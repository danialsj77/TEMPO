"""Command-line interface: ``tempo-drl {train,eval,enjoy,info}``.

Train, evaluate and watch TEMPO agents on any Gymnasium environment id without
writing Python::

    tempo-drl train --env CartPole-v1 --timesteps 50000 --seed 0 --save tempo_cartpole
    tempo-drl eval  --model tempo_cartpole.zip --env CartPole-v1 --episodes 10
    tempo-drl enjoy --model tempo_cartpole.zip --env CartPole-v1 --render

``--hyperparams`` takes ``key=value`` pairs for :class:`tempo_rl.TEMPO`. Values
are parsed with :func:`ast.literal_eval`; text that is not a Python literal
stays a string, so ``ent_coef=auto_0.1`` works. ``key.sub=value`` sets one entry
of a dict argument::

    tempo-drl train --env Pendulum-v1 --timesteps 20000 \\
        --hyperparams n_step=5 "policy_kwargs={'net_arch': [400, 300]}" \\
                      policy_kwargs.batch_norm=False

``python -m tempo_rl ...`` does the same as ``tempo-drl ...``. Only the public
API is used (``TEMPO``, ``evaluate_policy``, ``make_env``, ``EvalCallback``).
"""
from __future__ import annotations

import argparse
import ast
import difflib
import importlib
import inspect
import pathlib
import platform
import re
import sys
import time
import zipfile
from collections.abc import Sequence
from typing import Any

import numpy as np

from . import TEMPO, EvalCallback, __version__, evaluate_policy, make_env

#: The evaluation environment of ``train`` is seeded with ``seed + EVAL_SEED_OFFSET``.
EVAL_SEED_OFFSET = 10_000

#: TEMPO arguments that have their own flag and are refused in ``--hyperparams``.
_RESERVED = {"env": "--env", "seed": "--seed", "device": "--device", "verbose": "--verbose",
             "tensorboard_log": "--tensorboard", "env_kwargs": "--env-kwargs"}

_WORDS = {"true": True, "false": False, "none": None, "null": None}

_EPILOG = """\
examples:
  tempo-drl train --env CartPole-v1 --timesteps 50000 --seed 0
  tempo-drl train --env HalfCheetah-v5 --timesteps 1000000 --save runs/cheetah \\
      --eval-freq 10000 --log-dir runs/cheetah_eval --tensorboard runs/tb
  tempo-drl train --env LunarLander-v3 --timesteps 200000 --env-kwargs continuous=True \\
      --hyperparams n_envs=2 learning_rate=1e-3
  tempo-drl eval  --model tempo_CartPole-v1_s0.zip --env CartPole-v1 --episodes 20
  tempo-drl enjoy --model tempo_CartPole-v1_s0.zip --env CartPole-v1 --render
"""


class CLIError(Exception):
    """A usage error: reported as ``tempo-drl: error: ...`` with exit status 2."""


# ------------------------------------------------------------------ parsing
def _dotted(node: ast.AST) -> str | None:
    """``nn.Tanh`` / ``torch.nn.Tanh`` -> the dotted name, else None."""
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        return ".".join([node.id, *reversed(parts)])
    return None


def _node_value(node: ast.AST, text: str) -> Any:
    """Literal-like AST -> value. Bare names and dotted names become strings;
    ``dict(a=1)`` becomes a dict. Anything else (calls, arithmetic) is refused."""
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return _WORDS.get(node.id.lower(), node.id)
    if isinstance(node, ast.Attribute) and _dotted(node) is not None:
        return _dotted(node)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        items = [_node_value(e, text) for e in node.elts]
        return {ast.List: list, ast.Tuple: tuple, ast.Set: set}[type(node)](items)
    if isinstance(node, ast.Dict) and None not in node.keys:
        pairs = zip(node.keys, node.values, strict=True)
        return {_node_value(k, text): _node_value(v, text) for k, v in pairs}
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))):
        v = node.operand.value
        return -v if isinstance(node.op, ast.USub) else v
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "dict"
            and not node.args and all(kw.arg for kw in node.keywords)):
        return {kw.arg: _node_value(kw.value, text) for kw in node.keywords}
    raise CLIError(f"cannot parse value {text!r}: use Python literals such as 3e-4, "
                   "[256, 256], {'batch_norm': True} or a bare word such as auto")


def parse_value(text: str) -> Any:
    """Parse one ``--hyperparams`` / ``--env-kwargs`` value.

    ``ast.literal_eval`` first (``3e-4``, ``1_000_000``, ``[256, 256]``,
    ``(-0.3, -1.0, 0.5)``, ``{'batch_norm': True}``, ``None``); then a tolerant
    pass that also accepts ``true``/``false``/``null``, ``dict(a=1)`` and dotted
    names (``nn.Tanh``); anything that is not Python syntax (``auto_0.1``,
    ``cuda:0``, ``8x8``) is returned unchanged as a string.
    """
    text = text.strip()
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError, TypeError, MemoryError, RecursionError):
        pass
    try:
        node = ast.parse(text, mode="eval").body
    except SyntaxError:
        return text
    return _node_value(node, text)


def _set_nested(d: dict, path: list[str], value: Any, key: str, what: str) -> None:
    for part in path[:-1]:
        d = d.setdefault(part, {})
        if not isinstance(d, dict):
            raise CLIError(f"{what}: cannot set {key!r}, {part!r} is not a dict")
    if isinstance(d.get(path[-1]), dict) and isinstance(value, dict):
        d[path[-1]].update(value)
    else:
        d[path[-1]] = value


def parse_pairs(pairs: Sequence[str] | None, what: str) -> dict[str, Any]:
    """``["a=1", "b.c=True"]`` -> ``{"a": 1, "b": {"c": True}}`` (later pairs win)."""
    out: dict[str, Any] = {}
    for item in pairs or []:
        key, sep, raw = item.partition("=")
        key = key.strip()
        if not sep or not key or any(not p for p in key.split(".")):
            raise CLIError(f"{what} expects KEY=VALUE pairs, got {item!r}")
        _set_nested(out, key.split("."), parse_value(raw), key, what)
    return out


def _int_annotation(annotation: Any) -> bool:
    """True for ``int`` / ``int | None`` annotations (strings under PEP 563)."""
    if annotation is int:
        return True
    words = re.findall(r"[A-Za-z_]+", annotation if isinstance(annotation, str) else "")
    return "int" in words and "float" not in words


def _activation(name: str):
    import torch.nn as nn

    cls = getattr(nn, name.rsplit(".", 1)[-1], None)
    if not (isinstance(cls, type) and issubclass(cls, nn.Module)):
        raise CLIError(f"unknown activation_fn {name!r}: use a torch.nn class name such as "
                       "ReLU, Tanh, ELU or SiLU")
    return cls


def tempo_hyperparams(pairs: Sequence[str] | None) -> dict[str, Any]:
    """Parse and validate ``--hyperparams`` against ``TEMPO.__init__``'s signature."""
    hp = parse_pairs(pairs, "--hyperparams")
    params = inspect.signature(TEMPO.__init__).parameters
    open_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
    valid = sorted(k for k in params
                   if k != "self" and not k.startswith("_") and k not in _RESERVED)
    for key in list(hp):
        if key in _RESERVED:
            raise CLIError(f"set {key!r} with {_RESERVED[key]}, not with --hyperparams")
        if key not in params or key.startswith("_"):
            if open_kwargs and not key.startswith("_"):
                continue
            close = difflib.get_close_matches(key, valid, n=3)
            hint = (f"did you mean {' or '.join(close)}?" if close
                    else f"valid keys: {', '.join(valid)}")
            raise CLIError(f"unknown hyperparameter {key!r}; {hint}")
        value = hp[key]
        if (isinstance(value, float) and value.is_integer()
                and _int_annotation(params[key].annotation)):
            hp[key] = int(value)          # buffer_size=1e6 -> 1000000
    pk = hp.get("policy_kwargs")
    if pk is not None:
        if not isinstance(pk, dict):
            raise CLIError("policy_kwargs must be a dict, e.g. "
                           "\"policy_kwargs={'batch_norm': True}\"")
        if isinstance(pk.get("activation_fn"), str):
            pk["activation_fn"] = _activation(pk["activation_fn"])
    return hp


# ------------------------------------------------------------------ helpers
def _zip_path(path: str | pathlib.Path) -> pathlib.Path:
    """The file ``TEMPO.save(path)`` writes: ``.zip`` is appended when missing."""
    path = pathlib.Path(path)
    return path if path.suffix == ".zip" else pathlib.Path(f"{path}.zip")


def _default_save_path(env_id: str, seed: int | None) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", env_id).strip("-") or "env"
    return f"tempo_{name}" + ("" if seed is None else f"_s{seed}")


def _require(module: str, package: str, flag: str) -> None:
    try:
        importlib.import_module(module)
    except ImportError as e:
        raise CLIError(f"{flag} needs the {package!r} package: "
                       f"pip install {package} (or pip install 'tempo-drl[extra]')") from e


def _env_errors() -> tuple[type[BaseException], ...]:
    import gymnasium as gym

    return (gym.error.Error, ImportError)


def _make_env(env_id: str, env_kwargs: dict):
    try:
        return make_env(env_id, env_kwargs or None)
    except _env_errors() as e:
        raise CLIError(f"cannot create environment {env_id!r}: {e}") from e


def _seed_env(env, seed: int | None) -> None:
    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)


def _load_model(path: str, env, env_id: str, device: str) -> TEMPO:
    p = pathlib.Path(path)
    if not p.is_file():
        if not _zip_path(p).is_file():
            raise CLIError(f"model file not found: {path}")
        p = _zip_path(p)
    try:
        model = TEMPO.load(p, device=device)
    except (zipfile.BadZipFile, KeyError, ValueError) as e:
        raise CLIError(f"{p} is not a loadable TEMPO model: {e!r}") from e
    for what, model_space, env_space in (("observation", model.observation_space,
                                          env.observation_space),
                                         ("action", model.action_space, env.action_space)):
        if model_space != env_space:
            raise CLIError(f"{p} was trained with a different {what} space than {env_id}:\n"
                           f"  model: {model_space}\n  env:   {env_space}")
    return model


def _episode_line(index: int, reward: float, length: int) -> str:
    return f"episode {index:>4d}: return {float(reward):12.2f}   length {int(length):6d}"


def _summary(rewards, lengths) -> str:
    return (f"mean_reward={np.mean(rewards):.2f} +/- {np.std(rewards):.2f} "
            f"over {len(rewards)} episodes (mean length {np.mean(lengths):.1f})")


# ----------------------------------------------------------------- commands
def cmd_train(args: argparse.Namespace) -> int:
    hp = tempo_hyperparams(args.hyperparams)
    env_kwargs = parse_pairs(args.env_kwargs, "--env-kwargs")
    if args.tensorboard:
        _require("torch.utils.tensorboard", "tensorboard", "--tensorboard")
    if args.progress_bar:
        _require("tqdm", "tqdm", "--progress-bar")
    save_path = _zip_path(args.save or _default_save_path(args.env, args.seed))
    policy = hp.pop("policy", "MlpPolicy")

    try:
        model = TEMPO(policy, args.env, seed=args.seed, device=args.device, verbose=args.verbose,
                      tensorboard_log=args.tensorboard, env_kwargs=env_kwargs or None, **hp)
    except (*_env_errors(), NotImplementedError, ValueError) as e:
        raise CLIError(f"cannot set up TEMPO on {args.env!r}: {e}") from e

    callbacks, eval_env = [], None
    if args.eval_freq > 0 and args.n_eval > 0:
        eval_env = _make_env(args.env, env_kwargs)
        _seed_env(eval_env, None if args.seed is None else args.seed + EVAL_SEED_OFFSET)
        callbacks.append(EvalCallback(
            eval_env, n_eval_episodes=args.n_eval,
            eval_freq=max(args.eval_freq // model.n_envs, 1),   # counted in vectorised steps
            log_path=args.log_dir, best_model_save_path=args.log_dir, deterministic=True,
            verbose=args.verbose))

    if args.verbose >= 1:
        print(f"TEMPO {__version__} | {args.env} | {args.timesteps:,} steps | seed={args.seed} "
              f"| device={model.device} | n_envs={model.n_envs}", flush=True)
    start, status = time.time(), 0
    try:
        model.learn(args.timesteps, callback=callbacks or None, log_interval=args.log_interval,
                    progress_bar=args.progress_bar)
    except KeyboardInterrupt:
        status = 130
        print(f"\nInterrupted after {model.num_timesteps:,} steps; saving the current model.",
              file=sys.stderr)
    model.save(save_path)
    print(f"Saved model to {save_path} ({model.num_timesteps:,} steps, "
          f"{time.time() - start:.0f} s)")
    for env in (model.get_env(), eval_env):
        if env is not None:
            env.close()
    return status


def cmd_eval(args: argparse.Namespace) -> int:
    env = _make_env(args.env, parse_pairs(args.env_kwargs, "--env-kwargs"))
    try:
        model = _load_model(args.model, env, args.env, args.device)
        _seed_env(env, args.seed)
        rewards, lengths = evaluate_policy(model, env, n_eval_episodes=args.episodes,
                                           deterministic=not args.stochastic,
                                           return_episode_rewards=True)
    finally:
        env.close()
    if args.verbose >= 1:
        for i, (r, n) in enumerate(zip(rewards, lengths, strict=True), 1):
            print(_episode_line(i, r, n))
    print(_summary(rewards, lengths))
    return 0


def cmd_enjoy(args: argparse.Namespace) -> int:
    env_kwargs = parse_pairs(args.env_kwargs, "--env-kwargs")
    if args.render:
        env_kwargs.setdefault("render_mode", "human")
    env = _make_env(args.env, env_kwargs)
    rewards: list[float] = []
    lengths: list[int] = []
    try:
        model = _load_model(args.model, env, args.env, args.device)
        obs, _ = env.reset(seed=args.seed)
        for _ in range(args.episodes):
            total, steps, done = 0.0, 0, False
            while not done:
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                steps += 1
                done = terminated or truncated
            rewards.append(total)
            lengths.append(steps)
            print(_episode_line(len(rewards), total, steps), flush=True)
            obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
    except _env_errors() as e:        # e.g. rendering needs pygame
        raise CLIError(f"{args.env!r} failed while running: {e}") from e
    finally:
        env.close()
    if rewards:
        print(_summary(rewards, lengths))
    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Versions and devices, for bug reports."""
    import importlib.metadata as md

    import gymnasium
    import torch

    def version(dist: str) -> str:
        try:
            return md.version(dist)
        except md.PackageNotFoundError:
            return "not installed"

    rows = [
        ("tempo-drl", f"{__version__} ({pathlib.Path(__file__).resolve().parent})"),
        ("python", f"{platform.python_version()} ({platform.python_implementation()})"),
        ("platform", platform.platform()),
        ("torch", f"{torch.__version__} (threads={torch.get_num_threads()}, "
                  f"cuda={torch.cuda.is_available()}, mps={torch.backends.mps.is_available()})"),
        ("gymnasium", gymnasium.__version__),
        ("numpy", np.__version__),
    ]
    rows += [(d, version(d)) for d in ("stable-baselines3", "sb3-contrib", "mujoco", "box2d",
                                        "ale-py", "tensorboard", "tqdm", "rich")]
    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"{k.ljust(width)}  {v}")
    return 0


# ------------------------------------------------------------------- parser
def _count(text: str, allow_zero: bool = False) -> int:
    """``20000``, ``2e4`` or ``20_000`` -> 20000."""
    try:
        value = float(text.replace("_", ""))
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {text!r}") from None
    if not value.is_integer() or value < (0 if allow_zero else 1):
        raise argparse.ArgumentTypeError(
            f"expected a {'non-negative' if allow_zero else 'positive'} integer, got {text!r}")
    return int(value)


def _count0(text: str) -> int:
    return _count(text, allow_zero=True)


def build_parser(prog: str | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=prog or "tempo-drl",
        description="Train, evaluate and watch TEMPO agents on Gymnasium environments.",
        epilog=_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--device", default="auto",
                        help="auto (default: CUDA if available, else MPS for image observations or "
                             "networks with a layer of 512+ units, such as the default 1024-unit "
                             "critic, else CPU), cpu, cuda, cuda:1, mps")
    common.add_argument("--threads", type=_count, default=None, metavar="N",
                        help="number of PyTorch CPU threads (default: PyTorch's choice)")
    common.add_argument("--env-kwargs", nargs="+", action="extend", default=[],
                        metavar="KEY=VALUE",
                        help="keyword arguments for gymnasium.make, e.g. continuous=True")
    common.add_argument("--verbose", type=int, default=1, choices=(0, 1, 2),
                        help="0 quiet, 1 progress (default), 2 debug")

    t = sub.add_parser("train", parents=[common], help="train a TEMPO agent and save it",
                       description="Train TEMPO on a Gymnasium environment id and save a .zip "
                                   "model that TEMPO.load(), eval and enjoy can read.",
                       epilog=_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    t.add_argument("--env", required=True, metavar="ENV_ID",
                   help="Gymnasium id, e.g. CartPole-v1, Humanoid-v5, ALE/Pong-v5")
    t.add_argument("--timesteps", type=_count, required=True, metavar="N",
                   help="environment steps to train for (2e5 and 200_000 are accepted)")
    t.add_argument("--seed", type=int, default=None, help="random seed (default: random)")
    t.add_argument("--save", default=None, metavar="PATH",
                   help="model path; .zip is appended if missing "
                        "(default: tempo_<ENV_ID>[_s<SEED>].zip in the current directory)")
    t.add_argument("--eval-freq", type=_count0, default=10_000, metavar="F",
                   help="evaluate the deterministic policy every F environment steps "
                        "(default: 10000; 0 disables)")
    t.add_argument("--n-eval", type=_count0, default=5, metavar="E",
                   help="episodes per evaluation (default: 5)")
    t.add_argument("--log-dir", default=None, metavar="DIR",
                   help="write evaluations.npz and best_model.zip to DIR")
    t.add_argument("--tensorboard", default=None, metavar="DIR",
                   help="TensorBoard log directory (needs the tensorboard package)")
    t.add_argument("--hyperparams", nargs="+", action="extend", default=[], metavar="KEY=VALUE",
                   help="TEMPO arguments, e.g. n_step=5 learning_rate=1e-3 n_envs=4 "
                        "policy_kwargs.batch_norm=False")
    t.add_argument("--log-interval", type=_count0, default=4, metavar="EPISODES",
                   help="episodes between training-log tables (default: 4)")
    t.add_argument("--progress-bar", action="store_true", help="show a tqdm progress bar")
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", parents=[common], help="evaluate a saved model",
                       description="Evaluate a saved TEMPO model and print the mean return.")
    e.add_argument("--model", required=True, metavar="PATH", help="saved model (.zip)")
    e.add_argument("--env", required=True, metavar="ENV_ID")
    e.add_argument("--episodes", type=_count, default=10, metavar="N",
                   help="evaluation episodes (default: 10)")
    e.add_argument("--seed", type=int, default=None, help="seed of the evaluation episodes")
    e.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of using the deterministic policy")
    e.set_defaults(func=cmd_eval)

    j = sub.add_parser("enjoy", parents=[common], help="run (and render) a saved model",
                       description="Run a saved TEMPO model episode by episode, optionally "
                                   "rendering it in a window.")
    j.add_argument("--model", required=True, metavar="PATH", help="saved model (.zip)")
    j.add_argument("--env", required=True, metavar="ENV_ID")
    j.add_argument("--episodes", type=_count, default=5, metavar="N",
                   help="episodes to run (default: 5)")
    j.add_argument("--render", action="store_true",
                   help="render in a window (render_mode='human'; classic-control and "
                        "toy-text environments need pygame: "
                        "pip install 'gymnasium[classic-control]')")
    j.add_argument("--seed", type=int, default=None, help="seed of the first episode")
    j.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of using the deterministic policy")
    j.set_defaults(func=cmd_enjoy)

    i = sub.add_parser("info", help="print versions and devices (for bug reports)")
    i.set_defaults(func=cmd_info, threads=None)
    return parser


def main(argv: Sequence[str] | None = None, prog: str | None = None) -> int:
    """Entry point of the ``tempo-drl`` console script; returns the exit status."""
    parser = build_parser(prog)
    args = parser.parse_args(argv)
    try:
        if args.threads is not None:
            import torch

            torch.set_num_threads(args.threads)
        return int(args.func(args) or 0)
    except CLIError as err:
        parser.exit(2, f"{parser.prog}: error: {err}\n")
    return 2  # pragma: no cover  (parser.exit raises SystemExit)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
