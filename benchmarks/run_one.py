"""Train one (algorithm, variant, environment, seed) and record its learning curve.

Every algorithm is evaluated with the same protocol: every ``eval_freq``
environment steps, ``n_eval`` deterministic episodes on a separate environment
whose episode ``i`` is reset with seed ``100_000 + i`` (a held-out bank shared by
all algorithms and training seeds). Per-episode returns are stored.

    python -m benchmarks.run_one '{"algo": "TEMPO", "variant": "default", "env": "Hopper-v5", ...}'
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import sys
import time
import traceback

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
#: results root; override with TEMPO_BENCH_RESULTS (smoke runs, end-to-end tests)
RESULTS = pathlib.Path(os.environ.get("TEMPO_BENCH_RESULTS", ROOT / "benchmarks" / "results"))
EVAL_SEED_BASE = 100_000


def source_hash() -> str:
    import tempo_rl

    h = hashlib.sha256()
    for p in sorted(pathlib.Path(tempo_rl.__file__).parent.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def result_path(job: dict) -> pathlib.Path:
    return (RESULTS / job.get("suite", "main") / job["env"] / job["algo"] / job.get("variant", "default")
            / f"seed{job['seed']}.json")


# ---------------------------------------------------------------- evaluation
def evaluate(predict, env_id: str, n_eval: int, make_env) -> tuple[list, list]:
    import gymnasium as gym

    env = make_env()
    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    rets, lens = [], []
    for i in range(n_eval):
        obs, _ = env.reset(seed=EVAL_SEED_BASE + i)
        done, total, n = False, 0.0, 0
        while not done:
            action = predict(obs)
            if discrete:            # toy-text envs index dicts with the action
                action = int(np.asarray(action).reshape(-1)[0])
            obs, r, term, trunc, _ = env.step(action)
            total += float(r)
            n += 1
            done = term or trunc
        rets.append(total)
        lens.append(n)
    env.close()
    return rets, lens


# ------------------------------------------------------------------- TEMPO
def run_tempo(job: dict, record) -> dict:
    import torch
    from tempo_rl import TEMPO
    from tempo_rl.callbacks import BaseCallback
    from tempo_rl.env_util import make_env

    from .variants import tempo_kwargs

    kwargs = tempo_kwargs(job.get("variant", "default"), job["env"])
    entropy_offset = kwargs.pop("_physical_entropy", False)
    scale_through_truncation = kwargs.pop("_scale_through_truncation", False)
    kappa_scale = kwargs.pop("_kappa_scale", False)
    long_schedule = kwargs.pop("_long_entropy_schedule", None)
    audition_opts = kwargs.pop("_audition_opts", None)
    model = TEMPO("MlpPolicy", job["env"], seed=job["seed"], device=job.get("device", "cpu"),
                  env_kwargs=job.get("env_kwargs"), **kwargs)
    if scale_through_truncation:
        model._scale_resets_on_truncation = False
    if audition_opts:
        model._audition_opts = dict(audition_opts)
    if long_schedule is not None and not model.discrete:
        model._long_entropy_schedule = tuple(long_schedule)
    if kappa_scale and model.reward_scaler is not None:
        from tempo_rl.env_util import episode_horizon, horizon_kappa
        model.reward_scaler.kappa = horizon_kappa(model.gamma, episode_horizon(model.env))
    if entropy_offset and not model.discrete:
        model._entropy_log_offset = -float(np.log((model.action_space.high - model.action_space.low) / 2).sum())

    def predict(obs):
        return model.predict(obs, deterministic=True)[0]

    class Eval(BaseCallback):
        def _on_step(self):
            if self.num_timesteps % job["eval_freq"] < model.n_envs:
                record(self.num_timesteps, *evaluate(predict, job["env"], job["n_eval"],
                                                     lambda: make_env(job["env"], job.get("env_kwargs"))),
                       extra={"ent_coef": float(model.log_ent_coef.exp()),
                              "target_entropy": model.current_target_entropy()})
            return True

    model.learn(job["steps"], callback=Eval())
    cfg = {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict, tuple)) else str(v))
           for k, v in kwargs.items()}
    if scale_through_truncation:
        cfg["_scale_through_truncation"] = True
    if kappa_scale:
        cfg["_kappa_scale"] = True
    if long_schedule is not None:
        cfg["_long_entropy_schedule"] = list(long_schedule)
    if audition_opts:
        cfg["_audition_opts"] = dict(audition_opts)
    # the resolved architecture: the config above omits whatever the library defaults set
    net_arch = {k: [int(w) for w in v] for k, v in model.policy.net_arch.items()}
    out = {"config": cfg, "torch_threads": torch.get_num_threads(), "gamma": float(model.gamma),
           "net_arch": net_arch}
    if model.reward_scaler is not None and model.reward_scaler.kappa != 1.0:
        out["scale_kappa"] = float(model.reward_scaler.kappa)
    if model.horizon_audition:
        out["horizon_choice"] = model._horizon_choice
        out["audition"] = model._audition_log
    return out


# ------------------------------------------------------------------- SB3
def _schedule(v, total):
    if isinstance(v, str) and v.startswith("lin_"):
        x = float(v[4:])
        return lambda p: p * x
    return v


def _tuple_to_multidiscrete(env):
    """SB3 cannot take a ``Tuple`` observation space (Blackjack-v1: ``Tuple(Discrete(32), Discrete(11), Discrete(2))``):
    the same observation as ``MultiDiscrete([32, 11, 2])``, which SB3 one-hot encodes (45 inputs), exactly as TEMPO
    encodes the ``Tuple`` itself."""
    import gymnasium as gym

    class TupleToMultiDiscrete(gym.ObservationWrapper):
        def __init__(self, env):
            super().__init__(env)
            parts = env.observation_space.spaces
            self.start = np.array([int(s.start) for s in parts], dtype=np.int64)
            self.observation_space = gym.spaces.MultiDiscrete([int(s.n) for s in parts])

        def observation(self, obs):
            return np.asarray(obs, dtype=np.int64) - self.start

    return TupleToMultiDiscrete(env)


def run_sb3(job: dict, record) -> dict:
    if job["env"].startswith("CarRacing"):       # a pygame process: keep OpenCV, which SB3 imports, out (duplicate SDL)
        sys.modules.setdefault("cv2", None)
    import gymnasium as gym
    import sb3_contrib
    import stable_baselines3 as sb3
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    from tempo_rl.env_util import make_env

    from .zoo import zoo_hyperparams

    algo = job["algo"]
    cls = {"SAC": sb3.SAC, "TD3": sb3.TD3, "PPO": sb3.PPO, "DQN": sb3.DQN,
           "CrossQ": sb3_contrib.CrossQ}[algo]
    hp, tuned = zoo_hyperparams(algo, job["env"])
    hp.pop("n_timesteps", None)
    n_envs = hp.pop("n_envs", 1)
    normalize = hp.pop("normalize", False)
    noise_type, noise_std = hp.pop("noise_type", None), hp.pop("noise_std", None)
    for k in ("learning_rate", "clip_range"):
        if k in hp:
            hp[k] = _schedule(hp[k], job["steps"])

    adapted = []

    def env_fn():
        env = make_env(job["env"], job.get("env_kwargs"), preprocess=True)
        if job.get("device") == "mps" and isinstance(env.observation_space, gym.spaces.Box) \
                and env.observation_space.dtype == np.float64:
            # MPS has no float64; SB3 casts observations to float32 inside the policy anyway
            env = gym.wrappers.DtypeObservation(env, np.float32)
        if isinstance(env.observation_space, gym.spaces.Tuple) \
                and all(isinstance(s, gym.spaces.Discrete) for s in env.observation_space.spaces):
            env = _tuple_to_multidiscrete(env)
            adapted[:] = ["Tuple of Discrete -> MultiDiscrete (one-hot in SB3)"]
        return env

    venv = make_vec_env(env_fn, n_envs=n_envs, seed=job["seed"], vec_env_cls=DummyVecEnv)
    if normalize:   # True, or the zoo's dict form, e.g. {"norm_obs": False, "norm_reward": True} (CarRacing PPO)
        venv = VecNormalize(venv, gamma=hp.get("gamma", 0.99), clip_obs=10.0, clip_reward=10.0,
                            **{"norm_obs": True, "norm_reward": True, **(normalize if isinstance(normalize, dict) else {})})
    if noise_type is not None:
        d = venv.action_space.shape[0]
        noise_cls = NormalActionNoise if noise_type == "normal" else OrnsteinUhlenbeckActionNoise
        hp["action_noise"] = noise_cls(mean=np.zeros(d), sigma=noise_std * np.ones(d))
    policy = "CnnPolicy" if len(venv.observation_space.shape) == 3 else "MlpPolicy"
    model = cls(policy, venv, seed=job["seed"], device=job.get("device", "cpu"), verbose=0, **hp)

    def predict(obs):
        o = obs
        if normalize:
            o = venv.normalize_obs(np.asarray(obs)[None])[0]
        return model.predict(o, deterministic=True)[0]

    freq = job["eval_freq"]

    class Eval(BaseCallback):
        def __init__(self):
            super().__init__()
            self.next_eval = freq

        def _on_step(self):
            while self.num_timesteps >= self.next_eval:
                record(self.next_eval, *evaluate(predict, job["env"], job["n_eval"], env_fn))
                self.next_eval += freq
            return True

    model.learn(job["steps"], callback=Eval())
    out = {"config": {k: (v if isinstance(v, (int, float, str, bool, type(None), list, dict))
                          else repr(v)) for k, v in hp.items()},
           "zoo_tuned": tuned, "n_envs": n_envs, "normalize": normalize,
           "sb3_version": sb3.__version__, "sb3_contrib_version": sb3_contrib.__version__}
    if adapted:
        out["obs_adapter"] = adapted[0]
    return out


# -------------------------------------------------------------------- main
def use_snapshot(job: dict) -> str:
    """Import tempo_rl from a frozen per-suite copy when one exists.

    Library development continues while long benchmark suites run; a suite
    whose code is frozen in ``benchmarks/snapshots/<suite>/tempo_rl`` keeps
    running exactly that code for every job, however late the job starts.
    """
    snap = ROOT / "benchmarks" / "snapshots" / job.get("suite", "main")
    if (snap / "tempo_rl").is_dir():
        sys.path.insert(0, str(snap))
        return str(snap)
    return str(ROOT)


def main(job: dict) -> None:
    code_root = use_snapshot(job)
    import torch

    torch.set_num_threads(int(job.get("threads", 1)))
    out = result_path(job)
    if out.exists() and not job.get("force"):
        print(f"exists: {out}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    curve = {"steps": [], "returns": [], "lengths": [], "extra": []}

    def record(step, rets, lens, extra=None):
        curve["steps"].append(int(step))
        curve["returns"].append([float(r) for r in rets])
        curve["lengths"].append([int(n) for n in lens])
        curve["extra"].append(extra or {})
        print(f"[{job['env']} {job['algo']}/{job.get('variant', 'default')} s{job['seed']}] "
              f"step {step:>8d}  eval {np.mean(rets):10.2f}  ({time.time() - t0:7.0f}s)", flush=True)

    tempo_source_start = source_hash()      # the code this process imports now (the end hash below can describe later edits)
    t0 = time.time()
    runner = run_tempo if job["algo"] == "TEMPO" else run_sb3
    info = runner(job, record)
    import gymnasium
    import torch as _t

    result = dict(job=job, **curve, wall_s=time.time() - t0, info=info, tempo_source=source_hash(), tempo_source_start=tempo_source_start,
                  code_root=code_root,
                  versions=dict(python=platform.python_version(), torch=_t.__version__,
                                gymnasium=gymnasium.__version__, numpy=np.__version__,
                                platform=platform.platform()),
                  finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    tmp = out.with_suffix(".tmp")
    tmp.write_text(json.dumps(result, default=repr))
    os.replace(tmp, out)
    print(f"saved {out} ({result['wall_s'] / 60:.1f} min)", flush=True)


if __name__ == "__main__":
    try:
        main(json.loads(sys.argv[1]))
    except Exception:
        traceback.print_exc()
        sys.exit(1)
