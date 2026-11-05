"""Environment construction helpers.

``make_vec_env("Humanoid-v5", n_envs=1, seed=0)`` builds TEMPO's lock-step vector
of environments, each wrapped in ``RecordEpisodeStatistics``. Environment ids of
Atari games (``"ALE/Pong-v5"``) get the standard DQN preprocessing
automatically (84x84 grayscale, frame-skip 4, 4-frame stack, no-op starts);
``CarRacing`` gets a numpy-only grayscale/crop/frame-skip/frame-stack pipeline.
Pass ``wrapper_class`` to add your own wrappers.
"""
from __future__ import annotations

from collections import deque
from typing import Any, Callable

import gymnasium as gym
import numpy as np

from .vec_env import TempoVecEnv


class CarRacingPreprocess(gym.Wrapper):
    """CarRacing pixels -> uint8 ``[stack, 84, 84]``.

    Grayscale, crop the 96x96 frame to 84x84 (drops the status bar), repeat each
    action ``skip`` frames summing rewards, stack the last ``stack`` frames.
    Pure numpy: avoids importing OpenCV next to pygame/Box2D (duplicate SDL
    symbols on some platforms).
    """

    def __init__(self, env: gym.Env, skip: int = 4, stack: int = 4):
        super().__init__(env)
        self.skip, self.stack = skip, stack
        self.frames: deque = deque(maxlen=stack)
        self.observation_space = gym.spaces.Box(0, 255, (stack, 84, 84), dtype=np.uint8)

    @staticmethod
    def _proc(frame: np.ndarray) -> np.ndarray:
        gray = frame[:84, 6:90].astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
        return gray.astype(np.uint8)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        f = self._proc(obs)
        for _ in range(self.stack):
            self.frames.append(f)
        return np.stack(self.frames, 0), info

    def step(self, action):
        total, terminated, truncated, info = 0.0, False, False, {}
        for _ in range(self.skip):
            obs, r, terminated, truncated, info = self.env.step(action)
            total += float(r)
            if terminated or truncated:
                break
        self.frames.append(self._proc(obs))
        return np.stack(self.frames, 0), total, terminated, truncated, info


def is_atari_id(env_id: str) -> bool:
    return env_id.startswith("ALE/") or "NoFrameskip" in env_id


def _make_atari(env_id: str, **kwargs) -> gym.Env:
    import ale_py  # noqa: F401  (registers the ALE environments)

    gym.register_envs(ale_py)
    kwargs.setdefault("frameskip", 1)
    env = gym.make(env_id, **kwargs)
    env = gym.wrappers.AtariPreprocessing(env, noop_max=30, frame_skip=4, screen_size=84,
                                          terminal_on_life_loss=False, grayscale_obs=True,
                                          scale_obs=False)
    return gym.wrappers.FrameStackObservation(env, 4)


def make_env(env_id: str, env_kwargs: dict | None = None,
             wrapper_class: Callable[[gym.Env], gym.Env] | None = None,
             preprocess: bool = True) -> gym.Env:
    """One environment, with TEMPO's standard preprocessing for known pixel tasks."""
    env_kwargs = dict(env_kwargs or {})
    if preprocess and is_atari_id(env_id):
        env = _make_atari(env_id, **env_kwargs)
    else:
        env = gym.make(env_id, **env_kwargs)
        if preprocess and env_id.startswith("CarRacing"):
            env = CarRacingPreprocess(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    if wrapper_class is not None:
        env = wrapper_class(env)
    return env


def _has_episode_statistics(env: gym.Env) -> bool:
    e: Any = env
    while e is not None:
        if isinstance(e, gym.wrappers.RecordEpisodeStatistics) or type(e).__name__ == "Monitor":
            return True
        e = getattr(e, "env", None)
    return False


def _time_limit_wrapper_steps(env) -> int | None:
    """``max_episode_steps`` of the outermost ``TimeLimit`` wrapper, if any."""
    e: Any = env
    while e is not None:
        if isinstance(e, gym.wrappers.TimeLimit):
            return e._max_episode_steps
        e = getattr(e, "env", None)
    return None


def episode_horizon(env) -> int | None:
    """The time limit of an environment (TEMPO/SB3 vector env or ``gymnasium.Env``), if any."""
    try:
        if hasattr(env, "envs"):                      # TempoVecEnv / DummyVecEnv
            env = env.envs[0]
        elif hasattr(env, "venv"):                    # SB3VecEnvAdapter
            spec = env.venv.get_attr("spec")[0]
            horizon = getattr(spec, "max_episode_steps", None)
            if horizon is None:
                try:                                  # get_attr searches the wrapper chain
                    horizon = env.venv.get_attr("_max_episode_steps")[0]
                except Exception:  # noqa: BLE001 - no TimeLimit wrapper
                    horizon = None
            return horizon
        horizon = getattr(getattr(env, "spec", None), "max_episode_steps", None)
        return horizon if horizon is not None else _time_limit_wrapper_steps(env)
    except Exception:  # noqa: BLE001 - best effort
        return None


def auto_gamma(horizon: int | None, default: float = 0.99) -> float:
    """Discount from the episode time limit ``T``: ``clip(1 - 5/T, 0.95, 0.995)``.

    The effective horizon ``1/(1-gamma)`` becomes a fifth of the episode, the
    heuristic of TD-MPC2 (Hansen et al., 2024). Without a time limit: ``default``.
    """
    if not horizon:
        return default
    return float(min(0.995, max(0.95, 1.0 - 5.0 / float(horizon))))


def horizon_kappa(gamma: float, horizon: int | None) -> float:
    """Factor that makes the episodic reward scale consistent with bootstrapping through
    time limits: ``(1 - g_a^T) / (1 - gamma^T)`` with ``g_a = auto_gamma(T)`` when ``gamma``
    is longer than ``g_a``, else exactly 1.0 (research option, Screen 7b).

    The running return of :class:`~tempo_rl.utils.RewardScaler` restarts at every episode end,
    so it measures ``r (1 - gamma^T) / (1 - gamma)`` while the critic, which bootstraps through
    time limits, learns ``r / (1 - gamma)``. At ``gamma = 1 - 5/T`` the two agree up to
    ``e^-5``; for longer discounts the factor restores the agreement.
    """
    if not horizon:
        return 1.0
    g_a = auto_gamma(horizon)
    if gamma <= g_a:
        return 1.0
    return float((1.0 - g_a ** horizon) / (1.0 - gamma ** horizon))


def wrap_env(env: gym.Env) -> gym.Env:
    return env if _has_episode_statistics(env) else gym.wrappers.RecordEpisodeStatistics(env)


def make_vec_env(env_id: str | Callable[[], gym.Env], n_envs: int = 1, seed: int | None = None,
                 env_kwargs: dict | None = None,
                 wrapper_class: Callable[[gym.Env], gym.Env] | None = None) -> TempoVecEnv:
    """``n_envs`` copies of an environment id (or of an env factory), stepped in lock-step."""
    def factory():
        if callable(env_id):
            return wrap_env(env_id())
        return make_env(env_id, env_kwargs, wrapper_class)

    venv = TempoVecEnv([factory for _ in range(n_envs)])
    venv.seed(seed)
    return venv
