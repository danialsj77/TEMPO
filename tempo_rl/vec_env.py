"""Minimal vectorised-environment layer.

TEMPO steps a list of Gymnasium environments in lock-step (``n_envs`` defaults
to 1). Every step returns the *genuine* final observation of environments whose
episode just ended, which the replay buffer needs to bootstrap correctly through
time-limit truncations. Stable-Baselines3 ``VecEnv`` objects are accepted too and
adapted to the same interface.
"""
from __future__ import annotations

from typing import Any, Callable

import gymnasium as gym
import numpy as np


def stack_obs(obs_list):
    """List of per-env observations -> batched observation (array, dict or tuple)."""
    first = obs_list[0]
    if isinstance(first, dict):
        return {k: np.stack([o[k] for o in obs_list]) for k in first}
    if isinstance(first, tuple):
        return tuple(np.stack([o[i] for o in obs_list]) for i in range(len(first)))
    return np.stack([np.asarray(o) for o in obs_list])


class TempoVecEnv:
    """Lock-step list of Gymnasium environments with explicit final observations.

    ``reset()``/``step()`` follow the Stable-Baselines3 ``VecEnv`` API (what users
    see through ``model.get_env()``); TEMPO itself uses ``reset_all()``/``step_all()``,
    which return per-env lists plus the genuine final observations.
    """

    def __init__(self, env_fns: list[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        self.num_envs = len(self.envs)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space
        self._seeds: list[int | None] = [None] * self.num_envs
        self.render_mode = getattr(self.envs[0], "render_mode", None)

    def seed(self, seed: int | None) -> None:
        self._seeds = [None if seed is None else seed + i for i in range(self.num_envs)]
        for i, env in enumerate(self.envs):
            env.action_space.seed(None if seed is None else seed + i)

    def reset_all(self) -> list[Any]:
        obs = []
        for i, env in enumerate(self.envs):
            o, _ = env.reset(seed=self._seeds[i])
            self._seeds[i] = None       # seed only the first reset
            obs.append(o)
        return obs

    def step_all(self, actions):
        obs, rews, terms, truncs, infos, final_obs = [], [], [], [], [], {}
        for i, env in enumerate(self.envs):
            o, r, te, tr, info = env.step(actions[i])
            if te or tr:
                final_obs[i] = o
                o, _ = env.reset()
            obs.append(o)
            rews.append(r)
            terms.append(te)
            truncs.append(tr)
            infos.append(info)
        return (obs, np.asarray(rews, dtype=np.float32), np.asarray(terms, dtype=bool),
                np.asarray(truncs, dtype=bool), infos, final_obs)

    # Stable-Baselines3 VecEnv API ------------------------------------------------
    def reset(self):
        """Batched observation of every (re)started environment."""
        return stack_obs(self.reset_all())

    def step(self, actions):
        """``(obs, rewards, dones, infos)`` with auto-reset, as an SB3 ``VecEnv``.

        At an episode end ``infos[i]`` holds ``"terminal_observation"`` and
        ``"TimeLimit.truncated"``, and ``obs[i]`` is the first observation of the
        next episode.
        """
        obs, rews, terms, truncs, infos, final_obs = self.step_all(actions)
        infos = [dict(info) for info in infos]
        for i, fo in final_obs.items():
            infos[i]["terminal_observation"] = fo
            infos[i]["TimeLimit.truncated"] = bool(truncs[i] and not terms[i])
        return stack_obs(obs), rews, terms | truncs, infos

    def render(self):
        return self.envs[0].render()

    def close(self) -> None:
        for env in self.envs:
            env.close()


class SB3VecEnvAdapter:
    """Wraps a Stable-Baselines3 ``VecEnv`` to the :class:`TempoVecEnv` interface."""

    def __init__(self, venv):
        self.venv = venv
        self.num_envs = venv.num_envs
        self.observation_space = venv.observation_space
        self.action_space = venv.action_space
        self.render_mode = getattr(venv, "render_mode", None)

    @staticmethod
    def is_sb3_vec_env(env) -> bool:
        return all(hasattr(env, a) for a in ("num_envs", "step_async", "step_wait", "env_method"))

    def _split(self, obs):
        if isinstance(obs, dict):
            return [{k: v[i] for k, v in obs.items()} for i in range(self.num_envs)]
        if isinstance(obs, tuple):
            return [tuple(v[i] for v in obs) for i in range(self.num_envs)]
        return list(obs)

    def seed(self, seed):
        if seed is not None:
            self.venv.seed(seed)
            self.venv.action_space.seed(seed)

    def reset_all(self):
        return self._split(self.venv.reset())

    def reset(self):
        return self.venv.reset()

    def step(self, actions):
        return self.venv.step(np.asarray(actions))

    def step_all(self, actions):
        obs, rews, dones, infos = self.venv.step(np.asarray(actions))
        truncs = np.array([bool(d and info.get("TimeLimit.truncated", False))
                           for d, info in zip(dones, infos)])
        terms = np.asarray(dones, dtype=bool) & ~truncs
        final_obs = {i: info["terminal_observation"] for i, info in enumerate(infos)
                     if dones[i] and "terminal_observation" in info}
        return self._split(obs), np.asarray(rews, dtype=np.float32), terms, truncs, list(infos), final_obs

    def render(self):
        return self.venv.render()

    def close(self):
        self.venv.close()


def episode_info(info: dict):
    """(return, length) if ``info`` marks the end of a recorded episode, else None.

    Understands Gymnasium's ``RecordEpisodeStatistics`` and SB3's ``Monitor``.
    """
    ep = info.get("episode")
    if ep is None:
        return None
    return float(np.asarray(ep["r"]).reshape(-1)[0]), int(np.asarray(ep["l"]).reshape(-1)[0])
