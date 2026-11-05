"""``evaluate_policy`` with the Stable-Baselines3 signature and semantics."""
from __future__ import annotations

import warnings

import gymnasium as gym
import numpy as np

from .vec_env import SB3VecEnvAdapter, TempoVecEnv, episode_info, stack_obs


def _as_vec(env):
    if isinstance(env, (TempoVecEnv, SB3VecEnvAdapter)):
        return env
    if SB3VecEnvAdapter.is_sb3_vec_env(env):
        return SB3VecEnvAdapter(env)
    if isinstance(env, gym.Env):
        from .env_util import wrap_env

        wrapped = wrap_env(env)
        return TempoVecEnv([lambda: wrapped])
    raise TypeError(f"cannot evaluate on {env!r}")


def _is_monitored(venv) -> bool:
    """True if every sub-env reports episodes (RecordEpisodeStatistics / Monitor / VecMonitor).

    Then, as in SB3, only ``info["episode"]`` ends an evaluation episode, so a
    ``done`` emitted above the monitor (e.g. SB3's ``EpisodicLifeEnv`` life loss)
    is not counted as a whole episode.
    """
    if isinstance(venv, TempoVecEnv):
        from .env_util import _has_episode_statistics

        return all(_has_episode_statistics(e) for e in venv.envs)
    try:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import VecMonitor, is_vecenv_wrapped

        inner = venv.venv
        return bool(is_vecenv_wrapped(inner, VecMonitor) or all(inner.env_is_wrapped(Monitor)))
    except Exception:  # noqa: BLE001 - not an SB3 VecEnv we can inspect
        return False


def evaluate_policy(model, env, n_eval_episodes: int = 10, deterministic: bool = True,
                    render: bool = False, callback=None, reward_threshold: float | None = None,
                    return_episode_rewards: bool = False, warn: bool = True):
    """Run ``n_eval_episodes`` episodes and return ``(mean_reward, std_reward)``.

    With ``return_episode_rewards=True`` returns ``(episode_rewards, episode_lengths)``.
    Works with any object exposing ``predict(obs, state, episode_start, deterministic)``
    -- TEMPO models and Stable-Baselines3 models alike -- and with a Gymnasium
    environment, a TEMPO vector env, or an SB3 ``VecEnv``. Episodes are split
    across sub-environments exactly as in SB3 to avoid biasing towards short
    episodes. Returns are read from ``RecordEpisodeStatistics``/``Monitor`` when
    present, so reward-shaping wrappers do not distort them.
    """
    venv = _as_vec(env)
    n_envs = venv.num_envs
    counts = np.zeros(n_envs, dtype=int)
    targets = np.array([(n_eval_episodes + i) // n_envs for i in range(n_envs)], dtype=int)
    cur_r = np.zeros(n_envs)
    cur_l = np.zeros(n_envs, dtype=int)
    ep_rewards, ep_lengths = [], []
    monitored = _is_monitored(venv)
    obs = venv.reset_all()
    states = None
    starts = np.ones(n_envs, dtype=bool)
    saw_monitor = monitored
    while (counts < targets).any():
        batch = _stack(obs)
        actions, states = model.predict(batch, state=states, episode_start=starts,
                                        deterministic=deterministic)
        obs, rews, terms, truncs, infos, _ = venv.step_all(actions)
        cur_r += rews
        cur_l += 1
        dones = terms | truncs
        for i in range(n_envs):
            if callback is not None:
                callback(locals(), globals())
            if counts[i] < targets[i] and dones[i]:
                ep = episode_info(infos[i])
                if ep is not None:
                    saw_monitor = True
                    ep_rewards.append(ep[0])
                    ep_lengths.append(ep[1])
                    counts[i] += 1
                elif not monitored:
                    ep_rewards.append(float(cur_r[i]))
                    ep_lengths.append(int(cur_l[i]))
                    counts[i] += 1
            if dones[i]:
                cur_r[i], cur_l[i] = 0.0, 0
        starts = dones
        if render:
            venv.render()
    if warn and not saw_monitor:
        warnings.warn("evaluation env has no RecordEpisodeStatistics/Monitor wrapper: "
                      "returns are raw reward sums of the wrapped env", UserWarning)
    mean, std = float(np.mean(ep_rewards)), float(np.std(ep_rewards))
    if reward_threshold is not None:
        assert mean > reward_threshold, f"mean reward {mean:.2f} below threshold {reward_threshold:.2f}"
    if return_episode_rewards:
        return ep_rewards, ep_lengths
    return mean, std


_stack = stack_obs   # backwards-compatible name
