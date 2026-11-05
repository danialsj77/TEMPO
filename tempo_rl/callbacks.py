"""Training callbacks with the Stable-Baselines3 interface.

A callback's ``_on_step`` runs after every environment step; returning ``False``
stops training. ``model.learn(callback=...)`` accepts a callback, a list of
callbacks, or a plain function ``f(locals, globals) -> bool``.
"""
from __future__ import annotations

import os
import warnings
from typing import Any, Callable

import numpy as np


class BaseCallback:
    def __init__(self, verbose: int = 0):
        self.verbose = verbose
        self.model = None
        self.n_calls = 0
        self.num_timesteps = 0
        self.locals: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}
        self.parent: BaseCallback | None = None

    @property
    def training_env(self):
        return self.model.get_env() if self.model is not None else None

    @property
    def logger(self):
        return self.model.logger

    # lifecycle ------------------------------------------------------------
    def init_callback(self, model) -> None:
        self.model = model
        self._init_callback()

    def on_training_start(self, locals_: dict, globals_: dict) -> None:
        self.locals, self.globals = locals_, globals_
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def on_step(self) -> bool:
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps
        return self._on_step()

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def on_training_end(self) -> None:
        self._on_training_end()

    def update_locals(self, locals_: dict) -> None:
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: dict) -> None:
        pass

    # hooks to override -----------------------------------------------------
    def _init_callback(self) -> None:
        pass

    def _on_training_start(self) -> None:
        pass

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        pass

    def _on_training_end(self) -> None:
        pass


class ConvertCallback(BaseCallback):
    """Wraps ``f(locals, globals) -> bool``."""

    def __init__(self, callback: Callable, verbose: int = 0):
        super().__init__(verbose)
        self.callback = callback

    def _on_step(self) -> bool:
        return bool(self.callback(self.locals, self.globals)) if self.callback else True


class CallbackList(BaseCallback):
    def __init__(self, callbacks: list[BaseCallback]):
        super().__init__()
        self.callbacks = list(callbacks)

    def _init_callback(self):
        for c in self.callbacks:
            c.init_callback(self.model)
            c.parent = self.parent

    def _on_training_start(self):
        for c in self.callbacks:
            c.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self):
        for c in self.callbacks:
            c.on_rollout_start()

    def _on_step(self) -> bool:
        cont = True
        for c in self.callbacks:
            cont = c.on_step() and cont
        return cont

    def _on_rollout_end(self):
        for c in self.callbacks:
            c.on_rollout_end()

    def _on_training_end(self):
        for c in self.callbacks:
            c.on_training_end()

    def update_child_locals(self, locals_):
        for c in self.callbacks:
            c.update_locals(locals_)


class EventCallback(BaseCallback):
    """Base for callbacks that trigger a child callback on an event."""

    def __init__(self, callback: BaseCallback | None = None, verbose: int = 0):
        super().__init__(verbose)
        self.callback = callback
        if callback is not None:
            callback.parent = self

    def init_callback(self, model) -> None:
        super().init_callback(model)
        if self.callback is not None:
            self.callback.init_callback(self.model)

    def _on_training_start(self):
        if self.callback is not None:
            self.callback.on_training_start(self.locals, self.globals)

    def _on_event(self) -> bool:
        return self.callback.on_step() if self.callback is not None else True

    def update_child_locals(self, locals_):
        if self.callback is not None:
            self.callback.update_locals(locals_)


class EveryNTimesteps(EventCallback):
    def __init__(self, n_steps: int, callback: BaseCallback):
        super().__init__(callback)
        self.n_steps = n_steps
        self.last_time_trigger = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_time_trigger >= self.n_steps:
            self.last_time_trigger = self.num_timesteps
            return self._on_event()
        return True


class CheckpointCallback(BaseCallback):
    """Save the model every ``save_freq`` calls (i.e. vectorised steps)."""

    def __init__(self, save_freq: int, save_path: str, name_prefix: str = "rl_model",
                 save_replay_buffer: bool = False, verbose: int = 0):
        super().__init__(verbose)
        self.save_freq, self.save_path, self.name_prefix = save_freq, save_path, name_prefix
        self.save_replay_buffer = save_replay_buffer

    def _init_callback(self):
        os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps")
            self.model.save(path)
            if self.save_replay_buffer:
                self.model.save_replay_buffer(
                    os.path.join(self.save_path,
                                 f"{self.name_prefix}_replay_buffer_{self.num_timesteps}_steps"))
            if self.verbose >= 1:
                print(f"Saving model checkpoint to {path}.zip")
        return True


class StopTrainingOnRewardThreshold(BaseCallback):
    """Use as ``EvalCallback(callback_on_new_best=...)``: stop once the mean reward passes a threshold."""

    def __init__(self, reward_threshold: float, verbose: int = 0):
        super().__init__(verbose)
        self.reward_threshold = reward_threshold

    def _on_step(self) -> bool:
        assert self.parent is not None, "use StopTrainingOnRewardThreshold with an EvalCallback"
        cont = bool(self.parent.best_mean_reward < self.reward_threshold)
        if self.verbose >= 1 and not cont:
            print(f"Stopping training: mean reward {self.parent.best_mean_reward:.2f} "
                  f"is above the threshold {self.reward_threshold}")
        return cont


class StopTrainingOnMaxEpisodes(BaseCallback):
    def __init__(self, max_episodes: int, verbose: int = 0):
        super().__init__(verbose)
        self.max_episodes = max_episodes
        self.n_episodes = 0

    def _on_step(self) -> bool:
        dones = self.locals.get("dones")
        if dones is not None:
            self.n_episodes += int(np.sum(dones))
        return self.n_episodes < self.max_episodes


class StopTrainingOnNoModelImprovement(BaseCallback):
    """Use as ``EvalCallback(callback_after_eval=...)``."""

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super().__init__(verbose)
        self.max_no_improvement_evals, self.min_evals = max_no_improvement_evals, min_evals
        self.last_best = -np.inf
        self.no_improvement = 0

    def _on_step(self) -> bool:
        assert self.parent is not None, "use StopTrainingOnNoModelImprovement with an EvalCallback"
        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best:
                self.no_improvement = 0
            else:
                self.no_improvement += 1
        self.last_best = self.parent.best_mean_reward
        return self.no_improvement <= self.max_no_improvement_evals


class EvalCallback(EventCallback):
    """Periodically evaluate the deterministic policy on a separate environment.

    Mirrors SB3: ``eval_freq`` counts calls (vectorised steps); results go to
    ``log_path/evaluations.npz``; the best model is saved to
    ``best_model_save_path/best_model.zip``.
    """

    def __init__(self, eval_env, callback_on_new_best: BaseCallback | None = None,
                 callback_after_eval: BaseCallback | None = None, n_eval_episodes: int = 5,
                 eval_freq: int = 10_000, log_path: str | None = None,
                 best_model_save_path: str | None = None, deterministic: bool = True,
                 render: bool = False, verbose: int = 1, warn: bool = True):
        super().__init__(callback_after_eval, verbose)
        self.callback_on_new_best = callback_on_new_best
        if callback_on_new_best is not None:
            callback_on_new_best.parent = self
        self.eval_env = eval_env
        self.n_eval_episodes, self.eval_freq = n_eval_episodes, eval_freq
        self.deterministic, self.render, self.warn = deterministic, render, warn
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.best_model_save_path = best_model_save_path
        self.log_path = os.path.join(log_path, "evaluations") if log_path else None
        self.evaluations_timesteps: list[int] = []
        self.evaluations_results: list[list[float]] = []
        self.evaluations_length: list[list[int]] = []

    def _init_callback(self):
        if self.best_model_save_path:
            os.makedirs(self.best_model_save_path, exist_ok=True)
        if self.log_path:
            os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.n_calls % self.eval_freq != 0:
            return True
        from .evaluation import evaluate_policy

        rewards, lengths = evaluate_policy(self.model, self.eval_env,
                                           n_eval_episodes=self.n_eval_episodes,
                                           deterministic=self.deterministic, render=self.render,
                                           return_episode_rewards=True, warn=self.warn)
        self.evaluations_timesteps.append(self.num_timesteps)
        self.evaluations_results.append(list(rewards))
        self.evaluations_length.append(list(lengths))
        if self.log_path:
            np.savez(self.log_path, timesteps=self.evaluations_timesteps,
                     results=self.evaluations_results, ep_lengths=self.evaluations_length)
        mean_r, std_r = float(np.mean(rewards)), float(np.std(rewards))
        mean_l = float(np.mean(lengths))
        self.last_mean_reward = mean_r
        if self.verbose >= 1:
            print(f"Eval num_timesteps={self.num_timesteps}, "
                  f"episode_reward={mean_r:.2f} +/- {std_r:.2f}")
            print(f"Episode length: {mean_l:.2f} +/- {np.std(lengths):.2f}")
        self.logger.record("eval/mean_reward", mean_r)
        self.logger.record("eval/mean_ep_length", mean_l)
        self.logger.record("time/total_timesteps", self.num_timesteps)
        self.logger.dump(self.num_timesteps)
        cont = True
        if mean_r > self.best_mean_reward:
            if self.verbose >= 1:
                print("New best mean reward!")
            if self.best_model_save_path:
                self.model.save(os.path.join(self.best_model_save_path, "best_model"))
            self.best_mean_reward = mean_r
            if self.callback_on_new_best is not None:
                cont = self.callback_on_new_best.on_step()
        if self.callback is not None:
            cont = self._on_event() and cont
        return cont


class ProgressBarCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        try:
            from tqdm.auto import tqdm  # noqa: F401
        except ImportError as e:  # pragma: no cover
            raise ImportError("progress_bar=True needs tqdm: pip install tqdm") from e
        self.pbar = None

    def _on_training_start(self):
        from tqdm.auto import tqdm
        total = self.locals["total_timesteps"] - self.model.num_timesteps
        self.pbar = tqdm(total=total, unit="steps")

    def _on_step(self) -> bool:
        self.pbar.update(self.model.n_envs)
        return True

    def _on_training_end(self):
        if self.pbar is not None:
            self.pbar.refresh()
            self.pbar.close()


def make_callback(callback, progress_bar: bool = False) -> BaseCallback:
    if isinstance(callback, list):
        callback = CallbackList(callback)
    elif callback is None:
        callback = CallbackList([])
    elif not isinstance(callback, BaseCallback):
        if all(hasattr(callback, a) for a in ("init_callback", "on_training_start", "on_step",
                                                "update_locals", "on_training_end")):
            callback = CallbackList([callback])      # duck-typed, e.g. a Stable-Baselines3 callback
        elif callable(callback):
            callback = ConvertCallback(callback)
        else:
            warnings.warn(f"ignoring unsupported callback {callback!r}")
            callback = CallbackList([])
    if progress_bar:
        callback = CallbackList([callback, ProgressBarCallback()])
    return callback
