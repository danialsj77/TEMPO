"""Train TEMPO on your own task: a point on a square that must reach a goal.

The task is written here from scratch as a Gymnasium environment. The same script trains it with continuous
actions (a move in two dimensions) or with discrete actions (stay, right, left, up, down); TEMPO needs no
setting for either and no hyperparameter for the task.

    python examples/custom_env.py                      # continuous actions
    python examples/custom_env.py --discrete           # discrete actions
    python examples/custom_env.py --timesteps 50000 --device cpu
"""
from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from tempo_rl import TEMPO, evaluate_policy


class Reach(gym.Env):
    """A point on the square [-1, 1]^2 must reach a goal that is drawn at random in every episode.

    Observation: the position of the point and of the goal. Reward: minus the distance to the goal at every
    step. The episode ends when the point is within ``tolerance`` of the goal (or at the time limit).
    """

    MOVES = np.array([[0, 0], [1, 0], [-1, 0], [0, 1], [0, -1]], np.float32)

    def __init__(self, discrete: bool = False, step_size: float = 0.1, tolerance: float = 0.1):
        self.observation_space = spaces.Box(-1.0, 1.0, (4,), np.float32)
        self.action_space = spaces.Discrete(5) if discrete else spaces.Box(-1.0, 1.0, (2,), np.float32)
        self.discrete, self.step_size, self.tolerance = discrete, step_size, tolerance
        self.pos = self.goal = np.zeros(2)

    def _obs(self) -> np.ndarray:
        return np.concatenate([self.pos, self.goal]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = self.np_random.uniform(-1.0, 1.0, 2)
        self.goal = self.np_random.uniform(-1.0, 1.0, 2)
        return self._obs(), {}

    def step(self, action):
        move = self.MOVES[int(action)] if self.discrete else np.clip(action, -1.0, 1.0)
        self.pos = np.clip(self.pos + self.step_size * move, -1.0, 1.0)
        distance = float(np.linalg.norm(self.pos - self.goal))
        return self._obs(), -distance, distance < self.tolerance, False, {}


# The time limit belongs to the task: TEMPO reads it to set its discount.
gym.register(id="Reach-v0", entry_point=Reach, max_episode_steps=50)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--discrete", action="store_true", help="five discrete moves instead of a continuous one")
    ap.add_argument("--timesteps", type=int, default=20_000, help="training budget in environment steps")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    env_kwargs = dict(discrete=args.discrete)

    model = TEMPO("MlpPolicy", "Reach-v0", env_kwargs=env_kwargs, seed=args.seed, device=args.device, verbose=1)
    eval_env = gym.make("Reach-v0", **env_kwargs)
    before, _ = evaluate_policy(model, eval_env, n_eval_episodes=20)

    model.learn(total_timesteps=args.timesteps)
    after, std = evaluate_policy(model, eval_env, n_eval_episodes=20)
    print(f"\nmean return over 20 episodes: {before:.1f} before training, {after:.1f} +/- {std:.1f} after "
          f"{args.timesteps:,} steps")

    model.save("tempo_reach")                          # tempo_reach.zip
    model = TEMPO.load("tempo_reach")
    obs, _ = eval_env.reset(seed=args.seed)
    action, _ = model.predict(obs, deterministic=True)
    print("action of the loaded model for the first observation:", action)


if __name__ == "__main__":
    main()
