"""Train TEMPO on a Gymnasium task, save it, load it and evaluate it.

    python examples/quickstart.py                                  # CartPole-v1, 30,000 steps
    python examples/quickstart.py --env Pendulum-v1 --timesteps 20000
    python examples/quickstart.py --env LunarLander-v3 --timesteps 200000 --render

The call is the same for discrete actions (CartPole, LunarLander) and continuous actions (Pendulum).
"""
from __future__ import annotations

import argparse

import gymnasium as gym

from tempo_rl import TEMPO, EvalCallback, evaluate_policy


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", default="CartPole-v1", help="a Gymnasium environment id")
    ap.add_argument("--timesteps", type=int, default=30_000, help="training budget in environment steps")
    ap.add_argument("--device", default="auto", help="auto, cpu, cuda or mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--render", action="store_true", help="show the trained agent in a window")
    args = ap.parse_args()

    # an evaluation every tenth of the budget on a separate copy of the task
    evaluate = EvalCallback(gym.make(args.env), eval_freq=max(args.timesteps // 40, 1), n_eval_episodes=5, verbose=1)

    model = TEMPO("MlpPolicy", args.env, seed=args.seed, device=args.device, verbose=1)
    model.learn(total_timesteps=args.timesteps, callback=evaluate)
    model.save(f"tempo_{args.env}")

    model = TEMPO.load(f"tempo_{args.env}", device=args.device)
    mean, std = evaluate_policy(model, gym.make(args.env), n_eval_episodes=20)
    print(f"\n{args.env}: mean return {mean:.1f} +/- {std:.1f} over 20 episodes after {args.timesteps:,} steps")

    if args.render:
        env = gym.make(args.env, render_mode="human")
        obs, _ = env.reset(seed=args.seed)
        for _ in range(2000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                obs, _ = env.reset()
        env.close()


if __name__ == "__main__":
    main()
