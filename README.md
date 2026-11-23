# TEMPO: Tempered-Entropy Multi-Step Policy Optimization

TEMPO is an off-policy maximum-entropy actor-critic for discrete and continuous control that runs with one fixed
configuration on every task. This repository holds the library (`tempo_rl`) and the code of its benchmark: TEMPO against
SAC, TD3, CrossQ, PPO and DQN (Stable-Baselines3 / sb3-contrib) on 23 Gymnasium tasks, five seeds each.

## Use TEMPO on your task

```bash
pip install "git+https://github.com/danialsj77/TEMPO.git"
```

```python
from tempo_rl import TEMPO

model = TEMPO("MlpPolicy", "LunarLander-v3", verbose=1)   # any Gymnasium task, discrete or continuous actions
model.learn(total_timesteps=200_000)
model.save("tempo_lunarlander")

action, _ = model.predict(observation, deterministic=True)
```

There is no setting for the action type and no hyperparameter to choose for the task: TEMPO reads the action space, the
observation space and the time limit of the environment and follows the training budget. The interface is the one of
Stable-Baselines3 (`learn`, `predict`, `save`, `load`, callbacks, `evaluate_policy`).

- **[User guide](docs/guide.md)**: which spaces are supported, how to pass your own environment, what TEMPO sets for
  you, saving and loading, callbacks, the command line, every parameter.
- **[`examples/quickstart.py`](examples/quickstart.py)**: train, save, load and evaluate on a Gymnasium task.
- **[`examples/custom_env.py`](examples/custom_env.py)**: a task written from scratch and trained with continuous or
  discrete actions by the same code.

## Contents

| Path | What it is |
|---|---|
| `tempo_rl/` | the TEMPO library (Stable-Baselines3-style API: `TEMPO("MlpPolicy", env).learn(...)`) |
| `docs/`, `examples/` | the [user guide](docs/guide.md) and two runnable examples |
| `benchmarks/` | one run (`run_one.py`), the scheduler, the task registry (`plans.py`), the baselines' hyperparameters (`zoo.py`: RL-Baselines3-Zoo entries; `tuned_toy_text.json`: grid-search winners for the toy-text tasks), TEMPO configurations (`variants.py`), statistics (`report.py`) and the random-policy references (`random_scores.json`) |
| `notebooks/` | the harness: `runlab.py` (protocol, jobs, devices, result checks), `benchlab.py`, `figstyle.py`, `tunelab.py` (the toy-text hyperparameter search) |
| `Comparison/` | `complab.py` (comparison statistics, tables, figures) and `timelab.py` (training time of each algorithm measured alone) |
| `scripts/` | `run_all.py` (all runs), `compare.py` (the comparison), `time_all.py` (the timing) |
| `run_all.sh` | starts every run of the benchmark, 10 at a time by default |

## Installation for the benchmark

The library alone installs with the `pip` line above. The benchmark also needs the baselines and the simulators:
Python 3.11 and a CUDA build of PyTorch:

```bash
conda create -n tempo python=3.11 -y
conda activate tempo
pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/<cuda-tag>   # pick <cuda-tag> for your driver on pytorch.org
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## Running the benchmark

Start from a fresh clone, so that every result comes from the same machine.

```bash
bash run_all.sh --dry-run                                   # what would run (495 runs on a fresh clone)
bash run_all.sh --smoke --tasks Pendulum-v1,CartPole-v1     # a two-minute pipeline check in a temporary folder
nohup bash run_all.sh > run_all.log 2>&1 &                  # the benchmark, 10 runs at a time
JOBS=12 nohup bash run_all.sh > run_all.log 2>&1 &          # or 12 at a time
```

Each run trains one algorithm on one task with one seed for 500,000 environment steps and evaluates it every 10,000
steps (10 deterministic episodes on held-out reset seeds; 1,000 on Blackjack). TEMPO and CrossQ run on the GPU (CUDA when
available, else the Apple GPU), SAC, TD3, PPO and DQN on one CPU thread each. Results go to
`TEMPO_test/results/` and `Baselines_test/results/` (one JSON file per run with the whole learning curve), logs to
`TEMPO_test/logs/` and `Baselines_test/logs/`, and the progress to `logs/run_all_status.json`. Running the script again
resumes: finished runs are skipped. Only one copy runs at a time, and it refuses to start when PyTorch sees no GPU
(`--allow-cpu` overrides this). `python scripts/run_all.py --help` lists the options (`--kinds`, `--tasks`, `--gpu-jobs`,
`--carracing`, `--allow-cpu`). CarRacing-v3 is left out by default: it is not part of the comparison and takes days.

After the runs:

```bash
python scripts/time_all.py      # training time of every algorithm measured alone (nothing else may run meanwhile)
python scripts/compare.py       # statistics, tables and figures in Comparison/tables and Comparison/figures
```

## Protocol

- Tasks: classic control (CartPole, Acrobot, MountainCar, Pendulum, MountainCarContinuous), Box2D (LunarLander,
  LunarLanderContinuous, BipedalWalker), MuJoCo v5 (InvertedPendulum, InvertedDoublePendulum, Reacher, Pusher, Swimmer,
  Hopper, Walker2d, HalfCheetah, Ant, Humanoid, HumanoidStandup) and toy text (FrozenLake, CliffWalking with a 200-step
  limit, Blackjack, Taxi); Gymnasium 1.3.0.
- Seeds 10-14 for every algorithm; final score = mean return of the last three evaluations.
- TEMPO: the library defaults, the same configuration on every task.
- Baselines: Stable-Baselines3 / sb3-contrib 2.9.0 with the RL-Baselines3-Zoo hyperparameters of each task; where the Zoo
  has none for an algorithm, its Zoo MuJoCo entry (CrossQ's with the sb3-contrib default critic of 1024 x 1024 units
  instead of the Zoo's 2048 x 2048); PPO and DQN on the toy-text tasks with the winners of a grid search on development
  seeds 0-1 (`notebooks/tunelab.py`).

## License

MIT, see `LICENSE` (third-party notices in `THIRD_PARTY_NOTICES.md`).
