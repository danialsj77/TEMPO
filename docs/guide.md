# TEMPO

Tempered-Entropy Multi-Step Policy Optimization (TEMPO) is an off-policy maximum-entropy actor-critic for discrete and
continuous actions. It builds on Soft Actor-Critic and replaces every quantity that usually needs tuning for a task with
a schedule over training or a rule that reads the task: the entropy target falls during training, a novelty bonus acts
only early, the discount comes from the time limit and the reward scale from the rewards. One configuration is meant for
every task, so using TEMPO on a new task takes an environment and a training budget and nothing else.

The interface follows Stable-Baselines3: `TEMPO(policy, env)`, `learn`, `predict`, `save`, `load`, callbacks and
`evaluate_policy` have the same names and the same meaning.

- [Quick start](#quick-start)
- [Can I use it?](#can-i-use-it)
- [Your own task](#your-own-task)
- [What TEMPO sets for you](#what-tempo-sets-for-you)
- [What you choose](#what-you-choose)
- [Saving, loading and evaluating](#saving-loading-and-evaluating)
- [Callbacks and logging](#callbacks-and-logging)
- [Command line](#command-line)
- [Coming from Stable-Baselines3](#coming-from-stable-baselines3)
- [Parameters](#parameters)
- [Methods](#methods)
- [Questions](#questions)

## Quick start

```bash
pip install "git+https://github.com/danialsj77/TEMPO.git"
```

```python
import gymnasium as gym

from tempo_rl import TEMPO

model = TEMPO("MlpPolicy", "LunarLander-v3", verbose=1)   # the same line for any task
model.learn(total_timesteps=200_000)
model.save("tempo_lunarlander")

del model                                                 # to show saving and loading
model = TEMPO.load("tempo_lunarlander")

env = gym.make("LunarLander-v3", render_mode="human")
obs, info = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
```

Replace `"LunarLander-v3"` by `"Humanoid-v5"`, `"Taxi-v4"` or your own task. Nothing else changes: there is no setting
for the action type and no hyperparameter to choose.

## Can I use it?

| | |
|---|---|
| Discrete actions and continuous actions | yes, with the same configuration |
| Several environments at once | yes (four copies by default; Stable-Baselines3 `VecEnv`, including `SubprocVecEnv`, is accepted) |
| Recurrent policies | no |
| CPU, CUDA GPU, Apple GPU | yes (`device="auto"` picks one) |

Gymnasium spaces:

| Space | Action | Observation |
|---|---|---|
| `Discrete` | yes | yes |
| `Box` | yes | yes (vectors and images) |
| `MultiDiscrete` | no | yes |
| `MultiBinary` | no | yes |
| `Dict`, `Tuple` | no | yes (one level) |

Every observation space in this table has been run with discrete and with continuous actions, including saving and
loading. Three details of the action spaces:

- A `Box` action space must be one-dimensional in shape (a vector) and bounded. TEMPO works in the unit cube
  `[-1, 1]^d` inside and rescales to the bounds of the environment, so the bounds can be any finite numbers.
- A `Discrete` action space must start at 0.
- For `MultiDiscrete` or `MultiBinary` actions with few combinations, a `gymnasium.ActionWrapper` that numbers the
  combinations turns them into one `Discrete` space.

The three policy names `"MlpPolicy"`, `"CnnPolicy"` and `"MultiInputPolicy"` are accepted for familiarity. They are the
same class: the network is always built from the observation space (vectors are normalised, discrete parts are one-hot
encoded, images go through a convolutional encoder, and the parts of a dictionary are joined).

## Your own task

Write the task as a [Gymnasium environment](https://gymnasium.farama.org/introduction/create_custom_env/) and hand it to
TEMPO. [`examples/custom_env.py`](../examples/custom_env.py) is a complete program: a task written from scratch, trained
with continuous or with discrete actions by the same code.

```python
import gymnasium as gym

from tempo_rl import TEMPO

gym.register(id="MyTask-v0", entry_point=MyTask, max_episode_steps=200)   # MyTask is your gymnasium.Env class

model = TEMPO("MlpPolicy", "MyTask-v0", verbose=1)
model.learn(total_timesteps=100_000)
```

There are four ways to pass the environment. They train the same agent; they differ in how many copies collect data.

| You pass | Environments | Notes |
|---|---|---|
| an id, `TEMPO("MlpPolicy", "MyTask-v0")` | 4 | the default; `env_kwargs=dict(...)` goes to `gymnasium.make`, `n_envs` changes the number |
| a vector of copies, `make_vec_env(lambda: MyTask(), n_envs=4)` | as given | no registration needed; `from tempo_rl import make_vec_env` |
| a Stable-Baselines3 `VecEnv` | as given | for example `SubprocVecEnv`, to step slow environments in parallel processes |
| one instance, `TEMPO("MlpPolicy", MyTask())` | 1 | an instance cannot be copied; TEMPO then makes one update every four steps, so learning per step is unchanged and only data collection is slower (a warning says so) |

Three things make a task ready for TEMPO:

1. **A time limit.** TEMPO reads the time limit of the task to set its discount. Give it with `max_episode_steps` when
   you register the task, or wrap an instance in `gymnasium.wrappers.TimeLimit(env, 200)`. Without a time limit the
   discount is 0.99.
2. **`terminated` and `truncated` used as Gymnasium defines them.** Return `terminated=True` only when the episode
   really ends (goal reached, robot fell). The time limit sets `truncated`; TEMPO keeps bootstrapping through it.
3. **Bounded actions** for continuous control (see above).

Rewards need no scaling and observations no normalisation: TEMPO does both.

## What TEMPO sets for you

| Quantity | Read from | Rule |
|---|---|---|
| Policy | action space | `Discrete`: a categorical policy whose expectations are computed exactly. `Box`: a squashed Gaussian in the unit cube, rescaled to the bounds |
| Network inputs | observation space | vectors normalised with running statistics, discrete parts one-hot, images through a CNN |
| Discount | time limit `T` | `clip(1 - 5/T, 0.99, 0.995)`; 0.99 without a time limit |
| Reward scale | rewards seen in training | rewards are divided by a running scale of the discounted return, so the critic sees values in a fixed range whatever the size of the rewards |
| Entropy target | action space and budget | continuous: from 0 to -0.5 nats per action dimension over the first half of the budget, then to -2 by 80 %. Discrete: from 70 % to 5 % of the maximum entropy over the first half |
| Temperature | learned | adjusted so that the policy meets the entropy target |
| Novelty bonus | budget | weight from 5 to 0 over the first half of the budget, then off |
| Random warm-up | budget | 2 % of the budget, at least 1,000 and at most 10,000 steps |
| Updates | number of environments | one update of batch 1024 per step of the four environments (256 samples per environment step) |
| Device | hardware | CUDA if available, else the Apple GPU, else the CPU |

`model.scaling_summary()` returns the update settings that were resolved, `model.gamma` the discount, and
`model.current_target_entropy()` and `model.current_exploration_bonus()` the two schedules at the current step.

## What you choose

- **The budget, `learn(total_timesteps=...)`.** The schedules follow the fraction of the budget that has been used, so
  choose the budget before training and let the run finish. A run that is stopped early has not reached the low entropy
  target of the end of the schedule. If you do not know how long a task needs, start with a short budget and repeat with
  a longer one.
- **The seed**, `TEMPO(..., seed=0)`, for reproducible runs.
- **The device**, if you do not want the automatic choice. The default networks are large (two critics of 1024 units),
  so a GPU is the intended device; a run of 500,000 steps takes about a quarter of an hour on one recent GPU. On a CPU
  TEMPO works unchanged and is slower.

Everything else has a default that is the same for every task. The [parameters](#parameters) can still be changed.

## Saving, loading and evaluating

```python
from tempo_rl import TEMPO, evaluate_policy

model.save("my_agent")                                   # my_agent.zip: networks, optimisers, statistics, settings
model = TEMPO.load("my_agent")                           # for prediction
model = TEMPO.load("my_agent", env="MyTask-v0")          # with an environment: training can continue

mean, std = evaluate_policy(model, gym.make("MyTask-v0"), n_eval_episodes=20)

model.learn(total_timesteps=50_000, reset_num_timesteps=False)    # continue; the schedules span the new total
```

`predict(obs, deterministic=True)` returns the most probable action (the mean for continuous actions), and
`deterministic=False` samples from the policy. During and after training, `predict` answers with an average of the recent
actors (half-life 5,000 steps; not with image observations), which removes most short dips of the evaluated return. `save_replay_buffer` and
`load_replay_buffer` store the collected data separately.

## Callbacks and logging

```python
from tempo_rl import TEMPO, CheckpointCallback, EvalCallback

evaluate = EvalCallback(gym.make("MyTask-v0"), eval_freq=2_500, n_eval_episodes=10,       # every 10,000 steps
                        best_model_save_path="runs/best", log_path="runs/eval")
checkpoint = CheckpointCallback(save_freq=25_000, save_path="runs/checkpoints")           # every 100,000 steps

model = TEMPO("MlpPolicy", "MyTask-v0", verbose=1, tensorboard_log="runs/tb")
model.learn(total_timesteps=500_000, callback=[evaluate, checkpoint], progress_bar=True)
```

As in Stable-Baselines3, `eval_freq` and `save_freq` count steps of the vector of environments: with the default four
environments, 2,500 means every 10,000 environment steps. Available callbacks: `EvalCallback`, `CheckpointCallback`, `EveryNTimesteps`, `StopTrainingOnRewardThreshold`,
`StopTrainingOnMaxEpisodes`, `StopTrainingOnNoModelImprovement`, `CallbackList`, and your own subclass of
`BaseCallback`. With `verbose=1` a table is printed every few episodes: mean episode return and length, steps per
second, the losses, the temperature (`ent_coef`), the current entropy target, the bonus weight and the reward scale.
`tensorboard_log` writes the same values for TensorBoard (`pip install tensorboard`), and `progress_bar=True` needs
`tqdm` and `rich`.

## Command line

```bash
tempo-drl train --env CartPole-v1 --timesteps 50000 --seed 0 --save tempo_cartpole
tempo-drl eval  --model tempo_cartpole.zip --env CartPole-v1 --episodes 20
tempo-drl enjoy --model tempo_cartpole.zip --env CartPole-v1 --render
tempo-drl info                                           # versions and devices, for bug reports
```

`python -m tempo_rl ...` does the same. `tempo-drl train --help` lists the options (`--env-kwargs`, `--eval-freq`,
`--tensorboard`, `--hyperparams key=value ...`).

## Coming from Stable-Baselines3

| Stable-Baselines3 | TEMPO |
|---|---|
| `from stable_baselines3 import SAC` (continuous) or `DQN`, `PPO` (discrete) | `from tempo_rl import TEMPO` (both) |
| `SAC("MlpPolicy", env, **tuned_hyperparameters)` | `TEMPO("MlpPolicy", env)` |
| `model.learn(total_timesteps)`, `model.predict(obs)`, `model.save(path)`, `SAC.load(path)` | the same |
| `evaluate_policy`, `EvalCallback`, `CheckpointCallback`, `make_vec_env` | the same names, imported from `tempo_rl` |
| `VecNormalize` for observations and rewards | not needed: built in |
| `gamma`, `ent_coef`, `target_entropy`, `learning_starts` chosen per task | set by rule (see [above](#what-tempo-sets-for-you)) |

`evaluate_policy` from `tempo_rl` also accepts Stable-Baselines3 models, so one script can evaluate both.

## Parameters

`TEMPO(policy="MlpPolicy", env=None, ...)`. The defaults are the configuration that was used on every task of the
benchmark.

| Parameter | Default | Meaning |
|---|---|---|
| `policy` | `"MlpPolicy"` | `"MlpPolicy"`, `"CnnPolicy"`, `"MultiInputPolicy"` or `"auto"`; the network is built from the observation space in every case |
| `env` | | an environment id, a `gymnasium.Env`, a TEMPO vector environment or a Stable-Baselines3 `VecEnv` |
| `n_envs` | 4 | number of copies when `env` is an id |
| `env_kwargs` | `None` | keyword arguments for `gymnasium.make` when `env` is an id |
| `learning_rate` | 1e-3 | Adam learning rate at batch 256 (a number or a function of the remaining progress); rescaled for the batch size by `lr_rule` |
| `buffer_size` | 1,000,000 | capacity of the replay buffer in transitions; memory is reserved for the budget only, up to this capacity |
| `learning_starts` | `None` | random steps before learning; `None`: 2 % of the budget, between 1,000 and 10,000 |
| `batch_size` | 1024 | samples per update |
| `gamma` | `"auto"` | discount; `"auto"`: `clip(1 - 5/T, 0.99, 0.995)` from the time limit `T`, or a fixed number |
| `train_freq` | `None` | environment steps (per environment) between updates; `None`: so that 256 samples are processed per environment step |
| `gradient_steps` | 1 | updates per training phase |
| `n_step` | 3 | length of the multi-step return (1 gives the one-step target of SAC) |
| `policy_delay` | 3 | the actor and the temperature are updated every `policy_delay` critic updates |
| `ent_coef` | `"auto"` | temperature: learned (`"auto"`, `"auto_0.1"` for another initial value) or a fixed number |
| `target_entropy` | `"auto"` | `"auto"`: the tempered schedule; or a fixed number in nats, or a function of the remaining progress |
| `entropy_schedule` | `None` | the tempered schedule; `None`: `(0, -0.5, 0.5, -2, 0.8)` for continuous actions (nats per dimension: start, middle, fraction, end, fraction) and `(0.7, 0.05, 0.5)` for discrete actions (shares of the maximum entropy: start, end, fraction) |
| `critic_loss` | `"hl_gauss"` | categorical critic trained with the HL-Gauss cross-entropy, or `"mse"` |
| `reward_scaling` | `None` | divide rewards by a running scale of the return; `None`: on with the categorical critic |
| `exploration_bonus` | 5.0 | initial weight of the tempered novelty bonus; 0 switches it off |
| `exploration_noise` | `"pink"` | noise of the continuous policy during data collection: `"pink"` (correlated over an episode) or `"white"` |
| `policy_average` | 5000 | half-life, in environment steps, of the averaged actor that `predict` uses; `None`: the newest actor |
| `policy_kwargs` | `None` | network options, for example `dict(net_arch=dict(pi=[256, 256], qf=[1024, 1024]))` |
| `max_grad_norm` | `None` | gradient clipping (off by default) |
| `tensorboard_log` | `None` | directory for TensorBoard logs |
| `stats_window_size` | 100 | episodes averaged in the training log |
| `verbose` | 0 | 0: silent, 1: training table |
| `seed` | `None` | seed of PyTorch, NumPy, Python and the environments |
| `device` | `"auto"` | `"auto"`, `"cpu"`, `"cuda"` or `"mps"` |

Speed options (`gpu_replay`, `act_on_cpu`, `pipeline`, `fused_adam`, `amp`, `invariance`, `lr_rule`, `ref_batch`,
`brn_impl`) are on by default and are described in the docstring of the class (`help(TEMPO)`). They keep the replay
buffer on the training device, collect data with a CPU copy of the actor while the device trains, and re-count the
settings that depend on the batch size, so that one large update stands for four small ones.

## Methods

| Method | What it does |
|---|---|
| `learn(total_timesteps, callback=None, log_interval=4, tb_log_name="TEMPO", reset_num_timesteps=True, progress_bar=False)` | trains for `total_timesteps` environment steps and returns the model |
| `predict(observation, state=None, episode_start=None, deterministic=False)` | returns `(action, state)` for one observation or a batch |
| `save(path)` / `TEMPO.load(path, env=None, device="auto", custom_objects=None)` | writes and reads `path.zip` |
| `set_env(env)` / `get_env()` | changes or returns the training environment |
| `get_parameters()` / `set_parameters(path_or_dict)` | reads or loads the network parameters |
| `save_replay_buffer(path)` / `load_replay_buffer(path)` | writes and reads the collected data |
| `set_random_seed(seed)` | seeds every random generator |
| `current_target_entropy()` / `current_exploration_bonus()` | the two schedules at the current step |
| `scaling_summary()` | the resolved update settings (batch, updates per step, learning rates) |

## Questions

**How long should I train?** The benchmark used 500,000 steps for every task, and small tasks need far fewer. Because
the schedules follow the budget, a longer budget also means a longer exploration stage.

**My environment is slow.** Pass a Stable-Baselines3 `SubprocVecEnv` so that the copies step in parallel processes.

**My observations are images.** Use the same call. TEMPO builds a convolutional encoder for `uint8` images of shape
`(H, W, C)` or `(C, H, W)` with random-shift augmentation; Atari ids get the standard preprocessing through
`make_vec_env`. Images take much more memory in the replay buffer, so set `buffer_size` to what your machine holds.

**I have no GPU.** Add `device="cpu"`. Short budgets and small tasks are practical; long runs are slow because the
default critics are wide.

**Can I change a hyperparameter?** Yes, every default is an argument. The point of TEMPO is that you should not have to.

**Something does not work.** `tempo-drl info` prints the versions and devices; include its output in the issue.
