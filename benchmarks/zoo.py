"""RL Baselines3 Zoo hyperparameters (rl-baselines3-zoo master @ 630883a, fetched 2026-09-24).

Copied verbatim from hyperparams/{sac,td3,crossq,ppo,dqn}.yml with YAML anchors
resolved. The zoo's MuJoCo keys are *-v4* (and *-v2* for PPO on InvertedPendulum,
InvertedDoublePendulum, Reacher and HumanoidStandup); they are applied to the *-v5*
environments here (same robots, same reward structure). ``n_timesteps`` is
dropped: every algorithm gets the benchmark's budget. Strings ``"lin_X"`` are
linear schedules from X to 0 over that budget, as in the zoo.

Every baseline is tuned (since 2026-11-04; benchmarks/PROTOCOL.md): a MuJoCo
task without an entry for an algorithm gets that algorithm's zoo MuJoCo entry (the
``mujoco-defaults`` anchor the zoo applies to all its MuJoCo tasks), and the toy-text
tasks, which the zoo does not cover, get the PPO / DQN hyperparameters selected by the
grid search of notebooks/tunelab.py (``tuned_toy_text.json``; until that file exists
they use the library defaults, the zoo's ``default``). :func:`hp_source` names the source.
"""
import json
import pathlib

import torch.nn as nn

_MUJOCO = ["HalfCheetah", "Ant", "Hopper", "Walker2d", "Humanoid", "Swimmer"]
# MuJoCo tasks without a zoo entry for SAC / TD3 / CrossQ (2026-11-04, every baseline tuned): they get the algorithm's zoo
# MuJoCo entry, the `mujoco-defaults` anchor the zoo applies to all its MuJoCo tasks. HumanoidStandup (added 2026-11-03):
# crossq.yml has HumanoidStandup-v4 (that anchor) and ppo.yml a HumanoidStandup-v2 entry; sac.yml and td3.yml have none.
# PPO has -v2 entries for the first three and none for Pusher (its MuJoCo anchor, see PPO).
_MUJOCO_NO_ENTRY = ["InvertedPendulum", "InvertedDoublePendulum", "Reacher", "Pusher", "HumanoidStandup"]

SAC = {
    "MountainCarContinuous-v0": dict(learning_rate=3e-4, buffer_size=50_000, batch_size=512, ent_coef=0.1,
                                     train_freq=32, gradient_steps=32, gamma=0.9999, tau=0.01,
                                     learning_starts=0, use_sde=True,
                                     policy_kwargs=dict(log_std_init=-3.67, net_arch=[64, 64])),
    "Pendulum-v1": dict(learning_rate=1e-3),
    "LunarLanderContinuous-v3": dict(batch_size=256, learning_rate="lin_7.3e-4", buffer_size=1_000_000,
                                     ent_coef="auto", gamma=0.99, tau=0.01, train_freq=1, gradient_steps=1,
                                     learning_starts=10_000, policy_kwargs=dict(net_arch=[400, 300])),
    "BipedalWalker-v3": dict(learning_rate=7.3e-4, buffer_size=300_000, batch_size=256, ent_coef="auto",
                             gamma=0.98, tau=0.02, train_freq=64, gradient_steps=64, learning_starts=10_000,
                             use_sde=True, policy_kwargs=dict(log_std_init=-3, net_arch=[400, 300])),
}
for _e in (*_MUJOCO, *_MUJOCO_NO_ENTRY):
    SAC[f"{_e}-v5"] = dict(learning_starts=10_000)
SAC["Swimmer-v5"] = dict(learning_starts=10_000, gamma=0.999)

_TD3_MUJOCO = dict(learning_starts=10_000, noise_type="normal", noise_std=0.1, train_freq=1, gradient_steps=1,
                   learning_rate=1e-3, batch_size=256, policy_kwargs=dict(net_arch=[400, 300]))
TD3 = {
    "default": dict(train_freq=1, gradient_steps=1, noise_type="normal", noise_std=0.1),
    "MountainCarContinuous-v0": dict(noise_type="ornstein-uhlenbeck", noise_std=0.5, gradient_steps=1,
                                     train_freq=1, learning_rate=1e-3, batch_size=256,
                                     policy_kwargs=dict(net_arch=[400, 300])),
    "Pendulum-v1": dict(gamma=0.98, buffer_size=200_000, learning_starts=10_000, noise_type="normal",
                        noise_std=0.1, gradient_steps=1, train_freq=1, learning_rate=1e-3,
                        policy_kwargs=dict(net_arch=[400, 300])),
    "LunarLanderContinuous-v3": dict(gamma=0.98, buffer_size=200_000, learning_starts=10_000,
                                     noise_type="normal", noise_std=0.1, gradient_steps=1, train_freq=1,
                                     learning_rate=1e-3, policy_kwargs=dict(net_arch=[400, 300])),
    "BipedalWalker-v3": dict(gamma=0.98, buffer_size=200_000, learning_starts=10_000, noise_type="normal",
                             noise_std=0.1, gradient_steps=1, train_freq=1, learning_rate=1e-3,
                             policy_kwargs=dict(net_arch=[400, 300])),
}
for _e in (*_MUJOCO, *_MUJOCO_NO_ENTRY):
    TD3[f"{_e}-v5"] = dict(_TD3_MUJOCO)
TD3["Swimmer-v5"] = dict(_TD3_MUJOCO, gamma=0.9999)

# sb3-contrib CrossQ. The zoo's MuJoCo entry uses 2048x2048 critics; we use the
# library default 1024x1024 (the sb3-contrib authors' own speed/accuracy choice,
# and TEMPO's critic width).
_CROSSQ_MUJOCO = dict(buffer_size=1_000_000, learning_rate=1e-3, learning_starts=5000, policy_delay=3,
                      policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[1024, 1024])))
CROSSQ = {
    "MountainCarContinuous-v0": dict(learning_rate=7e-4, buffer_size=50_000, train_freq=32, gradient_steps=32,
                                     gamma=0.9999, learning_starts=100, use_sde=True, policy_delay=2,
                                     policy_kwargs=dict(use_expln=True, log_std_init=-1, net_arch=[64, 64])),
    "Pendulum-v1": dict(policy_delay=2, policy_kwargs=dict(net_arch=[256, 256])),
    "LunarLanderContinuous-v3": dict(buffer_size=1_000_000, learning_starts=10_000),
    "BipedalWalker-v3": dict(buffer_size=300_000, gamma=0.98, learning_starts=10_000,
                             policy_kwargs=dict(net_arch=dict(pi=[256, 256], qf=[1024, 1024]))),
}
for _e in (*_MUJOCO, *_MUJOCO_NO_ENTRY):
    CROSSQ[f"{_e}-v5"] = dict(_CROSSQ_MUJOCO)
CROSSQ["Swimmer-v5"] = dict(_CROSSQ_MUJOCO, gamma=0.999)

_PPO_TUNED_POLICY = dict(log_std_init=-2, ortho_init=False, activation_fn=nn.ReLU,
                         net_arch=dict(pi=[256, 256], vf=[256, 256]))
PPO = {
    "CartPole-v1": dict(n_envs=8, n_steps=32, batch_size=256, gae_lambda=0.8, gamma=0.98, n_epochs=20,
                        ent_coef=0.0, learning_rate="lin_0.001", clip_range="lin_0.2"),
    "Acrobot-v1": dict(normalize=True, n_envs=16, n_steps=256, gae_lambda=0.94, gamma=0.99, n_epochs=4,
                       ent_coef=0.0),
    "MountainCar-v0": dict(normalize=True, n_envs=16, n_steps=16, gae_lambda=0.98, gamma=0.99, n_epochs=4,
                           ent_coef=0.0),
    "MountainCarContinuous-v0": dict(normalize=True, n_envs=1, batch_size=256, n_steps=8, gamma=0.9999,
                                     learning_rate=7.77e-05, ent_coef=0.00429, clip_range=0.1, n_epochs=10,
                                     gae_lambda=0.9, max_grad_norm=5, vf_coef=0.19, use_sde=True,
                                     policy_kwargs=dict(log_std_init=-3.29, ortho_init=False)),
    "Pendulum-v1": dict(n_envs=4, n_steps=1024, gae_lambda=0.95, gamma=0.9, n_epochs=10, ent_coef=0.0,
                        learning_rate=1e-3, clip_range=0.2, use_sde=True, sde_sample_freq=4),
    "LunarLander-v3": dict(n_envs=16, n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999, n_epochs=4,
                           ent_coef=0.01),
    "LunarLanderContinuous-v3": dict(n_envs=16, n_steps=1024, batch_size=64, gae_lambda=0.98, gamma=0.999,
                                     n_epochs=4, ent_coef=0.01),
    "BipedalWalker-v3": dict(normalize=True, n_envs=32, n_steps=2048, batch_size=64, gae_lambda=0.95,
                             gamma=0.999, n_epochs=10, ent_coef=0.0, learning_rate=3e-4, clip_range=0.18),
    "Ant-v5": dict(normalize=True),
    "Swimmer-v5": dict(normalize=True, gamma=0.9999, n_envs=4, n_steps=1024, batch_size=256,
                       learning_rate=6e-4, gae_lambda=0.98),
    "HalfCheetah-v5": dict(normalize=True, n_envs=1, batch_size=64, n_steps=512, gamma=0.98,
                           learning_rate=2.0633e-05, ent_coef=0.000401762, clip_range=0.1, n_epochs=20,
                           gae_lambda=0.92, max_grad_norm=0.8, vf_coef=0.58096, policy_kwargs=_PPO_TUNED_POLICY),
    "Hopper-v5": dict(normalize=True, n_envs=1, batch_size=32, n_steps=512, gamma=0.999,
                      learning_rate=9.80828e-05, ent_coef=0.00229519, clip_range=0.2, n_epochs=5,
                      gae_lambda=0.99, max_grad_norm=0.7, vf_coef=0.835671, policy_kwargs=_PPO_TUNED_POLICY),
    "Walker2d-v5": dict(normalize=True, n_envs=1, batch_size=32, n_steps=512, gamma=0.99,
                        learning_rate=5.05041e-05, ent_coef=0.000585045, clip_range=0.1, n_epochs=20,
                        gae_lambda=0.95, max_grad_norm=1, vf_coef=0.871923),
    "Humanoid-v5": dict(normalize=True, n_envs=1, batch_size=256, n_steps=512, gamma=0.95,
                        learning_rate=3.56987e-05, ent_coef=0.00238306, clip_range=0.3, n_epochs=5,
                        gae_lambda=0.9, max_grad_norm=2, vf_coef=0.431892, policy_kwargs=_PPO_TUNED_POLICY),
    # added 2026-11-03: the zoo's only HumanoidStandup entry is the -v2 key (same robot and reward), applied to -v5
    "HumanoidStandup-v5": dict(normalize=True, n_envs=1, batch_size=32, n_steps=512, gamma=0.99,
                               learning_rate=2.55673e-05, ent_coef=3.62109e-06, clip_range=0.3, n_epochs=20,
                               gae_lambda=0.9, max_grad_norm=0.7, vf_coef=0.430793, policy_kwargs=_PPO_TUNED_POLICY),
    # added 2026-11-04 (every baseline tuned): the zoo's -v2 entries of these three tasks (same robots and rewards)
    "InvertedPendulum-v5": dict(normalize=True, n_envs=1, batch_size=64, n_steps=32, gamma=0.999,
                                learning_rate=0.000222425, ent_coef=1.37976e-07, clip_range=0.4, n_epochs=5,
                                gae_lambda=0.9, max_grad_norm=0.3, vf_coef=0.19816),
    "InvertedDoublePendulum-v5": dict(normalize=True, n_envs=1, batch_size=512, n_steps=128, gamma=0.98,
                                      learning_rate=0.000155454, ent_coef=1.05057e-06, clip_range=0.4, n_epochs=10,
                                      gae_lambda=0.8, max_grad_norm=0.5, vf_coef=0.695929),
    "Reacher-v5": dict(normalize=True, n_envs=1, batch_size=32, n_steps=512, gamma=0.9,
                       learning_rate=0.000104019, ent_coef=7.52585e-08, clip_range=0.3, n_epochs=5,
                       gae_lambda=1.0, max_grad_norm=0.9, vf_coef=0.950368),
    # Pusher has no zoo entry: the zoo's PPO MuJoCo anchor (`mujoco-defaults`: normalize, else defaults), as its Ant entry
    "Pusher-v5": dict(normalize=True),
    # added 2026-11-03: the zoo's CarRacing-v3 entry without its env_wrapper (frame-skip 2, 64x64 grey) and frame_stack 2:
    # every algorithm sees the shared pixel pipeline of tempo_rl.env_util.make_env instead (grey 84x84 crop, frame-skip
    # 4, 4-frame stack); its normalize dict normalises the rewards only (VecNormalize on images would break the CNN)
    "CarRacing-v3": dict(normalize=dict(norm_obs=False, norm_reward=True), n_envs=8, batch_size=128, n_steps=512,
                         gamma=0.99, gae_lambda=0.95, n_epochs=10, ent_coef=0.0, sde_sample_freq=4, max_grad_norm=0.5,
                         vf_coef=0.5, learning_rate="lin_1e-4", use_sde=True, clip_range=0.2,
                         policy_kwargs=dict(log_std_init=-2, ortho_init=False, activation_fn=nn.GELU,
                                            net_arch=dict(pi=[256], vf=[256]))),
}

DQN = {
    "CartPole-v1": dict(learning_rate=2.3e-3, batch_size=64, buffer_size=100_000, learning_starts=1000,
                        gamma=0.99, target_update_interval=10, train_freq=256, gradient_steps=128,
                        exploration_fraction=0.16, exploration_final_eps=0.04,
                        policy_kwargs=dict(net_arch=[256, 256])),
    "MountainCar-v0": dict(learning_rate=4e-3, batch_size=128, buffer_size=10_000, learning_starts=1000,
                           gamma=0.98, target_update_interval=600, train_freq=16, gradient_steps=8,
                           exploration_fraction=0.2, exploration_final_eps=0.07,
                           policy_kwargs=dict(net_arch=[256, 256])),
    "LunarLander-v3": dict(learning_rate=6.3e-4, batch_size=128, buffer_size=50_000, learning_starts=0,
                           gamma=0.99, target_update_interval=250, train_freq=4, gradient_steps=-1,
                           exploration_fraction=0.12, exploration_final_eps=0.1,
                           policy_kwargs=dict(net_arch=[256, 256])),
    "Acrobot-v1": dict(learning_rate=6.3e-4, batch_size=128, buffer_size=50_000, learning_starts=0,
                       gamma=0.99, target_update_interval=250, train_freq=4, gradient_steps=-1,
                       exploration_fraction=0.12, exploration_final_eps=0.1,
                       policy_kwargs=dict(net_arch=[256, 256])),
}

ZOO = {"SAC": SAC, "TD3": TD3, "CrossQ": CROSSQ, "PPO": PPO, "DQN": DQN}


#: (algorithm, task) pairs whose entry above is the algorithm's zoo MuJoCo anchor because the zoo has none for the task.
ANCHOR_ONLY = ({(a, f"{e}-v5") for a in ("SAC", "TD3") for e in _MUJOCO_NO_ENTRY}
               | {("CrossQ", f"{e}-v5") for e in _MUJOCO_NO_ENTRY if e != "HumanoidStandup"} | {("PPO", "Pusher-v5")})

#: PPO / DQN hyperparameters of the toy-text tasks (the zoo has none): the winners of the grid search of notebooks/tunelab.py
#: on development seeds 0-1 (Baselines_test/tune_toy_text.ipynb writes this file; benchmarks/PROTOCOL.md, 2026-11-04).
TUNED_HERE = pathlib.Path(__file__).with_name("tuned_toy_text.json")

#: Off-policy algorithms (they keep a replay buffer).
OFF_POLICY = ("SAC", "TD3", "CrossQ", "DQN")


def tuned_here() -> dict:
    """``{algo: {env_id: hyperparameters}}`` from :data:`TUNED_HERE`; empty while the grid search has not written it."""
    try:
        data = json.loads(TUNED_HERE.read_text())
    except FileNotFoundError:
        return {}
    return {a: {e: dict(d["hp"]) for e, d in v.items()} for a, v in data.items() if not a.startswith("_")}


def zoo_hyperparams(algo: str, env_id: str) -> tuple[dict, bool]:
    """(hyperparameters, tuned?): the zoo's entry for the task (:data:`ZOO`), else the grid-search winner of a toy-text
    task (:func:`tuned_here`), else the zoo's ``default`` entry (untuned). On a task of
    ``benchmarks.plans.REPLAY_CAPACITY`` an off-policy algorithm's ``buffer_size`` is capped at that
    capacity (a memory limit of the harness, not an RL-Zoo hyperparameter: ``tuned`` is unchanged)."""
    from .plans import REPLAY_CAPACITY

    table, here = ZOO[algo], tuned_here().get(algo, {})
    if env_id in table:
        hp, tuned = dict(table[env_id]), True
    elif env_id in here:
        hp, tuned = dict(here[env_id]), True
    else:
        hp, tuned = dict(table.get("default", {})), False
    if algo in OFF_POLICY and env_id in REPLAY_CAPACITY:
        hp["buffer_size"] = min(int(hp.get("buffer_size", REPLAY_CAPACITY[env_id])), REPLAY_CAPACITY[env_id])
    return hp, tuned


def hp_source(algo: str, env_id: str, short: bool = False) -> str:
    """Where :func:`zoo_hyperparams` takes ``algo``'s hyperparameters on ``env_id`` from (``short``: a few words)."""
    if env_id in ZOO[algo]:
        if (algo, env_id) in ANCHOR_ONLY:
            return "zoo MuJoCo entry" if short else "the algorithm's RL-Zoo MuJoCo entry, as the zoo has none for this task"
        return "zoo entry" if short else "RL-Zoo entry for this task"
    if env_id in tuned_here().get(algo, {}):
        return "grid search" if short else "grid search on development seeds 0-1, benchmarks/tuned_toy_text.json"
    return "library defaults" if short else ("library defaults" + (" + the zoo's 'default' entry" if ZOO[algo].get("default")
                                                                    else "") + " (untuned)")
