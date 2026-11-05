"""Job lists. Each job: algo, variant, env, seed, steps, eval_freq, n_eval, device, suite."""
from __future__ import annotations

SCREEN_ENVS = {  # env: (steps, eval_freq)
    "Humanoid-v5": (300_000, 10_000),
    "Ant-v5": (300_000, 10_000),
    "Hopper-v5": (300_000, 10_000),
    "LunarLander-v3": (150_000, 5_000),
}


def screen1() -> list[dict]:
    jobs = []
    for env, (steps, freq) in SCREEN_ENVS.items():
        for variant, device in (("ln", "cpu"), ("bn256", "cpu"), ("bn512", "mps"), ("bn512_n1", "mps")):
            if variant == "bn512_n1" and env != "Humanoid-v5":
                continue   # dropped for compute (2026-09-25): n-step ablation kept on Humanoid only
            for seed in (0, 1):
                cost = steps * (2.5 if "512" in variant else 1.5 if "bn" in variant else 1.0)
                jobs.append(dict(suite="screen1", algo="TEMPO", variant=variant, env=env, seed=seed,
                                 steps=steps, eval_freq=freq, n_eval=10, device=device, threads=1,
                                 cost=cost))
        if env == "Humanoid-v5":
            for seed in (0, 1):
                jobs.append(dict(suite="screen1", algo="TEMPO", variant="ln_phys", env=env, seed=seed,
                                 steps=steps, eval_freq=freq, n_eval=10, device="cpu", threads=1,
                                 cost=steps))
    return jobs


def smoke() -> list[dict]:
    return [dict(suite="smoke", algo=a, variant=v, env=e, seed=0, steps=s, eval_freq=f, n_eval=3,
                 device="cpu", threads=1)
            for a, v, e, s, f in [("TEMPO", "bn256", "Pendulum-v1", 3000, 1000),
                                  ("TEMPO", "ln", "CartPole-v1", 3000, 1000),
                                  ("SAC", "default", "Pendulum-v1", 3000, 1000),
                                  ("CrossQ", "default", "Hopper-v5", 3000, 1000),
                                  ("PPO", "default", "Hopper-v5", 3000, 1000),
                                  ("DQN", "default", "CartPole-v1", 3000, 1000),
                                  ("TD3", "default", "Pendulum-v1", 3000, 1000)]]


PLANS = {"screen1": screen1, "smoke": smoke}


# ------------------------------------------------------------------ main benchmark
#: env: (budget, eval_freq, family). Budgets are the sample-efficiency regime used
#: throughout the paper; every algorithm gets exactly the same number of env steps.
SUITE = {
    # classic control, discrete
    "CartPole-v1": (50_000, 2_500, "classic-discrete"),
    "Acrobot-v1": (100_000, 5_000, "classic-discrete"),
    "MountainCar-v0": (150_000, 5_000, "classic-discrete"),
    # classic control, continuous
    "Pendulum-v1": (20_000, 1_000, "classic-continuous"),
    "MountainCarContinuous-v0": (50_000, 2_500, "classic-continuous"),
    # Box2D
    "LunarLander-v3": (200_000, 10_000, "box2d-discrete"),
    "LunarLanderContinuous-v3": (200_000, 10_000, "box2d-continuous"),
    "BipedalWalker-v3": (300_000, 10_000, "box2d-continuous"),
    # MuJoCo
    "InvertedPendulum-v5": (30_000, 1_500, "mujoco"),
    "InvertedDoublePendulum-v5": (100_000, 5_000, "mujoco"),
    "Reacher-v5": (100_000, 5_000, "mujoco"),
    "Pusher-v5": (100_000, 5_000, "mujoco"),
    "Swimmer-v5": (300_000, 10_000, "mujoco"),
    "Hopper-v5": (300_000, 10_000, "mujoco"),
    "Walker2d-v5": (300_000, 10_000, "mujoco"),
    "HalfCheetah-v5": (300_000, 10_000, "mujoco"),
    "Ant-v5": (300_000, 10_000, "mujoco"),
    "Humanoid-v5": (300_000, 10_000, "mujoco"),
}
#: TEMPO-only capability checks against known optimal returns (no tuned baselines exist).
TOY_TEXT = {"FrozenLake-v1": (50_000, 2_500, "toy-text"), "CliffWalking-v1": (50_000, 2_500, "toy-text")}
#: extra gymnasium.make kwargs (CliffWalking has no time limit of its own)
ENV_KWARGS = {"CliffWalking-v1": {"max_episode_steps": 200}}
FINAL_SEEDS = (10, 11, 12, 13, 14)
DISCRETE = {"CartPole-v1", "Acrobot-v1", "MountainCar-v0", "LunarLander-v3", "FrozenLake-v1",
            "CliffWalking-v1", "Blackjack-v1", "Taxi-v4"}
#: relative wall-clock cost per env step (for scheduling longest-first)
COST = {"TEMPO": 1.5, "SAC": 1.2, "TD3": 0.7, "CrossQ": 2.5, "PPO": 0.1, "DQN": 0.2}

# ------------------------------------------------------------------ tasks added to the test folders (2026-11-03)
#: Four tasks that Baselines_test / TEMPO_test (notebooks 21-24) and Comparison/ run in addition to the 20 above, since
#: 2026-11-03 (benchmarks/PROTOCOL.md, addendum of 2026-11-03). They were never part of the earlier
#: benchmark, so SUITE / TOY_TEXT stay its record and these tasks have no earlier budget: the test protocol of the run
#: notebooks (notebooks/runlab.py, TEST_BUDGET / TEST_EVAL_FREQ) is their only one. env: family, in notebook order.
#: Taxi-v4 is the Taxi of gymnasium 1.3 (v3 is deprecated there; v4 only fixes the optional rainy / fickle modes).
ADDED = {"CarRacing-v3": "box2d-continuous", "HumanoidStandup-v5": "mujoco", "Blackjack-v1": "toy-text",
         "Taxi-v4": "toy-text"}
#: Every toy-text task of the test folders (earlier + added): judged against an exact optimum, baselines untuned references.
TOY_TEXT_ALL = (*TOY_TEXT, *(e for e, fam in ADDED.items() if fam == "toy-text"))
#: Pixel tasks (tempo_rl.env_util.make_env -> uint8 4x84x84 frames): SB3 uses CnnPolicy there; sb3-contrib CrossQ has
#: MlpPolicy only, so it cannot run on them.
IMAGE = ("CarRacing-v3",)
#: Replay capacity (transitions) of EVERY off-policy learner (TEMPO, SAC, TD3, CrossQ, DQN) on a task where the library
#: defaults (1M) do not fit in memory: 500k CarRacing observations take 14 GB in TEMPO's buffer and 28 GB in SB3's
#: (which stores obs and next_obs) on a 24 GB Mac. 100k is the usual capacity of pixel-based off-policy RL.
REPLAY_CAPACITY = {"CarRacing-v3": 100_000}


def baseline_jobs(algos=None, envs=None, seeds=FINAL_SEEDS, device="cpu") -> list[dict]:
    jobs = []
    for env, (steps, freq, fam) in SUITE.items():
        if envs is not None and env not in envs:
            continue
        pool = ("DQN", "PPO") if env in DISCRETE else ("SAC", "TD3", "CrossQ", "PPO")
        for algo in pool:
            if algos is not None and algo not in algos:
                continue
            for seed in seeds:
                jobs.append(dict(suite="final", algo=algo, variant="default", env=env, seed=seed,
                                 steps=steps, eval_freq=freq, n_eval=10, device=device, threads=1,
                                 cost=steps * COST[algo]))
    return jobs


def tempo_jobs(variant: str, envs=None, seeds=FINAL_SEEDS, device="cpu", toy_text=True) -> list[dict]:
    table = dict(SUITE, **(TOY_TEXT if toy_text else {}))
    return [dict(suite="final", algo="TEMPO", variant=variant, env=env, seed=seed, steps=steps,
                 eval_freq=freq, n_eval=10, device=device, threads=1, cost=steps * COST["TEMPO"],
                 env_kwargs=ENV_KWARGS.get(env))
            for env, (steps, freq, _) in table.items() if envs is None or env in envs
            for seed in seeds]


def baselines_cheap() -> list[dict]:
    """PPO and DQN everywhere, TD3 everywhere (cheap), SAC on the short tasks."""
    short = {e for e, (s, _, _) in SUITE.items() if s <= 100_000}
    return (baseline_jobs(algos=("PPO", "DQN")) + baseline_jobs(algos=("TD3",))
            + baseline_jobs(algos=("SAC",), envs=short))


def baselines_heavy_cpu() -> list[dict]:
    short = {e for e, (s, _, _) in SUITE.items() if s <= 100_000}
    long = set(SUITE) - short
    return baseline_jobs(algos=("SAC",), envs=long) + baseline_jobs(algos=("CrossQ",), envs=short)


def baselines_crossq_mps() -> list[dict]:
    short = {e for e, (s, _, _) in SUITE.items() if s <= 100_000}
    long = set(SUITE) - short
    jobs = baseline_jobs(algos=("CrossQ",), envs=long, device="mps")
    order = ["Humanoid-v5", "Ant-v5", "Hopper-v5", "Walker2d-v5", "HalfCheetah-v5", "BipedalWalker-v3",
             "LunarLanderContinuous-v3", "Swimmer-v5"]
    # CrossQ (1024-wide critics) runs at ~30 steps/s here, so it is scheduled by priority:
    # MuJoCo locomotion x seeds 10-12 first, then seeds 13-14, then the remaining tasks.
    loco = order[:5]
    for j in jobs:
        tier = 3 if (j["env"] in loco and j["seed"] <= 12) else 2 if j["env"] in loco else 1
        j["cost"] = tier * 100 + (len(order) - order.index(j["env"]))
    return jobs


PLANS.update(baselines_cheap=baselines_cheap, baselines_heavy_cpu=baselines_heavy_cpu,
             baselines_crossq_mps=baselines_crossq_mps)


# ------------------------------------------------------------------ screen 2
def screen2() -> list[dict]:
    """Entropy level (unit coordinates) and critic stabilisers on top of the BRN critic."""
    cont = {"Humanoid-v5": (300_000, 10_000, (2, 3)), "Ant-v5": (300_000, 10_000, (2,)),
            "Hopper-v5": (300_000, 10_000, (2,)), "Walker2d-v5": (300_000, 10_000, (2,)),
            "HalfCheetah-v5": (300_000, 10_000, (2,)), "BipedalWalker-v3": (300_000, 10_000, (2,)),
            "LunarLanderContinuous-v3": (200_000, 10_000, (2,))}
    disc = {"LunarLander-v3": (150_000, 5_000, (2,)), "Acrobot-v1": (100_000, 5_000, (2,))}
    jobs = []

    def add(env, steps, freq, seeds, variant):
        for seed in seeds:
            jobs.append(dict(suite="screen2", algo="TEMPO", variant=variant, env=env, seed=seed,
                             steps=steps, eval_freq=freq, n_eval=10, device="cpu", threads=1,
                             # priority: the e05-vs-base comparison first, then the stabilisers
                             cost=steps * {"bn256_e05": 4, "bn256": 3, "bn256_e05_hl": 2}.get(variant, 1)))
    for env, (steps, freq, seeds) in cont.items():
        for v in ("bn256_e05", "bn256_e05_hl", "bn256_e05_hl_lr"):
            add(env, steps, freq, seeds, v)
        if env not in ("Humanoid-v5", "Ant-v5", "Hopper-v5"):    # screen1 has bn256 there
            add(env, steps, freq, seeds, "bn256")
    for env, (steps, freq, seeds) in disc.items():
        for v in ("bn256", "bn256_e05_hl", "bn256_e05_hl_lr"):
            add(env, steps, freq, seeds, v)
    return jobs


PLANS["screen2"] = screen2


def screen2b() -> list[dict]:
    """Amendment 1: HL-Gauss without the e05 entropy schedule."""
    cont = {"Humanoid-v5": (2, 3), "Ant-v5": (2,), "Hopper-v5": (2,), "Walker2d-v5": (2,),
            "HalfCheetah-v5": (2,)}
    return [dict(suite="screen2", algo="TEMPO", variant="bn256_hl", env=env, seed=seed,
                 steps=300_000, eval_freq=10_000, n_eval=10, device="cpu", threads=1, cost=1)
            for env, seeds in cont.items() for seed in seeds]


PLANS["screen2b"] = screen2b


def diag_discrete() -> list[dict]:
    jobs = []
    for v in ("ln", "bn256_d03", "bn256_dfix005", "bn256_n1", "bn256_hl", "bn256_d03_hl"):
        jobs.append(dict(suite="diag", algo="TEMPO", variant=v, env="Acrobot-v1", seed=2,
                         steps=60_000, eval_freq=5_000, n_eval=10, device="cpu", threads=1, cost=1))
    return jobs


PLANS["diag_discrete"] = diag_discrete


# ------------------------------------------------------------------ final runs (2026-09-26)
def gpu_device() -> str | None:
    """The machine's GPU as a torch device name: ``"cuda"`` (NVIDIA) when available, else ``"mps"`` (Apple), else None."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def default_tempo_device() -> str:
    """Where the library-default TEMPO runs: the GPU (:func:`gpu_device`) for wide critics (>= 512 units, the
    rule of ``tempo_rl.tempo.get_device``) when there is one, else the CPU."""
    from tempo_rl.policies import DEFAULT_ARCH

    widest = max([*DEFAULT_ARCH["pi"], *DEFAULT_ARCH["qf"]])
    gpu = gpu_device()
    return gpu if widest >= 512 and gpu else "cpu"


def tempo_final() -> list[dict]:
    """TEMPO with the library defaults (no keyword arguments) on every task, seeds 10-14."""
    return tempo_jobs("default", device=default_tempo_device())


def baselines_sac_long() -> list[dict]:
    short = {e for e, (s, _, _) in SUITE.items() if s <= 100_000}
    return baseline_jobs(algos=("SAC",), envs=set(SUITE) - short)


def baselines_crossq_short_mps() -> list[dict]:
    short = {e for e, (s, _, _) in SUITE.items() if s <= 100_000}
    return baseline_jobs(algos=("CrossQ",), envs=short, device="mps")


PLANS.update(tempo_final=tempo_final, baselines_sac_long=baselines_sac_long,
             baselines_crossq_short_mps=baselines_crossq_short_mps)


# ------------------------------------------------------------------ screen 3 (2026-09-26)
LOCO = ("Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Walker2d-v5", "Hopper-v5")


def screen3() -> list[dict]:
    """TEMPO vs CrossQ at equal critic width; development seeds 4-5, 300k steps."""
    jobs = []
    variants = ("t1024", "t1024_e1", "t1024_mse", "t1024_n1")
    for i, env in enumerate(LOCO):
        for v in variants:
            for seed in (4, 5):
                # half the jobs on MPS (faster per job), the rest on CPU with 2 threads
                dev = "mps" if (seed == 4) else "cpu"
                jobs.append(dict(suite="screen3", algo="TEMPO", variant=v, env=env, seed=seed,
                                 steps=300_000, eval_freq=10_000, n_eval=10, device=dev,
                                 threads=1 if dev == "mps" else 2,
                                 cost=(4 - variants.index(v)) * 100 + (10 - i)))
    return jobs


PLANS["screen3"] = screen3


# ------------------------------------------------------------------ notebook helpers
def custom_jobs(suite: str, algo: str = "TEMPO", variants=("default",), envs=None, seeds=(10,),
                device: str = "cpu", threads: int = 1, smoke: bool = False,
                smoke_steps: int = 3000) -> list[dict]:
    """Job list for any (variants x tasks x seeds). ``smoke=True`` shrinks every run to
    ``smoke_steps`` steps (3 evaluations) so a whole notebook runs in minutes; smoke
    suites are prefixed ``smoke_`` so they never mix with real results."""
    table = dict(SUITE, **TOY_TEXT)
    envs = list(table) if envs is None else list(envs)
    jobs = []
    for env in envs:
        steps, freq, _ = table[env]
        if smoke:
            steps, freq = smoke_steps, max(1, smoke_steps // 3)
        for v in variants:
            for seed in seeds:
                jobs.append(dict(suite=("smoke_" + suite) if smoke else suite, algo=algo,
                                 variant=v, env=env, seed=int(seed), steps=steps, eval_freq=freq,
                                 n_eval=3 if smoke else 10, device=device, threads=threads,
                                 cost=steps * COST.get(algo, 1.0), env_kwargs=ENV_KWARGS.get(env)))
    return jobs


def option_a(smoke: bool = False) -> list[dict]:
    """Option A: TEMPO at CrossQ's critic width (1024) on the locomotion tasks, dev seeds 4-5."""
    return custom_jobs("screen3", variants=("t1024", "t1024_e1", "t1024_mse", "t1024_n1"),
                       envs=LOCO, seeds=(4, 5), smoke=smoke)


def option_b(smoke: bool = False) -> list[dict]:
    """Option B: learned representations and 2 updates/step + weight norm, dev seeds 4-5.

    Its reference was the library default of 2026-09-26, the 256-unit recipe, pinned since the
    default changed as ``bn256_e05_hl`` (never run; superseded by Screens 5 and 6)."""
    return custom_jobs("screen4", variants=("bn256_e05_hl", "sale", "wn_utd2", "sale_wn_utd2"),
                       envs=LOCO, seeds=(4, 5), smoke=smoke)


def missing_baselines(smoke: bool = False) -> list[dict]:
    """Every SB3 baseline run of the final suite that has no result yet."""
    from .run_one import result_path
    jobs = baselines_cheap() + baselines_sac_long() + baselines_crossq_mps() + \
        baselines_crossq_short_mps()
    if smoke:
        out = []
        for j in jobs:
            if j["seed"] == 10 and j["env"] in ("Pendulum-v1", "CartPole-v1"):
                out.append(dict(j, suite="smoke_final", steps=3000, eval_freq=1000, n_eval=3))
        return out
    return [j for j in jobs if not result_path(j).exists()]


PLANS.update(option_a=option_a, option_b=option_b, missing_baselines=missing_baselines)


# ------------------------------------------------------------------ beat-all screen (2026-09-28)
#: tasks where TEMPO lost, tied narrowly or is at risk (sparse rewards / long horizons)
BEAT_ALL_TASKS = ("Humanoid-v5", "Ant-v5", "HalfCheetah-v5", "Walker2d-v5", "Hopper-v5",
                  "Swimmer-v5", "BipedalWalker-v3", "LunarLanderContinuous-v3",
                  "MountainCarContinuous-v0", "Acrobot-v1", "MountainCar-v0")
BEAT_ALL_VARIANTS = ("tall", "tall_gauto", "t512_sale_gauto")


def beat_all(smoke: bool = False, variants=BEAT_ALL_VARIANTS, seeds=(6, 7)) -> list[dict]:
    """Development screen for 'beat every baseline': new dev seeds 6-7; 1024-wide runs on MPS."""
    jobs = custom_jobs("screen5", variants=variants, envs=BEAT_ALL_TASKS, seeds=seeds, smoke=smoke)
    for j in jobs:
        heavy = "1024" in str(__import__("benchmarks.variants", fromlist=["x"]).tempo_kwargs(
            j["variant"], j["env"]).get("policy_kwargs", {}).get("net_arch", ""))
        j["device"] = "mps" if heavy else "cpu"
    return jobs


PLANS["beat_all"] = beat_all


# ------------------------------------------------------------------ screen 6 (2026-09-29)
def screen6(smoke: bool = False) -> list[dict]:
    """C1 on every hard task; C2 (n=1) and C3 (gamma 0.999) where they are expected to matter."""
    jobs = custom_jobs("screen6", variants=("c1",), envs=BEAT_ALL_TASKS, seeds=(8, 9), smoke=smoke)
    jobs += custom_jobs("screen6", variants=("c2",), envs=("Humanoid-v5", "Hopper-v5", "Walker2d-v5",
                                                       "Ant-v5"), seeds=(8, 9), smoke=smoke)
    jobs += custom_jobs("screen6", variants=("c3",), envs=("Swimmer-v5", "Hopper-v5", "Ant-v5",
                                                       "HalfCheetah-v5", "Walker2d-v5"),
                        seeds=(8, 9), smoke=smoke)
    for j in jobs:
        j["device"] = "mps"          # all candidates use the 1024-wide critic
    return jobs


PLANS["screen6"] = screen6
