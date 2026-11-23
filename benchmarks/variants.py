"""Named TEMPO configurations used in the screening and benchmark runs."""
from __future__ import annotations

_BN = dict(learning_rate=1e-3, policy_delay=3, critic_loss="mse", n_step=3,
           entropy_schedule_continuous=(-0.3, -1.0, 0.5))

#: The screening variants are pinned explicitly (they were run with the library defaults
#: of their time, which later changed): LayerNorm critics + Polyak target networks,
#: MSE critic, lr 3e-4, no policy delay, entropy schedule -0.3 -> -1.0 nats/dim.
_LN = dict(learning_rate=3e-4, policy_delay=1, critic_loss="mse", n_step=3,
           entropy_schedule_continuous=(-0.3, -1.0, 0.5),
           policy_kwargs=dict(batch_norm=False, critic_layer_norm=True,
                              net_arch=dict(pi=[256, 256], qf=[256, 256])))

VARIANTS: dict[str, dict] = {
    # LayerNorm critics + Polyak target networks (the first TEMPO recipe, with
    # unit-coordinate entropy, observation normalisation and a 1M buffer).
    "ln": dict(_LN),
    # same, but entropy measured in physical action coordinates (first-version convention)
    "ln_phys": dict(_LN, _physical_entropy=True),
    # Batch-renormalised critics, joint forward pass, no target networks (CrossQ-style)
    "bn256": dict(_BN, policy_kwargs=dict(batch_norm=True, net_arch=dict(pi=[256, 256], qf=[256, 256]))),
    "bn512": dict(_BN, policy_kwargs=dict(batch_norm=True, net_arch=dict(pi=[256, 256], qf=[512, 512]))),
    "bn512_n1": dict(_BN, n_step=1,
                     policy_kwargs=dict(batch_norm=True, net_arch=dict(pi=[256, 256], qf=[512, 512]))),
}

# ---- screen 2: entropy level (scale-invariant, more exploratory) and critic stabilisers
def _bn(width: int, **kw) -> dict:
    out = dict(_BN, critic_loss="mse", n_step=3, entropy_schedule_continuous=(-0.3, -1.0, 0.5),
               policy_kwargs=dict(batch_norm=True, net_arch=dict(pi=[256, 256], qf=[width, width])))
    out.update(kw)
    if "entropy_schedule" in kw:
        out["entropy_schedule_continuous"] = out.pop("entropy_schedule")
    return out


for _w in (256, 512):
    VARIANTS[f"bn{_w}_e05"] = _bn(_w, entropy_schedule=(0.0, -0.5, 0.5))
    VARIANTS[f"bn{_w}_hl"] = _bn(_w, critic_loss="hl_gauss")
    VARIANTS[f"bn{_w}_e05_hl"] = _bn(_w, entropy_schedule=(0.0, -0.5, 0.5), critic_loss="hl_gauss")
    VARIANTS[f"bn{_w}_e05_hl_lr"] = _bn(_w, entropy_schedule=(0.0, -0.5, 0.5), critic_loss="hl_gauss",
                                        learning_rate=("linear", 1e-3, 1e-4))
    VARIANTS[f"bn{_w}_e05_lr"] = _bn(_w, entropy_schedule=(0.0, -0.5, 0.5),
                                     learning_rate=("linear", 1e-3, 1e-4))

# ---- discrete exploration diagnosis (Acrobot / MountainCar), 2026-09-25
VARIANTS["bn256_d03"] = _bn(256, entropy_schedule_discrete=(0.3, 0.05, 0.5))
VARIANTS["bn256_dfix005"] = _bn(256, entropy_schedule_discrete=(0.05, 0.05, 0.5))
VARIANTS["bn256_n1"] = _bn(256, n_step=1)
VARIANTS["bn256_d03_hl"] = _bn(256, entropy_schedule_discrete=(0.3, 0.05, 0.5), critic_loss="hl_gauss")

# ---- screen 3 (2026-09-26): beat CrossQ at equal critic width (1024)
_W1024 = dict(policy_kwargs=dict(batch_norm=True, net_arch=dict(pi=[256, 256], qf=[1024, 1024])))
VARIANTS["t1024"] = _bn(1024, entropy_schedule=(0.0, -0.5, 0.5), critic_loss="hl_gauss")           # A
VARIANTS["t1024_e1"] = _bn(1024, entropy_schedule=(-0.3, -1.0, 0.5), critic_loss="hl_gauss")       # B
VARIANTS["t1024_mse"] = _bn(1024, entropy_schedule=(0.0, -0.5, 0.5), critic_loss="mse")            # C
VARIANTS["t1024_n1"] = _bn(1024, entropy_schedule=(0.0, -0.5, 0.5), critic_loss="hl_gauss", n_step=1)  # D

#: the library's current defaults, untouched: what `TEMPO("MlpPolicy", env)` does
VARIANTS["default"] = {}


def tempo_kwargs(variant: str, env_id: str) -> dict:
    import copy

    if variant not in VARIANTS:
        raise KeyError(f"unknown TEMPO variant {variant!r}; known: {sorted(VARIANTS)}")
    kw = copy.deepcopy(VARIANTS[variant])
    from .plans import DISCRETE
    sched = kw.pop("entropy_schedule_continuous", None)
    dsched = kw.pop("entropy_schedule_discrete", None)
    n_cont = kw.pop("n_step_continuous", None)
    if n_cont is not None and env_id not in DISCRETE:   # continuous actions only (discrete ones keep "n_step")
        kw["n_step"] = tuple(n_cont) if isinstance(n_cont, (tuple, list)) else int(n_cont)
    if sched is not None and env_id not in DISCRETE:   # continuous schedules only
        kw["entropy_schedule"] = tuple(sched)
    if dsched is not None and env_id in DISCRETE:
        kw["entropy_schedule"] = tuple(dsched)
    lr = kw.get("learning_rate")
    if isinstance(lr, tuple) and lr[0] == "linear":
        from tempo_rl.utils import LinearSchedule
        kw["learning_rate"] = LinearSchedule(lr[1], lr[2])
    from .plans import REPLAY_CAPACITY
    if env_id in REPLAY_CAPACITY:   # memory limit of the harness on pixel tasks (every off-policy learner gets it)
        kw["buffer_size"] = min(int(kw.get("buffer_size", REPLAY_CAPACITY[env_id])), REPLAY_CAPACITY[env_id])
    return kw


# ---- option B (2026-09-26): ingredients CrossQ lacks, on top of a base configuration
def _with(base: dict, **policy_kwargs_extra) -> dict:
    import copy

    out = copy.deepcopy(base)
    pk = dict(out.get("policy_kwargs", {}))
    pk.update(policy_kwargs_extra)
    out["policy_kwargs"] = pk
    return out


VARIANTS["sale"] = _with({}, representation="sale")                       # learned embeddings
VARIANTS["wn_utd2"] = _with(dict(gradient_steps=2), weight_norm=True)      # 2 updates/step + WN
VARIANTS["sale_wn_utd2"] = _with(dict(gradient_steps=2), representation="sale", weight_norm=True)
VARIANTS["t1024_sale"] = _with(VARIANTS["t1024"], representation="sale")
VARIANTS["t1024_wn_utd2"] = _with(dict(VARIANTS["t1024"], gradient_steps=2), weight_norm=True)


# ---- "beat every baseline" candidates (2026-09-28)
#: 1024-wide critic (CrossQ's width) + SALE with TD7's settings (un-normalised zsa, rep lr 3e-4)
VARIANTS["tall"] = _with(VARIANTS["t1024"], representation="sale")
#: ... + horizon-aware discount (TD-MPC2 heuristic) for long-horizon tasks such as Swimmer
VARIANTS["tall_gauto"] = dict(_with(VARIANTS["t1024"], representation="sale"), gamma="fifth")
#: cheaper: 512-wide critic
VARIANTS["t512_sale_gauto"] = dict(
    _with(_bn(512, entropy_schedule=(0.0, -0.5, 0.5), critic_loss="hl_gauss"), representation="sale"),
    gamma="fifth")


# ---- screen 6 (2026-09-29): exploration and returns on top of the screen-5 winner
VARIANTS["c1"] = dict(VARIANTS["tall_gauto"], exploration_noise="pink", exploration_bonus=5.0)
VARIANTS["c2"] = dict(VARIANTS["c1"], n_step=1)
VARIANTS["c3"] = dict(VARIANTS["c1"], gamma=0.999)


# ---- 2026-09-30: C1 became the library default ------------------------------------------
#: Every variant above was run while the library defaults were the 256-wide recipe:
#: gamma 0.99, white Gaussian exploration, no novelty bonus, no learned representation,
#: 256-wide critics. They are pinned to those defaults explicitly, so that the new
#: defaults never change a configuration after it has been run. "default" stays {}:
#: whatever ``TEMPO("MlpPolicy", env)`` does now (C1 until 2026-10-22; the fast recipe since 2026-10-23, see the end).
_PRE_C1 = dict(gamma=0.99, exploration_noise="white", exploration_bonus=0.0)
_PRE_C1_POLICY = dict(representation=None, net_arch=dict(pi=[256, 256], qf=[256, 256]))


def _pin(cfg: dict) -> dict:
    import copy

    out = copy.deepcopy(cfg)
    for k, v in _PRE_C1.items():
        out.setdefault(k, v)
    pk = out.setdefault("policy_kwargs", {})
    for k, v in _PRE_C1_POLICY.items():
        pk.setdefault(k, copy.deepcopy(v))
    return out


for _name in list(VARIANTS):
    if _name != "default":
        VARIANTS[_name] = _pin(VARIANTS[_name])

#: the final recipe of the screens under its screening name (Screen 6); identical to "default" from 2026-09-30 to
#: 2026-10-22. Since 2026-10-23 the default is the fast recipe (DEFAULT_RECIPE below) and "c1" is the previous default.
FINAL_RECIPE = "c1"


# ---- screen 7 (2026-10-01): Swimmer. Can TEMPO learn a long horizon once its values fit the
# HL-Gauss support? The reward scaler's running return reset at time-limit truncations although
# the critic bootstraps through them; at gamma 0.9999 the scaled values were ~7x the support.
VARIANTS["c1_g9999_ts"] = dict(VARIANTS["c1"], gamma=0.9999, _scale_through_truncation=True)
VARIANTS["c1_g999_ts"] = dict(VARIANTS["c1"], gamma=0.999, _scale_through_truncation=True)

# ---- screen 7b (2026-10-01): horizon-consistent scale (kappa, exactly 1.0 at gamma="auto") with
# gamma 1 - 1/T; on Swimmer also a second entropy anneal after 50% (-0.5 -> -2.0 nats/dim by 80%)
# so that the deterministic gait may use near-full torque (amplitude-cap hypothesis)
VARIANTS["c1_g999_k"] = dict(VARIANTS["c1"], gamma=0.999, _kappa_scale=True)
VARIANTS["c1_g999_k_late"] = dict(VARIANTS["c1"], gamma=0.999, _kappa_scale=True,
                                  entropy_schedule_continuous=(0.0, -0.5, 0.5, -2.0, 0.8))

# ---- screen 7c (2026-10-02): C1 + horizon audition (tempo_rl/horizon.py): a shadow learner with
# gamma = 1 - 1/T and the kappa scale collects 25% of the episodes until the end of tempering (50%);
# the run continues with it only if its undiscounted training returns are clearly better. The long
# learner also anneals its entropy further after 50% (-0.5 -> -2.0 nats/dim by 80%; screen 7b).
VARIANTS["c1_aud"] = dict(VARIANTS["c1"], horizon_audition=True,
                          _long_entropy_schedule=(0.0, -0.5, 0.5, -2.0, 0.8))

# ---- screen 7d (2026-10-02): evidence-driven allocation. The shadow's share of new episodes
# rises from 25 % to 75 % as its time-matched lead grows from 1 to 3 standard errors (it never
# falls below 25 %); decision at 50 % unchanged.
VARIANTS["c1_aud2"] = dict(VARIANTS["c1_aud"], _audition_opts=dict(ADAPTIVE=True))


# ---- 2026-10-23: the fast recipe became the library default (benchmarks/PROTOCOL.md, addendum
# of 2026-10-23). TEMPO("MlpPolicy", env) now runs C1's update maths on the fast engine (4 environments, one update of
# batch 1024 per vector step, batch-size invariance rules with Adam's square-root rule, replay buffer on the device,
# pipelined CPU acting, fused Adam, bfloat16 critics on a GPU, closed-form Batch Renorm, fused twin critics). It is NOT
# bit-identical to C1 and its scores have not been screened; ~507 env-steps/s for one run alone on Hopper (MPS) vs 94.8.
# The screening variants of 2026-10-22/23 (fast2, fast4, fast4_late, fast4_recent, fast4_mean, fast4_actor1, fast4_all,
# fast4x, c1x, fast4_lin, fast4_lin_sa, fast4x_lin_sa, fast4_lin_sa_actor1) and their opt-in options were removed;
# Screen 8 was not run.
#: Every variant above was run (or defined) while the library default was the C1 engine. Like the pin of 2026-09-30,
#: they are pinned to that engine explicitly, so that the new default changes none of them: "c1" trains bit for bit
#: like TEMPO("MlpPolicy", env) of the library before 2026-10-23 (source hashes 9a854437201918a8 / 85f20dd464e3d145).
#: "default" stays {}: whatever TEMPO("MlpPolicy", env) does now (the fast recipe).
_C1_ENGINE = dict(n_envs=1, batch_size=256, train_freq=1, gpu_replay=False, act_on_cpu=False, pipeline=False,
                  fused_adam=False, amp=None, invariance="none", lr_rule="none", brn_impl="autograd")
_C1_ENGINE_POLICY = dict(fused_critics=False)


def _pin_engine(cfg: dict) -> dict:
    import copy

    out = copy.deepcopy(cfg)
    for k, v in _C1_ENGINE.items():
        out.setdefault(k, v)
    pk = out.setdefault("policy_kwargs", {})
    for k, v in _C1_ENGINE_POLICY.items():
        pk.setdefault(k, v)
    return out


for _name in list(VARIANTS):
    if _name != "default":
        VARIANTS[_name] = _pin_engine(VARIANTS[_name])

#: The library default from 2026-10-23 to 2026-11-12 spelled out (it keeps this meaning now that the default has changed,
#: as "c1" does): C1's settings (the pinned "c1" above) + the fast engine, i.e. the former variant "fast4x".
VARIANTS["fast"] = dict(VARIANTS["c1"], n_envs=4, batch_size=1024, train_freq=1, gpu_replay=True, act_on_cpu=True,
                        pipeline=True, fused_adam=True, amp="auto", invariance="samples", lr_rule="sqrt",
                        ref_batch=256, brn_impl="closed_form",
                        policy_kwargs=dict(VARIANTS["c1"]["policy_kwargs"], fused_critics=True))
#: the recipe "default" had to equal until 2026-11-12 (the configuration check of the TEMPO_test notebooks; see the end)
DEFAULT_RECIPE = "fast"
#: The fast recipe as TEMPO's default resolves it on image observations (2026-11-03, CarRacing-v3): the phase engine
#: (gpu_replay, act_on_cpu, pipeline) does not support images, so the default falls back to the classic engine with every
#: other setting unchanged ("fast" itself sets gpu_replay=True explicitly, which TEMPO refuses for images).
VARIANTS["fast_image"] = dict(VARIANTS["fast"], gpu_replay=False, act_on_cpu=False, pipeline=False)
#: the recipe "default" must equal on the pixel tasks of benchmarks.plans.IMAGE
DEFAULT_RECIPE_IMAGE = "fast_image"


# ---- 2026-11-13: TEMPO 2 became the library default (benchmarks/PROTOCOL.md, entry of 2026-11-13) -----------------------
# TEMPO("MlpPolicy", env) now runs the fast recipe with three changes: the automatic discount never falls below 0.99
# (gamma="auto": clip(1 - 5/T, 0.99, 0.995); the former rule is gamma="fifth"), continuous actions get a second, late
# anneal of the entropy target (-0.5 -> -2.0 nats per dimension between 50 % and 80 % of the budget), and predict /
# evaluation use an average of the actor (policy_average=5000 environment steps).
#: Every variant above was run (or defined) while the library default was the fast recipe (or an earlier one). Their
#: discount is explicit ("fifth" replaces the "auto" of their time: the same rule under its new name); the multi-step
#: horizon and the continuous entropy schedule are pinned where a variant left them to the library, and the averaged
#: policy, which did not exist, is pinned off, so that the new default
#: changes none of them ("fast" trains bit for bit like TEMPO("MlpPolicy", env) of the library with source hash
#: 2e417b4c09095fef, and evaluates with the newest actor as it did). "default" stays {}: whatever TEMPO("MlpPolicy", env)
#: does now.
for _name in list(VARIANTS):
    if _name != "default":
        VARIANTS[_name].setdefault("policy_average", None)
        VARIANTS[_name].setdefault("n_step", 3)
        if "entropy_schedule" not in VARIANTS[_name] and "entropy_schedule_continuous" not in VARIANTS[_name]:
            # "sale", "wn_utd2", "sale_wn_utd2" left it to the library: the continuous default of their time
            VARIANTS[_name]["entropy_schedule_continuous"] = (0.0, -0.5, 0.5)

#: The library default spelled out (identical to "default" today; it keeps this meaning if a default changes later).
#: The continuous entropy schedule applies to continuous actions only (tempo_kwargs); discrete actions keep the discrete
#: schedule of the library.
VARIANTS["tempo2"] = dict(VARIANTS["fast"], gamma="auto", entropy_schedule_continuous=(0.0, -0.5, 0.5, -2.0, 0.8),
                          policy_average=5000)
#: the recipe "default" must equal (the configuration check of the TEMPO_test notebooks, notebooks/runlab.py)
DEFAULT_RECIPE = "tempo2"
#: TEMPO 2 as the default resolves it on image observations (the classic engine; the averaged policy needs the phase engine
#: and is ignored there)
VARIANTS["tempo2_image"] = dict(VARIANTS["tempo2"], gpu_replay=False, act_on_cpu=False, pipeline=False)
DEFAULT_RECIPE_IMAGE = "tempo2_image"
#: research options, not part of the default (diagnostic runs of 2026-11-13, benchmarks/PROTOCOL.md):
#: the whole time limit as the horizon where only the time limit ends an episode (Swimmer gains, HalfCheetah loses) ...
VARIANTS["tempo2_limit"] = dict(VARIANTS["tempo2"], gamma="limit")
#: ... and one-step returns once tempering has ended, on continuous actions (HalfCheetah gains, Walker2d loses)
VARIANTS["tempo2_n1"] = dict(VARIANTS["tempo2"], n_step_continuous=(3, 1))


# ==== 2026-11-18: ablations of the default for the supplementary material (TEMPO_test/Ablations) ========================
# Each entry is the default ("tempo2", above) with exactly ONE part changed; nothing above this line is modified. The names
# start with "TEMPO-" (they are also the folder names of their results and the row names of the supplementary table).
# The notebook TEMPO_test/Ablations/TEMPO_ablations.ipynb runs them and stores their results in TEMPO_test/Ablations/results,
# never in TEMPO_test/results (which the main comparison reads).
def _ablation(**changes) -> dict:
    import copy

    out = copy.deepcopy(VARIANTS["tempo2"])
    out.update(changes)
    return out


#: tempered entropy target -> a constant target. Continuous actions: -1 nat per action dimension, SAC's standard target
#: -dim(A); discrete actions: 0.5 of the maximum entropy ln|A|. Written as a schedule whose start and end are equal, with the
#: anneal fraction 0.5 of the default kept, because the novelty bonus (and nothing else here) decays over that fraction:
#: the bonus anneal over the first half of training is unchanged.
VARIANTS["TEMPO-FixedEntropy"] = _ablation(entropy_schedule_continuous=(-1.0, -1.0, 0.5),
                                           entropy_schedule_discrete=(0.5, 0.5, 0.5))
#: no tempered novelty bonus
VARIANTS["TEMPO-NoBonus"] = _ablation(exploration_bonus=0.0)
#: predict / evaluation use the newest actor instead of the averaged one
VARIANTS["TEMPO-NoAverage"] = _ablation(policy_average=None)

# ---- 2026-11-22: two more ablations (the learned representation and the categorical critic loss)
#: HL-Gauss critic -> MSE regression critic. The library ties reward scaling to the critic loss (reward_scaling=None means
#: "on with HL-Gauss, off with MSE"), so it is set to True explicitly: only the loss changes.
VARIANTS["TEMPO-MSE"] = _ablation(critic_loss="mse", reward_scaling=True)
#: no SALE representation: actor and critics see the encoder features only (no zs / zsa embeddings)
VARIANTS["TEMPO-NoSALE"] = _ablation(policy_kwargs=dict(VARIANTS["tempo2"]["policy_kwargs"], representation=None))

#: the ablations in table order (each one applies to discrete and to continuous actions)
ABLATIONS = ("TEMPO-FixedEntropy", "TEMPO-NoBonus", "TEMPO-NoAverage", "TEMPO-MSE", "TEMPO-NoSALE")
