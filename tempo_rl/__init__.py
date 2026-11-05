"""TEMPO: Tempered-Entropy Multi-step Policy Optimization.

One off-policy reinforcement-learning algorithm, one default configuration, for
discrete and continuous actions and vector, image and dictionary observations::

    from tempo_rl import TEMPO

    model = TEMPO("MlpPolicy", "LunarLander-v3", verbose=1)
    model.learn(total_timesteps=200_000)
"""
from .callbacks import (BaseCallback, CallbackList, CheckpointCallback, EvalCallback,
                        EveryNTimesteps, StopTrainingOnMaxEpisodes,
                        StopTrainingOnNoModelImprovement, StopTrainingOnRewardThreshold)
from .env_util import make_env, make_vec_env
from .evaluation import evaluate_policy
from .policies import TempoPolicy
from .tempo import TEMPO
from .version import __version__

MlpPolicy = CnnPolicy = MultiInputPolicy = TempoPolicy

__all__ = [
    "TEMPO", "TempoPolicy", "MlpPolicy", "CnnPolicy", "MultiInputPolicy",
    "evaluate_policy", "make_env", "make_vec_env",
    "BaseCallback", "CallbackList", "CheckpointCallback", "EvalCallback", "EveryNTimesteps",
    "StopTrainingOnMaxEpisodes", "StopTrainingOnNoModelImprovement",
    "StopTrainingOnRewardThreshold", "__version__",
]
