# Third-party notices

This repository (the `tempo-drl` library in `tempo_rl/` and its benchmark code) is
released under the MIT License (see `LICENSE`). It contains code adapted from,
interfaces modelled on, and hyperparameters copied from the following MIT-licensed
projects. Their copyright notices are reproduced here as their licenses require.

## sb3-contrib

`tempo_rl/networks.py` (`BatchRenorm`) is adapted from the Batch Renormalization
layer of sb3-contrib (<https://github.com/Stable-Baselines-Team/stable-baselines3-contrib>).

```
MIT License

Copyright (c) 2020 Stable-Baselines Team
```

## SBX

`tempo_rl/networks.py` (`BatchRenorm`) also follows the Batch Renormalization layer
of SBX (<https://github.com/araffin/sbx>).

```
The MIT License

Copyright (c) 2022 Antonin Raffin
```

## RL Baselines3 Zoo

The baselines' hyperparameter tables in `benchmarks/zoo.py` are copied from
`hyperparams/{sac,td3,crossq,ppo,dqn}.yml` of RL Baselines3 Zoo
(<https://github.com/DLR-RM/rl-baselines3-zoo>, commit 630883a).

```
MIT License

Copyright (c) 2019 Antonin RAFFIN
```

## Stable-Baselines3

The public API (`TEMPO(...)`, `learn`, `predict`, `save`/`load`), the callback
classes and `evaluate_policy` follow the interfaces and semantics of
Stable-Baselines3 (<https://github.com/DLR-RM/stable-baselines3>) so that code
written for SB3 works with TEMPO.

```
The MIT License

Copyright (c) 2019 Antonin Raffin
```

These projects are distributed under the MIT License, whose full text is identical
to the one in `LICENSE` apart from the copyright line.
