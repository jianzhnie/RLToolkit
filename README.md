<!--

 * @Author: jianzhnie
 * @LastEditors: jianzhnie
 * @Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
 * Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
-->

<p align="center">
<img src="docs/images/logo.png" alt="logo" width="1000"/>
</p>
---

[![Documentation Status](https://readthedocs.org/projects/deep-rl-docs/badge/?version=latest)](https://deep-rl-docs.readthedocs.io/en/latest/?badge=latest)

## Overview

RLToolkit is a flexible and high-efficient reinforcement learning framework. RLToolkit ([website](https://github.com/jianzhnie/deep-rl-toolkit))) is developed for practitioners with the following advantages:

- **Reproducible**. We provide algorithms that stably reproduce the result of many influential reinforcement learning algorithms.

- **Extensible**. Build new algorithms quickly by inheriting the abstract class in the framework.

- **Reusable**.  Algorithms provided in the repository could be directly adapted to a new task by defining a forward network and training mechanism will be built automatically.

- **Elastic**: allows to elastically and automatically allocate computing resources on the cloud.

- **Lightweight**: the core codes <1,000 lines (check [Demo](./examples/tutorials/lesson3/DQN/train.py)).

- **Stable**: much more stable than [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) by utilizing various ensemble methods.


## Table of Content
- [Overview](#overview)
- [Table of Content](#table-of-content)
- [Abstractions](#abstractions)
  - [Model](#model)
  - [Algorithm](#algorithm)
  - [Agent](#agent)
- [Supported Algorithms](#supported-algorithms)
- [Supported Envs](#supported-envs)
- [Examples](#examples)
- [Experimental Demos](#experimental-demos)
- [Contributions](#contributions)
- [References](#references)
- [Citation](#citation)


## Abstractions

<p align="center">
<img src="./docs/images/abstractions.png" alt="abstractions" width="400"/>
</p>

RLToolkit aims to build an agent for training algorithms to perform complex tasks.
The main abstractions introduced by PARL that are used to build an agent recursively are the following:

### Model
`Model` is abstracted to construct the forward network which defines a policy network or critic network given state as input.

### Algorithm
`Algorithm` describes the mechanism to update parameters in `Model` and often contains at least one model.

### Agent
`Agent`, a data bridge between the environment and the algorithm, is responsible for data I/O with the outside environment and describes data preprocessing before feeding data into the training process.

## Supported Algorithms

RLToolkit implements the following model-free deep reinforcement learning (DRL) algorithms:

![../_images/rl_algorithms_9_15.svg](https://spinningup.openai.com/en/latest/_images/rl_algorithms_9_15.svg)

A non-exhaustive, but useful taxonomy of algorithms in modern RL.

<img src="docs/images/algorithms.png" alt="Coach Design" style="width: 800px;"/>

##  Supported Envs

- **OpenAI Gym**
- **Atari**
- **MuJoCo**
- **PyBullet**

For the details of DRL algorithms, please check out the educational webpage [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/).

## Examples
If you want to learn more about deep reinforcemnet learning, please read the [deep-rl-class](https://jianzhnie.github.io/machine-learning-wiki/#/deep-rl/) and run the [examples](https://github.com/jianzhnie/deep-rl-toolkit/blob/main/examples/tutorials).



[//]: # (Image References)
<p align="center">
<img src="docs/images/trained.gif" alt="logo" width="810"/>
</p>
<p align="center">
<img src="examples/tutorials/assets/img/breakout.gif" width = "200" height ="200"/> <img src="examples/tutorials/assets/img/spaceinvaders.gif" width = "200" height ="200"/> <img src="examples/tutorials/assets/img/seaquest.gif" width = "200" height ="200"/><img src="docs/images/Breakout.gif" width = "200" height ="200" alt="Breakout"/>
<br>
<p align="center">
<img src="docs/images/performance.gif" width = "265" height ="200" alt="NeurlIPS2018"/> <img src="docs/images/Half-Cheetah.gif" width = "265" height ="200" alt="Half-Cheetah"/> <img src="examples/tutorials/assets/img/snowballfight.gif" width = "265" height ="200"/>
<br>

- [QuickStart](./benchmark/quickstart/train.py)
- [DQN](./examples/tutorials/lesson3/DQN/train.py)
- [N-step-DQN](./examples/tutorials/lesson3/N-step-DQN/train.py)
- [Noisy-DQN](./examples/tutorials/lesson3/Noisy-DQN/train.py)
- [Rainbow](./examples/tutorials/lesson3/Rainbow/train.py)
- [PG](./examples/tutorials/lesson4/pg/train.py)
- [TRPO](./examples/tutorials/lesson4/trpo/train.py)
- [PPO](./examples/tutorials/lesson4/ppo/train.py)
- [AC](./examples/tutorials/lesson4/ac&a2c/train.py)
- [A2C](./examples/tutorials/lesson4/ac&a2c/train.py)
- [DDPG](./examples/tutorials/lesson5/ddpg-pendulum)
- [SAC](./examples/tutorials/lesson5/sac/train.py)
- [TD3](./examples/tutorials/lesson5/td3/train.py)
- [QMIX](./examples/tutorials/lesson6/qmix/train.py)
- [IDQN](./examples/tutorials/lesson6/idqn/train.py)
- [MADDPG](./examples/tutorials/lesson6/maddpg/train.py)
- [vdn](./examples/tutorials/lesson6/vdn/train.py)


## Experimental Demos

- **Quick start**
```python
# into demo dirs
cd  benchmark/quickstart/
# train
python train.py
```

**DNQ  example**
```python
# into demo dirs
cd  examples/tutorials/lesson3/DQN/
# train
python train.py
```

**PPO Example**
```python
# into demo dirs
cd  examples/tutorials/lesson3/DQN/
# train
python train.py
```

**DDPG for Pendulum-v1**

```python
# into demo dirs
cd  examples/tutorials/lesson5/ddpg/
# train
python train.py
```
...


## Contributions

We welcome any contributions to the codebase, but we ask that you please **do not** submit/push code that breaks the tests. Also, please shy away from modifying the tests just to get your proposed changes to pass them. As it stands, the tests on their own are quite minimal (instantiating environments, training agents for one step, etc.), so if they're breaking, it's almost certainly a problem with your code and not with the tests.

We're actively working on refactoring and trying to make the codebase cleaner and more performant as a whole. If you'd like to help us clean up some code, we'd strongly encourage you to also watch [Uncle Bob's clean coding lessons](https://www.youtube.com/playlist?list=PLmmYSbUCWJ4x1GO839azG_BBw8rkh-zOj) if you haven't already.

## References

1. Deep Q-Network (DQN) <sub><sup> ([V. Mnih et al. 2015](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf)) </sup></sub>
2. Double DQN (DDQN) <sub><sup> ([H. Van Hasselt et al. 2015](https://arxiv.org/abs/1509.06461)) </sup></sub>
3. Advantage Actor Critic (A2C)
4. Vanilla Policy Gradient (VPG)
5. Natural Policy Gradient (NPG) <sub><sup> ([S. Kakade et al. 2002](http://papers.nips.cc/paper/2073-a-natural-policy-gradient.pdf)) </sup></sub>
6. Trust Region Policy Optimization (TRPO) <sub><sup> ([J. Schulman et al. 2015](https://arxiv.org/abs/1502.05477)) </sup></sub>
7. Proximal Policy Optimization (PPO) <sub><sup> ([J. Schulman et al. 2017](https://arxiv.org/abs/1707.06347)) </sup></sub>
8. Deep Deterministic Policy Gradient (DDPG) <sub><sup> ([T. Lillicrap et al. 2015](https://arxiv.org/abs/1509.02971)) </sup></sub>
9. Twin Delayed DDPG (TD3) <sub><sup> ([S. Fujimoto et al. 2018](https://arxiv.org/abs/1802.09477)) </sup></sub>
10. Soft Actor-Critic (SAC) <sub><sup> ([T. Haarnoja et al. 2018](https://arxiv.org/abs/1801.01290)) </sup></sub>
11. SAC with automatic entropy adjustment (SAC-AEA) <sub><sup> ([T. Haarnoja et al. 2018](https://arxiv.org/abs/1812.05905)) </sup></sub>



## Citation

To cite this repository:

```
@misc{erl,
  author = {jianzhnie},
  title = {{RLToolkit}: An Easy  Deep Reinforcement Learning Toolkit},
  year = {2022},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/jianzhnie/deep-rl-toolkit}},
}
```
