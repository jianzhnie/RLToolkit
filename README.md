<!--
 * @Author: jianzhnie
 * @LastEditors: jianzhnie
 * @Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
 * Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
-->

# RLToolkit An Easy  Deep Reinforcement Learning Toolkit

RLToolkit ([website][(](https://github.com/jianzhnie/deep-rl-toolkit))) is developed for practitioners with
the following advantages:

- **Scalable**: fully exploits the parallelism of DRL algorithms at multiple levels, making it easily scale out to hundreds or thousands of computing nodes on a cloud platform, say, a [DGX SuperPOD platform](https://www.nvidia.com/en-us/data-center/dgx-superpod/) with thousands of GPUs.

- **Elastic**: allows to elastically and automatically allocate computing resources on the cloud.

- **Lightweight**: the core codes <1,000 lines (check [Demo]()).

- **Efficient**: in many testing cases (single GPU/multi-GPU/GPU cloud), we find it more efficient than [Ray RLlib](https://github.com/ray-project/ray).

- **Stable**: much much much more stable than [Stable Baselines 3](https://github.com/DLR-RM/stable-baselines3) by utilizing various ensemble methods.

RLToolkit implements the following model-free deep reinforcement learning (DRL) algorithms:

- **DDPG, TD3, SAC, PPO, REDQ** for continuous actions in single-agent environment,
- **DQN, Double DQN, D3QN, SAC** for discrete actions in single-agent environment,
- **QMIX, VDN, MADDPG, MAPPO, MATD3** in multi-agent environment.

For the details of DRL algorithms, please check out the educational
webpage [OpenAI Spinning Up](https://spinningup.openai.com/en/latest/).

RL supports the following simulators:

- **Isaac Gym** for massively parallel simulation,
- **OpenAI Gym, MuJoCo, PyBullet, FinRL** for benchmarking.

## Contents

- [RLToolkit An Easy  Deep Reinforcement Learning Toolkit](#rltoolkit-an-easy--deep-reinforcement-learning-toolkit)
  - [Contents](#contents)
  - [Experimental Demos](#experimental-demos)
  - [Contributions](#contributions)
  - [Citation:](#citation)



## Experimental Demos


## Contributions

We welcome any contributions to the codebase, but we ask that you please **do not** submit/push code that breaks the tests. Also, please shy away from modifying the tests just to get your proposed changes to pass them. As it stands, the tests on their own are quite minimal (instantiating environments, training agents for one step, etc.), so if they're breaking, it's almost certainly a problem with your code and not with the tests.

We're actively working on refactoring and trying to make the codebase cleaner and more performant as a whole. If you'd like to help us clean up some code, we'd strongly encourage you to also watch [Uncle Bob's clean coding lessons](https://www.youtube.com/playlist?list=PLmmYSbUCWJ4x1GO839azG_BBw8rkh-zOj) if you haven't already.


## Citation:

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
