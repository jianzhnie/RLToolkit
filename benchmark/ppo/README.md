<!--
 * @Author: jianzhnie
 * @Date: 2022-09-01 15:34:45
 * @LastEditors: jianzhnie
 * @LastEditTime: 2022-09-01 15:55:48
 * @Description:
 * Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
-->

## Reproduce PPO

Based on rltoolkit, the PPO algorithm of deep reinforcement learning has been reproduced, reaching the same level of indicators as the paper in mujoco benchmarks.

> Paper: PPO in [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)

### Mujoco/Atari games introduction

Please see [mujoco-py](https://github.com/openai/mujoco-py) to know more about Mujoco games or [atari](https://gym.openai.com/envs/#atari) to know more about Atari games.

### Benchmark result

#### 1. Mujoco games results

<p align="center">
<img src="https://github.com/benchmarking-rl/rltoolkit-experiments/blob/master/PPO/torch/mujoco_result.png" alt="mujoco-result"/>
</p>

#### 2. Atari games results

<p align="center">
<img src="https://github.com/benchmarking-rl/rltoolkit-experiments/blob/master/PPO/torch/atari_result.png" alt="atari-result"/>
</p>

- Each experiment was run three times with different seeds

## How to use

### Dependencies:

- python>=3.6.2
- pytorch
- gym==0.21.0
- mujoco-py==2.1.2.14

### Training:

```
# To train an agent for discrete action game (Atari: PongNoFrameskip-v4 by default)
python train.py

# To train an agent for continuous action game (Mujoco)
python train.py --env 'HalfCheetah-v2' --continuous_action
```
