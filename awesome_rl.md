# 强化学习入门

## 强化学习课程

### 周博磊

[https://www.bilibili.com/video/BV1LE411G7Xj?from=search&seid=9725909430531578664](https://link.zhihu.com/?target=https%3A//www.bilibili.com/video/BV1LE411G7Xj%3Ffrom%3Dsearch%26seid%3D9725909430531578664)

### 李宏毅

台大李宏毅教授的课程，这门课以Sutton的书籍作为教材，对强化学习里比较重要的概念和算法进行了教授。

### 王树森老师

课件：[https://github.com/wangshusen/DeepLearning](https://github.com/wangshusen/DeepLearning)

讲义：[https://github.com/wangshusen/DRL/blob/master/Notes_CN/chp3.pdf](https://github.com/wangshusen/DRL/blob/master/Notes_CN/chp3.pdf)

### RL China

<https://www.bilibili.com/video/av584041095/>

### 百度

<https://www.bilibili.com/video/BV1yv411i7xd?from=search&seid=9725909430531578664>

## 强化学习书+代码

### Reinforcement Learning: An Introduction

第一本强化学习的书，也是最经典的书，《Reinforcement Learning: An Introduction》 Richard S. Sutton and Andrew G. Barto。 Sutton在2012年Release出来的，更新之后的第二版。应该算是目前为止，关于强化学习，介绍最为详细，全面的教材之一。David Silver的强化学习视频也是根据这本教材展开，配合着看，更容易理解。

- **书籍配套代码1**

<https://github.com/ShangtongZhang/reinforcement-learning-an-introduction>

- **书籍配套代码2**

<https://github.com/dennybritz/reinforcement-learning>

- **配套视频课程：**

[DAVID SILVER] (<https://www.davidsilver.uk/>)

### openai  spinningup

OpenAI Spinning Up（ 五星推荐！）深度强化学习入门的最佳手册，代码短小精悍，性能也不错，包含了基础原理、关键论文、关键算法解释以及实现。[文档](https://spinningup.openai.com/en/latest/)十分良心，和代码基本一一对照。

- <https://spinningup.openai.com/en/latest/user/introduction.html>

## 强化学习开源工具箱

代码库经过野蛮生长的年代后终于趋于稳定，在学习、科研、生产的不同阶段都有了十分成熟的代码库，成熟的代码库不仅指好用的代码，还需要清晰的文档，配套的工具等。在此各推荐一个（少即是多，浓缩才是精华）如下：

### OpenAI Gym

目前强化学习编程实战常用的环境就是OpenAI的gym库了，支持Python语言编程。OpenAI Gym是一款用于研发和比较强化学习算法的工具包，它支持训练智能体（agent）做任何事——从行走到玩Pong或围棋之类的游戏都在范围中。

- <https://gym.openai.com/>

### Baselines

https://github.com/openai/baselinesgithub.com/openai/baselines

### stable-baselines3

 [stable-baselines3](https//github.com/DLR-RM/stable-baselines3) 由 OpenAI 的 baselines 发展而来，因为 baselines 不够稳定，于是有了 [stable-baselines](https//github.com/hill-a/stable-baselines)，接着有了 v2，再有了 PyTorch 版的 v3，目前由 DLR-RM 维护。不仅[文档](https://stable-baselines3.readthedocs.io/)清晰，还提供了很多常用环境和RL算法的调优超参数：[RL Baselines3 Zoo](https://github.com/DLR-RM/rl-baselines3-zoo).实现了几乎所有的强化学习算法。

### Ray/RLlib

[ray/rllib](https://github.com/ray-project/ray)。UC Berkeley 出品，工业级的强化学习库，优势在于分布式计算和自动调参，支持 TensorFlow/PyTorch，很多大企业比如谷歌、亚马逊、蚂蚁金服都在用。

## PyTorch  + RL  Algorithms

### 步步深入RL

这份Pytorch强化学习教程一共有八章，从DQN（Deep Q-Learning）开始，步步深入，最后向你展示Rainbow到底是什么。

不仅有Jupyter Notebook，作者还在Colab上配置好了代码，无需安装，你就能直观地感受到算法的效果，甚至还可以直接在手机上进行学习！

Github 地址： <https://github.com/Curt-Park/rainbow-is-all-you-need>

### CleanRL

CleanRL (Clean Implementation of RL Algorithms)
 tests ci   Code style: black Imports: isort

CleanRL is a Deep Reinforcement Learning library that provides high-quality single-file implementation with research-friendly features. The implementation is clean and simple, yet we can scale it to run thousands of experiments using AWS Batch. The highlight features of CleanRL are:

📜 Single-file implementation
Every detail about an algorithm variant is put into a single standalone file.
For example, our ppo_atari.py only has 340 lines of code but contains all implementation details on how PPO works with Atari games, so it is a great reference implementation to read for folks who do not wish to read an entire modular library.
📊 Benchmarked Implementation (7+ algorithms and 34+ games at <https://benchmark.cleanrl.dev>)
📈 Tensorboard Logging
🪛 Local Reproducibility via Seeding
🎮 Videos of Gameplay Capturing
🧫 Experiment Management with Weights and Biases
💸 Cloud Integration with docker and AWS

Github 地址：  <https://github.com/vwxyzjn/cleanrl>

### openai - spinningup

openai的spinningup：里面提供了经典Policy-based算法的复现，优点是写的通俗易懂上手简单，并且效果有保障，而且同时tf和Pytorch的支持；缺点是没有value-based的算法，做DQN系列的就没办法了

### rlpyt

UCB两个大佬开源的rlpyt：专门基于pytorch实现的rl框架，有单机/多机分配资源的黑科技，挂arxiv的paper里面介绍的也效果也不错。contributor以前也写过如何加速DQN训练的调参方法

### Deep Reinforcement Learning Algorithms with PyTorch

This repository contains PyTorch implementations of deep reinforcement learning algorithms and environments.

(To help you remember things you learn about machine learning in general write them in [Save All](https://saveall.ai/shared/deck/140&4&3K3uXPazkg4&github_links) and try out the public deck there about Fast AI's machine learning textbook.)

Github 地址： [p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch​](https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch.git)

已实现的算法包括：

- Deep Q Learning (**DQN**) (Mnih et al. 2013)
- **DQN with Fixed Q Targets** (Mnih et al. 2013)
- Double DQN (**DDQN**) (Hado van Hasselt et al. 2015)
- **DDQN with Prioritised Experience Replay** (Schaul et al. 2016)
- **Dueling DDQN** (Wang et al. 2016)
- **REINFORCE** (Williams et al. 1992)
- Deep Deterministic Policy Gradients (**DDPG**) (Lillicrap et al. 2016 )
- Twin Delayed Deep Deterministic Policy Gradients (**TD3**) (Fujimoto et al. 2018)
- Soft Actor-Critic (**SAC & SAC-Discrete**) (Haarnoja et al. 2018)
- Asynchronous Advantage Actor Critic (**A3C**) (Mnih et al. 2016)
- Syncrhonous Advantage Actor Critic (**A2C**)
- Proximal Policy Optimisation (**PPO**) (Schulman et al. 2017)
- DQN with Hindsight Experience Replay (**DQN-HER**) (Andrychowicz et al. 2018)
- DDPG with Hindsight Experience Replay (**DDPG-HER**) (Andrychowicz et al. 2018 )
- Hierarchical-DQN (**h-DQN**) (Kulkarni et al. 2016)
- Stochastic NNs for Hierarchical Reinforcement Learning (**SNN-HRL**) (Florensa et al. 2017)
- Diversity Is All You Need (**DIAYN**) (Eyensbach et al. 2018)

### PFRL

PFRL is a deep reinforcement learning library that implements various state-of-the-art deep reinforcement algorithms in Python using [PyTorch](https://github.com/pytorch/pytorch).

Github 地址：[GitHub - pfnet/pfrl: PFRL: a PyTorch-based deep reinforcement learning library](https://github.com/pfnet/pfrl)

实现的  RL Algorithms

| Algorithm                      | Discrete Action | Continous Action | Recurrent Model | Batch Training | CPU Async Training | Pretrained models* |
| ------------------------------ | --------------- | ---------------- | --------------- | -------------- | ------------------ | ------------------ |
| DQN (including DoubleDQN etc.) | ✓               | ✓ (NAF)          | ✓               | ✓              | x                  | ✓                  |
| Categorical DQN                | ✓               | x                | ✓               | ✓              | x                  | x                  |
| Rainbow                        | ✓               | x                | ✓               | ✓              | x                  | ✓                  |
| IQN                            | ✓               | x                | ✓               | ✓              | x                  | ✓                  |
| DDPG                           | x               | ✓                | x               | ✓              | x                  | ✓                  |
| A3C                            | ✓               | ✓                | ✓               | ✓ (A2C)        | ✓                  | ✓                  |
| ACER                           | ✓               | ✓                | ✓               | x              | ✓                  | x                  |
| PPO                            | ✓               | ✓                | ✓               | ✓              | x                  | ✓                  |
| TRPO                           | ✓               | ✓                | ✓               | ✓              | x                  | ✓                  |
| TD3                            | x               | ✓                | x               | ✓              | x                  | ✓                  |
| SAC                            | x               | ✓                | x               | ✓              | x                  | ✓                  |

### 清华天授（Tianshou）

天授（Tianshou）是纯 基于 PyTorch 代码的强化学习框架，与目前现有基于 TensorFlow 的强化学习库不同，天授的类继承并不复杂，API 也不是很繁琐。最重要的是，天授的训练速度非常快，我们试用 Pythonic 的 API 就能快速构建与训练 RL [智能体]()。

Tianshou的优势：

- 实现简洁
- 速度快
- 模块化
- 可复现性

目前天授支持的 RL 算法有如下几种：

- Policy Gradient (PG)
- Deep Q-Network (DQN)
- Double DQN (DDQN) with n-step returns
- Advantage Actor-Critic (A2C)
- Deep Deterministic Policy Gradient (DDPG)
- Proximal Policy Optimization (PPO)
- Twin Delayed DDPG (TD3)
- Soft Actor-Critic (SAC)

另外，对于以上代码天授还支持并行收集样本，并且所有算法均统一改写为基于 replay-buffer 的形式。

github 地址：[https://github.com/thu-ml/tianshou](https://github.com/thu-ml/tianshou)

## Tutorial

An Introduction to Reinforcement Learning Using OpenAI Gym
<https://www.gocoder.one/blog/rl-tutorial-with-openai-gym>

An Introduction to Reinforcement Learning with OpenAI Gym, RLlib, and Google Colab
<https://www.anyscale.com/blog/an-introduction-to-reinforcement-learning-with-openai-gym-rllib-and-google>

Intro to RLlib: Example Environments
<https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70>

Ray and RLlib for Fast and Parallel Reinforcement Learning
<https://towardsdatascience.com/ray-and-rllib-for-fast-and-parallel-reinforcement-learning-6d31ee21c96c>



## 知乎专栏

- [强化学习知识大讲堂](https://zhuanlan.zhihu.com/sharerl)

- 该专栏作者即为《[深入浅出强化学习：原理入门]()》一书的作者，专栏的讲解包括：入门篇、进阶篇、前沿篇和实践篇，深入浅出，内容翔实，是专门针对强化学习的知识大讲堂。

- [智能单元](https://zhuanlan.zhihu.com/intelligentunit)

- 该专栏涵盖的内容较广，主要包括深度学习和强化学习及其相应的实践应用，是知乎上深度学习和强化学习领域关注量最大的专栏，其中对强化学习的介绍也较浅显易懂。

- [神经网络与强化学习](https://zhuanlan.zhihu.com/c_101836530)

- 该专栏主要是作者关于强化学习经典入门书籍《Reinforcement Learning : An introduction》的读书笔记，因此，非常适合在啃该书的时候参考该专栏，以有更深入的理解。



## 强化学习算法学习流程概览：
### 离散动作：（Value Gradient ）
Q-table-learning -> DQN(Deep Q NetWork) -> Double DQN -> Dueling DQN ->Double Dueling DQN (D3QN) ->Twin Delayed DDPG (TD3 连续动作)

### 连续动作：（Policy Gradient）

Actor-Critic -> Advantage Actor-Critic(A2C) -> Asynchronus A2C (A3C) -> Deep Deterministic Policy Gradient (DDPG) -> Distributed Distributional DDPG (D4PG) -> Soft Actor-Critic (SAC) -> Trust Region Policy Optimization (TRPO) -> Generalized Advantage Estimation (GAE) -> Proximal Policy Optimization(PPO)
