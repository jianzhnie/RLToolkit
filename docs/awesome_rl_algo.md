# Deep Reinforment Learning 相关算法资料


## Awesome Repos
1. https://github.com/kengz/SLM-Lab

2. https://github.com/rail-berkeley/rlkit
3. https://github.com/ChenglongChen/pytorch-DRL


## DDPG DDPG([Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971))

Blog 推荐： [SAC(*Soft Actor-Critic*)阅读笔记](https://zhuanlan.zhihu.com/p/85003758)

DPG针对连续动作空间的控制任务在传统的PG（[Policy Gradient](https://papers.nips.cc/paper/1713-policy-gradient-methods-for-reinforcement-learning-with-function-approximation.pdf)，[OpenAI的PG教程](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html)）算法上做了改进，将策略函数的输出从一个分布（常用高斯分布）转变为一个唯一确定的动作（通常由一个向量表示，这也是“deterministic“的由来）。



**DPG的思路**：

以往的PG算法思路是建立累计收益(Cumulative Return)与策略的关系函数，随后调整策略以追求更大的收益。而DPG算法在根本上不同，DPG算法可以被视为 Q-learning 的连续动作空间版本，其思想在于直接利用critic(Q函数)找到可能的最优决策，随后用找到的最优决策来优化策略函数(actor)，也就是说策略调整完全依赖于critic而不用理会实际的收益。



## PPO([Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347))

Blog 推荐： [SAC(*Soft Actor-Critic*)阅读笔记](https://zhuanlan.zhihu.com/p/85003758)

PPO是TRPO([Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477))的简化版，二者的目标都是：**在PG算法的优化过程中，使性能单调上升，并且使上升的幅度尽量大。**

PPO同样使用了AC框架，不过相比DPG更加接近传统的PG算法，采用的是随机分布式的策略函数（Stochastic Policy），智能体（agent）每次决策时都要从策略函数输出的分布中采样，得到的样本作为最终执行的动作，因此天生具备探索环境的能力，不需要为了探索环境给决策加上扰动；PPO的重心会放到actor上，仅仅将critic当做一个预测状态好坏（在该状态获得的期望收益）的工具，策略的调整基准在于获取的收益，不是critic的导数。



个人理解的PPO是在传统的PG算法上加入了如下改进：

- 引入[importance sampling](https://medium.com/%40jonathan_hui/rl-importance-sampling-ebfb28b4a8c6)技巧，使PG算法成为可以利用过往数据的off-policy算法。

- 引入AC框架，一方面免去PG每次优化策略都需要计算收益（Return）的操作，另一方面可以利用critic计算单步决策的advantage。

- 使用了[GAE](https://arxiv.org/abs/1506.02438)，batch training，[replay buffer]() 等提高算法性能的技巧。

- 严格约束策略参数的更新速度，使策略的表现尽量单调上升。



##  SAC([Soft Actor-Critic](https://arxiv.org/abs/1801.01290))

Blog 推荐： [SAC(*Soft Actor-Critic*)阅读笔记](https://zhuanlan.zhihu.com/p/85003758)

SAC是基于最大熵（maximum entropy）这一思想发展的RL算法，采用与PPO类似的随机策略函数（Stochastic Policy），并且是一个off-policy，actor-critic算法，与其他RL算法最为不同的地方在于，SAC在优化策略以获取更高累计收益的同时，也会最大化策略的熵。

### 简单理解SAC的基本思想

将熵引入RL算法的好处为，可以让策略（policy）尽可能随机，agent可以更充分地探索状态空间 S，避免策略早早地落入局部最优点（local optimum），并且可以探索到多个可行方案来完成指定任务，提高抗干扰能力。
