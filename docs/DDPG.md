# 强化学习—DDPG算法原理详解

## 一、 概述

在[DQN](https://wanjun0511.github.io/2017/11/05/DQN/)中有讲过，DQN是一种 model free（无环境模型）, off-policy（产生行为的策略和进行评估的策略不一样）的强化学习算法。DDPG (Deep Deterministic Policy Gradient)算法也是model free, off-policy的，且同样使用了深度神经网络用于函数近似。但与DQN不同的是，DQN只能解决离散且维度不高的action spaces的问题，这一点请回忆DQN的神经网络的输出。而DDPG可以解决**连续动作空间**问题。另外，DQN是value based方法，即只有一个值函数网络，而DDPG是actor-critic方法，即既有值函数网络(critic)，又有策略网络(actor)。

DDPG算法原文链接： [DDPG](https://arxiv.org/pdf/1509.02971.pdf)

## 二、算法原理

在[基本概念](https://wanjun0511.github.io/2017/11/04/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-%E5%9F%BA%E6%9C%AC%E6%A6%82%E5%BF%B5/)中有说过，强化学习是一个反复迭代的过程，每一次迭代要解决两个问题：给定一个策略求值函数，和根据值函数来更新策略。

DDPG中使用一个神经网络来近似值函数，此值函数网络又称**critic网络**，它的输入是 action与observation \[a,s\]\[a,s\]，输出是Q(s,a)Q(s,a)；另外使用一个神经网络来近似策略函数，此policy网络又称**actor网络**，它的输入是observation ss，输出是action aa.

\[![critic-network](https://wanjun0511.github.io/2017/11/19/DDPG/critic.jpeg)
$$
critic: Q(s,a;ω)
$$

$$
target critic:  Q(s,a;ω−)
$$
\[![actor-network](https://wanjun0511.github.io/2017/11/19/DDPG/actor.jpeg)

$$
actor: a=π(s;θ)
$$

$$
target actor: a=π(s;θ−)
$$

这两个网络之间的联系是这样的：首先环境会给出一个obs，智能体根据**actor网络**（后面会讲到在此网络基础上增加噪声）做出决策action，环境收到此action后会给出一个奖励Rew，及新的obs。这个过程是一个step。此时我们要根据Rew去更新**critic网络**，然后沿**critic**建议的方向去更新**actor**网络。接着进入下一个step。如此循环下去，直到我们训练出了一个好的actor网络。

那么每次迭代如何更新这两个神经网络的参数呢？

与DQN一样，DDPG中也使用了target网络来保证参数的收敛。假设critic网络为Q(s,a;ω)Q(s,a;ω)， 它对应的target critic网络为Q(s,a;ω−)Q(s,a;ω−)。actor网络为π(s;θ)π(s;θ)，它对应的target actor网络为π(s;θ−)π(s;θ−)。

### 1、critic网络更新

critic网络用于值函数近似，更新方式与DQN中的类似。

target = Rt+1+γQ(St+1,π(St+1;θ−);ω−)

Loss=1/N∑t= (target−Q(St,at;ω))2

然后使用梯度下降法进行更新。注意，actor和critic都使用了target网络来计算target。

### 2、actor网络更新

actor网络用于参数化策略。这里涉及到强化学习中一个非常重要的概念：**策略梯度Policy Gradient**。

如何评价一个策略的好坏？首先我们要有一个目标，称为policy objective function，记为J(θ)。我们希望求得θ使得J(θ)J(θ)取得最大值。J(θ)J(θ)对θθ的导数 ▽θJ(θ)▽θJ(θ)即为**策略梯度**。

策略梯度这一块可以分为四种情况分别讨论：stochastic on-policy, stochastic off-policy, deterministic on-policy 和 deterministic off-policy。David Silver的课程中详细的介绍了第一种。DPG论文的第二部分讲了第二种，第四部分讲了第三四种。由于DDPG中的策略是deterministic的，本文只介绍最后两种。

直观上来说，我们应该朝着使得值函数QQ值增大的方向去更新策略的参数θθ。记策略为 a=πθ(s)a=πθ(s), J(πθ)=∫sdπ(s)Q(s,πθ(s))ds=Es∼dπ\[Q(s,πθ(s))\] ，有以下定理：

**Deterministic Policy Gradient Theorem**:
▽θJ(πθ)=∫sdπ(s)▽θπθ(s)▽aQπ(s,a)|a=πθ(s)ds=Es∼dπ\[▽θπθ(s)▽aQπ(s,a)|a=πθ(s)\]▽θJ(πθ)=∫sdπ(s)▽θπθ(s)▽aQπ(s,a)|a=πθ(s)ds=Es∼dπ\[▽θπθ(s)▽aQπ(s,a)|a=πθ(s)\]

确定性策略梯度定理提供了更新确定性策略的方法。将此方法用到Actor-Critic算法中：

#### (1) On-Policy Deterministic Actor-Critic

TD Error: δt=Rt+1+γQ(St+1,at+1;ω)−Q(St,at;ω)δt=Rt+1+γQ(St+1,at+1;ω)−Q(St,at;ω)
更新critic: Δω=αω⋅δt⋅▽ωQ(St,at;ω)Δω=αω⋅δt⋅▽ωQ(St,at;ω) (SARSA)
更新actor: Δθ=αθ⋅▽θπθ(St)▽aQ(St,at;ω)|a=πθ(s)Δθ=αθ⋅▽θπθ(St)▽aQ(St,at;ω)|a=πθ(s)

#### (2) Off-Policy Deterministic Actor-Critic

TD Error: δt=Rt+1+γQ(St+1,πθ(St+1);ω)−Q(St,at;ω)δt=Rt+1+γQ(St+1,πθ(St+1);ω)−Q(St,at;ω)
更新critic: Δω=αω⋅δt⋅▽ωQ(St,at;ω)Δω=αω⋅δt⋅▽ωQ(St,at;ω) (Q-Learning)
更新actor: Δθ=αθ⋅▽θπθ(St)▽aQ(St,at;ω)|a=πθ(s)Δθ=αθ⋅▽θπθ(St)▽aQ(St,at;ω)|a=πθ(s)

注意，在off-policy中，用于生成行为数据的策略和用于评估的策略不是同一个策略，也就是说，智能体实际上采取的action at+1at+1 不是由πθπθ生成的。假设它是由ββ生成的。在DDPG中，ββ策略是在ππ策略上增加了随机噪声random process，用来保证探索exploration。

理论上，这里引入了两种bias：一个是Deterministic Policy Gradient Theorem中的Qπ(s,a)Qπ(s,a)，我们实际上用的是它的近似函数Q(s,a;ω)Q(s,a;ω)；另一个是在off-policy中，行为策略与评估策略不同，理论上是需要引入importance sampling来进行修正的。实际上，这两个bias都通过满足了一个定理的条件来得以保证。详见Compatible Function Approximation.

## 三、算法整体流程

回顾DQN在Q-Learning基础上所做的改进：使用了深度神经网络做函数近似；使用经验回放；使用target网络。DDPG类似的也使用了深度神经网络，经验回放和target网络。不过DQN中的target更新是hard update，即每隔固定步数更新一次target网络，DDPG使用soft update，每一步都会更新target网络，只不过更新的幅度非常小。

附上原文的算法流程：
[![DDPG](https://wanjun0511.github.io/2017/11/19/DDPG/DDPG.png)](https://wanjun0511.github.io/2017/11/19/DDPG/DDPG.png)

[DDPG](https://wanjun0511.github.io/2017/11/19/DDPG/DDPG.png)

## 参考资料：

David Silver的课程：www0.cs.ucl.ac.uk/staff/D.Silver/web/Teaching.html
