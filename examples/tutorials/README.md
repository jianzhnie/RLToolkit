<!--
 * @Author: jianzhnie
 * @LastEditors: jianzhnie
 * @Description: RLToolKit is a flexible and high-efficient reinforcement learning framework.
 * Copyright (c) 2022 by jianzhnie@126.com, All Rights Reserved.
-->
## 《强化学习入门实践》课程示例

针对强化学习初学者，提供了入门课程，展示最基础的5个强化学习算法代码示例（注意：本课程示例均基于**静态图框架**编写）。

## 课程大纲
+ 一、强化学习(RL)初印象
    + RL概述、入门路线
    + 实践：环境搭建（[lesson1](lesson1/gridworld.py) 的代码提供了格子环境世界的渲染封装）
+ 二、基于表格型方法求解RL
    + MDP、状态价值、Q表格
    + 实践： [Sarsa](lesson2/sarsa)、[Q-learning](lesson2/q_learning)
+ 三、基于神经网络方法求解RL
    + 函数逼近方法
    + 实践：[DQN](lesson3/dqn)
+ 四、基于策略梯度求解RL
    + 策略近似、策略梯度
    + 实践：[Policy Gradient](lesson4/policy_gradient)
+ 五、连续动作空间上求解RL
    + 实战：[DDPG](lesson5/ddpg)


## 使用说明

### 安装依赖（注意：请务必安装对应的版本）

+ Python 3.6/3.7
+ gym==0.18.0
+ torch 1.5+

可以直接安装本目录下的 `requirements.txt` 来完成以上依赖版本的适配。

```
pip install -r requirements.txt
```

### 运行示例

进入每个示例对应的代码文件夹中，运行
```
python train.py
```