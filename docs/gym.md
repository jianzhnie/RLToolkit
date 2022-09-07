## gym.spaces.box，gym.spaces.discrete，gym.spaces.multi_discrete



1.discrete类

- Discrete类对应于一维离散空间
- 定义一个Discrete类的空间只需要一个参数n就可以了
- discrete space允许固定范围的非负数
- 每个时间步agent只采取离散空间中的一个动作，如离散空间中actions=[上、下、左、右]，一个时间步可能采取“上”这一个动作。

2.box类

- box类对应于多维连续空间
- Box空间可以定义多维空间，每一个维度可以用一个最低值和最大值来约束
- 定义一个多维的Box空间需要知道每一个维度的最小最大值，当然也要知道维数。

3.multidiscrete类

-  用于多维离散空间

- 多离散动作空间由一系列具有不同参数的离散动作空间组成

  - 它可以适应离散动作空间或连续（Box）动作空间

  - 表示游戏控制器或键盘非常有用，其中每个键都可以表示为离散的动作空间

  - 通过传递每个离散动作空间包含[min，max]的数组的数组进行参数化

  - 离散动作空间可以取从min到max的任何整数（包括两端值）



> 多智能体算法中在train开始的时候，把不同种类的动作建立成了各种不同的分布, 最后的动作输出的是分布，根据分布最后采样得到输出值。
>
> - Box 连续空间->DiagGaussianPdType （对角高斯概率分布）
> - Discrete离散空间->SoftCategoricalPdType（软分类概率分布）
> - MultiDiscrete连续空间->SoftMultiCategoricalPdType （多变量软分类概率分布）
> - 多二值变量连续空间->BernoulliPdType （伯努利概率分布）-



### 首先解释box，先看[gym官网](https://github.com/openai/gym/blob/master/gym/spaces/box.py)是如何定义的。

```python3
class Box(Space):
    """
    A (possibly unbounded) box in R^n. Specifically, a Box represents the
    Cartesian product of n closed intervals. Each interval has the form of one
    of [a, b], (-oo, b], [a, oo), or (-oo, oo).
    There are two common use cases:
    * Identical bound for each dimension::
        >>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)
    * Independent bound for each dimension::
        >>> Box(low=np.array([-1.0, -2.0]), high=np.array([2.0, 4.0]), dtype=np.float32)
        Box(2,)
    """
```

- 上述描述的直译：box（可能是无界的）在n维空间中。一个box代表n维封闭区间的笛卡尔积。每个区间都有[a, b]， (-oo, b)， [a, oo)，或(-oo, oo)的形式。

- - 需要注意的重点在于：box可以表示n维空间，并且区间有闭有开。

- 例子：每一维相同的限制：

- - 可以看到，此处Box的shape为(3, 4)，每一维区间的最小值为-1.0，最大值为2.0。

```python
>>> Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
        Box(3, 4)
```

- 如果不够直观，我们可以它sample出来看一下。

```python
import numpy as np
from gym.spaces.box import Box

my_box = Box(low=-1.0, high=2.0, shape=(3, 4), dtype=np.float32)
my_box_sample = my_box.sample()

print(my_box)
print(my_box_sample)

# 输出
Box(3, 4)
[[ 0.34270912 -0.17985763  1.7716838  -0.71165234]
 [ 0.5638914  -0.6311684  -0.28997722 -0.19067103]
 [-0.6750097   0.99941856  1.1923424  -0.9933872 ]]
```
