from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import beta


class BernoulliBandit(object):
    """伯努利多臂老虎机,输入K表示拉杆个数."""

    def __init__(self, num_bandit, probs=None):
        assert probs is None or len(probs) == num_bandit

        if probs is None:
            # 随机生成K个0～1的数,作为拉动每根拉杆的获奖概率
            self.probs = np.random.uniform(size=num_bandit)
        else:
            self.probs = probs

        self.best_idx = np.argmax(self.probs)  # 获奖概率最大的拉杆
        self.best_prob = self.probs[self.best_idx]  # 最大的获奖概率
        self.num_bandit = num_bandit

    def step(self, k):
        # 当玩家选择了k号拉杆后,根据拉动该老虎机的k号拉杆获得奖励的概率返回1（获奖）或0（未获奖）
        if np.random.rand() < self.probs[k]:
            reward = 1
        else:
            reward = 0
        return reward


class Solver(object):
    """多臂老虎机算法基本框架."""

    def __init__(self, bandit: BernoulliBandit):
        """bandit (Bandit): the target bandit to solve."""
        self.bandit = bandit
        self.counts = np.zeros(self.bandit.num_bandit)  # 每根拉杆的尝试次数
        self.regret = 0.  # 当前步的累积懊悔
        self.actions = []  # 维护一个列表,记录每一步的动作
        self.regrets = []  # 维护一个列表,记录每一步的累积懊悔

    def update_regret(self, k):
        # 计算累积懊悔并保存,k为本次动作选择的拉杆的编号
        curr_regret = self.bandit.best_prob - self.bandit.probs[k]
        self.regret += curr_regret
        self.regrets.append(self.regret)

    @property
    def estimated_probs(self):
        raise NotImplementedError

    def run_one_step(self):
        # 返回当前动作选择哪一根拉杆,由每个具体的策略实现
        raise NotImplementedError

    def run(self, num_steps):
        # 运行一定次数,num_steps为总运行次数
        for _ in range(num_steps):
            k = self.run_one_step()
            self.counts[k] += 1
            self.actions.append(k)
            self.update_regret(k)


class EpsilonGreedy(Solver):
    """epsilon贪婪算法,继承Solver类."""

    def __init__(self, bandit: BernoulliBandit, epsilon=0.01, init_prob=1.0):
        super(EpsilonGreedy, self).__init__(bandit)
        """
        eps (float): the probability to explore at each time step.
        init_prob (float): default to be 1.0; optimistic initialization
        """
        assert 0. <= epsilon <= 1.0
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.num_bandit)

    @property
    def estimated_probs(self):
        return self.estimates

    def run_one_step(self):
        if np.random.random() < self.epsilon:
            k = np.random.randint(0, self.bandit.num_bandit)  # 随机选择一根拉杆
        else:
            k = np.argmax(self.estimates)  # 选择期望奖励估值最大的拉杆
        reward = self.bandit.step(k)  # 得到本次动作的奖励
        self.estimates[k] += 1. / (self.counts[k] + 1) * (
            reward - self.estimates[k])
        return k


class DecayingEpsilonGreedy(Solver):
    """epsilon值随时间衰减的epsilon-贪婪算法,继承Solver类."""

    def __init__(self, bandit: BernoulliBandit, epsilon=1.0, init_prob=1.0):
        super(DecayingEpsilonGreedy, self).__init__(bandit)
        self.epsilon = epsilon
        # 初始化拉动所有拉杆的期望奖励估值
        self.estimates = np.array([init_prob] * self.bandit.num_bandit)
        self.total_count = 0

    @property
    def estimated_probs(self):
        return self.estimates

    def run_one_step(self):
        self.total_count += 1
        self.epsilon = 1.0 / self.total_count
        if np.random.random() < self.epsilon:  # epsilon值随时间衰减
            k = np.random.randint(0, self.bandit.num_bandit)
        else:
            k = np.argmax(self.estimates)

        reward = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (
            reward - self.estimates[k])

        return k


class UCB(Solver):
    """UCB算法,继承Solver类."""

    def __init__(self, bandit: BernoulliBandit, coef, init_prob=1.0):
        super(UCB, self).__init__(bandit)
        self.total_count = 0
        self.estimates = np.array([init_prob] * self.bandit.num_bandit)
        self.coef = coef

    @property
    def estimated_probs(self):
        return self.estimates

    def run_one_step(self):
        self.total_count += 1
        ucb = self.estimates + self.coef * np.sqrt(
            np.log(self.total_count) / (2 * (self.counts + 1)))
        # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self.estimates[k] += 1. / (self.counts[k] + 1) * (
            r - self.estimates[k])
        return k


class BayesianUCB(Solver):
    """Assuming Beta prior."""

    def __init__(self, bandit: BernoulliBandit, coef=3, init_a=1, init_b=1):
        super(BayesianUCB, self).__init__(bandit)
        self.coef = coef
        self._a = np.array([init_a] *
                           self.bandit.num_bandit)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.array([init_b] *
                           self.bandit.num_bandit)  # 列表,表示每根拉杆奖励为0的次数

    @property
    def estimated_probs(self):
        return self._a / (self._a + self._b)

    def run_one_step(self):
        ucb = self._a / (self._a + self._b) + beta.std(self._a,
                                                       self._b) * self.coef
        # 计算上置信界
        k = np.argmax(ucb)  # 选出上置信界最大的拉杆
        r = self.bandit.step(k)
        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k


class ThompsonSampling(Solver):
    """汤普森采样算法,继承Solver类."""

    def __init__(self, bandit: BernoulliBandit, init_a=1, init_b=1):
        super(ThompsonSampling, self).__init__(bandit)
        """
        init_a (int): initial value of a in Beta(a, b).
        init_b (int): initial value of b in Beta(a, b).
        """
        self._a = np.array([init_a] *
                           self.bandit.num_bandit)  # 列表,表示每根拉杆奖励为1的次数
        self._b = np.array([init_b] *
                           self.bandit.num_bandit)  # 列表,表示每根拉杆奖励为0的次数

    @property
    def estimated_probs(self):
        return self._a / (self._a + self._b)

    def run_one_step(self):
        samples = np.random.beta(self._a, self._b)  # 按照Beta分布采样一组奖励样本
        k = np.argmax(samples)  # 选出采样奖励最大的拉杆
        r = self.bandit.step(k)

        self._a[k] += r  # 更新Beta分布的第一个参数
        self._b[k] += (1 - r)  # 更新Beta分布的第二个参数
        return k


def plot_results(solvers, solver_names, figname):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称.

    Plot the results by multi-armed bandit solvers.
    Args:
        solvers (list<Solver>): All of them should have been fitted.
        solver_names (list<str)
        figname (str)
    """
    assert len(solvers) == len(solver_names)
    assert all(map(lambda s: isinstance(s, Solver), solvers))
    assert all(map(lambda s: len(s.regrets) > 0, solvers))

    bandit = solvers[0].bandit

    fig = plt.figure(figsize=(18, 5))
    fig.subplots_adjust(bottom=0.3, wspace=0.3)

    ax1 = fig.add_subplot(131)
    ax2 = fig.add_subplot(132)
    ax3 = fig.add_subplot(133)

    # Sub.fig. 1: Regrets in time.
    for i, s in enumerate(solvers):
        ax1.plot(range(len(s.regrets)), s.regrets, label=solver_names[i])

    ax1.set_xlabel('Time step')
    ax1.set_ylabel('Cumulative regret')
    ax1.legend(loc=9, bbox_to_anchor=(1.82, -0.25), ncol=5)
    ax1.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 2: Probabilities estimated by solvers.
    sorted_indices = sorted(
        range(bandit.num_bandit), key=lambda x: bandit.probs[x])
    ax2.plot(
        range(bandit.num_bandit), [bandit.probs[x] for x in sorted_indices],
        'k--',
        markersize=12)
    for s in solvers:
        ax2.plot(
            range(bandit.num_bandit),
            [s.estimated_probs[x] for x in sorted_indices],
            'x',
            markeredgewidth=2)
    ax2.set_xlabel('Actions sorted by ' + r'θ')
    ax2.set_ylabel('Estimated')
    ax2.grid('k', ls='--', alpha=0.3)

    # Sub.fig. 3: Action counts
    for s in solvers:
        ax3.plot(
            range(bandit.num_bandit),
            np.array(s.counts) / float(len(solvers[0].regrets)),
            ls='solid',
            lw=2)
    ax3.set_xlabel('Actions')
    ax3.set_ylabel('Frac. # trials')
    ax3.grid('k', ls='--', alpha=0.3)

    plt.savefig(figname)


def plot_results_(solvers, solver_names):
    """生成累积懊悔随时间变化的图像。输入solvers是一个列表,列表中的每个元素是一种特定的策略。
    而solver_names也是一个列表,存储每个策略的名称."""

    for idx, solver in enumerate(solvers):
        time_list = range(len(solver.regrets))
        plt.plot(time_list, solver.regrets, label=solver_names[idx])
    plt.xlabel('Time steps')
    plt.ylabel('Cumulative regrets')
    plt.title('%d-armed bandit' % solvers[0].bandit.num_bandit)
    plt.legend()
    plt.show()


def experiment(K, N):
    """Run a small experiment on solving a Bernoulli bandit with K slot
    machines, each with a randomly initialized reward probability.

    Args:
        K (int): number of slot machiens.
        N (int): number of time steps to try.
    """

    bandit = BernoulliBandit(K)
    print('Randomly generated Bernoulli bandit has reward probabilities:\n',
          bandit.probs)
    print('The best machine has index: {} and proba: {}'.format(
        max(range(K), key=lambda i: bandit.probs[i]), max(bandit.probs)))

    test_solvers = [
        # EpsilonGreedy(bandit, 0),
        # EpsilonGreedy(bandit, 1),
        EpsilonGreedy(bandit, 0.01),
        UCB(bandit, coef=1.0),
        BayesianUCB(bandit, 3, 1, 1),
        ThompsonSampling(bandit, 1, 1)
    ]
    names = [
        # 'Full-exploitation', 'Full-exploration',
        r'ϵ' + '-Greedy',
        'UCB1',
        'Bayesian UCB',
        'Thompson Sampling'
    ]

    for s in test_solvers:
        s.run(N)

    plot_results(test_solvers, names, 'results_K{}_N{}.png'.format(K, N))


def plot_single():

    np.random.seed(1)  # 设定随机种子,使实验具有可重复性
    K = 10  # 10
    bandit_10_arm = BernoulliBandit(K)
    print('随机生成了一个%d臂伯努利老虎机' % K)
    print('获奖概率最大的拉杆为%d号,其获奖概率为%.4f' %
          (bandit_10_arm.best_idx, bandit_10_arm.best_prob))

    np.random.seed(1)
    epsilon_greedy_solver = EpsilonGreedy(bandit_10_arm, epsilon=0.01)
    epsilon_greedy_solver.run(50000)
    print('epsilon-贪婪算法的累积懊悔为：', epsilon_greedy_solver.regret)
    plot_results_([epsilon_greedy_solver], ['EpsilonGreedy'])

    np.random.seed(0)
    epsilons = [1e-4, 0.01, 0.1, 0.25, 0.5, 1.0]
    epsilon_greedy_solver_list = [
        EpsilonGreedy(bandit_10_arm, epsilon=e) for e in epsilons
    ]
    epsilon_greedy_solver_names = ['epsilon={}'.format(e) for e in epsilons]
    for solver in epsilon_greedy_solver_list:
        solver.run(50000)

    plot_results_(epsilon_greedy_solver_list, epsilon_greedy_solver_names)

    np.random.seed(1)
    decaying_epsilon_greedy_solver = DecayingEpsilonGreedy(bandit_10_arm)
    decaying_epsilon_greedy_solver.run(50000)
    print('epsilon值衰减的贪婪算法的累积懊悔为：', decaying_epsilon_greedy_solver.regret)
    plot_results_([decaying_epsilon_greedy_solver], ['DecayingEpsilonGreedy'])

    np.random.seed(1)
    coef = 1  # 控制不确定性比重的系数
    UCB_solver = UCB(bandit_10_arm, coef)
    UCB_solver.run(50000)
    print('上置信界算法的累积懊悔为：', UCB_solver.regret)
    plot_results_([UCB_solver], ['UCB'])

    np.random.seed(1)
    thompson_sampling_solver = ThompsonSampling(bandit_10_arm)
    thompson_sampling_solver.run(50000)
    print('汤普森采样算法的累积懊悔为：', thompson_sampling_solver.regret)
    plot_results_([thompson_sampling_solver], ['ThompsonSampling'])


if __name__ == '__main__':
    experiment(10, 5000)
