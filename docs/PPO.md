

# PPO



## [Background](https://spinningup.openai.com/en/latest/algorithms/ppo.html#id4)

(Previously: [Background for TRPO](https://spinningup.openai.com/en/latest/algorithms/trpo.html#background))

PPO is motivated by the same question as TRPO: how can we take the biggest possible improvement step on a policy using the data we currently have, without stepping so far that we accidentally cause performance collapse? Where TRPO tries to solve this problem with a complex second-order method, PPO is a family of first-order methods that use a few other tricks to keep new policies close to old. PPO methods are significantly simpler to implement, and empirically seem to perform at least as well as TRPO.

There are two primary variants of PPO: PPO-Penalty and PPO-Clip.

**PPO-Penalty** approximately solves a KL-constrained update like TRPO, but penalizes the KL-divergence in the objective function instead of making it a hard constraint, and automatically adjusts the penalty coefficient over the course of training so that it’s scaled appropriately.

**PPO-Clip** doesn’t have a KL-divergence term in the objective and doesn’t have a constraint at all. Instead relies on specialized clipping in the objective function to remove incentives for the new policy to get far from the old policy.

Here, we’ll focus only on PPO-Clip (the primary variant used at OpenAI).

### [Quick Facts](https://spinningup.openai.com/en/latest/algorithms/ppo.html#id5)

- PPO is an on-policy algorithm.
- PPO can be used for environments with either discrete or continuous action spaces.
- The Spinning Up implementation of PPO supports parallelization with MPI.

### [Key Equations](https://spinningup.openai.com/en/latest/algorithms/ppo.html#id6)

PPO-clip updates policies via

![\theta_{k+1} = \arg \max_{\theta} \underset{s,a \sim \pi_{\theta_k}}{{\mathrm E}}\left[     L(s,a,\theta_k, \theta)\right],](https://spinningup.openai.com/en/latest/_images/math/96a52e61318720522e040e433c938ee829d54506.svg)

typically taking multiple steps of (usually minibatch) SGD to maximize the objective. Here ![L](https://spinningup.openai.com/en/latest/_images/math/3ffe1da701d78dd473975ebd2f875807611f7713.svg) is given by

![L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\; \text{clip}\left(\frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, 1 - \epsilon, 1+\epsilon \right) A^{\pi_{\theta_k}}(s,a) \right),](https://spinningup.openai.com/en/latest/_images/math/99621d5bcaccd056d6ca3aeb48a27bf8cc0e640c.svg)

in which ![\epsilon](https://spinningup.openai.com/en/latest/_images/math/c589a82739d7aa277bcf45e632d930d1c119b7ef.svg) is a (small) hyperparameter which roughly says how far away the new policy is allowed to go from the old.

This is a pretty complex expression, and it’s hard to tell at first glance what it’s doing, or how it helps keep the new policy close to the old policy. As it turns out, there’s a considerably simplified version [[1\]](https://spinningup.openai.com/en/latest/algorithms/ppo.html#id2) of this objective which is a bit easier to grapple with (and is also the version we implement in our code):

![L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}  A^{\pi_{\theta_k}}(s,a), \;\; g(\epsilon, A^{\pi_{\theta_k}}(s,a)) \right),](https://spinningup.openai.com/en/latest/_images/math/dd41a29292af3bc58c0c76bc7dba82a7355bf929.svg)

where

![g(\epsilon, A) = \left\{     \begin{array}{ll}     (1 + \epsilon) A & A \geq 0 \\     (1 - \epsilon) A & A < 0.     \end{array}     \right.](https://spinningup.openai.com/en/latest/_images/math/39f524858866b80e627840ba77a54360e3bac55e.svg)

To figure out what intuition to take away from this, let’s look at a single state-action pair ![(s,a)](https://spinningup.openai.com/en/latest/_images/math/4a1b4e2fc586f984a8edafbcae068c3f3c992402.svg), and think of cases.

**Advantage is positive**: Suppose the advantage for that state-action pair is positive, in which case its contribution to the objective reduces to

![L(s,a,\theta_k,\theta) = \min\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, (1 + \epsilon) \right)  A^{\pi_{\theta_k}}(s,a).](https://spinningup.openai.com/en/latest/_images/math/b4e46e01172264315e9e5d6c8bd2ced884d6602c.svg)

Because the advantage is positive, the objective will increase if the action becomes more likely—that is, if ![\pi_{\theta}(a|s)](https://spinningup.openai.com/en/latest/_images/math/400068784a9d13ffe96c61f29b4ab26ad5557376.svg) increases. But the min in this term puts a limit to how *much* the objective can increase. Once ![\pi_{\theta}(a|s) > (1+\epsilon) \pi_{\theta_k}(a|s)](https://spinningup.openai.com/en/latest/_images/math/cee08da41b29ab9355f2e4dac94de335c6eff03f.svg), the min kicks in and this term hits a ceiling of ![(1+\epsilon) A^{\pi_{\theta_k}}(s,a)](https://spinningup.openai.com/en/latest/_images/math/08d4d3bab53ce2aef0a6fd4d8e0e9f5cd0e4f7ca.svg). Thus: *the new policy does not benefit by going far away from the old policy*.

**Advantage is negative**: Suppose the advantage for that state-action pair is negative, in which case its contribution to the objective reduces to

![L(s,a,\theta_k,\theta) = \max\left( \frac{\pi_{\theta}(a|s)}{\pi_{\theta_k}(a|s)}, (1 - \epsilon) \right)  A^{\pi_{\theta_k}}(s,a).](https://spinningup.openai.com/en/latest/_images/math/b8b23f5e4578125c2d8fbfc66442629ff7a85fb5.svg)

Because the advantage is negative, the objective will increase if the action becomes less likely—that is, if ![\pi_{\theta}(a|s)](https://spinningup.openai.com/en/latest/_images/math/400068784a9d13ffe96c61f29b4ab26ad5557376.svg) decreases. But the max in this term puts a limit to how *much* the objective can increase. Once ![\pi_{\theta}(a|s) < (1-\epsilon) \pi_{\theta_k}(a|s)](https://spinningup.openai.com/en/latest/_images/math/82d6b288e893443689bf88b41b1f0f532c54f2f3.svg), the max kicks in and this term hits a ceiling of ![(1-\epsilon) A^{\pi_{\theta_k}}(s,a)](https://spinningup.openai.com/en/latest/_images/math/0aea7de5d8df7541d515b563b9c7bb0191e28b32.svg). Thus, again: *the new policy does not benefit by going far away from the old policy*.

What we have seen so far is that clipping serves as a regularizer by removing incentives for the policy to change dramatically, and the hyperparameter ![\epsilon](https://spinningup.openai.com/en/latest/_images/math/c589a82739d7aa277bcf45e632d930d1c119b7ef.svg) corresponds to how far away the new policy can go from the old while still profiting the objective.

> While this kind of clipping goes a long way towards ensuring reasonable policy updates, it is still possible to end up with a new policy which is too far from the old policy, and there are a bunch of tricks used by different PPO implementations to stave this off. In our implementation here, we use a particularly simple method: early stopping. If the mean KL-divergence of the new policy from the old grows beyond a threshold, we stop taking gradient steps.

When you feel comfortable with the basic math and implementation details, it’s worth checking out other implementations to see how they handle this issue!

| [[1\]](https://spinningup.openai.com/en/latest/algorithms/ppo.html#id1) | See [this note](https://drive.google.com/file/d/1PDzn9RPvaXjJFZkGeapMHbHGiWWW20Ey/view?usp=sharing) for a derivation of the simplified form of the PPO-Clip objective |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

### [Exploration vs. Exploitation](https://spinningup.openai.com/en/latest/algorithms/ppo.html#id7)

PPO trains a stochastic policy in an on-policy way. This means that it explores by sampling actions according to the latest version of its stochastic policy. The amount of randomness in action selection depends on both initial conditions and the training procedure. Over the course of training, the policy typically becomes progressively less random, as the update rule encourages it to exploit rewards that it has already found. This may cause the policy to get trapped in local optima.

### [Pseudocode](https://spinningup.openai.com/en/latest/algorithms/ppo.html#id8)

![](https://spinningup.openai.com/en/latest/_images/math/e62a8971472597f4b014c2da064f636ffe365ba3.svg)
