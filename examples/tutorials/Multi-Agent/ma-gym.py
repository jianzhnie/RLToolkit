import time

import gym

if __name__ == '__main__':

    env = gym.make('ma_gym:PongDuel-v0')
    done_n = [False for _ in range(env.n_agents)]
    ep_reward = 0

    obs_n = env.reset()
    while not all(done_n):
        env.render()
        time.sleep(0.1)
        obs_n, reward_n, done_n, info = env.step(env.action_space.sample())
        print(type(obs_n[0]))
        print(type(reward_n))
        ep_reward += sum(reward_n)
    env.close()
