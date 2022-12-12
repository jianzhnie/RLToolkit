from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from smac.env import StarCraft2Env
import numpy as np

# 独立的智能体在接收到观察和全局状态后会执行随机策略。
def main():
    env = StarCraft2Env(map_name="8m")
    env_info = env.get_env_info()

    n_actions = env_info["n_actions"]  # 获取动作维度 14
    n_agents = env_info["n_agents"]  # 存在多少个智能体 8
    print("n_agents: %d, n_actions: %d" % (n_agents, n_actions))
    n_episodes = 10
    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            # obs: list, length = 8, for 8 agents
            obs = env.get_obs()
            # state: shape (168, )
            state = env.get_state()
            env.render()  # Uncomment for rendering
            actions = []
            for agent_id in range(n_agents): # 对于每个智能体遍历循环
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print("Total reward in episode {} = {}".format(e, episode_reward))

    env.close()


if __name__ == "__main__":
    main()