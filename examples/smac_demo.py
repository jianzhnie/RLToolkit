import numpy as np
from smac.env import StarCraft2Env
from smac.env.pettingzoo import StarCraft2PZEnv


def main():
    env = StarCraft2PZEnv.env(map_name='8m')
    env = StarCraft2Env(map_name='8m')
    env_info = env.get_env_info()

    n_actions = env_info['n_actions']
    n_agents = env_info['n_agents']

    print('n_actions: ', n_actions, 'n_agents: ', n_agents)
    n_episodes = 100

    for e in range(n_episodes):
        env.reset()
        terminated = False
        episode_reward = 0

        while not terminated:
            obs = env.get_obs()
            state = env.get_state()
            obs = env.get_obs()
            print(obs.space.shape, state.space.shape)
            env.render()  # Uncomment for rendering
            actions = []
            for agent_id in range(n_agents):
                avail_actions = env.get_avail_agent_actions(agent_id)
                avail_actions_ind = np.nonzero(avail_actions)[0]
                action = np.random.choice(avail_actions_ind)
                actions.append(action)

            reward, terminated, _ = env.step(actions)
            episode_reward += reward

        print('Total reward in episode {} = {}'.format(e, episode_reward))

    env.close()


if __name__ == '__main__':
    main()
