import random

import gym
import numpy as np


def initialize_q_table(state_space, action_space):
    q_table = np.zeros((state_space, action_space))
    return q_table


def epsilon_greedy_policy(Qtable, state, epsilon):
    # Randomly generate a number between 0 and 1
    random_int = random.uniform(0, 1)
    # if random_int > greater than epsilon --> exploitation
    if random_int > epsilon:
        # Take the action with the highest value given a state
        # np.argmax can be useful here
        action = np.argmax(Qtable[state])
    # else --> exploration
    else:
        action = env.action_space.sample()

    return action


def greedy_policy(Qtable, state):
    # Exploitation: take the action with the highest state, action value
    action = np.argmax(Qtable[state])

    return action


def train(n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps,
          Qtable, env):
    for episode in range(n_training_episodes):
        # Reduce epsilon (because we need less and less exploration)
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(
            -decay_rate * episode)
        # Reset the environment
        state = env.reset()
        done = False

        # repeat
        for step in range(max_steps):
            # Choose the action At using epsilon greedy policy
            action = epsilon_greedy_policy(Qtable, state, epsilon)

            # Take action At and observe Rt+1 and St+1
            # Take the action (a) and observe the outcome state(s') and reward (r)
            new_state, reward, done, info = env.step(action)

            # Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]
            cur_state_value = Qtable[state][action]
            td_target = reward + gamma * np.max(Qtable[new_state])
            update_state_value = cur_state_value + learning_rate * (
                td_target - cur_state_value)

            Qtable[state][action] = update_state_value
            # Qtable[state][action] = Qtable[state][action] + learning_rate * (
            #     reward + gamma * np.max(Qtable[new_state]) -
            #     Qtable[state][action])
            # If done, finish the episode
            if done:
                break
            # Our state is the new state
            state = new_state
    return Qtable


def evaluate_agent(env, max_steps, n_eval_episodes, Q, seed):
    """Evaluate the agent for ``n_eval_episodes`` episodes and returns average
    reward and std of reward.

    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param Q: The Q-table
    :param seed: The evaluation seed array (for taxi-v3)
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        if seed:
            state = env.reset(seed=seed[episode])
        else:
            state = env.reset()
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            # Take the action (index) that have the maximum expected future reward given that state
            action = np.argmax(Q[state][:])
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', map_name='4x4', is_slippery=False)

    # We create our environment with gym.make("<name_of_the_environment>")
    env.reset()
    print('_____OBSERVATION SPACE_____ \n')
    print('Observation Space', env.observation_space)
    print('Sample observation',
          env.observation_space.sample())  # Get a random observation

    # Step2: Create and Initialize the Q-table
    state_space = env.observation_space.n
    print('There are ', state_space, ' possible states')

    action_space = env.action_space.n
    print('There are ', action_space, ' possible actions')

    # Let's create our Qtable of size (state_space, action_space) and initialized each values at 0 using np.zeros

    Qtable_frozenlake = initialize_q_table(state_space, action_space)
    # Step 3: Define the epsilon-greedy polic
    # Step 4: Define the greedy policy 🤖
    # Step 5: Define the hyperparameters ⚙️
    # Training parameters
    n_training_episodes = 10000  # Total training episodes
    learning_rate = 0.7  # Learning rate

    # Evaluation parameters
    n_eval_episodes = 100  # Total number of test episodes

    # Environment parameters
    env_id = 'FrozenLake-v1'  # Name of the environment
    max_steps = 99  # Max steps per episode
    gamma = 0.95  # Discounting rate
    eval_seed = []  # The evaluation seed of the environment

    # Exploration parameters
    epsilon = 1.0  # Exploration rate
    max_epsilon = 1.0  # Exploration probability at start
    min_epsilon = 0.05  # Minimum exploration probability
    decay_rate = 0.0005  # Exponential decay rate for exploration prob
    Qtable_frozenlake = train(n_training_episodes, min_epsilon, max_epsilon,
                              decay_rate, max_steps, Qtable_frozenlake, env)
    print(Qtable_frozenlake)

    # Evaluate our Agent
    mean_reward, std_reward = evaluate_agent(env, max_steps, n_eval_episodes,
                                             Qtable_frozenlake, eval_seed)
    print(f'Mean_reward={mean_reward:.2f} +/- {std_reward:.2f}')
