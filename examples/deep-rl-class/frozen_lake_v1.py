import datetime
import json
import random
from pathlib import Path

import gym
import imageio
import numpy as np
import pickle5 as pickle
from huggingface_hub import HfApi, Repository
from huggingface_hub.repocard import metadata_eval_result, metadata_save


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


def record_video(env, Qtable, out_directory, fps=1):
    images = []
    done = False
    state = env.reset(seed=random.randint(0, 500))
    img = env.render(mode='rgb_array')
    images.append(img)
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(Qtable[state][:])
        state, reward, done, info = env.step(
            action)  # We directly put next_state = state for recording logic
        img = env.render(mode='rgb_array')
        images.append(img)

    imageio.mimsave(out_directory,
                    [np.array(img) for i, img in enumerate(images)],
                    fps=fps)


def push_to_hub(repo_name,
                model,
                env,
                video_fps=1,
                local_repo_path='hub',
                commit_message='Push Q-Learning agent to Hub',
                token=None):
    eval_env = env
    # Step 1: Clone or create the repo
    # Create the repo (or clone its content if it's nonempty)
    api = HfApi()

    repo_url = api.create_repo(
        name=repo_name,
        token=token,
        private=False,
        exist_ok=True,
    )

    # Git pull
    repo_local_path = Path(local_repo_path) / repo_name
    repo = Repository(repo_local_path,
                      clone_from=repo_url,
                      use_auth_token=True)
    repo.git_pull()

    repo.lfs_track(['*.mp4'])

    # Step 1: Save the model
    if env.spec.kwargs.get('map_name'):
        model['map_name'] = env.spec.kwargs.get('map_name')
        if env.spec.kwargs.get('is_slippery', '') is False:
            model['slippery'] = False

    print(model)

    # Pickle the model
    with open(Path(repo_local_path) / 'q-learning.pkl', 'wb') as f:
        pickle.dump(model, f)

    # Step 2: Evaluate the model and build JSON
    mean_reward, std_reward = evaluate_agent(eval_env, model['max_steps'],
                                             model['n_eval_episodes'],
                                             model['qtable'],
                                             model['eval_seed'])

    # First get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
        'env_id': model['env_id'],
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'n_eval_episodes': model['n_eval_episodes'],
        'eval_datetime': eval_form_datetime,
    }
    # Write a JSON file
    with open(Path(repo_local_path) / 'results.json', 'w') as outfile:
        json.dump(evaluate_data, outfile)

    # Step 3: Create the model card
    # Env id
    env_name = model['env_id']
    if env.spec.kwargs.get('map_name'):
        env_name += '-' + env.spec.kwargs.get('map_name')

    if env.spec.kwargs.get('is_slippery', '') is False:
        env_name += '-' + 'no_slippery'

    metadata = {}
    metadata['tags'] = [
        env_name, 'q-learning', 'reinforcement-learning',
        'custom-implementation'
    ]

    # Add metrics
    eval = metadata_eval_result(
        model_pretty_name=repo_name,
        task_pretty_name='reinforcement-learning',
        task_id='reinforcement-learning',
        metrics_pretty_name='mean_reward',
        metrics_id='mean_reward',
        metrics_value=f'{mean_reward:.2f} +/- {std_reward:.2f}',
        dataset_pretty_name=env_name,
        dataset_id=env_name,
    )

    # Merges both dictionaries
    metadata = {**metadata, **eval}

    model_card = f"""
  # **Q-Learning** Agent playing **{env_id}**
  This is a trained model of a **Q-Learning** agent playing **{env_id}** .
  """

    model_card += """
  ## Usage
  ```python
  """

    model_card += f"""model = load_from_hub(repo_id="{repo_name}", filename="q-learning.pkl")

  # Don't forget to check if you need to add additional attributes (is_slippery=False etc)
  env = gym.make(model["env_id"])

  evaluate_agent(env, model["max_steps"], model["n_eval_episodes"], model["qtable"], model["eval_seed"])
  """

    model_card += """
  ```
  """

    readme_path = repo_local_path / 'README.md'
    readme = ''
    if readme_path.exists():
        with readme_path.open('r', encoding='utf8') as f:
            readme = f.read()
    else:
        readme = model_card

    with readme_path.open('w', encoding='utf-8') as f:
        f.write(readme)

    # Save our metrics to Readme metadata
    metadata_save(readme_path, metadata)

    # Step 4: Record a video
    video_path = repo_local_path / 'replay.mp4'
    record_video(env, model['qtable'], video_path, video_fps)

    # Push everything to hub
    print(f'Pushing repo {repo_name} to the Hugging Face Hub')
    repo.push_to_hub(commit_message=commit_message)

    print(
        f'Your model is pushed to the hub. You can view your model here: {repo_url}'
    )


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

    model = {
        'env_id': env_id,
        'max_steps': max_steps,
        'n_training_episodes': n_training_episodes,
        'n_eval_episodes': n_eval_episodes,
        'eval_seed': eval_seed,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'epsilon': epsilon,
        'max_epsilon': max_epsilon,
        'min_epsilon': min_epsilon,
        'decay_rate': decay_rate,
        'qtable': Qtable_frozenlake
    }

    from huggingface_hub import notebook_login
    notebook_login()
    username = 'jianzhnie'  # FILL THIS
    repo_name = 'q_FrozenLake_v1_4x4_noSlippery'
    push_to_hub(repo_name=f'{repo_name}', model=model, env=env)
