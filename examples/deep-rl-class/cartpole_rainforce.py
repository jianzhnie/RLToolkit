import datetime
import json
import os
from collections import deque
from pathlib import Path

import gym
import imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from huggingface_hub import HfApi, Repository
from huggingface_hub.repocard import metadata_eval_result, metadata_save
from torch.distributions import Categorical


class Policy(nn.Module):

    def __init__(self, s_size, a_size, h_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(s_size, h_size)
        self.fc2 = nn.Linear(h_size, a_size)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return F.softmax(x, dim=1)

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        m = Categorical(probs)
        action = m.sample()
        return action.item(), m.log_prob(action)


def reinforce(policy, optimizer, n_training_episodes, max_t, gamma,
              print_every):
    # Help us to calculate the score during the training
    scores_deque = deque(maxlen=100)
    scores = []
    # Line 3 of pseudocode
    for i_episode in range(1, n_training_episodes + 1):
        saved_log_probs = []
        rewards = []
        state = env.reset()
        # Line 4 of pseudocode
        for t in range(max_t):
            action, log_prob = policy.act(state)
            saved_log_probs.append(log_prob)
            state, reward, done, _ = env.step(action)
            rewards.append(reward)
            if done:
                break
        scores_deque.append(sum(rewards))
        scores.append(sum(rewards))

        # Line 6 of pseudocode: calculate the return
        # Here, we calculate discounts for instance [0.99^1, 0.99^2, 0.99^3, ..., 0.99^len(rewards)]
        discounts = [gamma**i for i in range(len(rewards) + 1)]
        # We calculate the return by sum(gamma[t] * reward[t])
        R = sum([a * b for a, b in zip(discounts, rewards)])

        # Line 7:
        policy_loss = []
        for log_prob in saved_log_probs:
            policy_loss.append(-log_prob * R)
        policy_loss = torch.cat(policy_loss).sum()

        # Line 8:
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        if i_episode % print_every == 0:
            print('Episode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_deque)))

    return scores


def evaluate_agent(env, max_steps, n_eval_episodes, policy):
    """Evaluate the agent for ``n_eval_episodes`` episodes and returns average
    reward and std of reward.

    :param env: The evaluation environment
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param policy: The Reinforce agent
    """
    episode_rewards = []
    for episode in range(n_eval_episodes):
        state = env.reset()
        done = False
        total_rewards_ep = 0

        for step in range(max_steps):
            action, _ = policy.act(state)
            new_state, reward, done, info = env.step(action)
            total_rewards_ep += reward

            if done:
                break
            state = new_state
        episode_rewards.append(total_rewards_ep)
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)

    return mean_reward, std_reward


def record_video(env, policy, out_directory, fps=30):
    images = []
    done = False
    state = env.reset()
    img = env.render(mode='rgb_array')
    images.append(img)
    while not done:
        # Take the action (index) that have the maximum expected future reward given that state
        action, _ = policy.act(state)
        state, reward, done, info = env.step(action)
        # We directly put next_state = state for recording logic
        img = env.render(mode='rgb_array')
        images.append(img)
    imageio.mimsave(
        out_directory, [np.array(img) for i, img in enumerate(images)],
        fps=fps)


def package_to_hub(repo_id,
                   model,
                   hyperparameters,
                   eval_env,
                   video_fps=30,
                   local_repo_path='hub',
                   commit_message='Push Reinforce agent to the Hub',
                   token=None):
    _, repo_name = repo_id.split('/')

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
    repo = Repository(
        repo_local_path, clone_from=repo_url, use_auth_token=True)
    repo.git_pull()

    repo.lfs_track(['*.mp4'])

    # Step 1: Save the model
    torch.save(model, os.path.join(repo_local_path, 'model.pt'))

    # Step 2: Save the hyperparameters to JSON
    with open(Path(repo_local_path) / 'hyperparameters.json', 'w') as outfile:
        json.dump(hyperparameters, outfile)

    # Step 2: Evaluate the model and build JSON
    mean_reward, std_reward = evaluate_agent(
        eval_env, hyperparameters['max_t'],
        hyperparameters['n_evaluation_episodes'], model)

    # First get datetime
    eval_datetime = datetime.datetime.now()
    eval_form_datetime = eval_datetime.isoformat()

    evaluate_data = {
        'env_id': hyperparameters['env_id'],
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'n_evaluation_episodes': hyperparameters['n_evaluation_episodes'],
        'eval_datetime': eval_form_datetime,
        'hyperparameters': hyperparameters,
    }
    # Write a JSON file
    with open(Path(repo_local_path) / 'results.json', 'w') as outfile:
        json.dump(evaluate_data, outfile)

    # Step 3: Create the model card
    # Env id
    env_name = hyperparameters['env_id']

    metadata = {}
    metadata['tags'] = [
        env_name, 'reinforce', 'reinforcement-learning',
        'custom-implementation', 'deep-rl-class'
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
  # **Reinforce** Agent playing **{env_id}**
  This is a trained model of a **Reinforce** agent playing **{env_id}** .
  To learn to use this model and train yours check Unit 5 of the Deep Reinforcement Learning Class: https://github.com/huggingface/deep-rl-class/tree/main/unit5
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
    record_video(env, model, video_path, video_fps)

    # Push everything to hub
    print(f'Pushing repo {repo_name} to the Hugging Face Hub')
    repo.push_to_hub(commit_message=commit_message)

    print(
        f'Your model is pushed to the hub. You can view your model here: {repo_url}'
    )


if __name__ == '__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    env_id = 'CartPole-v1'
    # Create the env
    env = gym.make(env_id)

    # Create the evaluation env
    eval_env = gym.make(env_id)

    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = env.action_space.n

    print('_____OBSERVATION SPACE_____ \n')
    print('The State Space is: ', s_size)
    print('Sample observation',
          env.observation_space.sample())  # Get a random observation

    print('\n _____ACTION SPACE_____ \n')
    print('The Action Space is: ', a_size)
    print('Action Space Sample',
          env.action_space.sample())  # Take a random action

    cartpole_hyperparameters = {
        'h_size': 32,
        'n_training_episodes': 10000,
        'n_evaluation_episodes': 100,
        'max_t': 1000,
        'gamma': 0.995,
        'lr': 1e-2,
        'env_id': env_id,
        'state_space': s_size,
        'action_space': a_size,
    }

    # Create policy and place it to the device
    cartpole_policy = Policy(cartpole_hyperparameters['state_space'],
                             cartpole_hyperparameters['action_space'],
                             cartpole_hyperparameters['h_size']).to(device)
    cartpole_optimizer = optim.Adam(
        cartpole_policy.parameters(), lr=cartpole_hyperparameters['lr'])

    scores = reinforce(cartpole_policy, cartpole_optimizer,
                       cartpole_hyperparameters['n_training_episodes'],
                       cartpole_hyperparameters['max_t'],
                       cartpole_hyperparameters['gamma'], 100)

    mean_reward, std_reward = evaluate_agent(
        eval_env, cartpole_hyperparameters['max_t'],
        cartpole_hyperparameters['n_evaluation_episodes'], cartpole_policy)
    print(mean_reward, std_reward)

    from huggingface_hub import notebook_login
    notebook_login()

    username = 'jianzhnie'
    repo_name = f'Reinforce-{env_id}'
    repo_id = f'{username}/{repo_name}'
    package_to_hub(
        repo_id,
        cartpole_policy,  # The model we want to save
        cartpole_hyperparameters,  # Hyperparameters
        eval_env,  # Evaluation environment
        video_fps=30,
        local_repo_path='hub',
        commit_message='Push Reinforce agent to the Hub',
    )
