import gym
# import pybullet_envs
from huggingface_hub import notebook_login
from huggingface_sb3 import package_to_hub
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

if __name__ == '__main__':
    env_id = 'Walker2DBulletEnv-v0'
    # Create the env
    env = gym.make(env_id)

    # Get the state space and action space
    s_size = env.observation_space.shape[0]
    a_size = env.action_space
    print('_____OBSERVATION SPACE_____ \n')
    print('The State Space is: ', s_size)
    print('Sample observation',
          env.observation_space.sample())  # Get a random observation
    print('\n _____ACTION SPACE_____ \n')
    print('The Action Space is: ', a_size)
    print('Action Space Sample',
          env.action_space.sample())  # Take a random action
    env = make_vec_env(env_id, n_envs=4)

    # Adding this wrapper to normalize the observation and the reward
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)

    model = A2C(
        policy='MlpPolicy',
        env=env,
        gae_lambda=0.9,
        gamma=0.99,
        learning_rate=0.00096,
        max_grad_norm=0.5,
        n_steps=5,
        vf_coef=0.4,
        ent_coef=0.0,
        tensorboard_log=f'work_dirs/{env_id}/tensorboard',
        policy_kwargs=dict(log_std_init=-2, ortho_init=False),
        normalize_advantage=False,
        use_rms_prop=True,
        use_sde=True,
        verbose=1)
    model.learn(200000)
    # Save the model and  VecNormalize statistics when saving the agent
    model.save(f'work_dirs/{env_id}/a2c-Walker2DBulletEnv-v0')
    env.save(f'work_dirs/{env_id}/vec_normalize.pkl')

    # Load the saved statistics
    eval_env = DummyVecEnv([lambda: gym.make('Walker2DBulletEnv-v0')])
    eval_env = VecNormalize.load(f'work_dirs/{env_id}/vec_normalize.pkl',
                                 eval_env)

    #  do not update them at test time
    eval_env.training = False
    # reward normalization is not needed at test time
    eval_env.norm_reward = False

    # Load the agent
    model = A2C.load(f'work_dirs/{env_id}/a2c-Walker2DBulletEnv-v0')

    mean_reward, std_reward = evaluate_policy(model, env)

    print(f'Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}')

    notebook_login()
    package_to_hub(
        model=model,
        model_name=f'a2c-{env_id}',
        model_architecture='A2C',
        env_id=env_id,
        eval_env=eval_env,
        repo_id=f'jianzhnie/a2c-v1-{env_id}',
        commit_message='Initial commit',
        logs=f'work_dirs/{env_id}')
