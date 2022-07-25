import gym
import numpy as np
import torch
import torchvision.transforms as T
from gym.spaces import Box
from gym.wrappers import FrameStack


class ResizeObservation(gym.ObservationWrapper):

    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        transforms = T.Compose([T.Resize(self.shape), T.Normalize(0, 255)])
        observation = transforms(observation).squeeze(0)
        return observation


class SkipFrame(gym.Wrapper):

    def __init__(self, env, skip):
        """Return only every `skip`-th frame."""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward."""
        total_reward = 0.0
        done = False
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info


class GrayScaleObservation(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = Box(
            low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def permute_orientation(self, observation):
        # permute [H, W, C] array to [C, H, W] tensor
        observation = np.transpose(observation, (2, 0, 1))
        observation = torch.tensor(observation.copy(), dtype=torch.float)
        return observation

    def observation(self, observation):
        observation = self.permute_orientation(observation)
        transform = T.Grayscale()
        observation = transform(observation)
        return observation


if __name__ == '__main__':
    import gym_super_mario_bros
    from nes_py.wrappers import JoypadSpace

    # Apply Wrappers to environment
    # Initialize Super Mario environment
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')

    # Limit the action-space to
    #   0. walk right
    #   1. jump right
    env = JoypadSpace(env, [['right'], ['right', 'A']])

    env.reset()
    next_state, reward, done, info = env.step(action=0)
    print(f'{next_state.shape},\n {reward},\n {done},\n {info}')
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, shape=84)
    env = FrameStack(env, num_stack=4)
