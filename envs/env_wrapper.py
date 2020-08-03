import pathlib

import numpy as np
import gym

from obstacle_tower_env import ObstacleTowerEnv
from legacy.env import LegacyCatcherEnv


def make_obstacle_tower_env(ctx):
    base_path = pathlib.Path(__file__).parent.absolute()
    path_to_binary = (base_path / '../obstacle-tower-env/ObstacleTower/obstacletower.x86_64').as_posix()
    env = ObstacleTowerEnvWrapper(environment_filename=path_to_binary,
                                  greyscale=False)
    return env


class ObstacleTowerEnvWrapper(ObstacleTowerEnv):
    @property
    def observation_space(self):
        return gym.spaces.Box(0, 255, shape=(3, 84, 84))

    def _preprocess_observation(self, obs):
        # transpose to channels first
        obs = np.transpose(obs, axes=(2, 0, 1))

        return obs

    def step(self, action):
        obs, reward, done, info = super().step(action)
        prep_obs = self._preprocess_observation(obs)
        return prep_obs, reward, done, info

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        prep_obs = self._preprocess_observation(obs)
        return prep_obs


class ObstacleTowerEnvStub:
    def inputDimensions(self):
        obs_shape = (1, 3, 84, 84)
        return [obs_shape]

    def observationType(self, subject):
        return np.uint8

    def nActions(self):
        return 54


class DebugObstacleTowerEnv(ObstacleTowerEnvWrapper):
    def __init__(self, *args, **kwargs):
        self._action_space = gym.spaces.Discrete(54)
        self._observation_space = gym.spaces.Box(0, 255, shape=(3, 84, 84))

        self._obs = np.ones((3, 84, 84), dtype=np.uint8)

    def reset(self):
        return self._obs

    def step(self, action):
        return self._obs, 0.0, False, {}


class CatcherEnv(LegacyCatcherEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = gym.spaces.Box(0, 1, shape=(1, 36, 36))
        self.action_space = gym.spaces.Discrete(2)

    def reset(self, mode=-1):
        obs = super(CatcherEnv, self).reset()
        return [np.zeros((36, 36))]

    def step(self, action):
        reward = self.act(action)
        obs = self.observe()
        done = self.inTerminalState()
        info = {}
        return obs, reward, done, info
