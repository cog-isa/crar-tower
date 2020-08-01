import pathlib

import numpy as np
import gym

from legacy.env import LegacyCatcherEnv


def make_obstacle_tower_env(ctx):
    from obstacle_tower_env import ObstacleTowerEnv

    base_path = pathlib.Path(__file__).parent.absolute()
    path_to_binary = (base_path / '../obstacle-tower-env/ObstacleTower/obstacletower.x86_64').as_posix()
    env = ObstacleTowerEnv(environment_filename=path_to_binary,
                           greyscale=True)
    return env


class ObstacleTowerEnvStub:
    def inputDimensions(self):
        frame_stack_size = 1
        obs_shape = (1, 84, 84)
        return [(frame_stack_size,) + obs_shape]

    def observationType(self, subject):
        return np.uint8

    def nActions(self):
        return 54


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
