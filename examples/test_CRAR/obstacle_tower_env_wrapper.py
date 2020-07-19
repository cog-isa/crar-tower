import numpy as np

from obstacle_tower_env import ObstacleTowerEnv


class ObstacleTowerEnvWrapper:
    """
    Translates calls from DEER framework to obstacle-tower-env
    """
    # stub for framework's needs
    VALIDATION_MODE = 0

    def __init__(self):
        path_to_binary = '../../obstacle-tower-env/ObstacleTower/obstacletower.x86_64'
        self._env = ObstacleTowerEnv(environment_filename=path_to_binary,
                                     greyscale=True)
        self._obs = None
        self._done = None

    def reset(self, mode):
        self._obs = self._env.reset()

    def act(self, action):
        obs, reward, done, info = self._env.step(action)

        self._obs = obs
        self._done = done

        return reward

    def inputDimensions(self):
        frame_stack_size = 1
        obs_shape = (84, 84, 1)
        return [(frame_stack_size,) + obs_shape]

    def observationType(self, subject):
        return np.uint8

    def nActions(self):
        return 54

    def observe(self):
        return [self._obs]

    def inTerminalState(self):
        return self._done