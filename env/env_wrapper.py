from obstacle_tower_env import ObstacleTowerEnv


def make_env():
    path_to_binary = '../obstacle-tower-env/ObstacleTower/obstacletower.x86_64'
    env = ObstacleTowerEnv(environment_filename=path_to_binary,
                           greyscale=True)
    return env


class EnvStub:
    def inputDimensions(self):
        frame_stack_size = 1
        obs_shape = (1, 84, 84)
        return [(frame_stack_size,) + obs_shape]

    def nActions(self):
        return 54