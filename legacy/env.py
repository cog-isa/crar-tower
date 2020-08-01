import numpy as np

from legacy.base_classes.environment import Environment


class LegacyCatcherEnv(Environment):
    VALIDATION_MODE = 0

    def __init__(self, rng, **kwargs):

        self._mode = -1
        self._mode_score = 0.0
        self._mode_episode_count = 0

        self._height = 10  # 15
        self._width = 10  # preferably an odd number so that it's symmetrical
        self._width_paddle = 1
        self._nx_block = 2  # self._width#2 #number of different x positions of the falling blocks
        self._higher_dim_obs = kwargs["higher_dim_obs"]
        self._reverse = kwargs["reverse"]

        if (self._nx_block == 1):
            self._x_block = self._width // 2
        else:
            rand = np.random.randint(self._nx_block)  # random selection of the pos for falling block
            self._x_block = rand * (
                        (self._width - 1) // (self._nx_block - 1))  # traduction in a number in [0,self._width] of rand

    def reset(self, mode=-1):
        if mode == LegacyCatcherEnv.VALIDATION_MODE:
            if self._mode != LegacyCatcherEnv.VALIDATION_MODE:
                self._mode = LegacyCatcherEnv.VALIDATION_MODE
                self._mode_score = 0.0
                self._mode_episode_count = 0
                np.random.seed(seed=11)  # Seed the generator so that the sequence of falling blocks is the same in validation
            else:
                self._mode_episode_count += 1
        elif self._mode != -1:  # and thus mode == -1
            self._mode = -1

        self.y = self._height - 1
        self.x = np.random.randint(self._width - self._width_paddle + 1)  # self._width//2
        if (self._nx_block == 1):
            self._x_block = self._width // 2
        else:
            rand = np.random.randint(self._nx_block)  # random selection of the pos for falling block
            self._x_block = rand * (
                        (self._width - 1) // (self._nx_block - 1))  # traduction in a number in [0,self._width] of rand

        return [1 * [self._height * [self._width * [0]]]]  # [0,0,1]+[0,1,0]

    def act(self, action):
        """Applies the agent action [action] on the environment.

        Parameters
        -----------
        action : int
            The action selected by the agent to operate on the environment. Should be an identifier
            included between 0 included and nActions() excluded.
        """

        if (action == 0):
            self.x = max(self.x - 1, 0)
        if (action == 1):
            self.x = min(self.x + 1, self._width - self._width_paddle)

        self.y = self.y - 1

        if (self.y == 0 and self.x > self._x_block - 1 - self._width_paddle and self.x <= self._x_block + 1):
            self.reward = 1
        elif (self.y == 0):
            self.reward = -1
        else:
            self.reward = 0

        self._mode_score += self.reward
        return self.reward

    def inputDimensions(self):
        if (self._higher_dim_obs == True):
            return [(1, (self._height + 2) * 3, (self._width + 2) * 3)]
        else:
            return [(1, self._height, self._width)]

    def observationType(self, subject):
        return np.float32

    def nActions(self):
        return 2

    def observe(self):
        obs = self.get_observation(self.y, self._x_block, self.x)
        return [obs]

    def get_observation(self, y, x_block, x):
        obs = np.zeros((self._height, self._width))
        obs[y, x_block] = 0.5
        obs[0, x - self._width_paddle + 1:x + 1] = 1

        if (self._higher_dim_obs == True):
            y_t = (1 + y) * 3
            x_block_t = (1 + x_block) * 3
            x_t = (1 + x) * 3
            obs = np.zeros(((self._height + 2) * 3, (self._width + 2) * 3))
            ball = np.array([[0, 0, 0.6, 0.8, 0.6, 0, 0],
                             [0., 0.6, 0.9, 1, 0.9, 0.6, 0],
                             [0., 0.85, 1, 1, 1, 0.85, 0.],
                             [0, 0.6, 0.9, 1, 0.9, 0.6, 0],
                             [0, 0, 0.6, 0.85, 0.6, 0, 0]])
            paddle = np.array([[0.5, 0.95, 1, 1, 1, 0.95, 0.5],
                               [0.9, 1, 1, 1, 1, 1, 0.9],
                               [0., 0., 0, 0, 0, 0., 0.]])

            obs[y_t - 2:y_t + 3, x_block_t - 3:x_block_t + 4] = ball
            obs[3:6, x_t - 3:x_t + 4] = paddle

        if (self._reverse == True):
            obs = -obs
            # plt.imshow(np.flip(obs,axis=0), cmap='gray_r')
            # plt.show()

        return obs

    def inTerminalState(self):
        if (self.y == 0):
            return True
        else:
            return False
