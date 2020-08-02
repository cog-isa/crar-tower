from keras import Input, Model
from keras import layers
import numpy as np


class RandomNetworkDistillator:
    def __init__(self, obs_space):
        self.obs_shape = obs_space.shape

        self.predictor_network = self._make_fc_network()
        self.target_network = self._make_fc_network()

        self.ext_reward_stats = RunningStats(shape=(1,))
        self.int_reward_stats = RunningStats(shape=(1,))
        self.obs_stats = RunningStats(shape=(100,))

    def _make_fc_network(self):
        input_item = Input(shape=self.obs_shape)
        h = layers.Dense(200, activation='relu')(input_item)
        out = layers.Dense(100, activation='relu')(h)
        model = Model(inputs=input_item, outputs=out)
        model.compile(loss='mse')
        return model

    def get_intrinsic_reward(self, obs):
        a = self.predictor_network.predict(obs)
        b = self.target_network.predict(obs)
        mse = np.mean((a - b) ** 2, axis=1)
        return mse

    def fit(self, obs):
        target = self.target_network.predict(obs)
        self.predictor_network.fit(x=obs, y=target, epochs=10)


class RunningStats:
    # This class which computes global stats is adapted & modified from:
    # https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean

        new_mean = self.mean + delta * batch_count / (self.count + batch_count)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        self.count = batch_count + self.count

    def normalize(self, buff, clip = 1, clip_state = False):
        self.update(np.array(buff))
        buff = (np.array(buff) - self.mean) / self.std
        if clip_state:
            buff = np.clip(buff, -clip, clip)
        return buff
