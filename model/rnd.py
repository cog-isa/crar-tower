from keras import Input, Model
from keras import layers, optimizers, initializers
import numpy as np


class RandomNetworkDistillator:
    def __init__(self, obs_space):
        self.obs_shape = obs_space.shape

        self._predictor_network = self._make_fc_network()
        self._target_network = self._make_fc_network()

        self.int_reward_stats = RunningStats(shape=(1,), is_running=False)
        self.obs_stats = RunningStats(shape=self.obs_shape, is_running=False)

    def distill(self, obs):
        # normalize obs
        normed_obs = self.obs_stats.normalize(obs)

        # compute intrinsic rewards
        target = self._target_network.predict(normed_obs)
        pred = self._predictor_network.predict(normed_obs)

        intrinsic_rewards = ((target - pred) ** 2).sum(axis=1)
        normed_int_rewards = self.int_reward_stats.normalize(intrinsic_rewards, clip_state=True, clip=0.3)

        # fit predictor
        loss = self._predictor_network.fit(x=normed_obs, y=target, epochs=10, verbose=0)

        return normed_int_rewards, loss

    def _make_fc_network(self):
        initializer = initializers.he_normal()

        input_item = Input(shape=(1,)+self.obs_shape)
        h = layers.Flatten()(input_item)
        h = layers.Dense(200, activation='relu', kernel_initializer=initializer)(h)
        out = layers.Dense(100, kernel_initializer=initializer)(h)
        model = Model(inputs=input_item, outputs=out)
        model.compile(loss='mse', optimizer=optimizers.RMSprop())
        return model


class RunningStats:
    # This class which computes global stats is adapted & modified from:
    # https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), is_running=True):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.std = np.ones(shape, 'float64')
        self.count = epsilon
        self.is_running = is_running

    def _update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_size = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_size)

    def _update_from_moments(self, batch_mean, batch_var, batch_size):
        if self.is_running:
            delta = batch_mean - self.mean
            new_mean = self.mean + delta * batch_size / (self.count + batch_size)
            m_a = self.var * self.count
            m_b = batch_var * batch_size
            M2 = m_a + m_b + np.square(delta) * self.count * batch_size / (self.count + batch_size)
            new_var = M2 / (self.count + batch_size)
        else:
            new_mean = batch_mean
            new_var = batch_var

        self.mean = new_mean
        self.var = new_var
        self.std = np.maximum(np.sqrt(self.var), 1e-6)
        self.count = batch_size + self.count

    def normalize(self, buff, clip=1.0, clip_state=False):
        self._update(np.array(buff))
        buff = (np.array(buff) - self.mean) / self.std
        if clip_state:
            buff = np.clip(buff, -clip, clip)
        return buff
