from warnings import warn

import numpy as np
from ray.rllib.policy.policy import Policy

from envs.env_wrapper import ObstacleTowerEnvStub
from legacy import params
from legacy.model import algo_original
from legacy.epsilon_greedy_policy import EpsilonGreedyPolicy
from legacy.replay.dataset import DataSet
from legacy.utils.exceptions import AgentError, AgentWarning, SliceError


class CrarPolicy(Policy):

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        # env_stub = ObstacleTowerEnvStub()
        env_stub = config['env']

        learning_algo = algo_original.CRAR(
                env_stub,
                config['rms_decay'],
                config['rms_epsilon'],
                config['momentum'],
                config['clip_norm'],
                config['freeze_interval'],
                config['batch_size'],
                config['update_rule'],
                config['rng'],
                double_Q=True,
                high_int_dim=params.HIGH_INT_DIM,
                internal_dim=3)

        self._legacy_init(config, env_stub, config['rng'], learning_algo)
        self._action_counter = 0

    def _legacy_init(self, config, env_stub, rng, learning_algo):
        """
        NeuralAgent's __init__
        """
        replay_memory_size = config['replay_memory_size']
        replay_start_size = max(env_stub.inputDimensions()[i][0] for i in range(len(env_stub.inputDimensions())))
        # replay_start_size = 100
        batch_size = config['batch_size']
        random_state = rng
        exp_priority = 0
        train_policy = None
        test_policy = EpsilonGreedyPolicy(learning_algo, env_stub.nActions(), rng, 0.1)
        only_full_history = True

        inputDims = env_stub.inputDimensions()
        if replay_start_size < max(inputDims[i][0] for i in range(len(inputDims))):
            raise AgentError("Replay_start_size should be greater than the biggest history of a state.")

        self._controllers = []
        self._environment = env_stub
        self._learning_algo = learning_algo
        self._replay_memory_size = replay_memory_size
        self._replay_start_size = replay_start_size
        self._batch_size = batch_size
        self._random_state = random_state
        self._exp_priority = exp_priority
        self._only_full_history = only_full_history
        self._dataset = DataSet(env_stub, max_size=replay_memory_size, random_state=random_state,
                                use_priority=self._exp_priority, only_full_history=self._only_full_history)
        self._tmp_dataset = None  # Will be created by startTesting() when necessary
        self._mode = -1
        self._mode_epochs_length = 0
        self._total_mode_reward = 0
        self._curr_ep_reward = 0
        self._in_episode = False
        self._selected_action = -1
        self._state = []
        for i in range(len(inputDims)):
            self._state.append(np.zeros(inputDims[i], dtype=float))
        if (train_policy == None):
            self._train_policy = EpsilonGreedyPolicy(learning_algo, env_stub.nActions(), random_state, 0.1)
        else:
            self._train_policy = train_policy
        if (test_policy == None):
            self._test_policy = EpsilonGreedyPolicy(learning_algo, env_stub.nActions(), random_state, 0.)
        else:
            self._test_policy = test_policy
        self.gathering_data = True  # Whether the agent is gathering data or not
        self.sticky_action = 1  # Number of times the agent is forced to take the same action as part of one actual time step

    def compute_actions(self,
                        obs_batch,
                        state_batches=None,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        explore=None,
                        timestep=None,
                        **kwargs):

        # print(f'compute_actions() with obs_batch len {len(obs_batch)}, id {self._action_counter}')
        # self._action_counter += 1

        def legacy_choose_action(state):
            if self._mode != -1:
                # Act according to the test policy if not in training mode
                action, V = self._test_policy.action(state, mode=self._mode, dataset=self._dataset)
            else:
                if self._dataset.n_elems > self._replay_start_size:
                    # follow the train policy
                    action, V = self._train_policy.action(state, mode=None,
                                                          dataset=self._dataset)
                else:
                    # Still gathering initial data: choose dummy action
                    action, V = self._train_policy.randomAction()
            return action

        actions = np.array([legacy_choose_action([obs])
                            for obs in obs_batch])

        #actions = np.array([self.action_space.sample()
        #                    for _ in obs_batch])

        return actions, [], {}

    def learn_on_batch(self, samples):
        # print('learn_on_batch()')
        # add samples to replay, one by one...
        obs = samples[samples.CUR_OBS]
        actions = samples[samples.ACTIONS]
        rewards = samples[samples.REWARDS]
        next_obs = samples[samples.NEXT_OBS]
        dones = samples[samples.DONES]
        for s, a, r, s_next, done in zip(obs, actions, rewards, next_obs, dones):
            self._dataset.addSample(s, a, r, done, priority=1)

        loss = None
        # sample a random batch of data and perform a Q-learning iteration
        # make sure replay has enough samples
        if self._dataset.n_elems > self._replay_start_size:
            try:
                if hasattr(self._learning_algo, 'nstep'):
                    observations, actions, rewards, terminals, rndValidIndices = self._dataset.randomBatch_nstep(
                        self._batch_size, self._learning_algo.nstep, self._exp_priority)
                    loss, loss_ind = self._learning_algo.train(observations, actions, rewards, terminals)
                else:
                    states, actions, rewards, next_states, terminals, rndValidIndices = self._dataset.randomBatch(
                        self._batch_size, self._exp_priority)
                    loss, loss_ind = self._learning_algo.train(states, actions, rewards, next_states, terminals)

                if self._exp_priority:
                    self._dataset.updatePriorities(pow(loss_ind, self._exp_priority) + 0.0001, rndValidIndices[1])

            except SliceError as e:
                warn("Training not done - " + str(e), AgentWarning)

        return {'q-loss': loss}

    def get_weights(self):
        weights = self._learning_algo.getAllParams()
        return weights

    def set_weights(self, weights):
        self._learning_algo.setAllParams(weights)


