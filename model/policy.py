import numpy as np
from ray.rllib.policy.policy import Policy
import wandb

from legacy import params
from legacy.model import algo_original
from legacy.epsilon_greedy_policy import EpsilonGreedyPolicy
from legacy.replay.dataset import DataSet
from legacy.utils.exceptions import AgentError, AgentWarning, SliceError
from legacy.model.controllers import LearningRateScheduler
from .rnd import RandomNetworkDistillator


class CrarPolicy(Policy):

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

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

        self._lr_scheduler = LearningRateScheduler(config['learning_rate'],
                                                   config['learning_rate_decay'],
                                                   decay_every=2000,
                                                   learning_algo=learning_algo)

        self._rnd = RandomNetworkDistillator(observation_space)
        self._step_idx = 0
        #wandb.init(project='crar-tower',
        #           group=config['experiment_id'])

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

        def legacy_choose_action(state):
            if self._dataset.n_elems > self._replay_start_size:
                # follow the train policy
                state[0] = state[0] / 255
                action, V = self._train_policy.action(state, mode=None,
                                                      dataset=self._dataset)
            else:
                # Still gathering initial data: choose dummy action
                action, V = self._train_policy.randomAction()
            return action

        actions = np.array([legacy_choose_action([obs[None]])
                            for obs in obs_batch])

        return actions, [], {}

    def learn_on_batch(self, samples):
        # add samples to replay, one by one...
        obs = samples[samples.CUR_OBS]
        actions = samples[samples.ACTIONS]
        rewards = samples[samples.REWARDS]
        next_obs = samples[samples.NEXT_OBS]
        dones = samples[samples.DONES]
        infos = samples[samples.INFOS]

        for s, a, r, s_next, done in zip(obs, actions, rewards, next_obs, dones):
            # hack from original framework
            s = [s]
            self._dataset.addSample(s, a, r, done, priority=1)

        q_loss = None
        transition_loss = None
        reward_loss = None
        gamma_loss = None
        report = {}

        intrinsic_rewards_total = []
        distill_loss = None
        if self._dataset.n_elems > self._replay_start_size:
            n_samples = len(obs)

            for _ in range(n_samples):
                states, actions, rewards, next_states, terminals, rndValidIndices = self._dataset.randomBatch(
                    self._batch_size, self._exp_priority)

                states[0] = states[0] / 255.

                # rnd
                if self.config['use_rnd']:
                    obs_batch = next_states[0]
                    intrinsic_rewards, distill_loss = self._rnd.distill(obs_batch)
                    rewards += intrinsic_rewards

                    intrinsic_rewards_total.extend(intrinsic_rewards)

                # train
                q_loss, loss_ind, transition_loss, reward_loss, gamma_loss = \
                    self._learning_algo.train(states, actions, rewards, next_states, terminals)

                if self._exp_priority:
                    self._dataset.updatePriorities(pow(loss_ind, self._exp_priority) + 0.0001, rndValidIndices[1])

            self._lr_scheduler.new_samples_seen(n_samples)

            # report rnd stats
            report['intrinsic_reward_max'] = max(intrinsic_rewards_total)
            report['intrinsic_reward_min'] = min(intrinsic_rewards_total)
            report['intrinsic_reward_mean'] = sum(intrinsic_rewards_total) / len(intrinsic_rewards_total)
            report['distill_loss'] = distill_loss.history['loss'][-1]

        # tmp stub for computing metric
        if 'current_floor' in infos[0]:
            max_floor = max(info["current_floor"] for info in infos)
            report['max_floor'] = max_floor

        out = {**report,
               'q_loss': q_loss,
               'transition_loss': transition_loss,
               'reward_loss': reward_loss,
               'gamma_loss': gamma_loss,
               'lr': self._lr_scheduler.lr,
               'learn_on_batch_step': self._step_idx}
        self._step_idx += 1
        return out

    def get_weights(self):
        weights = self._learning_algo.getAllParams()
        return weights

    def set_weights(self, weights):
        self._learning_algo.setAllParams(weights)


