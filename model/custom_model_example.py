import ray
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray import tune


def make_env():
    pass


class CrarPolicy(Policy):

    def __init__(self, observation_space, action_space, config):
        super().__init__(observation_space, action_space, config)

        # instantiate stuff

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
        # run planning
        pass

    def learn_on_batch(self, samples):
        # learning code
        pass

    def get_weights(self):
        return 0

    def set_weights(self, weights):
        pass


def training_workflow(config, reporter):
    # setup policy and evaluation actors

    env = make_env()
    policy = CrarPolicy(env.observation_space, env.aciton_space, {})
    worker = RolloutWorker.as_remote().remote(lambda c: make_env(), CrarPolicy)

    for _ in range(config['num_iters']):
        # broadcast weights to evaluation worker
        weights = ray.put({'default_policy': policy.get_weights()})
        worker.set_weights.remote(weights)

        # gather batch of samples
        samples = SampleBatch.concat_samples(
                    ray.get([worker.sample.remote()]))

        # improve policy
        policy.learn_on_batch(samples)

        reporter(**collect_metrics(remote_workers=[worker]))


if __name__ == '__main__':
    ray.init()

    config = {
        'num_workers': 1,
        'num_iters': 1000
    }

    tune.run(training_workflow,
             config=config)
