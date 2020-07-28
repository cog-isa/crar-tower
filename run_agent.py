import sys

import ray
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray import tune

from model.policy import CrarPolicy
from env.env_wrapper import make_env
from utils.tf_utils import tf_init_gpus

from legacy import params
from legacy.utils import default_parser

tf_init_gpus()


def training_workflow(config, reporter):

    parsed_params = default_parser.process_args(sys.argv[1:], params.Defaults)

    # setup policy and evaluation actors
    env = make_env()
    policy = CrarPolicy(env.observation_space, env.action_space, config=vars(parsed_params))
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
