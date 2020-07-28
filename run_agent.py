import sys

import numpy as np
import gym
import ray
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray import tune

from model.policy import CrarPolicy
from envs.env_wrapper import make_obstacle_tower_env
from envs.env_wrapper import CatcherEnv
from utils.tf_utils import tf_init_gpus

from legacy import params
from legacy.utils import default_parser

tf_init_gpus()


def training_workflow(config, reporter):

    parsed_params = default_parser.process_args(sys.argv[1:], params.Defaults)

    rng = np.random.RandomState(123456)

    # setup policy and evaluation actors
    # env = make_obstacle_tower_env()
    env = CatcherEnv(rng, higher_dim_obs=params.HIGHER_DIM_OBS, reverse=False)

    custom_config = vars(parsed_params)
    custom_config['env'] = env
    custom_config['rng'] = rng

    policy = CrarPolicy(env.observation_space, env.action_space, config=custom_config)
    worker = RolloutWorker.as_remote().remote(
        lambda c: CatcherEnv(rng, higher_dim_obs=params.HIGHER_DIM_OBS, reverse=False),
        CrarPolicy,
        policy_config=custom_config)

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
    ray.init(local_mode=True)

    config = {
        'num_workers': 1,
        'num_iters': 1000
    }

    tune.run(training_workflow,
             config=config)
