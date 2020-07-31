import sys
import os

import numpy as np
import gym
import ray
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray import tune
from wandb.ray import WandbLogger

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

    worker = RolloutWorker(
        lambda c: CatcherEnv(rng, higher_dim_obs=params.HIGHER_DIM_OBS, reverse=False),
        CrarPolicy,
        policy_config=custom_config)

    # policy = worker.get_policy()

    for _ in range(config['num_iters']):
        # broadcast weights to evaluation worker
        # weights = ray.put({'default_policy': policy.get_weights()})
        # worker.set_weights.remote(weights)

        # gather batch of samples
        samples = worker.sample()

        # improve policy
        # policy.learn_on_batch(samples)
        info = worker.learn_on_batch(samples)

        reporter(**collect_metrics(local_worker=worker), **info['default_policy'])


if __name__ == '__main__':
    ray.init(local_mode=True)

    config = {
        'num_workers': 0,
        'num_iters': 10000,
        'env_config': {
            'wandb': {
                'project': 'crar-tower'
            }
        }
    }

    tune.run(training_workflow,
             resources_per_trial={
                 "gpu": 2,
                 "cpu": os.cpu_count()
             },
             config=config,
             loggers=[WandbLogger])
