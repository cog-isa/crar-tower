import sys
import os

import numpy as np
import ray
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.metrics import collect_metrics
from ray import tune
import wandb
from wandb.ray import WandbLogger

from model.policy import CrarPolicy
from envs.env_wrapper import make_obstacle_tower_env, ObstacleTowerEnvStub, DebugObstacleTowerEnv
from envs.env_wrapper import CatcherEnv
from utils.tf_utils import tf_init_gpus

from legacy import params
from legacy.utils import default_parser


DEBUG = os.environ.get('DEBUG', False)
ENV = 'obstacle-tower'
USE_RND = False
USE_NSP_MOTIVATION = True


def training_workflow(config, reporter):
    tf_init_gpus()

    parsed_params = config['parsed_params']

    rng = np.random.RandomState(123456)

    if not DEBUG:
        if ENV == 'catcher':
            env_creator = lambda ctx: CatcherEnv(rng, higher_dim_obs=params.HIGHER_DIM_OBS, reverse=False)
            env_stub = env_creator(None)
        elif ENV == 'obstacle-tower':
            env_creator = make_obstacle_tower_env
            env_stub = ObstacleTowerEnvStub()
        else:
            assert False
    else:
        env_creator = lambda ctx: DebugObstacleTowerEnv()
        env_stub = ObstacleTowerEnvStub()

    custom_config = vars(parsed_params)
    custom_config['env'] = env_stub
    custom_config['rng'] = rng
    custom_config['use_rnd'] = config['use_rnd']
    custom_config['use_nsp_motivation'] = config['use_nsp_motivation']
    custom_config['experiment_id'] = config['experiment_id']

    n_steps = 100 if not DEBUG else 2
    worker = RolloutWorker(
        env_creator,
        CrarPolicy,
        policy_config=custom_config,
        rollout_fragment_length=n_steps)

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
    ray.init(local_mode=DEBUG,
             webui_host='127.0.0.1')

    parsed_params = default_parser.process_args(sys.argv[1:], params.Defaults)

    experiment_id = wandb.util.generate_id()
    config = {
        'num_workers': 0,
        'num_iters': 10000,
        'parsed_params': parsed_params,
        'use_rnd': USE_RND,
        'use_nsp_motivation': USE_NSP_MOTIVATION,
        'experiment_id': experiment_id,
        'env_config': {'wandb': {'project': 'crar-tower',
                                 'group': experiment_id}}
    }

    tune.run(training_workflow,
             resources_per_trial={
                 "gpu": 1,
                 "cpu": os.cpu_count()
             },
             config=config,
             loggers=[WandbLogger])
