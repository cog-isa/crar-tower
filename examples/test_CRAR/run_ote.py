import sys
import os
import logging

import numpy as np
from joblib import hash, dump
import tensorflow as tf
import wandb

from deer.default_parser import process_args
from deer.agent import NeuralAgent
from deer.learning_algos.CRAR_keras import CRAR
import deer.experiment.base_controllers as bc
from deer.policies import EpsilonGreedyPolicy
from examples.test_CRAR.obstacle_tower_env_wrapper import ObstacleTowerEnvWrapper

wandb.init(project='crar-tower')

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)


class Defaults:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    STEPS_PER_EPOCH = 2000
    EPOCHS = 50
    STEPS_PER_TEST = 500
    PERIOD_BTW_SUMMARY_PERFS = 1

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 2

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.0005
    LEARNING_RATE_DECAY = 0.9
    DISCOUNT = 0.9
    DISCOUNT_INC = 1
    DISCOUNT_MAX = 0.99
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0
    CLIP_NORM = 1.0
    EPSILON_START = 1.0
    EPSILON_MIN = 1.0
    EPSILON_DECAY = 10000
    UPDATE_FREQUENCY = 1
    REPLAY_MEMORY_SIZE = 1000000
    BATCH_SIZE = 32
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = False


HIGHER_DIM_OBS = True
HIGH_INT_DIM = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # --- Parse parameters ---
    parameters = process_args(sys.argv[1:], Defaults)
    if parameters.deterministic:
        rng = np.random.RandomState(123456)
    else:
        rng = np.random.RandomState()

    # --- Instantiate environment ---
    env = ObstacleTowerEnvWrapper()

    # --- Instantiate learning algorithm ---
    learning_algo = CRAR(
        env,
        parameters.rms_decay,
        parameters.rms_epsilon,
        parameters.momentum,
        parameters.clip_norm,
        parameters.freeze_interval,
        parameters.batch_size,
        parameters.update_rule,
        rng,
        double_Q=True,
        high_int_dim=HIGH_INT_DIM,
        internal_dim=3)

    test_policy = EpsilonGreedyPolicy(learning_algo, env.nActions(), rng, 0.1)  # 1.)

    # --- Instantiate agent ---
    agent = NeuralAgent(
        env,
        learning_algo,
        parameters.replay_memory_size,
        max(env.inputDimensions()[i][0] for i in range(len(env.inputDimensions()))),
        parameters.batch_size,
        rng,
        test_policy=test_policy)

    # --- Create unique filename for FindBestController ---
    h = hash(vars(parameters), hash_name="sha1")
    fname = "test_" + h
    print("The parameters hash is: {}".format(h))
    print("The parameters are: {}".format(parameters))

    # --- Bind controllers to the agent ---
    # Before every training epoch (periodicity=1), we want to print a summary of the agent's epsilon, discount and
    # learning rate as well as the training epoch number.
    agent.attach(bc.VerboseController(
        evaluate_on='epoch',
        periodicity=1))

    # As for the discount factor and the learning rate, one can update periodically the parameter of the epsilon-greedy
    # policy implemented by the agent. This controllers has a bit more capabilities, as it allows one to choose more
    # precisely when to update epsilon: after every X action, episode or epoch. This parameter can also be reset every
    # episode or epoch (or never, hence the resetEvery='none').
    agent.attach(bc.EpsilonController(
        initial_e=parameters.epsilon_start,
        e_decays=parameters.epsilon_decay,
        e_min=parameters.epsilon_min,
        evaluate_on='action',
        periodicity=1,
        reset_every='none'))

    # --- Run the experiment ---
    try:
        os.mkdir("params")
    except Exception:
        pass
    dump(vars(parameters), "params/" + fname + ".jldump")

    # agent.run(n_epochs=1, epoch_length=20000) #For collecting data off-policy
    # print "end gathering data"

    # During training epochs, we want to train the agent after every [parameters.update_frequency] action it takes.
    # Plus, we also want to display after each training episode (!= than after every training) the average bellman
    # residual and the average of the V values obtained during the last episode, hence the two last arguments.
    agent.attach(bc.TrainerController(
        evaluate_on='action',
        periodicity=parameters.update_frequency,
        show_episode_avg_V_value=True,
        show_avg_Bellman_residual=True))

    # Every epoch end, one has the possibility to modify the learning rate using a LearningRateController. Here we
    # wish to update the learning rate after every training epoch (periodicity=1), according to the parameters given.
    agent.attach(bc.LearningRateController(
        initial_learning_rate=parameters.learning_rate,
        learning_rate_decay=parameters.learning_rate_decay,
        periodicity=1))

    # Same for the discount factor.
    agent.attach(bc.DiscountFactorController(
        initial_discount_factor=parameters.discount,
        discount_factor_growth=parameters.discount_inc,
        discount_factor_max=parameters.discount_max,
        periodicity=1))

    # All previous controllers control the agent during the epochs it goes through. However, we want to interleave a
    # "validation epoch" between each training epoch ("one of two epochs", hence the periodicity=2). We do not want
    # these validation epoch to interfere with the training of the agent, which is well established by the
    # TrainerController, EpsilonController and alike. Therefore, we will disable these controllers for the whole
    # duration of the validation epochs interleaved this way, using the controllersToDisable argument of the
    # InterleavedTestEpochController. For each validation epoch, we want also to display the sum of all rewards
    # obtained, hence the showScore=True. Finally, we want to call the summarizePerformance method of ALE_env every
    # [parameters.period_btw_summary_perfs] *validation* epochs.
    agent.attach(bc.InterleavedTestEpochController(
        id=env.VALIDATION_MODE,
        epoch_length=parameters.steps_per_test,
        controllers_to_disable=[0, 1, 2, 3, 4],
        periodicity=2,
        show_score=True,
        summarize_every=1))

    # agent.gathering_data=False
    agent.run(parameters.epochs, parameters.steps_per_epoch)
