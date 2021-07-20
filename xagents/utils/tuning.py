from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import optuna
import tensorflow as tf

from xagents import PPO
from xagents.utils.cli import (agent_args, non_agent_args, off_policy_args,
                               train_args)
from xagents.utils.common import create_envs


class Objective:
    def __init__(
        self, agent_id, agent_known_args, non_agent_known_args, command_known_args
    ):
        self.agent_id = agent_id
        self.agent_args = agent_known_args
        self.non_agent_args = non_agent_known_args
        self.command_args = command_known_args

    def __call__(self, trial):
        return 1


# def optimize_agent(trial):
#     hparams = get_hparams(trial)
#     envs = create_envs('BreakoutNoFrameskip-v4', hparams['n_envs'])
#     optimizer = Adam(
#         hparams['learning_rate'],
#         epsilon=hparams['epsilon'],
#     )
#     model = create_model(envs[0].observation_space.shape, envs[0].action_space.n)
#     model.compile(optimizer)
#     agent = PPO(
#         envs,
#         model,
#         entropy_coef=hparams['entropy_coef'],
#         grad_norm=hparams['grad_norm'],
#         n_steps=hparams['n_steps'],
#         lam=hparams['lam'],
#         clip_norm=hparams['clip_norm'],
#         trial=trial,
#         log_frequency=1,
#         quiet=True,
#     )
#     steps = 250000
#     agent.fit(max_steps=steps)
#     current_rewards = np.around(np.mean(agent.total_rewards), 2)
#     if not np.isfinite(current_rewards):
#         current_rewards = 0
#     return current_rewards


def run_trial(
    agent_id,
    agent_known_args,
    non_agent_known_args,
    command_known_args,
    study,
    storage,
    direction='maximize',
    load_if_exists=True,
):
    optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    tf.get_logger().setLevel('CRITICAL')
    study = optuna.create_study(
        study_name=study,
        storage=storage,
        load_if_exists=load_if_exists,
        direction=direction,
    )
    objective = Objective(
        agent_id, agent_known_args, non_agent_known_args, command_known_args
    )
    study.optimize(objective, n_trials=1)
    # frame = study.trials_dataframe()
    # print(frame.loc[frame.shape[0] - 1])


def run_tuning(agent_id, non_agent_known_args, agent_known_args, command_known_args):
    trial_kwargs = {
        'agent_id': agent_id,
        'agent_known_args': agent_known_args,
        'non_agent_known_args': non_agent_known_args,
        'command_known_args': command_known_args,
        'study': command_known_args.study,
        'storage': command_known_args.storage,
    }
    with ProcessPoolExecutor(command_known_args.n_jobs) as executor:
        future_trials = [
            executor.submit(run_trial, **trial_kwargs)
            for _ in range(command_known_args.n_trials)
        ]
        for future_trial in as_completed(future_trials):
            future_trial.result()
