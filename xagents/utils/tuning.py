from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import optuna
import tensorflow as tf

import xagents
from xagents.utils.cli import agent_args, non_agent_args, off_policy_args
from xagents.utils.common import create_agent


class Objective:
    def __init__(
        self, agent_id, agent_known_args, non_agent_known_args, command_known_args
    ):
        self.agent_id = agent_id
        self.agent_args = agent_known_args
        self.non_agent_args = non_agent_known_args
        self.command_args = command_known_args
        self.arg_groups = [
            (
                vars(agent_known_args),
                {**agent_args, **xagents.agents[agent_id]['module'].cli_args},
                self.agent_args,
            ),
            (
                vars(non_agent_known_args),
                {**non_agent_args, **off_policy_args},
                self.non_agent_args,
            ),
        ]

    def set_trial_values(self, trial):
        for parsed_args, default_args, namespace in self.arg_groups:
            for arg, possible_values in parsed_args.items():
                hp_type = default_args[arg.replace('_', '-')].get('hp_type')
                trial_value = possible_values
                if isinstance(possible_values, list):
                    if hp_type and len(possible_values) == 1:
                        trial_value = possible_values[0]
                    elif hp_type == 'categorical':
                        trial_value = trial.suggest_categorical(arg, possible_values)
                    elif hp_type == 'log_uniform':
                        trial_value = trial.suggest_loguniform(arg, *possible_values)
                    elif hp_type == 'int':
                        trial_value = trial.suggest_int(arg, *possible_values)
                setattr(namespace, arg, trial_value)

    def __call__(self, trial):
        self.set_trial_values(trial)
        agent = create_agent(self.agent_id, vars(self.agent_args), self.non_agent_args)
        agent.fit(max_steps=self.command_args.trial_steps)
        trial_reward = np.around(np.mean(agent.total_rewards or [0]), 2)
        return trial_reward


def run_trial(
    agent_id,
    agent_known_args,
    non_agent_known_args,
    command_known_args,
):
    if not command_known_args.non_silent:
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        tf.get_logger().setLevel('ERROR')
        agent_known_args.quiet = True
    pruner = optuna.pruners.MedianPruner(command_known_args.warmup_trials)
    study = optuna.create_study(
        study_name=command_known_args.study,
        storage=command_known_args.storage,
        load_if_exists=True,
        direction='maximize',
        pruner=pruner,
    )
    objective = Objective(
        agent_id, agent_known_args, non_agent_known_args, command_known_args
    )
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study.optimize(objective, n_trials=1)


def run_tuning(agent_id, agent_known_args, non_agent_known_args, command_known_args):
    trial_kwargs = {
        'agent_id': agent_id,
        'agent_known_args': agent_known_args,
        'non_agent_known_args': non_agent_known_args,
        'command_known_args': command_known_args,
    }
    with ProcessPoolExecutor(command_known_args.n_jobs) as executor:
        future_trials = [
            executor.submit(run_trial, **trial_kwargs)
            for _ in range(command_known_args.n_trials)
        ]
        for future_trial in as_completed(future_trials):
            future_trial.result()
