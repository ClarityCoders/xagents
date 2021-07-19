import numpy as np
import optuna
from tensorflow.keras.optimizers import Adam

from xagents import PPO
from xagents.utils.common import create_envs


def get_hparams(trial):
    return {
        'n_steps': int(
            trial.suggest_categorical('n_steps', [2 ** i for i in range(2, 11)])
        ),
        'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-2),
        'epsilon': trial.suggest_loguniform('epsilon', 1e-7, 1e-1),
        'entropy_coef': trial.suggest_loguniform('entropy_coef', 1e-8, 2e-1),
        'n_envs': int(
            trial.suggest_categorical('n_envs', [2 ** i for i in range(4, 7)])
        ),
        'grad_norm': trial.suggest_uniform('grad_norm', 0.1, 10.0),
        'lam': trial.suggest_loguniform('lam', 0.65, 0.99),
        'clip_norm': trial.suggest_loguniform('clip_norm', 0.01, 10),
    }


def optimize_agent(trial):
    hparams = get_hparams(trial)
    envs = create_envs('BreakoutNoFrameskip-v4', hparams['n_envs'])
    optimizer = Adam(
        hparams['learning_rate'],
        epsilon=hparams['epsilon'],
    )
    model = create_model(envs[0].observation_space.shape, envs[0].action_space.n)
    model.compile(optimizer)
    agent = PPO(
        envs,
        model,
        entropy_coef=hparams['entropy_coef'],
        grad_norm=hparams['grad_norm'],
        n_steps=hparams['n_steps'],
        lam=hparams['lam'],
        clip_norm=hparams['clip_norm'],
        trial=trial,
        log_frequency=1,
    )
    steps = 250000
    agent.fit(max_steps=steps)
    current_rewards = np.around(np.mean(agent.total_rewards), 2)
    if not np.isfinite(current_rewards):
        current_rewards = 0
    return current_rewards


if __name__ == '__main__':
    pruner = optuna.pruners.PercentilePruner(10)
    study = optuna.create_study(
        study_name='ppo-trials-with-pruning',
        load_if_exists=True,
        storage="sqlite:///ppo.db",
        direction='maximize',
    )
    study.optimize(optimize_agent, n_trials=1)
    frame = study.trials_dataframe()
    print(frame.loc[frame.shape[0] - 1])
