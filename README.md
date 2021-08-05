[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]

<p>
  <a href="https://github.com/schissmantics/xagents/">
  </a>

  <h3 align="left">xagents - reusable, scalable, 
  performant reinforcement learning algorithms in tf2</h3>
  </p>

* [Installation](#1-installation)
* [Description](#2-description)
* [Features](#3-features)
  * [Tensorflow 2](#31-tensorflow-2x)
  * [wandb support](#32-wandb-support)
  * [Multiple environments (All agents)](#33-multiple-environments)
  * [Multiple memory-optimized replay buffers](#34-multiple-memory-optimized-replay-buffers)
  * [Command line options](#35-command-line-options)
  * [Intuitive hyperparameter tuning from cli](#36-intuitive-hyperparameter-tuning-from-cli)
  * [Early stopping / reduce on plateau](#37-early-stopping--reduce-on-plateau)
  * [Discrete and continuous action spaces](#38-discrete-and-continuous-action-spaces)
  * [Unit tests](#39-unit-tests)
  * [Models are loaded from .cfg files](#310-models-are-loaded-from-cfg-files)
  * [Training history checkpoints](#311-training-history-checkpoints)
  * [Reproducible results](#312-reproducible-results)
  * [Gameplay output to .jpg frames or .mp4 vid](#313-gameplay-output-to-jpg-frames-or-mp4-vid)
  * [Resumable training and history](#314-resume-training--history)
* [Usage](#4-usage)
  * [Training](#41-training)
  * [Playing](#42-playing)
  * [Tuning](#43-tuning)
* [Command line options](#5-command-line-options)
  * [Agent](#51-agent)
  * [General](#52-general)
  * [Training](#53-training)
  * [Playing](#54-playing)
  * [Tuning](#55-tuning)
* [Algorithms](#6-algorithms)
  * [A2C](#61-a2c)
  * [ACER](#62-acer)
  * [DDPG](#63-ddpg)
  * [DQN / DDQN](#64-dqn-ddqn)
  * [PPO](#65-ppo)
  * [TD3](#66-td3)
  * [TRPO](#67-trpo)
* [License](#7-license)
* [Show your support](#8-show-your-support)
* [Contact](#9-contact)

![breakout](/img/breakout.gif)
![pong](/img/pong.gif)
![carnival](/img/carnival.gif)
![pacman](/img/pacman.gif)
![gopher](/img/gopher.gif)

![bipedal-walker](/img/bipedal-walker.gif)
![lunar-lander](/img/lunar-lander.gif)


### **1. Installation**
___

```sh
git clone https://github.com/schissmantics/xagents
cd xagents
pip install .
```

**Notes:** 
* To be able to use atari environments, follow the instructions in [atari-py](https://github.com/openai/atari-py#roms)
to install [ROMS](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html).
  
* To be able to run the tests **remotely**, you need to run:

      pip install pytest-xvfb pyvirtualdisplay 

**Followed by:**

  * **For OSX cloud**:
    ```shell 
    brew install xquartz
    ```
  * **For linux cloud**:
    ```shell
    sudo apt-get install -y xvfb
    ```
    
**Verify installation**

```sh
xagents
```

**OUT:**

    xagents 1.0

    Usage:
        xagents <command> <agent> [options] [args]
    
    Available commands:
        train      Train given an agent and environment
        play       Play a game given a trained agent and environment
        tune       Tune hyperparameters given an agent, hyperparameter specs, and environment
    
    Use xagents <command> to see more info about a command
    Use xagents <command> <agent> to see more info about command + agent

<!-- DESCRIPTION -->
## **2. Description**
___
**xagents** is a tensorflow based mini-library which facilitates experimentation with
existing reinforcement learning algorithms, as well as the implementation of new ones. It
provides well tested components that can be easily modified or extended. The available
selection of algorithms can be used directly or through command line.

<!-- FEATURES -->
## **3. Features**
___
### **3.1. Tensorflow 2.x**

* All available agents are based on tensorflow 2.x.
* High performance training loops executed in graph mode.
* Keras models.

### **3.2. wandb support**

Visualization of the training is supported, as well as many other awesome features provided by [wandb](https://wandb.ai/site).

![wandb-agents](/img/wandb-agents.png)

### **3.3. Multiple environments**

All agents support multiple environments, which operations are conducted
in tensorflow graph. This boosts training speed without the overhead of creating
a process per environment. Atari and environments that return images, 
are wrapped in 
[LazyFrames](https://github.com/schissmantics/xagents/blob/d81e446bdd37d621fb4c3c1999a35306d70047b7/xagents/utils/common.py#L24) 
 which significantly lower memory usage.

### **3.4. Multiple memory-optimized replay buffers**

There are 2 kinds of replay buffers available:
 * [ReplayBuffer1](https://github.com/schissmantics/xagents/blob/d81e446bdd37d621fb4c3c1999a35306d70047b7/xagents/utils/buffers.py#L59) 
   which is deque-based (DQN, ACER).
 * [ReplayBuffer2](https://github.com/schissmantics/xagents/blob/d81e446bdd37d621fb4c3c1999a35306d70047b7/xagents/utils/buffers.py#L101) 
   which is numpy-based (DDPG, TD3).

Both support max size and initial size, and are usually
combined with [LazyFrames](https://github.com/schissmantics/xagents/blob/d81e446bdd37d621fb4c3c1999a35306d70047b7/xagents/utils/common.py#L24) 
for memory optimality.

### **3.5. Command line options**

All features are available through the command line. For more command line info,
check [command line options](#5-command-line-options)

### **3.6. Intuitive hyperparameter tuning from cli**

Command line tuning interface based on [optuna](https://optuna.org), which provides 
many hyperparameter features and types. 3 types are currently used by xagents:

* **Categorical**:
  
      xagents tune <agent> --env <env> --interesting-param <val1> <val2> <val3> # ...

* **Int / log uniform**:

      xagents tune <agent> --env <env> --interesting-param <min-val> <max-val>

And in both examples if `--interesting-param` is not specified, it will have the default value, 
or a fixed value, if only 1 value is specified. Also, some nice visualization options using 
[optuna.visualization.matplotlib](https://optuna.readthedocs.io/en/latest/reference/visualization/matplotlib.html):

![param-importances](/img/param-importances.png)

### **3.7. Early stopping / reduce on plateau.**

Early train stopping usually when plateau is reached for a pre-specified
n number of times without any improvement. Learning rate is
reduced by some pre-determined factor. To activate these features: 

    --divergence-monitoring-steps <train-steps-at-which-should-monitor>

### **3.8. Discrete and continuous action spaces**

|            | A2C   | ACER   | DDPG   | DQN   | PPO   | TD3   | TRPO   |
|:-----------|:------|:-------|:-------|:------|:------|:------|:-------|
| Discrete   | Yes   | Yes    | No     | Yes   | Yes   | No    | Yes    |
| Continuous | Yes   | No     | Yes    | No    | Yes   | Yes   | Yes    |

### **3.9. Unit tests**

Main components are covered using [pytest](https://docs.pytest.org/en/6.2.x/) unit tests. To be able to run the tests,
you need to install extra dependencies mentioned [earlier](#1-installation). More
unit tests will be added to agents in future versions.

### **3.10. Models are loaded from .cfg files**

To facilitate experimentation, and eliminate redundancy, all agents support
loading models by passing either `--model <model.cfg>` or `--actor-model <actor.cfg>` and 
`--critic-model <critic.cfg>`. If no models were passed, the default ones will be loaded.
A typical `model.cfg` file would look like:

    [convolutional-0]
    filters=32
    size=8
    stride=4
    activation=relu
    initializer=orthogonal
    gain=1.4142135
    
    [convolutional-1]
    filters=64
    size=4
    stride=2
    activation=relu
    initializer=orthogonal
    gain=1.4142135
    
    [convolutional-2]
    filters=64
    size=3
    stride=1
    activation=relu
    initializer=orthogonal
    gain=1.4142135
    
    [flatten-0]
    
    [dense-0]
    units=512
    activation=relu
    initializer=orthogonal
    gain=1.4142135
    common=1
    
    [dense-1]
    initializer=orthogonal
    gain=0.01
    output=1
    
    [dense-2]
    initializer=orthogonal
    gain=1.0
    output=1

Which should generate a keras model similar to this one with output units 6, and 1 respectively:

    Model: "model"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_1 (InputLayer)            [(None, 84, 84, 1)]  0                                            
    __________________________________________________________________________________________________
    conv2d (Conv2D)                 (None, 20, 20, 32)   2080        input_1[0][0]                    
    __________________________________________________________________________________________________
    conv2d_1 (Conv2D)               (None, 9, 9, 64)     32832       conv2d[0][0]                     
    __________________________________________________________________________________________________
    conv2d_2 (Conv2D)               (None, 7, 7, 64)     36928       conv2d_1[0][0]                   
    __________________________________________________________________________________________________
    flatten (Flatten)               (None, 3136)         0           conv2d_2[0][0]                   
    __________________________________________________________________________________________________
    dense (Dense)                   (None, 512)          1606144     flatten[0][0]                    
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 6)            3078        dense[0][0]                      
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 1)            513         dense[0][0]                      
    ==================================================================================================
    Total params: 1,681,575
    Trainable params: 1,681,575
    Non-trainable params: 0
    __________________________________________________________________________________________________

**Notes**

* You don't have to worry about this if you're going to use the default models,
  which are loaded automatically.
* `common=1` marks a layer to be reused by the following layers, which means
`dense-1` and `dense-2` are called on the output of `dense-0`.
* Initializer can be `orthogonal` or `glorot_uniform`, and to add more, 
you'll have to modify [xagents.utils.common.ModelReader.initializers](https://github.com/schissmantics/xagents/blob/d81e446bdd37d621fb4c3c1999a35306d70047b7/xagents/utils/common.py#L169).
* `output=1` marks a layer as output which will be appended to the outputs 
of the resulting [tf.keras.Model](https://www.tensorflow.org/api_docs/python/tf/keras/Model)
* Dense layers without units (output layers) will expect their respective units to be passed
to [xagents.utils.common.ModelReader](https://github.com/schissmantics/xagents/blob/d81e446bdd37d621fb4c3c1999a35306d70047b7/xagents/utils/common.py#L151).

### **3.11. Training history checkpoints**

Saving training history is available for further benchmarking / visualizing results.
This is achieved by specifying `--history-checkpoint <history.parquet>` which will result
in a `.parquet` that will be updated at each episode end. A sample data point will have these 
columns:

* `mean_reward` most recent mean of agent episode rewards.
* `best_reward` most recent best of agent episode rewards.
* `episode_reward` most recent episode reward.
* `step` most recent agent step.
* `time` training elapsed time.

Which enables producing plots similar to the ones below,
using [xagents.utils.common.plot_history](https://github.com/schissmantics/xagents/blob/d81e446bdd37d621fb4c3c1999a35306d70047b7/xagents/utils/common.py#L346)

![step-benchmark](/img/step-benchmark.jpg)
![time-benchmark](/img/time-benchmark.jpg)

### **3.12. Reproducible results**

All operation results are reproducible by passing `--seed <some-seed>` or `seed=some_seed` 
to agent constructor.

### **3.13. Gameplay output to .jpg frames or .mp4 vid**

Gameplay visual output can be saved to vid, by passing `--video-dir <some-dir>`
or to `.jpg` frames by passing `--frame-dir <some-dir>` to `play` command.

### **3.14. Resume training / history**

Weights are saved to `.tf` by specifying `--checkpoints <ckpt1.tf> <ckpt2.tf>`. To resume training,
`--weights <ckpt1.tf> <ckpt2.tf>` should load the weights saved earlier. If `--history-checkpoint <ckpt.parquet>`
is specified, the file is looked for and if found, further training history will be saved
to the same history `ckpt.parquet` and the agent metrics will be updated with the most
recent ones contained in the history file.

## **4. Usage**
___
All agents / commands are available through the command line.

    xagents <command> <agent> [options] [args]

**Note:** Unless called from command line with `--weights` passed,
all models passed to agents in code, should be loaded with weights 
beforehand, if called for resuming training or playing.

### **4.1. Training**

**Through command line**

    xagents train a2c --env PongNoFrameskip-v4 --n-env 16 --target-reward 19 --preprocess

**Through direct importing**

    import xagents
    from xagents import A2C
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('PongNoFrameskip-v4')
    model = ModelReader(
        xagents.agents['a2c']['model']['cnn'][0],
        output_units=[6, 1],
        input_shape=envs[0].observation_space.shape,
        optimizer='adam',
    ).build_model()
    agent = A2C(envs, model)

    
Then either `max_steps` or `target_reward` should be specified to start training:
    
    agent.fit(target_reward=19)

### **4.2. Playing**

**Through command line**

    xagents play a2c --env PongNoFrameskip-v4 --preprocess --weights <trained-a2c-weights> --render

**Through direct importing**

    import xagents
    from xagents import A2C
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('PongNoFrameskip-v4')
    model = ModelReader(
        xagents.agents['a2c']['model']['cnn'][0],
        output_units=[6, 1],
        input_shape=envs[0].observation_space.shape,
        optimizer='adam',
    ).build_model()
    model.load_weights(
        '/path/to/trained-weights.tf'
    ).expect_partial()
    agent = A2C(envs, model)
    agent.play(render=True)

**Save video**

    agent.play(video_dir='/path/to/video-dir/')

**or**

    xagents play a2c --video-dir /path/to/video-dir/  

**Save frames**

    agent.play(frame_dir='/path/to/frame-dir/')

**or** 

    xagents play a2c --frame-dir /path/to/frame-dir/

and all arguments can be combined `--video-dir <vid-dir> --frame-dir <frame-dir> --render`

### **4.3. Tuning**

**Notes**

* Due to an [issue](https://github.com/tensorflow/tensorflow/issues/50765) with tensorflow
  that causes occasional memory leaks, if trials are run consecutively using:
  
      study.optimize(objective, n_trials=100)
  
  The current implementation runs trials in separate processes that are killed after each trial,
  to release the resources. Therefore, you may find the suggested non-command line
  example different from optuna's [docs](https://optuna.readthedocs.io/en/stable/reference/generated/optuna.study.Study.html#optuna.study.Study.optimize).
* There are hyperparameters that accept min and max values, and others that support n values.
  To know which is what, check the `hp_type` in the help menu table. `categorical` takes
  any number of values, otherwise min and max.
* For more info about how the optimization algorithms work under the hood, you may want to 
check optuna [docs](https://optuna.readthedocs.io/en/stable/).
* Tuning from later stages of the training is available by passing `--weights <weights1.tf> <weights2.tf>`
  which loads agent respective model weights, and tuning starts from there.
* Only the hyperparameters selected are tuned, the rest will keep the default values
and will not be tuned or can have a single fixed value `--flag <val>`
* Also, due to tensorflow issue mentioned above, tensorflow logging is silenced
using `TF_CPP_MIN_LOG_LEVEL` environment variable to prevent each trial process 
from displaying the same import log messages over and over.

**Through command line**

    !TF_CPP_MIN_LOG_LEVEL=3 xagents tune ppo --env PongNoFrameskip-v4 --study ppo-carnival --storage sqlite:///ppo-carnival.db --trial-steps 500000 --n-trials 100 --warmup-trials 3 --preprocess --n-envs 16 32 --lr 1e-5 1e-2 --opt-epsilon 1e-7 1e-4 --gamma 0.9 0.999 --entropy-coef 0.01 0.2 --n-steps 16 32 64 128 --lam 0.7 0.99

**Through direct importing**

    import os
    
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    from concurrent.futures import ProcessPoolExecutor, as_completed
    
    import numpy as np
    import optuna
    import tensorflow as tf
    from tensorflow.keras.optimizers import Adam
    
    import xagents
    from xagents import PPO
    from xagents.utils.common import ModelReader, create_envs
    
    
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
        model_cfg = xagents.agents['ppo']['model']['cnn'][0]
        model = ModelReader(
            model_cfg,
            output_units=[envs[0].action_space.n, 1],
            input_shape=envs[0].observation_space.shape,
            optimizer=optimizer,
        ).build_model()
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
            quiet=True,
        )
        steps = 500000
        agent.fit(max_steps=steps)
        current_rewards = np.around(np.mean(agent.total_rewards), 2)
        if not np.isfinite(current_rewards):
            current_rewards = 0
        return current_rewards
    
    
    def run_trial():
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        tf.get_logger().setLevel('ERROR')
        study = optuna.create_study(
            study_name='ppo-example',
            storage='sqlite:///ppo-example.db',
            load_if_exists=True,
            direction='maximize',
        )
        optuna.logging.set_verbosity(optuna.logging.INFO)
        study.optimize(optimize_agent, n_trials=1)
    
    
    if __name__ == '__main__':
        for _ in range(100):
            with ProcessPoolExecutor(1) as executor:
                future_trials = [executor.submit(run_trial)]
                for future_trial in as_completed(future_trials):
                    future_trial.result()


<!-- COMMAND LINE OPTIONS -->
## **5. Command line options**
___
**Note:** Not all the flags listed below are available at once, and to know which 
ones are, respective to the command you passed, you can use:

    xagents <command>

or

    xagents <command> <agent>

which should list command + agent options combined

**Flags (Available for all agents)**

### **5.1. Agent**

  | flags                         | help                                                                         | default   | hp_type     |
  |:------------------------------|:-----------------------------------------------------------------------------|:----------|:------------|
  | --checkpoints                 | Path(s) to new model(s) to which checkpoint(s) will be saved during training | -         | -           |
  | --display-precision           | Number of decimals to be displayed                                           | 2         | -           |
  | --divergence-monitoring-steps | Steps after which, plateau and early stopping are active                     | -         | -           |
  | --early-stop-patience         | Minimum plateau reduces to stop training                                     | 3         | -           |
  | --gamma                       | Discount factor                                                              | 0.99      | log_uniform |
  | --history-checkpoint          | Path to .parquet file to save training history                               | -         | -           |
  | --log-frequency               | Log progress every n games                                                   | -         | -           |
  | --plateau-reduce-factor       | Factor multiplied by current learning rate when there is a plateau           | 0.9       | -           |
  | --plateau-reduce-patience     | Minimum non-improvements to reduce lr                                        | 10        | -           |
  | --quiet                       | If specified, no messages by the agent will be displayed                     | -         | -           |
  |                               | to the console                                                               |           |             |
  | --reward-buffer-size          | Size of the total reward buffer, used for calculating                        | 100       | -           |
  |                               | mean reward value to be displayed.                                           |           |             |
  | --seed                        | Random seed                                                                  | -         | -           |

### **5.2. General**

  | flags         | help                                                              | default   | hp_type     |
  |:--------------|:------------------------------------------------------------------|:----------|:------------|
  | --beta1       | Beta1 passed to a tensorflow.keras.optimizers.Optimizer           | 0.9       | log_uniform |
  | --beta2       | Beta2 passed to a tensorflow.keras.optimizers.Optimizer           | 0.999     | log_uniform |
  | --env         | gym environment id                                                | -         | -           |
  | --lr          | Learning rate passed to a tensorflow.keras.optimizers.Optimizer   | 0.0007    | log_uniform |
  | --max-frame   | If specified, max & skip will be applied during preprocessing     | -         | categorical |
  | --n-envs      | Number of environments to create                                  | 1         | categorical |
  | --opt-epsilon | Epsilon passed to a tensorflow.keras.optimizers.Optimizer         | 1e-07     | log_uniform |
  | --preprocess  | If specified, states will be treated as atari frames              | -         | -           |
  |               | and preprocessed accordingly                                      |           |             |
  | --weights     | Path(s) to model(s) weight(s) to be loaded by agent output_models | -         | -           |

### **5.3. Training**

  | flags             | help                                                                   |
  |:------------------|:-----------------------------------------------------------------------|
  | --max-steps       | Maximum number of environment steps, when reached, training is stopped |
  | --monitor-session | Wandb session name                                                     |
  | --target-reward   | Target reward when reached, training is stopped                        |

### **5.4. Playing**

  | flags             | help                                                     | default   |
  |:------------------|:---------------------------------------------------------|:----------|
  | --action-idx      | Index of action output by agent.model                    | 0         |
  | --frame-delay     | Delay between rendered frames                            | 0         |
  | --frame-dir       | Path to directory to save game frames                    | -         |
  | --frame-frequency | If --frame-dir is specified, save frames every n frames. | 1         |
  | --render          | If specified, the gameplay will be rendered              | -         |
  | --video-dir       | Path to directory to save the resulting gameplay video   | -         |

### **5.5. Tuning**

  | flags           | help                                                            | default   |
  |:----------------|:----------------------------------------------------------------|:----------|
  | --n-jobs        | Parallel trials                                                 | 1         |
  | --n-trials      | Number of trials to run                                         | 1         |
  | --non-silent    | tensorflow, optuna and agent are silenced at trial start        | -         |
  |                 | to avoid repetitive import messages at each trial start, unless |           |
  |                 | this flag is specified                                          |           |
  | --storage       | Database url                                                    | -         |
  | --study         | Name of optuna study                                            | -         |
  | --trial-steps   | Maximum steps for a trial                                       | 500000    |
  | --warmup-trials | warmup trials before pruning starts                             | 5         |

### **5.6. Off-policy (available to off-policy agents only)**

  | flags                 | help                       | default   | hp_type     |
  |:----------------------|:---------------------------|:----------|:------------|
  | --buffer-batch-size   | Replay buffer batch size   | 32        | categorical |
  | --buffer-initial-size | Replay buffer initial size | -         | int         |
  | --buffer-max-size     | Maximum replay buffer size | 10000     | int         |

<!-- ALGORITHMS -->
## **6. Algorithms**
___
**General notes**

* All the default hyperparameters don't work for all environments.
  Which means you either need to tune them according to the given environment,
  or pass previously tuned ones, in order to get good results.
* `--model <model.cfg>` or `--actor-model <actor.cfg>` and `--critic-model <critic.cfg>` are optional 
  which means, if not specified, the default model(s) will be loaded, so you don't have to worry about it.
* You can also use external models by passing them to agent constructor. If you do, you will have to ensure
  your models outputs match what the implementation expects, or modify it accordingly.
* For atari environments / the ones that return an image by default, use the `--preprocess` flag for image preprocessing.
* For checkpoints to be saved, `--checkpoints <checkpoint1.tf> <checkpoint2.tf>` should
be specified for the model(s) to be saved. The number of passed checkpoints should match the number
  of models the agent accepts.
* For loading weights either for resuming training or for playing a game `--weights <weights1.tf> <weights2.tf>`
and same goes for the weights, they should match the number of agent models.
* For using a random seed, a `seed=some_seed` should be passed to agent constructor and ModelReader constructor if
specified from code. If from the command line, all you need is to pass `--seed <some-seed>`
* To save training history, `history_checkpoint=some_history.parquet` should be specified
to agent constructor or alternatively using `--history-checkpoint <some-history.parquet>`. 
  If the history checkpoint exists, training metrics will automatically start from where it left.
  
### *6.1. A2C*

* *Number of models:* 1
* *Action spaces:* discrete and continuous

| flags             | help                                                 | default   | hp_type     |
|:------------------|:-----------------------------------------------------|:----------|:------------|
| --entropy-coef    | Entropy coefficient for loss calculation             | 0.01      | log_uniform |
| --grad-norm       | Gradient clipping value passed to tf.clip_by_value() | 0.5       | log_uniform |
| --model           | Path to model .cfg file                              | -         | -           |
| --n-steps         | Transition steps                                     | 5         | categorical |
| --value-loss-coef | Value loss coefficient for value loss calculation    | 0.5       | log_uniform |

**Command line**

     xagents train a2c --env PongNoFrameskip-v4 --target-reward 19 --n-envs 16 --preprocess --checkpoints a2c-pong.tf

OR

    xagents train a2c --env BipedalWalker-v3 --target-reward 100 --n-envs 16 --checkpoints a2c-bipedal-walker.tf

**Non-command line**
    
    from tensorflow.keras.optimizers import Adam

    import xagents
    from xagents import A2C
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('PongNoFrameskip-v4', 16)
    model_cfg = xagents.agents['a2c']['model']['cnn'][0]
    optimizer = Adam(learning_rate=7e-4)
    model = ModelReader(
        model_cfg,
        output_units=[envs[0].action_space.n, 1],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    agent = A2C(envs, model, checkpoints=['a2c-pong.tf'])
    agent.fit(target_reward=19)

And for `BipedalWalker-v3`, the only difference is that you have to specify `preprocess=False` to `create_envs()`

### *6.2. ACER*

* *Number of models:* 1
* *Action spaces:* discrete
* Due to implementation details, buffer batch size for ACER is `n-steps`.
  Therefore `buffer-batch-size` is set to 1.

| flags             | help                                                               | default   | hp_type     |
|:------------------|:-------------------------------------------------------------------|:----------|:------------|
| --delta           | delta param used for trust region update                           | 1         | log_uniform |
| --ema-alpha       | Moving average decay passed to tf.train.ExponentialMovingAverage() | 0.99      | log_uniform |
| --entropy-coef    | Entropy coefficient for loss calculation                           | 0.01      | log_uniform |
| --epsilon         | epsilon used in gradient updates                                   | 1e-06     | log_uniform |
| --grad-norm       | Gradient clipping value passed to tf.clip_by_value()               | 10        | log_uniform |
| --importance-c    | Importance weight truncation parameter.                            | 10.0      | log_uniform |
| --model           | Path to model .cfg file                                            | -         | -           |
| --n-steps         | Transition steps                                                   | 20        | categorical |
| --replay-ratio    | Lam value passed to np.random.poisson()                            | 4         | categorical |
| --trust-region    | True by default, if this flag is specified,                        | -         | -           |
|                   | trust region updates will be used                                  |           |             |
| --value-loss-coef | Value loss coefficient for value loss calculation                  | 0.5       | log_uniform |

**Command line**

    xagents train acer --env PongNoFrameskip-v4 --target-reward 19 --n-envs 16 --preprocess --checkpoints acer-pong.tf --buffer-max-size 5000 --buffer-initial-size 500 --trust-region

**Non-command line**

    from tensorflow.keras.optimizers import Adam
    
    import xagents
    from xagents import ACER
    from xagents.utils.buffers import ReplayBuffer1
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('PongNoFrameskip-v4', 16)
    buffers = [
        ReplayBuffer1(5000, initial_size=500, batch_size=1) for _ in range(len(envs))
    ]
    model_cfg = xagents.agents['acer']['model']['cnn'][0]
    optimizer = Adam(learning_rate=7e-4)
    model = ModelReader(
        model_cfg,
        output_units=[envs[0].action_space.n, envs[0].action_space.n],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    agent = ACER(envs, model, buffers, checkpoints=['acer-pong.tf'])
    agent.fit(target_reward=19)

### *6.3. DDPG*

* *Number of models:* 2
* *Action spaces:* continuous
* FPS varies because a different number of updates is executed at each train step, 
  unless `--gradient-steps` is specified.

| flags             | help                                                     | default   | hp_type     |
|:------------------|:---------------------------------------------------------|:----------|:------------|
| --actor-model     | Path to actor model .cfg file                            | -         | -           |
| --critic-model    | Path to critic model .cfg file                           | -         | -           |
| --gradient-steps  | Number of iterations per train step                      | -         | int         |
| --step-noise-coef | Coefficient multiplied by noise added to actions to step | 0.1       | log_uniform |
| --tau             | Value used for syncing target model weights              | 0.005     | log_uniform |

**Command line**

    xagents train ddpg --env BipedalWalker-v3 --target-reward 100 --n-envs 16 --checkpoints ddpg-actor-bipedal-walker.tf ddpg-critic-bipedal-walker.tf --buffer-max-size 1000000 --buffer-initial-size 25000 --buffer-batch-size 100

**Non-command line**

    from tensorflow.keras.optimizers import Adam
    
    import xagents
    from xagents import DDPG
    from xagents.utils.buffers import ReplayBuffer2
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('BipedalWalker-v3', 16, preprocess=False)
    buffers = [
        ReplayBuffer2(62500, slots=5, initial_size=1560, batch_size=8)
        for _ in range(len(envs))
    ]
    actor_model_cfg = xagents.agents['ddpg']['actor_model']['ann'][0]
    critic_model_cfg = xagents.agents['ddpg']['critic_model']['ann'][0]
    optimizer = Adam(learning_rate=7e-4)
    actor_model = ModelReader(
        actor_model_cfg,
        output_units=[envs[0].action_space.shape[0]],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    critic_model = ModelReader(
        actor_model_cfg,
        output_units=[1],
        input_shape=envs[0].observation_space.shape[0] + envs[0].action_space.shape[0],
        optimizer=optimizer,
    ).build_model()
    agent = DDPG(
        envs,
        actor_model,
        critic_model,
        buffers,
        checkpoints=['ddpg-actor-bipedal-walker.tf', 'ddpg-critic-bipedal-walker.tf'],
    )
    agent.fit(target_reward=100)

### *6.4. DQN-DDQN*

* *Number of models:* 1
* *Action spaces:* discrete

| flags                 | help                                                                    | default   | hp_type     |
|:----------------------|:------------------------------------------------------------------------|:----------|:------------|
| --double              | If specified, DDQN will be used                                         | -         | -           |
| --epsilon-decay-steps | Number of steps for `epsilon-start` to reach `epsilon-end`              | 150000    | int         |
| --epsilon-end         | Epsilon end value (minimum exploration rate)                            | 0.02      | log_uniform |
| --epsilon-start       | Starting epsilon value which is used to control random exploration.     | 1.0       | log_uniform |
|                       | It should be decremented and adjusted according to implementation needs |           |             |
| --model               | Path to model .cfg file                                                 | -         | -           |
| --target-sync-steps   | Sync target models every n steps                                        | 1000      | int         |

**Command line**

    xagents train dqn --env PongNoFrameskip-v4 --target-reward 19 --n-envs 3 --lr 1e-4 --preprocess --checkpoints dqn-pong.tf --buffer-max-size 50000 --buffer-initial-size 10000 --max-frame

**Non-command line**

    from tensorflow.keras.optimizers import Adam
    
    import xagents
    from xagents import DQN
    from xagents.utils.buffers import ReplayBuffer1
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('PongNoFrameskip-v4', 3, max_frame=True)
    buffers = [
        ReplayBuffer1(16666, initial_size=3333, batch_size=10) for _ in range(len(envs))
    ]
    model_cfg = xagents.agents['dqn']['model']['cnn'][0]
    optimizer = Adam(learning_rate=7e-4)
    model = ModelReader(
        model_cfg,
        output_units=[envs[0].action_space.n],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    agent = DQN(envs, model, buffers, checkpoints=['dqn-pong.tf'])
    agent.fit(target_reward=19)

**Note:** if you need a DDQN, specify `double=True` to the agent constructor or `--double`

### *6.5. PPO*

* *Number of models:* 1
* *Action spaces:* discrete, continuous

| flags               | help                                                 | default   | hp_type     |
|:--------------------|:-----------------------------------------------------|:----------|:------------|
| --advantage-epsilon | Value added to estimated advantage                   | 1e-08     | log_uniform |
| --clip-norm         | Clipping value passed to tf.clip_by_value()          | 0.1       | log_uniform |
| --entropy-coef      | Entropy coefficient for loss calculation             | 0.01      | log_uniform |
| --grad-norm         | Gradient clipping value passed to tf.clip_by_value() | 0.5       | log_uniform |
| --lam               | GAE-Lambda for advantage estimation                  | 0.95      | log_uniform |
| --mini-batches      | Number of mini-batches to use per update             | 4         | categorical |
| --model             | Path to model .cfg file                              | -         | -           |
| --n-steps           | Transition steps                                     | 128       | categorical |
| --ppo-epochs        | Gradient updates per training step                   | 4         | categorical |
| --value-loss-coef   | Value loss coefficient for value loss calculation    | 0.5       | log_uniform |

**Command line**

    xagents train ppo --env PongNoFrameskip-v4 --target-reward 19 --n-envs 16 --preprocess --checkpoints ppo-pong.tf

or

    xagents train ppo --env BipedalWalker-v3 --target-reward 200 --n-envs 16 --checkpoints ppo-bipedal-walker.tf

**Non-command line**

    from tensorflow.keras.optimizers import Adam
    
    import xagents
    from xagents import PPO
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('PongNoFrameskip-v4', 16)
    model_cfg = xagents.agents['ppo']['model']['cnn'][0]
    optimizer = Adam(learning_rate=7e-4)
    model = ModelReader(
        model_cfg,
        output_units=[envs[0].action_space.n, 1],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    agent = PPO(envs, model, checkpoints=['ppo-pong.tf'])
    agent.fit(target_reward=19)

### *6.6. TD3*

* *Number of models:* 3
* *Action spaces:* continuous
* TD3 constructor accepts only 2 models as input but accepts 3 for checkpoints or weights, 
because the second critic network will be cloned at runtime.
* FPS varies because a different number of updates is executed at each train step, 
  unless `--gradient-steps` is specified.

| flags               | help                                                               | default   | hp_type     |
|:--------------------|:-------------------------------------------------------------------|:----------|:------------|
| --actor-model       | Path to actor model .cfg file                                      | -         | -           |
| --critic-model      | Path to critic model .cfg file                                     | -         | -           |
| --gradient-steps    | Number of iterations per train step                                | -         | int         |
| --noise-clip        | Target noise clipping value                                        | 0.5       | log_uniform |
| --policy-delay      | Delay after which, actor weights and target models will be updated | 2         | categorical |
| --policy-noise-coef | Coefficient multiplied by noise added to target actions            | 0.2       | log_uniform |
| --step-noise-coef   | Coefficient multiplied by noise added to actions to step           | 0.1       | log_uniform |
| --tau               | Value used for syncing target model weights                        | 0.005     | log_uniform |

**Command line**

    xagents train td3 --env BipedalWalker-v3 --target-reward 300 --n-envs 16 --checkpoints td3-actor-bipedal-walker.tf td3-critic1-bipedal-walker.tf td3-critic2-bipedal-walker.tf --buffer-max-size 1000000 --buffer-initial-size 100 --buffer-batch-size 100

**Non-command line**

    from tensorflow.keras.optimizers import Adam
    
    import xagents
    from xagents import TD3
    from xagents.utils.buffers import ReplayBuffer2
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('BipedalWalker-v3', 16, preprocess=False)
    buffers = [
        ReplayBuffer2(62500, slots=5, initial_size=1560, batch_size=8)
        for _ in range(len(envs))
    ]
    actor_model_cfg = xagents.agents['td3']['actor_model']['ann'][0]
    critic_model_cfg = xagents.agents['td3']['critic_model']['ann'][0]
    optimizer = Adam(learning_rate=7e-4)
    actor_model = ModelReader(
        actor_model_cfg,
        output_units=[envs[0].action_space.shape[0]],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    critic_model = ModelReader(
        actor_model_cfg,
        output_units=[1],
        input_shape=envs[0].observation_space.shape[0] + envs[0].action_space.shape[0],
        optimizer=optimizer,
    ).build_model()
    agent = TD3(
        envs,
        actor_model,
        critic_model,
        buffers,
        checkpoints=[
            'td3-actor-bipedal-walker.tf',
            'td3-critic1-bipedal-walker.tf',
            'td3-critic2-bipedal-walker.tf',
        ],
    )
    agent.fit(target_reward=100)

### *6.7. TRPO*

* *Number of models:* 2
* *Action spaces:* discrete, continuous

| flags                   | help                                                           | default   | hp_type     |
|:------------------------|:---------------------------------------------------------------|:----------|:------------|
| --actor-iterations      | Actor optimization iterations per train step                   | 10        | int         |
| --actor-model           | Path to actor model .cfg file                                  | -         | -           |
| --advantage-epsilon     | Value added to estimated advantage                             | 1e-08     | log_uniform |
| --cg-damping            | Gradient conjugation damping parameter                         | 0.001     | log_uniform |
| --cg-iterations         | Gradient conjugation iterations per train step                 | 10        | -           |
| --cg-residual-tolerance | Gradient conjugation residual tolerance parameter              | 1e-10     | log_uniform |
| --clip-norm             | Clipping value passed to tf.clip_by_value()                    | 0.1       | log_uniform |
| --critic-iterations     | Critic optimization iterations per train step                  | 3         | int         |
| --critic-model          | Path to critic model .cfg file                                 | -         | -           |
| --entropy-coef          | Entropy coefficient for loss calculation                       | 0         | log_uniform |
| --fvp-n-steps           | Value used to skip every n-frames used to calculate FVP        | 5         | int         |
| --grad-norm             | Gradient clipping value passed to tf.clip_by_value()           | 0.5       | log_uniform |
| --lam                   | GAE-Lambda for advantage estimation                            | 1.0       | log_uniform |
| --max-kl                | Maximum KL divergence used for calculating Lagrange multiplier | 0.001     | log_uniform |
| --mini-batches          | Number of mini-batches to use per update                       | 4         | categorical |
| --n-steps               | Transition steps                                               | 512       | categorical |
| --ppo-epochs            | Gradient updates per training step                             | 4         | categorical |
| --value-loss-coef       | Value loss coefficient for value loss calculation              | 0.5       | log_uniform |

**Command line**

    xagents train trpo --env PongNoFrameskip-v4 --target-reward 19 --n-envs 16 --checkpoints trpo-actor-pong.tf trpo-critic-pong.tf --preprocess --lr 1e-3

or

    xagents train trpo --env BipedalWalker-v3 --target-reward 200 --n-envs 16 --checkpoints trpo-actor-pong.tf trpo-critic-pong.tf --lr 1e-3

**Non-command line**

    from tensorflow.keras.optimizers import Adam
    
    import xagents
    from xagents import TRPO
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('PongNoFrameskip-v4', 16)
    actor_model_cfg = xagents.agents['trpo']['actor_model']['cnn'][0]
    critic_model_cfg = xagents.agents['trpo']['critic_model']['cnn'][0]
    optimizer = Adam()
    actor_model = ModelReader(
        actor_model_cfg,
        output_units=[envs[0].action_space.n],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    critic_model = ModelReader(
        actor_model_cfg,
        output_units=[1],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    agent = TRPO(
        envs,
        actor_model,
        critic_model,
        checkpoints=[
            'trpo-actor-pong.tf',
            'trpo-critic-pong.tf',
        ],
    )
    agent.fit(target_reward=100)

## **7. License**
___

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## **8. Show your support**
___

Give a ⭐️ if this project helped you!

## **9. Contact**
___

schissmantics@outlook.com

Project link: https://github.com/schissmantics/xagents

[contributors-shield]: https://img.shields.io/github/contributors/schissmantics/xagents?style=flat-square
[contributors-url]: https://github.com/schissmantics/xagents/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/schissmantics/xagents?style=flat-square
[forks-url]: https://github.com/schissmantics/xagents/network/members
[stars-shield]: https://img.shields.io/github/stars/schissmantics/xagents?style=flat-square
[stars-url]: https://github.com/schissmantics/xagents/stargazers
[issues-shield]: https://img.shields.io/github/issues/schissmantics/xagents?style=flat-square
[issues-url]: https://github.com/schissmantics/xagents/issues
[license-shield]: https://img.shields.io/github/license/schissmantics/xagents
[license-url]: https://github.com/schissmantics/xagents/blob/master/LICENSE