<p>
  <a href="https://github.com/emadboctorx/xagents/">
  </a>

  <h3 align="left">xagents - reusable, scalable, 
  performant reinforcement learning algorithms in tf2</h3>
  </p>

* [Installation](#installation)
* [Description](#description)
* [Features](#features)
* [Usage](#usage)
  * [Training](#training)
  * [Playing](#playing)
* [Command line options](#command-line-options)
* [Algorithms](#algorithms)
  * [A2C](#a2c)
  * [ACER](#acer)
  * [DDPG](#ddpg)
  * [DQN / DDQN](#dqn-ddqn)
  * [PPO](#ppo)
  * [TD3](#td3)
  * [TRPO](#trpo)
* [License](#license)
* [Show your support](#show-your-support)
* [Contact](#contact)

![pong](/gifs/pong.gif)
![breakout](/gifs/breakout.gif)

![bipedal-walker](/gifs/bipedal-walker.gif)


### **Installation**
___

```sh
git clone https://github.com/emadboctorx/xagents
cd xagents
pip install .
```

**Notes:** 
* To be able to run atari environments, follow the instructions in [atari-py](https://github.com/openai/atari-py#roms)
to install [ROMS](http://www.atarimania.com/rom_collection_archive_atari_2600_roms.html).
  
* To be able to run the tests remotely, [pytest-xvfb](https://pypi.org/project/pytest-xvfb/) plugin will
be automatically installed but will require an additional step ...
  * For macOS users, you'll have to install [Xquartz](https://www.xquartz.org)
    ```shell 
    brew install xquartz
    ```
  * For linux users, you'll have to install [xvfb](https://www.x.org/releases/X11R7.6/doc/man/man1/Xvfb.1.xhtml)
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

<!-- DESCRIPTION -->
## **Description**
___
xagents is a tensorflow based mini-library which facilitates experimentation with
existing reinforcement learning algorithms, as well as the implementation of new ones. It
provides well tested components that can be easily modified or extended. The available
selection of algorithms can be used directly or through command line.

<!-- FEATURES -->
## **Features**

* Tensorflow 2.
* wandb support.
* Multiple environments (All agents).
* Command line options.
* Hyperparameter auto-tuning.
* Early stopping, reduce on plateau.
* Resume training and update metrics from last checkpoint.
* Discrete and continuous action spaces.
* Unit tests.
* Models are loaded from .cfg files.
* Training history checkpoints.
* Single / multiple training history plotting.
* Reproducible results.
* Gameplay output to .jpg frames or .mp4 vid.

## **Usage**

All agents are available through the command line.

    xagents <command> <agent> [options] [args]

**Note:** Unless called from command line with `--weights` passed,
all models passed to agents in code, should be loaded with weights 
beforehand, if called for resuming training or playing.

### **Training**

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

### **Playing**

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

If you need to save the game ...

**For video**

    agent.play(video_dir='/path/to/video-dir/')

or

    xagents play a2c --video-dir /path/to/video-dir/  

**For frames**

    agent.play(frame_dir='/path/to/frame-dir/')

or 

    xagents play a2c --frame-dir /path/to/frame-dir/

and all arguments can be combined `--video-dir <vid-dir> --frame-dir <frame-dir> --render`

<!-- COMMAND LINE OPTIONS -->
## **Command line options**

**Note:** Not all the flags listed below are available at once, and to know which 
ones are available respective to the command you passed, you can use:

    xagents <command>

or

    xagents <command> <agent>

which should list command + agent options combined

**Flags (Available for all agents)**

| flags                         | help                                                                         | required   | default   |
|:------------------------------|:-----------------------------------------------------------------------------|:-----------|:----------|
| --env                         | gym environment id                                                           | True       | -         |
| --n-envs                      | Number of environments to create                                             | -          | 1         |
| --preprocess                  | If specified, states will be treated as atari frames                         | -          | -         |
|                               | and preprocessed accordingly                                                 |            |           |
| --no-env-scale                | If specified, frames will not be scaled by preprocessor                      | -          | -         |
| --lr                          | Learning rate passed to a tensorflow.keras.optimizers.Optimizer              | -          | 0.0007    |
| --opt-epsilon                 | Epsilon passed to a tensorflow.keras.optimizers.Optimizer                    | -          | 1e-07     |
| --beta1                       | Beta1 passed to a tensorflow.keras.optimizers.Optimizer                      | -          | 0.9       |
| --beta2                       | Beta2 passed to a tensorflow.keras.optimizers.Optimizer                      | -          | 0.999     |
| --weights                     | Path(s) to model(s) weight(s) to be loaded by agent output_models            | -          | -         |
| --max-frame                   | If specified, max & skip will be applied during preprocessing                | -          | -         |
| --reward-buffer-size          | Size of the total reward buffer, used for calculating                        | -          | 100       |
|                               | mean reward value to be displayed.                                           |            |           |
| --gamma                       | Discount factor                                                              | -          | 0.99      |
| --display-precision           | Number of decimals to be displayed                                           | -          | 2         |
| --seed                        | Random seed                                                                  | -          | -         |
| --scale-inputs                | If specified, inputs will be scaled by agent                                 | -          | -         |
| --log-frequency               | Log progress every n games                                                   | -          | -         |
| --checkpoints                 | Path(s) to new model(s) to which checkpoint(s) will be saved during training | -          | -         |
| --history-checkpoint          | Path to .parquet file to save training history                               | -          | -         |
| --plateau-reduce-factor       | Factor multiplied by current learning rate when there is a plateau           | -          | 0.9       |
| --plateau-reduce-patience     | Minimum non-improvements to reduce lr                                        | -          | 10        |
| --early-stop-patience         | Minimum plateau reduces to stop training                                     | -          | 3         |
| --divergence-monitoring-steps | Steps after which, plateau and early stopping are active                     | -          | 500000    |
| --target-reward               | Target reward when reached, training is stopped                              | -          | -         |
| --max-steps                   | Maximum number of environment steps, when reached, training is stopped       | -          | -         |
| --monitor-session             | Wandb session name                                                           | -          | -         |

<!-- ALGORITHMS -->
## **Algorithms**

**General notes**

* All the default hyperparameters don't work for all environments.
  Which means you either need to tune them according to the given environment,
  or pass previously tuned ones, in order to get good results.
* `--model <model.cfg>` or `--actor-model <actor.cfg>` and `--critic-model <critic.cfg>` are optional 
  which means, if not specified, the default model(s) will be loaded, so you don't have to worry about it.
* You can also use external models by passing them to agent constructor. If you do, you will have to ensure
  your models outputs match what the implementation expects, or modify it accordingly.
* For atari environments / the ones that return an image by default, use the `--preprocess` flag for image preprocessing.
* Also for atari environments when specified with agents that use a replay buffer including ACER, DQN
  you should use `--no-env-scale` and `--scale-inputs` flags otherwise you'll get an error. This applies to training and playing.
* For checkpoints to be saved, `--checkpoints <checkpoint1.tf> <checkpoint2.tf>` should
be specified for the model(s) to be saved. The number of passed checkpoints should match the number
  of models the agent accepts.
* For loading weights either for resuming training or for playing a game `--weights <weights1.tf> <weights2.tf>`
and same goes for the weights, they should match the number of agent models.
* For using a random seed, a `seed=some_seed` should be passed to agent constructor and ModelReader constructor if
specified from code. If from the command line, all you need is to pass `--seed <some-seed>`
* To save training history `history_checkpoint=some_history.parquet` should be specified
to agent constructor or alternatively using `--history-checkpoint <some-history.parquet>`. 
  If the history checkpoint exists, training metrics will automatically start from where it left.
  
### *A2C*

* *Number of models:* 1
* *Action spaces:* discrete and continuous

| flags             | help                                                 | default   |
|:------------------|:-----------------------------------------------------|:----------|
| --model           | Path to model .cfg file                              | -         |
| --entropy-coef    | Entropy coefficient for loss calculation             | 0.01      |
| --value-loss-coef | Value loss coefficient for value loss calculation    | 0.5       |
| --grad-norm       | Gradient clipping value passed to tf.clip_by_value() | 0.5       |
| --n-steps         | Transition steps                                     | 5         |

**Command line**

     xagents train a2c --env PongNoFrameskip-v4 --target-reward 19 --n-envs 16 --preprocess --checkpoints a2c-pong.tf --opt-epsilon 1e-5 --beta1 0 beta2 0.99

OR

    xagents train a2c --env BipedalWalker-v3 --target-reward 100 --n-envs 16 --checkpoints a2c-bipedal-walker.tf

**Non-command line**
    
    from tensorflow.keras.optimizers import Adam

    import xagents
    from xagents import A2C
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('PongNoFrameskip-v4', 16)
    model_cfg = xagents.agents['a2c']['model']['cnn'][0]
    optimizer = Adam(learning_rate=7e-4, epsilon=1e-5, beta_1=0, beta_2=0.99)
    model = ModelReader(
        model_cfg,
        output_units=[envs[0].action_space.n, 1],
        input_shape=envs[0].observation_space.shape,
        optimizer=optimizer,
    ).build_model()
    agent = A2C(envs, model, checkpoints=['a2c-pong.tf'])
    agent.fit(target_reward=19)

And for `BipedalWalker-v3`, the only difference is that you have to specify `preprocess=False` to `create_envs()`

### *ACER*

* *Number of models:* 1
* *Action spaces:* discrete

| flags                 | help                                                               | default   |
|:----------------------|:-------------------------------------------------------------------|:----------|
| --model               | Path to model .cfg file                                            | -         |
| --entropy-coef        | Entropy coefficient for loss calculation                           | 0.01      |
| --value-loss-coef     | Value loss coefficient for value loss calculation                  | 0.5       |
| --grad-norm           | Gradient clipping value passed to tf.clip_by_value()               | 10        |
| --n-steps             | Transition steps                                                   | 20        |
| --ema-alpha           | Moving average decay passed to tf.train.ExponentialMovingAverage() | 0.99      |
| --replay-ratio        | Lam value passed to np.random.poisson()                            | 4         |
| --epsilon             | epsilon used in gradient updates                                   | 1e-06     |
| --importance-c        | Importance weight truncation parameter.                            | 10.0      |
| --delta               | delta param used for trust region update                           | 1         |
| --trust-region        | True by default, if this flag is specified,                        | -         |
|                       | trust region updates will be used                                  |           |
| --buffer-max-size     | Maximum replay buffer size                                         | 10000     |
| --buffer-initial-size | Replay buffer initial size                                         | -         |
| --buffer-batch-size   | Replay buffer batch size                                           | 32        |
| --buffer-n-steps      | Replay buffer transition step                                      | 1         |

**Command line**

    xagents train acer --env PongNoFrameskip-v4 --target-reward 19 --n-envs 16 --preprocess --checkpoints acer-pong.tf --buffer-max-size 5000 --buffer-initial-size 500 --buffer-batch-size 16 --trust-region --no-env-scale --scale-inputs

**Non-command line**

    from tensorflow.keras.optimizers import Adam
    
    import xagents
    from xagents import ACER
    from xagents.utils.buffers import ReplayBuffer1
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('PongNoFrameskip-v4', 16, scale_frames=False)
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
    agent = ACER(envs, model, buffers, checkpoints=['acer-pong.tf'], scale_inputs=True)
    agent.fit(target_reward=19)

### *DDPG*

* *Number of models:* 2
* *Action spaces:* continuous

| flags                 | help                                                     | default   |
|:----------------------|:---------------------------------------------------------|:----------|
| --actor-model         | Path to actor model .cfg file                            | -         |
| --critic-model        | Path to critic model .cfg file                           | -         |
| --gradient-steps      | Number of iterations per train step                      | -         |
| --tau                 | Value used for syncing target model weights              | 0.005     |
| --step-noise-coef     | Coefficient multiplied by noise added to actions to step | 0.1       |
| --buffer-max-size     | Maximum replay buffer size                               | 10000     |
| --buffer-initial-size | Replay buffer initial size                               | -         |
| --buffer-batch-size   | Replay buffer batch size                                 | 32        |
| --buffer-n-steps      | Replay buffer transition step                            | 1         |

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

### *DQN-DDQN*

* *Number of models:* 1
* *Action spaces:* discrete

| flags                 | help                                                                    | default   |
|:----------------------|:------------------------------------------------------------------------|:----------|
| --model               | Path to model .cfg file                                                 | -         |
| --double              | If specified, DDQN will be used                                         | -         |
| --epsilon-start       | Starting epsilon value which is used to control random exploration.     | 1.0       |
|                       | It should be decremented and adjusted according to implementation needs |           |
| --epsilon-end         | Epsilon end value (minimum exploration rate)                            | 0.02      |
| --epsilon-decay-steps | Number of steps for `epsilon-start` to reach `epsilon-end`              | 150000    |
| --target-sync-steps   | Sync target models every n steps                                        | 1000      |
| --n-steps             | Transition steps                                                        | 1         |
| --buffer-max-size     | Maximum replay buffer size                                              | 10000     |
| --buffer-initial-size | Replay buffer initial size                                              | -         |
| --buffer-batch-size   | Replay buffer batch size                                                | 32        |
| --buffer-n-steps      | Replay buffer transition step                                           | 1         |

**Command line**

    xagents train dqn --env PongNoFrameskip-v4 --target-reward 19 --n-envs 3 --lr 1e-4 --preprocess --checkpoints dqn-pong.tf --scale-inputs --no-env-scale --buffer-max-size 50000 --buffer-initial-size 10000 --max-frame

**Non-command line**

    from tensorflow.keras.optimizers import Adam
    
    import xagents
    from xagents import DQN
    from xagents.utils.buffers import ReplayBuffer1
    from xagents.utils.common import ModelReader, create_envs
    
    envs = create_envs('PongNoFrameskip-v4', 3, scale_frames=False, max_frame=True)
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
    agent = DQN(envs, model, buffers, checkpoints=['dqn-pong.tf'], scale_inputs=True)
    agent.fit(target_reward=19)

**Note:** if you need a DDQN, specify `double=True` to the agent constructor or `--double`

### *PPO*

* *Number of models:* 1
* *Action spaces:* discrete, continuous

| flags               | help                                                 | default   |
|:--------------------|:-----------------------------------------------------|:----------|
| --model             | Path to model .cfg file                              | -         |
| --entropy-coef      | Entropy coefficient for loss calculation             | 0.01      |
| --value-loss-coef   | Value loss coefficient for value loss calculation    | 0.5       |
| --grad-norm         | Gradient clipping value passed to tf.clip_by_value() | 0.5       |
| --n-steps           | Transition steps                                     | 128       |
| --lam               | GAE-Lambda for advantage estimation                  | 0.95      |
| --ppo-epochs        | Gradient updates per training step                   | 4         |
| --mini-batches      | Number of mini-batches to use per update             | 4         |
| --advantage-epsilon | Value added to estimated advantage                   | 1e-08     |
| --clip-norm         | Clipping value passed to tf.clip_by_value()          | 0.1       |

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

### *TD3*

* *Number of models:* 3
* *Action spaces:* continuous

| flags                 | help                                                               | default   |
|:----------------------|:-------------------------------------------------------------------|:----------|
| --actor-model         | Path to actor model .cfg file                                      | -         |
| --critic-model        | Path to critic model .cfg file                                     | -         |
| --gradient-steps      | Number of iterations per train step                                | -         |
| --tau                 | Value used for syncing target model weights                        | 0.005     |
| --step-noise-coef     | Coefficient multiplied by noise added to actions to step           | 0.1       |
| --policy-delay        | Delay after which, actor weights and target models will be updated | 2         |
| --policy-noise-coef   | Coefficient multiplied by noise added to target actions            | 0.2       |
| --noise-clip          | Target noise clipping value                                        | 0.5       |
| --buffer-max-size     | Maximum replay buffer size                                         | 10000     |
| --buffer-initial-size | Replay buffer initial size                                         | -         |
| --buffer-batch-size   | Replay buffer batch size                                           | 32        |
| --buffer-n-steps      | Replay buffer transition step                                      | 1         |

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

**Note:** TD3 accepts only 2 models as input but accepts 3 for checkpoints or weights, 
because the second critic network will be cloned at runtime.

### *TRPO*

* *Number of models:* 2
* *Action spaces:* discrete, continuous

| flags                   | help                                                           | default   |
|:------------------------|:---------------------------------------------------------------|:----------|
| --entropy-coef          | Entropy coefficient for loss calculation                       | 0         |
| --value-loss-coef       | Value loss coefficient for value loss calculation              | 0.5       |
| --grad-norm             | Gradient clipping value passed to tf.clip_by_value()           | 0.5       |
| --n-steps               | Transition steps                                               | 512       |
| --lam                   | GAE-Lambda for advantage estimation                            | 1.0       |
| --ppo-epochs            | Gradient updates per training step                             | 4         |
| --mini-batches          | Number of mini-batches to use per update                       | 4         |
| --advantage-epsilon     | Value added to estimated advantage                             | 1e-08     |
| --clip-norm             | Clipping value passed to tf.clip_by_value()                    | 0.1       |
| --actor-model           | Path to actor model .cfg file                                  | -         |
| --critic-model          | Path to critic model .cfg file                                 | -         |
| --max-kl                | Maximum KL divergence used for calculating Lagrange multiplier | 0.001     |
| --cg-iterations         | Gradient conjugation iterations per train step                 | 10        |
| --cg-residual-tolerance | Gradient conjugation residual tolerance parameter              | 1e-10     |
| --cg-damping            | Gradient conjugation damping parameter                         | 0.001     |
| --actor-iterations      | Actor optimization iterations per train step                   | 10        |
| --critic-iterations     | Critic optimization iterations per train step                  | 3         |
| --fvp-n-steps           | Value used to skip every n-frames used to calculate FVP        | 5         |

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

## **License**
___

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.

## **Show your support**
___

Give a ⭐️ if this project helped you!

## **Contact**
___

Emad Boctor - emad_1989@hotmail.com

Project link: https://github.com/emadboctorx/xagents