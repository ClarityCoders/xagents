from tensorflow.keras.optimizers import Adam

import xagents
from xagents import PPO
from xagents.utils.common import ModelReader, create_envs
import gym
from gym2048 import Game2048Env
import tensorflow as tf


import warnings
warnings.filterwarnings('ignore')

if tf.test.gpu_device_name():
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
else:
    print("Please install GPU version of TF")


envs = []
for i in range(2048):
    envs.append(Game2048Env())
print(len(envs))
model_cfg = xagents.agents['ppo']['model']['ann'][0]
optimizer = Adam(learning_rate=0.0001)
model = ModelReader(
    model_cfg,
    output_units=[envs[0].action_space.n, 1],
    input_shape=envs[0].observation_space.shape,
    optimizer=optimizer,
).build_model()
agent = PPO(envs, model, seed=42, checkpoints=[f'PPO-001-500M.tf'])
agent.fit(max_steps=500000000)