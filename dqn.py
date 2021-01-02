import os
from collections import deque
from time import perf_counter

import cv2
import gym
import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

from utils import AtariPreprocessor, ReplayBuffer


class DQN:
    def __init__(
        self,
        env,
        replay_buffer,
        batch_size=32,
        checkpoint=None,
        reward_buffer_size=100,
        epsilon_start=1.0,
        epsilon_end=0.01,
        frame_skips=4,
        resize_shape=(84, 84),
        state_buffer_size=2,
        n_steps=1,
        gamma=0.99,
        double=False,
    ):
        """
        Initialize agent settings.
        Args:
            env: gym environment that returns states as atari frames.
                last n observations in the form of (state, action, reward, done, new state)
            replay_buffer: ReplayBuffer object to use for memorizing transitions.
            batch_size: Training batch size.
            checkpoint: Path to .tf filename under which the trained model will be saved.
            reward_buffer_size: Size of the reward buffer that will hold the last n total
                rewards which will be used for calculating the mean reward.
            epsilon_start: Start value of epsilon that regulates exploration during training.
            epsilon_end: End value of epsilon which represents the minimum value of epsilon
                which will not be decayed further when reached.
            frame_skips: Number of frame skips to use per environment step.
            resize_shape: (m, n) dimensions for the frame preprocessor
            state_buffer_size: Size of the state buffer used by the frame preprocessor.
            n_steps: n-step transition for example given s1, s2, s3, s4 and n_step = 4,
                transition will be s1 -> s4 (defaults to 1, s1 -> s2)
            gamma: Discount factor used for gradient updates.
            double: If True, DDQN is used for gradient updates.
        """
        self.env = gym.make(env)
        self.env = AtariPreprocessor(
            self.env, frame_skips, resize_shape, state_buffer_size
        )
        self.input_shape = self.env.observation_space.shape
        self.main_model = self.create_model()
        self.target_model = self.create_model()
        self.buffer = replay_buffer
        self.batch_size = batch_size
        self.checkpoint_path = checkpoint
        self.total_rewards = deque(maxlen=reward_buffer_size)
        self.best_reward = -float('inf')
        self.mean_reward = -float('inf')
        self.state = self.env.reset()
        self.steps = 0
        self.frame_speed = 0
        self.last_reset_frame = 0
        self.epsilon_start = self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.games = 0
        self.n_steps = n_steps
        self.gamma = gamma
        self.double = double

    def create_model(self):
        """
        Create model that will be used for the main and target models.
        Returns:
            None
        """
        x0 = Input(self.input_shape)
        x = Conv2D(32, 8, 4, activation='relu')(x0)
        x = Conv2D(64, 4, 2, activation='relu')(x)
        x = Conv2D(64, 3, 1, activation='relu')(x)
        x = Flatten()(x)
        x = Dense(512, 'relu')(x)
        x = Dense(self.env.action_space.n)(x)
        return Model(x0, x)

    def get_action(self, training=True):
        """
        Generate action following an epsilon-greedy policy.
        Args:
            training: If False, no use of randomness will apply.

        Returns:
            A random action or Q argmax.
        """
        if training and np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        q_values = self.main_model.predict(np.expand_dims(self.state, 0))
        return np.argmax(q_values)

    def update(self, batch):
        """
        Update gradients given a batch.
        Args:
            batch: A batch of observations in the form of
                [[states], [actions], [rewards], [dones], [next states]]

        Returns:
            None
        """
        states, actions, rewards, dones, new_states = batch
        q_states = self.main_model.predict(states)
        if self.double:
            new_state_actions = np.argmax(self.main_model.predict(new_states), 1)
            new_state_q_values = self.target_model.predict(new_states)
            new_state_values = new_state_q_values[
                np.arange(self.batch_size), new_state_actions
            ]
        else:
            new_state_values = self.target_model.predict(new_states).max(1)
        new_state_values[dones] = 0
        target_values = np.copy(q_states)
        target_value_update = new_state_values * self.gamma ** self.n_steps + rewards
        state_action_values = target_values[np.arange(self.batch_size), actions]
        target_values[np.arange(self.batch_size), actions] = target_value_update
        self.main_model.fit(states, target_values, verbose=0)
        if self.buffer.priorities:
            squared_loss = (state_action_values - target_value_update) ** 2
            priorities = (
                self.buffer.current_weights * squared_loss + self.buffer.priority_bias
            )
            self.buffer.update_priorities(priorities)

    def checkpoint(self):
        """
        Save model weights if current reward > best reward.
        Returns:
            None
        """
        if self.best_reward < self.mean_reward:
            print(f'Best reward updated: {self.best_reward} -> {self.mean_reward}')
            if self.checkpoint_path:
                self.main_model.save_weights(self.checkpoint_path)
        self.best_reward = max(self.mean_reward, self.best_reward)

    def display_metrics(self, episode_reward):
        """
        Display progress metrics to the console.
        Args:
            episode_reward: Current episode reward.

        Returns:
            None
        """
        display_titles = (
            'frame',
            'games',
            'mean reward',
            'best reward',
            'episode reward',
            'epsilon',
            'speed',
        )
        display_values = (
            self.steps,
            self.games,
            self.mean_reward,
            self.best_reward,
            episode_reward,
            np.around(self.epsilon, 2),
            f'{round(self.frame_speed)} steps/s',
        )
        display = (
            f'{title}: {value}' for title, value in zip(display_titles, display_values)
        )
        print(', '.join(display))

    def update_metrics(self, episode_reward, start_time):
        """
        Update progress metrics.
        Args:
            episode_reward: Total reward per a single episode (game).
            start_time: Episode start time, used for calculating fps.

        Returns:
            None
        """
        self.games += 1
        self.checkpoint()
        self.total_rewards.append(episode_reward)
        self.frame_speed = (self.steps - self.last_reset_frame) / (
            perf_counter() - start_time
        )
        self.last_reset_frame = self.steps
        self.mean_reward = np.around(np.mean(self.total_rewards), 2)
        self.display_metrics(episode_reward)

    def fit(
        self,
        target_reward,
        decay_n_steps=150000,
        learning_rate=1e-4,
        update_target_steps=1000,
        monitor_session=None,
        weights=None,
        max_steps=None,
    ):
        """
        Train agent on a supported environment
        Args:
            target_reward: Target reward, if achieved, the training will stop
            decay_n_steps: Maximum steps that determine epsilon decay rate.
            learning_rate: Model learning rate shared by both main and target networks.
            update_target_steps: Update target model every n steps.
            monitor_session: Session name to use for monitoring the training with wandb.
            weights: Path to .tf trained model weights to continue training.
            max_steps: Maximum number of steps, if reached the training will stop.

        Returns:
            None
        """
        if monitor_session:
            wandb.init(name=monitor_session)
        episode_reward = 0
        start_time = perf_counter()
        optimizer = Adam(learning_rate)
        if weights:
            self.main_model.load_weights(weights)
            self.target_model.load_weights(weights)
        self.main_model.compile(optimizer, loss='mse')
        self.target_model.compile(optimizer, loss='mse')
        while True:
            if self.mean_reward >= target_reward:
                print(f'Reward achieved in {self.steps} steps!')
                break
            if max_steps and self.steps >= max_steps:
                print(f'Maximum steps exceeded')
                break
            self.steps += 1
            self.epsilon = max(
                self.epsilon_end, self.epsilon_start - self.steps / decay_n_steps
            )
            action = self.get_action()
            new_state, reward, done, info = self.env.step(action)
            episode_reward += reward
            self.buffer.append((self.state, action, reward, done, new_state))
            self.state = new_state
            if done:
                self.update_metrics(episode_reward, start_time)
                start_time = perf_counter()
                episode_reward = 0
                self.state = self.env.reset()
            if len(self.buffer) < self.buffer.initial_size:
                continue
            batch = self.buffer.get_sample()
            self.update(batch)
            if self.steps % update_target_steps == 0:
                self.target_model.set_weights(self.main_model.get_weights())

    def play(self, weights=None, video_dir=None, render=False, frame_dir=None):
        """
        Play and display a game.
        Args:
            weights: Path to trained weights, if not specified, the most recent
                model weights will be used.
            video_dir: Path to directory to save the resulting game video.
            render: If True, the game will be displayed.
            frame_dir: Path to directory to save game frames.

        Returns:
            None
        """
        if weights:
            self.main_model.load_weights(weights)
        if video_dir:
            self.env = gym.wrappers.Monitor(self.env, video_dir)
        self.state = self.env.reset()
        steps = 0
        for dir_name in (video_dir, frame_dir):
            os.makedirs(dir_name or '.', exist_ok=True)
        while True:
            if render:
                self.env.render()
            if frame_dir:
                frame = self.env.render(mode='rgb_array')
                cv2.imwrite(os.path.join(frame_dir, f'{steps:05d}.jpg'), frame)
            action = self.get_action(False)
            self.state, reward, done, info = self.env.step(action)
            if done:
                break
            steps += 1


if __name__ == '__main__':
    # import tensorflow as tf
    tf.compat.v1.disable_eager_execution()
    bf = ReplayBuffer(10000, prioritize=True)
    agn = DQN(
        'PongNoFrameskip-v4',
        bf,
        # checkpoint='pong_replay_buffer_test.tf',
        # n_steps=4,
        epsilon_end=0.02,
        # double=True,
    )
    agn.fit(19)
    # agn.play('/Users/emadboctor/Desktop/code/dqn-pong-19-model/pong_test.tf', render=True, video_dir='.')
