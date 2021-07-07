import random

import gym
import pytest

from xagents.utils.common import AtariWrapper


@pytest.mark.parametrize(
    'resize_shape, scale_frames',
    [[[random.randint(50, 100)] * 2, random.choice([True, False])] for _ in range(5)],
)
def test_atari_wrapper(resize_shape, scale_frames):
    env = gym.make('PongNoFrameskip-v4')
    env = AtariWrapper(env, resize_shape=resize_shape, scale_frames=scale_frames)
    reset_state = env.reset()
    state, *_ = env.step(env.action_space.sample())
    assert state.shape == reset_state.shape == (*resize_shape, 1)
    if scale_frames:
        assert (state < 1).all() and (reset_state < 1).all()
