import argparse

import numpy as np
import tensorflow as tf
import gym

from stable_baselines.a2c.utils import conv_to_fc
from stable_baselines.simple_ddpg import SimpleDDPG as DDPG
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv

class SimpleMlpPolicy(FeedForwardPolicy):
  """
    Policy object that implements actor critic, using a MLP (3 layers of 64)
  """
  def __init__(self, *args, **kwargs):
      super(SimpleMlpPolicy, self).__init__(*args, **kwargs,
                                         layers=[256, 256, 256],
                                         layer_norm=True,
                                         feature_extraction="mlp")

def main(args):
    """
    Train and save the DDPG model, for the FetchReach problem

    :param args: (ArgumentParser) the input arguments100000
    """
    env = SubprocVecEnv([lambda: gym.make('FetchReach-v1') for _ in range(128)])

    model = DDPG(
        env=env,
        policy=SimpleMlpPolicy,
        gamma=1.,
        actor_lr=1e-3,
        learning_starts=5000,
        verbose=1,
        batch_size=256,
        buffer_size=1000000,
        hindsight_mode=args.her,
        tensorboard_log=None,
    )
    assert model.goal_space is not None
    model.learn(total_timesteps=args.max_timesteps)

    model_filename = "fetchreach_model_{}.pkl".format(args.max_timesteps)
    print("Saving model to {}".format(model_filename))
    model.save(model_filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on goal Gridworld")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--her', default='none', type=str, help="HER type: final, future_k, none (default)")
    args = parser.parse_args()
    main(args)
