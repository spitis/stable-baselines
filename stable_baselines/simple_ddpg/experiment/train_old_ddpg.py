import argparse

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
from gym.envs.robotics import FetchReachEnv

from stable_baselines.ddpg import DDPG
from stable_baselines.ddpg.policies import DDPG_FeedForwardPolicy as FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

class SimpleMlpPolicy(FeedForwardPolicy):
  """
    Policy object that implements actor critic, using a MLP (2 layers of 256)
  """
  def __init__(self, *args, **kwargs):
      super(SimpleMlpPolicy, self).__init__(*args, **kwargs,
                                         layers=[256, 256],
                                         layer_norm=True,
                                         feature_extraction="mlp")

def main(args):
    """
    Train and save the DDPG model, for the FetchReach problem

    :param args: (ArgumentParser) the input arguments100000
    """
    if args.env == "FetchReach-v1":
      env = wrappers.FlattenDictWrapper(FetchReachEnv(reward_type='dense'), ['observation', 'desired_goal'])
    else:
      env = args.env
      
    model = DDPG(
        env=env,
        policy=SimpleMlpPolicy,
        gamma=0.99,
        actor_lr=1e-4,
        tau=0.001,
        verbose=1,
        batch_size=256,
        memory_limit=500000,
        observation_range=(-200., 200.),
        tensorboard_log="/tmp/ddpg_tensorboard/",
    )
    model.learn(total_timesteps=args.max_timesteps, tb_log_name="OLD_DDPG_{}_{}".format(args.env, args.tb), log_interval=1)

    model_filename = "ddpg_model_{}_{}.pkl".format(args.env, args.max_timesteps)
    print("Saving model to {}".format(model_filename))
    model.save(model_filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DDPG")
    parser.add_argument('--env', default='FetchReach-v1', type=str, help="Gym environment")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--tb', default='1', type=str, help="Tensorboard_name")
    args = parser.parse_args()
    main(args)
