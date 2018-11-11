import argparse

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
from gym.envs.robotics import FetchReachEnv
from envs.custom_fetch import CustomFetchReachEnv, CustomFetchPushEnv

from stable_baselines.a2c.utils import conv_to_fc
from stable_baselines.simple_ddpg import SimpleDDPG as DDPG
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv

class SimpleMlpPolicy(FeedForwardPolicy):
  """
    Policy object that implements actor critic, using a MLP (2 layers of 256)
  """
  def __init__(self, *args, **kwargs):
      super(SimpleMlpPolicy, self).__init__(*args, **kwargs,
                                         layers=[400,400],
                                         layer_norm=True,
                                         feature_extraction="mlp")

def main(args):
    """
    Train and save the DDPG model, for the FetchReach problem

    :param args: (ArgumentParser) the input arguments100000
    """
    if "FetchReach" in args.env:
      env = SubprocVecEnv([lambda: CustomFetchReachEnv() for _ in range(48)])
    elif "FetchPush" in args.env:
      env = SubprocVecEnv([lambda: CustomFetchPushEnv() for _ in range(48)])
    else:
      env = SubprocVecEnv([lambda: gym.make(args.env) for _ in range(48)])

    model = DDPG(
        env=env,
        policy=SimpleMlpPolicy,
        gamma=0.98,
        actor_lr=1e-3,
        learning_starts=2500,
        target_network_update_frac=0.0005,
        epsilon_random_exploration=0.2,
        verbose=1,
        batch_size=128,
        buffer_size=1000000,
        observation_range=(-200., 200.),
        hindsight_mode=args.her,
        tensorboard_log="/tmp/ddpg_tensorboard/",
    )
    model.learn(total_timesteps=args.max_timesteps, tb_log_name="DDPG_{}_{}".format(args.env, args.tb), log_interval=10)

    model_filename = "ddpg_model_{}_{}.pkl".format(args.env, args.max_timesteps)
    print("Saving model to {}".format(model_filename))
    model.save(model_filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DDPG")
    parser.add_argument('--env', default="FetchReach-v1", type=str, help="Gym environment")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--her', default='none', type=str, help="Hindsight mode (e.g., future_4 or final)")
    parser.add_argument('--tb', default='1', type=str, help="Tensorboard_name")
    args = parser.parse_args()
    main(args)
