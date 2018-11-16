import argparse

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
from envs.custom_fetch import CustomFetchReachEnv, CustomFetchPushEnv
from envs import discrete_to_box_wrapper
from envs.goal_grid import GoalGridWorldEnv

from stable_baselines.a2c.utils import conv_to_fc
from stable_baselines.simple_ddpg import SimpleDDPG as DDPG, make_feedforward_extractor, identity_extractor
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
                                         activ=tf.nn.relu,
                                         feature_extraction="mlp")

def main(args):
    """
    Train and save the DDPG model, for the FetchReach problem

    :param args: (ArgumentParser) the input arguments100000
    """
    if "CustomFetchReach" in args.env:
      env = SubprocVecEnv([CustomFetchReachEnv for _ in range(48)])
    elif "CustomFetchPush" in args.env:
      env = SubprocVecEnv([CustomFetchPushEnv for _ in range(48)])
    elif "GoalGrid" in args.env:
      grid_file = "{}.txt".format(args.room_file)
      env = SubprocVecEnv([lambda: discrete_to_box_wrapper(GoalGridWorldEnv(grid_size=5, max_step=50, grid_file=grid_file)) for _ in range(16)])
    else:
      env = SubprocVecEnv([lambda: gym.make(args.env) for _ in range(48)])

    if not args.folder:
      args.folder = '/tmp'

    model = DDPG(
        env=env,
        policy=SimpleMlpPolicy,
        gamma=0.96,
        actor_lr=1e-3,
        critic_lr=1e-4,
        learning_starts=1000,
        joint_feature_extractor=None,
        joint_goal_feature_extractor=None,
        clip_value_fn_range=(0.,1.),
        train_freq=10,
        target_network_update_frac=0.02,
        target_network_update_freq=20,
        epsilon_random_exploration=0.,
        action_noise='normal_0.5',
        critic_l2_regularization=0.,
        action_l2_regularization=1e-4,
        verbose=1,
        batch_size=128,
        buffer_size=2000000,
        hindsight_mode=args.her,
        tensorboard_log="{}/ddpg_tensorboard/".format(args.folder),
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
    parser.add_argument('--folder', default='/tmp', type=str, help="Tensorboard_folder")
    parser.add_argument('--room-file', default='room_5x5_empty', type=str,
                        help="Room type: room_5x5_empty (default), 2_room_9x9")
    args = parser.parse_args()
    main(args)
