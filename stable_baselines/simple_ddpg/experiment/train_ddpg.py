import argparse

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
from envs.custom_fetch import CustomFetchReachEnv, CustomFetchPushEnv, CustomFetchPushEnv6DimGoal, CustomFetchSlideEnv, CustomFetchSlideEnv9DimGoal
from envs import discrete_to_box_wrapper
from envs.goal_grid import GoalGridWorldEnv

from stable_baselines.a2c.utils import conv_to_fc
from stable_baselines.simple_ddpg import SimpleDDPG as DDPG, make_feedforward_extractor, identity_extractor
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.landmark_generator import RandomLandmarkGenerator

class SimpleMlpPolicy(FeedForwardPolicy):
  """
    Policy object that implements actor critic, using a MLP (2 layers of 256)
  """
  def __init__(self, *args, **kwargs):
      super(SimpleMlpPolicy, self).__init__(*args, **kwargs,
                                         layers=[256,256,256],
                                         layer_norm=True,
                                         activ=tf.nn.relu,
                                         feature_extraction="mlp")

def main(args):
    """
    Train and save the DDPG model, for the FetchReach problem

    :param args: (ArgumentParser) the input arguments100000
    """

    if args.env == "CustomFetchReach":
      env_fn = lambda: lambda: CustomFetchReachEnv()
    elif args.env == "CustomFetchPush6Dim":
      env_fn = lambda: lambda: CustomFetchPushEnv6DimGoal()
    elif args.env == "CustomFetchPush":
      env_fn = lambda: lambda: CustomFetchPushEnv()
    elif args.env == "CustomFetchSlide9Dim":
      env_fn = lambda: lambda: CustomFetchSlideEnv9DimGoal()
    elif args.env == "CustomFetchSlide":
      env_fn = lambda: lambda: CustomFetchSlideEnv()
    elif "GoalGrid" in args.env:
      grid_file = "{}.txt".format(args.room_file)
      env_fn = lambda: discrete_to_box_wrapper(GoalGridWorldEnv(grid_size=5, max_step=50, grid_file=grid_file))
    else:
      env_fn = lambda: gym.make(args.env)

    env = SubprocVecEnv([env_fn() for _ in range(12)])
    if not args.folder:
      args.folder = '/tmp'

    landmark_generator = None
    if args.landmark_training:
      landmark_generator = RandomLandmarkGenerator(100000, env_fn()())

    model = DDPG(
        env=env,
        policy=SimpleMlpPolicy,
        gamma=0.98,
        actor_lr=1e-3,
        critic_lr=args.critic_lr,
        learning_starts=2500,
        joint_feature_extractor=None,
        joint_goal_feature_extractor=None,
        clip_value_fn_range=(0.,1.),
        landmark_training=args.landmark_training,
        landmark_mode=args.landmark_mode,
        landmark_training_per_batch=args.landmark_k,
        landmark_width=args.landmark_w,
        landmark_generator=landmark_generator,
        train_freq=args.train_freq,
        target_network_update_frac=0.01,
        target_network_update_freq=args.train_freq * 10,
        epsilon_random_exploration=args.eexplore,
        action_noise=args.action_noise,
        critic_l2_regularization=0.,
        action_l2_regularization=args.action_l2,
        verbose=1,
        batch_size=240,
        buffer_size=1000000,
        hindsight_mode=args.her,
        tensorboard_log="{}/ler_new/".format(args.folder),
    )
    model.learn(total_timesteps=args.max_timesteps, tb_log_name="DDPG_{}_landmark-{}_{}_k-{}_w-{}_crlr-{}_tf-{}_al2-{}_{}_eps-{}_{}".format(args.env, 
      args.landmark_training, args.landmark_mode, args.landmark_k, args.landmark_w, args.critic_lr, args.train_freq, args.action_l2, args.action_noise, args.eexplore, args.tb), log_interval=100)

    model_filename = "ddpg_model_{}_{}_landmark-{}_{}_k-{}_w-{}_crlr-{}_tf-{}_{}_{}_{}.pkl".format(args.env, args.tb, args.landmark_training, args.landmark_mode, 
      args.landmark_k, args.landmark_w, args.critic_lr, args.train_freq, args.action_l2, args.action_noise, args.max_timesteps)
    print("Saving model to {}".format(model_filename))
    model.save(model_filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DDPG")
    parser.add_argument('--env', default="FetchReach-v1", type=str, help="Gym environment")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--her', default='none', type=str, help="Hindsight mode (e.g., future_4 or final)")
    parser.add_argument('--tb', default='1', type=str, help="Tensorboard_name")
    parser.add_argument('--folder', default='/scratch/gobi1/spitis/tb/', type=str, help="Tensorboard_folder")
    parser.add_argument('--room-file', default='room_5x5_empty', type=str,
                        help="Room type: room_5x5_empty (default), 2_room_9x9")
    parser.add_argument('--action_l2', default=5e-3, type=float, help="action l2 norm")
    parser.add_argument('--action_noise', default='normal_0.2', type=str, help="action noise")
    parser.add_argument('--eexplore', default=0.3, type=float, help="epsilon exploration")
    parser.add_argument('--landmark_training', default=0., type=float, help='landmark training coefficient')
    parser.add_argument('--landmark_mode', default='unidirectional', type=str, help='landmark training coefficient')
    parser.add_argument('--landmark_k', default=1, type=int, help='number of landmark trainings per batch')
    parser.add_argument('--landmark_w', default=1, type=int, help='number of steps landmarks can take')
    parser.add_argument('--train_freq', default=10, type=int, help='how often to train')
    parser.add_argument('--critic_lr', default=1e-3, type=float, help='critic_learning_rate')
    args = parser.parse_args()
    main(args)
