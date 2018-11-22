import argparse

import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
from envs.custom_fetch import CustomFetchReachEnv, CustomFetchPushEnv, CustomFetchPushEnv6DimGoal
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
                                         layers=[400,400,400],
                                         layer_norm=True,
                                         activ=tf.nn.relu,
                                         feature_extraction="mlp")

def fetch_reach_goal_to_prototype(batched_goals_tensor):
  return tf.pad(batched_goals_tensor, [[0,0],[0,7]])

def fetch_push_goal_to_prototype(batched_goals_tensor):
  tiled = tf.tile(batched_goals_tensor, [1, 2])
  tiled = tf.pad(tiled, [[0,0],[0,19]])
  return tiled

def main(args):
    """
    Train and save the DDPG model, for the FetchReach problem

    :param args: (ArgumentParser) the input arguments100000
    """
    goal_to_prototype = None

    if args.env == "CustomFetchReach":
      env = SubprocVecEnv([lambda: CustomFetchReachEnv(threshold_scale=np.sqrt(args.threshold_scale)) for _ in range(12)])
      goal_to_prototype = fetch_reach_goal_to_prototype
    elif args.env == "CustomFetchPush6Dim":
      env = SubprocVecEnv([lambda: CustomFetchPushEnv6DimGoal(threshold_scale=np.sqrt(args.threshold_scale)) for _ in range(12)])
      goal_to_prototype = fetch_push_goal_to_prototype
    elif args.env == "CustomFetchPush":
      env = SubprocVecEnv([lambda: CustomFetchPushEnv(threshold_scale=np.sqrt(args.threshold_scale)) for _ in range(12)])
      goal_to_prototype = fetch_push_goal_to_prototype
    elif "GoalGrid" in args.env:
      grid_file = "{}.txt".format(args.room_file)
      env = SubprocVecEnv([lambda: discrete_to_box_wrapper(GoalGridWorldEnv(grid_size=5, max_step=50, grid_file=grid_file)) for _ in range(12)])
    else:
      env = SubprocVecEnv([lambda: gym.make(args.env) for _ in range(12)])

    if not args.folder:
      args.folder = '/tmp'

    if not args.g2p:
      goal_to_prototype = None
    else:
      print('Using G2P:', goal_to_prototype)

    model = DDPG(
        env=env,
        policy=SimpleMlpPolicy,
        gamma=0.98,
        actor_lr=1e-3,
        critic_lr=1e-3,
        learning_starts=2500,
        joint_feature_extractor=None,
        joint_goal_feature_extractor=None,
        clip_value_fn_range=(0.,1.),
        goal_to_prototype_state=goal_to_prototype,
        landmark_training=args.landmark_training,
        train_freq=args.train_freq,
        target_network_update_frac=0.02,
        target_network_update_freq=40,
        epsilon_random_exploration=args.eexplore,
        action_noise=args.action_noise,
        critic_l2_regularization=0.,
        action_l2_regularization=args.action_l2,
        verbose=1,
        batch_size=480,
        buffer_size=2000000,
        hindsight_mode=args.her,
        tensorboard_log="{}/ddpg_tensorboard/".format(args.folder),
    )
    model.learn(total_timesteps=args.max_timesteps, tb_log_name="DDPG_{}_landmark-{}_tf-{}_al2-{}_{}_eps-{}_{}".format(args.env, args.landmark_training, args.train_freq, args.action_l2, args.action_noise, args.eexplore, args.tb), log_interval=10)

    model_filename = "ddpg_model_{}_landmark-{}_tf-{}_{}_{}_{}.pkl".format(args.env, args.landmark_training, args.train_freq, args.action_l2, args.action_noise, args.max_timesteps)
    print("Saving model to {}".format(model_filename))
    model.save(model_filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DDPG")
    parser.add_argument('--env', default="FetchReach-v1", type=str, help="Gym environment")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--her', default='none', type=str, help="Hindsight mode (e.g., future_4 or final)")
    parser.add_argument('--tb', default='1', type=str, help="Tensorboard_name")
    parser.add_argument('--folder', default='/h/spitis/tmp/', type=str, help="Tensorboard_folder")
    parser.add_argument('--room-file', default='room_5x5_empty', type=str,
                        help="Room type: room_5x5_empty (default), 2_room_9x9")
    parser.add_argument('--g2p', dest='g2p', action='store_true')
    parser.add_argument('--threshold_scale', default=1., type=float, help="squared scale for compute_reward threshhold with operating on states")
    parser.add_argument('--action_l2', default=5e-3, type=float, help="action l2 norm")
    parser.add_argument('--action_noise', default='ou_0.2', type=str, help="action noise")
    parser.add_argument('--eexplore', default=0.2, type=float, help="epsilon exploration")
    parser.add_argument('--landmark_training', default=0., type=float, help='landmark training coefficient')
    parser.add_argument('--train_freq', default=10, type=int, help='how often to train')
    parser.set_defaults(g2p=False)
    args = parser.parse_args()
    main(args)
