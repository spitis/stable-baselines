import argparse

import numpy as np
import tensorflow as tf
import gym
import os

from gym import wrappers
from envs.custom_fetch import CustomFetchReachEnv, CustomFetchPushEnv, CustomFetchPushEnv6DimGoal,\
  CustomFetchSlideEnv, CustomFetchSlideEnv9DimGoal, CustomFetchPushEnvMultiDimGoal
from envs import discrete_to_box_wrapper
from envs.goal_grid import GoalGridWorldEnv

from stable_baselines.simple_ddpg import SimpleDDPG as DDPG, make_feedforward_extractor, identity_extractor
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines.common.landmark_generator import RandomLandmarkGenerator, NearestNeighborLandmarkGenerator, NonScoreBasedVAEWithNNRefinement,\
  ScoreBasedVAEWithNNRefinement, FetchPushHeuristicGenerator
from stable_baselines.common import set_global_seeds

nonsaturating = False

class SimpleMlpPolicy(FeedForwardPolicy):
  """
    Policy object that implements actor critic, using a MLP (2 layers of 256)
  """
  def __init__(self, *args, **kwargs):
      global nonsaturating
      super(SimpleMlpPolicy, self).__init__(*args, **kwargs,
                                      layers=[400,400],
                                      layer_norm=True,
                                      activ=tf.nn.relu,
                                      feature_extraction="mlp",
                                      use_non_saturating_activation=nonsaturating)

def main(args):
    """
    Train and save the DDPG model, for the FetchReach problem

    :param args: (ArgumentParser) the input arguments100000
    """


    global nonsaturating
    nonsaturating = args.nonsaturating

    def make_env(env_fn, rank, seed=args.seed):
        """
        Utility function for multiprocessed env.
        
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = env_fn()
            env.seed(seed + rank)
            return env
        set_global_seeds(seed)
        return _init
    
    if args.env == "CustomFetchReach":
      env_fn = lambda: CustomFetchReachEnv()
    elif args.env == "CustomFetchPush6Dim":
      env_fn = lambda: CustomFetchPushEnv6DimGoal()
    elif args.env == "CustomFetchPush":
      env_fn = lambda: CustomFetchPushEnv()
    elif "CustomFetchPush_" in args.env:
      k = int(args.env.split('_')[1])
      env_fn = lambda: CustomFetchPushEnvMultiDimGoal(goal_dims = k)
    elif args.env == "CustomFetchSlide9Dim":
      env_fn = lambda: CustomFetchSlideEnv9DimGoal()
    elif args.env == "CustomFetchSlide":
      env_fn = lambda: CustomFetchSlideEnv()
    elif "GoalGrid" in args.env:
      grid_file = "{}.txt".format(args.room_file)
      env_fn = discrete_to_box_wrapper(GoalGridWorldEnv(grid_size=5, max_step=50, grid_file=grid_file))
    else:
      env_fn = lambda: gym.make(args.env)

    env = SubprocVecEnv([make_env(env_fn, i) for i in range(12)])

    landmark_generator = None
    if args.landmark_training:
      if args.landmark_gen == 'random':
        landmark_generator = RandomLandmarkGenerator(100000, make_env(env_fn, 1137)())
      elif args.landmark_gen == 'nn':
        landmark_generator = NearestNeighborLandmarkGenerator(100000, make_env(env_fn, 1137)())
      elif 'lervae' in args.landmark_gen:
        landmark_generator = NonScoreBasedVAEWithNNRefinement(int(args.landmark_gen.split('_')[1]), make_env(env_fn, 1137)(), refine_with_NN=args.refine_vae)
      elif 'scorevae' in args.landmark_gen:
        landmark_generator = ScoreBasedVAEWithNNRefinement(int(args.landmark_gen.split('_')[1]), make_env(env_fn, 1137)(), refine_with_NN=args.refine_vae)
      elif 'heur' in args.landmark_gen:
        landmark_generator = FetchPushHeuristicGenerator(100000, make_env(env_fn, 1137)())
      else:
        raise ValueError("Unsupported landmark_gen")

    clip_value = tuple([float(i) for i in args.clip_values.replace('neg','-').split('_')])

    model = DDPG(
        env=env,
        policy=SimpleMlpPolicy,
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        learning_starts=2500,
        joint_feature_extractor=None,
        joint_goal_feature_extractor=None,
        clip_value_fn_range=clip_value,
        landmark_training=args.landmark_training,
        landmark_mode=args.landmark_mode,
        landmark_training_per_batch=args.landmark_k,
        landmark_width=args.landmark_w,
        landmark_generator=landmark_generator,
        landmark_error=args.landmark_error,
        train_freq=args.train_freq,
        target_network_update_frac=0.01,
        target_network_update_freq=args.train_freq * args.update_factor,
        epsilon_random_exploration=args.eexplore,
        action_noise=args.action_noise,
        critic_l2_regularization=args.critic_l2,
        action_l2_regularization=args.action_l2,
        verbose=1,
        batch_size=360,
        buffer_size=1000000,
        hindsight_mode=args.her,
        tensorboard_log=os.path.join(args.folder, 'ddpg'),
        eval_env=make_env(env_fn, 1138)(),
        eval_every=10,
    )

    model_name = "ddpg_model_{}_{}_{}_landmark-{}_{}_ref{}_nonsat{}_{}_{}_k-{}_w-{}_crlr-{}_tf-{}_{}_{}_{}_{}_seed-{}_tb-{}".format(args.env, args.her, args.tb, 
      args.landmark_training, args.landmark_gen, args.refine_vae, args.nonsaturating, args.landmark_error, args.landmark_mode, args.landmark_k, args.landmark_w, args.critic_lr, 
      args.train_freq, args.action_l2, args.action_noise, args.eexplore, args.max_timesteps, args.seed, args.tb)

    model.learn(total_timesteps=args.max_timesteps, tb_log_name=model_name, log_interval=50)




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
    parser.add_argument('--critic_l2', default=0., type=float, help="critic weights l2 penalty")
    parser.add_argument('--action_noise', default='ou_0.2', type=str, help="action noise")
    parser.add_argument('--eexplore', default=0.3, type=float, help="epsilon exploration")
    parser.add_argument('--landmark_training', default=0., type=float, help='landmark training coefficient')
    parser.add_argument('--landmark_mode', default='bidirectional', type=str, help='landmark training coefficient')
    parser.add_argument('--landmark_k', default=1, type=int, help='number of landmark trainings per batch')
    parser.add_argument('--landmark_w', default=1, type=int, help='number of steps landmarks can take')
    parser.add_argument('--landmark_gen', default='random', type=str, help='landmark generator to use')
    parser.add_argument('--landmark_error', default='linear', type=str, help='landmark error type (linear or squared)')
    parser.add_argument('--train_freq', default=10, type=int, help='how often to train')
    parser.add_argument('--update_factor', default=8, type=int, help='how often to update (in number training steps)')
    parser.add_argument('--critic_lr', default=1e-3, type=float, help='critic_learning_rate')
    parser.add_argument('--actor_lr', default=1e-3, type=float, help='actor_learning_rate')
    parser.add_argument('--refine_vae', default=False, type=bool, help='use nearest neighbor to refine VAE')
    parser.add_argument('--nonsaturating', default=False, type=bool, help='use non-saturating activation for policy')
    parser.add_argument('--clip_values', default='0._1.', type=str, help='range for value clipping (0, 1) defualt')
    parser.add_argument('--gamma', default=0.98, type=float, help='discount factorcx')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()
    main(args)
