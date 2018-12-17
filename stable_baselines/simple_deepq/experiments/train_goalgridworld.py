"""
train_goalgridworld.py

Example usage:

python train_goalgridworld.py --model-type cnn --max-timestep 5000 --her future_4 --room-file room_5x5_empty

python train_goalgridworld.py --model-type cnn --max-timestep 10000 --her future_4 --room-file 2_room_9x9 --landmark-training 0.01

python train_goalgridworld.py --model-type cnn --max-timestep 40000 --her future_4 --room-file 4_room_13x13_outerwalls --landmark-training 0.05 --landmark_gen scorevae_30000 --refine_vae 1
"""
import argparse

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import conv_to_fc

from stable_baselines.simple_deepq import SimpleDQN as DQN
from stable_baselines.common.policies import FeedForwardPolicy
from stable_baselines.common.vec_env import SubprocVecEnv
from stable_baselines.common.landmark_generator import RandomLandmarkGenerator, NearestNeighborLandmarkGenerator, \
                                              NonScoreBasedImageVAEWithNNRefinement, ScoreBasedImageVAEWithNNRefinement
from envs import discrete_to_box_wrapper
from envs.goal_grid import GoalGridWorldEnv
from stable_baselines.common import set_global_seeds

def gridworld_cnn(scaled_images, **kwargs):
  """
    CNN for grid world.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
  activ = tf.nn.relu
  h = tf.layers.conv2d(scaled_images, filters=32, kernel_size=5, padding='SAME', activation=activ)
  h = tf.layers.conv2d(h, filters=64, kernel_size=5, padding='SAME', activation=activ)
  return conv_to_fc(h)

class GridWorldCnnPolicy(FeedForwardPolicy):
  """
    Policy object that implements actor critic, using a CNN
  """

  def __init__(self, *args, **kwargs):
      super(GridWorldCnnPolicy, self).__init__(*args, **kwargs,
                                    feature_extraction="cnn",
                                    cnn_extractor=gridworld_cnn, layers=[256])


class GridWorldMlpPolicy(FeedForwardPolicy):
  """
    Policy object that implements actor critic, using a MLP (3 layers of 64)
  """
  def __init__(self, *args, **kwargs):
      super(GridWorldMlpPolicy, self).__init__(*args, **kwargs,
                                         layers=[400,400],
                                         layer_norm=True,
                                         feature_extraction="mlp")

def main(args):
    """
    Train and save the DQN model, for the cartpole problem

    :param args: (ArgumentParser) the input arguments100000
    """

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

    grid_file = "{}.txt".format(args.room_file)
    # env = GoalGridWorldEnv(grid_size=5, max_step=40, grid_file=grid_file)
    env_fn = lambda: GoalGridWorldEnv(grid_size=5, max_step=32, grid_file=grid_file)

    env = SubprocVecEnv([make_env(env_fn, i) for i in range(12)])

    if args.model_type == "mlp":
        policy = GridWorldMlpPolicy
    elif args.model_type == "cnn":
        policy = GridWorldCnnPolicy

    landmark_generator = None
    if args.landmark_training:
      if args.landmark_gen == 'random':
        landmark_generator = RandomLandmarkGenerator(100000, make_env(env_fn, 1137)())
      elif args.landmark_gen == 'nn':
        landmark_generator = NearestNeighborLandmarkGenerator(100000, make_env(env_fn, 1137)())
      elif 'lervae' in args.landmark_gen:
        landmark_generator = NonScoreBasedImageVAEWithNNRefinement(int(args.landmark_gen.split('_')[1]), make_env(env_fn, 1137)(), refine_with_NN=args.refine_vae)
      elif 'scorevae' in args.landmark_gen:
        landmark_generator = ScoreBasedImageVAEWithNNRefinement(int(args.landmark_gen.split('_')[1]), make_env(env_fn, 1137)(), refine_with_NN=args.refine_vae)
      else:
        raise ValueError("Unsupported landmark_gen")

    model = DQN(
        env=env,
        policy=policy,
        gamma=args.gamma,
        learning_rate=1e-3,
        learning_starts=500,
        verbose=1,
        train_freq=args.train_freq,
        batch_size=128,
        buffer_size=200000,
        exploration_fraction=0.8,
        exploration_final_eps=0.05,
        target_network_update_frac=0.02,
        target_network_update_freq=20,
        hindsight_mode=args.her,
        hindsight_frac=0.8,
        landmark_training=args.landmark_training,
        landmark_mode=args.landmark_mode,
        landmark_training_per_batch=args.landmark_k,
        landmark_width=args.landmark_w,
        landmark_generator=landmark_generator,
        landmark_error=args.landmark_error,
        tensorboard_log="./dqn_goalgridworld_tensorboard/FinalReport/{}/".format(args.room_file),
        eval_env=GoalGridWorldEnv(grid_size=5, max_step=40, grid_file=grid_file),
        eval_every=10,
    )
    assert model.goal_space is not None
    model.learn(total_timesteps=args.max_timesteps, tb_log_name="DQN_her-{}_landmark-{}_generator-{}_seed-{}_{}".format(
                      args.her, args.landmark_training, args.landmark_gen, args.seed, args.tb))

    model_filename = "FinalReport/goalgridworld_model_model-{}_timestep-{}_room-{}_her-{}".format(args.model_type,
                                                        args.max_timesteps, args.room_file, args.her)
    if args.landmark_training > 0:
      model_filename += "_landmark-{}_generator-{}_refinevae-{}".format(args.landmark_training, 
                                                        args.landmark_gen,
                                                        args.refine_vae)
    model_filename += "_seed-{}.pkl".format(args.seed)

    print("Saving model to {}".format(model_filename))
    model.save(model_filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on goal Gridworld")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--model-type', default='mlp', type=str, help="Model type: cnn, mlp (default)")
    parser.add_argument('--her', default='none', type=str, help="HER type: final, future_k, none (default)")
    parser.add_argument('--room-file', default='room_5x5_empty', type=str,
                        help="Room type: room_5x5_empty (default), 2_room_9x9, 4_room_13x13_outerwalls")
    parser.add_argument('--landmark-training', default=0., type=float, help='landmark training coefficient')
    parser.add_argument('--landmark_mode', default='bidirectional', type=str, help='landmark training coefficient')
    parser.add_argument('--landmark_k', default=1, type=int, help='number of landmark trainings per batch')
    parser.add_argument('--landmark_w', default=1, type=int, help='number of steps landmarks can take')
    parser.add_argument('--landmark_gen', default='random', type=str, help='landmark generator to use')
    parser.add_argument('--landmark_error', default='linear', type=str, help='landmark error type (linear or squared)')
    parser.add_argument('--refine_vae', default=False, type=bool, help='use nearest neighbor to refine VAE')
    parser.add_argument('--train-freq', default=10, type=int, help='how often to train')
    parser.add_argument('--tb', default='1', type=str, help="Tensorboard_name")
    parser.add_argument('--gamma', default=0.97, type=float, help='discount factors')
    parser.add_argument('--seed', default=0, type=int, help='random seed')
    args = parser.parse_args()
    main(args)
