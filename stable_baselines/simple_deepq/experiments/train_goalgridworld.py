"""
train_goalgridworld.py

Example usage:

python train_goalgridworld.py --model-type cnn --max-timestep 5000 --her future_4 --room-file room_5x5_empty

"""
import argparse

import numpy as np
import tensorflow as tf

from stable_baselines.a2c.utils import conv_to_fc

from stable_baselines.simple_deepq import SimpleDQN as DQN
from stable_baselines.common.policies import FeedForwardPolicy

from envs.goal_grid import GoalGridWorldEnv

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
    grid_file = "{}.txt".format(args.room_file)
    env = GoalGridWorldEnv(grid_size=5, max_step=40, grid_file=grid_file)
    if args.model_type == "mlp":
        policy = GridWorldMlpPolicy
    elif args.model_type == "cnn":
        policy = GridWorldCnnPolicy

    model = DQN(
        env=env,
        policy=policy,
        learning_rate=1e-3,
        gamma=0.97,
        learning_starts=500,
        verbose=1,
        train_freq=5,
        batch_size=128,
        buffer_size=200000,
        exploration_fraction=0.8,
        exploration_final_eps=0.1,
        target_network_update_frac=0.02,
        target_network_update_freq=20,
        hindsight_mode=args.her,
        hindsight_frac=0.8,
        tensorboard_log="./dqn_goalgridworld_tensorboard/{}".format(args.room_file),
    )
    assert model.goal_space is not None
    model.learn(total_timesteps=args.max_timesteps, tb_log_name="DQN_{}".format(args.her))

    model_filename = "goalgridworld_model_{}_{}_{}.pkl".format(args.model_type, args.max_timesteps, args.room_file)
    print("Saving model to {}".format(model_filename))
    model.save(model_filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train DQN on goal Gridworld")
    parser.add_argument('--max-timesteps', default=100000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--model-type', default='mlp', type=str, help="Model type: cnn, mlp (default)")
    parser.add_argument('--her', default='none', type=str, help="HER type: final, future_k, none (default)")
    parser.add_argument('--room-file', default='room_5x5_empty', type=str,
                        help="Room type: room_5x5_empty (default), 2_room_9x9")
    args = parser.parse_args()
    main(args)
