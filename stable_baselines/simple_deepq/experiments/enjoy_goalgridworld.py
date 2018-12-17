"""
enjoy_goalgridworld.py

Example usage:

python enjoy_goalgridworld.py --model-type cnn --max-timestep 5000 --room-file room_5x5_empty

"""

import argparse

import gym

from stable_baselines.simple_deepq import SimpleDQN as DQN
from stable_baselines.deepq import MlpPolicy
from train_goalgridworld import GridWorldCnnPolicy
from envs.goal_grid import GoalGridWorldEnv
from stable_baselines.common.landmark_generator import RandomLandmarkGenerator, NearestNeighborLandmarkGenerator, \
                                            NonScoreBasedImageVAEWithNNRefinement, ScoreBasedImageVAEWithNNRefinement
from stable_baselines.common import set_global_seeds

def main(args):
    """
    Run a trained model for the goal grid world problem

    :param args: (ArgumentParser) the input arguments
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
    env = GoalGridWorldEnv(grid_size=9, max_step=32, grid_file=grid_file)
    env_fn = lambda: env

    model_filename = "FinalReport/goalgridworld_model_model-{}_timestep-{}_room-{}_her-{}".format(args.model_type,
                                                        args.max_timesteps, args.room_file, args.her)
    if args.landmark_training > 0:
      model_filename += "_landmark-{}_generator-{}_refinevae-{}".format(args.landmark_training, 
                                                        args.landmark_gen,
                                                        args.refine_vae)
    model_filename += "_seed-{}.pkl".format(args.seed)
    model = DQN.load(model_filename, env, goal_space=env.observation_space.spaces['desired_goal'])

    # Load the landmark generator, if applicable, separately
    if args.landmark_training and args.landmark_gen != "random":
        if "lervae" in args.landmark_gen:
            landmark_generator = NonScoreBasedImageVAEWithNNRefinement(int(args.landmark_gen.split('_')[1]), make_env(env_fn, 1137)(), refine_with_NN=args.refine_vae)
        elif "scorevae" in args.landmark_gen:
            landmark_generator = ScoreBasedImageVAEWithNNRefinement(int(args.landmark_gen.split('_')[1]), make_env(env_fn, 1137)(), refine_with_NN=args.refine_vae)

        model.landmark_generator = landmark_generator.load(model_filename + ".gen")

    max_num_eps = 100
    num_eps = 0
    num_success = 0

    while (num_eps < max_num_eps):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if not args.no_render:
                env.render(mode="RGB") # When the number of objects is 3 (agent, empty, wall)
            action, _ = model.predict(obs)
            values, _ = model.get_value(obs) # Get values of each action
            landmarks = model.landmark_generator.generate(obs['observation'], np.array([action]), obs['desired_goal'])
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)
        num_eps += 1
        num_success += episode_rew

    print("Average success rate: {}".format(num_success/max_num_eps))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enjoy trained DQN on Goal GridWorld")
    parser.add_argument('--no-render', default=False, action="store_true", help="Disable rendering")
    parser.add_argument('--max-timesteps', default=1000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--model-type', default='mlp', type=str, help="Model type: cnn, mlp (default)")
    parser.add_argument('--room-file', default='room_5x5_empty', type=str,
                        help="Room type: room_5x5_empty (default), 2_room_9x9")
    parser.add_argument('--her', default='none', type=str, help="HER type: final, future_k, none (default)")
    parser.add_argument('--landmark-training', default=0., type=float, help='landmark training coefficient')
    parser.add_argument('--landmark_gen', default='random', type=str, help='landmark generator to use')
    parser.add_argument('--refine_vae', default=False, type=bool, help='use nearest neighbor to refine VAE')
    parser.add_argument('--seed', default=0, type=int, help='random seed')

    args = parser.parse_args()
    main(args)
