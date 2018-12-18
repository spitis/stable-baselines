"""
enjoy_goalgridworld.py

Example usage:

python enjoy_goalgridworld.py --model-type cnn --max-timestep 5000 --room-file room_5x5_empty

"""

import argparse

import gym
import numpy as np

from stable_baselines.simple_deepq import SimpleDQN as DQN
from stable_baselines.deepq import MlpPolicy
from train_goalgridworld import GridWorldCnnPolicy
from envs.goal_grid import GoalGridWorldEnv
from stable_baselines.common.landmark_generator import RandomLandmarkGenerator, NearestNeighborLandmarkGenerator, NonScoreBasedImageVAEWithNNRefinement, ScoreBasedImageVAEWithNNRefinement
from stable_baselines.common import set_global_seeds

import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams.update({'font.size': 20})

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

    # Create a heatmap
    # Load the current map
    grid = env.grid
    num_objects = len(env.objects) # Number of channels per cell

    # landmark_scores = tf.clip_by_value(landmark_lower_bound / qa_tm1_g, 0., 1.05)

    # Decide on a starting state
    s_start = np.copy(grid)
    s_start[0,0] = 1 # Start is at top left corner
    
    # Decide on a final state
    s_goal = np.copy(grid)
    s_goal[-1,-1] = 1 # End is at the bottom right corner

    # Compute v_s_g
    v_s_g = []
    obs = {"observation":env.one_hot(s_start, num_objects), "desired_goal": env.one_hot(s_goal, num_objects)}
    v_s_g = model.get_value(obs)[0]
    v_s_g = max(v_s_g)

    # Manually create all the landmark location
    landmark_scores = np.zeros(grid.shape)
    landmark_ratios = np.zeros(grid.shape)
    for r in range(0, grid.shape[0]):
        for c in range(0, grid.shape[1]):
            # Use the current row, col to place the agent, if the cell value is 0
            if grid[r,c] == 0:
                curr_landmark = np.copy(grid)
                curr_landmark[r,c] = 1

                # Compute V(s0, 1) and V(l, g) 
                obs = {"observation": env.one_hot(s_start, num_objects),
                            "desired_goal": env.one_hot(curr_landmark, num_objects)}
                v_s_l = max(model.get_value(obs)[0])
                
                obs = {"observation": env.one_hot(curr_landmark, num_objects),
                            "desired_goal": env.one_hot(s_goal, num_objects)}
                v_l_g = max(model.get_value(obs)[0])

                landmark_scores[r,c] = (v_s_l*v_l_g*model.gamma)/v_s_g
                landmark_ratios[r,c] = np.log(v_s_l/v_l_g)

    # Visualize the start state
    plt.figure()
    plt.imshow(env.one_hot(s_start, num_objects))
    plt.title("Start State")
    plt.savefig("figs/2-room-9x9_start_state.pdf", format='pdf', dpi=100)

    # Visualize the goal state
    plt.figure()
    plt.imshow(env.one_hot(s_goal, num_objects))
    plt.title("Goal State")
    plt.savefig("figs/2-room-9x9_goal_state.pdf", format='pdf', dpi=100)

    # Visualize landmark_score heatmap
    print(landmark_scores)
    plt.imshow(landmark_scores)
    plt.colorbar()
    plt.title("Landmark Score")
    plt.savefig("figs/2-room-9x9_landmark-scores-heatmap.pdf", format='pdf', dpi=100)

    # Grayscale version
    plt.figure()
    plt.imshow(landmark_scores, cmap="gray")
    plt.colorbar()
    plt.title("Landmark Score")
    plt.savefig("figs/2-room-9x9_landmark-scores-heatmap_gray.pdf", format='pdf', dpi=100)

    # Visualize landmark_ratio heatmap
    plt.figure()
    plt.imshow(landmark_ratios)
    plt.colorbar()
    plt.title("Landmark Ratio")
    plt.savefig("figs/2-room-9x9_landmark-ratios-heatmap.pdf", format='pdf', dpi=100)

    # Grayscale version
    plt.figure()
    plt.imshow(landmark_ratios, cmap='gray')
    plt.colorbar()
    plt.title("Landmark Ratio")
    plt.savefig("figs/2-room-9x9_landmark-ratios-heatmap_gray.pdf", format='pdf', dpi=100)

    # TODO: Visualize the landmark generator?!
    # TODO: Visualize the landmark score/ratio for the ideal optimal value function?!


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
