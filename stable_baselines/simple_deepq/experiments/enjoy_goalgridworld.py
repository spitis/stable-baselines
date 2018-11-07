import argparse

import gym

from stable_baselines.simple_deepq import SimpleDQN as DQN
from stable_baselines.deepq import MlpPolicy
from train_goalgridworld import GridWorldCnnPolicy
from envs.goal_grid import GoalGridWorldEnv

def main(args):
    """
    Run a trained model for the goal grid world problem

    :param args: (ArgumentParser) the input arguments
    """
    grid_file = 'room_5x5_empty.txt'
    env = GoalGridWorldEnv(grid_size=5, max_step=12, grid_file=grid_file)
    model_filename = "goalgridworld_model_{}_{}.pkl".format(args.model_type, args.max_timesteps)
    model = DQN.load(model_filename, env, goal_space=env.observation_space.spaces['desired_goal'])

    max_num_eps = 10
    num_eps = 0
    num_success = 0
    while (num_eps < max_num_eps):
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            if not args.no_render:
                env.render()
            action, _ = model.predict(obs)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
        print("Episode reward", episode_rew)
        num_eps += 1
        num_success += episode_rew
        # No render is only used for automatic testing
        if args.no_render:
            break
    print("Average success rate: {}".format(num_success/max_num_eps))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Enjoy trained DQN on Goal GridWorld")
    parser.add_argument('--no-render', default=False, action="store_true", help="Disable rendering")
    parser.add_argument('--max-timesteps', default=1000, type=int, help="Maximum number of timesteps")
    parser.add_argument('--model-type', default='mlp', type=str, help="Model type: cnn, mlp (default)")
    args = parser.parse_args()
    main(args)
io