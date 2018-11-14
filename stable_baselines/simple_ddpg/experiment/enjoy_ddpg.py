import argparse

import gym
from gym import wrappers
from envs.custom_fetch import CustomFetchReachEnv, CustomFetchPushEnv

from stable_baselines.simple_ddpg import SimpleDDPG as DDPG

def main(args):
    """
    Run a trained DDPG model

    :param args: (ArgumentParser) the input arguments
    """
    if args.env == "FetchReach-v1":
      env = CustomFetchReachEnv()
    elif "FetchPush" in args.env:
      env = CustomFetchPushEnv()
    else:
      env = gym.make(args.env)
    model_filename = "ddpg_model_{}_{}.pkl".format(args.env, args.max_timesteps)
    model = DDPG.load(model_filename, env)

    max_num_eps = 20
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
    parser = argparse.ArgumentParser(description="Enjoy trained DDPG")
    parser.add_argument('--env', default="FetchReach-v1", type=str, help="Gym environment")
    parser.add_argument('--no-render', default=False, action="store_true", help="Disable rendering")
    parser.add_argument('--max-timesteps', default=1000, type=int, help="Maximum number of timesteps")
    args = parser.parse_args()
    main(args)