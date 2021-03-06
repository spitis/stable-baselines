import argparse

import gym
from gym import wrappers
from envs.custom_fetch import CustomFetchReachEnv, CustomFetchPushEnv, CustomFetchPushEnv6DimGoal, CustomFetchSlideEnv, CustomFetchSlideEnv9DimGoal

from stable_baselines.simple_ddpg import SimpleDDPG as DDPG

def main(args):
    """
    Run a trained DDPG model

    :param args: (ArgumentParser) the input arguments
    """

    if args.env == "CustomFetchReach":
      env = CustomFetchReachEnv()
    elif args.env == "CustomFetchPush6Dim":
      env = CustomFetchPushEnv6DimGoal()
    elif args.env == "CustomFetchPush":
      env = CustomFetchPushEnv()
    elif args.env == "CustomFetchSlide":
      env = CustomFetchSlideEnv()
    elif args.env == "CustomFetchSlide9Dim":
      env = CustomFetchSlideEnv9DimGoal()
    else:
      env = gym.make(args.env)
    model_filename = args.checkpoint
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
    parser.add_argument('--checkpoint', default="", type=str, help="Checkpoint file to load")
    parser.add_argument('--no-render', default=False, action="store_true", help="Disable rendering")
    args = parser.parse_args()
    main(args)