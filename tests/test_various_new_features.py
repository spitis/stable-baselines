import numpy as np
import gym
from gym import spaces
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines.common import set_global_seeds

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.
    
    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environment you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env = gym.make(env_id)
        env.seed(seed + rank)
        return env
    set_global_seeds(seed)
    return _init

def test_replay_buffer_add_batch():
    """
    test the replay buffer's add_batch functionality
    """
    pass


def test_vecenv_with_goalenvs():
  """
  test that VecEnv works with goal-environments (see https://blog.openai.com/ingredients-for-robotics-research/)
  """
  env_id = "FetchReach-v1"
  env = make_env(env_id, 0)()

  obs_space = env.observation_space
  assert isinstance(obs_space, spaces.Dict)

  num_cpu = 3
  for fn in [SubprocVecEnv, DummyVecEnv]:
    vecenv = fn([make_env(env_id, i) for i in range(num_cpu)])
    obs = vecenv.reset()
    assert isinstance(obs, dict), "vectorized goal envs should produce dict observations!"
    for key in ['observation', 'desired_goal', 'achieved_goal']:
      assert key in obs, "observation dictionary is missing the {} key!".format(key)
    
    assert obs['observation'].shape[0] == num_cpu, "first dim of obs should be batch_size = num_envs"
    assert obs['observation'].shape[1:] == obs_space.spaces['observation'].shape, "obs has incorrect shape!"
  
    assert vecenv.compute_reward == env.compute_reward, "vecenv should inherit the compute reward fn!"