import numpy as np

from gym.envs.robotics.fetch_env import FetchEnv, goal_distance
from gym.envs.robotics.fetch.slide import FetchSlideEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch.push import FetchPushEnv
from gym.envs.robotics.fetch.reach import FetchReachEnv


class CustomFetchReachEnv(FetchReachEnv):

  def __init__(self, max_step=50):
    super().__init__(reward_type='sparse')
    self.max_step = max_step
    self.num_step = 0

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.
    d = goal_distance(achieved_goal, goal)
    return (d <= self.distance_threshold).astype(np.float32)

  def step(self, action):
    action = np.clip(action, self.action_space.low, self.action_space.high)
    self._set_action(action)
    self.sim.step()
    self._step_callback()
    self.num_step += 1
    obs = self._get_obs()

    reward = self.compute_reward(obs['achieved_goal'], self.goal, None)
    done = 1. if self.num_step >= self.max_step else reward
    info = None

    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs



class CustomFetchPushEnv(FetchPushEnv):

  def __init__(self, max_step=50, reward_scale = 1.):
    super().__init__(reward_type='sparse')
    self.max_step = max_step
    self.num_step = 0
    self.reward_scale = reward_scale

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.
    d = goal_distance(achieved_goal, goal)
    return (d <= self.distance_threshold).astype(np.float32) * self.reward_scale

  def step(self, action):
    action = np.clip(action, self.action_space.low, self.action_space.high)
    self._set_action(action)
    self.sim.step()
    self._step_callback()
    self.num_step += 1
    obs = self._get_obs()

    reward = self.compute_reward(obs['achieved_goal'], self.goal, None)
    done = 1. if self.num_step >= self.max_step else reward.astype(np.bool).astype(np.float32)
    info = None

    return obs, reward, done, info

  def reset(self):
    obs = super().reset()
    self.num_step = 0
    return obs