import numpy as np

from gym.envs.robotics.fetch_env import FetchEnv, goal_distance
from gym.envs.robotics.fetch.slide import FetchSlideEnv
from gym.envs.robotics.fetch.pick_and_place import FetchPickAndPlaceEnv
from gym.envs.robotics.fetch.push import FetchPushEnv
from gym.envs.robotics.fetch.reach import FetchReachEnv

from gym.envs.robotics import rotations, utils


class CustomFetchReachEnv(FetchReachEnv):

  def __init__(self, max_step=50):
    super().__init__(reward_type='sparse')
    self.max_step = max_step
    self.num_step = 0

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.
    d = goal_distance(achieved_goal, goal)
    return (d <= self.distance_threshold).astype(np.float32)

  def goal_extraction_function(self, batched_states):
    return batched_states[:,:3]

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

  def goal_extraction_function(self, batched_states):
    return batched_states[:,3:6]

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

class CustomFetchPushEnv6DimGoal(FetchPushEnv):

  def __init__(self, max_step=50):
    super().__init__(reward_type='sparse')
    self.max_step = max_step
    self.num_step = 0

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.
    d = goal_distance(achieved_goal, goal)
    return (d <= self.distance_threshold * np.sqrt(2.)).astype(np.float32)

  def goal_extraction_function(self, batched_states):
    return batched_states[:,3:9]

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    if self.has_object:
      object_pos = self.sim.data.get_site_xpos('object0')
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
      # velocities
      object_velp = self.sim.data.get_site_xvelp('object0') * dt
      object_velr = self.sim.data.get_site_xvelr('object0') * dt
      # gripper state
      object_rel_pos = object_pos - grip_pos
      object_velp -= grip_velp
    else:
      object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if not self.has_object:
      achieved_goal = grip_pos.copy()
    else:
      achieved_goal = np.concatenate([np.squeeze(object_pos), np.squeeze(object_rel_pos)])

    obs = np.concatenate([
      grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
      object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
    ])

    return {
      'observation': obs.copy(),
      'achieved_goal': achieved_goal.copy(),
      'desired_goal': self.goal.copy(),
    }

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')
    self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
    self.sim.forward()

  def _sample_goal(self):
    if self.has_object:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
      goal += self.target_offset
      goal[2] = self.height_offset
      if self.target_in_the_air and self.np_random.uniform() < 0.5:
        goal[2] += self.np_random.uniform(0, 0.45)
    else:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    return np.concatenate([goal, [0., 0., -0.05]])

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



class CustomFetchSlideEnv(FetchSlideEnv):

  def __init__(self, max_step=70, pos_threshold = 0.07, speed_threshold = 0.05):
    super().__init__(reward_type='sparse')
    self.max_step = max_step
    self.num_step = 0
    self.pos_threshold = pos_threshold
    self.speed_threshold = speed_threshold

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.
    pos = goal_distance(achieved_goal[:3], goal[:3])
    speed = goal_distance(achieved_goal[3:6], goal[3:6])
    return ((pos <= self.pos_threshold) and (speed <= self.speed_threshold)).astype(np.float32)

  def goal_extraction_function(self, batched_states):
    return batched_states[:,[3,4,5,14,15,16]]

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

    
  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    if self.has_object:
      object_pos = self.sim.data.get_site_xpos('object0')
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
      # velocities
      object_velp = self.sim.data.get_site_xvelp('object0') * dt
      object_velr = self.sim.data.get_site_xvelr('object0') * dt
      # gripper state
      object_rel_pos = object_pos - grip_pos
      object_velp -= grip_velp
    else:
      object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if not self.has_object:
      achieved_goal = grip_pos.copy()
    else:
      achieved_goal = np.concatenate([np.squeeze(object_pos), np.squeeze(object_velp)])

    obs = np.concatenate([
      grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
      object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
    ])

    return {
      'observation': obs.copy(),
      'achieved_goal': achieved_goal.copy(),
      'desired_goal': self.goal.copy(),
    }

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')
    self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
    self.sim.forward()

  def _sample_goal(self):
    if self.has_object:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
      goal += self.target_offset
      goal[2] = self.height_offset
      if self.target_in_the_air and self.np_random.uniform() < 0.5:
        goal[2] += self.np_random.uniform(0, 0.45)
    else:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    return np.concatenate([goal, [0., 0., 0.]])


class CustomFetchSlideEnv9DimGoal(FetchSlideEnv):

  def __init__(self, max_step=70, pos_threshold = 0.07, speed_threshold = 0.05):
    super().__init__(reward_type='sparse')
    self.max_step = max_step
    self.num_step = 0
    self.pos_threshold = pos_threshold
    self.speed_threshold = speed_threshold

  def compute_reward(self, achieved_goal, goal, info):
    # Compute distance between goal and the achieved goal.
    pos = goal_distance(achieved_goal[:6], goal[:6])
    speed = goal_distance(achieved_goal[6:9], goal[6:9])
    return ((pos <= self.pos_threshold * np.sqrt(2)) and (speed <= self.speed_threshold)).astype(np.float32)

  def goal_extraction_function(self, batched_states):
    return batched_states[:,[3,4,5,0,1,2,14,15,16]]

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

  def _get_obs(self):
    # positions
    grip_pos = self.sim.data.get_site_xpos('robot0:grip')
    dt = self.sim.nsubsteps * self.sim.model.opt.timestep
    grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
    robot_qpos, robot_qvel = utils.robot_get_obs(self.sim)
    if self.has_object:
      object_pos = self.sim.data.get_site_xpos('object0')
      # rotations
      object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
      # velocities
      object_velp = self.sim.data.get_site_xvelp('object0') * dt
      object_velr = self.sim.data.get_site_xvelr('object0') * dt
      # gripper state
      object_rel_pos = object_pos - grip_pos
      object_velp -= grip_velp
    else:
      object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
    gripper_state = robot_qpos[-2:]
    gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

    if not self.has_object:
      achieved_goal = grip_pos.copy()
    else:
      achieved_goal = np.concatenate([np.squeeze(object_pos), grip_pos, np.squeeze(object_velp)])

    obs = np.concatenate([
      grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
      object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
    ])

    return {
      'observation': obs.copy(),
      'achieved_goal': achieved_goal.copy(),
      'desired_goal': self.goal.copy(),
    }

  def _render_callback(self):
    # Visualize target.
    sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    site_id = self.sim.model.site_name2id('target0')
    self.sim.model.site_pos[site_id] = self.goal[:3] - sites_offset[0]
    self.sim.forward()

  def _sample_goal(self):
    if self.has_object:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
      goal += self.target_offset
      goal[2] = self.height_offset
      if self.target_in_the_air and self.np_random.uniform() < 0.5:
        goal[2] += self.np_random.uniform(0, 0.45)
    else:
      goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-0.15, 0.15, size=3)
    return np.concatenate([goal, self.initial_gripper_xpos[:3], [0., 0., 0.]])

