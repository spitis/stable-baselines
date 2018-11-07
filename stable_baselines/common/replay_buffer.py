import numpy as np
from collections import OrderedDict
import multiprocessing as mp
from stable_baselines.common.vec_env import CloudpickleWrapper

def worker_init(compute_reward_fn_wrapper):
  global process_trajectory
  process_trajectory = compute_reward_fn_wrapper.var

def worker_fn(trajectory):
  global process_trajectory
  return process_trajectory(trajectory)

class RingBuffer(object):
  """This is a collections.deque in numpy, with pre-allocated memory"""

  def __init__(self, maxlen, shape, dtype='float32'):
    """
    A buffer object, when full restarts at the initial position

    :param maxlen: (int) the max number of numpy objects to store
    :param shape: (tuple) the shape of the numpy objects you want to store
    :param dtype: (str) the name of the type of the numpy object you want to store
    """
    self.maxlen = maxlen
    self.start = 0
    self.length = 0
    self.shape = shape
    self.data = np.zeros((maxlen, ) + shape).astype(dtype)

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    if idx < 0 or idx >= self.length:
      raise KeyError()
    return self.data[(self.start + idx) % self.maxlen]

  def get_batch(self, idxs):
    """
    get the value at the indexes

    :param idxs: (int or numpy int) the indexes
    :return: (np.ndarray) the stored information in the buffer at the asked positions
    """
    return self.data[(self.start + idxs) % self.length]

  def append(self, var):
    """
    Append an object to the buffer

    :param var: (np.ndarray) the object you wish to add
    """
    if self.length < self.maxlen:
      # We have space, simply increase the length.
      self.length += 1
    elif self.length == self.maxlen:
      # No space, "remove" the first item.
      self.start = (self.start + 1) % self.maxlen
    else:
      # This should never happen.
      raise RuntimeError()

    self.data[(self.start + self.length - 1) % self.maxlen] = var

  def _append_batch_with_space(self, var):
    """
    Append a batch of objects to the buffer, *assuming* there is space.

    :param var: (np.ndarray) the batched objects you wish to add
    """
    len_batch = len(var)
    start_pos = (self.start + self.length) % self.maxlen

    self.data[start_pos : start_pos + len_batch] = var
    
    if self.length < self.maxlen:
      self.length += len_batch
      assert self.length <= self.maxlen, "this should never happen!"
    else:
      self.start = (self.start + len_batch) % self.maxlen
  
  def append_batch(self, var):
    """
    Append a batch of objects to the buffer.

    :param var: (np.ndarray) the batched objects you wish to add
    """
    len_batch = len(var)
    assert len_batch < self.maxlen, 'trying to add a batch that is too big!'
    start_pos = (self.start + self.length) % self.maxlen
    
    if start_pos + len_batch <= self.maxlen:
      # If there is space, add it
      self._append_batch_with_space(var)
    else:
      # No space, so break it into two batches for which there is space
      first_batch, second_batch = np.split(var, [self.maxlen - start_pos])
      self._append_batch_with_space(first_batch)
      # use append on second call in case len_batch > self.maxlen
      self._append_batch_with_space(second_batch)

class ReplayBuffer(object):
  def __init__(self, limit, item_shape):
    """
    The replay buffer object

    :param limit: (int) the max number of transitions to store
    :param item_shape: a list of tuples of (str) item name and (tuple) the shape for item
      Ex: [("observations0", env.observation_space.shape),\
          ("actions",env.action_space.shape),\
          ("rewards", (1,)),\
          ("observations1",env.observation_space.shape ),\
          ("terminals1", (1,))]
    """
    self.limit = limit

    self.items = OrderedDict()

    for name, shape in item_shape:
      self.items[name] = RingBuffer(limit, shape=shape)

  def sample(self, batch_size):
    """
    sample a random batch from the buffer

    :param batch_size: (int) the number of element to sample for the batch
    :return: (list) the sampled batch
    """
    # Draw such that we always have a proceeding element.
    batch_idxs = np.random.randint(low=1, high=(self.size - 1), size=batch_size)

    transition = []
    for buf in self.items.values():
      item = buf.get_batch(batch_idxs)
      transition.append(item)

    return transition

  def add(self, *items):
    """
    Appends a single transition to the buffer

    :param items: a list of values for the transition to append to the replay buffer,
        in the item order that we initialized the ReplayBuffer with.
    """
    for buf, value in zip(self.items.values(), items):
      buf.append(value)

  def add_batch(self, *items):
    """
    Append a batch of transitions to the buffer.

    :param items: a list of batched transition values to append to the replay buffer,
        in the item order that we initialized the ReplayBuffer with.
    """
    for buf, batched_values in zip(self.items.values(), items):
      buf.append_batch(batched_values)

  def __len__(self):
    return self.size

  @property
  def size(self):
    # Get the size of the RingBuffer on the first item type
    return len(next(iter(self.items.values())))

class EpisodicBuffer(object):
  def __init__(self, n_subbuffers, process_trajectory_fn, n_cpus=None):
    """
    A simple buffer for storing full length episodes (as a  list of lists).

    :param n_subbuffers: (int) the number of subbuffers to use
    """
    self._main_buffer = []
    self.n_subbuffers = n_subbuffers
    self._subbuffers = [[] for _ in range(n_subbuffers)]
    n_cpus = n_cpus or n_subbuffers

    self.pool = mp.Pool(n_cpus, initializer=worker_init, initargs=(CloudpickleWrapper(process_trajectory_fn),))

  def commit_subbuffer(self, i):
    """
    Adds the i-th subbuffer to the main_buffer, then clears it. 
    """
    self._main_buffer.append(self._subbuffers[i])
    self._subbuffers[i] = []

  def add_to_subbuffer(self, i, item):
    """
    Adds item to i-th subbuffer.
    """
    self._subbuffers[i].append(item)

  def __len__(self):
    return len(self._main_buffer)

  def process_trajectories(self):
    """
    Processes trajectories
    """
    return self.pool.map(worker_fn, self._main_buffer)

  def clear_main_buffer(self):
    self._main_buffer = []

  def clear_all(self):
    self._main_buffer = []
    self._subbuffers = [[] for _ in range(self.n_subbuffers)]

  def close(self):
    self.pool.close()


def her_final(trajectory, compute_reward):
  """produces hindsight experiences where desired_goal is replaced with final achieved_goal"""
  final_achieved_goal = trajectory[-1][4]
  if np.allclose(final_achieved_goal, trajectory[-1][5]):
    return [] # don't add successful trajectories twice
  hindsight_trajectory = []
  for o1, action, reward, o2, achieved_goal, desired_goal in trajectory:
    hindsight_trajectory.append([o1, action, compute_reward(achieved_goal, final_achieved_goal, None), o2, 0., final_achieved_goal])
  hindsight_trajectory[-1][4] = [1.] #set final done = 1
  return hindsight_trajectory

def her_future(trajectory, k, compute_reward):
  """produces hindsight experiences where desired_goal is replaced with future achieved_goals"""
  achieved_goals = np.array([transition[4] for transition in trajectory])
  len_ag = len(achieved_goals)
  achieved_goals_range = np.array(range(len_ag))
  hindsight_experiences = []
  for i, (o1, action, _, o2, achieved_goal, _) in enumerate(trajectory):
    sampled_goals = np.random.choice(achieved_goals_range[i:], min(k, len_ag - i), replace=False)
    sampled_goals = achieved_goals[sampled_goals]
    for g in sampled_goals:
      reward = compute_reward(achieved_goal, g, None)
      hindsight_experiences.append([o1, action, reward, o2, reward, g])
  return hindsight_experiences