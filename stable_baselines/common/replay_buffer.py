import numpy as np
from collections import OrderedDict


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

  @property
  def size(self):
    # Get the size of the RingBuffer on the first item type
    return len(next(iter(self.items.values())))
