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
        self.data = np.zeros((maxlen,) + shape).astype(dtype)

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
        return self.data[(self.start + idxs) % self.maxlen]

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


"""seems like this is unnecessary? Used to wrap return of ReplayBuffer.sample()
def array_min2d(arr):
    "
    cast to np.ndarray, and make sure it is of 2 dim

    :param arr: ([Any]) the array to clean
    :return: (np.ndarray) the cleaned array
    "
    arr = np.array(arr)
    if arr.ndim >= 2:
        return arr
    return arr.reshape(-1, 1)
"""

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

        self.items = []

        for name, shape in item_shape:
            self.items.append((name, RingBuffer(limit, shape=shape)))

    def sample(self, batch_size):
        """
        sample a random batch from the buffer

        :param batch_size: (int) the number of element to sample for the batch
        :return: (list) the sampled batch
        """
        # Draw such that we always have a proceeding element.
        batch_idxs = np.random.randint(low=1, high=(self.size - 1), size=batch_size)

        transition = []
        for name, buf in self.items:
            item = buf.get_batch(batch_idxs)
            transition.append(item)

        return transition

    def add(self, *items):
        """
        Append a transition to the buffer

        :param items: a list of values for the transition to append to the replay buffer,
            in the item order that we initialized the ReplayBuffer with.
        """
        for i, value in enumerate(items):
            self.items[i][1].append(value)

    @property
    def size(self):
        # Get the size of the RingBuffer on the first item type
        return len(self.items[0][1])
