import numpy as np
import tensorflow as tf


class ActionNoise(object):
  """
    The action noise base class
    """

  def reset(self):
    """
        call end of episode reset for the noise
        """
    pass


class NormalActionNoise(ActionNoise):
  """
    A gaussian action noise

    :param mean: (float) the mean value of the noise
    :param sigma: (float) the scale of the noise (std here)
    """

  def __init__(self, mean, sigma):
    self._mu = mean
    self._sigma = sigma

  def __call__(self):
    return np.random.normal(self._mu, self._sigma)

  def __repr__(self):
    return 'NormalActionNoise(mu={}, sigma={})'.format(self._mu, self._sigma)


class OrnsteinUhlenbeckActionNoise(ActionNoise):
  """
    A Ornstein Uhlenbeck action noise, this is designed to aproximate brownian motion with friction.

    Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab

    :param mean: (float) the mean of the noise
    :param sigma: (float) the scale of the noise
    :param theta: (float) the rate of mean reversion
    :param dt: (float) the timestep for the noise
    :param initial_noise: ([float]) the initial value for the noise output, (if None: 0)
    """

  def __init__(self, mean, sigma, theta=.15, dt=1e-2, initial_noise=None):
    self._theta = theta
    self._mu = mean
    self._sigma = sigma
    self._dt = dt
    self.initial_noise = initial_noise
    self.noise_prev = None
    self.reset()

  def __call__(self):
    noise = self.noise_prev + self._theta * (self._mu - self.noise_prev) * self._dt + \
            self._sigma * np.sqrt(self._dt) * np.random.normal(size=self._mu.shape)
    self.noise_prev = noise
    return noise

  def reset(self):
    """
        reset the Ornstein Uhlenbeck noise, to the initial position
        """
    self.noise_prev = self.initial_noise if self.initial_noise is not None else np.zeros_like(self._mu)

  def __repr__(self):
    return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self._mu, self._sigma)


class OUNoiseTensorflow():
  """
    :param action_space_dims: (int) size of action space
    :param mean: (float) the mean of the noise
    :param sigma: (float) the scale of the noise
    :param theta: (float) the rate of mean reversion
    :param dt: (float) the timestep for the noise
    :param max_parallel_episodes (int): the max number of separate episode noises that will be needed
    """

  def __init__(self, action_space_dims, sigma=0.2, theta=0.15, dt=0.02, max_parallel_episodes=128):
    with tf.variable_scope('ou_action_noise'):
      self.noise = tf.get_variable(
          'current_noise',
          shape=(max_parallel_episodes, action_space_dims),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.constant_initializer(0.))
      self.mean = tf.zeros_like(self.noise)
      self.reset_noise_ph = tf.placeholder_with_default(np.zeros((max_parallel_episodes, 1), dtype=np.float32), (None, 1))

      reset_noise = tf.pad(self.reset_noise_ph, [[ 0, max_parallel_episodes - tf.shape(self.reset_noise_ph)[0] ], [0,0]])
      updated_noise = self.noise + theta * (self.mean - self.noise) * dt + sigma * np.sqrt(dt) * tf.random_normal(
          tf.shape(self.noise))
      reseted_noise = self.mean

      self.update_noise_op = self.noise.assign(reset_noise * reseted_noise + (1. - reset_noise) * updated_noise)

  def __call__(self, batched_deterministic_actions):
    with tf.control_dependencies([self.update_noise_op]):
      return batched_deterministic_actions + tf.slice(self.noise, [0, 0], tf.shape(batched_deterministic_actions))

class NormalNoiseTensorflow():
  """
  A gaussian action noise

  :param mean: (float) the mean value of the noise
  :param sigma: (float) the scale of the noise (std here)
  """
  def __init__(self, sigma=0.2):
    self.sigma = 0.2
    self.reset_noise_ph = tf.placeholder(tf.float32) # to make it have a consistent interface with ou. 

  def __call__(self, batched_deterministic_actions):
    return batched_deterministic_actions + tf.random_normal(tf.shape(batched_deterministic_actions), stddev=self.sigma)

