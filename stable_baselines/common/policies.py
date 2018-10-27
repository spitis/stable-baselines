from abc import ABC

import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as tf_layers
from gym.spaces import Discrete, Box, MultiBinary, MultiDiscrete

from stable_baselines.a2c.utils import conv, linear, conv_to_fc, batch_to_seq, seq_to_batch, lstm
from stable_baselines.common.distributions import make_proba_dist_type


def nature_cnn(scaled_images, **kwargs):
  """
    CNN from Nature paper.

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
  activ = tf.nn.relu
  layer_1 = activ(
      conv(scaled_images, 'c1', n_filters=32, filter_size=8, stride=4, init_scale=np.sqrt(2),
           **kwargs))
  layer_2 = activ(
      conv(layer_1, 'c2', n_filters=64, filter_size=4, stride=2, init_scale=np.sqrt(2), **kwargs))
  layer_3 = activ(
      conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
  return conv_to_fc(layer_3)


def observation_input(ob_space, batch_size=None, name='Ob', scale=False):
  """
    Build observation input with encoding depending on the observation space type

    When using Box ob_space, the input will be normalized between [1, 0] on the bounds ob_space.low and ob_space.high.

    :param ob_space: (Gym Space) The observation space
    :param batch_size: (int) batch size for input
                       (default is None, so that resulting input placeholder can take tensors with any batch size)
    :param name: (str) tensorflow variable name for input placeholder
    :param scale: (bool) whether or not to scale the input
    :return: (TensorFlow Tensor, TensorFlow Tensor) input_placeholder, processed_input_tensor
    """
  if isinstance(ob_space, Discrete):
    input_x = tf.placeholder(shape=(batch_size,), dtype=tf.int32, name=name)
    processed_x = tf.to_float(tf.one_hot(input_x, ob_space.n))
    return input_x, processed_x

  elif isinstance(ob_space, Box):
    input_x = tf.placeholder(shape=(batch_size,) + ob_space.shape, dtype=ob_space.dtype, name=name)
    processed_x = tf.to_float(input_x)
    # rescale to [1, 0] if the bounds are defined
    if (scale and not np.any(np.isinf(ob_space.low)) and not np.any(np.isinf(ob_space.high)) and
        np.any((ob_space.high - ob_space.low) != 0)):

      # equivalent to processed_x / 255.0 when bounds are set to [255, 0]
      processed_x = ((processed_x - ob_space.low) / (ob_space.high - ob_space.low))
    return input_x, processed_x

  elif isinstance(ob_space, MultiBinary):
    input_x = tf.placeholder(shape=(batch_size, ob_space.n), dtype=tf.int32, name=name)
    processed_x = tf.to_float(input_x)
    return input_x, processed_x

  elif isinstance(ob_space, MultiDiscrete):
    input_x = tf.placeholder(shape=(batch_size, len(ob_space.nvec)), dtype=tf.int32, name=name)
    processed_x = tf.concat([
        tf.to_float(tf.one_hot(input_split, ob_space.nvec[i]))
        for i, input_split in enumerate(tf.split(input_x, len(ob_space.nvec), axis=-1))
    ], axis=-1)
    return input_x, processed_x

  else:
    raise NotImplementedError("Error: the model does not support input space of type {}".format(
        type(ob_space).__name__))


class BasePolicy(ABC):
  """
    The base policy object

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param scale: (bool) whether or not to scale the input
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param dueling: (bool) For DQN only: if true double the output MLP to compute a baseline for action scores
    :param is_DQN: (bool) For DQN only: whether it is a DQN
    :param goal_space: (Gym Space) The goal space of the goal-based environment
    :param goal_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for goal placeholder
        and the processed goal placeholder respectivly
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False, *,
               layers=None, scale=False, obs_phs=None, dueling=True, is_DQN=False, goal_space=None,
               goal_phs=None):
    self.n_env = n_env
    self.n_steps = n_steps

    if layers is None:
      layers = [512]
    self.layers = layers
    assert len(layers) >= 1, "Error: must have at least one hidden layer for the policy."

    with tf.name_scope("input"):
      # observation placeholders; raw & preprocessed (e.g., change to float or rescale)
      if obs_phs is None:
        self.obs_ph, self.processed_x = observation_input(ob_space, n_batch, scale=scale)
      else:
        self.obs_ph, self.processed_x = obs_phs

      # masks/states placeholders for LSTM policies
      self.masks_ph = tf.placeholder(tf.float32, [n_batch], name="masks_ph")  # mask (done t-1)
      self.states_ph = tf.placeholder(tf.float32, [self.n_env, n_lstm * 2],
                                      name="states_ph")  # states

      # action placeholder
      self.action_ph = None

      # goal placeholder, if applicable for the environment
      if goal_space is not None:
        if goal_phs is None:
          self.goal_ph, self.processed_g = observation_input(goal_space, n_batch, scale=scale)
        else:
          self.goal_ph, self.processed_g = goal_phs
      else:
        self.goal_ph, self.processed_g = None, None

    self.sess = sess
    self.reuse = reuse
    self.ob_space = ob_space
    self.ac_space = ac_space
    self.goal_space = goal_space
    self.trainable_vars = None

    # from old ActorCriticPolicy class, but now with DQN arg
    # DQN uses a categorical distribution, with action probablities computed as softmax over q-values
    # epsilon-greedy exploration and whatnot is done separately in the DQN algorithm (deepq/build_graph.py)
    self.pdtype = make_proba_dist_type(ac_space, is_DQN=is_DQN)
    self.is_discrete = isinstance(ac_space, Discrete)
    self.policy = None
    self.proba_distribution = None
    self.value_fn = None
    self.deterministic_action = None
    self.initial_state = None

    # from old DQN class
    self.q_values = None
    if self.is_discrete:
      self.n_actions = ac_space.n
      self.dueling = dueling

    self.is_DQN = is_DQN

  def _setup_and_validate(self):
    """
    Sets up the distibutions, actions, and value, and validates the policy object
    """
    assert self.trainable_vars is not None

    with tf.variable_scope("output", reuse=True):
      assert self.policy is not None and self.proba_distribution is not None and self.value_fn is not None
      self.action = self.proba_distribution.sample()
      self.deterministic_action = self.proba_distribution.mode()
      self.neglogp = self.proba_distribution.neglogp(self.action)
      self.policy_proba = self.policy
      if self.is_discrete:
        self.policy_proba = tf.nn.softmax(self.policy_proba)
      self._value = self.value_fn[:, 0]

  def step(self, obs, state=None, mask=None, deterministic=True, goal=None):
    """
        Returns the policy for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions. (used for DQN)
        :return: ([float], [float], [float], [float]) actions, values, states, neglogp
        """
    raise NotImplementedError

  def proba_step(self, obs, state=None, mask=None, goal=None):
    """
        Returns the action probability for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) the action probability
        """
    feed_dict = {self.obs_ph: obs}
    if state is not None:
      feed_dict[self.states_ph] = state
    if mask is not None:
      feed_dict[self.masks_ph] = mask
    if goal is not None:
      feed_dict[self.goal_ph] = goal
    return self.sess.run(self.policy_proba, feed_dict=feed_dict)

  def value(self, obs, *, action=None, state=None, mask=None, goal=None):
    """
        Returns the value for a single step

        :param obs: ([float] or [int]) The current observation of the environment
        :param action: ([float] or [int]) The action for computing Q values (used in DQN/DDPG)
        :param state: ([float]) The last states (used in recurrent policies)
        :param mask: ([float]) The last masks (used in recurrent policies)
        :return: ([float]) The associated value of the action
        """
    feed_dict = {self.obs_ph: obs}
    if action is not None:
      feed_dict[self.action_ph] = action
    if state is not None:
      feed_dict[self.states_ph] = state
    if mask is not None:
      feed_dict[self.masks_ph] = mask
    if goal is not None:
      feed_dict[self.goal_ph] = goal
    return self.sess.run(self._value, feed_dict=feed_dict)


class FeedForwardPolicy(BasePolicy):
  """
    Policy object that implements actor critic, using a feed forward neural network.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network for the policy (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectively
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    :param layer_norm: (bool) enable layer normalisation
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param is_DQN: (bool) if True, enables dueling (duplication because dueling should apply to more than DQN)
    :param use_action_ph: (Tensorflow Tensor) batch_size x flattened_ac_space_size tensor
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, layers=None,
               cnn_extractor=nature_cnn, feature_extraction="cnn", obs_phs=None, layer_norm=False,
               dueling=True, is_DQN=False, action_ph=None, goal_space=None, **kwargs):

    super(FeedForwardPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch,
                                            n_lstm=256, dueling=dueling, is_DQN=is_DQN, reuse=reuse,
                                            layers=layers, scale=(feature_extraction == "cnn"),
                                            obs_phs=obs_phs, goal_space=goal_space)

    with tf.variable_scope("model", reuse=reuse):
      if feature_extraction == "cnn":
        extracted_features = cnn_extractor(self.processed_x, **kwargs)
      else:
        extracted_features = tf.layers.flatten(self.processed_x)

      if self.action_ph is not None:
        extracted_features = tf.concat(axis=1, values=[extracted_features, self.action_ph])

      activ = tf.nn.relu
      pi_h = extracted_features
      vf_h = extracted_features
      for i, layer_size in enumerate(self.layers):
        pi_h = tf.layers.dense(pi_h, layer_size, name='pi_fc' + str(i))
        vf_h = tf.layers.dense(vf_h, layer_size, name='vf_fc' + str(i))
        if layer_norm:
          pi_h = tf_layers.layer_norm(pi_h, center=True, scale=True)
          vf_h = tf_layers.layer_norm(vf_h, center=True, scale=True)
        pi_h = activ(pi_h)
        vf_h = activ(vf_h)

      value_fn = linear(vf_h, 'vf', 1)
      pi_latent = pi_h
      vf_latent = vf_h

      self.proba_distribution, self.policy, self.q_values = \
          self.pdtype.proba_distribution_from_latent(pi_latent, vf_latent, init_scale=0.01)

      # Dueling, currently intended for DQN only
      if self.is_DQN and self.dueling:
        with tf.variable_scope("dueling_state_value"):
          state_out = extracted_features
          for i, layer_size in enumerate(self.layers):
            state_out = tf.layers.dense(state_out, layer_size, activation=None)
            if layer_norm:
              state_out = tf_layers.layer_norm(state_out, center=True, scale=True)
            state_out = tf.nn.relu(state_out)
          state_score = tf.layers.dense(state_out, 1, activation=None)
        action_scores_mean = tf.reduce_mean(self.q_values, axis=1)
        action_scores_centered = self.q_values - tf.expand_dims(action_scores_mean, axis=1)
        self.q_values = state_score + action_scores_centered
        self.policy = self.q_values
        self.proba_distribution = self.pdtype.proba_distribution_from_flat(self.q_values)

      self.trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope=tf.get_variable_scope().name)

    self.value_fn = value_fn
    self.initial_state = None
    self._setup_and_validate()

  def step(self, obs, state=None, mask=None, deterministic=False, only_action=False):

    if only_action:  #more efficient for DQN
      if deterministic:
        action = self.sess.run(self.deterministic_action, {self.obs_ph: obs})
      else:
        action = self.sess.run(self.action, {self.obs_ph: obs})
      return action, None, None, None

    if deterministic:
      action, value, neglogp = self.sess.run([self.deterministic_action, self._value, self.neglogp],
                                             {self.obs_ph: obs})
    else:
      action, value, neglogp = self.sess.run([self.action, self._value, self.neglogp],
                                             {self.obs_ph: obs})
    return action, value, self.initial_state, neglogp


class MlpPolicy(FeedForwardPolicy):
  """
    Policy object that implements actor critic, using a MLP (2 layers of 64)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, dueling=True,
               is_DQN=False, goal_space=None, **_kwargs):
    super(MlpPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                    feature_extraction="mlp", dueling=dueling, is_DQN=is_DQN,
                                    goal_space=goal_space, **_kwargs)


class CnnPolicy(FeedForwardPolicy):
  """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, dueling=True,
               is_DQN=False, goal_space=None, **_kwargs):
    super(CnnPolicy, self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                                    feature_extraction="cnn", dueling=dueling, is_DQN=is_DQN,
                                    goal_space=goal_space, **_kwargs)


class LnMlpPolicy(FeedForwardPolicy):
  """
    Policy object that implements DQN policy, using a MLP (2 layers of 64), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, obs_phs=None,
               dueling=True, goal_space=None, is_DQN=False, **_kwargs):
    super(LnMlpPolicy,
          self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                         feature_extraction="mlp", obs_phs=obs_phs, layer_norm=True,
                         dueling=dueling, is_DQN=is_DQN, goal_space=goal_space, **_kwargs)


class LnCnnPolicy(FeedForwardPolicy):
  """
    Policy object that implements DQN policy, using a CNN (the nature CNN), with layer normalisation

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param obs_phs: (TensorFlow Tensor, TensorFlow Tensor) a tuple containing an override for observation placeholder
        and the processed observation placeholder respectivly
    :param dueling: (bool) if true double the output MLP to compute a baseline for action scores
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, obs_phs=None,
               dueling=True, is_DQN=False, goal_space=None, **_kwargs):
    super(LnCnnPolicy,
          self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=reuse,
                         feature_extraction="cnn", obs_phs=obs_phs, dueling=dueling, is_DQN=is_DQN,
                         layer_norm=True, goal_space=goal_space, **_kwargs)


class LstmPolicy(BasePolicy):
  """
    Policy object that implements actor critic, using LSTMs.

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param layers: ([int]) The size of the Neural network before the LSTM layer  (if None, default to [64, 64])
    :param cnn_extractor: (function (TensorFlow Tensor, ``**kwargs``): (TensorFlow Tensor)) the CNN feature extraction
    :param layer_norm: (bool) Whether or not to use layer normalizing LSTMs
    :param feature_extraction: (str) The feature extraction type ("cnn" or "mlp")
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False,
               layers=None, cnn_extractor=nature_cnn, layer_norm=False, feature_extraction="cnn",
               dueling=True, is_DQN=False, goal_space=None, **kwargs):

    super(LstmPolicy,
          self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=n_lstm,
                         reuse=reuse, layers=layers, scale=(feature_extraction == "cnn"),
                         dueling=dueling, is_DQN=is_DQN, goal_space=goal_space)

    with tf.variable_scope("model", reuse=reuse):
      if feature_extraction == "cnn":
        extracted_features = cnn_extractor(self.processed_x, **kwargs)
      else:
        activ = tf.tanh
        extracted_features = tf.layers.flatten(self.processed_x)
        for i, layer_size in enumerate(self.layers):
          extracted_features = activ(
              linear(extracted_features, 'pi_fc' + str(i), n_hidden=layer_size,
                     init_scale=np.sqrt(2)))
      input_sequence = batch_to_seq(extracted_features, self.n_env, n_steps)
      masks = batch_to_seq(self.masks_ph, self.n_env, n_steps)
      rnn_output, self.snew = lstm(input_sequence, masks, self.states_ph, 'lstm1', n_hidden=n_lstm,
                                   layer_norm=layer_norm)
      rnn_output = seq_to_batch(rnn_output)
      value_fn = linear(rnn_output, 'vf', 1)

      self.proba_distribution, self.policy, self.q_values = \
          self.pdtype.proba_distribution_from_latent(rnn_output, rnn_output)

      self.trainable_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                              scope=tf.get_variable_scope().name)

    self.value_fn = value_fn
    self.initial_state = np.zeros((self.n_env, n_lstm * 2), dtype=np.float32)
    self._setup_and_validate()

  def step(self, obs, state=None, mask=None, deterministic=False):
    if deterministic:
      return self.sess.run([self.deterministic_action, self._value, self.snew, self.neglogp], {
          self.obs_ph: obs,
          self.states_ph: state,
          self.masks_ph: mask
      })
    else:
      return self.sess.run([self.action, self._value, self.snew, self.neglogp], {
          self.obs_ph: obs,
          self.states_ph: state,
          self.masks_ph: mask
      })


class CnnLstmPolicy(LstmPolicy):
  """
    Policy object that implements actor critic, using LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False,
               dueling=True, is_DQN=False, goal_space=None, **_kwargs):
    super(CnnLstmPolicy,
          self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=n_lstm,
                         reuse=reuse, layer_norm=False, feature_extraction="cnn", dueling=dueling,
                         is_DQN=is_DQN, goal_space=goal_space, **_kwargs)


class CnnLnLstmPolicy(LstmPolicy):
  """
    Policy object that implements actor critic, using a layer normalized LSTMs with a CNN feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False,
               dueling=True, is_DQN=False, goal_space=None, **_kwargs):
    super(CnnLnLstmPolicy,
          self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=n_lstm,
                         reuse=reuse, layer_norm=True, feature_extraction="cnn", dueling=dueling,
                         is_DQN=is_DQN, goal_space=goal_space, **_kwargs)


class MlpLstmPolicy(LstmPolicy):
  """
    Policy object that implements actor critic, using LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False,
               dueling=True, is_DQN=False, goal_space=None, **_kwargs):
    super(MlpLstmPolicy,
          self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=n_lstm,
                         reuse=reuse, layer_norm=False, feature_extraction="mlp", dueling=dueling,
                         is_DQN=is_DQN, goal_space=goal_space, **_kwargs)


class MlpLnLstmPolicy(LstmPolicy):
  """
    Policy object that implements actor critic, using a layer normalized LSTMs with a MLP feature extraction

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param n_lstm: (int) The number of LSTM cells (for recurrent policies)
    :param reuse: (bool) If the policy is reusable or not
    :param kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

  def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=256, reuse=False,
               dueling=True, is_DQN=False, goal_space=None, **_kwargs):
    super(MlpLnLstmPolicy,
          self).__init__(sess, ob_space, ac_space, n_env, n_steps, n_batch, n_lstm=n_lstm,
                         reuse=reuse, layer_norm=True, feature_extraction="mlp", dueling=dueling,
                         is_DQN=is_DQN, goal_space=goal_space, **_kwargs)


_policy_registry = {
    "CnnPolicy": CnnPolicy,
    "CnnLstmPolicy": CnnLstmPolicy,
    "CnnLnLstmPolicy": CnnLnLstmPolicy,
    "MlpPolicy": MlpPolicy,
    "MlpLstmPolicy": MlpLstmPolicy,
    "MlpLnLstmPolicy": MlpLnLstmPolicy,
    "LnMlpPolicy": LnMlpPolicy,
    "LnCnnPolicy": LnCnnPolicy,
}


def get_policy_from_name(name):
  """
    returns the registed policy from the base type and name

    :param base_policy_type: (BasePolicy) the base policy object
    :param name: (str) the policy name
    :return: (base_policy_type) the policy
    """
  return _policy_registry[name]


def register_policy(name, policy):
  """
    returns the registed policy from the base type and name

    :param name: (str) the policy name
    :param policy: (subclass of BasePolicy) the policy
    """
  _policy_registry[name] = policy