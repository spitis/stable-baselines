from abc import ABC, abstractmethod
import os
import glob

import cloudpickle
import numpy as np
import gym
import tensorflow as tf

from stable_baselines.common import set_global_seeds
from stable_baselines.common.policies import LstmPolicy, get_policy_from_name, BasePolicy
from stable_baselines.common.vec_env import VecEnvWrapper, VecEnv, DummyVecEnv
from stable_baselines import logger
from stable_baselines.a2c.utils import total_episode_reward_logger


class BaseRLModel(ABC):
  """
    The base RL model

    :param policy: (BasePolicy) Policy object
    :param env: (Gym environment) The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param requires_vec_env: (bool) Does this model require a vectorized environment
    """

  def __init__(self,
               policy,
               env,
               verbose=0,
               *,
               requires_vec_env,
               _init_setup_model=None):
    if isinstance(policy, str):
      self.policy = get_policy_from_name(policy)
    else:
      self.policy = policy
    self.env = env
    self.verbose = verbose
    self._requires_vec_env = requires_vec_env
    self.observation_space = None
    self.action_space = None
    self.goal_space = None
    self.n_envs = None
    self._vectorize_action = False
    self.model = None  # This is the instatiated policy object, i.e., self.model = self.policy(...)

    # used to be in ActorCriticRLModel
    self.sess = None
    self.initial_state = None
    self.step = lambda: None
    self.params = None

    # For the SimpleRLModel subclass (yes it should be there, but will eventually be merged anyways)
    self.exploration = None
    self.task_step = None
    self.global_step = None
    self.graph = None
    self.tensorboard_log = None
    self.eval_env = None
    self.eval_every = 10

    if env is not None:
      if isinstance(env, str):
        if self.verbose >= 1:
          print(
              "Creating environment from the given name, wrapped in a DummyVecEnv."
          )
        self.env = env = DummyVecEnv([lambda: gym.make(env)])

      # Check if the environment is a goal-oriented type based on their observation space
      # Goal oriented Gym space have observation_space as dict
      if type(env.observation_space) == gym.spaces.dict_space.Dict:
        self.observation_space = env.observation_space.spaces["observation"]
        # Assume that desired and achieved goal have the same space
        self.goal_space = env.observation_space.spaces["desired_goal"]
      else:
        self.observation_space = env.observation_space

      self.action_space = env.action_space
      if requires_vec_env:
        if isinstance(env, VecEnv):
          self.n_envs = env.num_envs
        else:
          self.env = env = DummyVecEnv([lambda: env])
          self.n_envs = 1
          #raise ValueError(
          #    "Error: the model requires a vectorized environment, please use a VecEnv wrapper."
          #)
      else:
        if isinstance(env, VecEnv):
          if env.num_envs == 1:
            self.env = _UnvecWrapper(env)
            self._vectorize_action = True
          else:
            raise ValueError(
                "Error: the model requires a non vectorized environment or a single vectorized"
                " environment.")
        self.n_envs = 1

  def get_env(self):
    """
        returns the current environment (can be None if not defined)

        :return: (Gym Environment) The current environment
        """
    return self.env

  def set_env(self, env):
    """
        Checks the validity of the environment, and if it is coherent, set it as the current environment.

        :param env: (Gym Environment) The environment for learning a policy
        """
    if env is None and self.env is None:
      if self.verbose >= 1:
        print("Loading a model without an environment, "
              "this model cannot be trained until it has a valid environment.")
      return
    elif env is None:
      raise ValueError(
          "Error: trying to replace the current environment with None")

    # sanity checking the environment
    if type(env.observation_space) == gym.spaces.dict_space.Dict:
        assert self.observation_space == env.observation_space.spaces["observation"], \
            "Error: the Goal-based environment passed must have at least the same observation space as the model was trained on."
    else:
        assert self.observation_space == env.observation_space, \
            "Error: the environment passed must have at least the same observation space as the model was trained on."
    assert self.action_space == env.action_space, \
        "Error: the environment passed must have at least the same action space as the model was trained on."
    if self._requires_vec_env:
      if not isinstance(env, VecEnv):
        env = DummyVecEnv([lambda: env])
      assert isinstance(env, VecEnv), \
          "Error: the environment passed is not a vectorized environment, however {} requires it".format(
              self.__class__.__name__)
      assert not issubclass(self.policy, LstmPolicy) or self.n_envs == env.num_envs, \
          "Error: the environment passed must have the same number of environments as the model was trained on." \
          "This is due to the Lstm policy not being capable of changing the number of environments."
      self.n_envs = env.num_envs
    else:
      # for models that dont want vectorized environment, check if they make sense and adapt them.
      # Otherwise tell the user about this issue
      if isinstance(env, VecEnv):
        if env.num_envs == 1:
          env = _UnvecWrapper(env)
          self._vectorize_action = True
        else:
          raise ValueError(
              "Error: the model requires a non vectorized environment or a single vectorized "
              "environment.")
      else:
        self._vectorize_action = False

      self.n_envs = 1

    self.env = env

  @abstractmethod
  def _setup_model(self):
    """
        Create all the functions and tensorflow graphs necessary to train the model
        """
    pass

  def _setup_learn(self, seed):
    """
        check the environment, set the seed, and set the logger

        :param seed: (int) the seed value
        """
    if self.env is None:
      raise ValueError(
          "Error: cannot train the model without a valid environment, please set an environment with"
          "set_env(self, env) method.")
    if seed is not None:
      set_global_seeds(seed)

  @abstractmethod
  def learn(self,
            total_timesteps,
            callback=None,
            seed=None,
            log_interval=100,
            tb_log_name="run"):
    """
        Return a trained model.

        :param total_timesteps: (int) The total number of samples to train on
        :param seed: (int) The initial seed for training, if None: keep current seed
        :param callback: (function (dict, dict)) function called at every steps with state of the algorithm.
            It takes the local and global variables.
        :param log_interval: (int) The number of timesteps before logging.
        :param tb_log_name: (str) the name of the run for tensorboard log
        :return: (BaseRLModel) the trained model
        """
    pass

  def predict(self,
              observation,
              state=None,
              mask=None,
              deterministic=False,
              goal=None):
    """
        Get the model's action from an observation

        :param observation: (np.ndarray) the input observation
        :param state: (np.ndarray) The last states (can be None, used in recurrent policies)
        :param mask: (np.ndarray) The last masks (can be None, used in recurrent policies)
        :param deterministic: (bool) Whether or not to return deterministic actions.
        :param goal: (np.ndarray) the goal (can be None, used in goal-based environment)
        :return: (np.ndarray, np.ndarray) the model's action and the next state (used in recurrent policies)
        """
    if state is None:
      state = self.initial_state
    if mask is None:
      mask = [False for _ in range(self.n_envs)]

    observation = np.array(observation)
    vectorized_env = self._is_vectorized_observation(observation,
                                                     self.observation_space)

    observation = observation.reshape((-1,) + self.observation_space.shape)
    actions, _, states, _ = self.model.step(
        observation, state, mask, deterministic=deterministic)

    if not vectorized_env:
      if state is not None:
        raise ValueError(
            "Error: The environment must be vectorized when using recurrent policies."
        )
      actions = actions[0]

    return actions, states

  @abstractmethod
  def save(self, save_path):
    """
        Save the current parameters to file

        :param save_path: (str) the save location
        """
    # self._save_to_file(save_path, data={}, params=None)
    raise NotImplementedError()

  @classmethod
  def load(cls, load_path, env=None, **kwargs):
    data, params = cls._load_from_file(load_path)

    model = cls(policy=data["policy"], env=None, _init_setup_model=False)
    model.__dict__.update(data)
    model.__dict__.update(kwargs)
    model.set_env(env)
    model._setup_model()

    restores = []
    for param, loaded_p in zip(model.params, params):
      restores.append(param.assign(loaded_p))
    model.sess.run(restores)

    return model

  @staticmethod
  def _save_to_file(save_path, data=None, params=None):
    _, ext = os.path.splitext(save_path)
    if ext == "":
      save_path += ".pkl"

    with open(save_path, "wb") as file:
      cloudpickle.dump((data, params), file)

  @staticmethod
  def _load_from_file(load_path):
    if not os.path.exists(load_path):
      if os.path.exists(load_path + ".pkl"):
        load_path += ".pkl"
      else:
        raise ValueError(
            "Error: the file {} could not be found".format(load_path))

    with open(load_path, "rb") as file:
      data, params = cloudpickle.load(file)

    return data, params

  @staticmethod
  def _softmax(x_input):
    """
        An implementation of softmax.

        :param x_input: (numpy float) input vector
        :return: (numpy float) output vector
        """
    x_exp = np.exp(x_input.T - np.max(x_input.T, axis=0))
    return (x_exp / x_exp.sum(axis=0)).T

  @staticmethod
  def _is_vectorized_observation(observation, observation_space):
    """
        For every observation type, detects and validates the shape,
        then returns whether or not the observation is vectorized.

        :param observation: (np.ndarray) the input observation to validate
        :param observation_space: (gym.spaces) the observation space
        :return: (bool) whether the given observation is vectorized or not
        """
    if isinstance(observation_space, gym.spaces.Box):
      if observation.shape == observation_space.shape:
        return False
      elif observation.shape[1:] == observation_space.shape:
        return True
      else:
        raise ValueError("Error: Unexpected observation shape {} for ".format(
            observation.shape) + "Box environment, please use {} ".format(
                observation_space.shape) +
                         "or (n_env, {}) for the observation shape."
                         .format(", ".join(map(str, observation_space.shape))))
    elif isinstance(observation_space, gym.spaces.Discrete):
      if observation.shape == (
      ):  # A numpy array of a number, has shape empty tuple '()'
        return False
      elif len(observation.shape) == 1:
        return True
      else:
        raise ValueError(
            "Error: Unexpected observation shape {} for ".format(
                observation.shape) +
            "Discrete environment, please use (1,) or (n_env, 1) for the observation shape."
        )
    elif isinstance(observation_space, gym.spaces.MultiDiscrete):
      if observation.shape == (len(observation_space.nvec),):
        return False
      elif len(observation.shape) == 2 and observation.shape[1] == len(
          observation_space.nvec):
        return True
      else:
        raise ValueError(
            "Error: Unexpected observation shape {} for MultiDiscrete ".format(
                observation.shape) + "environment, please use ({},) or ".format(
                    len(observation_space.nvec)) +
            "(n_env, {}) for the observation shape.".format(
                len(observation_space.nvec)))
    elif isinstance(observation_space, gym.spaces.MultiBinary):
      if observation.shape == (observation_space.n,):
        return False
      elif len(observation.shape
              ) == 2 and observation.shape[1] == observation_space.n:
        return True
      else:
        raise ValueError(
            "Error: Unexpected observation shape {} for MultiBinary ".format(
                observation.shape) +
            "environment, please use ({},) or ".format(observation_space.n) +
            "(n_env, {}) for the observation shape.".format(observation_space.n)
        )
    else:
      raise ValueError(
          "Error: Cannot determine if the observation is vectorized with the space type {}."
          .format(observation_space))


class SimpleRLModel(BaseRLModel):

  @abstractmethod
  def _setup_model(self):
    """Create tensorflow graph / training ops"""
    pass

  @abstractmethod
  def _setup_new_task(self, total_timesteps):
    pass

  @abstractmethod
  def _get_action_for_single_obs(self, obs):
    pass

  @abstractmethod
  def _process_experience(self, obs, action, rew, new_obs, done):
    pass

  def evaluate(self, n_episodes):
    """evaluates model for n_episodes"""
    env = self.eval_env
    assert env is not None, "Must set an eval_env in order to evaluate!"
    results = []
    
    for _ in range(n_episodes):
      obs, done = env.reset(), False
      reward = 0.
      steps = 0
      while not done and reward < 1.:
        action, _ = self.predict(obs)
        obs, rew, done, _ = env.step(action)
        reward += rew
        steps += 1
      results.append((float(reward > 0.), steps))
    return results

  def learn(self,
            total_timesteps,
            callback=None,
            seed=None,
            log_interval=100,
            tb_log_name="DQN"):
    """Assumes VecEnv"""

    with SetVerbosity(self.verbose), TensorboardWriter(
        self.graph, self.tensorboard_log, tb_log_name) as writer:
      self._setup_learn(seed)
      self._setup_new_task(total_timesteps=total_timesteps)

      obses = self.env.reset()

      # probably shouldn't have two different bookkeeping mechanisms here, but whatever
      tb_episode_rewards = np.zeros((self.n_envs,))
      legacy_episode_rewards_per_env = np.zeros((self.n_envs,))
      legacy_episode_rewards = []
      test_successes, test_steps, best_eval = [], [], 0.

      data_path = writer.get_logdir()
      if not os.path.exists(data_path):
        os.makedirs(data_path)
      
      with open(os.path.join(data_path, 'config.txt'), 'w') as f:
        for atr, val in self.__dict__.items():
          if isinstance(val, (str, float, int)):
            f.write("{} - {}\n".format(atr, val))
      
      with open(os.path.join(data_path, 'test_results.txt'), 'w') as f:

        for _ in range(total_timesteps // self.n_envs):
          if callback is not None:
            callback(locals(), globals())

          # Take action and update exploration to the newest value
          actions = self._get_action_for_single_obs(obses)
          new_obses, rewards, dones, _ = self.env.step(actions)


          # Do the learning and fetch tensorboard summaries
          summaries = self._process_experience(obses, actions, rewards, new_obses, dones)

          # Tensorboard logging
          if writer is not None and summaries:
            for summary in summaries:
              writer.add_summary(summary, self.global_step)
              ep_rewards = np.expand_dims(rewards, 1)
              ep_dones = np.expand_dims(dones, 1)
              tb_episode_rewards = total_episode_reward_logger(tb_episode_rewards, ep_rewards, ep_dones, writer, len(legacy_episode_rewards))

          # Command line and evaluation logging
          legacy_episode_rewards_per_env += rewards
          for idx in np.argwhere(dones):
            legacy_episode_rewards.append(legacy_episode_rewards_per_env[idx[0]])
            legacy_episode_rewards_per_env[idx[0]] = 0.

            num_episodes = len(legacy_episode_rewards)
            if self.eval_env is not None and num_episodes % self.eval_every == 0:
              sucs, stps = self.evaluate(1)[0]
              test_successes.append(sucs)
              test_steps.append(stps)
              mean_eval = np.mean(test_successes[-100:])
              mean_steps = np.mean(test_steps[-100:])
              f.write("Step {}---Test {}---Last100 {}\n".format(num_episodes, test_successes[-1], mean_eval))
              summary = tf.Summary(value=[tf.Summary.Value(tag="test_reward", simple_value=test_successes[-1])])
              writer.add_summary(summary, num_episodes)
              summary = tf.Summary(value=[tf.Summary.Value(tag="test_reward/steps", simple_value=test_steps[-1])])
              writer.add_summary(summary, num_episodes)
              if (len(test_successes) + 1) % 20 == 0:
                print("Evaluation perf for last 100 evaluations: {}".format(mean_eval))
                if mean_eval > best_eval:
                  print("Beat previous best eval performance of {}! Saving model...".format(best_eval))
                  best_eval = mean_eval
                  self.save(os.path.join(data_path, 'best_eval.pkl'))


            if self.verbose >= 1 and log_interval is not None and num_episodes % log_interval == 0:
              if len(legacy_episode_rewards[-101:-1]) == 0:
                mean_100ep_reward = -np.inf
              else:
                mean_100ep_reward = round(float(np.mean(legacy_episode_rewards[-101:-1])), 2)

              logger.record_tabular("steps", self.task_step)
              logger.record_tabular("episodes", num_episodes)
              logger.record_tabular("repl_buff_len", len(self.replay_buffer))
              if hasattr(self, 'replay_buffer_hindsight') and self.replay_buffer_hindsight is not None:
                logger.record_tabular("repl_buff_hindsight_len", len(self.replay_buffer_hindsight))
              if hasattr(self, 'landmark_generator') and self.landmark_generator is not None:
                logger.record_tabular("landmark_gen_length", len(self.landmark_generator))
              logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
              if self.exploration:
                logger.record_tabular(
                    "% time spent exploring",
                    int(100 * self.exploration.value(self.task_step)))
              logger.dump_tabular()

          obses = new_obses

    return self


class _UnvecWrapper(VecEnvWrapper):

  def __init__(self, venv):
    """
        Unvectorize a vectorized environment, for vectorized environment that only have one environment

        :param venv: (VecEnv) the vectorized environment to wrap
        """
    super().__init__(venv)
    assert venv.num_envs == 1, "Error: cannot unwrap a environment wrapper that has more than one environment."

  def reset(self):
    return self.venv.reset()[0]

  def step_async(self, actions):
    self.venv.step_async([actions])

  def step_wait(self):
    actions, values, states, information = self.venv.step_wait()
    return actions[0], float(values[0]), states[0], information[0]

  def render(self, mode='human'):
    return self.venv.render(mode=mode)


class SetVerbosity:

  def __init__(self, verbose=0):
    """
        define a region of code for certain level of verbosity

        :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
        """
    self.verbose = verbose

  def __enter__(self):
    self.tf_level = os.environ.get('TF_CPP_MIN_LOG_LEVEL', '0')
    self.log_level = logger.get_level()
    self.gym_level = gym.logger.MIN_LEVEL

    if self.verbose <= 1:
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    if self.verbose <= 0:
      logger.set_level(logger.DISABLED)
      gym.logger.set_level(gym.logger.DISABLED)

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.verbose <= 1:
      os.environ['TF_CPP_MIN_LOG_LEVEL'] = self.tf_level

    if self.verbose <= 0:
      logger.set_level(self.log_level)
      gym.logger.set_level(self.gym_level)


class TensorboardWriter:

  def __init__(self, graph, tensorboard_log_path, tb_log_name):
    """
        Create a Tensorboard writer for a code segment, and saves it to the log directory as its own run

        :param graph: (Tensorflow Graph) the model graph
        :param tensorboard_log_path: (str) the save path for the log (can be None for no logging)
        :param tb_log_name: (str) the name of the run for tensorboard log
        """
    self.graph = graph
    self.tensorboard_log_path = tensorboard_log_path
    self.tb_log_name = tb_log_name
    self.writer = None

  def __enter__(self):
    if self.tensorboard_log_path is not None:
      save_path = os.path.join(
          self.tensorboard_log_path, "{}_{}".format(
              self.tb_log_name,
              self._get_latest_run_id() + 1))
      self.writer = tf.summary.FileWriter(save_path, graph=self.graph)
    return self.writer

  def _get_latest_run_id(self):
    """
        returns the latest run number for the given log name and log path,
        by finding the greatest number in the directories.

        :return: (int) latest run number
        """
    max_run_id = 0
    for path in glob.glob(self.tensorboard_log_path +
                          "/{}_[0-9]*".format(self.tb_log_name)):
      file_name = path.split("/")[-1]
      ext = file_name.split("_")[-1]
      if self.tb_log_name == "_".join(file_name.split(
          "_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
        max_run_id = int(ext)
    return max_run_id

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.writer is not None:
      self.writer.add_graph(self.graph)
      self.writer.flush()
