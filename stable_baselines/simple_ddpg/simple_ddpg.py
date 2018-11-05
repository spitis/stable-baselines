import tensorflow as tf
import numpy as np
import gym
import trfl

from stable_baselines.common import tf_util, SimpleRLModel, SetVerbosity
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.replay_buffer import ReplayBuffer
from stable_baselines.simple_ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines.common.policies import get_policy_from_name


class SimpleDDPG(SimpleRLModel):
  """
    Simplified version of DDPG model class. DDPG paper: https://arxiv.org/pdf/1509.02971.pdf
 
    :param actor_policy: (BasePolicy or str) The policy model to use for actor
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param actor_lr: (float) the actor learning rate
    :param critic_lr: (float) the critic learning rate

    :param critic_policy: (BasePolicy or str) Policy model to use for critic (if None (default), uses actor_policy)
    :param joint_feature_extractor: (function) extracts features to be shared by actor & critic (if None, identity_extractor)
    :param rescale_input: (bool) whether or not to rescale the input to [0., 1.] (e.g., for images with values in [0,255])
 
    :param noise_type: (str) the noises ('normal' or 'ou') to use 
     
    :param buffer_size: (int) size of the replay buffer
    :param train_freq: (int) update the model every `train_freq` steps
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
 
    :param target_network_update_frac: (float) fraction by which to update the target network every time.
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.


    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

  def __init__(self, policy, env, gamma=0.99, actor_lr=1e-4, critic_lr=1e-3, *,
               critic_policy=None, joint_feature_extractor=None, rescale_input=True,
               action_noise=None, param_noise=False, buffer_size=50000, train_freq=1, batch_size=32,
               learning_starts=1000, target_network_update_frac=0.001, target_network_update_freq=1,
               verbose=0, tensorboard_log=None, grad_norm_clipping=10., _init_setup_model=True):

    super(SimpleDDPG, self).__init__(policy=policy, env=env, verbose=verbose,
                                     requires_vec_env=True)

    self.gamma = gamma
    self.actor_lr = actor_lr
    self.critic_lr = critic_lr

    self.actor_policy = self.policy
    self.critic_policy = self.policy if critic_policy is None else critic_policy
    if isinstance(self.critic_policy, str):
        self.critic_policy = get_policy_from_name(self.critic_policy)

    self.joint_feature_extractor = joint_feature_extractor or identity_extractor
    self.rescale_input = rescale_input
    
    self.action_noise = action_noise
    if self.action_space is not None and action_noise is None:
      self.action_noise = OrnsteinUhlenbeckActionNoise(
          mean=np.zeros(self.action_space.shape[-1]),
          sigma=0.2 * np.ones(self.action_space.shape[-1]))

    self.param_noise = param_noise
    if param_noise:
      raise NotImplementedError('param_noise to be added later')

    self.learning_starts = learning_starts
    self.train_freq = train_freq
    self.batch_size = batch_size
    self.buffer_size = buffer_size

    self.target_network_update_frac = target_network_update_frac
    self.target_network_update_freq = target_network_update_freq

    self.grad_norm_clipping = grad_norm_clipping

    self.tensorboard_log = tensorboard_log

    # Below props are set in self._setup_new_task()
    self.reset = None
    self.global_step = 0
    self.task_step = 0
    self.replay_buffer = None
    self.exploration = None

    # Several additional props to be set in self._setup_model()
    if _init_setup_model:
      self._setup_model()

  def _setup_model(self):
    with SetVerbosity(self.verbose):
      assert isinstance(self.action_space, gym.spaces.Box), \
          "Error: SimpleDDPG only supports gym.spaces.Box action space."

      self.graph = tf.Graph()
      with self.graph.as_default():
        self.sess = tf_util.make_session(graph=self.graph)
        ob_space = self.observation_space

        with tf.variable_scope("ddpg"):

          # main network
          with tf.variable_scope('main_network', reuse=False):
            obs_ph = tf.placeholder(
                shape=(None,) + ob_space.shape, dtype=ob_space.dtype, name='obs_ph')
            processed_x = rescale_obs_ph(tf.to_float(obs_ph), ob_space, self.rescale_input)
            main_input, main_joint_tvars = self.joint_feature_extractor(processed_x)

            action_ph = tf.placeholder(
                shape=(None,) + self.action_space.shape, dtype=self.action_space.dtype,
                name='action_ph')

            with tf.variable_scope('actor', reuse=False):
              actor_branch = self.actor_policy(self.sess, ob_space, self.action_space, n_env=1,
                                              n_steps=1, n_batch=None, obs_phs=(obs_ph, main_input),
                                              goal_space=self.goal_space)
            with tf.variable_scope('critic', reuse=False):
              critic_branch = self.critic_policy(
                  self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None,
                  obs_phs=(obs_ph,
                          main_input), goal_space=self.goal_space, action_ph=action_ph)
            with tf.variable_scope('critic', reuse=True):
              max_critic_branch = self.critic_policy(
                  self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None,
                  obs_phs=(obs_ph,
                          main_input), goal_space=self.goal_space, action_ph=actor_branch.policy)

            if not self.param_noise:
              act = actor_branch.policy + self.action_noise()
            else:
              raise NotImplementedError('param_noise to be added later')

          # target network
          with tf.variable_scope('target_network', reuse=False):
            target_obs_ph = tf.placeholder(
                shape=(None,) + ob_space.shape, dtype=ob_space.dtype, name='target_obs_ph')
            processed_x = rescale_obs_ph(tf.to_float(target_obs_ph), ob_space, self.rescale_input)
            target_input, target_joint_tvars = self.joint_feature_extractor(processed_x)

            with tf.variable_scope('actor', reuse=False):
              target_actor_branch = self.actor_policy(
                  self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None,
                  obs_phs=(target_obs_ph, target_input), goal_space=self.goal_space)
            with tf.variable_scope('critic', reuse=False):
              target_critic_branch = self.critic_policy(
                  self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None,
                  obs_phs=(target_obs_ph, target_input), goal_space=self.goal_space,
                  action_ph=target_actor_branch.policy)

        # variables for main and target networks
        main_vars = main_joint_tvars + actor_branch.trainable_vars + critic_branch.trainable_vars
        target_vars = target_joint_tvars + target_actor_branch.trainable_vars +\
                      target_critic_branch.trainable_vars

        with tf.variable_scope("loss"):

          # note: naming conventions from trfl (see https://github.com/deepmind/trfl/blob/master/docs/index.md)

          #""" CRITIC LOSS """

          # placeholders for bellman equation
          r_t = tf.placeholder(tf.float32, [None], name="reward")
          done_mask_ph = tf.placeholder(tf.float32, [None], name="done")

          # gamma
          pcont_t = tf.constant([self.gamma])
          pcont_t = tf.tile(pcont_t, tf.shape(r_t))
          pcont_t *= (1 - done_mask_ph) * pcont_t

          # target q values based on 1-step bellman (no double q for ddpg)
          l2_loss, loss_info = trfl.td_learning(tf.squeeze(critic_branch.value_fn, 1), r_t, pcont_t,
                                        tf.squeeze(target_critic_branch.value_fn, 1))
          tf_util.NOT_USED(loss_info)  # loss_info is named_tuple with target values and td_error
          mean_critic_loss = tf.reduce_mean(l2_loss)

          tf.summary.scalar("critic_td_error", tf.reduce_mean(loss_info.td_error))
          tf.summary.histogram("critic_td_error", loss_info.td_error)
          tf.summary.scalar("critic_loss", mean_critic_loss)

          #""" ACTOR LOSS """

          a_max = actor_branch.policy
          q_max = tf.squeeze(max_critic_branch.value_fn, 1)

          l2_loss, loss_info = trfl.dpg(q_max, a_max, dqda_clipping=self.grad_norm_clipping)
          tf_util.NOT_USED(loss_info)
          mean_actor_loss = tf.reduce_mean(l2_loss)

          tf.summary.scalar("actor_loss", mean_actor_loss)

          #""" TOTAL LOSS"""

          loss = mean_critic_loss * self.critic_lr / self.actor_lr + mean_actor_loss

          # compute optimization op (potentially with gradient clipping)
          optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)

          gradients = optimizer.compute_gradients(loss, var_list=main_vars)
          if self.grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
              if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)

          training_step = optimizer.apply_gradients(gradients)

        with tf.name_scope('update_target_network_ops'):
          update_target_network = []
          for var, var_target in zip(
              sorted(main_vars, key=lambda v: v.name), sorted(target_vars, key=lambda v: v.name)):
            new_target = self.target_network_update_frac       * var +\
                        (1 - self.target_network_update_frac) * var_target
            update_target_network.append(var_target.assign(new_target))
          update_target_network = tf.group(*update_target_network)

        with tf.variable_scope("input_info", reuse=False):
          tf.summary.scalar('rewards', tf.reduce_mean(r_t))
          tf.summary.histogram('rewards', r_t)
          if len(obs_ph.shape) == 3:
            tf.summary.image('observation', obs_ph)
          else:
            tf.summary.histogram('observation', obs_ph)

        self._act = act
        self._train_step = training_step
        self._summary_op = tf.summary.merge_all()
        self._obs1_ph = obs_ph
        self._action_ph = action_ph
        self._reward_ph = r_t
        self._obs2_ph = target_obs_ph
        self._dones_ph = done_mask_ph
        self._goal_ph = None  # TODO
        self.update_target_network = update_target_network
        self.model = actor_branch

        with tf.variable_scope("ddpg"):
          self.params = tf.trainable_variables()

        # Initialize the parameters and copy them to the target network.
        tf_util.initialize(self.sess)
        self.sess.run(self.update_target_network)

        self.summary = tf.summary.merge_all()

  def _setup_new_task(self, total_timesteps):
    """Sets up new task by reinitializing step, replay buffer, and exploration schedule"""
    self.task_step = 0
    self.reset = np.ones([self.n_envs, 1])

    items = [("observations0", self.observation_space.shape), ("actions", self.action_space.shape),
             ("rewards", (1,)), ("observations1", self.observation_space.shape), ("terminals1",
                                                                                  (1,))]
    if self.goal_space is not None:
      items += [("achieved_goal", self.env.observation_space.spaces['achieved_goal'].shape),
                ("desired_goal", self.env.observation_space.spaces['desired_goal'].shape)]

    self.replay_buffer = ReplayBuffer(self.buffer_size, items)

  def _get_action_for_single_obs(self, obs):
    """Called during training loop to get online action (with exploration)"""
    if self.goal_space is not None:
      feed_dict = {
          self.model.obs_ph: np.array(obs["observation"]),
          self.model.goal_ph: np.array(obs["desired_goal"]),
      }
    else:
      feed_dict = {
          self.model.obs_ph: np.array(obs), 
      }

    return self.sess.run(self._act, feed_dict=feed_dict)

  def _process_experience(self, obs, action, rew, new_obs, done):
    """Called during training loop after action is taken; includes learning;
              returns a summary"""
    
    done = np.expand_dims(done.astype(np.float32),1)
    rew = np.expand_dims(rew, 1)

    goal_agent = self.goal_space is not None

    # Reset the episode if done
    self.reset = done

    # Store transition in the replay buffer.
    if goal_agent:
      self.replay_buffer.add_batch(obs['observation'], action, rew, new_obs['observation'], \
                          done, new_obs['achieved_goal'], new_obs['desired_goal'])
    else:
      self.replay_buffer.add_batch(obs, action, rew, new_obs, done)

    summaries = []
    self.global_step += self.n_envs
    for i in range(self.n_envs):
      self.task_step += 1
      # If have enough data, train on it.
      if self.task_step > self.learning_starts:
        if self.task_step % self.train_freq == 0:
          if goal_agent:
            obses_t, actions, rewards, obses_tp1, dones, achieved_g, desired_g =\
                                                          self.replay_buffer.sample(self.batch_size)
          else:
            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)

          rewards = np.squeeze(rewards, 1)
          dones = np.squeeze(dones, 1)

          feed_dict = {
              self._obs1_ph: obses_t,
              self._action_ph: actions,
              self._reward_ph: rewards,
              self._obs2_ph: obses_tp1,
              self._dones_ph: dones
          }
          if goal_agent:
            feed_dict[self._goal_ph]: desired_g

          _, summary = self.sess.run([self._train_step, self._summary_op], feed_dict=feed_dict)
          summaries.append(summary)

        if self.task_step % self.target_network_update_freq == 0:
          self.sess.run(self.update_target_network)

    return summaries

  def predict(self, observation, state=None, mask=None, deterministic=True, goal=None):
    observation = np.array(observation)
    vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

    observation = observation.reshape((-1,) + self.observation_space.shape)
    actions = self.sess.run(self.model.policy, {self._obs1_ph: observation})

    if not vectorized_env:
        actions = actions[0]

    return actions, None

  def save(self, save_path):
    # Things set in the __init__ method should be saved here, because the model is called with default args on load(),
    # which are subsequently updated using this dict.
    data = {
        "gamma": self.gamma,
        'actor_lr': self.actor_lr,
        'critic_lr': self.critic_lr,
        'actor_policy': self.policy,
        'critic_policy': self.critic_policy,
        'joint_feature_extractor': self.joint_feature_extractor,
        'action_noise': self.action_noise,
        "param_noise": self.param_noise,
        "learning_starts": self.learning_starts,
        "train_freq": self.train_freq,
        "batch_size": self.batch_size,
        "buffer_size": self.buffer_size,
        "target_network_update_frac": self.target_network_update_frac,
        "target_network_update_freq": self.target_network_update_freq,
        "grad_norm_clipping": self.grad_norm_clipping,
        "tensorboard_log": self.tensorboard_log,
        "verbose": self.verbose,
        "observation_space": self.observation_space,
        "action_space": self.action_space,
        "policy": self.policy,
        "n_envs": self.n_envs,
        "_vectorize_action": self._vectorize_action
    }

    # Model paramaters to be restored
    params = self.sess.run(self.params)

    self._save_to_file(save_path, data=data, params=params)

def identity_extractor(input):
  trainable_variables = []
  return input, trainable_variables

def rescale_obs_ph(obs_ph, ob_space, rescale_input):
  if (rescale_input and not np.any(np.isinf(ob_space.low)) and
      not np.any(np.isinf(ob_space.high)) and np.any((ob_space.high - ob_space.low) != 0)):
    # equivalent to processed_x / 255.0 when bounds are set to [0, 255]
    processed_x = ((obs_ph - ob_space.low) / (ob_space.high - ob_space.low))
    return processed_x