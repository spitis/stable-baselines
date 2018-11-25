import tensorflow as tf
import numpy as np
import gym
import trfl
import copy

from types import FunctionType
from itertools import chain

from stable_baselines.common import tf_util, SimpleRLModel, SetVerbosity
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.replay_buffer import ReplayBuffer, EpisodicBuffer, her_final, her_future, her_future_with_states, HerFutureAchievedPastActual
from stable_baselines.common.landmark_generator import AbstractLandmarkGenerator
from stable_baselines.simple_ddpg.noise import OUNoiseTensorflow, NormalNoiseTensorflow
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
    :param joint_goal_feature_extractor: (function) extracts goal features to be shared by actor & critic (if None, joint_feature_extractor)

    :param rescale_input: (bool) whether or not to rescale the input to [0., 1.] (e.g., for images with values in [0,255])
    :param rescale_goal: (bool) whether or not to rescale the goal to [0., 1.] (if None, rescale_input)
    :param normalize_input: (bool) whether to  normalize input (default True)
    :param normalize_goal: (bool) whether to  normalize goal (if None, normalize_input)
    :param observation_pre_norm_range: (tuple) range for observation clipping
    :param observation_post_norm_range: (tuple) range for observation clipping after normalization
    :param goal_pre_norm_range: (tuple) range for goal clipping (if None, observation_pre_norm_range)
    :param goal_post_norm_range: (tuple) range for goal clipping after normalization (if None, observation_post_norm_range)
    :param clip_value_fn_range: (tuple) range for value function target clipping
    
    :param action_noise: (str) the noises ('normal' or 'ou') to use 
    :param epsilon_random_exploration: (float) the episilon to use for random exploration
    :param param_noise: (bool) whether to use parameter noise (not supported)
     
    :param buffer_size: (int) size of the replay goal_ph
    :param train_freq: (int) update the model every `train_freq` steps
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
 
    :param target_network_update_frac: (float) fraction by which to update the target network every time.
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.

    :param hindsight_mode: (str) e.g., "final", "none", "future_4"
    :param grad_norm_clipping: (float) amount of gradient norm clipping
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param eval_env: (env) the gym environment on which to run evaluations
    :param eval_every: (int) how often (in training episodes) to run an evaluation episode
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

  def __init__(self, policy, env, gamma=0.99, actor_lr=1e-4, critic_lr=1e-3, *, critic_policy=None, joint_feature_extractor=None, 
               joint_goal_feature_extractor=None, rescale_input=False, rescale_goal=None, normalize_input=True, normalize_goal=None, 
               observation_pre_norm_range=(-200., 200.), goal_pre_norm_range=None, observation_post_norm_range=(-5., 5.), 
               goal_post_norm_range=None, clip_value_fn_range=None, landmark_training=False, landmark_mode='unidirectional', 
               landmark_training_per_batch=1, landmark_width=1, landmark_generator=None, action_noise='ou_0.2', 
               epsilon_random_exploration=0., param_noise=False, buffer_size=50000, train_freq=1, batch_size=32, learning_starts=1000, 
               target_network_update_frac=0.001, target_network_update_freq=1, hindsight_mode=None, grad_norm_clipping=10.,
               critic_l2_regularization=1e-2, action_l2_regularization=0., verbose=0, tensorboard_log=None, eval_env=None, eval_every=10, 
               _init_setup_model=True):

    super(SimpleDDPG, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True)

    self.gamma = gamma
    self.actor_lr = actor_lr
    self.critic_lr = critic_lr

    self.actor_policy = self.policy
    self.critic_policy = self.policy if critic_policy is None else critic_policy
    if isinstance(self.critic_policy, str):
      self.critic_policy = get_policy_from_name(self.critic_policy)

    self.joint_feature_extractor = joint_feature_extractor or identity_extractor
    self.joint_goal_feature_extractor = joint_goal_feature_extractor or self.joint_feature_extractor

    self.rescale_input = rescale_input
    if rescale_goal is not None:
      self.rescale_goal = rescale_goal
    else:
      self.rescale_goal = self.rescale_input

    self.normalize_input = normalize_input
    if normalize_goal is not None:
      self.normalize_goal = normalize_goal
    else:
      self.normalize_goal = self.normalize_input
    self.obs_rms = None
    self.g_rms = None

    self.observation_pre_norm_range = observation_pre_norm_range
    if goal_pre_norm_range is not None:
      self.goal_pre_norm_range = goal_pre_norm_range
    else:
      self.goal_pre_norm_range = self.observation_pre_norm_range

    self.observation_post_norm_range = observation_post_norm_range
    if goal_post_norm_range is not None:
      self.goal_post_norm_range = goal_post_norm_range
    else:
      self.goal_post_norm_range = self.observation_post_norm_range

    self.clip_value_fn_range = clip_value_fn_range

    self.landmark_training = landmark_training
    self.landmark_mode = landmark_mode
    self.landmark_training_per_batch = landmark_training_per_batch
    self.landmark_width = landmark_width
    self.landmark_generator = landmark_generator
    
    self.action_noise = action_noise
    self.action_noise_fn = None
    self.epsilon_random_exploration = epsilon_random_exploration

    self.param_noise = param_noise
    if param_noise:
      raise NotImplementedError('param_noise to be added later')

    self.learning_starts = learning_starts
    self.train_freq = train_freq
    self.batch_size = batch_size
    self.buffer_size = buffer_size

    self.target_network_update_frac = target_network_update_frac
    self.target_network_update_freq = target_network_update_freq

    self.hindsight_mode = hindsight_mode
    self.hindsight_frac = 0.

    self.grad_norm_clipping = grad_norm_clipping
    self.critic_l2_regularization = critic_l2_regularization
    self.action_l2_regularization = action_l2_regularization

    self.tensorboard_log = tensorboard_log
    self.eval_env = eval_env
    self.eval_every = eval_every

    # Below props are set in self._setup_new_task()
    self.reset = None
    self.hindsight_subbuffer = None
    self.hindsight_fn = None
    self.global_step = 0
    self.task_step = 0
    self.replay_buffer = None
    self.replay_buffer_hindsight = None
    self.state_buffer = None
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
        ob_space, goal_space = self.observation_space, self.goal_space

        with tf.variable_scope("ddpg"):

          # input normalization statistics
          if self.normalize_input:
            with tf.variable_scope('obs_normalizer'):
              self.obs_rms = RunningMeanStd(shape=ob_space.shape)

            if goal_space is not None:
              with tf.variable_scope('goal_normalizer'):
                self.g_rms = RunningMeanStd(shape=self.goal_space.shape)

          # main network
          with tf.variable_scope('main_network', reuse=False):

            # main network placeholder/joint feature extraction
            obs_ph, update_obs_rms, _, main_input, main_joint_tvars = feature_extractor(
                reuse=False, space=ob_space, ph_name='obs_ph', rescale=self.rescale_input, 
                rms=self.obs_rms, make_rms_update_op=True, pre_norm_range=self.observation_pre_norm_range, 
                post_norm_range=self.observation_post_norm_range, joint_feature_extractor=self.joint_feature_extractor)

            goal_phs = goal_ph = landmark_state_ph = landmark_goal_ph = None
            update_g_rms = tf.no_op()

            # main network goal placeholder/joint feature extraction
            if self.goal_space is not None:
              goal_ph, update_g_rms, processed_g, main_goal, main_goal_joint_tvars = feature_extractor(
                  reuse=tf.AUTO_REUSE, space=goal_space, ph_name='goal_ph', rescale=self.rescale_goal, 
                  rms=self.g_rms, make_rms_update_op=True, pre_norm_range=self.goal_pre_norm_range, 
                  post_norm_range=self.goal_post_norm_range, joint_feature_extractor=self.joint_goal_feature_extractor)

              main_joint_tvars = list(set(main_joint_tvars + main_goal_joint_tvars))
              goal_phs = (goal_ph, main_goal)

              if self.landmark_training:
                landmark_state_ph, _, _, landmark_state, _ = feature_extractor(
                    reuse=True, space=ob_space, ph_name='landmark_state_ph', rescale=self.rescale_input, 
                    rms=self.obs_rms, make_rms_update_op=False, pre_norm_range=self.observation_pre_norm_range, 
                    post_norm_range=self.observation_post_norm_range, joint_feature_extractor=self.joint_feature_extractor)

                landmark_goal_ph, _, _, landmark_goal, _ = feature_extractor(
                    reuse=True, space=goal_space, ph_name='landmark_goal_ph', rescale=self.rescale_goal, 
                    rms=self.g_rms, make_rms_update_op=False, pre_norm_range=self.goal_pre_norm_range, 
                    post_norm_range=self.goal_post_norm_range, joint_feature_extractor=self.joint_goal_feature_extractor)

            action_ph = tf.placeholder(
                shape=(None, ) + self.action_space.shape, dtype=self.action_space.dtype, name='action_ph')

            # main network actor / critic branches
            with tf.variable_scope('actor', reuse=False):
              actor_branch = self.actor_policy(
                  self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None, obs_phs=(obs_ph, main_input), 
                  goal_phs=goal_phs, goal_space=self.goal_space)

            with tf.variable_scope('critic', reuse=False):
              critic_branch = self.critic_policy(
                  self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None, obs_phs=(obs_ph, main_input), 
                  goal_phs=goal_phs, goal_space=self.goal_space, action_ph=action_ph)

            with tf.variable_scope('critic', reuse=True):
              max_critic_branch = self.critic_policy(
                  self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None, obs_phs=(obs_ph, main_input), 
                  goal_phs=goal_phs, goal_space=self.goal_space, action_ph=actor_branch.policy)

            # landmark bound components
            if self.landmark_training:
              landmark_critics_s_lg = []
              landmark_critics_l_g  = []
              joined_landmark_state_and_goal = tf.concat([landmark_state, landmark_goal], axis=1)
              
              for k in range(self.landmark_training_per_batch):
                
                if k > 1:
                  shuffled_landmark_state_and_goal = tf.random_shuffle(joined_landmark_state_and_goal)
                else:
                  shuffled_landmark_state_and_goal = joined_landmark_state_and_goal

                landmark_state, landmark_goal = tf.split(shuffled_landmark_state_and_goal, (ob_space.shape[0], goal_space.shape[0]), 1)

                # v(s, lg)

                with tf.variable_scope('critic', reuse=True):
                  landmark_critic_s_lg = self.critic_policy(
                      self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None, obs_phs=(obs_ph, main_input), 
                      goal_phs=(landmark_goal_ph, landmark_goal), goal_space=self.goal_space, action_ph=action_ph)
                  
                landmark_critics_s_lg.append(landmark_critic_s_lg)

                # v(l, g)

                with tf.variable_scope('actor', reuse=True):
                  landmark_actor_l_g = self.actor_policy(
                      self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None, obs_phs=(landmark_state_ph, 
                      landmark_state), goal_phs=goal_phs, goal_space=self.goal_space)

                with tf.variable_scope('critic', reuse=True):
                  landmark_critic_l_g = self.critic_policy(
                      self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None, obs_phs=(landmark_state_ph, 
                      landmark_state), goal_phs=goal_phs, goal_space=self.goal_space, action_ph=tf.stop_gradient(landmark_actor_l_g.policy))

                landmark_critics_l_g.append(landmark_critic_l_g)

            # exploratory action computation (and normalization statistic collection)
            if not self.param_noise:
              noise, sigma = self.action_noise.split('_')
              sigma = float(sigma)
              if noise == 'ou':
                self.action_noise_fn = OUNoiseTensorflow(self.action_space.shape[-1], sigma=float(sigma))
              elif noise == 'normal':
                self.action_noise_fn = NormalNoiseTensorflow(sigma=sigma)
              with tf.control_dependencies([update_obs_rms, update_g_rms]):
                act = self.action_noise_fn(actor_branch.policy)
                act = epsilon_exploration_wrapper_box(self.epsilon_random_exploration, self.action_space, act)

                assert np.allclose(self.action_space.low, self.action_space.low[0])
                assert np.allclose(self.action_space.high, self.action_space.high[0])

                act = tf.clip_by_value(act, self.action_space.low[0], self.action_space.high[0])
            else:
              raise NotImplementedError('param_noise to be added later')

          # target network placeholders & feature extraction
          with tf.variable_scope('target_network', reuse=False):
            target_obs_ph, _, _, target_input, target_joint_tvars = feature_extractor(
                reuse=False,
                space=ob_space,
                ph_name='target_obs_ph',
                rescale=self.rescale_input,
                rms=self.obs_rms,
                make_rms_update_op=False,
                pre_norm_range=self.observation_pre_norm_range,
                post_norm_range=self.observation_post_norm_range,
                joint_feature_extractor=self.joint_feature_extractor)

            target_goal_phs = None
            if self.goal_space is not None:
              with tf.variable_scope('feature_extraction', reuse=tf.AUTO_REUSE):
                target_goal, main_goal_joint_tvars = self.joint_goal_feature_extractor(processed_g)

              target_joint_tvars = list(set(target_joint_tvars + main_goal_joint_tvars))
              target_goal_phs = (goal_ph, target_goal)

            # target network actor/critic branches
            with tf.variable_scope('actor', reuse=False):
              target_actor_branch = self.actor_policy(
                  self.sess, ob_space, self.action_space, n_env=1, n_steps=1, n_batch=None,
                  obs_phs=(target_obs_ph, target_input), goal_phs=target_goal_phs, goal_space=self.goal_space)

            with tf.variable_scope('critic', reuse=False):
              target_critic_branch = self.critic_policy(
                  self.sess,
                  ob_space,
                  self.action_space,
                  n_env=1,
                  n_steps=1,
                  n_batch=None,
                  obs_phs=(target_obs_ph, target_input),
                  goal_phs=target_goal_phs,
                  goal_space=self.goal_space,
                  action_ph=target_actor_branch.policy)

          # variables for main and target networks
          main_vars = main_joint_tvars + actor_branch.trainable_vars + critic_branch.trainable_vars
          target_vars = target_joint_tvars + target_actor_branch.trainable_vars +\
                        target_critic_branch.trainable_vars

        with tf.variable_scope("loss"):

          #""" CRITIC LOSS """

          # placeholders for bellman equation
          r_t = tf.placeholder(tf.float32, [None], name="reward")
          done_mask_ph = tf.placeholder(tf.float32, [None], name="done")

          # gamma
          pcont_t = tf.constant([self.gamma])
          pcont_t = tf.tile(pcont_t, tf.shape(r_t))
          pcont_t *= (1 - done_mask_ph) * pcont_t

          # target q values based on 1-step bellman (no double q for ddpg)
          preds = tf.squeeze(critic_branch.value_fn, 1)
          l2_loss, loss_info = trfl.td_learning(preds, r_t, pcont_t, tf.squeeze(target_critic_branch.value_fn, 1))
          tf_util.NOT_USED(l2_loss)
          targets = loss_info.target
          if self.clip_value_fn_range is not None:
            targets = tf.clip_by_value(targets, self.clip_value_fn_range[0], self.clip_value_fn_range[1])
          mean_critic_loss = tf.reduce_mean(0.5 * tf.square(preds - targets))

          # add regularizer
          if self.critic_l2_regularization > 0.:
            mean_critic_loss += tf.contrib.layers.apply_regularization(
                tf.contrib.layers.l2_regularizer(self.critic_l2_regularization), critic_branch.trainable_vars)

          # landmark training
          landmark_scores = None
          if self.landmark_training:
            landmark_losses = []
            
            for k in range(self.landmark_training_per_batch):
              if self.landmark_mode == 'unidirectional':
                landmark_lower_bound = tf.stop_gradient(
                    landmark_critics_s_lg[k].value_fn * landmark_critics_l_g[k].value_fn * (self.gamma  ** self.landmark_width))
              elif self.landmark_mode == 'bidirectional':
                landmark_lower_bound = landmark_critics_s_lg[k].value_fn * landmark_critics_l_g[k].value_fn * (self.gamma  ** self.landmark_width)
              else:
                raise ValueError('landmark_mode must be one of "unidirectional" or "bidirectional"')
              landmark_losses.append(tf.maximum(0., landmark_lower_bound - critic_branch.value_fn))
              if k == 0:
                landmark_scores = landmark_lower_bound / critic_branch.value_fn

            landmark_losses = tf.concat(landmark_losses, 0)
            tf.summary.histogram('landmark_losses', landmark_losses)

            mean_landmark_loss = self.landmark_training * tf.reduce_mean(landmark_losses)

          tf.summary.scalar("critic_td_error", tf.reduce_mean(loss_info.td_error))
          tf.summary.histogram("critic_td_error", loss_info.td_error)
          tf.summary.scalar("critic_loss", mean_critic_loss)
          if self.landmark_training:
            tf.summary.scalar("landmark_loss", mean_landmark_loss)

          #""" ACTOR LOSS """

          a_max = actor_branch.policy
          q_max = tf.squeeze(max_critic_branch.value_fn, 1)

          l2_loss, loss_info = trfl.dpg(q_max, a_max, dqda_clipping=self.grad_norm_clipping)
          tf_util.NOT_USED(loss_info)
          mean_actor_loss = tf.reduce_mean(l2_loss)

          # add action norm regularizer
          if self.action_l2_regularization > 0.:
            mean_actor_loss += self.action_l2_regularization * tf.reduce_mean(
                tf.square((actor_branch.unadjusted_policy - 0.5) * 2))

          tf.summary.scalar("actor_loss", mean_actor_loss)


          if self.landmark_training:
            total_loss = (mean_critic_loss + mean_landmark_loss) * self.critic_lr / self.actor_lr + mean_actor_loss
          else:
            total_loss = mean_critic_loss * self.critic_lr / self.actor_lr + mean_actor_loss

        with tf.variable_scope("optimization"):

          # compute optimization op (potentially with gradient clipping)
          optimizer = tf.train.AdamOptimizer(learning_rate=self.actor_lr)

          gradients = optimizer.compute_gradients(
              total_loss, var_list=main_joint_tvars + actor_branch.trainable_vars + critic_branch.trainable_vars)
          if self.grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
              if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)

          training_step = optimizer.apply_gradients(gradients)

        with tf.name_scope('update_target_network_ops'):
          init_target_network = []
          update_target_network = []
          for var, var_target in zip(
              sorted(main_vars, key=lambda v: v.name), sorted(target_vars, key=lambda v: v.name)):
            new_target = self.target_network_update_frac       * var +\
                         (1 - self.target_network_update_frac) * var_target
            update_target_network.append(var_target.assign(new_target))
            init_target_network.append(var_target.assign(var))
          update_target_network = tf.group(*update_target_network)
          init_target_network = tf.group(*init_target_network)

        with tf.variable_scope("input_info", reuse=False):
          for action_dim in range(self.action_space.shape[0]):
            tf.summary.histogram('policy_dim_{}'.format(action_dim), a_max[:, action_dim])
            tf.summary.histogram('explor_dim_{}'.format(action_dim), action_ph[:, action_dim])
          tf.summary.histogram('targets', targets)
          tf.summary.scalar('rewards', tf.reduce_mean(r_t))
          tf.summary.histogram('rewards', r_t)
          if len(obs_ph.shape) == 3:
            tf.summary.image('observation', obs_ph)
          else:
            tf.summary.histogram('observation', obs_ph)

        with tf_util.COMMENT("attribute assignments:"):
          self._act = act
          self._train_step = training_step
          self._summary_op = tf.summary.merge_all()
          self._obs1_ph = obs_ph
          self._action_ph = action_ph
          self._reward_ph = r_t
          self._obs2_ph = target_obs_ph
          self._dones_ph = done_mask_ph
          self._goal_ph = goal_ph
          self._landmark_state_ph = landmark_state_ph
          self._landmark_goal_ph = landmark_goal_ph
          self._landmark_scores = landmark_scores
          self.update_target_network = update_target_network
          self.model = actor_branch
          self.target_model = target_actor_branch

          self.params = tf.global_variables("ddpg")

        with tf_util.COMMENT("graph initialization"):
          # Initialize the parameters and copy them to the target network.
          tf_util.initialize(self.sess)
          self.sess.run(init_target_network)

          self.summary = tf.summary.merge_all()

  def _setup_new_task(self, total_timesteps):
    """Sets up new task by reinitializing step, replay buffer, and exploration schedule"""
    self.task_step = 0
    self.reset = np.ones([self.n_envs, 1])

    if self.hindsight_mode == 'final':
      self.hindsight_fn = lambda trajectory: her_final(trajectory, self.env.compute_reward)
      self.hindsight_frac = 0.5
    elif isinstance(self.hindsight_mode, str) and 'future_' in self.hindsight_mode:
      _, k = self.hindsight_mode.split('_')
      self.hindsight_fn = lambda trajectory: her_future(trajectory, int(k), self.env.compute_reward)
      self.hindsight_frac = 1. - 1. / (1. + float(k))
    elif isinstance(self.hindsight_mode, str) and 'futureactual_' in self.hindsight_mode:
      _, k, p = self.hindsight_mode.split('_')
      self.hindsight_fn = HerFutureAchievedPastActual(int(k), int(p), self.env.compute_reward)
      self.hindsight_frac = 1. - 1. / (1. + float(k + p))
    else:
      self.hindsight_fn = None

    items = [("observations0", self.observation_space.shape), ("actions", self.action_space.shape), ("rewards", (1, )),
             ("observations1", self.observation_space.shape), ("terminals1", (1, ))]

    hindsight_items = copy.deepcopy(items)

    if self.goal_space is not None:
      items += [("desired_goal", self.env.observation_space.spaces['desired_goal'].shape)]

      hindsight_items += [("desired_goal", self.env.observation_space.spaces['desired_goal'].shape)]

      if self.landmark_training:
        if isinstance(self.landmark_generator, FunctionType):
          self.landmark_generator = self.landmark_generator(self.buffer_size, self.env)
        elif self.landmark_generator is None:
          self.state_buffer = ReplayBuffer(self.buffer_size,
                                        [("state", self.env.observation_space.spaces['observation'].shape),
                                        ("as_goal", self.env.observation_space.spaces['achieved_goal'].shape)])
        else:
          assert isinstance(self.landmark_generator, AbstractLandmarkGenerator)

    self.replay_buffer = ReplayBuffer(int((1 - self.hindsight_frac) * self.buffer_size), items)
    if self.hindsight_fn is not None:
      self.replay_buffer_hindsight = ReplayBuffer(int(self.hindsight_frac * self.buffer_size), hindsight_items)

    self.hindsight_subbuffer = EpisodicBuffer(self.n_envs, self.hindsight_fn, n_cpus=min(self.n_envs, 8))

  def _get_action_for_single_obs(self, obs):
    """Called during training loop to get online action (with exploration)"""
    if self.goal_space is not None:
      feed_dict = {
          self.model.obs_ph: np.array(obs["observation"]),
          self.model.goal_ph: np.array(obs["desired_goal"]),
          self.action_noise_fn.reset_noise_ph: self.reset,
      }
    else:
      feed_dict = {
          self.model.obs_ph: np.array(obs),
          self.action_noise_fn.reset_noise_ph: self.reset,
      }

    return self.sess.run(self._act, feed_dict=feed_dict)

  def _process_experience(self, obs, action, rew, new_obs, done):
    """Called during training loop after action is taken; includes learning;
              returns a summary"""

    expanded_done = np.expand_dims(done.astype(np.float32), 1)
    rew = np.expand_dims(rew, 1)

    goal_agent = self.goal_space is not None

    # Reset the episode if done
    self.reset = expanded_done

    # Store transition in the replay buffer, and hindsight subbuffer
    if goal_agent:
      self.replay_buffer.add_batch(obs['observation'], action, rew, new_obs['observation'], expanded_done,
                                   new_obs['desired_goal'])

      if self.landmark_training:
        if self.landmark_generator is not None:
          self.landmark_generator.add_state_data(obs['observation'], obs['achieved_goal'])
        else:
          self.state_buffer.add_batch(obs['observation'], obs['achieved_goal'])

    else:
      self.replay_buffer.add_batch(obs, action, rew, new_obs, expanded_done)

    if goal_agent and self.hindsight_fn is not None:
      for idx in range(self.n_envs):
        # add the transition to the HER subbuffer for that worker
        self.hindsight_subbuffer.add_to_subbuffer(idx, [
            obs['observation'][idx], action[idx], rew[idx], new_obs['observation'][idx], new_obs['achieved_goal'][idx],
            new_obs['desired_goal'][idx]
        ])
        if done[idx]:
          # commit the subbuffer
          self.hindsight_subbuffer.commit_subbuffer(idx)
          if len(self.hindsight_subbuffer) == self.n_envs:
            for hindsight_experience in chain.from_iterable(self.hindsight_subbuffer.process_trajectories()):
              self.replay_buffer_hindsight.add(*hindsight_experience)
            self.hindsight_subbuffer.clear_main_buffer()

    summaries = []
    self.global_step += self.n_envs
    for _ in range(self.n_envs):
      self.task_step += 1
      # If have enough data, train on it.
      if self.task_step > self.learning_starts:
        if self.task_step % self.train_freq == 0:
          if goal_agent:
            if self.replay_buffer_hindsight is not None and len(
                self.replay_buffer_hindsight) and self.hindsight_frac > 0.:
              hindsight_batch_size = round(self.batch_size * self.hindsight_frac)
              real_batch_size = self.batch_size - hindsight_batch_size

              # Sample from real batch
              obses_t, actions, rewards, obses_tp1, dones, desired_g = \
                  self.replay_buffer.sample(real_batch_size)

              # Sample from hindsight batch
              obses_t_hs, actions_hs, rewards_hs, obses_tp1_hs, dones_hs, desired_g_hs =\
                self.replay_buffer_hindsight.sample(hindsight_batch_size)

              # Concatenate the two
              obses_t = np.concatenate([obses_t, obses_t_hs], 0)
              actions = np.concatenate([actions, actions_hs], 0)
              rewards = np.concatenate([rewards, rewards_hs], 0)
              obses_tp1 = np.concatenate([obses_tp1, obses_tp1_hs], 0)
              dones = np.concatenate([dones, dones_hs], 0)

              desired_g = np.concatenate([desired_g, desired_g_hs], 0)

            else:
              obses_t, actions, rewards, obses_tp1, dones, desired_g =\
                                                          self.replay_buffer.sample(self.batch_size)
              desired_g_hs = None

            if self.landmark_training:
              if self.landmark_generator is not None:
                landmark_states, landmark_goals = self.landmark_generator.generate(obses_t, desired_g)
              else:
                landmark_states, landmark_goals = self.state_buffer.sample(self.batch_size)

          else:
            obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)

          rewards = np.squeeze(rewards, 1)
          dones = np.squeeze(dones, 1)

          feed_dict = {
              self._obs1_ph: obses_t,
              self._action_ph: actions,
              self._reward_ph: rewards,
              self._obs2_ph: obses_tp1,
              self._dones_ph: dones,
          }

          if goal_agent:
            feed_dict[self._goal_ph] = desired_g

            if self.landmark_training:
              feed_dict[self._landmark_state_ph] = landmark_states
              feed_dict[self._landmark_goal_ph] = landmark_goals

          if self.landmark_generator is not None:
            _, landmark_scores, summary = self.sess.run([self._train_step, self._landmark_scores, self._summary_op], 
              feed_dict=feed_dict)
            self.landmark_generator.assign_scores(landmark_scores)

          else:
            _, summary = self.sess.run([self._train_step, self._summary_op], feed_dict=feed_dict)
          summaries.append(summary)

        if self.task_step % self.target_network_update_freq == 0:
          self.sess.run(self.update_target_network)

    return summaries

  def predict(self, observation, state=None, mask=None, deterministic=True, goal=None):
    goal_agent = self.goal_space is not None
    if not goal_agent:
      observation = np.array(observation)
    else:
      desired_goal = np.array(observation['desired_goal'])
      desired_goal = desired_goal.reshape((-1, ) + self.goal_space.shape)
      observation = np.array(observation['observation'])

    vectorized_env = self._is_vectorized_observation(observation, self.observation_space)
    observation = observation.reshape((-1, ) + self.observation_space.shape)

    if goal_agent:
      actions = self.sess.run(self.model.policy, {self._obs1_ph: observation, self._goal_ph: desired_goal})
    else:
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
        'joint_goal_feature_extractor': self.joint_goal_feature_extractor,
        'rescale_input': self.rescale_input,
        'rescale_goal': self.rescale_goal,
        'normalize_input': self.normalize_input,
        'normalize_goal': self.normalize_goal,
        'observation_pre_norm_range': self.observation_pre_norm_range,
        'observation_post_norm_range': self.observation_post_norm_range,
        'goal_pre_norm_range': self.goal_pre_norm_range,
        'goal_post_norm_range': self.goal_post_norm_range,
        'clip_value_fn_range': self.clip_value_fn_range,
        'landmark_training': self.landmark_training,
        'landmark_mode': self.landmark_mode,
        'landmark_training_per_batch': self.landmark_training_per_batch,
        'landmark_width': self.landmark_width,
        'landmark_generator': self.landmark_generator,
        'action_noise': self.action_noise,
        'epsilon_random_exploration': self.epsilon_random_exploration,
        "param_noise": self.param_noise,
        "learning_starts": self.learning_starts,
        "train_freq": self.train_freq,
        "batch_size": self.batch_size,
        "buffer_size": self.buffer_size,
        "target_network_update_frac": self.target_network_update_frac,
        "target_network_update_freq": self.target_network_update_freq,
        'hindsight_mode': self.hindsight_mode,
        "grad_norm_clipping": self.grad_norm_clipping,
        'critic_l2_regularization': self.critic_l2_regularization,
        'action_l2_regularization': self.action_l2_regularization,
        "tensorboard_log": self.tensorboard_log,
        "verbose": self.verbose,
        "observation_space": self.observation_space,
        "action_space": self.action_space,
        "goal_space": self.goal_space,
        "policy": self.policy,
        "n_envs": self.n_envs,
        "_vectorize_action": self._vectorize_action,
        'eval_env': self.eval_env,
        'eval_every': self.eval_every,
    }

    # Model paramaters to be restored
    params = self.sess.run(self.params)

    self._save_to_file(save_path, data=data, params=params)


def normalize(tensor, stats):
  """
    normalize a tensor using a running mean and std

    :param tensor: (TensorFlow Tensor) the input tensor
    :param stats: (RunningMeanStd) the running mean and std of the input to normalize
    :return: (TensorFlow Tensor) the normalized tensor
    """
  if stats is None:
    return tensor
  return (tensor - stats.mean) / stats.std


def identity_extractor(input_tensor):
  trainable_variables = []
  return input_tensor, trainable_variables


def make_feedforward_extractor(layers=[256], activation=tf.nn.relu, scope=None):
  def feed_forward_extractor(input_tensor):
    if scope is not None:
      with tf.variable_scope(scope):
        tvars = []
        x = input_tensor
        for layer_size in layers:
          layer = tf.layers.Dense(layer_size, activation)
          x = layer(x)
          tvars += layer.trainable_variables
        return x, tvars
    else:
      tvars = []
      x = input_tensor
      for layer_size in layers:
        layer = tf.layers.Dense(layer_size, activation)
        x = layer(x)
        tvars += layer.trainable_variables
      return x, tvars

  return feed_forward_extractor


def rescale_obs_ph(obs_ph, ob_space, rescale_input):
  processed_x = tf.to_float(obs_ph)
  if (rescale_input and not np.any(np.isinf(ob_space.low)) and not np.any(np.isinf(ob_space.high))
      and np.any((ob_space.high - ob_space.low) != 0)):
    # equivalent to processed_x / 255.0 when bounds are set to [0, 255]
    processed_x = ((obs_ph - ob_space.low) / (ob_space.high - ob_space.low))
  return processed_x


def feature_extractor(reuse, space, ph_name, rescale, rms, make_rms_update_op, pre_norm_range, post_norm_range,
                      joint_feature_extractor):
  with tf.variable_scope('feature_extraction', reuse=reuse):
    obs_ph = tf.placeholder(shape=(None, ) + space.shape, dtype=space.dtype, name=ph_name)
    processed_x = rescale_obs_ph(tf.to_float(obs_ph), space, rescale)
    processed_x = tf.clip_by_value(processed_x, pre_norm_range[0], pre_norm_range[1])
    if make_rms_update_op is not None:
      update_rms = rms.make_update_op(processed_x)
    else:
      update_rms = None
    processed_x = normalize(processed_x, rms)
    processed_x = tf.clip_by_value(processed_x, post_norm_range[0], post_norm_range[1])

    ouput, joint_tvars = joint_feature_extractor(processed_x)

    return obs_ph, update_rms, processed_x, ouput, joint_tvars


class RunningMeanStd(object):
  def __init__(self, epsilon=1e-2, shape=()):
    """
        calulates the running mean and std of a data stream
        https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm

        :param epsilon: (float) helps with arithmetic issues
        :param shape: (tuple) the shape of the data stream's output
        """
    self._sum = tf.get_variable(
        dtype=tf.float64, shape=shape, initializer=tf.constant_initializer(0.0), name="runningsum", trainable=False)
    self._sumsq = tf.get_variable(
        dtype=tf.float64,
        shape=shape,
        initializer=tf.constant_initializer(epsilon),
        name="runningsumsq",
        trainable=False)
    self._count = tf.get_variable(
        dtype=tf.float64, shape=(), initializer=tf.constant_initializer(epsilon), name="count", trainable=False)
    self.shape = shape

    self.mean = tf.to_float(self._sum / self._count)
    self.std = tf.sqrt(tf.maximum(tf.to_float(self._sumsq / self._count) - tf.square(self.mean), 1e-2))

  def make_update_op(self, batched_unnormalized_data):
    """
        update the running mean and std using batched unnormalized data
        (batch here means the shape is prefixed by an extra batch_size axis)

        :param batched_unnormalized_data: (tensor)
        """
    data = tf.to_double(batched_unnormalized_data)
    summed_data = tf.reduce_sum(data, axis=0)
    sumsqed_data = tf.reduce_sum(tf.square(data), axis=0)
    count_data = tf.to_double(tf.shape(data)[0])

    update_sum = tf.assign_add(self._sum, summed_data)
    update_sumsq = tf.assign_add(self._sumsq, sumsqed_data)
    update_count = tf.assign_add(self._count, count_data)

    return tf.group([update_sum, update_sumsq, update_count])


def epsilon_exploration_wrapper_box(epsilon, action_space, policy_action):
  """
  Epsilon % takes random action in the action_space, else takes policy_action.
  """
  assert isinstance(action_space, gym.spaces.Box)

  low_as_batch = tf.expand_dims(action_space.low, 0)
  high_as_batch = tf.expand_dims(action_space.high, 0)

  batch_size = tf.shape(policy_action)[0]
  random_actions = tf.random_uniform(tf.shape(policy_action)) * (high_as_batch - low_as_batch) + low_as_batch
  chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < epsilon
  return tf.where(chose_random, random_actions, policy_action)
