import tensorflow as tf
import numpy as np
import gym
import trfl
import copy

from types import FunctionType
from itertools import chain

from stable_baselines.common import tf_util, SimpleRLModel, SetVerbosity
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.replay_buffer import ReplayBuffer, EpisodicBuffer, her_final, her_future, her_future_landmark,\
                                                   HerFutureAchievedPastActual, HerFutureAchievedPastActualLandmark
from stable_baselines.common.policies import observation_input
from stable_baselines.common.landmark_generator import AbstractLandmarkGenerator

class SimpleDQN(SimpleRLModel):
  """
    Simplified version of DQN model class. DQN paper: https://arxiv.org/pdf/1312.5602.pdf

    :param policy: (BasePolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer

    :param exploration_fraction: (float) fraction of entire training period over which gamme is annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param param_noise: (bool) Whether or not to apply noise to the parameters of the policy.
        
    :param buffer_size: (int) size of the replay buffer
    :param train_freq: (int) update the model every `train_freq` steps
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts

    :param target_network_update_frac: (float) fraction by which to update the target network every time.
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.

    :param hindsight_mode: (str) e.g., "final", "none", "future_4"

    :param double_q: (bool) whether to use double q learning
    :param grad_norm_clipping: (float) amount of gradient norm clipping

    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

  def __init__(self,
               policy,
               env,
               gamma=0.99,
               learning_rate=5e-4,
               *,
               exploration_fraction=0.1,
               exploration_final_eps=0.02,
               param_noise=False,
               buffer_size=50000,
               train_freq=1,
               batch_size=32,
               learning_starts=1000,
               target_network_update_frac=1.,
               target_network_update_freq=500,
               hindsight_mode=None,
               hindsight_frac=0.,
               landmark_training=False,
               landmark_training_per_batch=1,
               landmark_generator=None,
               landmark_mode='unidirectional',
               landmark_error='linear',
               landmark_width=1,
               double_q=True,
               grad_norm_clipping=10.,
               verbose=0,
               tensorboard_log=None,
               eval_env=None,
               eval_every=10,
               _init_setup_model=True):

    super(SimpleDQN, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=True)

    self.learning_rate = learning_rate
    self.gamma = gamma

    self.exploration_final_eps = exploration_final_eps
    self.exploration_fraction = exploration_fraction
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
    self.hindsight_frac = hindsight_frac

    self.landmark_training = landmark_training
    self.landmark_training_per_batch = landmark_training_per_batch
    self.landmark_generator = landmark_generator
    self.landmark_mode = landmark_mode
    self.landmark_error = landmark_error
    self.landmark_width = landmark_width

    self.double_q = double_q
    self.grad_norm_clipping = grad_norm_clipping

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
    # The reason for _init_setup_model = False is to set the action/env space from a saved model, without
    # loading an environment (e.g., to do transfer learning)
    if _init_setup_model:
      self._setup_model()

  def _setup_model(self):
    with SetVerbosity(self.verbose):
      assert isinstance(self.action_space, gym.spaces.Discrete), \
          "Error: SimpleDQN only supports gym.spaces.Discrete action space."

      self.graph = tf.Graph()
      with self.graph.as_default():
        self.sess = tf_util.make_session(graph=self.graph)
        ob_space = self.observation_space

        with tf.variable_scope("deepq"):

          # policy function
          policy = self.policy(
                self.sess,
                self.observation_space,
                self.action_space,
                n_env=self.n_envs,
                n_steps=1,
                n_batch=None,
                is_DQN=True,
                goal_space=self.goal_space)

          # exploration placeholders & online action with exploration noise
          epsilon_ph = tf.placeholder_with_default(0., shape=(), name="epsilon_ph")
          threshold_ph = tf.placeholder_with_default(0., shape=(), name="param_noise_threshold_ph")
          reset_ph = tf.placeholder(tf.float32, shape=[None, 1], name="reset_ph")

          # Set these to None for now
          goal_phs = goal_ph = goal_or_goalstate_ph = landmark_state_ph = landmark_goal_ph = None

          if not self.param_noise:
            act = epsilon_greedy_wrapper(policy, epsilon_ph)
          else:
            act = param_noise_wrapper(policy, reset_ph=reset_ph, threshold_ph=threshold_ph)

          # create target q network for training
          with tf.variable_scope("target_q_func", reuse=False):
            target_policy = self.policy(
                self.sess,
                self.observation_space,
                self.action_space,
                self.n_envs,
                1,
                None,
                reuse=False,
                is_DQN=True,
                goal_space=self.goal_space)

          # setup double q network; because of the outer_scope_getter, this reuses policy variables
          with tf.variable_scope("double_q", reuse=True, custom_getter=tf_util.outer_scope_getter("double_q")):
            double_policy = self.policy(
                self.sess,
                self.observation_space,
                self.action_space,
                self.n_envs,
                1,
                None,
                reuse=True,
                obs_phs=(target_policy.obs_ph, target_policy.processed_x),
                is_DQN=True,
                goal_space=self.goal_space,
                goal_phs=(target_policy.goal_ph, target_policy.processed_g))

          # landmark placeholder and processing
          if self.landmark_training:
            landmarks_q_s_lg = []
            landmarks_q_l_g = []

            # HC: Sketchy...borrowing processing code from Policies
            # and assuming goal processing is the same as observation processing
            landmark_goal_ph, landmark_goal = observation_input(self.goal_space, batch_size=None, scale=policy.scale,
                                                                name='landmark_goal')
            landmark_state_ph, landmark_state = observation_input(ob_space, batch_size=None, scale=policy.scale,
                                                                  name='landmark_state')

            joined_landmark_state_and_goal = tf.concat([landmark_state, landmark_goal], axis=1)

            for k in range(self.landmark_training_per_batch):

              if k > 1:
                  shuffled_landmark_state_and_goal = tf.random_shuffle(joined_landmark_state_and_goal)
              else:
                  shuffled_landmark_state_and_goal = joined_landmark_state_and_goal

              landmark_state, landmark_goal = tf.split(shuffled_landmark_state_and_goal,
                                                       (ob_space.shape[0], self.goal_space.shape[0]), 1)

              # Q(s, a, lg)
              with tf.variable_scope("q_landmark_s_lg", reuse=True, custom_getter=tf_util.outer_scope_getter("q_landmark_s_lg")):
                landmark_q_s_lg = self.policy(
                  self.sess,
                  self.observation_space,
                  self.action_space,
                  self.n_envs,
                  1,
                  None,
                  reuse=True,
                  obs_phs=(policy.obs_ph, policy.processed_x),
                  is_DQN=True,
                  goal_space=self.goal_space,
                  goal_phs=(landmark_goal_ph, landmark_goal))

              landmarks_q_s_lg.append(landmark_q_s_lg)

              # Q(l, a*, g)
              with tf.variable_scope("q_landmark_l_g", reuse=True, custom_getter=tf_util.outer_scope_getter("q_landmark_l_g")):
                landmark_q_l_g = self.policy(
                  self.sess,
                  self.observation_space,
                  self.action_space,
                  self.n_envs,
                  1,
                  None,
                  reuse=True,
                  obs_phs=(landmark_state_ph, landmark_state),
                  is_DQN=True,
                  goal_space=self.goal_space,
                  goal_phs=(policy.goal_ph, policy.processed_g))

              landmarks_q_l_g.append(landmark_q_l_g)

        with tf.variable_scope("loss"):

          # note: naming conventions from trfl (see https://github.com/deepmind/trfl/blob/master/docs/index.md)

          # placeholders for bellman equation
          a_tm1 = tf.placeholder(tf.int32, [None], name="action")
          r_t = tf.placeholder(tf.float32, [None], name="reward")
          done_mask_ph = tf.placeholder(tf.float32, [None], name="done")

          # gamma
          pcont_t = tf.constant([self.gamma])
          pcont_t = tf.tile(pcont_t, tf.shape(r_t))
          pcont_t *= (1 - done_mask_ph) * pcont_t

          # target q values based on 1-step bellman
          if self.double_q:
            l2_loss, loss_info = trfl.double_qlearning(policy.q_values, a_tm1, r_t, pcont_t, target_policy.q_values,
                                                       double_policy.q_values)
          else:
            l2_loss, loss_info = trfl.qlearning(policy.q_values, a_tm1, r_t, pcont_t, target_policy.q_values)

          tf_util.NOT_USED(l2_loss)  # because using huber loss (next line)

          mean_huber_loss = tf.reduce_mean(tf_util.huber_loss(loss_info.td_error))

          landmark_scores, landmark_ratios = None, None
          if self.landmark_training:
            landmark_losses = []

            for k in range(self.landmark_training_per_batch):
              # Q(s_t, a_t, l_g)
              qa_tm1_s_lg = trfl.indexing_ops.batched_index(landmarks_q_s_lg[k].q_values, a_tm1)

              # max_a Q(l, a, g)
              q_l_g = tf.reduce_max(landmarks_q_l_g[k].q_values, axis=1)

              # Q(s_t, a_t, g)
              qa_tm1_g = trfl.indexing_ops.batched_index(policy.q_values, a_tm1)

              if self.landmark_mode == 'unidirectional':
                landmark_lower_bound = tf.stop_gradient(qa_tm1_s_lg * q_l_g * (self.gamma ** self.landmark_width))

              elif self.landmark_mode == 'bidirectional':
                landmark_lower_bound = qa_tm1_s_lg * q_l_g * (self.gamma ** self.landmark_width)
              else:
                raise ValueError('landmark_mode must be one of "unidirectional" or "bidirectional"')

              if self.landmark_error == 'linear':
                landmark_losses.append(tf.maximum(0., landmark_lower_bound - qa_tm1_g))
              elif self.landmark_error == 'squared':
                landmark_losses.append(tf.square(tf.maximum(0., landmark_lower_bound - qa_tm1_g)))
              else:
                raise ValueError('Unsupported landmark_error!')

              if k == 0:
                landmark_scores = tf.expand_dims(tf.clip_by_value(landmark_lower_bound / qa_tm1_g, 0., 1.05), 1)
                landmark_ratios = tf.log(
                  tf.clip_by_value(landmarks_q_s_lg[k].value_fn / landmarks_q_l_g[k].value_fn, 1e-1, 1e1))
                tf.summary.histogram('landmark_scores', landmark_scores)
                tf.summary.histogram('landmark_ratios', landmark_ratios)

            landmark_losses = tf.concat(landmark_losses, 0)
            tf.summary.histogram('landmark_losses', landmark_losses)
            mean_huber_loss += self.landmark_training * tf.reduce_mean(landmark_losses)

          tf.summary.scalar("td_error", tf.reduce_mean(loss_info.td_error))
          tf.summary.histogram("td_error", loss_info.td_error)
          tf.summary.scalar("loss", mean_huber_loss)
          if self.landmark_training:
            tf.summary.scalar("landmark_loss", tf.reduce_mean(landmark_losses))

          # compute optimization op (potentially with gradient clipping)
          optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)

          gradients = optimizer.compute_gradients(mean_huber_loss, var_list=policy.trainable_vars)
          if self.grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
              if grad is not None:
                gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)

          training_step = optimizer.apply_gradients(gradients)

        with tf.name_scope('update_target_network_ops'):
          init_target_network = []
          update_target_network = []
          for var, var_target in zip(
              sorted(policy.trainable_vars, key=lambda v: v.name),
              sorted(target_policy.trainable_vars, key=lambda v: v.name)):
            new_target = self.target_network_update_frac       * var +\
                         (1 - self.target_network_update_frac) * var_target
            update_target_network.append(var_target.assign(new_target))
            init_target_network.append(var_target.assign(var))
          update_target_network = tf.group(*update_target_network)
          init_target_network = tf.group(*init_target_network)

        with tf.variable_scope("input_info", reuse=False):
          tf.summary.scalar('rewards', tf.reduce_mean(r_t))
          tf.summary.histogram('rewards', r_t)
          if len(policy.obs_ph.shape) == 3:
            tf.summary.image('observation', policy.obs_ph)
          else:
            tf.summary.histogram('observation', policy.obs_ph)

        with tf_util.COMMENT("attribute assignments:"):
          self._act = act
          self._train_step = training_step
          self._obs1_ph = policy.obs_ph
          self._action_ph = a_tm1
          self._reward_ph = r_t
          self._obs2_ph = target_policy.obs_ph
          self._dones_ph = done_mask_ph
          self._goal_ph = policy.goal_ph
          self._goal2_ph = target_policy.goal_ph
          self._landmark_state_ph = landmark_state_ph
          self._landmark_goal_ph = landmark_goal_ph
          self._landmark_scores = landmark_scores
          self._landmark_ratios = landmark_ratios
          self.update_target_network = update_target_network
          self.init_target_network = init_target_network
          self.model = policy
          self.target_model = target_policy
          self.epsilon_ph = epsilon_ph
          self.reset_ph = reset_ph
          self.threshold_ph = threshold_ph

          with tf.variable_scope("deepq"):
            self.params = tf.trainable_variables()

        # Log the state, action, goal, and landmark state
        # with tf.variable_scope("input_info_viz", reuse=False):
        tf.summary.image('input state', tf.cast(self._obs1_ph, tf.float32), max_outputs=1)
        tf.summary.image('input goal', tf.cast(self._goal_ph, tf.float32), max_outputs=1)
        action_shape = tf.reshape(self._action_ph, [-1, self.action_space.n, 1, 1])
        tf.summary.image('input action', tf.cast(action_shape, tf.float32), max_outputs=1)
        if self.landmark_training:
          tf.summary.image('generated landmark', tf.cast(self._landmark_goal_ph, tf.float32), max_outputs=1)

        with tf_util.COMMENT("attribute assignments:"):
          self._summary_op = tf.summary.merge_all()
        
        # Initialize the parameters and copy them to the target network.
        with tf_util.COMMENT("graph initialization"):
          tf_util.initialize(self.sess)
          self.sess.run(self.init_target_network)

          self.summary = tf.summary.merge_all()

  def _setup_new_task(self, total_timesteps):
    """Sets up new task by reinitializing step, replay buffer, and exploration schedule"""
    self.task_step = 0
    self.reset = np.ones([self.n_envs, 1])

    items = [("observations0", self.observation_space.shape), ("actions", self.action_space.shape), ("rewards", (1, )),
             ("observations1", self.observation_space.shape), ("terminals1", (1, ))]
    if self.goal_space is not None:
      items += [("desired_goal", self.env.observation_space.spaces['desired_goal'].shape)]

    self.replay_buffer = ReplayBuffer(self.buffer_size, items)

    if self.hindsight_mode == 'final':
      self.hindsight_fn = lambda trajectory: her_final(trajectory, self.env.compute_reward)
    elif isinstance(self.hindsight_mode, str) and 'future_' in self.hindsight_mode:
      _, k = self.hindsight_mode.split('_')
      if self.landmark_generator is not None:
        # Assume that landmark generator needs to have a separate landmark buffer containing [s,a,l,g] tuples
        self.hindsight_fn = lambda trajectory: her_future_landmark(trajectory, int(k), self.env.compute_reward)
      else:
        self.hindsight_fn = lambda trajectory: her_future(trajectory, int(k), self.env.compute_reward)
      self.hindsight_frac = 1. - 1. / (1. + float(k))
    elif isinstance(self.hindsight_mode,
                    str) and 'futureactual_' in self.hindsight_mode:
      _, k, p = self.hindsight_mode.split('_')
      if self.landmark_generator is not None:
        self.hindsight_fn = HerFutureAchievedPastActualLandmark(int(k), int(p), self.env.compute_reward)
      else:
        self.hindsight_fn = HerFutureAchievedPastActual(int(k), int(p), self.env.compute_reward)
      self.hindsight_frac = 1. - 1. / (1. + float(k + p))
    else:
      self.hindsight_fn = None

    # Add additional fields for the hindsight replay buffer, if using landmark
    # When using landmark, current observation becomes the landmark when goal_space
    # is the same as the observation space, for now
    hindsight_items = copy.deepcopy(items)

    # Create a secondary replay buffer
    if self.hindsight_fn is not None:
        self.replay_buffer_hindsight = ReplayBuffer(self.buffer_size, hindsight_items)
    
    self.hindsight_subbuffer = EpisodicBuffer(self.n_envs, self.hindsight_fn, n_cpus=min(self.n_envs, 8))

    if self.goal_space is not None and self.landmark_training:
      if isinstance(self.landmark_generator, FunctionType):
          self.landmark_generator = self.landmark_generator(self.buffer_size, self.env)
      elif self.landmark_generator is None:
          self.state_buffer = ReplayBuffer(self.buffer_size, [
            ("state", self.env.observation_space.spaces['observation'].shape),
            ("as_goal", self.env.observation_space.spaces['achieved_goal'].shape)])
      else:
          assert isinstance(self.landmark_generator, AbstractLandmarkGenerator)

    # Create the schedule for exploration starting from 1.
    self.exploration = LinearSchedule(
        schedule_timesteps=int(self.exploration_fraction * total_timesteps),
        initial_p=1.0,
        final_p=self.exploration_final_eps)

  def _get_action_for_single_obs(self, obs):
    """Called during training loop to get online action (with exploration)"""
    if self.goal_space is not None:
      feed_dict = {
          self.model.obs_ph: np.array(obs["observation"]),
          self.model.goal_ph: np.array(obs["desired_goal"]),
          self.epsilon_ph: self.exploration.value(self.task_step),
          self.reset_ph: self.reset
      }
    else:
      feed_dict = {
          self.model.obs_ph: np.array(obs),
          self.epsilon_ph: self.exploration.value(self.task_step),
          self.reset_ph: self.reset
      }

    return self.sess.run(self._act, feed_dict=feed_dict)

  def _process_experience(self, obs, action, rew, new_obs, done):
    """Called during training loop after action is taken; includes learning;
        returns a summary"""
    summaries = []
    expanded_done = np.expand_dims(done.astype(np.float32), 1)
    rew = np.expand_dims(rew, 1)

    goal_agent = self.goal_space is not None

    # Reset the episode if done
    self.reset = expanded_done

    # Store transition in the replay buffer, and hindsight subbuffer
    if goal_agent:
      self.replay_buffer.add_batch(obs['observation'], action, rew, new_obs['observation'], expanded_done, new_obs['desired_goal'])

      if self.landmark_training:
        if self.landmark_generator is not None:
          self.landmark_generator.add_state_data(obs['observation'], obs['achieved_goal'])
        else:
          self.state_buffer.add_batch(obs['observation'], obs['achieved_goal'])
    else:
      self.replay_buffer.add_batch(obs, action, rew, new_obs, expanded_done)

    if self.hindsight_fn is not None:
      for idx in range(self.n_envs):
        # add the transition to the HER subbuffer for that worker
        self.hindsight_subbuffer.add_to_subbuffer(
            idx, [obs['observation'][idx], action[idx], rew[idx], new_obs['observation'][idx], new_obs['achieved_goal'][idx], new_obs['desired_goal'][idx]])
        if done[idx]:
          # commit the subbuffer
          self.hindsight_subbuffer.commit_subbuffer(idx)
          if len(self.hindsight_subbuffer) == self.n_envs:
            if self.landmark_generator is not None:
              # Using landmarks will return also the landmark transitions
              hindsight_experiences, landmark_experiences = zip(*self.hindsight_subbuffer.process_trajectories())
            else:
              hindsight_experiences = self.hindsight_subbuffer.process_trajectories()

            # for hindsight_experience in chain.from_iterable(self.hindsight_subbuffer.process_trajectories()):
            #   self.replay_buffer_hindsight.add(*hindsight_experience)

            for hindsight_experience in chain.from_iterable(hindsight_experiences):
              self.replay_buffer_hindsight.add(*hindsight_experience)

            if self.landmark_generator is not None:
              s, a, l, g = [np.array(a) for a in zip(*chain.from_iterable(landmark_experiences))]

              additional = None
              if hasattr(self.landmark_generator,
                         'get_scores_with_experiences') and self.landmark_generator.get_scores_with_experiences:
                  feed_dict = {
                      self._obs1_ph: s,
                      self._action_ph: a,
                      self._goal_ph: g,
                      self._landmark_state_ph: l,
                      self._landmark_goal_ph: self.landmark_generator.goal_extraction_function(l),
                  }

                  landmark_scores, landmark_ratios = self.sess.run(
                      [self._landmark_scores, self._landmark_ratios],
                      feed_dict=feed_dict)

                  additional = np.concatenate([landmark_scores, landmark_ratios], 1)

              landmark_summaries = self.landmark_generator.add_landmark_experience_data(s, a, l, g, additional)
              if landmark_summaries:
                summaries.append(landmark_summaries)
            self.hindsight_subbuffer.clear_main_buffer()

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
                obses_t_hs, actions_hs, rewards_hs, obses_tp1_hs, dones_hs, desired_g_hs = \
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

            # Sample from landmark state buffer, if applicable
            if self.landmark_training:
              if self.landmark_generator is not None:
                landmark_states, landmark_goals = self.landmark_generator.generate(obses_t, actions, desired_g)
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
            feed_dict[self._goal2_ph] = desired_g  # Assuming that the goal does not change in episode

            if self.landmark_training:
              feed_dict[self._landmark_state_ph] = landmark_states
              feed_dict[self._landmark_goal_ph] = landmark_goals

          if self.landmark_generator is not None:
            _, landmark_scores, landmark_ratios, summary = self.sess.run(
              [self._train_step, self._landmark_scores, self._landmark_ratios, self._summary_op],
              feed_dict=feed_dict)
            landmark_ratios = np.squeeze(landmark_ratios, 1)
            self.landmark_generator.assign_scores(landmark_scores, landmark_ratios)

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
      actions = self.sess.run(self.target_model.deterministic_action, {
          self._obs2_ph: observation,
          self._goal2_ph: desired_goal
      })
    else:
      actions = self.sess.run(self.target_model.deterministic_action, {self._obs2_ph: observation})

    if not vectorized_env:
      actions = actions[0]

    return actions, None

  def get_value(self, observation, state=None, goal=None):
    """
    Gets the Q values for each action given the current observation.

    :param observation:
    :param state:
    :param goal:
    :return:
    """
    goal_agent = self.goal_space is not None

    if not goal_agent:
      observation = np.array(observation)
    else:
      desired_goal = np.array(observation['desired_goal'])
      desired_goal = desired_goal.reshape((-1,) + self.goal_space.shape)
      observation = np.array(observation['observation'])

    vectorized_env = self._is_vectorized_observation(observation, self.observation_space)
    observation = observation.reshape((-1,) + self.observation_space.shape)

    if goal_agent:
      values = self.sess.run(self.target_model.q_values, {
        self._obs2_ph: observation,
        self._goal2_ph: desired_goal,
      })
    else:
      values = self.sess.run(self.target_model.q_values, {self._obs2_ph: observation})

    if not vectorized_env:
      values = values[0]

    return values, None

  def save(self, save_path):
    # Things set in the __init__ method should be saved here, because the model is called with default args on load(),
    # which are subsequently updated using this dict.
    data = {
        "learning_rate": self.learning_rate,
        "gamma": self.gamma,
        "exploration_final_eps": self.exploration_final_eps,
        "exploration_fraction": self.exploration_fraction,
        "param_noise": self.param_noise,
        "learning_starts": self.learning_starts,
        "train_freq": self.train_freq,
        "batch_size": self.batch_size,
        "buffer_size": self.buffer_size,
        "target_network_update_frac": self.target_network_update_frac,
        "target_network_update_freq": self.target_network_update_freq,
        'hindsight_mode': self.hindsight_mode,
        'landmark_training': self.landmark_training,
        'landmark_mode': self.landmark_mode,
        'landmark_training_per_batch': self.landmark_training_per_batch,
        'landmark_width': self.landmark_width,
        'landmark_error': self.landmark_error,
        "double_q": self.double_q,
        "grad_norm_clipping": self.grad_norm_clipping,
        "tensorboard_log": self.tensorboard_log,
        "verbose": self.verbose,
        "observation_space": self.observation_space,
        "action_space": self.action_space,
        "policy": self.policy,
        "n_envs": self.n_envs,
        "_vectorize_action": self._vectorize_action
    }

    if self.landmark_generator is not None:
      generator_save_path = save_path + ".gen"
      self.landmark_generator.save(generator_save_path)

    # Model paramaters to be restored
    params = self.sess.run(self.params)

    self._save_to_file(save_path, data=data, params=params)


def epsilon_greedy_wrapper(policy, epsilon_placeholder):
  """
    Given policy and epsilon_placeholder returns a batch_size x 1 Tensor representing e-greedy actions.
    """
  deterministic_actions = tf.argmax(policy.q_values, axis=1)
  batch_size = tf.shape(policy.obs_ph)[0]
  random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=policy.ac_space.n, dtype=tf.int64)
  chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < epsilon_placeholder
  return tf.where(chose_random, random_actions, deterministic_actions)


def param_noise_wrapper(policy, reset_ph, threshold_ph, scale=True):
  """
    Given policy and stated args, returns a batch_size x 1 Tensor representing actions after parameter noise.
    """
  raise NotImplementedError('param_noise not yet implemented')
