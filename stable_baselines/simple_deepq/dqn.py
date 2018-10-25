from functools import partial
 
import tensorflow as tf
import numpy as np
import gym
 
from stable_baselines import logger
from stable_baselines.common import tf_util, SimpleRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.replay_buffer import ReplayBuffer
 
class SimpleDQN(SimpleRLModel):
    """
    Simplified version of DQN model class. DQN paper: https://arxiv.org/pdf/1312.5602.pdf
 
    :param policy: (BasePolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
 
    :param exploration_fraction: (float) fraction of entire training period over which gamme is annealed
    :param exploration_final_eps: (float) final value of random action probability
     
    :param buffer_size: (int) size of the replay buffer
    :param train_freq: (int) update the model every `train_freq` steps
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
 
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
 
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """
 
    def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, *, exploration_fraction=0.1,
                 exploration_final_eps=0.02, buffer_size=50000, train_freq=1, batch_size=32, 
                 learning_starts=1000, target_network_update_freq=500, verbose=0, tensorboard_log=None,
                 _init_setup_model=True):
 
        super(SimpleDQN, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False)
 
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.buffer_size = buffer_size
 
        self.target_network_update_freq = target_network_update_freq
         
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
         
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log
 
        self.graph = None
        self.sess = None
        self._train_step = None
        self.model = None
        self.update_target_network = None
        self.act = None
        self.task_step = None
        self.global_step = 0
        self.replay_buffer = None
        self.exploration = None
        self.params = None
        self.summary = None
 
        self.double_q = True
        self.grad_norm_clipping = 10.
 
        if _init_setup_model:
            self._setup_model()
 
    def _setup_model(self):
        with SetVerbosity(self.verbose):        
            assert isinstance(self.action_space, gym.spaces.Discrete), \
                "Error: SimpleDQN only supports gym.spaces.Discrete action space."
            self.graph = tf.Graph()
            with self.graph.as_default():
                self.sess = tf_util.make_session(graph=self.graph)
                n_actions = self.action_space.n
     
                with tf.variable_scope("deepq"):
 
                    # epsilon for e-greedy exploration
                    eps_ph = tf.placeholder_with_default(0., shape=(), name="epsilon_ph")
 
                    # policy function
                    policy = self.policy(self.sess, self.observation_space, self.action_space, n_env=1, n_steps=1, n_batch=None, is_DQN=True)
                    deterministic_actions = tf.argmax(policy.q_values, axis=1)
 
                    batch_size = tf.shape(policy.obs_ph)[0]
                    random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=n_actions, dtype=tf.int64)
                    chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps_ph
                    epsilon_greedy_actions = tf.where(chose_random, random_actions, deterministic_actions)
 
                    act = epsilon_greedy_actions
 
                     
                    # create target q network evaluation
                    with tf.variable_scope("target_q_func", reuse=False):
                        target_policy = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None, reuse=False, is_DQN=True)
 
                    # variables for each
                    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")
                    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                                        scope=tf.get_variable_scope().name + "/target_q_func")
 
                    # setup double q network
                    # because of the outer_scope_getter, this reuses the variables in the outer scope (i.e., the main network vars)
                    with tf.variable_scope("double_q", reuse=True, custom_getter=tf_util.outer_scope_getter("double_q")):
                        double_policy = self.policy(self.sess, self.observation_space, self.action_space, 1, 1, None, reuse=True, 
                                                obs_phs=(target_policy.obs_ph, target_policy.processed_x), is_DQN=True)
                        double_q_values = double_policy.q_values
 
                with tf.variable_scope("loss"):
 
                    # placeholders for bellman equation
                    act_t_ph = tf.placeholder(tf.int32, [None], name="action")
                    rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
                    done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
 
                    # estimated q values for given states/actions
                    estimates = tf.reduce_sum(policy.q_values * tf.one_hot(act_t_ph, n_actions), axis=1)
 
                    # target q values based on 1-step bellman
                    if self.double_q:
                        q_tp1_best_using_online_net = tf.argmax(double_q_values, axis=1)
                        q_tp1_best = tf.reduce_sum(target_policy.q_values * tf.one_hot(q_tp1_best_using_online_net, n_actions), axis=1)
                    else:
                        q_tp1_best = tf.reduce_max(target_policy.q_values, axis=1)
                    q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best
                    targets = rew_t_ph + self.gamma * q_tp1_best_masked
 
                    # td error, using huber_loss
                    td_error = estimates - tf.stop_gradient(targets)
                    mean_huber_loss = tf.reduce_mean(tf_util.huber_loss(td_error))
 
                    tf.summary.scalar("td_error", tf.reduce_mean(td_error))
                    tf.summary.histogram("td_error", td_error)
                    tf.summary.scalar("loss", mean_huber_loss)
 
                    # compute optimization op (potentially with gradient clipping)
                    optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
 
                    gradients = optimizer.compute_gradients(mean_huber_loss, var_list=q_func_vars)
                    if self.grad_norm_clipping is not None:
                        for i, (grad, var) in enumerate(gradients):
                            if grad is not None:
                                gradients[i] = (tf.clip_by_norm(grad, self.grad_norm_clipping), var)
                     
                    training_step = optimizer.apply_gradients(gradients)
 
                with tf.name_scope('update_target_network_ops'):
                    update_target_network = []
                    for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                                sorted(target_q_func_vars, key=lambda v: v.name)):
                        update_target_network.append(var_target.assign(var))
                    update_target_network = tf.group(*update_target_network)
 
                with tf.variable_scope("input_info", reuse=False):
                    tf.summary.scalar('rewards', tf.reduce_mean(rew_t_ph))
                    tf.summary.histogram('rewards', rew_t_ph)
                    if len(policy.obs_ph.shape) == 3:
                        tf.summary.image('observation', policy.obs_ph)
                    else:
                        tf.summary.histogram('observation', policy.obs_ph)
 
                self.act = act
                self._train_step = training_step
                self._summary_op = tf.summary.merge_all()
                self._obs1_ph = policy.obs_ph
                self._action_ph = act_t_ph
                self._reward_ph = rew_t_ph
                self._obs2_ph = target_policy.obs_ph
                self._dones_ph = done_mask_ph
                self.update_target_network = update_target_network
                self.model = policy
                self.eps_ph = eps_ph
                 
                with tf.variable_scope("deepq"):
                    self.params = tf.trainable_variables()
 
                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.sess.run(self.update_target_network)
 
                self.summary = tf.summary.merge_all()
 
    def _setup_new_task(self, total_timesteps):
        """Sets up new task by reinitializing step, replay buffer, and exploration schedule"""
        self.task_step = 0
 
        items = [("observations0", self.observation_space.shape),\
                    ("actions", self.action_space.shape),\
                    ("rewards", (1,)),\
                    ("observations1", self.observation_space.shape),\
                    ("terminals1", (1,))]
 
        self.replay_buffer = ReplayBuffer(self.buffer_size, items)    
 
        # Create the schedule for exploration starting from 1.
        self.exploration = LinearSchedule(schedule_timesteps=int(self.exploration_fraction * total_timesteps),
                                            initial_p=1.0,
                                            final_p=self.exploration_final_eps)
 
    def _get_action_for_single_obs(self, obs):
        """Called during training loop to get online action (with exploration)"""
        feed_dict = {
            self.model.obs_ph : np.array(obs)[np.newaxis], # adds to new axis to convert to batch size of 1
            self.eps_ph: self.exploration.value(self.task_step)
        }
        return self.sess.run(self.act, feed_dict = feed_dict)[0] # indexes into batch to get first (and only) action
 
    def _process_experience(self, obs, action, rew, new_obs, done):
        """Called during training loop after action is taken; includes learning;
        returns a summary"""
         
        summary = None
        self.task_step += 1
        self.global_step += 1
 
        # Store transition in the replay buffer.
        self.replay_buffer.add(obs, action, rew, new_obs, float(done))
 
        # If have enough data, train on it.
        if self.task_step > self.learning_starts:
            if self.task_step % self.train_freq == 0:
                obses_t, actions, rewards, obses_tp1, dones = self.replay_buffer.sample(self.batch_size)
                rewards = np.squeeze(rewards, 1)
                dones = np.squeeze(dones, 1)
                 
                feed_dict = {
                    self._obs1_ph : obses_t,
                    self._action_ph : actions,
                    self._reward_ph : rewards,
                    self._obs2_ph : obses_tp1,
                    self._dones_ph : dones
                }
                 
                _, summary = self.sess.run([self._train_step, self._summary_op], feed_dict=feed_dict)
 
            if self.task_step % self.target_network_update_freq == 0:
                self.sess.run(self.update_target_network)
         
        return summary
 
    def predict(self, observation, state=None, mask=None, deterministic=True):
        return super(SimpleDQN, self).predict(observation, state=state, mask=mask, deterministic=deterministic)
 
    # unclear to SP what exactly to save here. 
    def save(self, save_path):
        # params
        data = {
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "buffer_size": self.buffer_size,
 
            "target_network_update_freq": self.target_network_update_freq,
 
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
 
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "tensorboard_log": self.tensorboard_log,
 
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action
        }
 
        params = self.sess.run(self.params)
 
        self._save_to_file(save_path, data=data, params=params)