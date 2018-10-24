from functools import partial

import tensorflow as tf
import numpy as np
import gym

from stable_baselines import logger
from stable_baselines.common import tf_util, BaseRLModel, SetVerbosity, TensorboardWriter
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.vec_env import VecEnv
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.common.replay_buffer import ReplayBuffer
from stable_baselines.a2c.utils import find_trainable_variables, total_episode_reward_logger

class SimpleDQN(BaseRLModel):
    """
    Simplified version of DQN model class. DQN paper: https://arxiv.org/pdf/1312.5602.pdf

    :param policy: (BasePolicy or str) The policy model to use (MlpPolicy, CnnPolicy, LnMlpPolicy, ...)
    :param env: (Gym environment or str) The environment to learn from (if registered in Gym, can be str)
    :param gamma: (float) discount factor
    :param learning_rate: (float) learning rate for adam optimizer
    :param buffer_size: (int) size of the replay buffer
    :param exploration_fraction: (float) fraction of entire training period over which gamme is annealed
    :param exploration_final_eps: (float) final value of random action probability
    :param train_freq: (int) update the model every `train_freq` steps. set to None to disable printing
    :param batch_size: (int) size of a batched sampled from replay buffer for training
    :param checkpoint_freq: (int) how often to save the model. This is so that the best version is restored at the
            end of the training. If you do not wish to restore the best version set this variable to None.
    :param checkpoint_path: (str) replacement path used if you need to log to somewhere other than tmp directory.
    :param learning_starts: (int) how many steps of the model to collect transitions for before learning starts
    :param target_network_update_freq: (int) update the target network every `target_network_update_freq` steps.
    :param verbose: (int) the verbosity level: 0 none, 1 training information, 2 tensorflow debug
    :param tensorboard_log: (str) the log location for tensorboard (if None, no logging)
    :param _init_setup_model: (bool) Whether or not to build the network at the creation of the instance
    """

    def __init__(self, policy, env, gamma=0.99, learning_rate=5e-4, buffer_size=50000, exploration_fraction=0.1,
                 exploration_final_eps=0.02, train_freq=1, batch_size=32, checkpoint_freq=10000, checkpoint_path=None,
                 learning_starts=1000, target_network_update_freq=500, verbose=0, tensorboard_log=None,
                 _init_setup_model=True):

        super(SimpleDQN, self).__init__(policy=policy, env=env, verbose=verbose, requires_vec_env=False)

        self.checkpoint_path = checkpoint_path
        self.learning_starts = learning_starts
        self.train_freq = train_freq
        self.batch_size = batch_size
        self.target_network_update_freq = target_network_update_freq
        self.checkpoint_freq = checkpoint_freq
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.buffer_size = buffer_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tensorboard_log = tensorboard_log

        self.graph = None
        self.sess = None
        self._train_step = None
        self.step_model = None
        self.update_target_network = None
        self.act = None
        self.proba_step = None
        self.replay_buffer = None
        self.exploration = None
        self.params = None
        self.summary = None
        self.episode_reward = None

        self.double_q = True
        self.grad_norm_clipping = 10.

        if _init_setup_model:
            self.setup_model()

    def setup_model(self):
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
                self.step_model = policy
                self.eps_ph = eps_ph
                
                self.proba_step = self.step_model.proba_step
                with tf.variable_scope("deepq"):
                    self.params = tf.trainable_variables()

                # Initialize the parameters and copy them to the target network.
                tf_util.initialize(self.sess)
                self.sess.run(self.update_target_network)

                self.summary = tf.summary.merge_all()

    def learn(self, total_timesteps, callback=None, seed=None, log_interval=100, tb_log_name="DQN"):
        with SetVerbosity(self.verbose), TensorboardWriter(self.graph, self.tensorboard_log, tb_log_name) as writer:
            self._setup_learn(seed)

            # Create the replay buffer
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

            episode_rewards = [0.0]
            obs = self.env.reset()
            self.episode_reward = np.zeros((1,))

            for step in range(total_timesteps):
                if callback is not None:
                    callback(locals(), globals())
                # Take action and update exploration to the newest value
                eps = self.exploration.value(step)

                action = self.sess.run(self.act, feed_dict = {self.step_model.obs_ph : np.array(obs)[None], 
                    self.eps_ph: eps})[0]
                new_obs, rew, done, _ = self.env.step(action)
                # Store transition in the replay buffer.
                self.replay_buffer.add(obs, action, rew, new_obs, float(done))
                obs = new_obs

                if writer is not None:
                    ep_rew = np.array([rew]).reshape((1, -1))
                    ep_done = np.array([done]).reshape((1, -1))
                    self.episode_reward = total_episode_reward_logger(self.episode_reward, ep_rew, ep_done, writer,
                                                                      step)

                episode_rewards[-1] += rew
                if done:
                    if not isinstance(self.env, VecEnv):
                        obs = self.env.reset()
                    episode_rewards.append(0.0)

                if step > self.learning_starts and step % self.train_freq == 0:
                    # Minimize the error in Bellman's equation on a batch sampled from replay buffer.
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
                    if writer is not None:
                        # run loss backprop with summary, but once every 100 steps save the metadata
                        # (memory, compute time, ...)
                        #if (1 + step) % 100 == 0:
                        #    #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                        #    #run_metadata = tf.RunMetadata()
                        #    _, summary = self._train_step(obses_t, actions, rewards, obses_tp1,
                        #                                          dones, sess=self.sess, options=run_options,
                        #                                          run_metadata=run_metadata)
                        #    writer.add_run_metadata(run_metadata, 'step%d' % step)
                        #else:
                        _, summary = self.sess.run([self._train_step, self._summary_op], feed_dict=feed_dict)
                        writer.add_summary(summary, step)
                    else:
                        _ = self.sess.run(self._train_step, feed_dict=feed_dict)
        
                if step > self.learning_starts and step % self.target_network_update_freq == 0:
                    # Update target network periodically.
                    self.sess.run(self.update_target_network)

                if len(episode_rewards[-101:-1]) == 0:
                    mean_100ep_reward = -np.inf
                else:
                    mean_100ep_reward = round(float(np.mean(episode_rewards[-101:-1])), 1)

                num_episodes = len(episode_rewards)
                if self.verbose >= 1 and done and log_interval is not None and len(episode_rewards) % log_interval == 0:
                    logger.record_tabular("steps", step)
                    logger.record_tabular("episodes", num_episodes)
                    logger.record_tabular("mean 100 episode reward", mean_100ep_reward)
                    logger.record_tabular("% time spent exploring", int(100 * self.exploration.value(step)))
                    logger.dump_tabular()

        return self

    def predict(self, observation, state=None, mask=None, deterministic=True):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        with self.sess.as_default():
            actions, _, _, _ = self.step_model.step(observation, deterministic=deterministic, only_action=True)

        if not vectorized_env:
            actions = actions[0]

        return actions, None

    def action_probability(self, observation, state=None, mask=None):
        observation = np.array(observation)
        vectorized_env = self._is_vectorized_observation(observation, self.observation_space)

        observation = observation.reshape((-1,) + self.observation_space.shape)
        
        actions_proba = self.proba_step(observation, state, mask)

        if not vectorized_env:
            if state is not None:
                raise ValueError("Error: The environment must be vectorized when using recurrent policies.")
            actions_proba = actions_proba[0]

        return actions_proba

    def save(self, save_path):
        # params
        data = {
            "checkpoint_path": self.checkpoint_path,
            "learning_starts": self.learning_starts,
            "train_freq": self.train_freq,
            "batch_size": self.batch_size,
            "target_network_update_freq": self.target_network_update_freq,
            "checkpoint_freq": self.checkpoint_freq,
            "exploration_final_eps": self.exploration_final_eps,
            "exploration_fraction": self.exploration_fraction,
            "learning_rate": self.learning_rate,
            "gamma": self.gamma,
            "verbose": self.verbose,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "policy": self.policy,
            "n_envs": self.n_envs,
            "_vectorize_action": self._vectorize_action
        }

        params = self.sess.run(self.params)

        self._save_to_file(save_path, data=data, params=params)
