import tensorflow as tf

from stable_baselines.common import tf_util

def build_train(q_func, ob_space, ac_space, optimizer, sess, grad_norm_clipping=None, gamma=1.0, double_q=True,
                scope="deepq"):
    """
    Creates the train function:

    :param q_func: (BasePolicy) the policy
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param reuse: (bool) whether or not to reuse the graph variables
    :param optimizer: (tf.train.Optimizer) optimizer to use for the Q-learning objective.
    :param sess: (TensorFlow session) The current TensorFlow session
    :param grad_norm_clipping: (float) clip gradient norms to this value. If None no clipping is performed.
    :param gamma: (float) discount rate.
    :param double_q: (bool) if true will use Double Q Learning (https://arxiv.org/abs/1509.06461). In general it is a
        good idea to keep it enabled.
    :param scope: (str or VariableScope) optional scope for variable_scope.

    :return: (tuple)

        act: (function (TensorFlow Tensor, bool, float): TensorFlow Tensor) function to select and action given
            observation. See the top of the file for details.
        train: (function (Any, numpy float, numpy float, Any, numpy bool, numpy float): numpy float)
            optimize the error in Bellman's equation. See the top of the file for details.
        update_target_network: (function) copy the parameters from optimized Q function to the target Q function.
            See the top of the file for details.
        step_model: (BasePolicy) Policy for evaluation
    """
    n_actions = ac_space.n
    
    with tf.variable_scope(scope):

        # epsilon for e-greedy exploration
        eps_ph = tf.placeholder_with_default(0., shape=(), name="epsilon_ph")

        policy = q_func(sess, ob_space, ac_space, n_env=1, n_steps=1, n_batch=None, is_DQN=True)
        deterministic_actions = tf.argmax(policy.q_values, axis=1)

        batch_size = tf.shape(policy.obs_ph)[0]
        random_actions = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=n_actions, dtype=tf.int64)
        chose_random = tf.random_uniform(tf.stack([batch_size]), minval=0, maxval=1, dtype=tf.float32) < eps_ph
        epsilon_greedy_actions = tf.where(chose_random, random_actions, deterministic_actions)

        act = epsilon_greedy_actions

        # q network evaluation

        q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name + "/model")
        # target q network evaluation

        with tf.variable_scope("target_q_func", reuse=False):
            target_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=False, is_DQN=True)
        target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=tf.get_variable_scope().name + "/target_q_func")

        # compute estimate of best possible value starting from state at t + 1
        double_q_values = None
        double_obs_ph = target_policy.obs_ph
        if double_q:
            with tf.variable_scope("double_q", reuse=True, custom_getter=tf_util.outer_scope_getter("double_q")):
                double_policy = q_func(sess, ob_space, ac_space, 1, 1, None, reuse=True, is_DQN=True)
                double_q_values = double_policy.q_values
                double_obs_ph = double_policy.obs_ph

    with tf.variable_scope("loss"):
        # set up placeholders
        act_t_ph = tf.placeholder(tf.int32, [None], name="action")
        rew_t_ph = tf.placeholder(tf.float32, [None], name="reward")
        done_mask_ph = tf.placeholder(tf.float32, [None], name="done")
        importance_weights_ph = tf.placeholder(tf.float32, [None], name="weight")

        # q scores for actions which we know were selected in the given state.
        q_t_selected = tf.reduce_sum(policy.q_values * tf.one_hot(act_t_ph, n_actions), axis=1)

        # compute estimate of best possible value starting from state at t + 1
        if double_q:
            q_tp1_best_using_online_net = tf.argmax(double_q_values, axis=1)
            q_tp1_best = tf.reduce_sum(target_policy.q_values * tf.one_hot(q_tp1_best_using_online_net, n_actions), axis=1)
        else:
            q_tp1_best = tf.reduce_max(target_policy.q_values, axis=1)
        q_tp1_best_masked = (1.0 - done_mask_ph) * q_tp1_best

        # compute RHS of bellman equation
        q_t_selected_target = rew_t_ph + gamma * q_tp1_best_masked

        # compute the error (potentially clipped)
        td_error = q_t_selected - tf.stop_gradient(q_t_selected_target)
        errors = tf_util.huber_loss(td_error)
        weighted_error = tf.reduce_mean(importance_weights_ph * errors)

        tf.summary.scalar("td_error", tf.reduce_mean(td_error))
        tf.summary.histogram("td_error", td_error)
        tf.summary.scalar("loss", weighted_error)

        # update_target_network op that copies Q network to target Q network
        update_target_network = []
        for var, var_target in zip(sorted(q_func_vars, key=lambda v: v.name),
                                   sorted(target_q_func_vars, key=lambda v: v.name)):
            update_target_network.append(var_target.assign(var))
        update_target_network = tf.group(*update_target_network)

        # compute optimization op (potentially with gradient clipping)
        gradients = optimizer.compute_gradients(weighted_error, var_list=q_func_vars)
        if grad_norm_clipping is not None:
            for i, (grad, var) in enumerate(gradients):
                if grad is not None:
                    gradients[i] = (tf.clip_by_norm(grad, grad_norm_clipping), var)

    with tf.variable_scope("input_info", reuse=False):
        tf.summary.scalar('rewards', tf.reduce_mean(rew_t_ph))
        tf.summary.histogram('rewards', rew_t_ph)
        tf.summary.scalar('importance_weights', tf.reduce_mean(importance_weights_ph))
        tf.summary.histogram('importance_weights', importance_weights_ph)
        if len(policy.obs_ph.shape) == 3:
            tf.summary.image('observation', policy.obs_ph)
        else:
            tf.summary.histogram('observation', policy.obs_ph)

    optimize_expr = optimizer.apply_gradients(gradients)

    summary = tf.summary.merge_all()

    # Create callable functions
    train = tf_util.function(
        inputs=[
            policy.obs_ph,
            act_t_ph,
            rew_t_ph,
            target_policy.obs_ph,
            double_obs_ph,
            done_mask_ph,
            importance_weights_ph
        ],
        outputs=[summary, td_error],
        updates=[optimize_expr]
    )

    return act, train, update_target_network, policy, eps_ph
