
import tensorflow as tf

def build_actor(states, bounds, action_size, trainable, scope):
    """
    Builds actor CNN in the current tensorflow model under given scope name

    :param states: batch of states for learning or prediction
    :param bounds: list of minimum and maximum values of continuous action
    :param action_size: size of continuous action
    :param trainable: boolean that permits to fit neural network in the associated scope if value is True
    :param scope: builds an actor network under a given scope name, to be reused under this name

    :return: actions chosen by the actor network for each state of the batch
    """
    with tf.variable_scope(scope):

        conv_i = tf.layers.conv1d(states[:,0:60,:], filters = 40,kernel_size = 50, strides = 1, padding = "same", trainable=trainable,
                                 activation=tf.nn.relu, name='conv_i')
        conv_v = tf.layers.conv1d(states[:,60:120,:], filters=40,kernel_size = 50, strides = 1, padding = "same", trainable=trainable,
                                 activation=tf.nn.relu, name='conv_v')
        conv_i_2 = tf.layers.conv1d(conv_i, filters=30, kernel_size=20,
                                    trainable=trainable, padding = "same",
                                    activation=tf.nn.relu, name='conv_i_2')
        conv_i_2 = tf.layers.max_pooling1d(conv_i_2,pool_size = 2, strides = 1)

        conv_v_2 = tf.layers.conv1d(conv_v, filters=20, kernel_size=20,
                                    trainable=trainable, padding = "same",
                                    activation=tf.nn.relu, name='conv_v_2')
        conv_v_2 = tf.layers.max_pooling1d(conv_v_2,pool_size = 2, strides = 1)
        hidden_i = tf.layers.dense(conv_i_2, 120, trainable=trainable,
                                   activation=tf.nn.relu, name='dense_i')
        hidden_v = tf.layers.dense(conv_v_2, 120, trainable=trainable,
                                   activation=tf.nn.relu, name='dense_v')
        hidden_i_2 = tf.layers.dense(hidden_i, 60, trainable=trainable,
                                   activation=tf.nn.relu, name='dense_i_2')
        hidden_v_2 = tf.layers.dense(hidden_v, 60, trainable=trainable,
                                   activation=tf.nn.relu, name='dense_v_2')
        merge = tf.concat([hidden_i_2,hidden_v_2],axis = 1)
        merge_flat = tf.contrib.layers.flatten(merge,scope=scope)
        hidden_i_v = tf.layers.dense(merge_flat, 80, trainable=trainable,
                                   activation=tf.nn.relu, name='dense_i_v')
        hidden_i_v_2 = tf.layers.dense(hidden_i_v, 40, trainable=trainable,
                                   activation=tf.nn.relu, name='dense_i_v_2')
        hidden_i_v_3 = tf.layers.dense(hidden_i_v_2, 20, trainable=trainable,
                                       activation=tf.nn.relu, name='dense_i_v_3')
        actions_unscaled = tf.layers.dense(hidden_i_v_3, action_size,
                                           trainable=trainable, name='dense_i_v_out')
        # bound the actions to the valid range
        low_bound, high_bound = bounds
        valid_range = high_bound - low_bound
        actions = low_bound + tf.nn.sigmoid(actions_unscaled) * valid_range
    return actions


def build_critic(states, actions, trainable, reuse, scope):
    """
    Builds actor CNN in the current tensorflow model under given scope name

    :param states: batch of states for learning or prediction of Q-value
    :param action: batch of actions for learning or prediction of Q-value
    :param trainable: boolean that permits to fit neural network in the associated scope if value is True
    :param reuse: boolean that determines if the networks has to be built or reused when build_critic function is called
    :param scope: builds an actor network under a given scope name, to be reused under this name

    :return: q_values for each state-action pair of the given batch
    """
    with tf.variable_scope(scope):
        # states_actions = tf.concat([states, actions], axis=1)

        conv_i = tf.layers.conv1d(states[:, 0:60, :], reuse = reuse, filters=40, kernel_size=50, strides=1, padding="same",
                                  trainable=trainable,
                                  activation=tf.nn.relu, name='conv_i')
        conv_v = tf.layers.conv1d(states[:, 60:120, :], filters=40, kernel_size=50, strides=1, padding="same",
                                  trainable=trainable, reuse = reuse,
                                  activation=tf.nn.relu, name='conv_v')
        conv_i_2 = tf.layers.conv1d(conv_i, filters=30, kernel_size=20,
                                    trainable=trainable, padding="same", reuse = reuse,
                                    activation=tf.nn.relu, name='conv_i_2')
        conv_i_2 = tf.layers.max_pooling1d(conv_i_2, pool_size=2, strides = 1)

        conv_v_2 = tf.layers.conv1d(conv_v, filters=20, kernel_size=20,
                                    trainable=trainable, padding="same", reuse= reuse,
                                    activation=tf.nn.relu, name='conv_v_2')
        conv_v_2 = tf.layers.max_pooling1d(conv_v_2, pool_size=2, strides = 1)
        hidden_i = tf.layers.dense(conv_i_2, 120, trainable=trainable, reuse = reuse,
                                   activation=tf.nn.relu, name='dense_i')
        hidden_v = tf.layers.dense(conv_v_2, 120, trainable=trainable, reuse = reuse,
                                   activation=tf.nn.relu, name='dense_v')
        hidden_i_2 = tf.layers.dense(hidden_i, 60, trainable=trainable, reuse = reuse,
                                     activation=tf.nn.relu, name='dense_i_2')
        hidden_v_2 = tf.layers.dense(hidden_v, 60, trainable=trainable, reuse = reuse,
                                     activation=tf.nn.relu, name='dense_v_2')
        merge = tf.concat([hidden_i_2, hidden_v_2], axis=1)
        merge_flat = tf.contrib.layers.flatten(merge, scope=scope)
        hidden_i_v = tf.layers.dense(merge_flat, 80, trainable=trainable, reuse = reuse,
                                     activation=tf.nn.relu, name='dense_i_v')
        # hidden_a = tf.layers.dense(actions, 1, trainable=trainable,reuse=reuse,
                                  #   activation=tf.nn.relu, name='dense_a')
        merge_with_action = tf.concat([tf.expand_dims(hidden_i_v,axis=2),actions],axis = 1)
        merge_with_action_flat = tf.contrib.layers.flatten(merge_with_action)

        hidden_i_v_2 = tf.layers.dense(merge_with_action_flat, 40, trainable=trainable,reuse=reuse,
                                       activation=tf.nn.relu, name='dense_i_v_a')
        hidden_i_v_3 = tf.layers.dense(hidden_i_v_2, 20, trainable=trainable, reuse=reuse,
                                      activation=tf.nn.relu, name='dense_i_v_a_2')
        q_values = tf.layers.dense(hidden_i_v_3, 1,
                                   trainable=trainable, reuse = reuse, name='dense_i_v_a_out')
    return q_values


def get_vars(scope, trainable):
    if trainable:
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope)
    else:
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)


def copy_vars(src_vars, dest_vars, tau, name):
    update_dest = []
    for src_var, dest_var in zip(src_vars, dest_vars):
        op = dest_var.assign(tau * src_var + (1 - tau) * dest_var)
        update_dest.append(op)
    return tf.group(*update_dest, name=name)


def l2_regularization(vars):
    reg = 0
    for var in vars:
        if not 'bias' in var.name:
            reg += 1e-6 * 0.5 * tf.nn.l2_loss(var)
    return reg
