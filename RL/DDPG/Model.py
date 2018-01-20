
import tensorflow as tf
import settings as settings
import numpy as np

def build_actor(states, bounds, action_size, trainable, scope):
    with tf.variable_scope(scope):

        conv_i = tf.layers.conv1d(states[:,0:30,:], filters = settings.ACTOR_NETWORK_FILTERS,kernel_size = 20, strides = 1, trainable=trainable,
                                 activation=tf.nn.relu, name='conv_i')
        conv_v = tf.layers.conv1d(states[:,30:60,:], filters=settings.ACTOR_NETWORK_FILTERS,kernel_size = 20, strides = 1, trainable=trainable,
                                 activation=tf.nn.relu, name='conv_v')
        # hidden_i = tf.layers.dense(conv_i, 20, trainable=trainable,
                                #   activation=tf.nn.relu, name='dense_i')
        # hidden_v = tf.layers.dense(conv_v, 20, trainable=trainable,
                                #   activation=tf.nn.relu, name='dense_v')
        merge = tf.concat([conv_i,conv_v],axis = 1)
        merge_flat = tf.contrib.layers.flatten(merge,scope=scope)
        hidden_i_v = tf.layers.dense(merge_flat, 40, trainable=trainable,
                                   activation=tf.nn.relu, name='dense_i_v')
        hidden_i_v_2 = tf.layers.dense(hidden_i_v, 8, trainable=trainable,
                                   activation=tf.nn.relu, name='dense_i_v_2')
        actions_unscaled = tf.layers.dense(hidden_i_v_2, action_size,
                                           trainable=trainable, name='dense_i_v_3')
        # bound the actions to the valid range
        low_bound, high_bound = bounds
        valid_range = high_bound - low_bound
        actions = low_bound + tf.nn.sigmoid(actions_unscaled) * valid_range
    return actions


def build_critic(states, actions, trainable, reuse, scope):
    #temp_vec = np.zeros(30)
    #temp_vec[29] = 1.0
    with tf.variable_scope(scope):
        # states_actions = tf.concat([states, actions], axis=1)

        conv_i = tf.layers.conv1d(states[:, 0:30,:], reuse=reuse, filters=settings.ACTOR_NETWORK_FILTERS, kernel_size=20, strides=1, trainable=trainable,
                                  activation=tf.nn.relu, name='conv_i')
        conv_v = tf.layers.conv1d(states[:, 30:60,:], reuse = reuse, filters=settings.ACTOR_NETWORK_FILTERS, kernel_size=20, strides=1, trainable=trainable,
                                  activation=tf.nn.relu, name='conv_v')
        # hidden_i = tf.layers.dense(conv_i, 20, trainable=trainable,reuse=reuse,
                    #               activation=tf.nn.relu, name='dense_i')
        #hidden_v = tf.layers.dense(conv_v, 20, trainable=trainable,reuse=reuse,
                          #         activation=tf.nn.relu, name='dense_v')
        merge = tf.concat([conv_i,conv_v],axis = 1)
        merge_flat = tf.contrib.layers.flatten(merge)

        hidden_i_v = tf.layers.dense(merge_flat, 40, trainable=trainable,reuse=reuse,
                                     activation=tf.nn.relu, name='dense_i_v')
        hidden_a = tf.layers.dense(actions, 1, trainable=trainable,reuse=reuse,
                                     activation=tf.nn.relu, name='dense_a')
        merge_with_action = tf.concat([tf.expand_dims(hidden_i_v,axis=2),hidden_a],axis = 1)
        merge_with_action_flat = tf.contrib.layers.flatten(merge_with_action)

        hidden_i_v_2 = tf.layers.dense(merge_with_action_flat, 10, trainable=trainable,reuse=reuse,
                                       activation=tf.nn.relu, name='dense_i_v_2')
        q_values = tf.layers.dense(hidden_i_v_2, 1,
                                   trainable=trainable, reuse=reuse,
                                   name='dense_3')
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
