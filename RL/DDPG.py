import random
from collections import deque

import tensorflow as tf
import numpy as np

from DDPGNetworks import Network


class DDPGAgent:
    '''
    The aim of this class is to learn an optimal policy via an actor-critic structure with 2 separated Convolutional Neural Networks
<<<<<<< HEAD
    It uses the Deep Deterministic Policy Gradient to update tha actor network.
    This model deals with a continuous space of actions on the rudder, chosen between lower_bound and upper_bound.

=======
    It uses the Deep Deterministic Policy Gradient to update the actor network.
    This model deals with a continuous space of actions on the rudder, chosen between lower_bound and upper_bound
>>>>>>> 720fd95e20ec768898d45dccd1e514934a38f83f
    :param int state_size: length of the state input (for convolutionnal layers).
    :param int action_size:  number of continuous action output by the network.
    :param float lower_bound: minimum value for rudder action.
    :param float upper_bound: maximum value for rudder action.
    :param tensorflow.session sess: initialized tensorflow session within which the agent will be trained.

    :ivar deque memory: last-in first-out list of the batch buffer.
    :ivar float gamma:  discount factor.
    :ivar float epsilon: exploration rate.
    :ivar float epsilon_min: smallest exploration rate that we want to converge to.
    :ivar float epsilon_decay: decay factor that we apply after each replay.
    :ivar float actor_learning_rate: the learning rate of the NN of actor.
    :ivar float critic_learning_rate: the learning rate of the NN of critic.
    :ivar float update_target: update factor of the Neural Networks for each fit to target
    :ivar DDPGNetworks.Network network: tensorflow model which defines actor and critic convolutional neural networks features
    '''

    def __init__(self, state_size, action_size, lower_bound, upper_bound, sess):
        self.state_size = state_size
        self.action_size = action_size
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.sess = sess
        self.memory = deque(maxlen=100000)
        self.gamma = 0.97  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.10
        self.epsilon_decay = 0.50
        self.epsilon_decay_period = 700
        self.actor_learning_rate = 0.0005
        self.critic_learning_rate = 0.0005
        self.update_target = 0.001

        '''
        Definition of the neural networks   
        '''

        self.network = Network(self.state_size, self.action_size,
                               self.lower_bound, self.upper_bound, self.actor_learning_rate,
                               self.critic_learning_rate, self.gamma, self.update_target)
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(self.network.target_init)


    '''
    add s,a,r's' to mini-batch
    '''

    def remember(self, state, action, reward, next_state):
        """
        Remember a transition defined by an action `action` taken from a state `state` yielding a transition to a next
        state `next_state` and a reward `reward`. [s, a ,r, s']

        :param np.array state: initial state (s).
        :param int action: action (a).
        :param float reward: reward received from transition (r).
        :param np.array next_state: final state (s').
        """
        self.memory.append((state, action, reward, next_state))

    def act(self,state):
        """
        Calculate the action given by the Actor network's current weights

        :param state: state in which we want to chose an action.
        :return: the greedy action according to actor network
        """
        s = np.reshape([state[0,:], state[1,:]], [1,2 * self.state_size, 1])
        a, = self.sess.run(self.network.behaviour,
                           feed_dict={self.network.state_ph: s})
        return a

    def noise_decay(self,e):
        """
        Applies decay on noisy epsilon-greedy actions

        :param e: current episode playing during learning
        """
        if e % self.epsilon_decay_period ==0 and self.epsilon > self.epsilon_min and e != 0:
            self.epsilon *= self.epsilon_decay  # exploration decay
            print("EPSILON AFTER DECAY: {}".format(self.epsilon))

    def act_epsilon_greedy(self,state):
        """
        With probability epsilon, returns a random action between bounds
        With probability 1 - epsilon, returns the action given by the Actor network's current weights

        :param state: state in which we want to chose an action.
        :return: a random action or the action given by actor
        """
        alea = np.random.rand()
        if alea <= self.epsilon:
            print("NOISY ACTION")
            a = np.random.uniform(self.lower_bound, self.upper_bound)
        else:
            print("CHOSEN ACTION")
            s = np.reshape([state[0, :], state[1, :]], [1, 2 * self.state_size, 1])
            a, = self.sess.run(self.network.behaviour,
                               feed_dict={self.network.state_ph: s})
        return a

    def evaluate(self, state, action):
        """
        Evaluate the Q-value of a state-action pair  using the critic neural network.

        :param np.array state: state that we want to evaluate.
        :param float action: action that we want to evaluate (has to be between permitted bounds)
        :return: The continuous action value.
        """
        s = np.reshape([state[0, :], state[1, :]], (1,2*self.state_size, 1))
        a = np.reshape(action, (1,self.action_size, 1))
        q = self.sess.run(
            self.network.prediction,
            feed_dict={
                self.network.state_ph: s,
                self.network.action_ph: a})
        return q

    '''
    Core of the algortihm --> Actor and critic networks update on minibatch according to the current weight of the networks
    '''

    def replay(self, batch_size):
        """
        Performs an update of both actor and critic networks on a minibatch chosen among the experience replay memory.

        :param batch_size: number of samples used in the experience replay memory for the fit.
        :return: the average losses for actor and critic over the replay batch.
        """
        minibatch = random.sample(self.memory, batch_size)
        S = []
        S_ = []
        A = []
        R = []
        END = np.ones(batch_size)
        for state, action, reward, next_state in minibatch:  # We iterate over the minibatch
            S.append(np.reshape([state[0,:], state[1,:]], (2*self.state_size,1)))
            A.append(np.reshape(action,(self.action_size,1) ))
            R.append(reward)
            S_.append(np.reshape(next_state, (2*self.state_size,1)))

        q, _, _, critic_loss, actor_loss = self.sess.run(
            [self.network.q_values_of_given_actions, self.network.critic_train_op, self.network.actor_train_op,
             self.network.critic_loss, self.network.actor_loss],
            feed_dict={
                self.network.state_ph: np.asarray(S),
                self.network.action_ph: np.asarray(A),
                self.network.reward_ph: np.asarray(R),
                self.network.next_state_ph: np.asarray(S_),
                self.network.is_not_done_ph: np.asarray(END)})
        #actor_losses = np.array(actor_scores.history['loss'])
        #critic_losses = np.array(critic_scores.history['loss'])
        #actor = np.sum(actor_losses) / len(actor_losses)
        #critic = np.sum(critic_losses) / len(critic_losses)
        return actor_loss, critic_loss

    def load(self, name):
        """
        Load the weights of the 2 networks saved in the file into :ivar network
        :param name: name of the file containing the weights to load
        """
        saver = tf.train.Saver()
        saver.restore(self.sess, name+".ckpt")


    def save(self, name):
        """
        Save the weights of both of the networks into a .ckpt tensorflow session file
        :param name: Name of the file where the weights are saved
        """
        saver = tf.train.Saver()
        save_path = saver.save(self.sess, name+".ckpt")
        print("Model saved in path: %s" % save_path)