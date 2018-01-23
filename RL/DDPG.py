import random
from collections import deque

import numpy as np
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Flatten, merge
from keras.layers.core import Reshape
from keras.models import Model
from keras.optimizers import Adam


class DDPGAgent:
    '''
    The aim of this class is to learn an optimal policy via an actor-critic structure with 2 separated Convolutional Neural Networks
    It uses the Deep Deterministic Policy Gradient to update tha actor network.
    This model deals with a continuous space of actions on the rudder, chosen between lower_bound and upper_bound

    :ivar int state_size: shape of the input (for convolutionnal layers).
    :ivar int action_size:  number of action output by the network.
    :ivar deque memory: last-in first-out list of the batch.
    :ivar float gamma:  discount factor.
    :ivar float epsilon: exploration rate.
    :ivar float epsilon_min: smallest exploration rate that we want to converge to.
    :ivar float epsilon_decay: decay factor that we apply after each replay.
    :ivar float learning_rate: the learning rate of the NN.
    :ivar keras.model actor_model: NN, i.e the model containing the weight of the policy estimator.
    :ivar keras.model critic_model: NN, i.e the model containing the weight of the critic value estimator.
    :ivar float lower_bound: minimum value for rudder action
    :ivar float upper_bound: maximum value for rudder action
    '''

    def __init__(self, state_size, action_size, lower_bound, upper_bound):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=80)
        self.gamma = 0.975  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.95
        self.learning_rate = 0.0001
        self.actor_model = self._build_actor_model()
        self.critic_model = self._build_critic_model()
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    '''
    Definition of the neural network    
    '''

    def _build_actor_model(self):
        """
        Build the different layers of the convolutional neural network.

        :return: The model of the actor neural network.
        """
        inp1 = Input(shape=(self.state_size, 1))
        conv1 = Convolution1D(40, 50, padding='same', dilation_rate=2, activation='relu', )(inp1)
        conv11 = Convolution1D(30, 20, padding='same', dilation_rate=2, activation='relu', )(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv11)
        # drop_1 = Dropout(0.25)(pool_1)
        dense1 = Dense(120, activation='relu')(pool1)
        out1 = Dense(60, activation='relu')(dense1)

        inp2 = Input(shape=(self.state_size, 1))
        conv2 = Convolution1D(40, 50, padding='same', dilation_rate=2, activation='relu')(inp2)
        conv21 = Convolution1D(20, 20, padding='same', dilation_rate=2, activation='relu')(conv2)
        pool2 = MaxPooling1D(pool_size=2)(conv21)
        # drop_2 = Dropout(0.25)(pool_2)
        dense2 = Dense(120, activation='relu')(pool2)
        out2 = Dense(60, activation='relu')(dense2)

        merged = merge([out1, out2], mode='concat', concat_axis=1)
        merged = Flatten()(merged)
        dense_m1 = Dense(80, activation='relu')(merged)
        dense_m2 = Dense(40, activation='relu')(dense_m1)
        dense = Dense(20, activation='relu')(dense_m2)
        out = Dense(1, activation='sigmoid')(dense)

        model = Model([inp1, inp2], out)
        model.compile(loss='mse',  # using the cross-entropy loss function
                      optimizer=Adam(lr=self.learning_rate),  # using the Adam optimiser
                      metrics=['accuracy'])  # reporting the accuracy

        return model

    def _build_critic_model(self):
        """
        Build the different layers of the convolutional neural network.

        :return: The model of the critic neural network.
        """
        inp1 = Input(shape=(self.state_size, 1))
        conv1 = Convolution1D(40, 50, padding='same', dilation_rate=2, activation='relu', )(inp1)
        conv11 = Convolution1D(30, 20, padding='same', dilation_rate=2, activation='relu', )(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv11)
        # drop_1 = Dropout(0.25)(pool_1)
        dense1 = Dense(120, activation='relu')(pool1)
        out1 = Dense(60, activation='relu')(dense1)

        inp2 = Input(shape=(self.state_size, 1))
        conv2 = Convolution1D(40, 50, padding='same', dilation_rate=2, activation='relu')(inp2)
        conv21 = Convolution1D(20, 20, padding='same', dilation_rate=2, activation='relu')(conv2)
        pool2 = MaxPooling1D(pool_size=2)(conv21)
        # drop_2 = Dropout(0.25)(pool_2)
        dense2 = Dense(120, activation='relu')(pool2)
        out2 = Dense(60, activation='relu')(dense2)

        merged = merge([out1, out2], mode='concat', concat_axis=1)
        merged = Flatten()(merged)
        dense_m1 = Dense(80, activation='relu')(merged)

        inp3 = Input(shape=(self.action_size,1))
        dense_m1 = Reshape((80,1))(dense_m1)
        merged_with_a = merge([dense_m1,inp3], mode = 'concat', concat_axis=1)
        merged_with_a = Flatten()(merged_with_a)
        dense_m2 = Dense(40, activation='relu')(merged_with_a)
        dense = Dense(20, activation='relu')(dense_m2)

        out = Dense(1, activation='linear')(dense)

        model = Model([inp1, inp2, inp3], out)
        model.compile(loss='mse',  # using the cross-entropy loss function
                      optimizer=Adam(lr=self.learning_rate),  # using the Adam optimiser
                      metrics=['accuracy'])  # reporting the accuracy

        return model

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
        sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
        sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
        action = self.actor_model.predict([sub_state1, sub_state2])
        return action

    def act_epsilon_greedy(self,state):
        alea = np.random.rand()
        if alea <= self.epsilon:
            print("NOISY ACTION")
            action = np.random.uniform(self.lower_bound, self.upper_bound)
        else:
            print("CHOSEN ACTION")
            sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
            sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
            action_unscaled = self.actor_model.predict([sub_state1, sub_state2])
            action = self.lower_bound + action_unscaled * (self.upper_bound - self.lower_bound)
        return action

    def evaluate(self, state,action):
        """
        Evaluate the Q-value of a state-action pair  using the critic neural network.

        :param np.array state: state that we want to evaluate.
        :return: The actions values as a vector.
        """
        sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
        sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
        act_value = self.critic_model.predict([sub_state1, sub_state2],action)
        return act_value[0]

    '''
    Core of the algortihm --> Actor and critic networks update on minibatch according to the current weight of the networks
    '''

    def replay(self, batch_size):
        """
        Perform the learning on a the experience replay memory.

        :param batch_size: number of samples used in the experience replay memory for the fit.
        :return: the average loss over the replay batch.
        """
        minibatch = random.sample(self.memory, batch_size)
        actor_loss_list = []
        critic_loss_list = []
        X1 = []
        X2 = []
        Y_actor = []
        ACTIONS = []
        Y_critic = []
        for state, action, reward, next_state in minibatch:  # We iterate over the minibatch

            sub_next_state1 = np.reshape(next_state[0, :], [1, self.state_size, 1])
            sub_next_state2 = np.reshape(next_state[1, :], [1, self.state_size, 1])

            next_action = self.actor_model.predict([sub_next_state1,sub_next_state2])
            sub_next_action = np.reshape(next_action,[1,self.action_size,1])

            critic_target = reward + self.gamma * \
                              self.critic_model.predict([sub_next_state1,sub_next_state2,sub_next_action]) # y = r + gamma Q(s',a')
            sub_action = np.reshape(action,[1,self.action_size,1])
            sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
            sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])

            actor_target = self.critic_model.predict([sub_state1, sub_state2, sub_action])

            X1.append(state[0,:])
            X2.append(state[1,:])
            ACTIONS.append(action)
            Y_actor.append(actor_target)
            Y_critic.append(critic_target)

        X1 = np.reshape(X1,[batch_size,self.state_size,1])
        X2 = np.reshape(X2,[batch_size,self.state_size,1])
        Y_actor = np.reshape(Y_actor,(batch_size,self.action_size))
        ACTIONS = np.reshape(ACTIONS,(batch_size,self.action_size,1))
        Y_critic = np.reshape(Y_critic,(batch_size, self.action_size))

        critic_scores = self.critic_model.fit(x=[X1,X2,ACTIONS], y=Y_critic, epochs=1,
                                              verbose=0,
                                              batch_size=batch_size)  # fit for the previous action value --> update the weights in the network

        actor_scores = self.actor_model.fit(x = [X1,X2], y = Y_actor ,epochs = 1, verbose=0, batch_size=batch_size)

        actor_losses = np.array(actor_scores.history['loss'])
        critic_losses = np.array(critic_scores.history['loss'])

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # exploration decay
            print("EPSILON: {}".format(self.epsilon))
        actor = np.sum(actor_losses) / len(actor_losses)
        critic = np.sum(critic_losses) / len(critic_losses)
        return actor, critic

    def load(self, name):
        self.actor_model.load_weights(name)
        self.critic_model.load_weights(name)


    def save(self, name):
        self.actor_model.save_weights(name)
        self.critic_model.save_weights(name)