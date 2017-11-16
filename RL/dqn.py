import random
from collections import deque

import numpy as np
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Dropout, Flatten, merge
from keras.models import Model
from keras.optimizers import Adam

from sim import Simulator

EPISODES = 1000


class DQNAgent:
    '''
    Constructor:
    state_size is a shape of the input (for convolutionnal layers):
    action_size is the number of action output by the network
    memory is last-in first-out list of the batch
    gamma is the discount factor
    epsilon the exploration rate
    epsilon_min the smallest exploration rate that we want to converge to
    epsilon_decay is the decay factor that we apply after each replay
    learning_rate is the learning rate of the NN
    model is the NN, i.e the model containing the weight of the value estimator.
    '''

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    '''
    Definition of the neural network
    '''

    def _build_model(self):
        inp_1 = Input(shape=(self.state_size, 1))
        conv_1 = Convolution1D(32, 3, dilation_rate=2, padding='same', activation='relu')(inp_1)
        pool_1 = MaxPooling1D(pool_size=2)(conv_1)
        drop_1 = Dropout(0.25)(pool_1)
        dense_1 = Dense(30)(drop_1)
        out_1 = Dense(2, activation='sigmoid')(dense_1)

        inp_2 = Input(shape=(self.state_size, 1))
        conv_2 = Convolution1D(32, 3, dilation_rate=2, padding='same', activation='relu')(inp_2)
        pool_2 = MaxPooling1D(pool_size=2)(conv_2)
        drop_2 = Dropout(0.25)(pool_2)
        dense_2 = Dense(30)(drop_2)
        out_2 = Dense(2, activation='sigmoid')(dense_2)

        merged = merge([out_1, out_2], mode='concat', concat_axis=1)
        merged = Flatten()(merged)
        out = Dense(2, activation='sigmoid')(merged)

        model = Model([inp_1, inp_2], out)
        model.compile(loss='mse',  # using the cross-entropy loss function
                      optimizer=Adam(lr=self.learning_rate),  # using the Adam optimiser
                      metrics=['accuracy'])  # reporting the accuracy
        return model

    '''
    add s,a,r's' to mini-batch
    '''

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def actUnderPolicy(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        if (state[0, self.state_size - 1] < 14 * Simulator.TORAD):
            action = 0
        else:
            action = 1
        return action

    '''
    Act ε-greedy with respect to the actual Q-value output by the network
    '''

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
        sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
        act_values = self.model.predict([sub_state1, sub_state2])
        return np.argmax(act_values[0])  # returns action

    '''
    Core of the algortihm --> Q update according to the current weight of the network
    '''

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state in minibatch:  # We iterate over the minibatch
            target = reward + self.gamma * \
                              np.amax(self.model.predict(
                                  [np.reshape(next_state[0, :], [1, self.state_size, 1]),
                                   np.reshape(next_state[1, :], [1, self.state_size, 1])])[
                                          0])  # delta=r+amax(Q(s',a')
            sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
            sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
            target_f = self.model.predict([sub_state1, sub_state2])
            target_f[0][action] = target
            # target_f[0][action] = target_f[0][action]+ self.learning_rate*(target - target_f[0][action]) # changes the action value of the action taken
            sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
            sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
            self.model.fit(x=[sub_state1, sub_state2], y=target_f, epochs=1,
                           verbose=0)  # fit for the previous action value --> update the weights in the network
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # exploration decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
