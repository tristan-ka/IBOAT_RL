import random
from collections import deque

import numpy as np
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Flatten, merge
from keras.models import Model
from keras.optimizers import Adam

from sim.Simulator import TORAD

EPISODES = 1000


class ValueLearner:
    '''
    Constructor:
    state_size is a shape of the input (for convolutionnal layers):
    action_size is the number of action output by the network
    memory is last-in first-out list of the batch
    gamma is the discount factor
    learning_rate is the learning rate of the NN
    model is the NN, i.e the model containing the weight of the value estimator.
    '''

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        self.gamma = 0.95  # discount rate
        self.learning_rate = 0.0001
        self.model = self._build_model()

    '''
    Definition of the neural network
    '''

    def _build_model(self):

        inp_1 = Input(shape=(self.state_size, 1))
        conv_1 = Convolution1D(50, 5, padding='same', dilation_rate=2, activation='relu')(inp_1)
        pool_1 = MaxPooling1D(pool_size=2)(conv_1)
        # drop_1 = Dropout(0.25)(pool_1)
        dense_1 = Dense(40, activation='relu')(pool_1)
        #dense_11 = Dense(40, activation='relu')(dense_1)
        out_1 = Dense(20, activation='relu')(dense_1)

        inp_2 = Input(shape=(self.state_size, 1))
        conv_2 = Convolution1D(50, 5, padding='same', dilation_rate=2, activation='relu')(inp_2)
        pool_2 = MaxPooling1D(pool_size=2)(conv_2)
        # drop_2 = Dropout(0.25)(pool_2)
        dense_2 = Dense(40, activation='relu')(pool_2)
        #dense_22 = Dense(40, activation='relu')(dense_2)
        out_2 = Dense(20, activation='relu')(dense_2)

        merged = merge([out_1, out_2], mode='concat', concat_axis=1)
        merged = Flatten()(merged)
        out = Dense(1, activation='linear')(merged)

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

    '''
    Act with respect to policy that we want to learn
    '''

    def actUnderPolicy(self, state):

        if (state[0, self.state_size - 1] < 14 * TORAD):
            action = 0
        else:
            action = 1
        return action


    def evaluateValue(self,state):
        sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
        sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
        value=self.model.predict([sub_state1, sub_state2])
        return value[0]
    '''
    Core of the algortihm --> Q update according to the current weight of the network
    '''

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        loss_list=[]
        for state, action, reward, next_state in minibatch:  # We iterate over the minibatch
            target = reward + self.gamma * \
                              self.model.predict(
                                  [np.reshape(next_state[0, :], [1, self.state_size, 1]),
                                   np.reshape(next_state[1, :], [1, self.state_size, 1])])[
                                  0]# delta=r+Q_pi(s',a')
            sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
            sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
            scores=self.model.fit(x=[sub_state1, sub_state2], y=target, epochs=1,
                           verbose=0)  # fit for the previous action value --> update the weights in the network
            loss_list.append(scores.history['loss'])
            # loss, accuracy = self.model.evaluate(x=[sub_state1, sub_state2], y=target_f)
            # print("Learning loss : {}, Learning accuracy : {}".format(loss, accuracy))
        losses=np.array([loss_list])
        return np.sum(losses)/len(losses)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
