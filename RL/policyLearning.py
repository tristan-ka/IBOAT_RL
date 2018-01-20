import random
import sys
from collections import deque

import numpy as np
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Flatten, merge
from keras.models import Model
from keras.optimizers import Adam

sys.path.append("../sim/")
from Simulator import TORAD


class PolicyLearner:
    '''
    The aim of this class is to learn the Q-value of the action defined by a policy.

    :ivar int state_size: shape of the input (for convolutionnal layers).
    :ivar int action_size:  number of action output by the network.
    :ivar deque memory: last-in first-out list of the batch.
    :ivar float gamma:  discount factor.
    :ivar float epsilon: exploration rate.
    :ivar float epsilon_min: smallest exploration rate that we want to converge to.
    :ivar float epsilon_decay: decay factor that we apply after each replay.
    :ivar float learning_rate: the learning rate of the NN.
    :ivar keras.model model: NN, i.e the model containing the weight of the value estimator.
    '''

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1)
        self.gamma = 0.975  # discount rate
        self.epsilon = 0.2  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 1
        self.learning_rate = 0.0001
        self.model = self._build_model()

        self.policy_angle = 16*TORAD
        self.attach_angle = 6*TORAD
        self.stall = 0

    '''
    Definition of the neural network    
    '''

    def _build_model(self):
        """
        Build the different layers of the neural network.

        :return: The model of the neural network.
        """
        inp1 = Input(shape=(self.state_size, 1))
        conv1 = Convolution1D(40, 50, padding='same', dilation_rate=2, activation='relu', )(inp1)
        #conv11 = Convolution1D(30, 20, padding='same', dilation_rate=2, activation='relu', )(conv1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        # drop_1 = Dropout(0.25)(pool_1)
        dense1 = Dense(80, activation='relu')(pool1)
        out1 = Dense(40, activation='relu')(dense1)

        inp2 = Input(shape=(self.state_size, 1))
        conv2 = Convolution1D(40, 50, padding='same', dilation_rate=2, activation='relu')(inp2)
        #conv21 = Convolution1D(20, 20, padding='same', dilation_rate=2, activation='relu')(conv2)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        # drop_2 = Dropout(0.25)(pool_2)
        dense2 = Dense(80, activation='relu')(pool2)
        out2 = Dense(40, activation='relu')(dense2)

        merged = merge([out1, out2], mode='concat', concat_axis=1)
        merged = Flatten()(merged)
        #dense_m1 = Dense(80, activation='relu')(merged)
        dense_m2 = Dense(40, activation='relu')(merged)
        dense = Dense(20, activation='relu')(dense_m2)

        out = Dense(2, activation='linear')(dense)

        model = Model([inp1, inp2], out)
        model.compile(loss='mse',  # using the cross-entropy loss function
                      optimizer=Adam(lr=self.learning_rate),  # using the Adam optimiser
                      metrics=['accuracy'])  # reporting the accuracy

        return model

    '''
    add s,a,r's' to mini-batch
    '''

    def remember(self, state, action, reward, next_state, stall):
        """
        Remember a transition defined by an action `action` taken from a state `state` yielding a transition to a next
        state `next_state` and a reward `reward`. [s, a ,r, s']

        :param np.array state: initial state (s).
        :param int action: action (a).
        :param float reward: reward received from transition (r).
        :param np.array next_state: final state (s').
        :param int stall: flow state in the final state (s').
        """
        self.memory.append((state, action, reward, next_state, stall))

    '''
    Act with respect to policy that we want to learn
    '''

    def init_stall(self, mean, mdp):
        if mdp.simulator.hdg[0] + mean + mdp.simulator.sail_pos > self.policy_angle:
            self.stall=1
        else:
            self.stall=0
    
    def get_stall(self):
        return self.stall

    def actUnderPolicy(self, state):
        """
        Does the same as :meth:`actDeterministicallyUnderPolicy` instead that the returned action
        is sometime taken randomly.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            if self.stall == 1:
                action = 1
                if state[0, self.state_size - 1] < (self.attach_angle + 1 * TORAD):
                    self.stall = 0
            elif self.stall == 0:
                action = 0
                if (state[0, self.state_size - 1] > (self.policy_angle - 1 * TORAD)):
                    self.stall = 1
        return action

    def actDeterministicallyUnderPolicy(self, state):
        """
        Policy that reattaches when the angle of attack goes higher than 16 degree

        :param np.array state: state for which we want to know the policy action.
        :return: the policy action.
        """
        if self.stall == 1:
            action = 1
            if state[0, self.state_size - 1] < (self.attach_angle + 1 * TORAD):
                self.stall = 0
        elif self.stall == 0:
            action = 0
            if (state[0, self.state_size - 1] > (self.policy_angle - 1 * TORAD)):
                self.stall = 1

        return action

    def actRandomly(self):
        return random.randint(0, 1)

    def evaluateNextAction(self, stall):
        '''
        Evaluate the next action without updating the stall state in order to use it during the experience replay
        :param np.array state: state for which we want to know the policy action.
        :return: the policy action.
        '''
        if stall == 1:
            action = 1
        elif stall == 0:
            action = 0

        return action

    def evaluate(self, state):
        """
        Evaluate the Q-value of the two actions in a given state using the neural network.

        :param np.array state: state that we want to evaluate.
        :return: The actions values as a vector.
        """
        sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
        sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
        act_values = self.model.predict([sub_state1, sub_state2])
        return act_values[0]

    def act(self, state):
        """
        Calculate the action that yields the maximum Q-value.

        :param state: state in which we want to chose an action.
        :return: the greedy action.
        """
        sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
        sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
        act_values = self.model.predict([sub_state1, sub_state2])
        # print((act_values[0]))
        print((act_values[0][0] - act_values[0][1]))
        return np.argmax(act_values[0])  # returns action

    '''
    Core of the algortihm --> Q update according to the current weight of the network
    '''

    def replay(self, batch_size):
        """
        Perform the learning on a the experience replay memory.

        :param batch_size: number of samples used in the experience replay memory for the fit.
        :return: the average loss over the replay batch.
        """
        minibatch = random.sample(self.memory, batch_size)
        loss_list = []
        for state, action, reward, next_state, stall in minibatch:  # We iterate over the minibatch
            next_action = self.evaluateNextAction(stall)  # We identify wich action to take in the resulting state (s')
            target = reward + self.gamma * \
                              self.model.predict(
                                  [np.reshape(next_state[0, :], [1, self.state_size, 1]),
                                   np.reshape(next_state[1, :], [1, self.state_size, 1])])[
                                  0][next_action]  # delta=r+Q_pi(s',a')

            sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
            sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
            target_f = self.model.predict([sub_state1, sub_state2])
            target_f[0][action] = target  # changes the action value of the action taken
            sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
            sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
            scores = self.model.fit(x=[sub_state1, sub_state2], y=target_f, epochs=1,
                                    verbose=0,
                                    batch_size=32)  # fit for the previous action value --> update the weights in the network
            loss_list.append(scores.history['loss'])
            losses = np.array([loss_list])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # exploration decay
        return np.sum(losses) / len(losses)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
