import random
import sys
sys.path.append("../sim/")
from collections import deque
from Simulator import TORAD


import numpy as np
import keras
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Flatten, merge
from keras.models import Model
from keras.optimizers import Adam

# The n-step learner needs to simulate with MDP agent
from mdp import MDP


class DQNAgent:
    """
    DQN agent

    :ivar np.shape() state_size: shape of the input.
    :ivar int action_size: number of actions.
    :ivar deque() memory: memory as a list.
    :ivar float gamma: Discount rate.
    :ivar float epsilon: exploration rate.
    :ivar float epsilon_min: minimum exploration rate.
    :ivar float epsilon_decay: decay of the exploration rate.
    :ivar_float learning_rate: initial learning rate for the gradient descent
    :ivar keras.model model: neural network model
    """

    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.learning_rate = 0.0001
        self.model = self._build_model()

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

        out = Dense(2, activation='linear')(dense)

        model = Model([inp1, inp2], out)
        model.compile(loss='mse',  # using the cross-entropy loss function
                      optimizer=Adam(lr=self.learning_rate),  # using the Adam optimiser
                      metrics=['accuracy'])  # reporting the accuracy

        return model


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


    def actDeterministically(self, state):
        sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
        sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
        act_values = self.model.predict([sub_state1, sub_state2])
        return np.argmax(act_values[0])  # returns action

    def act(self, state):
        """

        Act ε-greedy with respect to the actual Q-value output by the network.
        :param state: State from which we want to use the network to compute the action to take.
        :return: a random action with probability ε or the greedy action with probability 1-ε.
        """
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
        sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
        act_values = self.model.predict([sub_state1, sub_state2])
        return np.argmax(act_values[0])  # returns action


    def replay(self, batch_size):
        """

        Core of the algorithm --> Q update according to the current weight of the network.
        :param int batch_size: Batch size for the batch gradient descent.
        :return:
        """
        minibatch = random.sample(self.memory, batch_size)
        X1 = []
        X2 = []
        Y = []
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
            X1.append(state[0, :])
            X2.append(state[1, :])
            Y.append(target_f)

        X1 = np.array(X1)
        X1 = np.reshape(X1, [batch_size, self.state_size, 1])
        X2 = np.array(X2)
        X2 = np.reshape(X2, [batch_size, self.state_size, 1])
        Y = np.array(Y)
        Y = np.reshape(Y, [batch_size, self.action_size])
        scores = self.model.fit([X1, X2], Y, epochs=1,
                                verbose=0,
                                batch_size=batch_size)  # fit for the previous action value --> update the weights in the network
        loss = scores.history['loss']

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # exploration decay
        return loss

    def saveModel(self, name):
        self.model.save(name)

    def loadModel(self, name):
        self.model = keras.models.load_model(name)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)

class MultiStepDQNAgent:
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

    def __init__(self, state_size, action_size, n_step, mean, std, duration_history, duration_simulation, delta_t):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        self.gamma = 0.95  # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.00001
        self.model = self._build_model()
        self.duration_history = duration_history
        self.duration_simulation = duration_simulation
        self.delta_t = delta_t
        self.n_step = n_step
        self.mean = mean * TORAD
        self.std = std * TORAD

    '''
    Definition of the neural network    
    '''

    def _build_model(self):


        # On reprend le CNN qui a appris la Policy

        inp1 = Input(shape=(self.state_size, 1))
        # bn1 = BatchNormalization(axis=1)(inp1)
        conv1 = Convolution1D(20, 10, padding='same', dilation_rate=2, activation='relu', )(inp1)
        pool1 = MaxPooling1D(pool_size=2)(conv1)
        # drop_1 = Dropout(0.25)(pool_1)
        dense1 = Dense(60, activation='relu')(pool1)
        out1 = Dense(30, activation='relu')(dense1)

        inp2 = Input(shape=(self.state_size, 1))
        # bn2 = BatchNormalization(axis=1)(inp2)
        conv2 = Convolution1D(20, 10, padding='same', dilation_rate=2, activation='relu')(inp2)
        pool2 = MaxPooling1D(pool_size=2)(conv2)
        # drop_2 = Dropout(0.25)(pool_2)
        dense2 = Dense(60, activation='relu')(pool2)
        out2 = Dense(30, activation='relu')(dense2)

        merged = merge([out1, out2], mode='concat', concat_axis=1)
        merged = Flatten()(merged)
        #merged_norm = BatchNormalization(axis=1)(merged)
        dense = Dense(40, activation='relu')(merged)
        dense2 = Dense(40, activation='relu')(dense)
        out = Dense(2, activation='linear')(dense2)

        model = Model([inp1, inp2], out)
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
    Fonctions relatives à l'apprentissage et l'utilisation de la Q-fonction
    '''

    def evaluate(self, state):
        sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
        sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
        act_values = self.model.predict([sub_state1, sub_state2])
        return act_values[0]

    def actDeterministically(self, state):
        sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
        sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
        act_values = self.model.predict([sub_state1, sub_state2])
        # print((act_values[0]))
        # print((act_values[0][0] - act_values[0][1]))
        return np.argmax(act_values[0])  # returns action

    def act(self, state):
        ra = np.random.rand()
        if ra <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
            sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
            act_values = self.model.predict([sub_state1, sub_state2])
            # print((act_values[0]))
            # print((act_values[0][0] - act_values[0][1]))
            return np.argmax(act_values[0])  # returns action

    '''
    Core of the algortihm --> Q update according to the current weight of the network
    '''

    def replay(self, batch_size):
        """
         Performs an update with n-step return target of the network on a minibatch chosen among the experience replay memory.

         :param batch_size: number of samples used in the experience replay memory for the fit.
         :return: the average loss over the replay batch.
         """
        minibatch = random.sample(self.memory, batch_size)
        loss_list = []
        count = 0
        for state, action, reward, next_state in minibatch:  # We iterate over the minibatch
            print("BATCH numéro:")
            count = count +1
            print(count)
            target = reward
            state_prime = next_state
            # Tirage d'un vent gaussien sur les n steps d'anticipation
            WH = np.random.uniform(self.mean - self.std, self.mean + self.std, size=(self.n_step-1,10))
            # Initialisation d'un MDP pour les n_step transitions: hdg0 = - WH + i + SP
            agent = MDP(self.duration_history, self.duration_simulation,self.delta_t)
            #state_test = agent.initializeMDP((40*TORAD + next_state[0,len(next_state[0]) - 1] - WH[0,9]),WH[0,:])
            agent.initializeState(state_prime)
            # Simulation des n_step dans le futur
            for k in range(1,self.n_step-1):
                next_action = self.act(state_prime) # On obtient le a qui maximise le q précédemment évalué
                print("k = ")
                print(k)
                print("angle et action entreprise correspondante")
                print(state_prime[0,len(state_prime-1)]/TORAD)
                print(next_action)
                state_prime, next_reward = agent.transition(next_action,WH[k,:])
                target = target + self.gamma**k * next_reward

            target = target + self.gamma**(self.n_step-1)* \
                              np.amax(self.model.predict(
                                  [np.reshape(state_prime[0, :], [1, self.state_size, 1]),
                                   np.reshape(state_prime[1, :], [1, self.state_size, 1])])[
                                          0])  # delta=r+amax(Q(s',a')
            sub_state1 = np.reshape(state[0, :], [1, self.state_size, 1])
            sub_state2 = np.reshape(state[1, :], [1, self.state_size, 1])
            target_f = self.model.predict([sub_state1, sub_state2])
            target_f[0][action] = target  # changes the action value of the action taken

            scores = self.model.fit(x=[sub_state1, sub_state2], y=target_f, epochs=1,
                                        verbose=0,
                                        batch_size=1)  # fit for the previous action value --> update the weights in the network
            loss_list.append(scores.history['loss'])
            losses = np.array([loss_list])
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay  # exploration decay
        return np.sum(losses) / len(losses)

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)