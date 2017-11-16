import math
import random
from collections import deque

import numpy as np
from Hysteresis import Hysteresis
from keras.layers import Input, Convolution1D, MaxPooling1D, Dense, Dropout, Flatten, merge
from keras.models import Model
from mdp import MDP

from sim import Simulator

TORAD = math.pi / 180

'''
MDP Parameters
'''
history_duration = 3  # Duration of state history [s]
mdp_step = 1  # Step between each state transition [s]
time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition

mdp = MDP(history_duration, mdp_step, time_step)

'''
Environment Parameters
'''
WS = 7  # Wind Speed
mean = 0  # Wind heading
std = 0.1 * TORAD  # Wind noise
wind_samples = mdp.size  # Number of wind sample
WH = np.random.normal(mean, std, size=wind_samples)

'''
Memeory generation --> Hysteresis

Note that we create one single speed calculator that we use to generate different experiences because it
saves the state of the flow and the previous incidences
Be carefull to generate experiences in a chronological way !!!

'''
speedCalculator = Hysteresis()

state_size = mdp.size
action_size = 2

i_rand = random.uniform(Simulator.IMIN, Simulator.IMAX)
i = i_rand * np.ones(mdp.size)
state = mdp.initializeMDP(i, WS, speedCalculator)
batch_size = 32

memory = deque(maxlen=2000)

SIMULATION_TIME = 10

for time in range(SIMULATION_TIME):
    WH = np.random.uniform(mean - std, mean + std, size=wind_samples)
    action = mdp.policy(15 * TORAD)
    next_state, reward = mdp.transition(action, WS, WH, speedCalculator)
    memory.append((state, action, reward, next_state))
    state = next_state

minibatch = random.sample(memory, 1)

# for state, action, reward, next_state in minibatch:
#     print(state)
#     print(reward)
#     print(next_state)

inp_1 = Input(shape=(state_size, 1))
conv_1 = Convolution1D(2, 3, padding='same', activation='relu')(inp_1)
pool_1 = MaxPooling1D(pool_size=2)(conv_1)
drop_1 = Dropout(0.25)(pool_1)
dense_1 = Dense(30)(drop_1)
out_1 = Dense(2, activation='sigmoid')(dense_1)

inp_2 = Input(shape=(state_size, 1))
conv_2 = Convolution1D(2, 3, padding='same', activation='relu')(inp_2)
pool_2 = MaxPooling1D(pool_size=2)(conv_2)
drop_2 = Dropout(0.25)(pool_2)
dense_2 = Dense(30)(drop_2)
out_2 = Dense(2, activation='sigmoid')(dense_2)

merged = merge([out_1, out_2], mode='concat',concat_axis=1)
merged=Flatten()(merged)

out = Dense(2, activation='sigmoid')(merged)

model = Model([inp_1, inp_2], out)
output=np.array([0,0])
output=np.reshape(output,[1,2])
model.compile(loss='mse',  # using the cross-entropy loss function
              optimizer='adam',  # using the Adam optimiser
              metrics=['accuracy'])  # reporting the accuracy


x_input1=np.reshape(state[0, :],[1,30,1])
x_input2=np.reshape(state[1, :],[1,30,1])
model.fit(x=[x_input1,x_input2], y=output)