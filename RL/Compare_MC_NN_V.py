import random

import matplotlib.pyplot as plt
import numpy as np

from sim.Hysteresis_backup import Hysteresis
from sim.Simulator import TORAD
from sim.mdp import MDP
from valueLearning import ValueLearner

'''
MDP Parameters
'''
history_duration = 5  # Duration of state history [s]
mdp_step = 1  # Step between each state transition [s]
time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
delay = 5  # delay in idx --> = tau=delay*time_step
mdp = MDP(history_duration, mdp_step, time_step, delay)

'''
Environment Parameters
'''
WS = 7  # Wind Speed
mean = 0  # Wind heading
std = 0 * TORAD  # Wind noise
wind_samples = mdp.size  # Number of wind sample
WH = np.random.normal(mean, std, size=wind_samples)
i_rand_vec = ([2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26])
'''
Memeory generation --> Hysteresis

Note that we create one single speed calculator that we use to generate different experiences because it
saves the state of the flow and the previous incidences
Be carefull to generate experiences in a chronological way !!!

'''
speedCalculator = Hysteresis()

state_size = mdp.size
action_size = 2
agent = ValueLearner(state_size, action_size)
agent.load("value_learning_reward")
i_rand = random.sample(i_rand_vec, 1)[0]
i = i_rand * np.ones(mdp.size) * TORAD

state = mdp.initializeMDP(i, WS, speedCalculator)
values = []
rewards = []
monte_carlo_values=[]

i = np.ones(0)
v = np.ones(0)
wind_heading = np.ones(0)
actions = [0, 1]
SIMULATION_TIME=200
for time in range(SIMULATION_TIME):
    WH = np.random.uniform(mean - std, mean + std, size=wind_samples)
    action=random.sample(actions,1)[0]
    print(action)
    # action = agent.actUnderPolicy(state)
    next_state, reward = mdp.transition(action, WS, WH, speedCalculator)
    print("Time: {}, reward: {}".format(time, reward))
    values.append(agent.evaluateValue(state)[0])
    rewards.append(reward)
    state = next_state
    i = np.concatenate([i, mdp.extractSimulationData()[0, :]])
    v = np.concatenate([v, mdp.extractSimulationData()[1, :]])
    wind_heading = np.concatenate([wind_heading, WH[0:10]])

rewards=np.array(rewards)



for ii in range (SIMULATION_TIME):
    t = np.linspace(0, len(rewards)-1-ii, len(rewards)-ii)
    print(np.shape(t))
    gamma_vec=np.power(agent.gamma,t)
    gamma_r = gamma_vec * rewards[ii:len(rewards)]
    monte_carlo_values.append(np.sum(gamma_r))

monte_carlo_values=np.array(monte_carlo_values)
values=np.array(values)


time_vec=np.linspace(1,SIMULATION_TIME,SIMULATION_TIME)
time=time_vec[1:int(SIMULATION_TIME/8)]
monte_carlo_values=monte_carlo_values[1:int(SIMULATION_TIME/2)]
values=values[1:int(SIMULATION_TIME/8)]

time_vec = np.linspace(0, SIMULATION_TIME, int((SIMULATION_TIME) / time_step))
# Plotting results
f, axarr = plt.subplots(3, sharex=True)
axarr[0].scatter(time_vec[1:int(SIMULATION_TIME/8/time_step)], v[1:int(SIMULATION_TIME/8/time_step)])
axarr[1].scatter(time_vec[1:int(SIMULATION_TIME/8/time_step)], i[1:int(SIMULATION_TIME/8/time_step)] / TORAD)

# Plotting comparison of values
axarr[2].plot(time,monte_carlo_values,label='Return from MC')
axarr[2].plot(time,values,label='Value from NN')
axarr[0].set_ylabel("v [m/s]")
axarr[1].set_ylabel("Hdg")
plt.xlabel("time [s]")

plt.xlabel("t [s]")
plt.legend()
# axarr[2].legend("MC","NN")






