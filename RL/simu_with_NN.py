import math

import matplotlib.pyplot as plt
import numpy as np
from sim.Hysteresis_backup import Hysteresis
from sim.mdp import MDP

from policyLearning import PolicyLearner

TORAD = math.pi / 180

'''
MDP Parameters
'''
history_duration = 5  # Duration of state history [s]
mdp_step = 1  # Step between each state transition [s]
time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
delay=5 # delay in idx --> = tau=delay*time_step
mdp = MDP(history_duration, mdp_step, time_step,delay)

'''
Environment Parameters
'''
WS = 7  # Wind Speed
mean = 0  # Wind heading
std = 0 * TORAD  # Wind noise
wind_samples = mdp.size  # Number of wind sample
WH = np.random.normal(mean, std, size=wind_samples)

'''
Memeory generation --> Hysteresis

Note that we create one single speed calculator that we use to generate different experiences because it
saves the state of the flow and the previous incidences
Be carefull to generate experiences in a chronological way !!!

'''
speedCalculator = Hysteresis()

'''
MDP INIT
'''
i = 10 * TORAD * np.ones(mdp.size)
mdp.initializeMDP(i, WS, speedCalculator)

state_size = mdp.size
action_size = 2
agent = PolicyLearner(state_size, action_size)
batch_size = 50

agent.load("Networks/New_reward_stop_condition")

i = np.ones(0)
v = np.ones(0)
wind_heading = np.ones(0)

SIMULATION_TIME = 100
for time in range(SIMULATION_TIME):
    WH = np.random.uniform(mean - std, mean + std, size=wind_samples)
    action = agent.act(mdp.s)
    print(mdp.s[0,state_size-1]/TORAD)
    print(action)
    mdp.transition(action, WS, WH, speedCalculator)
    i = np.concatenate([i, mdp.extractSimulationData()[0, :]])
    v = np.concatenate([v, mdp.extractSimulationData()[1, :]])
    wind_heading = np.concatenate([wind_heading, WH[0:10]])

time_vec = np.linspace(0, SIMULATION_TIME, int((SIMULATION_TIME) / time_step))

f, axarr2 = plt.subplots(3, sharex=True)
axarr2[0].scatter(time_vec, v)
axarr2[1].scatter(time_vec, i / TORAD)
axarr2[2].scatter(time_vec, (i + wind_heading) / TORAD)