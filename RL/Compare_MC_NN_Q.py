
from __future__ import unicode_literals

import matplotlib
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
import matplotlib.pyplot as plt
import numpy as np
import random
from policyLearning import PolicyLearner
from sim.Hysteresis_backup import Hysteresis
from sim.Simulator import TORAD
from sim.mdp import MDP

'''
MDP Parameters
'''
history_duration = 3  # Duration of state history [s]
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

policy_angle = 13

state_size = mdp.size
action_size = 2
agent = PolicyLearner(state_size, action_size, policy_angle)
agent.load("../Networks/policy_learning_i13")
i_rand = random.sample(i_rand_vec, 1)[0]
i = i_rand * np.ones(mdp.size) * TORAD
i = 2 * TORAD * np.ones(mdp.size)

speedCalculator = Hysteresis()
state = mdp.initializeMDP(i, WS, speedCalculator)

Q1 = []
Q2 = []
SIMULATION_TIME = 320
monte_carlo_Q = np.zeros([2, SIMULATION_TIME])
NN_Q = np.zeros([2, SIMULATION_TIME])
# For data visualisation
i = np.ones(0)
v = np.ones(0)
wind_heading = np.ones(0)

actions = [0, 1]

for time in range(SIMULATION_TIME):
    print(time)
    WH = np.random.uniform(mean - std, mean + std, size=wind_samples)
    policy_action = agent.actDeterministicallyUnderPolicy(state)

    # Copying the mdp and the hysteresis for the calculation of "then following policy part"

    # In order not to erase hystory of the real simulation but for computing the "then following policy part"


    next_state, reward = mdp.transition(policy_action, WS, WH, speedCalculator)
    mdp_tmp2 = mdp.copy()
    speedCalculator_tmp2 = speedCalculator.copy()
    speedCalculator_tmp1 = speedCalculator.copy()
    mdp_tmp1 = mdp.copy()

    if policy_action == 1:
        other_action = 0
    if policy_action == 0:
        other_action = 1

    reward_1 = []
    reward_2 = []

    # Reward for the on policy action then following policy
    reward_1.append(reward)
    # Reward for the other action then following policy
    next_state_tmp2, reward_tmp2 = mdp_tmp2.transition(other_action, WS, WH, speedCalculator_tmp2)
    reward_2.append(reward_tmp2)

    policy_action_tmp1 = agent.actDeterministicallyUnderPolicy(next_state)
    policy_action_tmp2 = agent.actDeterministicallyUnderPolicy(next_state_tmp2)
    for tt in range(time + 1, SIMULATION_TIME):
        # Reward for the on policy action then following policy
        next_state_tmp1, reward_tmp1 = mdp_tmp1.transition(policy_action_tmp1, WS, WH, speedCalculator_tmp1)
        policy_action_tmp1 = agent.actDeterministicallyUnderPolicy(next_state_tmp1)
        reward_1.append(reward_tmp1)
        print(reward_tmp1)
        # Reward for the other action then following policy
        next_state_tmp2, reward_tmp2 = mdp_tmp2.transition(policy_action_tmp2, WS, WH, speedCalculator_tmp2)
        policy_action_tmp2 = agent.actDeterministicallyUnderPolicy(next_state_tmp2)
        reward_2.append(reward_tmp2)
        print(reward_tmp2)
    kk = np.linspace(0, SIMULATION_TIME - time - 1, SIMULATION_TIME - time)
    gamma_powk = np.power(agent.gamma, kk)

    reward_1 = np.array(reward_1)
    reward_2 = np.array(reward_2)

    monte_carlo_Q[policy_action, time] = np.sum(reward_1 * gamma_powk)
    monte_carlo_Q[other_action, time] = np.sum(reward_2 * gamma_powk)

    NN_Q[0, time] = agent.evaluate(state)[0]
    NN_Q[1, time] = agent.evaluate(state)[1]
    state = next_state
    # For data visualisation
    i = np.concatenate([i, mdp.extractSimulationData()[0, :]])
    v = np.concatenate([v, mdp.extractSimulationData()[1, :]])
    wind_heading = np.concatenate([wind_heading, WH[0:10]])

time_vec = np.linspace(1, SIMULATION_TIME, SIMULATION_TIME)
time = time_vec[1:int(SIMULATION_TIME / 8)]
monte_carlo_Q = monte_carlo_Q[:, 1:int(SIMULATION_TIME / 8)]
NN_Q = NN_Q[:, 1:int(SIMULATION_TIME / 8)]

time_vec = np.linspace(0, SIMULATION_TIME, int((SIMULATION_TIME) / time_step))
# Plotting results
f, axarr = plt.subplots(2, sharex=True)
# axarr[0].plot(time_vec[1:int(SIMULATION_TIME / 8 / time_step)], v[1:int(SIMULATION_TIME / 8 / time_step)])
axarr[0].plot(time_vec[1:int(SIMULATION_TIME / 8 / time_step)], i[1:int(SIMULATION_TIME / 8 / time_step)] / TORAD)
#
# # Plotting comparison of values

axarr[1].plot(time, NN_Q[0, :], label=r'$Q_{\pi}(s,a=$"bear-off"$)$')
axarr[1].plot(time, monte_carlo_Q[0, :], label=r'$G_t(s,a=$"bear-off"$)$ (MC)')
axarr[1].plot(time, NN_Q[1, :], label=r'$Q_{\pi}(s,a=$"luff"$)$')
axarr[1].plot(time, monte_carlo_Q[1, :], label=r'$G_t(s,a=$"luff"$)$ (MC)')
# axarr[0].set_ylabel("v [m/s]")
axarr[0].set_ylabel("Hdg")
plt.xlabel("t [s]")
plt.legend()

axarr[0].grid(True)
gridlines = axarr[0].get_xgridlines() + axarr[0].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

axarr[1].grid(True)
gridlines = axarr[1].get_xgridlines() + axarr[1].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

f2,ax=plt.subplots()
ax.plot(time, NN_Q[0, :], label=r'$Q_{\pi}(s,a=$"bear-off"$)$')
ax.plot(time, monte_carlo_Q[0, :], label=r'$G_t(s,a=$"bear-off"$)$ (MC)')
ax.plot(time, NN_Q[1, :], label=r'$Q_{\pi}(s,a=$"luff"$)$')
ax.plot(time, monte_carlo_Q[1, :], label=r'$G_t(s,a=$"luff"$)$ (MC)')
plt.xlabel("t [s]")
plt.legend()

ax.grid(True)
gridlines = ax.get_xgridlines() + ax.get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')

# axarr[2].grid(True)
# gridlines = axarr[2].get_xgridlines() + axarr[2].get_ygridlines()
# for line in gridlines:
#     line.set_linestyle('-.')

# # axarr[2].legend("MC","NN")
#
#
#
#
#
#
