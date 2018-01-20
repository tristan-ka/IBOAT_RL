import sys

sys.path.append("../sim/")

import matplotlib.pyplot as plt
import numpy as np

from Sailor import PrioritizedLearningSailor

from sim.Simulator import TORAD
from sim.mdp import RealistMDP

## Comparaison déterministe entre un modèle DQN et le dicounted return de Monte Carlo

'''
MDP Parameters
'''
history_duration = 3  # Duration of state history [s]
mdp_step = 1  # Step between each state transition [s]
time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
mdp = RealistMDP(history_duration, mdp_step, time_step)

'''
WIND CONDITIONS
'''
mean = 45 * TORAD
std = 0 * TORAD
wind_samples = 10
WH = np.random.uniform(mean - std, mean + std, size=10)

'''
Hyper parameters
'''
n_step = 3

'''
MDP and agent INITIALISATION
'''
hdg0 = 6 * TORAD * np.ones(10)
state = mdp.initializeMDP(hdg0, WH)
action_size = 2
# agent = MultiStepLearningSailor(mdp.size, action_size, n_step,mean/TORAD,std/TORAD,history_duration, mdp_step, time_step)
agent = PrioritizedLearningSailor(mdp.size, action_size)
CNN = "deep_Q_learning_memory_80_epochs_100_epsilon_50"
agent.load(CNN)


Q1 = []
Q2 = []
SIMULATION_TIME = 400
monte_carlo_Q = np.zeros([2, SIMULATION_TIME])
NN_Q = np.zeros([2, SIMULATION_TIME])

# For data visualisation
i = np.ones(0)
v = np.ones(0)

wind_heading = np.ones(0)

actions = [0, 1]

for time in range(SIMULATION_TIME):
    print(time)
    print("i : {}".format(state[0, mdp.size - 1] / TORAD))
    WH = np.random.uniform(mean - std, mean + std, size=10)
    policy_action = agent.act(state)
    print("action : {}".format(policy_action))

    mdp_tmp2 = mdp.copy()
    next_state, reward = mdp.transition(policy_action, WH)
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
    next_state_tmp2, reward_tmp2 = mdp_tmp2.transition(other_action, WH)
    reward_2.append(reward_tmp2)

    policy_action_tmp1 = agent.act(next_state)
    policy_action_tmp2 = agent.act(next_state_tmp2)

    for tt in range(time + 1, SIMULATION_TIME):
        # Reward for the on policy action then following policy
        next_state_tmp1, reward_tmp1 = mdp_tmp1.transition(policy_action_tmp1, WH)
        policy_action_tmp1 = agent.act(next_state_tmp1)
        reward_1.append(reward_tmp1)
        #print(reward_tmp1)
        # Reward for the other action then following policy
        next_state_tmp2, reward_tmp2 = mdp_tmp2.transition(policy_action_tmp2, WH)
        policy_action_tmp2 = agent.act(next_state_tmp2)
        reward_2.append(reward_tmp2)
        #print(reward_tmp2)
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


time_vec = np.linspace(0, SIMULATION_TIME, SIMULATION_TIME)
time = time_vec[0:int(SIMULATION_TIME / 8)]
monte_carlo_Q = monte_carlo_Q[:, 0:int(SIMULATION_TIME / 8)]
NN_Q = NN_Q[:, 0:int(SIMULATION_TIME / 8)]

time_vec = np.linspace(0, SIMULATION_TIME, int((SIMULATION_TIME) / time_step))
# Plotting results
f, axarr = plt.subplots(3, sharex=True)
# axarr[0].plot(time_vec[1:int(SIMULATION_TIME / 8 / time_step)], v[1:int(SIMULATION_TIME / 8 / time_step)])
axarr[0].plot(time_vec[1:int(SIMULATION_TIME / 8 / time_step)], i[1:int(SIMULATION_TIME / 8 / time_step)] / TORAD)
#
# # Plotting comparison of values
axarr[1].plot(time_vec[1:int(SIMULATION_TIME / 8 / time_step)], v[1:int(SIMULATION_TIME / 8 / time_step)])
axarr[2].plot(time, NN_Q[0, :], label=r'$Q_{\pi}(s,a=$"bear-off"$)$')
axarr[2].plot(time, monte_carlo_Q[0, :], label=r'$G_t(s,a=$"bear-off"$)$ (MC)')
axarr[2].plot(time, NN_Q[1, :], label=r'$Q_{\pi}(s,a=$"luff"$)$')
axarr[2].plot(time, monte_carlo_Q[1, :], label=r'$G_t(s,a=$"luff"$)$ (MC)')
# axarr[0].set_ylabel("v [m/s]")
axarr[0].set_ylabel("angle of attack")
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

f2, ax = plt.subplots()
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
plt.show()
