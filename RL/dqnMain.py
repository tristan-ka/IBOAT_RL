import sys

sys.path.append("../sim/")
from math import *

import matplotlib.pyplot as plt
import numpy as np
from Simulator import TORAD
from dqn import DQNAgent
from environment import wind
from mdp import MDP
import random

'''
MDP Parameters
'''
mdp = MDP(duration_history=3, duration_simulation=1, delta_t=0.1)

'''
Environment Parameters
'''
w = wind(mean=45 * TORAD, std=0 * TORAD, samples=10)
WH = w.generateWind()

hdg0 = 0 * np.ones(10)
mdp.initializeMDP(hdg0, WH)

agent = DQNAgent(mdp.size, action_size=2)
#agent.load("../Networks/lighter_archi")
batch_size = 50

EPISODES = 500
hdg0_rand_vec=[-3, 0, 3, 6, 9, 12, 15, 18, 21]

loss_of_episode = []
i = []
v = []
r = []
for e in range(EPISODES):
    WH = w.generateWind()
    hdg0_rand = random.choice(hdg0_rand_vec) * TORAD
    hdg0 = hdg0_rand * np.ones(10)

    mdp.simulator.hyst.reset()

    #  We reinitialize the memory of the flow
    state = mdp.initializeMDP(hdg0, WH)
    loss_sim_list = []
    for time in range(80):
        WH = w.generateWind()
        action = agent.act(state)
        next_state, reward = mdp.transition(action, WH)
        agent.remember(state, action, reward, next_state)  # store the transition + the state flow in the
        # final state !!
        state = next_state
        if len(agent.memory) >= batch_size:
            loss_sim_list.append(agent.replay(batch_size))
            # For data visualisation
            i.append(mdp.s[0, -1])
            v.append(mdp.s[1, -1])
            r.append(mdp.reward)

    loss_over_simulation_time = np.sum(np.array([loss_sim_list])[0]) / len(np.array([loss_sim_list])[0])
    loss_of_episode.append(loss_over_simulation_time)
    print("Initial Heading : {}".format(hdg0_rand))
    print("----------------------------")
    print("episode: {}/{}, Mean Loss = {}".format(e, EPISODES, loss_over_simulation_time))
    print("----------------------------")
agent.save("../Networks/dqn-test")

# plt.semilogy(np.linspace(1, EPISODES, EPISODES), np.array(loss_of_episode))
# plt.xlabel("Episodes")
# plt.ylabel("Cost")

f, axarr = plt.subplots(4, sharex=True)
axarr[0].plot(np.array(i[floor(len(i) / 2):len(i) - 1]) / TORAD)
axarr[1].plot(v[floor(len(i) / 2):len(i) - 1])
axarr[2].plot(r[floor(len(i) / 2):len(i) - 1])
axarr[3].semilogy(loss_sim_list[floor(len(i) / 2):len(i) - 1])
axarr[0].set_ylabel("angle of attack")
axarr[1].set_ylabel("v")
axarr[2].set_ylabel("r")
axarr[3].set_ylabel("cost")
plt.show()
