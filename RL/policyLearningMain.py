import sys

sys.path.append("../sim/")

import matplotlib.pyplot as plt
import numpy as np
from math import *

from policyLearning import PolicyLearner

from Simulator import TORAD
from mdp import MDP
from environment import wind

'''
MDP Parameters
'''
mdp = MDP(duration_history=6, duration_simulation=1, delta_t=0.1)

'''
WIND CONDITIONS
'''
w = wind(mean=45 * TORAD, std=0 * TORAD, samples = 10)
WH = w.generateWind()

hdg0 = 0 * np.ones(10)
mdp.initializeMDP(hdg0, WH)

agent = PolicyLearner(mdp.size, action_size=2, batch_size=32)
#agent.load("../Networks/epsilon_pi")
EPISODES = 1
count_luff = 0
count_bear_off = 0

loss_of_episode = []
i = []
v = []
r = []
for e in range(EPISODES):
    WH = w.generateWind()
    hdg0_rand = 0 * TORAD
    if hdg0_rand + w.mean + mdp.simulator.sail_pos > agent.policy_angle:
        agent.init_stall(1, mdp)
    else:
        agent.init_stall(0, mdp)

    hdg0 = hdg0_rand * np.ones(10)
    # initialize the incidence randomly
    mdp.simulator.hyst.reset()  #
    #  We reinitialize the memory of the flow

    state = mdp.initializeMDP(hdg0, WH)
    loss_sim_list = []
    for time in range(50000):
        w.generateWind()
        action = agent.actUnderPolicy(state)
        if action == 0:
            count_bear_off += 1
        elif action == 1:
            count_luff += 1
        next_state, reward = mdp.transition(action, WH)
        agent.remember(state, action, reward, next_state, agent.stall)
        state = next_state
        if len(agent.memory) >= agent.batch_size:
            loss_sim_list.append(agent.replay(agent.batch_size))
            print("time: {}, Loss = {}".format(time, loss_sim_list[-1]))
            # For data visualisation
            i.append(mdp.s[0, -1])
            v.append(mdp.s[1, -1])
            r.append(mdp.reward)

    # loss_over_simulation_time = np.sum(np.array([loss_sim_list])[0]) / len(np.array([loss_sim_list])[0])
    # loss_of_episode.append(loss_over_simulation_time)
    print("Initial Heading : {}".format(hdg0_rand))
    # print("episode: {}/{}, Mean Loss = {}"
    # .format(e, EPISODES, loss_over_simulation_time))

print("n_luff : {}".format(count_luff))
print("n_bear_off : {}".format(count_bear_off))

agent.save("../Networks/test-policy-learning")

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
