import sys

sys.path.append("../sim/")

import matplotlib.pyplot as plt
import numpy as np
import random

from Sailor import PrioritizedLearningSailor
#from Sailor import DQLearningSailor
#from Sailor import MultiStepLearningSailor

from sim.Simulator import TORAD
from sim.mdp import RealistMDP

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
Random initial conditions 
'''
hdg0_rand_vec=(0,2,4,6,8,10,16,18,20,22)

action_size = 2

'''
Hyper - Paramètres
'''
n_step = 3

agent = PrioritizedLearningSailor(mdp.size, action_size)
batch_size = 80

EPISODES = 100

count_luff=0
count_bear_off=0

loss_of_episode = []
for e in range(EPISODES):
    WH = np.random.uniform(mean - std, mean + std, size=10)
    hdg0_rand = random.sample(hdg0_rand_vec, 1)[0]

    hdg0 = hdg0_rand * TORAD * np.ones(10)
    # initialize the incidence randomly
      #
    #  We reinitialize the memory of the flow
    state = mdp.initializeMDP(hdg0, WH)
    mdp.simulator.hyst.reset()
    loss_sim_list = []
    for time in range(40):
        # print(time)
        WH = np.random.uniform(mean - std, mean + std, size=wind_samples)
        action = agent.act(state)
        if action == 0:
            count_bear_off+=1
        elif action == 1:
            count_luff+=1
        next_state, reward = mdp.transition(action, WH)
        agent.remember(state, action, reward, next_state)
        state = next_state
        if len(agent.memory) > batch_size:
            loss_sim_list.append(agent.replay(batch_size))
    loss_over_simulation_time = np.sum(np.array([loss_sim_list])[0]) / len(np.array([loss_sim_list])[0])
    loss_of_episode.append(loss_over_simulation_time)
    print("episode: {}/{}, Mean Loss = {}"
          .format(e, EPISODES, loss_over_simulation_time))
print("n_luff : {}".format(count_luff))
print("n_bear_off : {}".format(count_bear_off))
agent.save("deep_Q_learning_memory_80_epochs_100_epsilon_50")
plt.semilogy(np.linspace(1, EPISODES, EPISODES), np.array(loss_of_episode))
plt.xlabel("Episodes")
plt.ylabel("Cost")
plt.show()
