import sys

sys.path.append("../sim/")

import matplotlib.pyplot as plt
import numpy as np
import random

from DDPG import DDPGAgent

from Simulator import TORAD
from mdp import ContinuousMDP

'''
MDP Parameters
'''
history_duration = 6  # Duration of state history [s]
mdp_step = 1  # Step between each state transition [s]
time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
lower_bound = -3.0
upper_bound = 3.0
mdp = ContinuousMDP(history_duration, mdp_step, time_step,lower_bound,upper_bound)

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
hdg0_rand_vec=(0,2,4,6,8,10,13,15,17,20)

action_size = 2
action_size_DDPG = 1

'''
Initialize DDPG Agent
'''

agent = DDPGAgent(mdp.size, action_size_DDPG,lower_bound,upper_bound)
batch_size = 32

EPISODES = 1000

count_luff=0
count_bear_off=0

'''
Start of training phase
'''

actor_loss_of_episode = []
critic_loss_of_episode = []
for e in range(EPISODES):
    WH = np.random.uniform(mean - std, mean + std, size=10)
    hdg0_rand = random.sample(hdg0_rand_vec, 1)[0]

    hdg0 = hdg0_rand * TORAD * np.ones(10)
    # initialize the incidence randomly
      #
    #  We reinitialize the memory of the flow
    state = mdp.initializeMDP(hdg0, WH)
    mdp.simulator.hyst.reset()
    actor_loss_sim_list = []
    critic_loss_sim_list = []

    for time in range(30):
        # print(time)
        WH = np.random.uniform(mean - std, mean + std, size=wind_samples)
        action = agent.act_epsilon_greedy(state)
        print("the last seen (i,v) is: {}".format((state[0,59]/TORAD,state[1,59])))
        print("the action is: {}".format(action))
        if action <= 0:
            count_bear_off+=1
        elif action > 0:
            count_luff+=1
        next_state, reward = mdp.transition(action, WH)
        # Reward shaping
        # reward = 1 - 15 *(next_state[0,59]-15.9*TORAD)**2# normalized reward shaping to get close to stall incidence
        # reward = state[1,59]/2.1
        # if state[0,59]>0.3:
            #reward= reward-0.5
        print("the reward is: {}".format(reward))
        agent.remember(state, action, reward, next_state)
        state = next_state
        if len(agent.memory) >= batch_size:
            a_loss, c_loss = agent.replay(batch_size)
            actor_loss_sim_list.append(a_loss)
            critic_loss_sim_list.append(c_loss)
            print("Actor network loss: {}".format(a_loss))
            print("Critic network loss: {}".format(c_loss))

    actor_loss_over_simulation_time = np.sum(np.array([actor_loss_sim_list])) / len(np.array([actor_loss_sim_list]))
    actor_loss_of_episode.append(actor_loss_over_simulation_time)
    critic_loss_over_simulation_time = np.sum(np.array([critic_loss_sim_list])) / len(np.array([critic_loss_sim_list]))
    critic_loss_of_episode.append(critic_loss_over_simulation_time)

    print("episode: {}/{}, Actor Mean Loss = {}"
          .format(e, EPISODES, actor_loss_over_simulation_time))
    print("episode: {}/{}, Critic Mean Loss = {}"
          .format(e, EPISODES, critic_loss_over_simulation_time))
print("n_luff : {}".format(count_luff))
print("n_bear_off : {}".format(count_bear_off))
agent.save("DDPG_learning")
plt.semilogy(np.linspace(1, EPISODES, EPISODES), np.array(actor_loss_of_episode))
plt.xlabel("Episodes")
plt.ylabel("Cost")
plt.semilogy(np.linspace(1, EPISODES, EPISODES), np.array(critic_loss_of_episode))
plt.xlabel("Episodes")
plt.ylabel("Cost")
plt.show()
