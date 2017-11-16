import math
import random

import numpy as np
import matplotlib.pyplot as plt
from policyLearning import PolicyLearner
from sim.Hysteresis_backup import Hysteresis
from sim.mdp import MDP

TORAD = math.pi / 180

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
speedCalculator = Hysteresis()

state_size = mdp.size
action_size = 2
policy_angle=13
agent = PolicyLearner(state_size, action_size,policy_angle)
batch_size = 32



EPISODES = 1000

loss_of_episode=[]
for e in range(EPISODES):
    i_rand = random.sample(i_rand_vec, 1)[0]
    i = i_rand * np.ones(mdp.size) * TORAD
    # initialize the incidence randomly
    speedCalculator.reset()  #
    #  We reinitialize the memory of the flow
    state = mdp.initializeMDP(i, WS, speedCalculator)
    loss_sim_list = []
    for time in range(75):
        # print(time)
        WH = np.random.uniform(mean - std, mean + std, size=wind_samples)
        action = agent.actUnderPolicy(state)
        # print(state[0, mdp.size - 1] * Simulator.TODEG)
        # print(action)
        next_state, reward = mdp.transition(action, WS, WH, speedCalculator)
        agent.remember(state, action, reward, next_state)
        state = next_state
        if len(agent.memory) > batch_size:
            loss_sim_list.append(agent.replay(batch_size))
    loss_over_simulation_time = np.sum(np.array([loss_sim_list])[0]) / len(np.array([loss_sim_list])[0])
    loss_of_episode.append(loss_over_simulation_time)
    print("episode: {}/{}, Mean Loss = {}"
          .format(e, EPISODES,loss_over_simulation_time))

agent.save("policy_learning_i13")
plt.semilogy(np.linspace(1,EPISODES,EPISODES),np.array(loss_of_episode))
plt.xlabel("Episodes")
plt.ylabel("Cost")

