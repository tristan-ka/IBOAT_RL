import math
import random

import numpy as np
from Hysteresis import Hysteresis
from mdp import MDP

from dqn import DQNAgent
from sim import Simulator

TORAD = math.pi / 180

'''
MDP Parameters
'''
history_duration = 3  # Duration of state history [s]
mdp_step = 1  # Step between each state transition [s]
time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
delay=5
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

state_size = mdp.size
action_size = 2
agent = DQNAgent(state_size, action_size)
batch_size = 32
EPISODES = 1

for e in range(EPISODES):
    i_rand = random.uniform(Simulator.IMIN, Simulator.IMAX)
    i = i_rand * np.ones(mdp.size)
    # initialize the incidence randomly
    speedCalculator.reset()  #
    #  We reinitialize the memory of the flow
    state = mdp.initializeMDP(i, WS, speedCalculator)
    for time in range(500):
        print(time)
        WH = np.random.uniform(mean - std, mean + std, size=wind_samples)
        action = agent.act(state)
        next_state, reward = mdp.transition(action, WS, WH, speedCalculator)
        agent.remember(state, action, reward, next_state)
        state = next_state
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)


        score_state=math.fabs(agent.act(state)[0,mdp.size]-14*TORAD)
    print("episode: {}/{}, score: {}, r: {:.2}"
              .format(e, EPISODES, score_state, reward))





# env = gym.make('CartPole-v1')
#     state_size = env.observation_space.shape[0]
#     action_size = env.action_space.n
#     agent = DQNAgent(state_size, action_size)
#     # agent.load("./save/cartpole-master.h5")
#     done = False
#     batch_size = 32
#
#     for e in range(EPISODES):
#         state = env.reset()
#         state = np.reshape(state, [1, state_size])
#         for time in range(500):
#             # env.render()
#             action = agent.act(state)
#             next_state, reward, done, _ = env.step(action)
#             reward = reward if not done else -10
#             next_state = np.reshape(next_state, [1, state_size])
#             agent.remember(state, action, reward, next_state, done)
#             state = next_state
#             if done:
#                 print("episode: {}/{}, score: {}, e: {:.2}"
#                       .format(e, EPISODES, time, agent.epsilon))
#                 break
#         if len(agent.memory) > batch_size:
#             agent.replay(batch_size)
