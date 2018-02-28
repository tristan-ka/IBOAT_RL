import sys

sys.path.append("../sim/")
import tensorflow as tf
#import matplotlib.pyplot as plt
import numpy as np
import random

from DDPG import DDPGAgent

from Simulator import TORAD
from mdp import ContinuousMDP

from environment import wind


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
w = wind(mean=mean, std=std, samples = wind_samples)
WH = w.generateWind()


'''
Random initial conditions 
'''
# hdg0_rand_vec=(0,2,4,6,8,10,13,15,17,20)
hdg0_rand_vec=(0,7,13)
action_size = 2
action_size_DDPG = 1

'''
Initialize Simulation
'''

batch_size = 32

EPISODES = 1000

count_luff=0
count_bear_off=0

'''
Start of training phase
'''

actor_loss_of_episode = []
critic_loss_of_episode = []
Q_predictions_3 = []
Q_predictions_minus_3 = []
Q_predictions_0 = []

tf.reset_default_graph()
with tf.Session() as sess:
    agent = DDPGAgent(mdp.size, action_size_DDPG, lower_bound, upper_bound, sess)
    for e in range(EPISODES):
        WH = w.generateWind()
        hdg0_rand = random.sample(hdg0_rand_vec, 1)[0]

        hdg0 = hdg0_rand * TORAD * np.ones(10)
        # initialize the incidence randomly
          #
        #  We reinitialize the memory of the flow
        state = mdp.initializeMDP(hdg0, WH)
        mdp.simulator.hyst.reset()
        actor_loss_sim_list = []
        critic_loss_sim_list = []

        for time in range(80):
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
            if state[0,59]>19*TORAD:
                reward= reward-0.5
            print("the reward is: {}".format(reward))
            agent.remember(state, action, reward, next_state)

            # For Critic visualization during learning
            # print("Q value for -3,0,3 in current state: {},{},{}".format(agent.evaluate(state,-2.99),agent.evaluate(state,0),agent.evaluate(state,2.99)))

            state = next_state
        if len(agent.memory) >= batch_size:
            a_loss, c_loss = agent.replay(batch_size)
            actor_loss_sim_list.append(a_loss)
            critic_loss_sim_list.append(c_loss)
            print("Actor network loss: {}".format(a_loss))
            print("Critic network loss: {}".format(c_loss))

        # We apply decay on epsilon greedy actions
        agent.noise_decay(e)

        # We save CNN weights every 5000 epochs
        if e % 5000 == 0 and e != 0:
            agent.save("../Networks/DDPG/Slow_Learning_"+ str(e) +"_epochs")

        # Critic evaluation every epoch
        Q_predictions_3.append(list(agent.evaluate(state, 3.0))[0])
        Q_predictions_minus_3.append(list(agent.evaluate(state, -3.0))[0])
        Q_predictions_0.append(list(agent.evaluate(state, 0))[0])

        actor_loss_over_simulation_time = np.sum(np.array([actor_loss_sim_list])) / len(np.array([actor_loss_sim_list]))
        actor_loss_of_episode.append(actor_loss_over_simulation_time)
        critic_loss_over_simulation_time = np.sum(np.array([critic_loss_sim_list])) / len(np.array([critic_loss_sim_list]))
        critic_loss_of_episode.append(critic_loss_over_simulation_time)
        print("episode: {}/{}, Actor Mean Loss = {}"
              .format(e, EPISODES, actor_loss_over_simulation_time))
        print("episode: {}/{}, Critic Mean Loss = {}"
              .format(e, EPISODES, critic_loss_over_simulation_time))

    # Save final tensorflow session
    agent.save("../Networks/DDPG/Slow_Learning_FullNoise_1000")

# Save test elements
commentaire = "Slow_Learning_FullNoise_1000"
np.save("../Networks/DDPG/actor_loss_"+commentaire,np.array(actor_loss_of_episode))
np.save("../Networks/DDPG/critic_loss_"+commentaire,np.array(critic_loss_of_episode))
np.save("../Networks/DDPG/Q3_"+commentaire, np.array(Q_predictions_3))
np.save("../Networks/DDPG/Qminus3_"+commentaire, np.array(Q_predictions_minus_3))
np.save("../Networks/DDPG/Q0_"+commentaire, np.array(Q_predictions_0))

print("n_luff : {}".format(count_luff))
print("n_bear_off : {}".format(count_bear_off))
#plt.semilogy(np.linspace(1, EPISODES, EPISODES), np.array(actor_loss_of_episode))
#plt.xlabel("Episodes")
#plt.ylabel("Cost")
#plt.semilogy(np.linspace(1, EPISODES, EPISODES), np.array(critic_loss_of_episode))
#plt.xlabel("Episodes")
#plt.ylabel("Cost")
#plt.show()
