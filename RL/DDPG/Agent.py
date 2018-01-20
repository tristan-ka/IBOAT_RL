import sys
import tensorflow as tf
import numpy as np

import random

from QNetwork import Network

from ExperienceBuffer import ExperienceBuffer

from Simulator import TORAD
from mdp import ContinuousMDP

from Displayer import DISPLAYER
import settings as settings

class Agent:

    def __init__(self, sess, history_duration, mdp_step, time_step):
        print("Initializing the agent...")

        self.sess = sess
        self.state_size = int(history_duration / time_step)
        self.action_size = 1 # In terms of dimension, our action is just a float
        self.low_bound = settings.LOW_BOUND
        self.high_bound = settings.HIGH_BOUND
        self.mdp = ContinuousMDP(history_duration, mdp_step, time_step, self.low_bound, self.high_bound)

        self.buffer = ExperienceBuffer()

        print("Creation of the actor-critic network")
        self.network = Network(self.state_size, self.action_size,
                               self.low_bound, self.high_bound)

        self.sess.run(tf.global_variables_initializer())

    def run(self):

        self.total_steps = 0
        self.sess.run(self.network.target_init)

        mean = 45 * TORAD
        std = 0 * TORAD
        wind_samples = 10
        WH = np.random.uniform(mean - std, mean + std, size=10)

        hdg0_rand_vec=(0,2,4,6,8,16,18,20,22)
        # hdg0_rand_vec=(9,14)

        count_luff=0
        count_bear_off=0

        # For exploration visualization (decision of policy, stall, end of stall)
        luff_chosen = 0
        bear_off_chosen = 0
        n_stall = 0
        n_attach = 0
        mat = [0,0]

        epsilon = settings.EPSILON
        noise_scale = settings.NOISE_SCALE_INIT * \
                   (self.high_bound - self.low_bound)

        for ep in range(1, settings.EPISODES+1):
            print("EPISODE NUMBER:")
            print(ep)
            episode_reward = 0
            episode_step = 0
            
            WH = np.random.uniform(mean - std, mean + std, size=10)
            print("WIND HEADING NOISE:")
            print(WH)
            hdg0_rand = random.sample(hdg0_rand_vec, 1)[0]

            hdg0 = hdg0_rand * TORAD * np.ones(10)
            
            # Initial state
            s = self.mdp.initializeMDP(hdg0, WH)
            self.mdp.simulator.hyst.reset()

            # Apply decay on exploration noise process
            #noise_process = np.zeros(self.action_size)
            if ep % settings.FREQ_DECAY ==0:
                # noise_scale = (noise_scale *settings.NOISE_DECAY)
                epsilon = epsilon*settings.NOISE_DECAY

            while episode_step < settings.TIME_PER_EPISODE:

                # choose action based on deterministic policy
                print("CURRENT STATE")
                new_s = np.reshape([s[0],s[1]],[2*self.state_size,1])
                print(new_s)
                a, = self.sess.run(self.network.actions,
                                   feed_dict={self.network.state_ph: [new_s]})
                print("ACTION CHOISIE")
                print(a)
                if a > 0:
                    bear_off_chosen+=1
                elif a < 0:
                    luff_chosen+=1

                # add temporally-correlated exploration noise to action
                #noise_process = np.random.uniform(self.low_bound,self.high_bound)
                #noise_process = settings.EXPLO_THETA * \
                 #   (settings.EXPLO_MU - noise_process) + \
                  #  settings.EXPLO_SIGMA * np.random.randn(self.action_size)

                # Test with epsilon greedy action-taking
                alea = np.random.rand()
                if alea <= epsilon:
                    print("TIRAGE ALEATOIRE")
                    print(alea)
                    print("ACTION BRUITEE")
                    a = np.ones(np.shape(a)) * np.random.uniform(self.low_bound,self.high_bound)
                    print(a)

                # a += noise_scale * noise_process
                if a > 0:
                    count_bear_off+=1
                    if s[0,29]>15.2*np.pi/180:
                        mat[0] += 1
                elif a < 0:
                    count_luff+=1
                    if s[0,29]>15.2*np.pi/180:
                        mat[1] += 1

                if a>self.high_bound:
                    a[0] = self.high_bound
                if a<self.low_bound:
                    a[0] = self.low_bound

                s_, r = self.mdp.transition(a, WH)

                # Stall counting
                if (s_[1, 25] < s_[1, 24] and s_[1, 15] > s_[1, 14]) or (s_[1, 20] < s_[1, 19] and s_[1, 15] > s_[1, 14]):
                    n_stall += 1

                # Attach counting
                if (s_[1, 25] > s_[1, 24] and s_[1, 15] < s_[1, 14]) or (s_[1, 20] > s_[1, 19] and s_[1, 15] < s_[1, 14]):
                    n_attach += 1

                episode_reward += r
                # Printing i and v at 5 last episodes
                if settings.EPISODES - ep <= settings.PLOT_I_V:
                    DISPLAYER.add_i_v(s[0,29],s[1,29])
                new_s_ = np.reshape([s_[0],s_[1]],[2*self.state_size,1])

                self.buffer.add((new_s, a, r,new_s_ ,1.0))

                # update network weights to fit a minibatch of experience
                if self.total_steps % settings.TRAINING_FREQ == 0 and \
                        len(self.buffer) >= settings.BUFFER_SIZE:

                    minibatch = self.buffer.sample()
                    q, _, _,critic_loss,actor_loss = self.sess.run([self.network.q_values_of_given_actions, self.network.critic_train_op, self.network.actor_train_op, self.network.critic_loss, self.network.actor_loss],
                                         feed_dict={
                        self.network.state_ph: np.asarray([elem[0] for elem in minibatch]),
                        self.network.action_ph: np.asarray([elem[1] for elem in minibatch]),
                        self.network.reward_ph: np.asarray([elem[2] for elem in minibatch]),
                        self.network.next_state_ph: np.asarray([elem[3] for elem in minibatch]),
                        self.network.is_not_done_ph: np.asarray([elem[4] for elem in minibatch])})

                    DISPLAYER.add_critic_loss(critic_loss)
                    DISPLAYER.add_actor_loss(actor_loss)
                    if settings.EPISODES - ep <= settings.PLOT_I_V:
                        q_luff = self.sess.run(
                            self.network.prediction,
                            feed_dict={
                                self.network.state_ph: np.asarray([new_s]),
                                self.network.action_ph: np.asarray([[[-0.99]]])})
                        q_bear_off = self.sess.run(
                            self.network.prediction,
                            feed_dict={
                                self.network.state_ph: np.asarray([new_s]),
                                self.network.action_ph: np.asarray([[[0.99]]])})
                        DISPLAYER.add_q(q[0],q_luff[0],q_bear_off[0])

                    # update target networks
                    _ = self.sess.run(self.network.update_targets)

                s = s_
                episode_step += 1
                self.total_steps += 1

            if ep % settings.DISP_EP_REWARD_FREQ == 0:
                print('Episode %2i, Reward: %7.3f, Steps: %i, Final noise scale: %7.3f' %
                      (ep, episode_reward, episode_step, noise_scale))
            # Reward totale reçue
            DISPLAYER.add_reward(episode_reward)
            
        print("n_luff : {}".format(count_luff))
        print("n_bear_off : {}".format(count_bear_off))
        print("luff_chosen : {}".format(luff_chosen))
        print("bear_off_chosen : {}".format(bear_off_chosen))
        print("stall : {}".format(n_stall))
        print("attach : {}".format(n_attach))
        print("final epsilon : {}".format(epsilon))
        print("stall exploration matrix : {}".format(mat))


