import sys

import matplotlib.pyplot as plt

sys.path.append("../sim/")
import numpy as np
from policyLearning import PolicyLearner
from Simulator import TORAD
from mdp import MDP

'''
MISCELLANEOUS FUNCTIONS
'''


def rollOut(time, SIMULATION_TIME, agent, mdp, action, WH):
    reward_list = []
    for tt in range(time + 1, SIMULATION_TIME):
        next_state, reward = mdp.transition(action, WH)
        action = agent.actDeterministicallyUnderPolicy(next_state)
        reward_list.append(reward)
    return reward_list


class ValidateNetwork:
    def __init__(self, hist_duration, mdp_step, time_step, action_size, mean, std, wind_samples, hdg0, src_file,
                 sim_time):
        self.mdp = MDP(hist_duration, mdp_step, time_step)
        self.agent = PolicyLearner(self.mdp.size, action_size)
        self.agent.load(src_file)
        self.mean = mean
        self.std = std
        self.ws = wind_samples
        self.hdg0 = hdg0
        self.src = src_file
        self.sim_time = sim_time

    def generateWind(self):
        return np.random.uniform(self.mean - self.std, self.mean + self.std, size=self.ws)

    def generateQplots(self):

        WH = self.generateWind()
        hdg0 = self.hdg0 * TORAD * np.ones(self.ws)

        state = self.mdp.initializeMDP(hdg0, WH)

        agent1 = PolicyLearner(self.mdp.size, self.agent.action_size)
        agent2 = PolicyLearner(self.mdp.size, self.agent.action_size)
        agent1.load(self.src)
        agent2.load(self.src)

        monte_carlo_Q = np.zeros([2, self.sim_time])
        NN_Q = np.zeros([2, self.sim_time])
        i = np.ones(0)
        v = np.ones(0)
        wind_heading = np.ones(0)

        for time in range(self.sim_time):
            print("t = {} s".format(time))

            agent1.stall = self.agent.get_stall()
            agent2.stall = self.agent.get_stall()

            WH = self.generateWind()
            policy_action = self.agent.actDeterministicallyUnderPolicy(state)

            mdp_tmp2 = self.mdp.copy()
            next_state, reward = self.mdp.transition(policy_action, WH)
            mdp_tmp1 = self.mdp.copy()

            if policy_action == 1:
                other_action = 0
            if policy_action == 0:
                other_action = 1

            next_state_tmp2, reward_tmp2 = mdp_tmp2.transition(other_action, WH)

            policy_action_tmp1 = agent1.actDeterministicallyUnderPolicy(next_state)
            policy_action_tmp2 = agent2.actDeterministicallyUnderPolicy(next_state_tmp2)

            reward_1 = np.array([reward] + rollOut(time, self.sim_time, agent1, mdp_tmp1, policy_action_tmp1, WH))
            reward_2 = np.array(
                [reward_tmp2] + rollOut(time, self.sim_time, agent2, mdp_tmp2, policy_action_tmp2, WH))

            kk = np.linspace(0, self.sim_time - time - 1, self.sim_time - time)
            gamma_powk = np.power(self.agent.gamma, kk)

            monte_carlo_Q[policy_action, time] = np.sum(reward_1 * gamma_powk)
            monte_carlo_Q[other_action, time] = np.sum(reward_2 * gamma_powk)

            NN_Q[0, time] = self.agent.evaluate(self.mdp.s)[0]
            NN_Q[1, time] = self.agent.evaluate(self.mdp.s)[1]

            state = next_state

            # For data visualisation
            i = np.concatenate([i, self.mdp.extractSimulationData()[0, :]])
            v = np.concatenate([v, self.mdp.extractSimulationData()[1, :]])
            wind_heading = np.concatenate([wind_heading, WH[0:10]])

        cut = int(self.sim_time / 8)
        time_vec = np.linspace(0, self.sim_time, self.sim_time)
        time = time_vec[0:cut]
        monte_carlo_Q = monte_carlo_Q[:, 0:cut]
        NN_Q = NN_Q[:, 0:cut]

        time_vec = np.linspace(0, self.sim_time, int((self.sim_time) / self.mdp.dt))

        f, axarr = plt.subplots(3, sharex=True)
        axarr[0].plot(time_vec[1:int(self.sim_time / 8 / self.mdp.dt)],
                      i[1:int(self.sim_time / 8 / self.mdp.dt)] / TORAD)
        axarr[1].plot(time_vec[1:int(self.sim_time / 8 / self.mdp.dt)], v[1:int(self.sim_time / 8 / self.mdp.dt)])
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
        # ax.plot(time, G_pi, label='policy return')
        plt.xlabel("t [s]")
        plt.legend()

        ax.grid(True)
        gridlines = ax.get_xgridlines() + ax.get_ygridlines()
        for line in gridlines:
            line.set_linestyle('-.')

        plt.show()

        return f
