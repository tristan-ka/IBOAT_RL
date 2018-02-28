import copy

import numpy as np

import Realistic_Simulator
from ctypes import *


class mdp_realistic:
    """
    Markov Decision process modelization of the transition
    :ivar float history_duration: Duration of the memory.
    :ivar float simulation_duration: Duration of the memory.
    :ivar float mdp_step: Sampling period for decision making measures
    :ivar int size: size of the first dimension of the state.
    :ivar float dt: time step between each value of the state.
    :ivar np.array() s: state containing the history of angles of attacks and velocities.
    :ivar range idx_memory: indices corresponding to the values shared by two successive states.
    :ivar Realistic Simulator simulator: Simulator used to compute new values after a transition.
    :ivar float reward: reward associated with a transition.
    :ivar float discount: discount factor.
    :ivar float action: action for transition.
    """
    def __init__(self, mdp_duration, history_duration, mdp_step, delta_t):
        self.history_duration = history_duration
        self.simulation_duration = mdp_duration
        self.mdp_step = mdp_step
        self.size = int(history_duration / delta_t)
        self.dt = delta_t
        self.s = np.array((self.size, 2))
        self.idx_memory = range(int(self.mdp_step / self.dt), self.size)
        self.simulator = Realistic_Simulator.Realistic_Simulator()
        self.reward = None
        self.discount = None
        self.action = None

    def copy(self):
        """
        Copy the MDP object

        :return: Deep copy of the object.
        :rtype: MDP
        """
        return copy.deepcopy(self)

    def initializeMDP(self, hdg0, sailpos, speed0, truewaterheading, truewaterspeed, truewindspeed, truewindheading):
        """
        Initialization of the Markov Decision Process, filling the values of the history memory
        :param hdg0: Initial cap
        :param speed0: Initial speed
        :param truewaterheading: Absolute water heading
        :param truewaterspeed: Absolute water speed
        :param truewindspeed: Absolute wind speed
        :param truewindheading: Absolute wind heading
        :param truewindheading_std: Wind heading variance
        :return: state s
        """

        self.simulator.U_hdg_ref = c_double(hdg0)  # main variable
        self.simulator.U_sailpos = c_double(sailpos)
        self.simulator.U_truewindspeed = c_double(truewindspeed)
        self.simulator.U_truewindheading = c_double(truewindheading)
        self.simulator.U_truewaterspeed = c_double(truewaterspeed)
        self.simulator.U_truewaterheading = c_double(truewaterheading)
        self.simulator.U_hdg0 = c_double(hdg0)
        self.simulator.U_speed0 = c_double(speed0)

        i, sog = self.simulator.step(self.history_duration,self.dt)

        self.s = np.array([i, sog])
        print(self.s)
        self.reward = np.sum(sog[self.size - int(self.mdp_step / self.dt):]) / len(
            sog[self.size - int(self.mdp_step / self.dt):]) / Realistic_Simulator.BSTRESH
        return self.s

    def transition(self, action):
        """
        Computes the mdp state when an action is applied.
        :param action: Action to perform
        :return: state, reward
        """

        self.action = action
        
        
        if action == 0:
            delta_hdg = 1
        elif action == 1:
            delta_hdg = -1
        else:
            delta_hdg = 0

        self.simulator.U_hdg_ref.value = float(self.simulator.U_hdg_ref.value) + delta_hdg
        i, sog = self.simulator.step(self.mdp_step, self.dt)

        self.s = np.array(
            [np.concatenate([self.s[0, self.idx_memory], i]),
             np.concatenate([self.s[1, self.idx_memory], sog])])

        self.reward = np.sum(sog) / len(sog) / Realistic_Simulator.BSTRESH
        return self.s, self.reward


    def extractSimulationData(self):
        """
        Recover data from the simulation and add it to the state
        :return: state s
        """
        return self.s[:, int((self.history_duration - self.mdp_step) / self.dt) + 1:len(self.s[0, :])]