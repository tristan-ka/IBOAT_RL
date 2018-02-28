import copy
import math

import Simulator
import numpy as np

TORAD = math.pi / 180


class MDP:
    """
    Markov Decision process modelization of the transition

    :ivar float history_duration: Duration of the memory.
    :ivar float simulation_duration: Duration of the memory.
    :ivar int size: size of the first dimension of the state.
    :ivar float dt: time step between each value of the state.
    :ivar np.array() s: state containing the history of angles of attacks and velocities.
    :ivar range idx_memory: indices corresponding to the values shared by two successive states.
    :ivar Simulator simulator: Simulator used to compute new values after a transition.
    :ivar float reward: reward associated with a transition.
    :ivar float discount: discount factor.
    :ivar float action: action for transition.

    """

    def __init__(self, duration_history, duration_simulation, delta_t):
        self.history_duration = duration_history
        self.simulation_duration = duration_simulation
        self.size = int(duration_history / delta_t)

        self.dt = delta_t
        self.s = np.array((self.size, 2))
        self.idx_memory = range(int(self.simulation_duration / self.dt), self.size)

        self.simulator = None
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

    '''
    Uses a vector of incidence to compute first state 
    '''

    def initializeMDP(self, hdg0, WH):
        """
        Initialization of the Markov Decicison Process.

        :param hdg0: initial heading of the boat.
        :type hdg0: float

        :param WH np.array(): Vector of wind heading.


        :return: s initialized state
        :rtype: np.array()

        """
        self.simulator = Simulator.Simulator(self.simulation_duration, self.dt)

        # Delay of the dynamic
        # be carefull with initialisation : due to delay we must initialize the taustep+1 first angles
        self.simulator.hdg = copy.deepcopy(hdg0)

        self.simulator.computeNewValues(0, WH)

        fill = np.zeros(int(self.size - self.simulator.size))
        self.s = np.array([np.concatenate([self.simulator.hdg, fill]), np.concatenate([self.simulator.vmg, fill])])

        self.reward = np.sum(self.simulator.vmg) / self.simulator.size
        return self.s

    def transition(self, action, WH):
        """
        Computes the mdp state when an action is applied.
        :param action: action to make. Action should be either 0 or 1. If action is 0 then the heading is increased by one degree.
        If the action is 1 the heading is decreased by one degree
        :param WH: vector of wind heading seen during the period of action making.
        :return: the state and reward yielded by taking action action from the current state of the mdp.
        """

        if action != 0 and action != 1:
            raise ValueError("Invalid action. Could not generate transition.")

        self.action = action

        if action == 0:
            delta_hdg = 1 * TORAD
        if action == 1:
            delta_hdg = -1 * TORAD

        hdg, vmg = self.simulator.computeNewValues(delta_hdg, WH)

        self.s = np.array(
            [np.concatenate([self.s[0, self.idx_memory], hdg + WH + self.simulator.sail_pos]),
             np.concatenate([self.s[1, self.idx_memory], vmg])])

        self.reward = np.sum(self.simulator.vmg) / self.simulator.size / Simulator.BSTRESH

        return self.s, self.reward

    def extractSimulationData(self):
        """
        Extract the new simulated data. Gets rid of the simulation data that are present more than twice in the state memory.
        :return: The data of interest in the state memory.
        """
        return self.s[:, int((self.history_duration - self.simulation_duration) / self.dt):len(self.s[0, :])]

    def policy(self, i_treshold):
        if (self.s[0, self.size - 1] < i_treshold):
            action = 0
        else:
            action = 1
        return action


class ContinuousMDP:
    """
        Markov Decision process modelization of the transition
        Based on Realistic Simulator of Iboat autonomous sailboat provided by Simulink
        Compatible with continuous action. See documentation of :class:`MDP` for all undocumented method as they are the same.

        :ivar float history_duration: Duration of the memory.
        :ivar float simulation_duration: Duration of the memory.
        :ivar int size: size of the first dimension of the state.
        :ivar float dt: time step between each value of the state.
        :ivar np.array() s: state containing the history of angles of attacks and velocities.
        :ivar range idx_memory: indices corresponding to the values shared by two successive states.
        :ivar Simulator simulator: Simulator used to compute new values after a transition.
        :ivar float reward: reward associated with a transition.
        :ivar float discount: discount factor.
        :ivar float action: action for transition.

        """

    def __init__(self, duration_history, duration_simulation, delta_t, LOWER_BOUND, UPPER_BOUND):
        self.history_duration = duration_history
        self.simulation_duration = duration_simulation
        self.size = int(duration_history / delta_t)
        self.LOWER_BOUND = LOWER_BOUND
        self.UPPER_BOUND = UPPER_BOUND
        self.dt = delta_t
        self.s = np.array((self.size, 2))
        self.idx_memory = range(int(self.simulation_duration / self.dt), self.size)

        self.simulator = None

        self.reward = None
        self.discount = None
        self.action = None

    def copy(self):
        return copy.deepcopy(self)

    '''
    Uses a state matrix or a vector of incidence (Hdg and WH) to compute first state 
    '''

    def initializeState(self, state):
        self.simulator = Simulator.RealistSimulator(self.simulation_duration, self.dt)
        self.s = state

    def initializeMDP(self, hdg0, WH):
        self.simulator = Simulator.RealistSimulator(self.simulation_duration, self.dt)

        # Delay of the dynamic
        # be carefull with initialisation : due to delay we must initialize the taustep+1 first angles
        self.simulator.hdg = copy.deepcopy(hdg0)

        self.simulator.computeNewValues(0, WH)

        fill = np.zeros(int(self.size - self.simulator.size))
        self.s = np.array([np.concatenate([self.simulator.hdg, fill]), np.concatenate([self.simulator.vmg, fill])])

        self.reward = np.sum(self.simulator.vmg) / self.simulator.size
        return self.s

    def transition(self, action, WH):
        if action < self.LOWER_BOUND or action > self.UPPER_BOUND:
            raise ValueError("Action out of bound. Could not generate transition.")

        self.action = action

        delta_hdg = action * TORAD

        hdg, vmg = self.simulator.computeNewValues(delta_hdg, WH)

        self.s = np.array(
            [np.concatenate([self.s[0, self.idx_memory], hdg + WH + self.simulator.sail_pos]),
             np.concatenate([self.s[1, self.idx_memory], vmg])])

        self.reward = np.sum(self.simulator.vmg) / self.simulator.size / Simulator.BSTRESH

        return self.s, self.reward

    def extractSimulationData(self):
        return self.s[:, int((self.history_duration - self.simulation_duration) / self.dt):len(self.s[0, :])]
