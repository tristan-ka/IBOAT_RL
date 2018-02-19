import copy
import math

import numpy as np

import Simulator_realistic

class mdp_realistic:

	def __init__(self, mdp_duration, hdg0, speed0, history_duration, mdp_step, delta_t):
		self.history_duration = history_duration
		self.mdp_step = mdp_step
		self.size = int(history_duration / delta_t)
		self.dt = delta_t
		self.s = np.array((self.size, 2))
		self.idx_memory = range(int(self.mdp_step / self.dt), self.size)
		self.simulator = Simulator_realistic.Simulator_realistic(mdp_duration, hdg0, speed0)
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

	def initializeMDP(self, truewindheading, truewindheading_std):
		i,sog = self.simulator.step(0, 
									truewindheading, 
									truewindheading_std, 
									self.history_duration, 
									self.dt)
		self.s = np.array([i,sog])
		print(self.s)
		self.reward = np.sum(sog[self.size-int(self.mdp_step / self.dt):])/len(sog[self.size-int(self.mdp_step / self.dt):]) / Simulator_realistic.BSTRESH
		return self.s


	def computeState(self, action, truewindheading, truewindheading_std):
		"""
		Computes the mdp state when an action is applied.
		:param action:
		:param WH:
		:return:
		"""
		if action != 0 and action != 1:
			raise ValueError("Invalid action. Could not generate transition.")

		self.action = action

		if action == 0:
			delta_hdg = 1
		if action == 1:
			delta_hdg = -1 

		i, sog = self.simulator.step(delta_hdg, truewindheading, truewindheading_std, self.mdp_step, self.dt)

		self.s = np.array(
			[np.concatenate([self.s[0, self.idx_memory], i]),
			np.concatenate([self.s[1, self.idx_memory], sog])])

		self.reward = np.sum(sog) / len(sog) / Simulator_realistic.BSTRESH

	'''
	Update the current state from previous state 
	'''

	def transition(self, action, truewindheading, truewindheading_std):
		self.computeState(action, truewindheading, truewindheading_std)
		return self.s, self.reward

	def extractSimulationData(self):
		return self.s[:, int((self.history_duration - self.mdp_step) / self.dt)+1:len(self.s[0, :])]

