import sys
sys.path.append("../../../sim/")
import Realistic_mdp

import os

from parameters import MAX_EPISODE_STEPS


class Environment:

    def __init__(self):
        '''
        self.env_no_frame_skip = gym.make(ENV)
        self.env = gym.wrappers.SkipWrapper(FRAME_SKIP)(self.env_no_frame_skip)
        self.render = False
        self.images = []
        '''
        self.history_duration = 6  # Duration of state history [s]
        self.mdp_duration = MAX_EPISODE_STEPS
        self.mdp_step = 1  # Step between each state transition [s]
        self.time_step = 0.1  # time step [s] <-> 10Hz frequency of data acquisition
        self.low_bound = -3.0
        self.high_bound = 3.0
        self.action_size = 1
        self.truewindspeed = 10
        self.truewaterheading = 0
        self.truewaterspeed = 0
        self.sailpos = 40
        self.speed0 = 0
        self.mdp = Realistic_mdp.mdp_realistic(self.mdp_duration, self.history_duration, self.mdp_step, self.time_step)
    def get_state_size(self):
        #try:
        return 61 #(self.mdp.size)
        #except AttributeError:
        #    return list(self.env.observation_space.shape)

    def get_action_size(self):
        #try:
        return self.action_size
        #except AttributeError:
        #    return self.env.action_space.shape[0]

    def get_bounds(self):
        return self.low_bound,self.high_bound #self.env.action_space.low, self.env.action_space.high

    #def set_render(self, render):
    #    self.render = render

    def reset(self,hdg0,WH):
        s = self.mdp.initializeMDP(hdg0[0], self.sailpos,self.speed0,self.truewaterheading, self.truewaterspeed,self.truewindspeed,WH[0])
        return s
        #return self.env.reset()

    def act(self, action,WH):
        #if self.render:
        #    self.env.render()
        next_state, reward = self.mdp.transition(action) #self.env.step(action)
        return next_state, reward 

    #def close(self):
    #    self.env.close()
