import copy
import math

from numpy import loadtxt
from scipy.interpolate import interp1d


i_dc = loadtxt("../Data/i_dc.txt", comments="#", delimiter=",", unpack=False)
v_dc = loadtxt("../Data/v_dc.txt", comments="#", delimiter=",", unpack=False)
i_c = loadtxt("../Data/i_c.txt", comments="#", delimiter=",", unpack=False)
v_c = loadtxt("../Data/v_c.txt", comments="#", delimiter=",", unpack=False)

V_c = interp1d(i_c, v_c)
V_dc = interp1d(i_dc, v_dc)

i_recol = 6 * math.pi / 180
i_decr = 16 * math.pi / 180


class Hysteresis:
    """
    Class that remembers the state of the flow and that computes the velocity for a given angle of attack of the wind
    on the wingsail.
    :ivar float e: state of the flow (0 if attached and 1 if detached)

    """
    def __init__(self):
        self.e = 0

    def copy(self):
        '''

        :return: a deepcopy of the object
        '''
        return copy.deepcopy(self)

    def reset(self):
        '''
        Reset the memory of the flow.
        '''
        self.e = 0

    def calculateSpeed(self, i):
        '''
        Calculate the velocity from angle of attack.

        :param float i: angle of attack
        :return: v - Boat velocity
        :rtype: float

        '''
        if self.e == 0 and i < i_decr:
            v = V_c(i)
            self.i_prev=i
        elif self.e == 0 and i >= i_decr:
            self.e=1
            v=V_dc(i)
            self.i_prev=i
        elif self.e==1 and i>=i_recol:
            self.e=1
            v=V_dc(i)
        elif self.e==1 and i < i_recol:
            self.e=0
            v=V_c(i)
        else:
            raise ValueError("Speed calculation crashed")
        return (v)

