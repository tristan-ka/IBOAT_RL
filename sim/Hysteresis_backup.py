import copy
import math

import numpy as np
from numpy import loadtxt
from scipy.interpolate import interp1d

# Data extracted from matlab simulation
WS_ref = 7  # knts
I = loadtxt("../Data/i_py.txt", comments="#", delimiter=",", unpack=False)  # rad
BS = loadtxt("../Data/V_py.txt", comments="#", delimiter=",", unpack=False)  # m/s

i_decr = 0.17802  # 0.1506


class Hysteresis:
    """
    Computes the value of the velocity of the boat from incidence value RWH (i)
    Here we suppose that the sail position is align with the direction of the boat so that the
    RWH is equal to the incidence

    Object that has internal memory of :
        - state flow
        - different incidences :
            - previous incidence
            - incidence of the constant regime
            - incidence corresponding to the flow transition from plateau to reattached
    """

    def __init__(self):

        self.e = 0
        self.iav = 0
        self.i_recol = 0
        self.i_plat = 0

    def copy(self):
        return copy.deepcopy(self)

    def reset(self):
        self.e = 0
        self.iav = 0
        self.i_recol = 0
        self.i_plat = 0

    def calculateSpeed(self, RWH, WS):

        # Check if the wind conditions are corresponding to an upwind sailing operation
        # RWH in rad
        # WS in knts
        if RWH > I[len(I) - 1] or RWH < I[0]:
            print(RWH)
            raise ValueError("The wind conditions are not in the studied range")

        # Scaling from reference table
        BS2 = WS / WS_ref * BS
        idx_stall = np.argmax(BS2)
        V = interp1d(I, BS2)

        #  State of the flow disjonction
        if (self.e == 0 and math.fabs(RWH) <= i_decr):
            VMG = V(RWH)
            self.iav = RWH
        elif (self.e == 0 and math.fabs(RWH) >= i_decr):
            self.e = 1
            VMG = V(RWH)
            self.iav = RWH
        elif (self.e == 1 and math.fabs(RWH) >= math.fabs(self.iav)):
            VMG = V(RWH)
            self.iav = RWH
            self.e = 1
        elif (self.e == 1 and math.fabs(RWH) < math.fabs(self.iav)):
            # Pas d'update de iav
            VMG = V(self.iav)
            if VMG < 0:
                self.i_recol = BS2[0]
            if VMG > BS2[idx_stall] - 0.01:
                self.i_recol = I[idx_stall]
            else:
                f = interp1d(BS2[0:idx_stall], I[0:idx_stall])
                self.i_recol = f(VMG)
            self.i_plat = self.iav
            self.e = 2
        elif (self.e == 2 and self.i_recol < math.fabs(RWH) and math.fabs(RWH) < math.fabs(self.i_plat)):
            # Pas d'update de iav
            VMG = V(self.iav)
        elif (self.e == 2 and math.fabs(RWH) >= math.fabs(self.i_plat)):
            self.e = 1
            VMG = V(RWH)
            self.iav = RWH
        elif (self.e == 2 and math.fabs(RWH) <= math.fabs(self.i_recol)):
            self.e = 0
            VMG = V(RWH)
            self.iav = RWH
        else:
            raise ValueError("Speed calculation did not work")

        return VMG
