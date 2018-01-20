import math

from Hysteresis import Hysteresis
import matplotlib.pyplot as plt
import numpy as np

# Conversion Constants
TORAD = math.pi / 180
TODEG = 180 / math.pi
TOMPS = 0.514444
TOKNTS = 1 / TOMPS

IMAX = 0.51436 - 0 * TORAD
IMIN = 0 * TORAD

BSINIT = 0.0069354000000000004
BSFIN = 1.8741000000000001
BSSTALL = 1.5

BSTRESH = 2.4


class Simulator:
    """
    Simulator object : It simulates a simplified dynamic of the boat with the config.

    :ivar float time_step: time step of the simulator, corresponds to the frequency of data acquisition.
    :ivar float size: size of the simulation.
    :ivar int delay: delay between the heading command and its activation.
    :ivar float sail_pos: position of the windsail [rad].
    :ivar float hdg_target: target heading towards which we want the boat to sail.
    :ivar np.array() hdg: array of size **size** that stores the heading of the boat [rad].
    :ivar np.array() vmg: array of size **size** that stores the velocity made good.
    :ivar Hysteresis hyst: Memory of the flow state during the simulations.
    :raise ValueError: if the size of the simulation is zero or less.
    """

    def __init__(self, duration, time_step):
        if duration <= 0:
            raise ValueError("Simulation duration must be greater than 0")

        self.time_step = time_step
        self.size = int(duration / time_step)
        self.delay = 5
        self.sail_pos = -40 * TORAD
        self.hdg_target = 0 * TORAD
        self.hdg = np.zeros(self.size)
        self.vmg = np.zeros(self.size)
        self.hyst = Hysteresis()

    def getLength(self):
        return self.size

    def getTimeStep(self):
        return self.time_step

    def getHdg(self, k):
        return self.hdg[k]

    def updateHdg(self, k, inc):
        self.hdg[k] = inc

    def incrementHdg(self, k, delta_hdg):
        if k > 0:
            self.hdg[k] = self.hdg[k - 1] + delta_hdg
        else:
            self.hdg[k] = self.hdg[k]

    def updateVMG(self, k, vmg):
        if k > self.size - 1:
            raise ValueError("Index out of bounds")
        self.vmg[k] = vmg

    def computeNewValues(self, delta_hdg, WH):
        Hdg_tmp = self.hdg[len(self.hdg) - 1]

        self.hdg[0:self.delay] = Hdg_tmp

        saturationMin = False
        saturationMax =False

        for jj in range(self.size):
            RWH = self.hdg[jj] + WH[jj] + self.sail_pos
            self.incrementDelayHdg(jj, delta_hdg)

            # Saturation
            if RWH > IMAX:
                saturationMax = True
            elif RWH < IMIN:
                saturationMin = True
            else:
                self.vmg[jj] = self.hyst.calculateSpeed(RWH) * math.cos(self.hdg_target - self.hdg[jj])

        if saturationMin == True:
            for jj in range(self.size):
                self.hdg[jj] = Hdg_tmp
                self.vmg[jj] = BSINIT
        if saturationMax == True :
            for jj in range(self.size):
                self.hdg[jj] = Hdg_tmp
                self.vmg[jj] = BSSTALL

        return self.hdg, self.vmg

    def incrementDelayHdg(self, k, delta_hdg):
        # Saturation
        if (k + self.delay < self.size):
            self.hdg[k + self.delay] = self.hdg[k] + delta_hdg
        elif k >= self.size - self.delay and k < self.size:
            self.hdg[k] = self.hdg[k]

    def plot(self):
        time = np.linspace(0, self.size * self.time_step, self.size)
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].scatter(time, self.vmg)
        axarr[1].scatter(time, self.hdg * TODEG)

        for k in range(2):
            axarr[k].grid(True)
            axarr[k].set_xlabel('t [s]')
        axarr[0].set_ylabel('VMG [m/s]')
        axarr[1].set_ylabel('Heading [°]')

class RealistSimulator:
    """
    Simulator object : It simulates a simplified dynamic of the boat with the config.

    :ivar float time_step: time step of the simulator, corresponds to the frequency of data acquisition.
    :ivar float size: size of the simulation.
    :ivar int delay: delay between the heading command and its activation.
    :ivar float sail_pos: position of the windsail [rad].
    :ivar float hdg_target: target heading towards which we want the boat to sail.
    :ivar np.array() hdg: array of size **size** that stores the heading of the boat [rad].
    :ivar np.array() vmg: array of size **size** that stores the velocity made good.
    :ivar Hysteresis hyst: Memory of the flow state during the simulations.
    :raise ValueError: if the size of the simulation is zero or less.
    """

    def __init__(self, duration, time_step):
        if duration <= 0:
            raise ValueError("Simulation duration must be greater than 0")

        self.time_step = time_step
        self.size = int(duration / time_step)
        self.delay = 5
        self.sail_pos = -40 * TORAD
        self.hdg_target = 0 * TORAD
        self.hdg = np.zeros(self.size)
        self.vmg = np.zeros(self.size)
        self.hyst = Hysteresis()

    def getLength(self):
        return self.size

    def getTimeStep(self):
        return self.time_step

    def getHdg(self, k):
        return self.hdg[k]

    def updateHdg(self, k, inc):
        self.hdg[k] = inc

    def incrementHdg(self, k, delta_hdg):
        if k > 0:
            self.hdg[k] = self.hdg[k - 1] + delta_hdg
        else:
            self.hdg[k] = self.hdg[k]

    def updateVMG(self, k, vmg):
        if k > self.size - 1:
            raise ValueError("Index out of bounds")
        self.vmg[k] = vmg

    def computeNewValues(self, delta_hdg, WH):
        Hdg_tmp = self.hdg[len(self.hdg) - 1]

        self.hdg[0:self.delay] = Hdg_tmp

        saturation = False

        for jj in range(self.size):
            RWH = self.hdg[jj] + WH[jj] + self.sail_pos
            self.incrementDelayHdg(jj, delta_hdg)

            # Saturation
            if RWH > IMAX:
                saturation = True
            elif RWH < IMIN:
                saturation = True
            else:
                self.vmg[jj] = self.hyst.calculateSpeed(RWH) * math.cos(self.hdg_target - self.hdg[jj])
        if saturation == True:
            for jj in range(self.size):
                self.hdg[jj] = Hdg_tmp
                self.vmg[jj] = self.hyst.calculateSpeed(Hdg_tmp + WH[jj] + self.sail_pos) * math.cos(
                    self.hdg_target - self.hdg[jj])

        return self.hdg, self.vmg

    def incrementDelayHdg(self, k, delta_hdg):
        # Saturation
        if (k + self.delay < self.size):
            self.hdg[k + self.delay] = self.hdg[k] + delta_hdg
        elif k >= self.size - self.delay and k < self.size:
            self.hdg[k] = self.hdg[k]

    def plot(self):
        time = np.linspace(0, self.size * self.time_step, self.size)
        f, axarr = plt.subplots(2, sharex=True)
        axarr[0].scatter(time, self.vmg)
        axarr[1].scatter(time, self.hdg * TODEG)

        for k in range(2):
            axarr[k].grid(True)
            axarr[k].set_xlabel('t [s]')
        axarr[0].set_ylabel('VMG [m/s]')
        axarr[1].set_ylabel('Heading [°]')
