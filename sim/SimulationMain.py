import math

import numpy as np
from Hysteresis import Hysteresis

from sim import Simulator

# Conversion Constants
TORAD = math.pi / 180
TODEG = 180 / math.pi
TOMPS = 0.514444
TOKNTS = 1 / TOMPS
IMAX = 0.479970000000000-0.1001
IMIN = 0+0.1001

# Test of the simulator object

# Initalisation :
delta_t=0.1
t_history=10
simulation = Simulator.Simulator(20, delta_t)


# Note that we create one single speed calculator that we use to generate different experiences because it
# saves the state of the flow and the previous incidences
# Be carefull to generate experiences in a chronological way !!!
speedCalculator = Hysteresis()

taustep = 2
# be carefull with initialisation : due to delay we must initialize the taustep+1 first angles
tau = taustep * simulation.time_step;
simulation.hdg[0:taustep + 1] = 5 * TORAD

# Wind Generation
WS = 7
mean = 0
std = 0.1 * TORAD  # 2.5*TORAD
wind_samples = simulation.getLength()
WH = np.random.normal(mean, std, size=wind_samples)
RWH = np.zeros(simulation.getLength())
print(simulation.time_step*simulation.size)
# SIMULATION
for i in range(simulation.getLength()):
    if i < simulation.getLength() / 2-1:
        simulation.incrementDelayHdg(i, taustep, 1 * TORAD)
    else:
        simulation.incrementDelayHdg(i, taustep, -0.8 * TORAD)

    RWH[i] = simulation.getHdg(i) + WH[i]
    simulation.updateVMG(i, speedCalculator.calculateSpeed(RWH[i], WS))

simulation.plot()

# f, axarr2 = plt.subplots(2, sharex=True)
# axarr2[0].scatter(simulation.time, WH)
# axarr2[1].scatter(simulation.time, RWH)
