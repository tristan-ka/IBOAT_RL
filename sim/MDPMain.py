import math

import matplotlib.pyplot as plt
import mdp
import numpy as np

# %%
TORAD = math.pi / 180

'''
MDP PARAMETERS
'''
history_duration = 3
mdp_step = 1
time_step = 0.1
SP = -40 * TORAD
mdp = mdp.MDP(history_duration, mdp_step, time_step)

'''
WIND CONDITIONS
'''
mean = 45 * TORAD
std = 0 * TORAD
wind_samples = 10
WH = np.random.uniform(mean - std, mean + std, size=10)

'''
MDP INIT
'''
hdg0 = 0 * np.ones(10)
state = mdp.initializeMDP(hdg0, WH)

'''
Generation of a simulation
'''

SIMULATION_TIME = 100

i = np.ones(0)
vmg = np.ones(0)
wind_heading = np.ones(0)


for time in range(SIMULATION_TIME):
    print('t = {0} s'.format(time))
    action = 0
    WH = np.random.uniform(mean - std, mean + std, size=10)
    if time < SIMULATION_TIME / 2:
        action = 0
    else:
        action = 1
    nex_state, reward = mdp.transition(action, WH)
    next_state = state
    i = np.concatenate([i, mdp.extractSimulationData()[0, :]])
    vmg = np.concatenate([vmg, mdp.extractSimulationData()[1, :]])
    wind_heading = np.concatenate([wind_heading, WH])

time_vec = np.linspace(0, SIMULATION_TIME, int((SIMULATION_TIME) / time_step))
hdg = i - wind_heading - SP

v = vmg / np.cos(0 - hdg)

# %%
f, axarr = plt.subplots(3, sharex=True)
vmg_plot = axarr[0].plot(time_vec, vmg, label="VMG")
v_plot = axarr[0].plot(time_vec, v, label="v")
axarr[0].legend()
axarr[0].set_ylabel("v [m/s]")

i_plot = axarr[1].scatter(time_vec, i / TORAD, label='i [°]')
axarr[1].set_ylabel("i [°]")
hdg_plot = axarr[2].scatter(time_vec, hdg / TORAD, label='Hdg [°]')
axarr[2].set_ylabel("Hdg [°]")
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

axarr[2].grid(True)
gridlines = axarr[2].get_xgridlines() + axarr[2].get_ygridlines()
for line in gridlines:
    line.set_linestyle('-.')
