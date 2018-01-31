Package Sim
================

This package contains all the classes required to build a simulation for the learning. In this small paragraph, the physic of the simulator is described
so that the reader can better understand the implementation.

We need the boat to be in a configuration when it sails upwind so that the flow around the sail is attached and the sail works as a wing.
To generate the configuration we first assume that the boat as a target heading `hdg_target` = 0. The boat as a certain heading `hdg` with respect to the north
and faces an upcoming wind of heading `WH`. To lower the number of parameters at stake we consider that the wind has a **constant speed** of 15 knts.
The sail is oriented with respect to the boat heading with an angle `sail_pos` = -40°. The angle of attack of the wind on the sail is therefore equal to
`i = hdg + WH + sail_pos`. This angle equation can be well understood thanks to the following image.


.. image:: scheme.png
   :width: 200px
   :height: 180px
   :scale: 200 %
   :align: center

The action taken to change the angle of attack are changes of boat heading `delta_hdg`. We therefore assume that `sail_pos` is constant and equal to -40°.
The wind heading is fixed to `WH` = 45°. Finally, there is a delay between the command and the change of heading of τ = 0.5 seconds. The simulator can be
represented with the following block diagram. It contains a delay and an hysteresis block that are variables of the simulator class.

.. image:: block_diagrams.png
   :width: 200px
   :height: 100px
   :scale: 200 %
   :align: center


Simulator
------------------------------


.. automodule:: Simulator
	:members:
	:undoc-members:
	:show-inheritance:



.. warning::
   Be careful, the delay is expressed has an offset of index. the delay in s is equal to delay*time_step



Hysteresis
------------------------------

.. automodule:: Hysteresis
	:members:
	:undoc-members:


Environment
------------------------------

.. automodule:: environment
	:members:
	:undoc-members:



Markov Decision Process (MDP)
------------------------------

.. automodule:: mdp
	:members:
	:undoc-members:

.. note:: The class variable simulation_duration defines the frequency of action taking. The reward is the average of the new velocities computed after each transition.

Tutorial
---------------

To visualize how a simulation can be generated we provide a file MDPmain.py that creates a simulation where the heading is first increase and then decrease.

.. code-block:: python
   :emphasize-lines: 36

   TORAD = math.pi / 180

   history_duration = 3
   mdp_step = 1
   time_step = 0.1
   SP = -40 * TORAD
   mdp = mdp.MDP(history_duration, mdp_step, time_step)

   mean = 45 * TORAD
   std = 0 * TORAD
   wind_samples = 10
   WH = np.random.uniform(mean - std, mean + std, size=10)

   hdg0 = 0 * TORAD * np.ones(wind_samples)
   state = mdp.initializeMDP(hdg0, WH)

   SIMULATION_TIME = 100

   i = np.ones(0)
   vmg = np.ones(0)
   wind_heading = np.ones(0)

   for time in range(SIMULATION_TIME):
       print('t = {0} s'.format(time))
       action = 0
       WH = np.random.uniform(mean - std, mean + std, size=wind_samples)
       if time < SIMULATION_TIME / 4:
           action = 0
       elif time < SIMULATION_TIME / 2:
           action = 1
       elif time < 3 * SIMULATION_TIME / 4:
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


This results in the following value for the velocity, angle of attack and heading.

.. image:: Figure_1.png
   :width: 200px
   :height: 200px
   :scale: 300 %
   :alt: alternate text
   :align: center
