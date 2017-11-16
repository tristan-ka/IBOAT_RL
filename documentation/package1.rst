Package Sim
================

This package contains all the classes required to build a simulation for the learning.

The simulator uses

Simulator
------------------------------

.. automodule:: Simulator
	:members:
	:undoc-members:
	:show-inheritance:

.. note:: The configuration of the boat is illustrated in the following figure:


.. image:: scheme.png
   :width: 200px
   :height: 180px
   :scale: 200 %
.. image:: block_diagrams.png
   :width: 200px
   :height: 100px
   :scale: 200 %

.. warning::
   Be careful, the delay is expressed has an offset of index. the delay in s is equal to delay*time_step



Hysteresis
------------------------------

.. automodule:: hysteresis
	:members:
	:undoc-members:

Markov Decision Process (MDP)
------------------------------

.. automodule:: mdp
	:members:
	:undoc-members:

.. note:: The class variable simulation_duration defines the frequency of action taking. The reward is the average of the new velocities computed after each transition.