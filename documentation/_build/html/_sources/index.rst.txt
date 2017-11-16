.. IBOAT RL documentation master file, created by
   sphinx-quickstart on Sat Nov 11 18:59:10 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to IBOAT RL's documentation!
====================================

This is my introduction to this project

Requirements
---------------

The project depends on the following extensions :

1. NumPy for the data structures (http://www.numpy.org) |pic1|
2. Matplotlib for the visualisation (https://matplotlib.org) |pic2|
3. Keras for the convolutional neural network models (https://keras.io) |pic3|

.. |pic1| image:: numpy.jpeg
   :width: 200px
   :height: 70px
   :scale: 50 %

.. |pic2| image:: matplotlib.jpeg
   :width: 200px
   :height: 70px
   :scale: 50 %

.. |pic3| image:: keras.png
   :width: 200px
   :height: 60px
   :scale: 50 %

Contents
---------------
.. toctree::
   :maxdepth: 2

   Sim <package1.rst>

   RL <package2.rst>


Tutorial
---------------

To visualize how a simulation can be generated we provide a file MDPmain.py that creates a simulation where the heading is first increase and then decrease.

.. code-block:: python
   :emphasize-lines: 13

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
       next_state, reward = mdp.transition(action, WH)
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
   :scale: 200 %
   :alt: alternate text
   :align: center


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
