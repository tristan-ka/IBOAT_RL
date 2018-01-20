.. IBOAT RL documentation master file, created by
   sphinx-quickstart on Sat Nov 11 18:59:10 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.


Welcome to IBOAT RL's documentation!
====================================

A brief context
-----------------
This project presents **Reinforcement Learning** as a solution to control systems with a **large hysteresis**. We consider an
autonomous sailing robot (IBOAT) which sails upwind. In this configuration, the wingsail is almost aligned with the upcoming wind. It thus operates like
a classical wing to push the boat forward. If the angle of attack of the wind coming on the wingsail is too great, the flow around the wing detaches leading to
a **marked decrease of the boat's speed**.

Hysteresis such as stall are hard to model. We therefore proposes an **end-to-end controller** which learns the stall behavior and
builds a policy that avoids it. Learning is performed on a simplified transition model representing the stochastic environment and the dynamic of the boat.

On this page, you will find the documentation of the simplified simulator of the boat as well as the documentation of the reinforcement learning tools. Each package
contains tutorials to better understand how the code can be used


Requirements
---------------

The project depends on the following extensions :

1. NumPy for the data structures (http://www.numpy.org)
2. Matplotlib for the visualisation (https://matplotlib.org)
3. Keras for the convolutional neural network models (https://keras.io)

|pic1| |pic2| |pic3|

.. |pic1| image:: numpy.jpeg
   :width: 200px
   :height: 70px
   :scale: 50 %
   :align: top

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




Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
