Welcome to IBOAT RL's documentation!
====================================

A brief context
---------------
This project presents **Reinforcement Learning** as a solution to control systems with a **large hysteresis**. We consider an
autonomous sailing robot (IBOAT) which sails upwind. In this configuration, the wingsail is almost aligned with the upcoming wind. It thus operates like
a classical wing to push the boat forward. If the angle of attack of the wind coming on the wingsail is too great, the flow around the wing detaches leading to
a **marked decrease of the boat's speed**.

Hysteresis such as stall are hard to model. We therefore proposes an **end-to-end controller** which learns the stall behavior and
builds a policy that avoids it. Learning is performed on a simplified transition model representing the stochastic environment and the dynamic of the boat.

Learning is performed on two types of simulators, A **proof of concept** is first carried out on a simplified simulator of the boat coded in Python. The second phase of the project consist of trying to control a **more realisitic**  model of the boat. For this purpose we use a dynamic library which is derived using the Code Generation tools in Simulink. The executable C are then feeded to Python using the "ctypes" library.

Prerequisites and code documentation
------------------------------------

This project is using Python 3.*. The documentation as well as the prerequisites can be found here:

https://tristan-ka.github.io/IBOAT_RL/


Usage
-----

This repositroy is intended to be a **source of information** for future work on end-to-end control of system with large hysteresis. It provides a solid base to dig further on this topic. The tools that are provided are the following :

- A realistic and fast simulator implemented in C++.
- Two reinforcement learning algorithms which have been tested on a simplified simulator
- A fully integrated environment to play with these tools

Built With
----------



Acknowledgments
---------------

