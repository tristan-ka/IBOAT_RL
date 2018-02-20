Package Realistic Simulator
=============================

This package contains all the classes required to build a realistic simulation of the boat. It uses the dynamic library :file:`libBoatModel.so` which implements the accurate dynamic of the boat. It is coded in C++ in order to speed up the calculation process and hence the learning.

Realisitic Simulator
---------------------

.. automodule:: Simulator_realistic
    :members:
    :undoc-members:
    :show-inheritance:

.. warning::
    Be careful to use the libBoatModel library corresponding to your OS (Linux or Mac)

Realistic MDP
---------------

.. automodule:: mdp_realistic
    :members:
    :undoc-members:
    :show-inheritance:
