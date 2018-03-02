Package Realistic Simulator
=============================

This package contains all the classes required to build a realistic simulation of the boat. It uses the dynamic library :file:`libBoatModel.so` which implements the accurate dynamic of the boat. It is coded in C in order to speed up the calculation process and hence the learning.

Realisitic Simulator
---------------------

.. automodule:: Realistic_Simulator
    :members:
    :show-inheritance:

.. warning::
    Be careful to use the libBoatModel library corresponding to your OS (Linux or Mac).

If the dynamic library is not correctly compiled one can recompile it using the source file in the folder /libs/SourceLibs thanks to the command :

.. code-block:: python

    >> gcc -shared -o libBoatModel.so -fPIC Model_2_EXPORT_Discrete.c Model_2_EXPORT_Discrete_data.c rt_look.c rt_look1d.c

For further assistance please read :download:`this file <SIMULINK_TO_C__PYTHON.pdf>`.

Realistic MDP
---------------

.. automodule:: mdp_realistic
    :members:
    :undoc-members:
    :show-inheritance:
