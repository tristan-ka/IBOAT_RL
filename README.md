<h1 align="center">
  Welcome to IBOAT RL's project
  <br>
  <a href="http://www.amitmerchant.com/electron-markdownify"><img src="img/IBOAT_logo.png" alt="Markdownify" width="200"></a>
  <br>
</h1>

<h1 align="center">
  <a href="PyPI - Python Version">
    <img src="https://img.shields.io/pypi/pyversions/Django.svg">
  </a>
  <a href="https://tristan-ka.github.io/IBOAT_RL/">
    <img src="https://readthedocs.org/projects/ansicolortags/badge/?version=latest)">
  </a>
  <a href="https://www.isae-supaero.fr/fr/">
    <img src="https://img.shields.io/badge/university-ISAE-blue.svg">
  </a>
</h1>

<h4 align="center">Controling autonomous boat with RL </h4>



<p align="center">
  <a href="#a-brief-context">Context</a> •
  <a href="#prerequisites-and-code-documentation">Documentation</a> •
  <a href="#usage">Usage</a> •
  <a href="#getting-started">Getting started</a> •
  <a href="#tests">Tests</a> •
  <a href="#built-with">Tools</a> •
  <a href="#related">Related</a> •
  <a href="#acknowledgments">Acknowledgments</a>
</p>


## A brief context

This project presents **Reinforcement Learning** as a solution to control systems with a **large hysteresis**. We consider an
autonomous sailing robot (IBOAT) which sails upwind. In this configuration, the wingsail is almost aligned with the upcoming wind. It thus operates like
a classical wing to push the boat forward. If the angle of attack of the wind coming on the wingsail is too great, the flow around the wing detaches leading to
a **marked decrease of the boat's speed**.

Hysteresis such as stall are hard to model. We therefore propose an **end-to-end controller** which learns the stall behavior and
builds a policy that avoids it. Learning is performed on a simplified transition model representing the stochastic environment and the dynamic of the boat.

Learning is performed on two types of simulators. A **proof of concept** is first carried out on a simplified simulator of the boat coded in Python. The second phase of the project consist of trying to control a **more realisitic**  model of the boat. For this purpose we use a dynamic library which is derived using the Code Generation tools in Simulink. The executable C are then feeded to Python using the "ctypes" library.

## Prerequisites and code documentation


The documentation as well as the prerequisites can be found on the following webpage :

[![Documentation Status](https://readthedocs.org/projects/ansicolortags/badge/?version=latest)](https://tristan-ka.github.io/IBOAT_RL/)


## Usage

This repositroy is intended to be a **source of information** for future work on end-to-end control of system with large hysteresis. It provides a solid base to dig further on this topic. The tools that are provided are the following :

- A realistic and fast simulator implemented in C++.
- Two reinforcement learning algorithms which have been tested on a simplified simulator.
- A fully integrated environment to play with these tools.

## Getting started


The notebook [demonstrator.ipynb](demonstrator.ipynb) contains different scripts to help users getting familiar with the different scripts. One simply has to have installed ipython or Jupyter notebook (http://jupyter.readthedocs.io/en/latest/install.html) and type the following command in a shell :
```
jupyter notebook
```
And open the file.

# Tests

In this repo, all the files finishing by Main.py are test files. 

- In package sim, the following files can be run to generate simulations and understand how they work:
    * [SimulationMain.py](SimulationMain.py) - generate a simulation using the simplified simulator.
    * [MDPMain.py](MDPMain.py) - generate trajectories via mdp transitions and using the simplified simulator.
    * [Realistic_MDPMain.py](Realistic_MDPMain.py) - generate trajectories via mdp transitions and using the realistic      simulator.
  
- In package RL, the following files can be run to train models:
    * [policyLearningMain.py](policyLearningMain.py) - train a network to learn the Q-values of a policy.
    * [dqnMain.py](dqnMain.py) - find the optimal policy to control the Iboat using the DQN algorithm (discret set of actions).
    * [DDPGMain.py](DDPGMain.py) - find the optimal policy to control the Iboat using the DDPG algorithm (continuous set of actions).


## Built With

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg)](https://www.python.org/)
[![made-with-sphinx-doc](https://img.shields.io/badge/Made%20with-Sphinx-1f425f.svg)](https://www.sphinx-doc.org/)

## Related 

This project falls under the IBOAT project and is related to the repo: 

[![github](https://img.shields.io/website-up-down-green-red/https/naereen.github.io.svg)](https://github.com/PBarde/IBoatPIE)

It tackles the problem of long-term path planning under uncertainty for offshore sailing using parallel MCTS.

## Acknowledgments

This project has been carried out with the help of:

* [Yves Brière](https://personnel.isae-supaero.fr/yves-briere/) - Professor of automatics at ISAE-Supaero.
* [Emmanuel Rachelson](https://github.com/erachelson) - Professor in reinforcement learning at ISAE-Supaero.
* [Valentin Guillet](https://github.com/Val95240/RL-Agents) - ISAE-Supaero student who has implemented various RL algorithms.

## Authors

* **Tristan Karch** - *Initial work* - Implementation of simplified simulator and proof of concept with Deep Q-Learning algorithm. Also responsible for the documentation management.
* **Nicolas Megel** - Implementation of DDPG algorithm and responsible for the project management.
* **Albert Bonet** - Simulink expert responsible for the realisitic simulator implementation and compilation.
