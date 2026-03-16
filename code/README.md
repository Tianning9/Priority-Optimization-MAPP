This folder contains the Python implementation used for the simulations and experiments described in the MSc thesis:
**Priority Optimization for Sequential Multi-Agent Path Planning**

The code implements a simulation framework for evaluating optimization methods in sequential multi-agent path planning.

---

## Requirements

The code requires Python 3 and the following libraries:

- torch
- gpytorch
- botorch
- pybullet
- numpy
- matplotlib

Install the dependencies using:

pip install torch gpytorch botorch pybullet numpy matplotlib

---

## Running the Code

Run the main script from this directory:

python main.py

This will initialize the simulation environment, generate multi-agent scenarios, run the optimization algorithms, and visualize the results.

## Multiple Scenarios

The project allows different problem instances to be tested.  
You can modify parameters in the code to generate different cases, such as changing:

- the number of agents
- map size
- obstacle configuration
- start and goal positions

The provided scripts include a default configuration that reproduces the experiments used in the thesis.



