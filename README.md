# Parasol
Parasol is framework for writing generalized, parameterized network algorithms and automatically optimizing those parameters.

## Setup
Parasol is built on top of [Lucid](https://github.com/PrincetonUniversity/lucid). Instructions in installing Lucid can be found in the linked repo.

## Writing a Parasol Program
### Lucid Code and Measurement Functions
A Parasol program is a Lucid file with symbolic values describing parameters and extern functions (defined in a Python file) that record measurements during simulation. For each of our examples, we provide a ``*.dpt`` file containing the Lucid code and a ``*.py`` file of the same name containing the measurement functions.

### Objective Function
The objective function is contained in a Python class that should contain the following functions: ``gen_traffic`` (optionally creates a json file for simulation from a given pcap), ``init_iteration`` (includes an necessary setup, called before each simulation), ``calc_cost`` (the objective function, takes measurements as input and returns a numeric score). We provide objective functions for each example - ``*opt.py``.

### Optimization Input
The Parasol optimizer takes as input a json file specifying different optimization parameters. In particular, it details the symbolic values in the program and their bounds, the optimization strategy (``bayesian``, ``simannealing``, ``neldermead``, ``exhaustive``), whether to preprocess (``"optalgo": preprocess"``) or not (``"optalgo"`` is the same as the optimization strategy), the time budget (``stop_time``), the file containing measurements taken during simulation (``outputfiles``), and various parameters for optimization strategies. We provide default values for each of these, but users are free to specify their own. This repo contains examples of input json files for each example (``*opt.json``).

## Running the Optimization
### Preprocessing only
To only preprocess (without optimizing after) and generate a file containing preprocessed solutions, run ``python3 $(OPT_DIR)/optimize.py *optvars.json --notrafficgen --preprocessingonly``

### Optimization
To run optimization after producing a preprocessed solution file, run ``python3 $(OPT_DIR)/optimize.py *optvars.json --timetest --shortcut``. To preprocess and run optimization, remove the ``--shortcut`` flag. We provide makefiles for each example that include these commands.


## Trace Optimization
This version of Parasol optimizes with different traces, while keep symbolic parameters constant. 


