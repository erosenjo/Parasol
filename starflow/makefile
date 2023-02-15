.PHONY: opttest interp compile layout assemble test exhaustive optimize_timetest dfg preprocess_dfg
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py starflowoptvars.json --notrafficgen

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py starflowoptvars.json --notrafficgen --nopruning --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py starflowoptvars.json --notrafficgen --nopruning --preprocessingonly --efficient

exhaustive:
	python3 $(OPT_DIR)/optimize.py starflowoptvars.json --nopruning --exhaustive --shortcut --efficient

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py starflowoptvars.json --timetest --nopruning --shortcut --efficient

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state starflow.dpt --spec starflow.json

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc starflow.dpt build --symb starflow.symb --silent

#layout
layout:
	$(LUCID_DIR)/bin/layout.py starflow.dpt --symb starflow.symb

#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py starflow.dpt --symb starflow.symb --dfg

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../starflow_test.json
