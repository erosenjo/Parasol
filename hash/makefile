.PHONY: opttest exhaustive optimize_timetest interp compile layout assemble test layout dfg preprocess_dfg
LUCID_DIR = .
OPT_DIR = optimization_new/

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py hashopt.json --notrafficgen --nopruning --preprocessingonly

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py hashopt.json --notrafficgen --nopruning --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py hashopt.json --notrafficgen --nopruning --preprocessingonly --efficient


exhaustive:
	python3 $(OPT_DIR)/optimize.py hashopt.json --nopruning --exhaustive --shortcut

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py hashopt.json --timetest --nopruning --shortcut

optimize_notimetest:
	python3 $(OPT_DIR)/optimize.py hashopt.json

#layout
layout:
	$(LUCID_DIR)/bin/layout.py hash.dpt --symb hash.symb

#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py hash.dpt --symb hash.symb --dfg

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state hash.dpt 

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc hash.dpt build --symb hash.symb --silent 

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../hashmodule_test.json
