.PHONY: opttest exhaustive optimize_timetest interp compile layout assemble test layout
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py hashopt.json --notrafficgen --nopruning --preprocessingonly

exhaustive:
	python3 $(OPT_DIR)/optimize.py hashopt.json --nopruning --exhaustive --shortcut

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py hashopt.json --timetest --nopruning --shortcut

#dfg compile and layout
layout:
	$(LUCID_DIR)/bin/layout.py hash.dpt --symb hash.symb

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
