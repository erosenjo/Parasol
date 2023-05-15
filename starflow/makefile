.PHONY: opttest interp compile layout assemble test optimize_timetest dfg preprocess_dfg
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py starflowopt.json --notrafficgen

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py starflowopt.json --notrafficgen --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py starflowopt.json --notrafficgen --preprocessingonly

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py starflowopt.json --timetest --shortcut

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state starflow.dpt 

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc starflow.dpt --symb starflow.symb --silent

#layout
layout:
	$(LUCID_DIR)/bin/layout.py starflow.dpt --symb starflow.symb

#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py starflow.dpt --symb starflow.symb --dfg

# compile P4 to tofino
assemble:
	cd lucid_tofino_build; make lucid_tofino_build	 

# test P4 on the asic model
test:
	cd lucid_tofino_build; make test ../starflow_test.json
