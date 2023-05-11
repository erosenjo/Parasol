i.PHONY: opttest interp compile layout assemble test optimize_timetest dfg preprocess_dfg
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --notrafficgen

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --notrafficgen --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --notrafficgen --preprocessingonly

optimize_timetest:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --shortcut --timetest

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state precision.dpt 

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc precision.dpt --symb precision.symb --silent 

#layout
layout:
	$(LUCID_DIR)/bin/layout.py precision.dpt --symb precision.symb

#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py precision.dpt --symb precision.symb --dfg

# compile P4 to tofino
assemble:
	cd lucid_tofino_build; make lucid_tofino_build	 

# test P4 on the asic model
test:
	cd lucid_tofino_build; make test ../precision_test.json
