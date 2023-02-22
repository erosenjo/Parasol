i.PHONY: opttest interp compile layout assemble test exhaustive optimize_timetest dfg preprocess_dfg
LUCID_DIR = .
OPT_DIR = optimization_new/

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --notrafficgen

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --notrafficgen --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --notrafficgen --preprocessingonly --efficient


exhaustive:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --exhaustive --shortcut

optimize_timetest:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --shortcut --timetest

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state precision.dpt --spec precision.json

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc precision.dpt build --symb precision.symb --silent 

#layout
layout:
	$(LUCID_DIR)/bin/layout.py precision.dpt --symb precision.symb

#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py precision.dpt --symb precision.symb --dfg

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../precision_test.json
