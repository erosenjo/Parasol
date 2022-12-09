i.PHONY: opttest interp compile layout assemble test exhaustive optimize_timetest dfg preprocess_dfg
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --notrafficgen

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --notrafficgen --nopruning --preprocessingonly --dfg

exhaustive:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --nopruning --exhaustive --shortcut

optimize_timetest:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --nopruning --shortcut

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
