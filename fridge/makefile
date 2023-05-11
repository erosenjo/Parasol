.PHONY: opttest optimize_timetest interp compile layout assemble test layout dfg preprocess_dfg
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py fridgeopt.json --notrafficgen --preprocessingonly

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py fridgeopt.json --notrafficgen --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py fridgeopt.json --notrafficgen --preprocessingonly 

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py fridgeopt.json --timetest --shortcut

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state fridge.dpt 

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc fridge.dpt --symb fridge.symb --silent

layout:
	$(LUCID_DIR)/bin/layout.py fridge.dpt --symb fridge.symb

#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py fridge.dpt --symb fridge.symb --dfg

# compile P4 to tofino
assemble:
	cd lucid_tofino_build; make lucid_tofino_build	 

# test P4 on the asic model
test:
	cd lucid_tofino_build; make test ../fridge_test.json
