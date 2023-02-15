.PHONY: opttest exhaustive optimize_timetest interp compile layout assemble test layout dfg preprocess_dfg
LUCID_DIR = .
OPT_DIR = optimization_new/

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py fridgeopt.json --notrafficgen --nopruning --preprocessingonly

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py fridgeopt.json --notrafficgen --nopruning --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py fridgeopt.json --notrafficgen --nopruning --preprocessingonly --efficient


exhaustive:
	python3 $(OPT_DIR)/optimize.py fridgeopt.json --nopruning --exhaustive --shortcut

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py fridgeopt.json --timetest --nopruning --shortcut

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state fridge.dpt --spec fridge.json

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc fridge.dpt build --symb fridge.symb --silent

layout:
	$(LUCID_DIR)/bin/layout.py fridge.dpt --symb fridge.symb

#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py fridge.dpt --symb fridge.symb --dfg

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../fridge_test.json
