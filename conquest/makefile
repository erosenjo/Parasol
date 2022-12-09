.PHONY: opttest interp compile layout assemble test preprocess_dfg exhaustive optimize_timetest dfg
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py conquestopt.json --notrafficgen --shortcut --nopruning

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py conquestopt.json --notrafficgen --nopruning --preprocessingonly --dfg

exhaustive:
	python3 $(OPT_DIR)/optimize.py conquestopt.json --nopruning --exhaustive --shortcut

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py conquestopt.json --timetest --nopruning --shortcut

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state conquest.dpt

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc conquest.dpt build --symb conquest.symb --silent

#layout
layout:
	$(LUCID_DIR)/bin/layout.py conquest.dpt --symb conquest.symb
#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py conquest.dpt --symb conquest.symb --dfg

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../conquest_test.json
