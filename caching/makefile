.PHONY: opttest interp compile layout assemble test
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py cacheopt.json --notrafficgen --preprocessingonly

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py cacheopt.json --timetest

exhaustive:
	python3 $(OPT_DIR)/optimize.py cacheopt.json --nopruning --exhaustive --shortcut

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state caching.dpt

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc caching.dpt build --symb caching.symb --silent

#dfg compile and layout
layout:
	$(LUCID_DIR)/bin/layout.py caching.dpt --symb caching.symb

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../cmscache.json
