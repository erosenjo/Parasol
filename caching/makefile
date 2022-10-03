.PHONY: interp compile assemble test
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_noilp

# run p4all optimizer
optimize:
	python3 $(OPT_DIR)/optimize.py cacheopt_full.json

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py cacheopt_full.json --timetest

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state caching.dpt

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc caching.dpt build --symb caching.symb --silent

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../cmscache.json
