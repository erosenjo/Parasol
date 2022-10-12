.PHONY: opttest interp compile layout assemble test
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py precisionopt.json --notrafficgen

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state precision.dpt --spec precision.json

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc precision.dpt build --symb precision.symb --silent 

#dfg compile and layout
layout:
	$(LUCID_DIR)/bin/layout.py precision.dpt --symb precision.symb

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../precision_test.json
