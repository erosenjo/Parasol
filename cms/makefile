.PHONY: opttest interp compile layout assemble test
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py cmsoptvars.json --notrafficgen

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state cms_sym.dpt

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc cms_sym.dpt build --symb cms_sym.symb --silent

#dfg compile and layout
layout:
	$(LUCID_DIR)/bin/layout.py cms_sym.dpt --symb cms_sym.symb

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../cms_sym.json
