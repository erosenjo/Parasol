.PHONY: opttest exhaustive optimize_timetest interp compile layout assemble test dfg preprocess_dfg
LUCID_DIR = .
OPT_DIR = optimization_new/

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py cmsoptvars.json --notrafficgen --shortcut

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py cmsoptvars.json --notrafficgen --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py cmsoptvars.json --notrafficgen --preprocessingonly --efficient

exhaustive:
	python3 $(OPT_DIR)/optimize.py cmsoptvars.json --exhaustive --shortcut

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py cmsoptvars.json --timetest --shortcut

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state cms_sym.dpt

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc cms_sym.dpt build --symb cms_sym.symb 

#layout
layout:
	$(LUCID_DIR)/bin/layout.py cms_sym.dpt --symb cms_sym.symb
#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py cms_sym.dpt --symb cms_sym.symb --dfg


# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../cms_sym.json
