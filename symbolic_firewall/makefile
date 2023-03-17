.PHONY: opttest interp compile layout assemble test optimize_timetest dfg preprocess_dfg
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

SOURCE := stateful_firewall
# SOURCE := $(shell basename "*.dpt" .dpt)

# compile lucid itself
lucid: 
	{ \
	cd ../.. ;\
	make ;\
	cd - ;\
	}

# run p4all optimizer
opttest: 
	python3 $(OPT_DIR)/optimize.py stateful_firewallopt.json --notrafficgen

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py stateful_firewallopt.json --notrafficgen --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py stateful_firewallopt.json --notrafficgen --preprocessingonly

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py stateful_firewallopt.json --timetest --shortcut

# run interpreter
interp: 
	$(LUCID_DIR)/dpt $(SOURCE).dpt --symb $(SOURCE).symb --suppress-final-state

# compile Lucid to P4 without symbolics
ccompile: 
	$(LUCID_DIR)/dptc $(SOURCE)_concrete.dpt ip_harness.p4 linker_config.json $(SOURCE)_build

# compile Lucid to P4 with symbolics
compile: 
	$(LUCID_DIR)/dptc $(SOURCE).dpt build --symb $(SOURCE).symb --silent

#layout
layout:
	$(LUCID_DIR)/bin/layout.py stateful_firewall.dpt --symb stateful_firewall.symb

#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py stateful_firewall.dpt --symb stateful_firewall.symb --dfg

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../$(SOURCE)_asictest.json
