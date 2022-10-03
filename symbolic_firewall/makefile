.PHONY: interp compile assemble test
LUCID_DIR = /media/data/mh43/lucid

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
optimize: lucid
	python3 ../optimization/optimize.py stateful_firewallopt.json

# run interpreter
interp: 
	$(LUCID_DIR)/dpt $(SOURCE).dpt --symb $(SOURCE).symb --suppress-final-state

# compile Lucid to P4 without symbolics
ccompile: 
	$(LUCID_DIR)/dptc $(SOURCE)_concrete.dpt ip_harness.p4 linker_config.json $(SOURCE)_build

# compile Lucid to P4 with symbolics
compile: 
	$(LUCID_DIR)/dptc $(SOURCE).dpt build --symb $(SOURCE).symb --silent

# compile P4 to tofino
assemble:
	cd $(SOURCE)_build; make build	 

# test P4 on the asic model
test:
	cd $(SOURCE)_build; make test ../$(SOURCE)_asictest.json
