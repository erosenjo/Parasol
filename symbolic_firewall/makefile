.PHONY: interp compile assemble test

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
interp: lucid
	../../lucid/dpt $(SOURCE).dpt --symb $(SOURCE).symb

# compile Lucid to P4 without symbolics
ccompile: lucid
	../../lucid/dptc $(SOURCE)_concrete.dpt ip_harness.p4 linker_config.json $(SOURCE)_build

# compile Lucid to P4 with symbolics
compile: lucid
	../../dptc $(SOURCE).dpt $(SOURCE)_build --symb $(SOURCE).symb --silent

# compile P4 to tofino
assemble:
	cd $(SOURCE)_build; make build	 

# test P4 on the asic model
test:
	cd $(SOURCE)_build; make test ../$(SOURCE)_asictest.json
