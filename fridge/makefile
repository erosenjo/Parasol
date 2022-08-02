.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt --suppress-final-state fridge.dpt --spec fridge.json

# compile Lucid to P4
compile: 
	../../dptc fridge.dpt ip_harness.p4 linker_config.json fridge_build --symb fridge.symb 

# compile P4 to tofino
assemble:
	cd fridge_build; make build	 

# test P4 on the asic model
test:
	cd fridge_build; make test ../fridge_test.json
