.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt --suppress-final-state precision.dpt --spec precision.json

# compile Lucid to P4
compile: 
	../../dptc precision.dpt ip_harness.p4 linker_config.json precision_build --symb precision.symb 

# compile P4 to tofino
assemble:
	cd precision_build; make build	 

# test P4 on the asic model
test:
	cd precision_build; make test ../precision_test.json
