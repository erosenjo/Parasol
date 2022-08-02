.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt --suppress-final-state starflow.dpt --spec starflow.json

# compile Lucid to P4
compile: 
	../../dptc starflow.dpt ip_harness.p4 linker_config.json starflow_build --symb starflow.symb 

# compile P4 to tofino
assemble:
	cd starflow_build; make build	 

# test P4 on the asic model
test:
	cd starflow_build; make test ../starflow_test.json
