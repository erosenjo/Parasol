.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt bloom_module.dpt --spec bloom_module.json

# compile Lucid to P4
compile: 
	../../dptc bloom_module.dpt ip_harness.p4 linker_config.json bloom_module_build --symb bloom_module.symb 

# compile P4 to tofino
assemble:
	cd bloom_module_build; make build	 

# test P4 on the asic model
test:
	cd bloom_module_build; make test ../bloom_module_test.json
