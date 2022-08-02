.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt --suppress-final-state hash.dpt 

# compile Lucid to P4
compile: 
	../../dptc hash.dpt ip_harness.p4 linker_config.json hash_build --symb hash.symb 

# compile P4 to tofino
assemble:
	cd hash_build; make build	 

# test P4 on the asic model
test:
	cd hashmodule_build; make test ../hashmodule_test.json
