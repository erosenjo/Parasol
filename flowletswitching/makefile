.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt --suppress-final-state flowletswitching.dpt

# compile Lucid to P4
compile: 
	../../dptc flowletswitching.dpt ip_harness.p4 linker_config.json flowletswitching_build --symb flowletswitching.symb

# compile P4 to tofino
assemble:
	cd flowletswitching_build; make build	 

# test P4 on the asic model
test:
	cd flowletswitching_build; make test ../flowletswitching_test.json
