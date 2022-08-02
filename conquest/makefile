.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt --suppress-final-state conquest.dpt

# compile Lucid to P4
compile: 
	../../dptc noextern_conquest.dpt ip_harness.p4 linker_config.json conquest_build --symb conquest.symb

# compile P4 to tofino
assemble:
	cd conquest_build; make build	 

# test P4 on the asic model
test:
	cd conquest_build; make test ../conquest_test.json
