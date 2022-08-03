.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt --suppress-final-state cmscache.dpt

# compile Lucid to P4
compile: 
	../../dptc cmscache.dpt ip_harness.p4 linker_config.json cmscache_build --symb cmscache.symb

# compile P4 to tofino
assemble:
	cd cmscache_build; make build	 

# test P4 on the asic model
test:
	cd cmscache_build; make test ../cmscache.json
