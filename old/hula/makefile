.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt hula.dpt --spec hula.json

# compile Lucid to P4
compile: 
	../../dptc hula.dpt ip_harness.p4 linker_config.json hula_build 

# compile P4 to tofino
assemble:
	cd hula_build; make build	 

# test P4 on the asic model
test:
	cd hula_build; make test ../hula_test.json
