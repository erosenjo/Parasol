.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt cms.dpt --spec cms.json

# compile Lucid to P4
compile: 
	../../dptc cms.dpt ip_harness.p4 linker_config.json cms_build

# compile P4 to tofino
assemble:
	cd cms_build; make build	 

# test P4 on the asic model
test:
	cd cms_build; make test ../cms_test.json
