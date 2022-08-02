.PHONY: interp compile assemble test

# run interpreter
interp: 
	../../dpt --suppress-final-state rtt.dpt --spec rtt.json

# compile Lucid to P4
compile: 
	../../dptc noextern_rtt.dpt ip_harness.p4 linker_config.json rtt_build --symb rtt.symb 

# compile P4 to tofino
assemble:
	cd rtt_build; make build	 

# test P4 on the asic model
test:
	cd rtt_build; make test ../rtt_test.json
