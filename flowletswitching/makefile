.PHONY: interp compile assemble test
LUCID_DIR = /media/data/mh43/lucid

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state flowletswitching.dpt

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc flowletswitching.dpt build --symb flowletswitching.symb --silent

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd flowletswitching_build; make test ../flowletswitching_test.json
