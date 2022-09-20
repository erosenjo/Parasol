.PHONY: interp compile assemble test
LUCID_DIR = /media/data/mh43/lucid

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state rtt.dpt --spec rtt.json

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc rtt.dpt build --symb rtt.symb --silent 

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../rtt_test.json
