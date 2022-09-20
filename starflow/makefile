.PHONY: interp compile assemble test
LUCID_DIR = /media/data/mh43/lucid

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state starflow.dpt --spec starflow.json

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc starflow.dpt build --symb starflow.symb --silent

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../starflow_test.json
