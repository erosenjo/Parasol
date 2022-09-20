.PHONY: interp compile assemble test
LUCID_DIR = /media/data/mh43/lucid

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state fridge.dpt --spec fridge.json

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc fridge.dpt build --symb fridge.symb --silent

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../fridge_test.json
