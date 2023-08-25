.PHONY: opttest interp compile layout assemble test optimize_timetest dfg preprocess_dfg
LUCID_DIR = ../../../lucid
OPT_DIR = ../optimizer

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py rttopt.json --notrafficgen --pair --shortcut

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py rttopt.json --notrafficgen --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py rttopt.json --notrafficgen --preprocessingonly

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py rttopt.json --timetest --pair --shortcut

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state rtt.dpt 

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc rtt.dpt --symb rtt.symb --silent 

#layout
layout:
	$(LUCID_DIR)/bin/layout.py rtt.dpt --symb rtt.symb

#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py rtt.dpt --symb rtt.symb --dfg

# compile P4 to tofino
assemble:
	cd lucid_tofino_build; make lucid_tofino_build	 

# test P4 on the asic model
test:
	cd lucid_tofino_build; make test ../rtt_test.json
