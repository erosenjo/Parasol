.PHONY: opttest interp compile layout assemble test optimize_timetest dfg preprocess_dfg
LUCID_DIR = .
OPT_DIR = optimization_new/

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py rttoptvars.json --notrafficgen --pair --shortcut

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py rttoptvars.json --notrafficgen --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py rttoptvars.json --notrafficgen --preprocessingonly

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py rttoptvars.json --timetest --pair --shortcut

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state rtt.dpt --spec rtt.json

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc rtt.dpt build --symb rtt.symb --silent 

#layout
layout:
	$(LUCID_DIR)/bin/layout.py rtt.dpt --symb rtt.symb

#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py rtt.dpt --symb rtt.symb --dfg

# compile P4 to tofino
assemble:
	cd build; make build	 

# test P4 on the asic model
test:
	cd build; make test ../rtt_test.json
