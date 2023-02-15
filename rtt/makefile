.PHONY: opttest interp compile layout assemble test exhaustive optimize_timetest dfg preprocess_dfg
LUCID_DIR = /media/data/mh43/lucid
OPT_DIR = /media/data/mh43/Lucid4All/optimization_new

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py rttoptvars.json --notrafficgen --pair --nopruning --shortcut

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py rttoptvars.json --notrafficgen --nopruning --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py rttoptvars.json --notrafficgen --nopruning --preprocessingonly --efficient

exhaustive:
	python3 $(OPT_DIR)/optimize.py rttoptvars.json --nopruning --exhaustive --pair --shortcut

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py rttoptvars.json --timetest --nopruning --pair --shortcut

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
