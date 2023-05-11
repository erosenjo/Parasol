.PHONY: opttest optimize_timetest trace_optimize interp compile interactive layout assemble test dfg preprocess_dfg
LUCID_DIR = ../../../lucid
OPT_DIR = ../

# run optimization process (using layout, no traffic gen)
opttest:
	python3 $(OPT_DIR)/optimize.py cmsopt.json --notrafficgen --shortcut

preprocess_dfg:
	python3 $(OPT_DIR)/optimize.py cmsopt.json --notrafficgen --preprocessingonly --dfg

preprocess_efficient:
	python3 $(OPT_DIR)/optimize.py cmsopt.json --notrafficgen --preprocessingonly

# run p4all optimizer, time test
optimize_timetest:
	python3 $(OPT_DIR)/optimize.py cmsopt.json --timetest --shortcut

# trace optimize w/ constant symbolics
trace_optimize:
	python3 $(OPT_DIR)/trace_optimize.py trace_cmsopt.json

# run interpreter
interp: 
	$(LUCID_DIR)/dpt --suppress-final-state trace_cms.dpt

# run interpreter in interactive mode
interactive:
	$(LUCID_DIR)/dpt -i --suppress-final-state trace_cms.dpt

# compile Lucid to P4
compile: 
	$(LUCID_DIR)/dptc trace_cms.dpt --symb trace_cms.symb 

#layout
layout:
	$(LUCID_DIR)/bin/layout.py trace_cms.dpt --symb trace_cms.symb
#dfg compile
dfg:
	$(LUCID_DIR)/bin/layout.py trace_cms.dpt --symb trace_cms.symb --dfg


# compile P4 to tofino
assemble:
	cd lucid_tofino_build; make lucid_tofino_build	 

# test P4 on the asic model
test:
	cd lucid_tofino_build; make test ../trace_cms.json
