# Copyright (C) Panua Technologies, 2023

# Set C++ compiler and compiler flags
# CXX = icx    # Intel
CXX = cl      # MSVC
CXX_FLAGS = -Ox -EHcs

# Set C compiler and compiler flags
# CC = icx     # Intel
CC = cl       # MSVC
CC_FLAGS = -Ox

# Set Fortran compiler and compiler flags
F77 = ifx      # Intel
F77_FLAGS = -Ox

# Base directory of the Panua Ipopt installation
IPOPT_INSTALL_DIR = $(shell pwd)/..

# Number of cores to be tested
NUM_CORES = 4 2 1

# Name of large PDE instances
LARGE_CASES = MBndryCntrl_3D_27 # MBndryCntrl_3D_27BT

# Dimension of the scalable problem
PROBSIZE = 25

# extensions for object and executable files
OBJEXT = obj
EXEEXT = .exe

# Exectuables that are built for the tests
EXES = hs071_cpp$(EXEEXT) hs071_c$(EXEEXT) \
	hs071_f$(EXEEXT) solve_problem$(EXEEXT)

IPOPT_LIB = $(IPOPT_INSTALL_DIR)/lib/libipopt.lib

INCFLAGS = -I$(IPOPT_INSTALL_DIR)/include

TESTLOGS = $(EXES:=.log)
CASESLOGS = $(LARGE_CASES:=.plog)

SHELL = /bin/bash

all: test

hs071_cpp$(EXEEXT): hs071_nlp.$(OBJEXT) hs071_main.$(OBJEXT)
	$(CXX) -o $@ $(CXX_FLAGS) $? $(IPOPT_LIB)

hs071_c$(EXEEXT): hs071_c.$(OBJEXT)
	$(CC) -o $@ $(CC_FLAGS) $? $(IPOPT_LIB)

hs071_f$(EXEEXT): hs071_f.$(OBJEXT)
	$(F77) -o $@ $(F77_FLAGS) $? $(IPOPT_LIB)

solve_problem$(EXEEXT): \
	solve_problem.$(OBJEXT) \
	MittelmannBndryCntrlDiri3D.$(OBJEXT) \
	MittelmannBndryCntrlDiri3D_27.$(OBJEXT) \
	MittelmannBndryCntrlDiri3Dsin.$(OBJEXT) \
	RegisteredTNLP.$(OBJEXT)
	$(CXX) -o $@ $(CXX_FLAGS) $? $(IPOPT_LIB)

test: $(EXES) ampl.log $(TESTLOGS) $(CASESLOGS)
	@echo To rerun the tests, type "make testclean" first

solve_problem$(EXEEXT).log:
	@touch $@

%.plog: solve_problem$(EXEEXT)
	@echo
	@echo "Testing executable $< for problem $* with size $(PROBSIZE)" | tee $@
	@echo "  for core counts $(NUM_CORES), output stored in $@" | tee -a $@
	@(export PATH="${PATH}:../lib"; \
	for i in $(NUM_CORES); do \
	echo; \
	echo Running with $$i core\(s\):; \
	time (OMP_NUM_THREADS=$$i ./solve_problem$(EXEEXT) $* $(PROBSIZE) >> $@); \
	grep -q "EXIT: Optimal" $@ ; \
	if [ $$? = 0 ]; then echo "  ....Succeeded"; else echo "  ....FAILED"; fi; \
	done) | tee -a $@

ampl.log: mytoy.nl
	@echo
	@echo Testing AMPL executable, output stored in $@ | tee $@
	@(PATH="${PATH}:../lib" ../bin/ipopt$(EXEEXT) $< ) >> $@; \
	grep -q "EXIT: Optimal" $@ ; \
	( if [ $$? = 0 ]; then echo "  ....Succeeded"; else echo "  ....FAILED"; fi) | tee -a $@

%.log: %
	@echo
	@echo Testing executable $<, output stored in $@ | tee $@
	@(PATH="${PATH}:../lib" ./$< ) >> $@; \
	grep -q "EXIT: Optimal" $@ ; \
	( if [ $$? = 0 ]; then echo "  ....Succeeded"; else echo "  ....FAILED"; fi) | tee -a $@

%.$(OBJEXT): %.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCFLAGS) $< -o $@

%.$(OBJEXT): %.c
	$(CC) -c $(CC_FLAGS) $(INCFLAGS) $< -o $@

%.$(OBJEXT): %.f
	$(F77) -c $(F77_FLAGS) $(INCFLAGS) $< -o $@

clean: testclean
	rm -f *.$(OBJEXT) $(EXES) ipopt.out ampl.log

testclean:
	rm -f $(TESTLOGS) $(CASESLOGS) ipopt.out ampl.log
