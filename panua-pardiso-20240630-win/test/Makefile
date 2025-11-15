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

# Base directory of the Panua Pardiso installation
PARDISO_INSTALL_DIR = $(shell pwd)/..

# Pardiso library
PARDISO_LIB = $(PARDISO_INSTALL_DIR)/lib/libpardiso.lib

# Exectuables that are built for the tests
EXES = pardiso_sym.exe pardiso_unsym.exe pardiso_unsym_complex.exe \
	pardiso_sym_f.exe pardiso_unsym_f.exe pardiso_unsym_complex_f.exe \
	pardiso_sym_schur.exe pardiso_unsym_complex_schur.exe pardiso_unsym_schur.exe \
	laplace.exe helmholtz.exe 

INCFLAGS = -I$(PARDISO_INSTALL_DIR)/include

TESTLOGS = $(EXES:.exe=.log)

SHELL = /bin/bash

all: test

test: $(EXES) $(TESTLOGS)
	@echo To rerun the tests, type "make testclean" first

laplace.log: laplace.exe
	@echo
	@echo Testing executable $<, output stored in $@ | tee $@
	@PATH="${PATH}:../lib"  OMP_NUM_THREADS=1 ./$< 10 1 >> $@; \
	grep -q "EXIT: Completed" $@ ; \
	( if [ $$? = 0 ]; then echo "  ....Succeeded"; else echo "  ....FAILED"; fi) | tee -a $@

helmholtz.log: helmholtz.exe
	@echo
	@echo Testing executable $<, output stored in $@ | tee $@
	@PATH="${PATH}:../lib"  OMP_NUM_THREADS=1 ./$< 10 >> $@; \
	grep -q "EXIT: Completed" $@ ; \
	( if [ $$? = 0 ]; then echo "  ....Succeeded"; else echo "  ....FAILED"; fi) | tee -a $@

%.log: %.exe
	@echo
	@echo Testing executable $<, output stored in $@ | tee $@
	@PATH="${PATH}:../lib"  OMP_NUM_THREADS=1 ./$< >> $@; \
	grep -q "EXIT: Completed" $@ ; \
	( if [ $$? = 0 ]; then echo "  ....Succeeded"; else echo "  ....FAILED"; fi) | tee -a $@

%.exe: %.c
	$(CC) $(CC_FLAGS) $(INCFLAGS) $< $(PARDISO_LIB) $(LIBS) -o $@

%.exe: %.cpp
	$(CXX) $(CXX_FLAGS) $(INCFLAGS) $< $(PARDISO_LIB) $(LIBS) -o $@

%.exe: %.f
	$(F77) $(F77_FLAGS) $(INCFLAGS) $< $(PARDISO_LIB) $(LIBS) -o $@

%.$(OBJEXT): %.cpp
	$(CXX) -c $(CXX_FLAGS) $(INCFLAGS) $< -o $@

%.$(OBJEXT): %.c
	$(CC) -c $(CC_FLAGS) $(INCFLAGS) $< -o $@

%.$(OBJEXT): %.f
	$(F77) -c $(F77_FLAGS) $(INCFLAGS) $< -o $@

clean: testclean
	rm -f *.$(OBJEXT) $(EXES)

testclean:
	rm -f $(TESTLOGS) $(CASESLOGS) *.csr
