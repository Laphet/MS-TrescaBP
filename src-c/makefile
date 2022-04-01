PETSC_DIR = ~/projects/petsc-3.16.5
PETSC_ARCH = arch-linux-c-debug

Experiment : Experiment.o TrescaBP.o Homogenization.o Context.o
	$(LINK.C) -o $@ $^ $(LDLIBS)

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules
include ${PETSC_DIR}/lib/petsc/conf/test