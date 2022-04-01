#ifndef HOMOGENIZATION_H_ /* Include guard */
#define HOMOGENIZATION_H_
#include "Context.h"
#include <petsc.h>

#define CORRS_NUM 3

typedef struct Homogenization
{
    Context ctx;
    DM cell_dm;
    Vec corrs[CORRS_NUM];
    double C_eff[CFF_LEN];
} Homogenization;

PetscErrorCode Homogenization_init_(Homogenization *self, unsigned int grids_on_cell, cell_cff_func cell_cff, const void *paras);

PetscErrorCode Homogenization_solve(Homogenization *self);

PetscErrorCode Homogenization_final_(Homogenization *self);

#endif