#ifndef TRESCA_BP_H_ /* Include guard */
#define TRESCA_BP_H_
#include "Context.h"
#include <petsc.h>

extern const unsigned int MAX_ITERS;
extern const double BT_BETA;
extern const double BT_TZERO;

typedef struct TrescaBP
{
    Context ctx;
    unsigned int grids_on_time;
    double T;
    double tau;
    s2d_t1d_to_2d_func bdy_f;
    s1d_t1d_to_2d_func tr_f1;
    s1d_t1d_to_2d_func tr_f2;
    double Tresca_bnd;
    DM dmn_dm;
    Vec u_init;
    Mat Amat;
} TrescaBP;

PetscErrorCode TrescaBP_init_(TrescaBP *self, unsigned int prd, unsigned int grids_on_cell, cell_cff_func cell_cff, void *paras);

PetscErrorCode TrescaBP_set_conds(TrescaBP *self, unsigned int grids_on_time, double T, s2d_t1d_to_2d_func bdy_f, s1d_t1d_to_2d_func tr_f1, s1d_t1d_to_2d_func tr_f2, double Tresca_bnd);

PetscErrorCode TrescaBP_get_rhs(TrescaBP *self, Vec u_minus, double t, Vec rhs);

PetscErrorCode TrescaBP_get_next_u(TrescaBP *self, Vec u_minus, Vec rhs, Vec u, int *info);

PetscErrorCode TrescaBP_solve(TrescaBP *self, Vec u);

PetscErrorCode TrescaBP_final_(TrescaBP *self);

#endif