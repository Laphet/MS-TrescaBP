#include "Context.h"
#include <stdlib.h>
#include <math.h>

const double QUAD_CORD[QUAD_ORDER] = {-0.77459667, 0.0, 0.77459667};
const double QUAD_WGHT[QUAD_ORDER] = {0.55555556, 0.88888888, 0.55555556};
const double TOL = 1.0e-5;

const double default_T = 1.0;
const double default_Tresca_bnd = 0.004;
const double E0 = 77.2;
const double E1 = 117.0;
const double nu0 = 0.33;
const double nu1 = 0.43;

double base_val_at_quad_pnt[LOC_NDS][QUAD_PNTS] = {{0.0}};
double base_grad_val_at_quad_pnt[LOC_NDS][DIM][QUAD_PNTS] = {{{0.0}}};
double base_grad2_val_at_quad_pnt[LOC_NDS] = {0.25, -0.25, -0.25, 0.25};
double quad_wghts[QUAD_PNTS] = {0.0};

void get_locbase_val(unsigned short loc_ind, double x, double y, double *val)
{
    switch (loc_ind)
    {
    case 0:
        val[0] = 0.25 * (1.0 - x) * (1.0 - y);
        break;
    case 1:
        val[0] = 0.25 * (1.0 + x) * (1.0 - y);
        break;
    case 2:
        val[0] = 0.25 * (1.0 - x) * (1.0 + y);
        break;
    case 3:
        val[0] = 0.25 * (1.0 + x) * (1.0 + y);
        break;
    default:
        break;
    }
}

void get_locbase_grad_val(unsigned short loc_ind, double x, double y, double *grad_val)
{
    switch (loc_ind)
    {
    case 0:
        grad_val[0] = -0.25 * (1.0 - y);
        grad_val[1] = -0.25 * (1.0 - x);
        break;
    case 1:
        grad_val[0] = 0.25 * (1.0 - y);
        grad_val[1] = -0.25 * (1.0 + x);
        break;
    case 2:
        grad_val[0] = -0.25 * (1.0 + y);
        grad_val[1] = 0.25 * (1.0 - x);
        break;
    case 3:
        grad_val[0] = 0.25 * (1.0 + y);
        grad_val[1] = 0.25 * (1.0 + x);
        break;
    default:
        break;
    }
}

void get_C_from_E_nu(double E, double nu, double *C)
{
    double lambda_3d = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
    double mu = E / (2.0 * (1.0 + nu));
    double lambda_2d = 2 * lambda_3d * mu / (lambda_3d + 2.0 * mu);
    C[0] = lambda_2d + 2.0 * mu;
    C[1] = lambda_2d + 2.0 * mu;
    C[2] = mu;
    C[3] = 0.0;
    C[4] = 0.0;
    C[5] = lambda_2d;
}

void default_tr_f1(double x, double t, double *val)
{
    val[0] = 0.01 * sin(2.0 * M_PI * x) * t;
    val[1] = 0.0;
}

void default_tr_f2(double y, double t, double *val)
{
    val[0] = 0.08 * (1.25 - y) * t;
    val[1] = -0.01 * t;
}

void default_bdy_f(double x, double y, double t, double *val)
{
    val[0] = 0.0;
    val[1] = -1.4e-4;
}

void default_cell_cff(double x, double y, const void *paras, double *C)
{
    if ((1.0 / 8.0 <= x && x <= 7.0 / 8.0 && 3.0 / 8.0 <= y && y <= 5.0 / 8.0) || (3.0 / 8.0 <= x && x <= 5.0 / 8.0 && 1.0 / 8.0 <= y && y <= 7.0 / 8.0))
        get_C_from_E_nu(E0, nu0, C);
    else
        get_C_from_E_nu(E1, nu1, C);
}

void connected_cell_cff(double x, double y, const void *paras, double *C)
{
    if ((0.0 / 8.0 <= x && x <= 8.0 / 8.0 && 3.0 / 8.0 <= y && y <= 5.0 / 8.0) || (3.0 / 8.0 <= x && x <= 5.0 / 8.0 && 0.0 / 8.0 <= y && y <= 8.0 / 8.0))
        get_C_from_E_nu(E0, nu0, C);
    else
        get_C_from_E_nu(E1, nu1, C);
}

void const_cell_cff(double x, double y, const void *paras, double *C)
{
    double *paras_ = (double *)paras;
    C[0] = paras_[0];
    C[1] = paras_[1];
    C[2] = paras_[2];
    C[3] = paras_[3];
    C[4] = paras_[4];
    C[5] = paras_[5];
}

void Context_init_(Context *ctx, unsigned int prd, unsigned int grids_on_cell, cell_cff_func cell_cff, const void *paras)
{
    int loc_nd_ind, quad_ind, quad_ind_x, quad_ind_y;
    double grad_val[DIM], x, y;
    for (quad_ind = 0; quad_ind < QUAD_PNTS; ++quad_ind)
    {
        quad_ind_y = quad_ind / QUAD_ORDER;
        quad_ind_x = quad_ind % QUAD_ORDER;
        for (loc_nd_ind = 0; loc_nd_ind < LOC_NDS; ++loc_nd_ind)
        {
            y = QUAD_CORD[quad_ind_y];
            x = QUAD_CORD[quad_ind_x];
            get_locbase_val(loc_nd_ind, x, y, &base_val_at_quad_pnt[loc_nd_ind][quad_ind]);
            get_locbase_grad_val(loc_nd_ind, x, y, &grad_val[0]);
            base_grad_val_at_quad_pnt[loc_nd_ind][0][quad_ind] = grad_val[0];
            base_grad_val_at_quad_pnt[loc_nd_ind][1][quad_ind] = grad_val[1];
        }
        quad_wghts[quad_ind] = QUAD_WGHT[quad_ind_x] * QUAD_WGHT[quad_ind_y];
    }

    ctx->prd = prd;
    ctx->grids_on_cell = grids_on_cell;
    ctx->grids_on_dmn = prd * grids_on_cell;
    ctx->hh = 1.0 / grids_on_cell;
    ctx->h = 1.0 / (prd * grids_on_cell);
    ctx->total_elems = (prd * grids_on_cell) * (prd * grids_on_cell);
    ctx->total_nds = (prd * grids_on_cell + 1) * (prd * grids_on_cell + 1);
    ctx->total_fdms = DIM * (prd * grids_on_cell + 1) * (prd * grids_on_cell + 1);
    ctx->total_elems_on_cell = grids_on_cell * grids_on_cell;
    ctx->total_nds_on_cell = (grids_on_cell + 1) * (grids_on_cell + 1);
    ctx->total_fdms_on_cell = DIM * grids_on_cell * grids_on_cell;
    ctx->cff_data = (double *)malloc(CFF_LEN * grids_on_cell * grids_on_cell * sizeof(double));
    double C[CFF_LEN], hh = 1.0 / grids_on_cell;
    unsigned short i;
    unsigned int sub_elem_ind_x, sub_elem_ind_y;
    unsigned int j;
    for (j = 0; j < grids_on_cell * grids_on_cell; ++j)
    {
        sub_elem_ind_x = j % grids_on_cell;
        sub_elem_ind_y = j / grids_on_cell;
        x = (0.5 + sub_elem_ind_x) * hh;
        y = (0.5 + sub_elem_ind_y) * hh;
        cell_cff(x, y, paras, &C[0]);
        for (i = 0; i < CFF_LEN; ++i)
        {
            ctx->cff_data[j * CFF_LEN + i] = C[i];
        }
    }
}

void Context_final_(Context *ctx)
{
    free(ctx->cff_data);
}