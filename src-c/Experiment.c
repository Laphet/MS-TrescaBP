#include "TrescaBP.h"
#include "Homogenization.h"
#include <math.h>
#include <time.h>
#include "mpi.h"

const unsigned int GRIDS_ON_CELL = 8;
const unsigned int GRIDS_ON_TIME = 128;
const int MAX_LEN_TIMESTR = 26;
const int MAX_LEN_FILENAME = 24;

void get_timestr(char timestr[])
{
    time_t timer;
    struct tm *tm_info;
    timer = time(NULL);
    tm_info = localtime(&timer);
    strftime(timestr, MAX_LEN_TIMESTR, "%Y-%m-%d %H:%M:%S", tm_info);
}

void _get_val_quad_pnt(unsigned short quad_ind, double u0, double u1, double u2, double u3, double *val)
{
    *val = 0.0;
    *val += u0 * base_val_at_quad_pnt[0][quad_ind];
    *val += u1 * base_val_at_quad_pnt[1][quad_ind];
    *val += u2 * base_val_at_quad_pnt[2][quad_ind];
    *val += u3 * base_val_at_quad_pnt[3][quad_ind];
}

void _get_grad_val_quad_pnt(unsigned short quad_ind, double u0, double u1, double u2, double u3, double *val0, double *val1)
{
    *val0 = 0.0;
    *val0 += u0 * base_grad_val_at_quad_pnt[0][0][quad_ind];
    *val0 += u1 * base_grad_val_at_quad_pnt[1][0][quad_ind];
    *val0 += u2 * base_grad_val_at_quad_pnt[2][0][quad_ind];
    *val0 += u3 * base_grad_val_at_quad_pnt[3][0][quad_ind];
    *val1 = 0.0;
    *val1 += u0 * base_grad_val_at_quad_pnt[0][1][quad_ind];
    *val1 += u1 * base_grad_val_at_quad_pnt[1][1][quad_ind];
    *val1 += u2 * base_grad_val_at_quad_pnt[2][1][quad_ind];
    *val1 += u3 * base_grad_val_at_quad_pnt[3][1][quad_ind];
}

void _get_grad2_val_quad_pnt(double u0, double u1, double u2, double u3, double *val)
{
    *val = 0.0;
    *val += u0 * base_grad2_val_at_quad_pnt[0];
    *val += u1 * base_grad2_val_at_quad_pnt[1];
    *val += u2 * base_grad2_val_at_quad_pnt[2];
    *val += u3 * base_grad2_val_at_quad_pnt[3];
}

PetscErrorCode get_errors(Vec u_ms, Vec u_homo, TrescaBP ms_bvp, TrescaBP homo_bvp, Homogenization homo, PetscScalar *errors)
{
    PetscErrorCode ierr;

    int elem_ind_x_start, elem_ind_x_end, nd_ind_x_len, elem_ind_x, elem_ind_y_start, elem_ind_y_end, nd_ind_y_len, elem_ind_y;
    int _elem_ind_x_start, _nd_ind_x_len, _elem_ind_y_start, _nd_ind_y_len;
    ierr = DMDAGetCorners(ms_bvp.dmn_dm, &elem_ind_x_start, &elem_ind_y_start, NULL, &nd_ind_x_len, &nd_ind_y_len, NULL);
    CHKERRQ(ierr);
    ierr = DMDAGetCorners(homo_bvp.dmn_dm, &_elem_ind_x_start, &_elem_ind_y_start, NULL, &_nd_ind_x_len, &_nd_ind_y_len, NULL);
    CHKERRQ(ierr);
    if ((elem_ind_x_start != _elem_ind_x_start) || (elem_ind_y_start != _elem_ind_y_start) || (nd_ind_x_len != _nd_ind_x_len) || (nd_ind_y_len != _nd_ind_y_len))
    {
        ierr = PetscPrintf(PETSC_COMM_WORLD, "!Error: Two DMs are not consistent!\n");
        CHKERRQ(ierr);
        return ierr;
    }

    PetscScalar l2_ref = 0.0, h1_ref = 0.0;
    errors[0] = 0.0; /* L2 error u_ms - u_0 */
    errors[1] = 0.0; /* H1 error u_ms - u_0 */
    errors[2] = 0.0; /* H1 error u_ms - u_0 - correctors */

    PetscScalar ***chi11, ***chi22, ***chi12;
    ierr = DMDAVecGetArrayDOFRead(homo.cell_dm, homo.corrs[0], &chi11);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOFRead(homo.cell_dm, homo.corrs[1], &chi22);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOFRead(homo.cell_dm, homo.corrs[2], &chi12);
    CHKERRQ(ierr);

    Vec u_ms_loc, u_homo_loc; /* Need to destroy. */
    PetscScalar ***u_ms_array, ***u_homo_array;

    ierr = DMCreateLocalVector(ms_bvp.dmn_dm, &u_ms_loc);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocal(ms_bvp.dmn_dm, u_ms, INSERT_VALUES, u_ms_loc);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOFRead(ms_bvp.dmn_dm, u_ms_loc, &u_ms_array);
    CHKERRQ(ierr);

    ierr = DMCreateLocalVector(homo_bvp.dmn_dm, &u_homo_loc);
    CHKERRQ(ierr);
    ierr = DMGlobalToLocal(homo_bvp.dmn_dm, u_homo, INSERT_VALUES, u_homo_loc);
    CHKERRQ(ierr);
    ierr = DMDAVecGetArrayDOFRead(homo_bvp.dmn_dm, u_homo_loc, &u_homo_array);
    CHKERRQ(ierr);

    elem_ind_x_end = elem_ind_x_start + nd_ind_x_len < ms_bvp.ctx.grids_on_dmn + 1 ? elem_ind_x_start + nd_ind_x_len : ms_bvp.ctx.grids_on_dmn;
    elem_ind_y_end = elem_ind_y_start + nd_ind_y_len < ms_bvp.ctx.grids_on_dmn + 1 ? elem_ind_y_start + nd_ind_y_len : ms_bvp.ctx.grids_on_dmn;
    int sub_elem_ind_x, sub_elem_ind_y;
    unsigned short quad_ind;
    PetscScalar hh = ms_bvp.ctx.hh, h = ms_bvp.ctx.h;
    PetscScalar loc_quad_val_h1_corr, loc_val_corrs[CORRS_NUM][DIM][QUAD_PNTS], loc_grad_val_corrs[CORRS_NUM][DIM][DIM][QUAD_PNTS], loc_grad_val_diff[DIM][DIM][QUAD_PNTS], loc_grad_val_homo[DIM][DIM][QUAD_PNTS];
    double val, grad_val0, grad_val1, grad2_val0, grad2_val1, u0, u1, u2, u3;
    for (elem_ind_x = elem_ind_x_start; elem_ind_x < elem_ind_x_end; ++elem_ind_x)
        for (elem_ind_y = elem_ind_y_start; elem_ind_y < elem_ind_y_end; ++elem_ind_y)
        {
            sub_elem_ind_x = elem_ind_x % ms_bvp.ctx.grids_on_cell;
            sub_elem_ind_y = elem_ind_y % ms_bvp.ctx.grids_on_cell;
            ierr = PetscMemzero(&loc_val_corrs[0][0][0], sizeof(loc_val_corrs));
            ierr = PetscMemzero(&loc_grad_val_corrs[0][0][0][0], sizeof(loc_grad_val_corrs));
            ierr = PetscMemzero(&loc_grad_val_homo[0][0][0], sizeof(loc_grad_val_homo));
            ierr = PetscMemzero(&loc_grad_val_diff[0][0][0], sizeof(loc_grad_val_diff));
            CHKERRQ(ierr);

            u0 = u_homo_array[elem_ind_y][elem_ind_x][0];
            u1 = u_homo_array[elem_ind_y][elem_ind_x + 1][0];
            u2 = u_homo_array[elem_ind_y + 1][elem_ind_x][0];
            u3 = u_homo_array[elem_ind_y + 1][elem_ind_x + 1][0];
            _get_grad2_val_quad_pnt(u0, u1, u2, u3, &grad2_val0);
            u0 = u_homo_array[elem_ind_y][elem_ind_x][1];
            u1 = u_homo_array[elem_ind_y][elem_ind_x + 1][1];
            u2 = u_homo_array[elem_ind_y + 1][elem_ind_x][1];
            u3 = u_homo_array[elem_ind_y + 1][elem_ind_x + 1][1];
            _get_grad2_val_quad_pnt(u0, u1, u2, u3, &grad2_val1);

            for (quad_ind = 0; quad_ind < QUAD_PNTS; ++quad_ind)
            {
                u0 = u_ms_array[elem_ind_y][elem_ind_x][0] - u_homo_array[elem_ind_y][elem_ind_x][0];
                u1 = u_ms_array[elem_ind_y][elem_ind_x + 1][0] - u_homo_array[elem_ind_y][elem_ind_x + 1][0];
                u2 = u_ms_array[elem_ind_y + 1][elem_ind_x][0] - u_homo_array[elem_ind_y + 1][elem_ind_x][0];
                u3 = u_ms_array[elem_ind_y + 1][elem_ind_x + 1][0] - u_homo_array[elem_ind_y + 1][elem_ind_x + 1][0];
                _get_val_quad_pnt(quad_ind, u0, u1, u2, u3, &val);
                _get_grad_val_quad_pnt(quad_ind, u0, u1, u2, u3, &grad_val0, &grad_val1);
                errors[0] += 0.25 * h * h * val * val * quad_wghts[quad_ind];
                errors[1] += (grad_val0 * grad_val0 + grad_val1 * grad_val1) * quad_wghts[quad_ind];
                loc_grad_val_diff[0][0][quad_ind] = grad_val0;
                loc_grad_val_diff[1][0][quad_ind] = grad_val1;

                u0 = u_ms_array[elem_ind_y][elem_ind_x][1] - u_homo_array[elem_ind_y][elem_ind_x][1];
                u1 = u_ms_array[elem_ind_y][elem_ind_x + 1][1] - u_homo_array[elem_ind_y][elem_ind_x + 1][1];
                u2 = u_ms_array[elem_ind_y + 1][elem_ind_x][1] - u_homo_array[elem_ind_y + 1][elem_ind_x][1];
                u3 = u_ms_array[elem_ind_y + 1][elem_ind_x + 1][1] - u_homo_array[elem_ind_y + 1][elem_ind_x + 1][1];
                _get_val_quad_pnt(quad_ind, u0, u1, u2, u3, &val);
                _get_grad_val_quad_pnt(quad_ind, u0, u1, u2, u3, &grad_val0, &grad_val1);
                errors[0] += 0.25 * h * h * val * val * quad_wghts[quad_ind];
                errors[1] += (grad_val0 * grad_val0 + grad_val1 * grad_val1) * quad_wghts[quad_ind];
                loc_grad_val_diff[0][1][quad_ind] = grad_val0;
                loc_grad_val_diff[1][1][quad_ind] = grad_val1;

                u0 = u_homo_array[elem_ind_y][elem_ind_x][0];
                u1 = u_homo_array[elem_ind_y][elem_ind_x + 1][0];
                u2 = u_homo_array[elem_ind_y + 1][elem_ind_x][0];
                u3 = u_homo_array[elem_ind_y + 1][elem_ind_x + 1][0];
                _get_val_quad_pnt(quad_ind, u0, u1, u2, u3, &val);
                _get_grad_val_quad_pnt(quad_ind, u0, u1, u2, u3, &grad_val0, &grad_val1);
                l2_ref += 0.25 * h * h * val * val * quad_wghts[quad_ind];
                h1_ref += (grad_val0 * grad_val0 + grad_val1 * grad_val1) * quad_wghts[quad_ind];
                loc_grad_val_homo[0][0][quad_ind] = grad_val0;
                loc_grad_val_homo[1][0][quad_ind] = grad_val1;
                u0 = u_homo_array[elem_ind_y][elem_ind_x][1];
                u1 = u_homo_array[elem_ind_y][elem_ind_x + 1][1];
                u2 = u_homo_array[elem_ind_y + 1][elem_ind_x][1];
                u3 = u_homo_array[elem_ind_y + 1][elem_ind_x + 1][1];
                _get_val_quad_pnt(quad_ind, u0, u1, u2, u3, &val);
                _get_grad_val_quad_pnt(quad_ind, u0, u1, u2, u3, &grad_val0, &grad_val1);
                l2_ref += 0.25 * h * h * val * val * quad_wghts[quad_ind];
                h1_ref += (grad_val0 * grad_val0 + grad_val1 * grad_val1) * quad_wghts[quad_ind];
                loc_grad_val_homo[0][1][quad_ind] = grad_val0;
                loc_grad_val_homo[1][1][quad_ind] = grad_val1;

                u0 = chi11[sub_elem_ind_y][sub_elem_ind_x][0];
                u1 = chi11[sub_elem_ind_y][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][0];
                u2 = chi11[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][sub_elem_ind_x][0];
                u3 = chi11[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][0];
                _get_val_quad_pnt(quad_ind, u0, u1, u2, u3, &val);
                _get_grad_val_quad_pnt(quad_ind, u0, u1, u2, u3, &grad_val0, &grad_val1);
                loc_val_corrs[0][0][quad_ind] = val;
                loc_grad_val_corrs[0][0][0][quad_ind] = grad_val0;
                loc_grad_val_corrs[0][1][0][quad_ind] = grad_val1;
                u0 = chi11[sub_elem_ind_y][sub_elem_ind_x][1];
                u1 = chi11[sub_elem_ind_y][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][1];
                u2 = chi11[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][sub_elem_ind_x][1];
                u3 = chi11[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][1];
                _get_val_quad_pnt(quad_ind, u0, u1, u2, u3, &val);
                _get_grad_val_quad_pnt(quad_ind, u0, u1, u2, u3, &grad_val0, &grad_val1);
                loc_val_corrs[0][1][quad_ind] = val;
                loc_grad_val_corrs[0][0][1][quad_ind] = grad_val0;
                loc_grad_val_corrs[0][1][1][quad_ind] = grad_val1;

                u0 = chi22[sub_elem_ind_y][sub_elem_ind_x][0];
                u1 = chi22[sub_elem_ind_y][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][0];
                u2 = chi22[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][sub_elem_ind_x][0];
                u3 = chi22[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][0];
                _get_val_quad_pnt(quad_ind, u0, u1, u2, u3, &val);
                _get_grad_val_quad_pnt(quad_ind, u0, u1, u2, u3, &grad_val0, &grad_val1);
                loc_val_corrs[1][0][quad_ind] = val;
                loc_grad_val_corrs[1][0][0][quad_ind] = grad_val0;
                loc_grad_val_corrs[1][1][0][quad_ind] = grad_val1;
                u0 = chi22[sub_elem_ind_y][sub_elem_ind_x][1];
                u1 = chi22[sub_elem_ind_y][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][1];
                u2 = chi22[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][sub_elem_ind_x][1];
                u3 = chi22[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][1];
                _get_val_quad_pnt(quad_ind, u0, u1, u2, u3, &val);
                _get_grad_val_quad_pnt(quad_ind, u0, u1, u2, u3, &grad_val0, &grad_val1);
                loc_val_corrs[1][1][quad_ind] = val;
                loc_grad_val_corrs[1][0][1][quad_ind] = grad_val0;
                loc_grad_val_corrs[1][1][1][quad_ind] = grad_val1;

                u0 = chi12[sub_elem_ind_y][sub_elem_ind_x][0];
                u1 = chi12[sub_elem_ind_y][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][0];
                u2 = chi12[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][sub_elem_ind_x][0];
                u3 = chi12[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][0];
                _get_val_quad_pnt(quad_ind, u0, u1, u2, u3, &val);
                _get_grad_val_quad_pnt(quad_ind, u0, u1, u2, u3, &grad_val0, &grad_val1);
                loc_val_corrs[2][0][quad_ind] = val;
                loc_grad_val_corrs[2][0][0][quad_ind] = grad_val0;
                loc_grad_val_corrs[2][1][0][quad_ind] = grad_val1;
                u0 = chi12[sub_elem_ind_y][sub_elem_ind_x][1];
                u1 = chi12[sub_elem_ind_y][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][1];
                u2 = chi12[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][sub_elem_ind_x][1];
                u3 = chi12[(sub_elem_ind_y + 1) % homo.ctx.grids_on_cell][(sub_elem_ind_x + 1) % homo.ctx.grids_on_cell][1];
                _get_val_quad_pnt(quad_ind, u0, u1, u2, u3, &val);
                _get_grad_val_quad_pnt(quad_ind, u0, u1, u2, u3, &grad_val0, &grad_val1);
                loc_val_corrs[2][1][quad_ind] = val;
                loc_grad_val_corrs[2][0][1][quad_ind] = grad_val0;
                loc_grad_val_corrs[2][1][1][quad_ind] = grad_val1;
            }
            for (quad_ind = 0; quad_ind < QUAD_PNTS; ++quad_ind)
            {
                /* k=1, alpha=1 */
                loc_quad_val_h1_corr = loc_grad_val_diff[0][0][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[0][0][0][quad_ind] * loc_grad_val_homo[0][0][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[1][0][0][quad_ind] * loc_grad_val_homo[1][1][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[2][0][0][quad_ind] * loc_grad_val_homo[0][1][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[2][0][0][quad_ind] * loc_grad_val_homo[1][0][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_val_corrs[2][0][quad_ind] * grad2_val0; /* j=2, beta=1 */
                loc_quad_val_h1_corr -= 2.0 / hh * loc_val_corrs[1][0][quad_ind] * grad2_val1; /* j=2, beta=2 */
                errors[2] += loc_quad_val_h1_corr * loc_quad_val_h1_corr * quad_wghts[quad_ind];

                /* k=2, alpha=2 */
                loc_quad_val_h1_corr = loc_grad_val_diff[1][1][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[0][1][1][quad_ind] * loc_grad_val_homo[0][0][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[1][1][1][quad_ind] * loc_grad_val_homo[1][1][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[2][1][1][quad_ind] * loc_grad_val_homo[0][1][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[2][1][1][quad_ind] * loc_grad_val_homo[1][0][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_val_corrs[0][1][quad_ind] * grad2_val0; /* j=1, beta=1 */
                loc_quad_val_h1_corr -= 2.0 / hh * loc_val_corrs[2][1][quad_ind] * grad2_val1; /* j=1, beta=2 */
                errors[2] += loc_quad_val_h1_corr * loc_quad_val_h1_corr * quad_wghts[quad_ind];

                /* k=1, alpha=2 */
                loc_quad_val_h1_corr = loc_grad_val_diff[0][1][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[0][0][1][quad_ind] * loc_grad_val_homo[0][0][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[1][0][1][quad_ind] * loc_grad_val_homo[1][1][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[2][0][1][quad_ind] * loc_grad_val_homo[0][1][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[2][0][1][quad_ind] * loc_grad_val_homo[1][0][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_val_corrs[2][1][quad_ind] * grad2_val0; /* j=2, beta=1 */
                loc_quad_val_h1_corr -= 2.0 / hh * loc_val_corrs[1][1][quad_ind] * grad2_val1; /* j=2, beta=2 */
                errors[2] += loc_quad_val_h1_corr * loc_quad_val_h1_corr * quad_wghts[quad_ind];

                /* k=2, alpha=1 */
                loc_quad_val_h1_corr = loc_grad_val_diff[1][0][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[0][1][0][quad_ind] * loc_grad_val_homo[0][0][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[1][1][0][quad_ind] * loc_grad_val_homo[1][1][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[2][1][0][quad_ind] * loc_grad_val_homo[0][1][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_grad_val_corrs[2][1][0][quad_ind] * loc_grad_val_homo[1][0][quad_ind];
                loc_quad_val_h1_corr -= 2.0 / hh * loc_val_corrs[0][0][quad_ind] * grad2_val0; /* j=1, beta=1 */
                loc_quad_val_h1_corr -= 2.0 / hh * loc_val_corrs[2][0][quad_ind] * grad2_val1; /* j=1, beta=2 */
                errors[2] += loc_quad_val_h1_corr * loc_quad_val_h1_corr * quad_wghts[quad_ind];
            }
        }

    ierr = DMDAVecRestoreArrayDOFRead(ms_bvp.dmn_dm, u_ms_loc, &u_ms_array);
    CHKERRQ(ierr);
    ierr = VecDestroy(&u_ms_loc);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayDOFRead(homo_bvp.dmn_dm, u_homo_loc, &u_homo_array);
    CHKERRQ(ierr);
    ierr = VecDestroy(&u_homo_loc);
    CHKERRQ(ierr);

    ierr = DMDAVecRestoreArrayDOFRead(homo.cell_dm, homo.corrs[0], &chi11);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayDOFRead(homo.cell_dm, homo.corrs[1], &chi22);
    CHKERRQ(ierr);
    ierr = DMDAVecRestoreArrayDOFRead(homo.cell_dm, homo.corrs[2], &chi12);
    CHKERRQ(ierr);

    int process_rank;
    MPI_Comm_rank(PETSC_COMM_WORLD, &process_rank);
    // printf("pid=%d, before MPI_Reduce.\n", process_rank);
    double global_l2 = 0.0, global_h1 = 0.0, global_h1_corr = 0.0, global_l2_ref = 0.0, global_h1_ref = 0.0;
    MPI_Reduce(&errors[0], &global_l2, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&errors[1], &global_h1, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&errors[2], &global_h1_corr, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&l2_ref, &global_l2_ref, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    MPI_Reduce(&h1_ref, &global_h1_ref, 1, MPI_DOUBLE, MPI_SUM, 0, PETSC_COMM_WORLD);
    if (process_rank == 0)
    {
        errors[0] = sqrt(global_l2) / sqrt(global_l2_ref);
        errors[1] = sqrt(global_h1) / sqrt(global_h1_ref);
        errors[2] = sqrt(global_h1_corr) / sqrt(global_h1_ref);
    }
    else
    {
        errors[0] = sqrt(errors[0]) / sqrt(l2_ref);
        errors[1] = sqrt(errors[1]) / sqrt(h1_ref);
        errors[2] = sqrt(errors[2]) / sqrt(h1_ref);
    }
    // printf("pid=%d, after MPI_Reduce, l2=%.5f, h1=%.5f, h1_corr=%.5f.\n", process_rank, errors[0], errors[1], errors[2]);

    // errors[0] = sqrt(errors[0]) / sqrt(l2_ref);
    // errors[1] = sqrt(errors[1]) / sqrt(h1_ref);
    // errors[2] = sqrt(errors[2]) / sqrt(h1_ref);

    return ierr;
}

PetscErrorCode test_set(char op, Homogenization *homo, TrescaBP *ms_bvp, TrescaBP *homo_bvp)
{
    PetscErrorCode ierr = 0;
    unsigned int prd, _grids_on_time, _grids_on_cell;
    switch (op)
    {
    case:
        'x' : prd = 2;
        _grids_on_time = 32;
        _grids_on_cell = 8;
        ierr = Homogenization_init_(homo, _grids_on_cell, default_cell_cff, NULL);
        ierr = Homogenization_solve(homo);
        ierr = TrescaBP_init_(ms_bvp, prd, _grids_on_cell, default_cell_cff, NULL);
        ierr = TrescaBP_set_conds(ms_bvp, _grids_on_time, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        ierr = TrescaBP_init_(homo_bvp, prd, _grids_on_cell, const_cell_cff, &homo->C_eff[0]);
        ierr = TrescaBP_set_conds(homo_bvp, _grids_on_time, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        break;
    case 'a':
        ierr = Homogenization_init_(homo, GRIDS_ON_CELL, default_cell_cff, NULL);
        ierr = Homogenization_solve(homo);
        prd = 2;
        ierr = TrescaBP_init_(ms_bvp, prd, GRIDS_ON_CELL, default_cell_cff, NULL);
        ierr = TrescaBP_set_conds(ms_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        ierr = TrescaBP_init_(homo_bvp, prd, GRIDS_ON_CELL, const_cell_cff, &homo->C_eff[0]);
        ierr = TrescaBP_set_conds(homo_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        break;
    case 'b':
        ierr = Homogenization_init_(homo, GRIDS_ON_CELL, default_cell_cff, NULL);
        ierr = Homogenization_solve(homo);
        prd = 4;
        ierr = TrescaBP_init_(ms_bvp, prd, GRIDS_ON_CELL, default_cell_cff, NULL);
        ierr = TrescaBP_set_conds(ms_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        ierr = TrescaBP_init_(homo_bvp, prd, GRIDS_ON_CELL, const_cell_cff, &homo->C_eff[0]);
        ierr = TrescaBP_set_conds(homo_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        break;
    case 'c':
        ierr = Homogenization_init_(homo, GRIDS_ON_CELL, default_cell_cff, NULL);
        ierr = Homogenization_solve(homo);
        prd = 8;
        ierr = TrescaBP_init_(ms_bvp, prd, GRIDS_ON_CELL, default_cell_cff, NULL);
        ierr = TrescaBP_set_conds(ms_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        ierr = TrescaBP_init_(homo_bvp, prd, GRIDS_ON_CELL, const_cell_cff, &homo->C_eff[0]);
        ierr = TrescaBP_set_conds(homo_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        break;
    case 'd':
        ierr = Homogenization_init_(homo, GRIDS_ON_CELL, default_cell_cff, NULL);
        ierr = Homogenization_solve(homo);
        prd = 16;
        ierr = TrescaBP_init_(ms_bvp, prd, GRIDS_ON_CELL, default_cell_cff, NULL);
        ierr = TrescaBP_set_conds(ms_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        ierr = TrescaBP_init_(homo_bvp, prd, GRIDS_ON_CELL, const_cell_cff, &homo->C_eff[0]);
        ierr = TrescaBP_set_conds(homo_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        break;
    case 'e':
        ierr = Homogenization_init_(homo, GRIDS_ON_CELL, connected_cell_cff, NULL);
        ierr = Homogenization_solve(homo);
        prd = 2;
        ierr = TrescaBP_init_(ms_bvp, prd, GRIDS_ON_CELL, connected_cell_cff, NULL);
        ierr = TrescaBP_set_conds(ms_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        ierr = TrescaBP_init_(homo_bvp, prd, GRIDS_ON_CELL, const_cell_cff, &homo->C_eff[0]);
        ierr = TrescaBP_set_conds(homo_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        break;
    case 'f':
        ierr = Homogenization_init_(homo, GRIDS_ON_CELL, connected_cell_cff, NULL);
        ierr = Homogenization_solve(homo);
        prd = 4;
        ierr = TrescaBP_init_(ms_bvp, prd, GRIDS_ON_CELL, connected_cell_cff, NULL);
        ierr = TrescaBP_set_conds(ms_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        ierr = TrescaBP_init_(homo_bvp, prd, GRIDS_ON_CELL, const_cell_cff, &homo->C_eff[0]);
        ierr = TrescaBP_set_conds(homo_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        break;
    case 'g':
        ierr = Homogenization_init_(homo, GRIDS_ON_CELL, connected_cell_cff, NULL);
        ierr = Homogenization_solve(homo);
        prd = 8;
        ierr = TrescaBP_init_(ms_bvp, prd, GRIDS_ON_CELL, connected_cell_cff, NULL);
        ierr = TrescaBP_set_conds(ms_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        ierr = TrescaBP_init_(homo_bvp, prd, GRIDS_ON_CELL, const_cell_cff, &homo->C_eff[0]);
        ierr = TrescaBP_set_conds(homo_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        break;
    case 'h':
        ierr = Homogenization_init_(homo, GRIDS_ON_CELL, connected_cell_cff, NULL);
        ierr = Homogenization_solve(homo);
        prd = 16;
        ierr = TrescaBP_init_(ms_bvp, prd, GRIDS_ON_CELL, connected_cell_cff, NULL);
        ierr = TrescaBP_set_conds(ms_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        ierr = TrescaBP_init_(homo_bvp, prd, GRIDS_ON_CELL, const_cell_cff, &homo->C_eff[0]);
        ierr = TrescaBP_set_conds(homo_bvp, GRIDS_ON_TIME, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
        break;
    default:
        break;
    }
    return ierr;
}

int main(int argc, char *argv[])
{
    PetscErrorCode ierr;

    char timestr[MAX_LEN_TIMESTR];
    ierr = PetscInitialize(&argc, &argv, NULL, NULL);
    if (ierr)
        return ierr;
    /*
    {
        int pid, sum = 0;
        MPI_Comm_rank(PETSC_COMM_WORLD, &pid);
        MPI_Reduce(&pid, &sum, 1, MPI_INT, MPI_SUM, 0, PETSC_COMM_WORLD);
        printf("pid=%d, sum=%d.\n", pid, sum);
    }
    */
    char op[] = "a";
    ierr = PetscOptionsGetString(NULL, NULL, "-op", op, sizeof(op), NULL);
    CHKERRQ(ierr);
    Homogenization homo;
    TrescaBP ms_bvp, homo_bvp;
    ierr = test_set(op[0], &homo, &ms_bvp, &homo_bvp);
    get_timestr(timestr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s...... Option=[%s], epsilon=1/%d, h=1/%d, tau=1%d.\n", timestr, op, ms_bvp.ctx.prd, ms_bvp.ctx.grids_on_dmn, ms_bvp.grids_on_time);
    CHKERRQ(ierr);
    PetscScalar errors[3 * ms_bvp.grids_on_time], max_error_l2 = 0.0, max_error_h1 = 0.0, max_error_h1_corr = 0.0;

    Vec u_ms_minus, u_ms, rhs_ms, u_homo_minus, u_homo, rhs_homo; /* Need to destroy. */

    ierr = VecDuplicate(ms_bvp.u_init, &u_ms_minus);
    CHKERRQ(ierr);
    ierr = VecDuplicate(ms_bvp.u_init, &u_ms);
    CHKERRQ(ierr);
    ierr = VecCopy(ms_bvp.u_init, u_ms);
    CHKERRQ(ierr);
    ierr = VecDuplicate(ms_bvp.u_init, &rhs_ms);
    CHKERRQ(ierr);

    ierr = VecDuplicate(homo_bvp.u_init, &u_homo_minus);
    CHKERRQ(ierr);
    ierr = VecDuplicate(homo_bvp.u_init, &u_homo);
    CHKERRQ(ierr);
    ierr = VecCopy(homo_bvp.u_init, u_homo);
    CHKERRQ(ierr);
    ierr = VecDuplicate(homo_bvp.u_init, &rhs_homo);
    CHKERRQ(ierr);

    unsigned int time_ind;
    int info_ms, info_homo;
    PetscScalar t;

    double time_threshold = 0.00, prg_bar;
    for (time_ind = 1; time_ind <= 1; ++time_ind)
    {
        t = (PetscScalar)time_ind * ms_bvp.tau;

        ierr = VecZeroEntries(rhs_ms);
        CHKERRQ(ierr);
        ierr = VecCopy(u_ms, u_ms_minus);
        CHKERRQ(ierr);
        ierr = TrescaBP_get_rhs(&ms_bvp, u_ms_minus, t, rhs_ms);
        ierr = TrescaBP_get_next_u(&ms_bvp, u_ms_minus, rhs_ms, u_ms, &info_ms);

        ierr = VecZeroEntries(rhs_homo);
        CHKERRQ(ierr);
        ierr = VecCopy(u_homo, u_homo_minus);
        CHKERRQ(ierr);
        ierr = TrescaBP_get_rhs(&homo_bvp, u_homo_minus, t, rhs_homo);
        ierr = TrescaBP_get_next_u(&homo_bvp, u_homo_minus, rhs_homo, u_homo, &info_homo);

        ierr = get_errors(u_ms, u_homo, ms_bvp, homo_bvp, homo, &errors[3 * (time_ind - 1)]);

        max_error_l2 = errors[3 * (time_ind - 1)] >= max_error_l2 ? errors[3 * (time_ind - 1)] : max_error_l2;
        max_error_h1 = errors[3 * (time_ind - 1) + 1] >= max_error_h1 ? errors[3 * (time_ind - 1) + 1] : max_error_h1;
        max_error_h1_corr = errors[3 * (time_ind - 1) + 2] >= max_error_h1_corr ? errors[3 * (time_ind - 1) + 2] : max_error_h1_corr;

        prg_bar = (double)time_ind / (double)ms_bvp.grids_on_time;
        if (prg_bar >= time_threshold)
        {
            get_timestr(timestr);
            ierr = PetscPrintf(PETSC_COMM_WORLD, "%s...... [%.2f%%], iter_ms=%d, iter_homo=%d.\n", timestr, 100.0 * prg_bar, info_ms, info_homo);
            CHKERRQ(ierr);
            time_threshold += 0.01;
        }
    }
    get_timestr(timestr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "%s...... l2=%.5f, h1=%.5f, h1_corr=%.5f.\n", timestr, max_error_l2, max_error_h1, max_error_h1_corr);
    CHKERRQ(ierr);

    PetscViewer viewer;
    char file_name[MAX_LEN_FILENAME];

    snprintf(file_name, MAX_LEN_FILENAME, "u_ms-op-%s.dat", op);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_name, FILE_MODE_WRITE, &viewer);
    CHKERRQ(ierr);
    ierr = VecView(u_ms, viewer);
    CHKERRQ(ierr);

    snprintf(file_name, MAX_LEN_FILENAME, "u_homo-op-%s.dat", op);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, file_name, FILE_MODE_WRITE, &viewer);
    CHKERRQ(ierr);
    ierr = VecView(u_homo, viewer);
    CHKERRQ(ierr);

    snprintf(file_name, MAX_LEN_FILENAME, "corr11-op-%s.dat", op);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF, file_name, FILE_MODE_WRITE, &viewer);
    CHKERRQ(ierr);
    ierr = VecView(homo.corrs[0], viewer);
    CHKERRQ(ierr);

    snprintf(file_name, MAX_LEN_FILENAME, "corr22-op-%s.dat", op);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF, file_name, FILE_MODE_WRITE, &viewer);
    CHKERRQ(ierr);
    ierr = VecView(homo.corrs[1], viewer);
    CHKERRQ(ierr);

    snprintf(file_name, MAX_LEN_FILENAME, "corr12-op-%s.dat", op);
    ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF, file_name, FILE_MODE_WRITE, &viewer);
    CHKERRQ(ierr);
    ierr = VecView(homo.corrs[2], viewer);
    CHKERRQ(ierr);

    ierr = VecDestroy(&u_ms_minus);
    CHKERRQ(ierr);
    ierr = VecDestroy(&u_ms);
    CHKERRQ(ierr);
    ierr = VecDestroy(&rhs_ms);
    CHKERRQ(ierr);
    ierr = VecDestroy(&u_homo_minus);
    CHKERRQ(ierr);
    ierr = VecDestroy(&u_homo);
    CHKERRQ(ierr);
    ierr = VecDestroy(&rhs_homo);
    CHKERRQ(ierr);

    ierr = TrescaBP_final_(&ms_bvp);
    ierr = TrescaBP_final_(&homo_bvp);
    ierr = Homogenization_final_(&homo);

    PetscFinalize();
}