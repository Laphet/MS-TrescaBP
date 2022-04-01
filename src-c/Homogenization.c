#include "Homogenization.h"

PetscErrorCode Homogenization_init_(Homogenization *self, unsigned int grids_on_cell, cell_cff_func cell_cff, const void *paras)
{

    Context_init_(&self->ctx, 1, grids_on_cell, cell_cff, paras);
    PetscErrorCode ierr;

    /* Create DM object on the cell domain. */
    ierr = DMDACreate2d(PETSC_COMM_SELF, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, grids_on_cell, grids_on_cell, PETSC_DECIDE, PETSC_DECIDE, DIM, 1, NULL, NULL, &self->cell_dm);
    CHKERRQ(ierr);
    /* Set up DM first! */
    ierr = DMSetUp(self->cell_dm);
    CHKERRQ(ierr);

    /* Initialize corrector Vecs. */
    unsigned short i;
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = DMCreateGlobalVector(self->cell_dm, &(self->corrs[i]));
        CHKERRQ(ierr);
    }

    /* Initialize C_eff. */
    ierr = PetscMemzero(&self->C_eff[0], sizeof(self->C_eff));
    return ierr;
}

PetscErrorCode Homogenization_solve(Homogenization *self)
{
    PetscErrorCode ierr;
    Mat Amat;
    ierr = DMCreateMatrix(self->cell_dm, &Amat);
    CHKERRQ(ierr);
    Vec rhs[CORRS_NUM], rhs_l[CORRS_NUM];
    unsigned short i;
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = DMCreateGlobalVector(self->cell_dm, &rhs[i]);
        CHKERRQ(ierr);
        ierr = DMCreateLocalVector(self->cell_dm, &rhs_l[i]);
        CHKERRQ(ierr);
    }
    PetscInt elem_ind_x_start, elem_ind_y_start, elem_ind_x_len, elem_ind_y_len, elem_ind_x, elem_ind_y, elem_ind;
    ierr = DMDAGetCorners(self->cell_dm, &elem_ind_x_start, &elem_ind_y_start, NULL, &elem_ind_x_len, &elem_ind_y_len, NULL);
    CHKERRQ(ierr);
    MatStencil row[LOC_FDMS], col[LOC_FDMS];
    PetscScalar ***array0, ***array1, ***array2;
    ierr = DMDAVecGetArrayDOF(self->cell_dm, rhs_l[0], &array0);
    ierr = DMDAVecGetArrayDOF(self->cell_dm, rhs_l[1], &array1);
    ierr = DMDAVecGetArrayDOF(self->cell_dm, rhs_l[2], &array2);
    CHKERRQ(ierr);
    /*
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = DMDAVecGetArrayDOF(self->cell_dm, rhs_l[i], &array[i]);
        CHKERRQ(ierr);
    }
    */
    /* Construct the linear systems. */
    double *C, C_ext[DIM][DIM][DIM][DIM];
    unsigned short loc_nd_ind_row, loc_nd_ind_col, quad_ind, val_ind;
    PetscInt row_nd_ind_x, row_nd_ind_y, col_nd_ind_x, col_nd_ind_y;
    PetscScalar values[LOC_FDMS * LOC_FDMS];
    for (elem_ind_y = elem_ind_y_start; elem_ind_y < elem_ind_y_start + elem_ind_y_len; ++elem_ind_y)
        for (elem_ind_x = elem_ind_x_start; elem_ind_x < elem_ind_x_start + elem_ind_x_len; ++elem_ind_x)
        {
            elem_ind = elem_ind_y * self->ctx.grids_on_cell + elem_ind_x;
            C = &self->ctx.cff_data[elem_ind * CFF_LEN];
            C_ext[0][0][0][0] = C[0]; /* C_1111 = C11 */
            C_ext[0][0][0][1] = C[4]; /* C_1112 = C13 */
            C_ext[0][0][1][0] = C[4]; /* C_1121 = C_1112 = C13 */
            C_ext[0][0][1][1] = C[5]; /* C_1122 = C12 */
            C_ext[0][1][0][0] = C[4]; /* C_1211 = C_1112 = C13 */
            C_ext[0][1][0][1] = C[2]; /* C_1212 = C33 */
            C_ext[0][1][1][0] = C[2]; /* C_1221 = C_1212 = C33 */
            C_ext[0][1][1][1] = C[3]; /* C_1222 = C2212 = C23 */

            C_ext[1][0][0][0] = C[4]; /* C_2111 = C1112 = C13 */
            C_ext[1][0][0][1] = C[2]; /* C_2112 = C1212 = C33 */
            C_ext[1][0][1][0] = C[2]; /* C_2121 = C1212 = C33 */
            C_ext[1][0][1][1] = C[3]; /* C_2122 = C2212 = C23 */
            C_ext[1][1][0][0] = C[5]; /* C_2211 = C1122 = C12 */
            C_ext[1][1][0][1] = C[3]; /* C_2212 = C23 */
            C_ext[1][1][1][0] = C[3]; /* C_2221 = C_2212 = C23 */
            C_ext[1][1][1][1] = C[1]; /* C_2222 = C22 */

            PetscMemzero(&values[0], sizeof(values));
            for (loc_nd_ind_row = 0; loc_nd_ind_row < LOC_NDS; ++loc_nd_ind_row)
            {
                row_nd_ind_x = (elem_ind_x + loc_nd_ind_row % 2) % self->ctx.grids_on_cell;
                row_nd_ind_y = (elem_ind_y + loc_nd_ind_row / 2) % self->ctx.grids_on_cell;
                for (quad_ind = 0; quad_ind < QUAD_PNTS; ++quad_ind)
                {
                    array0[row_nd_ind_y][row_nd_ind_x][0] -= 0.5 * self->ctx.hh * C_ext[0][0][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array0[row_nd_ind_y][row_nd_ind_x][0] -= 0.5 * self->ctx.hh * C_ext[1][0][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];
                    array1[row_nd_ind_y][row_nd_ind_x][0] -= 0.5 * self->ctx.hh * C_ext[0][0][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array1[row_nd_ind_y][row_nd_ind_x][0] -= 0.5 * self->ctx.hh * C_ext[1][0][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];
                    array2[row_nd_ind_y][row_nd_ind_x][0] -= 0.5 * self->ctx.hh * C_ext[0][0][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array2[row_nd_ind_y][row_nd_ind_x][0] -= 0.5 * self->ctx.hh * C_ext[1][0][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];

                    array0[row_nd_ind_y][row_nd_ind_x][1] -= 0.5 * self->ctx.hh * C_ext[0][1][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array0[row_nd_ind_y][row_nd_ind_x][1] -= 0.5 * self->ctx.hh * C_ext[1][1][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];
                    array1[row_nd_ind_y][row_nd_ind_x][1] -= 0.5 * self->ctx.hh * C_ext[0][1][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array1[row_nd_ind_y][row_nd_ind_x][1] -= 0.5 * self->ctx.hh * C_ext[1][1][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];
                    array2[row_nd_ind_y][row_nd_ind_x][1] -= 0.5 * self->ctx.hh * C_ext[0][1][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array2[row_nd_ind_y][row_nd_ind_x][1] -= 0.5 * self->ctx.hh * C_ext[1][1][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];
                    /*  
                        fdm_m = m * XY  # m'=0
                        loc_rhs[0, fdm_m] += C_ext[0, 0, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[0, fdm_m] += C_ext[1, 0, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)
                        loc_rhs[1, fdm_m] += C_ext[0, 0, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[1, fdm_m] += C_ext[1, 0, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)
                        loc_rhs[2, fdm_m] += C_ext[0, 0, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[2, fdm_m] += C_ext[1, 0, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)

                        fdm_m = m * XY + 1  # m'=1
                        loc_rhs[0, fdm_m] += C_ext[0, 1, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[0, fdm_m] += C_ext[1, 1, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)
                        loc_rhs[1, fdm_m] += C_ext[0, 1, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[1, fdm_m] += C_ext[1, 1, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)
                        loc_rhs[2, fdm_m] += C_ext[0, 1, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[2, fdm_m] += C_ext[1, 1, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)
                     */
                }
                row[loc_nd_ind_row * DIM].i = row_nd_ind_x;
                row[loc_nd_ind_row * DIM].j = row_nd_ind_y;
                row[loc_nd_ind_row * DIM].c = 0;
                row[loc_nd_ind_row * DIM + 1].i = row_nd_ind_x;
                row[loc_nd_ind_row * DIM + 1].j = row_nd_ind_y;
                row[loc_nd_ind_row * DIM + 1].c = 1;

                for (loc_nd_ind_col = 0; loc_nd_ind_col < LOC_NDS; ++loc_nd_ind_col)
                {
                    col_nd_ind_x = (elem_ind_x + loc_nd_ind_col % 2) % self->ctx.grids_on_cell;
                    col_nd_ind_y = (elem_ind_y + loc_nd_ind_col / 2) % self->ctx.grids_on_cell;
                    col[loc_nd_ind_col * DIM].i = col_nd_ind_x;
                    col[loc_nd_ind_col * DIM].j = col_nd_ind_y;
                    col[loc_nd_ind_col * DIM].c = 0;
                    col[loc_nd_ind_col * DIM + 1].i = col_nd_ind_x;
                    col[loc_nd_ind_col * DIM + 1].j = col_nd_ind_y;
                    col[loc_nd_ind_col * DIM + 1].c = 1;
                    /*
                        fdm_m, fdm_n = m * XY, n * XY  # m'=0, n'=0
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 0, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 0, 1, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 0, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 0, 1, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        fdm_m, fdm_n = m * XY, n * XY + 1  # m'=0, n'=1
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 1, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 1, 1, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 1, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 1, 1, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        fdm_m, fdm_n = m * XY + 1, n * XY  # m'=1, n'=0
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 0, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 0, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 0, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 0, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        fdm_m, fdm_n = m * XY + 1, n * XY + 1  #m'=1, n'=1
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 1, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 1, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 1, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 1, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                    */
                    for (quad_ind = 0; quad_ind < QUAD_PNTS; ++quad_ind)
                    {
                        val_ind = DIM * loc_nd_ind_row * LOC_FDMS + DIM * loc_nd_ind_col;
                        values[val_ind] += C_ext[0][0][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[0][0][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][0][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][0][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        val_ind = DIM * loc_nd_ind_row * LOC_FDMS + DIM * loc_nd_ind_col + 1;
                        values[val_ind] += C_ext[0][1][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[0][1][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][1][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][1][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        val_ind = (DIM * loc_nd_ind_row + 1) * LOC_FDMS + DIM * loc_nd_ind_col;
                        values[val_ind] += C_ext[0][0][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[0][0][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][0][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][0][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        val_ind = (DIM * loc_nd_ind_row + 1) * LOC_FDMS + DIM * loc_nd_ind_col + 1;
                        values[val_ind] += C_ext[0][1][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[0][1][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][1][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][1][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                    }
                }
            }
            ierr = MatSetValuesStencil(Amat, LOC_FDMS, row, LOC_FDMS, col, values, ADD_VALUES);
            CHKERRQ(ierr);
        }
    ierr = DMDAVecRestoreArrayDOF(self->cell_dm, rhs_l[0], &array0);
    ierr = DMDAVecRestoreArrayDOF(self->cell_dm, rhs_l[1], &array1);
    ierr = DMDAVecRestoreArrayDOF(self->cell_dm, rhs_l[2], &array2);
    CHKERRQ(ierr);
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = DMLocalToGlobal(self->cell_dm, rhs_l[i], ADD_VALUES, rhs[i]);
        CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    /* Set NullSpace of Amat */
    /*
    Vec nullvec[2];
    ierr = DMCreateGlobalVector(self->cell_dm, &nullvec[0]);
    ierr = DMCreateGlobalVector(self->cell_dm, &nullvec[1]);
    PetscScalar ***nullvec_array[2];
    ierr = DMDAVecGetArrayDOF(self->cell_dm, nullvec[0], &nullvec_array[0]);
    ierr = DMDAVecGetArrayDOF(self->cell_dm, nullvec[1], &nullvec_array[1]);
    CHKERRQ(ierr);
    for (elem_ind_y = 0; elem_ind_y < self->ctx.grids_on_cell; ++elem_ind_y)
        for (elem_ind_x = 0; elem_ind_x < self->ctx.grids_on_cell; ++elem_ind_x)
        {
            nullvec_array[0][0][elem_ind_y][elem_ind_x] = 1.0;
            nullvec_array[0][1][elem_ind_y][elem_ind_x] = 0.0;
            nullvec_array[1][0][elem_ind_y][elem_ind_x] = 0.0;
            nullvec_array[1][1][elem_ind_y][elem_ind_x] = 1.0;
        }
    ierr = DMDAVecRestoreArrayDOF(self->cell_dm, nullvec[0], &nullvec_array[0]);
    ierr = DMDAVecRestoreArrayDOF(self->cell_dm, nullvec[1], &nullvec_array[1]);
    CHKERRQ(ierr);
    ierr = VecNormalize(nullvec[0], NULL);
    ierr = VecNormalize(nullvec[1], NULL);
    CHKERRQ(ierr);
    MatNullSpace msp;
    ierr = MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, 2, nullvec, &msp);
    CHKERRQ(ierr);
    ierr = MatSetNullSpace(Amat, msp);
    CHKERRQ(ierr);
    */

    /*
    double norm2_rhs[CORRS_NUM], norm2_rhs_l[CORRS_NUM];
    for (i = 0; i < CORRS_NUM; ++i)
    {
        VecNorm(rhs[i], NORM_2, &norm2_rhs[i]);
        VecNorm(rhs_l[i], NORM_2, &norm2_rhs_l[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "rhs[0]=%.5e, rhs[1]=%.5e, rhs[2]=%.5e.\n", norm2_rhs[0], norm2_rhs[1], norm2_rhs[2]);
    PetscPrintf(PETSC_COMM_WORLD, "rhs_l[0]=%.5e, rhs_l[1]=%.5e, rhs_l[2]=%.5e.\n", norm2_rhs_l[0], norm2_rhs_l[1], norm2_rhs_l[2]);
    */
    /* Solve the linear systems */
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, Amat, Amat);
    CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, TOL, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT);
    CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);
    CHKERRQ(ierr);
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = KSPSolve(ksp, rhs[i], self->corrs[i]);
        CHKERRQ(ierr);
    }
    /*
    double norm2_corrs[CORRS_NUM];
    for (i = 0; i < CORRS_NUM; ++i)
    {
        VecNorm(self->corrs[i], NORM_2, &norm2_corrs[i]);
    }
    PetscPrintf(PETSC_COMM_WORLD, "corr[0]=%.5e, corr[1]=%.5e, corr[2]=%.5e.\n", norm2_corrs[0], norm2_corrs[1], norm2_corrs[2]);
    */

    /* Calculate the homogenized coefficients. */
    for (elem_ind_y = 0; elem_ind_y < self->ctx.grids_on_cell; ++elem_ind_y)
        for (elem_ind_x = 0; elem_ind_x < self->ctx.grids_on_cell; ++elem_ind_x)
        {
            elem_ind = elem_ind_y * self->ctx.grids_on_cell + elem_ind_x;
            C = &self->ctx.cff_data[elem_ind * CFF_LEN];
            for (i = 0; i < CFF_LEN; ++i)
                self->C_eff[i] += self->ctx.hh * self->ctx.hh * C[i];
        }
    PetscScalar dot_val = 0.0;
    VecTDot(rhs[0], self->corrs[0], &dot_val);
    self->C_eff[0] -= dot_val;
    VecTDot(rhs[1], self->corrs[1], &dot_val);
    self->C_eff[1] -= dot_val;
    VecTDot(rhs[2], self->corrs[2], &dot_val);
    self->C_eff[2] -= dot_val;
    VecTDot(rhs[1], self->corrs[2], &dot_val);
    self->C_eff[3] -= dot_val;
    VecTDot(rhs[0], self->corrs[2], &dot_val);
    self->C_eff[4] -= dot_val;
    VecTDot(rhs[0], self->corrs[1], &dot_val);
    self->C_eff[5] -= dot_val;
    /* Free memory */
    ierr = KSPDestroy(&ksp);
    /*
    ierr = MatNullSpaceDestroy(&msp);
    ierr = VecDestroy(&nullvec[0]);
    ierr = VecDestroy(&nullvec[1]);
    */
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = VecDestroy(&rhs[i]);
        ierr = VecDestroy(&rhs_l[i]);
    }
    ierr = MatDestroy(&Amat);
    CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode _d_Homogenization_solve(const Context ctx, Vec **corrs, PetscScalar *C_eff[])
{
    PetscErrorCode ierr;
    DM dm;
    ierr = DMDACreate2d(PETSC_COMM_SELF, DM_BOUNDARY_PERIODIC, DM_BOUNDARY_PERIODIC, DMDA_STENCIL_BOX, ctx.grids_on_cell, ctx.grids_on_cell, PETSC_DECIDE, PETSC_DECIDE, DIM, 1, NULL, NULL, &dm);
    CHKERRQ(ierr);
    /* Set up DM first! */
    ierr = DMSetUp(dm);
    CHKERRQ(ierr);
    Mat Amat;
    ierr = DMCreateMatrix(dm, &Amat);
    CHKERRQ(ierr);
    Vec rhs[CORRS_NUM], rhs_l[CORRS_NUM];
    int i;
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = DMCreateGlobalVector(dm, &rhs[i]);
        CHKERRQ(ierr);
        ierr = DMCreateLocalVector(dm, &rhs_l[i]);
        CHKERRQ(ierr);
    }
    /* Initialize the vectors [corrs] before solving linear systems! */
    ierr = VecDuplicateVecs(rhs[0], CORRS_NUM, corrs);
    CHKERRQ(ierr);
    PetscInt elem_ind_x_start, elem_ind_y_start, elem_ind_x_len, elem_ind_y_len, elem_ind_x, elem_ind_y, elem_ind;
    ierr = DMDAGetCorners(dm, &elem_ind_x_start, &elem_ind_y_start, NULL, &elem_ind_x_len, &elem_ind_y_len, NULL);
    CHKERRQ(ierr);
    MatStencil row[LOC_FDMS], col[LOC_FDMS];
    /*
    PetscScalar ***array0, ***array1, ***array2;
    ierr = DMDAVecGetArrayDOF(dm, rhs_l[0], &array0);
    ierr = DMDAVecGetArrayDOF(dm, rhs_l[1], &array1);
    ierr = DMDAVecGetArrayDOF(dm, rhs_l[2], &array2);
    */
    PetscScalar ***array[CORRS_NUM];
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = DMDAVecGetArrayDOF(dm, rhs_l[i], &array[i]);
        CHKERRQ(ierr);
    }
    CHKERRQ(ierr);
    /* Construct the linear systems. */
    double *C, C_ext[DIM][DIM][DIM][DIM];
    int loc_nd_ind_row, loc_nd_ind_col, quad_ind, val_ind;
    PetscInt row_nd_ind_x, row_nd_ind_y, col_nd_ind_x, col_nd_ind_y;
    PetscScalar values[LOC_FDMS * LOC_FDMS];
    for (elem_ind_y = elem_ind_y_start; elem_ind_y < elem_ind_y_start + elem_ind_y_len; ++elem_ind_y)
        for (elem_ind_x = elem_ind_x_start; elem_ind_x < elem_ind_x_start + elem_ind_x_len; ++elem_ind_x)
        {
            elem_ind = elem_ind_y * ctx.grids_on_cell + elem_ind_x;
            C = &ctx.cff_data[elem_ind * CFF_LEN];
            C_ext[0][0][0][0] = C[0]; /* C_1111 = C11 */
            C_ext[0][0][0][1] = C[4]; /* C_1112 = C13 */
            C_ext[0][0][1][0] = C[4]; /* C_1121 = C_1112 = C13 */
            C_ext[0][0][1][1] = C[5]; /* C_1122 = C12 */
            C_ext[0][1][0][0] = C[4]; /* C_1211 = C_1112 = C13 */
            C_ext[0][1][0][1] = C[2]; /* C_1212 = C33 */
            C_ext[0][1][1][0] = C[2]; /* C_1221 = C_1212 = C33 */
            C_ext[0][1][1][1] = C[3]; /* C_1222 = C2212 = C23 */

            C_ext[1][0][0][0] = C[4]; /* C_2111 = C1112 = C13 */
            C_ext[1][0][0][1] = C[2]; /* C_2112 = C1212 = C33 */
            C_ext[1][0][1][0] = C[2]; /* C_2121 = C1212 = C33 */
            C_ext[1][0][1][1] = C[3]; /* C_2122 = C2212 = C23 */
            C_ext[1][1][0][0] = C[5]; /* C_2211 = C1122 = C12 */
            C_ext[1][1][0][1] = C[3]; /* C_2212 = C23 */
            C_ext[1][1][1][0] = C[3]; /* C_2221 = C_2212 = C23 */
            C_ext[1][1][1][1] = C[1]; /* C_2222 = C22 */

            PetscMemzero(&values[0], sizeof(values));
            for (loc_nd_ind_row = 0; loc_nd_ind_row < LOC_NDS; ++loc_nd_ind_row)
            {
                row_nd_ind_x = (elem_ind_x + loc_nd_ind_row % 2) % ctx.grids_on_cell;
                row_nd_ind_y = (elem_ind_y + loc_nd_ind_row / 2) % ctx.grids_on_cell;
                for (quad_ind = 0; quad_ind < QUAD_PNTS; ++quad_ind)
                {
                    array[0][0][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[0][0][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array[0][0][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[1][0][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];
                    array[1][0][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[0][0][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array[1][0][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[1][0][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];
                    array[2][0][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[0][0][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array[2][0][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[1][0][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];

                    array[0][1][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[0][1][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array[0][1][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[1][1][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];
                    array[1][1][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[0][1][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array[1][1][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[1][1][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];
                    array[2][1][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[0][1][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * quad_wghts[quad_ind];
                    array[2][1][row_nd_ind_y][row_nd_ind_x] -= 0.5 * ctx.hh * C_ext[1][1][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * quad_wghts[quad_ind];
                    /*  
                        fdm_m = m * XY  # m'=0
                        loc_rhs[0, fdm_m] += C_ext[0, 0, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[0, fdm_m] += C_ext[1, 0, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)
                        loc_rhs[1, fdm_m] += C_ext[0, 0, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[1, fdm_m] += C_ext[1, 0, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)
                        loc_rhs[2, fdm_m] += C_ext[0, 0, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[2, fdm_m] += C_ext[1, 0, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)

                        fdm_m = m * XY + 1  # m'=1
                        loc_rhs[0, fdm_m] += C_ext[0, 1, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[0, fdm_m] += C_ext[1, 1, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)
                        loc_rhs[1, fdm_m] += C_ext[0, 1, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[1, fdm_m] += C_ext[1, 1, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)
                        loc_rhs[2, fdm_m] += C_ext[0, 1, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :], quad_wghts)
                        loc_rhs[2, fdm_m] += C_ext[1, 1, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :], quad_wghts)
                     */
                }
                row[loc_nd_ind_row * DIM].i = row_nd_ind_x;
                row[loc_nd_ind_row * DIM].j = row_nd_ind_y;
                row[loc_nd_ind_row * DIM].c = 0;
                row[loc_nd_ind_row * DIM + 1].i = row_nd_ind_x;
                row[loc_nd_ind_row * DIM + 1].j = row_nd_ind_y;
                row[loc_nd_ind_row * DIM + 1].c = 1;

                for (loc_nd_ind_col = 0; loc_nd_ind_col < LOC_NDS; ++loc_nd_ind_col)
                {
                    col_nd_ind_x = (elem_ind_x + loc_nd_ind_col % 2) % ctx.grids_on_cell;
                    col_nd_ind_y = (elem_ind_y + loc_nd_ind_col / 2) % ctx.grids_on_cell;
                    col[loc_nd_ind_col * DIM].i = col_nd_ind_x;
                    col[loc_nd_ind_col * DIM].j = col_nd_ind_y;
                    col[loc_nd_ind_col * DIM].c = 0;
                    col[loc_nd_ind_col * DIM + 1].i = col_nd_ind_x;
                    col[loc_nd_ind_col * DIM + 1].j = col_nd_ind_y;
                    col[loc_nd_ind_col * DIM + 1].c = 1;
                    /*
                        fdm_m, fdm_n = m * XY, n * XY  # m'=0, n'=0
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 0, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 0, 1, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 0, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 0, 1, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        fdm_m, fdm_n = m * XY, n * XY + 1  # m'=0, n'=1
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 1, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 1, 1, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 1, 0, 0] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 1, 1, 0] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        fdm_m, fdm_n = m * XY + 1, n * XY  # m'=1, n'=0
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 0, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 0, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 0, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 0, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        fdm_m, fdm_n = m * XY + 1, n * XY + 1  #m'=1, n'=1
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 1, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[0, 1, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 0, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 1, 0, 1] * np.dot(base_grad_val_at_quad_pnt[m, 0, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                        loc_Amat[fdm_m, fdm_n] += C_ext[1, 1, 1, 1] * np.dot(base_grad_val_at_quad_pnt[m, 1, :] * base_grad_val_at_quad_pnt[n, 1, :], quad_wghts)
                    */
                    for (quad_ind = 0; quad_ind < QUAD_PNTS; ++quad_ind)
                    {
                        val_ind = DIM * loc_nd_ind_row * LOC_FDMS + DIM * loc_nd_ind_col;
                        values[val_ind] += C_ext[0][0][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[0][0][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][0][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][0][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        val_ind = DIM * loc_nd_ind_row * LOC_FDMS + DIM * loc_nd_ind_col + 1;
                        values[val_ind] += C_ext[0][1][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[0][1][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][1][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][1][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        val_ind = (DIM * loc_nd_ind_row + 1) * LOC_FDMS + DIM * loc_nd_ind_col;
                        values[val_ind] += C_ext[0][0][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[0][0][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][0][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][0][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        val_ind = (DIM * loc_nd_ind_row + 1) * LOC_FDMS + DIM * loc_nd_ind_col + 1;
                        values[val_ind] += C_ext[0][1][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[0][1][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][1][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        values[val_ind] += C_ext[1][1][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                    }
                }
            }
            ierr = MatSetValuesStencil(Amat, LOC_FDMS, row, LOC_FDMS, col, values, ADD_VALUES);
            CHKERRQ(ierr);
        }
    /*
    ierr = DMDAVecRestoreArrayDOF(dm, rhs_l[0], &array[0]);
    ierr = DMDAVecRestoreArrayDOF(dm, rhs_l[1], &array[1]);
    ierr = DMDAVecRestoreArrayDOF(dm, rhs_l[2], &array[2]);
    */
    CHKERRQ(ierr);
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = DMDAVecRestoreArrayDOF(dm, rhs_l[i], &array[i]);
        CHKERRQ(ierr);
        ierr = DMLocalToGlobal(dm, rhs_l[i], ADD_VALUES, rhs[i]);
        CHKERRQ(ierr);
        ierr = VecAssemblyBegin(rhs[i]);
        CHKERRQ(ierr);
        ierr = VecAssemblyEnd(rhs[i]);
        CHKERRQ(ierr);
    }
    ierr = MatAssemblyBegin(Amat, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);

    /* Set NullSpace of Amat */
    Vec nullvec[2];
    ierr = DMCreateGlobalVector(dm, &nullvec[0]);
    ierr = DMCreateGlobalVector(dm, &nullvec[1]);
    PetscScalar ***nullvec_array[2];
    ierr = DMDAVecGetArrayDOF(dm, nullvec[0], &nullvec_array[0]);
    ierr = DMDAVecGetArrayDOF(dm, nullvec[1], &nullvec_array[1]);
    CHKERRQ(ierr);
    for (elem_ind_y = 0; elem_ind_y < ctx.grids_on_cell; ++elem_ind_y)
        for (elem_ind_x = 0; elem_ind_x < ctx.grids_on_cell; ++elem_ind_x)
        {
            nullvec_array[0][0][elem_ind_y][elem_ind_x] = 1.0;
            nullvec_array[0][1][elem_ind_y][elem_ind_x] = 0.0;
            nullvec_array[1][0][elem_ind_y][elem_ind_x] = 0.0;
            nullvec_array[1][1][elem_ind_y][elem_ind_x] = 1.0;
        }
    ierr = DMDAVecRestoreArrayDOF(dm, nullvec[0], &nullvec_array[0]);
    ierr = DMDAVecRestoreArrayDOF(dm, nullvec[1], &nullvec_array[1]);
    CHKERRQ(ierr);
    ierr = VecAssemblyBegin(nullvec[0]);
    ierr = VecAssemblyEnd(nullvec[0]);
    ierr = VecAssemblyBegin(nullvec[1]);
    ierr = VecAssemblyEnd(nullvec[1]);
    CHKERRQ(ierr);
    ierr = VecNormalize(nullvec[0], NULL);
    ierr = VecNormalize(nullvec[1], NULL);
    CHKERRQ(ierr);
    MatNullSpace msp;
    ierr = MatNullSpaceCreate(PETSC_COMM_SELF, PETSC_FALSE, 2, nullvec, &msp);
    CHKERRQ(ierr);
    ierr = MatSetNullSpace(Amat, msp);
    CHKERRQ(ierr);
    /* Solve the linear systems */
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_SELF, &ksp);
    CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, Amat, Amat);
    CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp);
    CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);
    CHKERRQ(ierr);
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = KSPSolve(ksp, rhs[i], (*corrs)[i]);
        CHKERRQ(ierr);
    }
    /* Calculate the homogenized coefficients. */
    *C_eff = (double *)malloc(CFF_LEN * sizeof(double));
    PetscMemzero(*C_eff, CFF_LEN * sizeof(double));
    for (elem_ind_y = elem_ind_y_start; elem_ind_y < elem_ind_y_start + elem_ind_y_len; ++elem_ind_y)
        for (elem_ind_x = elem_ind_x_start; elem_ind_x < elem_ind_x_start + elem_ind_x_len; ++elem_ind_x)
        {
            elem_ind = elem_ind_y * ctx.grids_on_cell + elem_ind_x;
            C = &ctx.cff_data[elem_ind * CFF_LEN];
            for (i = 0; i < CFF_LEN; ++i)
                (*C_eff)[i] += ctx.hh * ctx.hh * C[i];
        }
    PetscScalar dot_val;
    VecTDot(rhs[0], (*corrs)[0], &dot_val);
    (*C_eff)[0] -= dot_val;
    VecTDot(rhs[1], (*corrs)[1], &dot_val);
    (*C_eff)[1] -= dot_val;
    VecTDot(rhs[2], (*corrs)[2], &dot_val);
    (*C_eff)[2] -= dot_val;
    VecTDot(rhs[1], (*corrs)[2], &dot_val);
    (*C_eff)[3] -= dot_val;
    VecTDot(rhs[0], (*corrs)[2], &dot_val);
    (*C_eff)[4] -= dot_val;
    VecTDot(rhs[0], (*corrs)[1], &dot_val);
    (*C_eff)[5] -= dot_val;
    /* Free memory */
    ierr = KSPDestroy(&ksp);
    ierr = MatNullSpaceDestroy(&msp);
    ierr = VecDestroy(&nullvec[0]);
    ierr = VecDestroy(&nullvec[1]);
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = VecDestroy(&rhs[i]);
        ierr = VecDestroy(&rhs_l[i]);
    }
    ierr = MatDestroy(&Amat);
    ierr = DMDestroy(&dm);
    CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode Homogenization_final_(Homogenization *self)
{
    PetscErrorCode ierr;
    Context_final_(&(self->ctx));
    ierr = DMDestroy(&(self->cell_dm));
    CHKERRQ(ierr);
    int i;
    for (i = 0; i < CORRS_NUM; ++i)
    {
        ierr = VecDestroy(&(self->corrs[i]));
        CHKERRQ(ierr);
    }
    return ierr;
}

int _p_Homogenization_main(int argc, char *argv[])
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscErrorCode ierr;
    Homogenization homo;
    Homogenization_init_(&homo, 64, default_cell_cff, NULL);
    /*PetscPrintf(PETSC_COMM_WORLD, "h=1/%d\n", homo.ctx.grids_on_cell); */
    ierr = Homogenization_solve(&homo);
    CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "C_eff[1111]=%.5e, C_eff[2222]=%.5e, C_eff[1212]=%.5e, C_eff[1222]=%.5e, C_eff[1211]=%.5e, C_eff[1122]=%.5e.\n", homo.C_eff[0], homo.C_eff[1], homo.C_eff[2], homo.C_eff[3], homo.C_eff[4], homo.C_eff[5]);
    Homogenization_final_(&homo);
    PetscFinalize();
    return 0;
}