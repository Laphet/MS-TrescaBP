#include "TrescaBP.h"

const unsigned int MAX_ITERS = 1000000;
const double BT_BETA = 0.5;
const double BT_TZERO = 1.0;

PetscBool _p_TrescaBP_is_clamped_fdm(TrescaBP *self, MatStencil fdm)
{
    if (fdm.j == 0 && fdm.c == 1)
        return PETSC_TRUE;
    else if (fdm.i == self->ctx.grids_on_dmn)
        return PETSC_TRUE;
    else
        return PETSC_FALSE;
}

PetscErrorCode TrescaBP_init_(TrescaBP *self, unsigned int prd, unsigned int grids_on_cell, cell_cff_func cell_cff, void *paras)
{
    PetscErrorCode ierr;
    Context_init_(&self->ctx, prd, grids_on_cell, cell_cff, paras);
    /* Create DM object on the whole domain. */
    ierr = DMDACreate2d(PETSC_COMM_WORLD, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DMDA_STENCIL_BOX, self->ctx.grids_on_dmn + 1, self->ctx.grids_on_dmn + 1, PETSC_DECIDE, PETSC_DECIDE, DIM, 1, NULL, NULL, &self->dmn_dm);
    CHKERRQ(ierr);
    /* Set up DM first! */
    ierr = DMSetUp(self->dmn_dm);
    CHKERRQ(ierr);
    /* Initialize u_init. */
    ierr = DMCreateGlobalVector(self->dmn_dm, &self->u_init);
    ierr = VecZeroEntries(self->u_init);
    /* Initialize Amat. */
    ierr = DMCreateMatrix(self->dmn_dm, &self->Amat);
    return ierr;
}

PetscErrorCode TrescaBP_set_conds(TrescaBP *self, unsigned int grids_on_time, double T, s2d_t1d_to_2d_func bdy_f, s1d_t1d_to_2d_func tr_f1, s1d_t1d_to_2d_func tr_f2, double Tresca_bnd)
{
    self->grids_on_time = grids_on_time;
    self->T = T;
    self->tau = T / grids_on_time;
    self->bdy_f = bdy_f;
    self->tr_f1 = tr_f1;
    self->tr_f2 = tr_f2;
    self->Tresca_bnd = Tresca_bnd;

    PetscErrorCode ierr;

    int elem_ind_x_start, elem_ind_x_end, nd_ind_x_len, elem_ind_x, elem_ind_y_start, elem_ind_y_end, nd_ind_y_len, elem_ind_y, sub_elem_ind, row_nd_ind_x, row_nd_ind_y, col_nd_ind_x, col_nd_ind_y;
    ierr = DMDAGetCorners(self->dmn_dm, &elem_ind_x_start, &elem_ind_y_start, NULL, &nd_ind_x_len, &nd_ind_y_len, NULL);
    CHKERRQ(ierr);
    elem_ind_x_end = elem_ind_x_start + nd_ind_x_len < self->ctx.grids_on_dmn + 1 ? elem_ind_x_start + nd_ind_x_len : self->ctx.grids_on_dmn;
    elem_ind_y_end = elem_ind_y_start + nd_ind_y_len < self->ctx.grids_on_dmn + 1 ? elem_ind_y_start + nd_ind_y_len : self->ctx.grids_on_dmn;
    MatStencil row[LOC_FDMS], col[LOC_FDMS];
    PetscScalar values[LOC_FDMS * LOC_FDMS];
    double *C, C_ext[DIM][DIM][DIM][DIM];
    unsigned short loc_nd_ind_row, loc_nd_ind_col, quad_ind, val_ind;

    for (elem_ind_y = elem_ind_y_start; elem_ind_y < elem_ind_y_end; ++elem_ind_y)
        for (elem_ind_x = elem_ind_x_start; elem_ind_x < elem_ind_x_end; ++elem_ind_x)
        {
            sub_elem_ind = (elem_ind_y % self->ctx.grids_on_cell) * self->ctx.grids_on_cell + (elem_ind_x % self->ctx.grids_on_cell);
            C = &self->ctx.cff_data[sub_elem_ind * CFF_LEN];
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
                row_nd_ind_x = elem_ind_x + loc_nd_ind_row % 2;
                row_nd_ind_y = elem_ind_y + loc_nd_ind_row / 2;
                row[loc_nd_ind_row * DIM].i = row_nd_ind_x;
                row[loc_nd_ind_row * DIM].j = row_nd_ind_y;
                row[loc_nd_ind_row * DIM].c = 0;
                row[loc_nd_ind_row * DIM + 1].i = row_nd_ind_x;
                row[loc_nd_ind_row * DIM + 1].j = row_nd_ind_y;
                row[loc_nd_ind_row * DIM + 1].c = 1;
                for (loc_nd_ind_col = 0; loc_nd_ind_col < LOC_NDS; ++loc_nd_ind_col)
                {
                    col_nd_ind_x = elem_ind_x + loc_nd_ind_col % 2;
                    col_nd_ind_y = elem_ind_y + loc_nd_ind_col / 2;
                    col[loc_nd_ind_col * DIM].i = col_nd_ind_x;
                    col[loc_nd_ind_col * DIM].j = col_nd_ind_y;
                    col[loc_nd_ind_col * DIM].c = 0;
                    col[loc_nd_ind_col * DIM + 1].i = col_nd_ind_x;
                    col[loc_nd_ind_col * DIM + 1].j = col_nd_ind_y;
                    col[loc_nd_ind_col * DIM + 1].c = 1;
                    for (quad_ind = 0; quad_ind < QUAD_PNTS; ++quad_ind)
                    {
                        val_ind = DIM * loc_nd_ind_row * LOC_FDMS + DIM * loc_nd_ind_col;
                        if (!_p_TrescaBP_is_clamped_fdm(self, row[DIM * loc_nd_ind_row]) && !_p_TrescaBP_is_clamped_fdm(self, col[DIM * loc_nd_ind_col]))
                        {
                            values[val_ind] += C_ext[0][0][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[0][0][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[1][0][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[1][0][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        }
                        val_ind = DIM * loc_nd_ind_row * LOC_FDMS + DIM * loc_nd_ind_col + 1;
                        if (!_p_TrescaBP_is_clamped_fdm(self, row[DIM * loc_nd_ind_row]) && !_p_TrescaBP_is_clamped_fdm(self, col[DIM * loc_nd_ind_col + 1]))
                        {
                            values[val_ind] += C_ext[0][1][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[0][1][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[1][1][0][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[1][1][1][0] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        }
                        val_ind = (DIM * loc_nd_ind_row + 1) * LOC_FDMS + DIM * loc_nd_ind_col;
                        if (!_p_TrescaBP_is_clamped_fdm(self, row[DIM * loc_nd_ind_row + 1]) && !_p_TrescaBP_is_clamped_fdm(self, col[DIM * loc_nd_ind_col]))
                        {
                            values[val_ind] += C_ext[0][0][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[0][0][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[1][0][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[1][0][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        }
                        val_ind = (DIM * loc_nd_ind_row + 1) * LOC_FDMS + DIM * loc_nd_ind_col + 1;
                        if (!_p_TrescaBP_is_clamped_fdm(self, row[DIM * loc_nd_ind_row + 1]) && !_p_TrescaBP_is_clamped_fdm(self, col[DIM * loc_nd_ind_col + 1]))
                        {
                            values[val_ind] += C_ext[0][1][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[0][1][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][0][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[1][1][0][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][0][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                            values[val_ind] += C_ext[1][1][1][1] * base_grad_val_at_quad_pnt[loc_nd_ind_row][1][quad_ind] * base_grad_val_at_quad_pnt[loc_nd_ind_col][1][quad_ind] * quad_wghts[quad_ind];
                        }
                    }
                }
            }
            ierr = MatSetValuesStencil(self->Amat, LOC_FDMS, row, LOC_FDMS, col, values, ADD_VALUES);
            CHKERRQ(ierr);
        }
    ierr = MatAssemblyBegin(self->Amat, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    ierr = MatAssemblyEnd(self->Amat, MAT_FINAL_ASSEMBLY);
    CHKERRQ(ierr);
    /* test
    PetscScalar norm_test1;
    ierr = MatNorm(self->Amat, NORM_1, &norm_test1);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Current norm of Amat=%.5e.\n", norm_test1);
    test end */
    return ierr;
}

void _p_TrescaBP_get_loc_rhs_bdy_f(TrescaBP *self, unsigned int elem_ind_x, unsigned int elem_ind_y, double t, double *val)
{
    unsigned short quad_ind, quad_ind_x, quad_ind_y;
    double x_center = ((double)elem_ind_x + 0.5) * self->ctx.h, y_center = ((double)elem_ind_y + 0.5) * self->ctx.h, x, y, h = self->ctx.h;
    double val_f[DIM];
    MatStencil fdm;
    for (quad_ind = 0; quad_ind < QUAD_PNTS; ++quad_ind)
    {
        quad_ind_x = quad_ind % QUAD_ORDER;
        quad_ind_y = quad_ind / QUAD_ORDER;
        x = x_center + 0.5 * self->ctx.h * QUAD_CORD[quad_ind_x];
        y = y_center + 0.5 * self->ctx.h * QUAD_CORD[quad_ind_y];
        self->bdy_f(x, y, t, &val_f[0]);
        fdm.i = elem_ind_x;
        fdm.j = elem_ind_y;
        fdm.c = 0;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[0] += 0.25 * h * h * val_f[0] * base_val_at_quad_pnt[0][quad_ind] * quad_wghts[quad_ind];
        fdm.i = elem_ind_x;
        fdm.j = elem_ind_y;
        fdm.c = 1;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[1] += 0.25 * h * h * val_f[1] * base_val_at_quad_pnt[0][quad_ind] * quad_wghts[quad_ind];
        fdm.i = elem_ind_x + 1;
        fdm.j = elem_ind_y;
        fdm.c = 0;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[2] += 0.25 * h * h * val_f[0] * base_val_at_quad_pnt[1][quad_ind] * quad_wghts[quad_ind];
        fdm.i = elem_ind_x + 1;
        fdm.j = elem_ind_y;
        fdm.c = 1;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[3] += 0.25 * h * h * val_f[1] * base_val_at_quad_pnt[1][quad_ind] * quad_wghts[quad_ind];
        fdm.i = elem_ind_x;
        fdm.j = elem_ind_y + 1;
        fdm.c = 0;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[4] += 0.25 * h * h * val_f[0] * base_val_at_quad_pnt[2][quad_ind] * quad_wghts[quad_ind];
        fdm.i = elem_ind_x;
        fdm.j = elem_ind_y + 1;
        fdm.c = 1;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[5] += 0.25 * h * h * val_f[1] * base_val_at_quad_pnt[2][quad_ind] * quad_wghts[quad_ind];
        fdm.i = elem_ind_x + 1;
        fdm.j = elem_ind_y + 1;
        fdm.c = 0;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[6] += 0.25 * h * h * val_f[0] * base_val_at_quad_pnt[3][quad_ind] * quad_wghts[quad_ind];
        fdm.i = elem_ind_x + 1;
        fdm.j = elem_ind_y + 1;
        fdm.c = 1;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[7] += 0.25 * h * h * val_f[1] * base_val_at_quad_pnt[3][quad_ind] * quad_wghts[quad_ind];
    }
}

void _p_TrescaBP_get_loc_rhs_tr_f1(TrescaBP *self, unsigned int elem_ind_x, unsigned int elem_ind_y, double t, double *val)
{
    if (elem_ind_y != self->ctx.grids_on_dmn - 1)
        return;
    unsigned short quad_ind_x;
    double x_center = ((double)elem_ind_x + 0.5) * self->ctx.h, x, h = self->ctx.h;
    double val_f1[DIM], val_base;
    MatStencil fdm;
    for (quad_ind_x = 0; quad_ind_x < QUAD_ORDER; ++quad_ind_x)
    {
        x = x_center + 0.5 * h * QUAD_CORD[quad_ind_x];
        self->tr_f1(x, t, &val_f1[0]);

        get_locbase_val(2, QUAD_CORD[quad_ind_x], 1.0, &val_base);
        fdm.i = elem_ind_x;
        fdm.j = elem_ind_y + 1;
        fdm.c = 0;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[4] += 0.5 * h * val_f1[0] * val_base * QUAD_WGHT[quad_ind_x];
        fdm.i = elem_ind_x;
        fdm.j = elem_ind_y + 1;
        fdm.c = 1;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[5] += 0.5 * h * val_f1[1] * val_base * QUAD_WGHT[quad_ind_x];

        get_locbase_val(3, QUAD_CORD[quad_ind_x], 1.0, &val_base);
        fdm.i = elem_ind_x + 1;
        fdm.j = elem_ind_y + 1;
        fdm.c = 0;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[6] += 0.5 * h * val_f1[0] * val_base * QUAD_WGHT[quad_ind_x];
        fdm.i = elem_ind_x + 1;
        fdm.j = elem_ind_y + 1;
        fdm.c = 1;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[7] += 0.5 * h * val_f1[1] * val_base * QUAD_WGHT[quad_ind_x];
    }
}

void _p_TrescaBP_get_loc_rhs_tr_f2(TrescaBP *self, unsigned int elem_ind_x, unsigned int elem_ind_y, double t, double *val)
{
    if (elem_ind_x != 0)
        return;
    unsigned short quad_ind_y;
    double y_center = ((double)elem_ind_y + 0.5) * self->ctx.h, y, h = self->ctx.h;
    double val_f2[DIM], val_base;
    MatStencil fdm;
    for (quad_ind_y = 0; quad_ind_y < QUAD_ORDER; ++quad_ind_y)
    {
        y = y_center + 0.5 * h * QUAD_CORD[quad_ind_y];
        self->tr_f2(y, t, &val_f2[0]);

        get_locbase_val(0, -1.0, QUAD_CORD[quad_ind_y], &val_base);
        fdm.i = elem_ind_x;
        fdm.j = elem_ind_y;
        fdm.c = 0;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[0] += 0.5 * h * val_f2[0] * val_base * QUAD_WGHT[quad_ind_y];
        fdm.i = elem_ind_x;
        fdm.j = elem_ind_y;
        fdm.c = 1;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[1] += 0.5 * h * val_f2[1] * val_base * QUAD_WGHT[quad_ind_y];

        get_locbase_val(2, -1.0, QUAD_CORD[quad_ind_y], &val_base);
        fdm.i = elem_ind_x;
        fdm.j = elem_ind_y + 1;
        fdm.c = 0;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[4] += 0.5 * h * val_f2[0] * val_base * QUAD_WGHT[quad_ind_y];
        fdm.i = elem_ind_x;
        fdm.j = elem_ind_y + 1;
        fdm.c = 1;
        if (!_p_TrescaBP_is_clamped_fdm(self, fdm))
            val[5] += 0.5 * h * val_f2[1] * val_base * QUAD_WGHT[quad_ind_y];
    }
}

PetscErrorCode TrescaBP_get_rhs(TrescaBP *self, Vec u_minus, double t, Vec rhs)
{
    PetscErrorCode ierr;
    int elem_ind_x_start, elem_ind_x_end, nd_ind_x_len, elem_ind_x, elem_ind_y_start, elem_ind_y_end, nd_ind_y_len, elem_ind_y;
    ierr = DMDAGetCorners(self->dmn_dm, &elem_ind_x_start, &elem_ind_y_start, NULL, &nd_ind_x_len, &nd_ind_y_len, NULL);
    CHKERRQ(ierr);
    Vec rhs_l, rhs_temp, neg_u_minus;
    ierr = DMCreateLocalVector(self->dmn_dm, &rhs_l);
    CHKERRQ(ierr);
    ierr = DMCreateGlobalVector(self->dmn_dm, &rhs_temp);
    CHKERRQ(ierr);
    ierr = VecDuplicate(self->u_init, &neg_u_minus);
    CHKERRQ(ierr);
    PetscScalar ***array;
    ierr = DMDAVecGetArrayDOF(self->dmn_dm, rhs_l, &array);
    CHKERRQ(ierr);
    elem_ind_x_end = elem_ind_x_start + nd_ind_x_len < self->ctx.grids_on_dmn + 1 ? elem_ind_x_start + nd_ind_x_len : self->ctx.grids_on_dmn;
    elem_ind_y_end = elem_ind_y_start + nd_ind_y_len < self->ctx.grids_on_dmn + 1 ? elem_ind_y_start + nd_ind_y_len : self->ctx.grids_on_dmn;
    PetscScalar val[LOC_FDMS];
    for (elem_ind_x = elem_ind_x_start; elem_ind_x < elem_ind_x_end; ++elem_ind_x)
        for (elem_ind_y = elem_ind_y_start; elem_ind_y < elem_ind_y_end; ++elem_ind_y)
        {
            ierr = PetscMemzero(&val[0], sizeof(val));
            CHKERRQ(ierr);
            _p_TrescaBP_get_loc_rhs_bdy_f(self, elem_ind_x, elem_ind_y, t, &val[0]);
            _p_TrescaBP_get_loc_rhs_tr_f1(self, elem_ind_x, elem_ind_y, t, &val[0]);
            _p_TrescaBP_get_loc_rhs_tr_f2(self, elem_ind_x, elem_ind_y, t, &val[0]);
            array[elem_ind_y][elem_ind_x][0] += val[0];
            array[elem_ind_y][elem_ind_x][1] += val[1];
            array[elem_ind_y][elem_ind_x + 1][0] += val[2];
            array[elem_ind_y][elem_ind_x + 1][1] += val[3];
            array[elem_ind_y + 1][elem_ind_x][0] += val[4];
            array[elem_ind_y + 1][elem_ind_x][1] += val[5];
            array[elem_ind_y + 1][elem_ind_x + 1][0] += val[6];
            array[elem_ind_y + 1][elem_ind_x + 1][1] += val[7];
        }
    ierr = DMDAVecRestoreArrayDOF(self->dmn_dm, rhs_l, &array);
    CHKERRQ(ierr);
    ierr = DMLocalToGlobal(self->dmn_dm, rhs_l, ADD_VALUES, rhs_temp);
    CHKERRQ(ierr);
    /* test
    PetscScalar norm_test0, norm_test1;
    ierr = VecNorm(rhs_temp, NORM_2, &norm_test0);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Current norm of rhs_temp=%.5e.\n", norm_test0);
    test end */

    /*
    ierr = VecZeroEntries(neg_u_minus);
    CHKERRQ(ierr);
    */
    ierr = VecAXPY(neg_u_minus, -1.0, u_minus);
    CHKERRQ(ierr);
    /* test
    ierr = VecNorm(neg_u_minus, NORM_2, &norm_test1);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Current time t=%.5f, curren norm of neg_u_minus=%.5e.\n", t, norm_test1);
    test end */
    ierr = MatMultAdd(self->Amat, neg_u_minus, rhs_temp, rhs);
    CHKERRQ(ierr);
    /* test
    ierr = MatNorm(self->Amat, NORM_1, &norm_test1);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Current time t=%.5f, curren norm of Amat=%.5e.\n", t, norm_test1);
    test end
    test
    PetscScalar norm_test0, norm_test1;
    ierr = VecNorm(rhs, NORM_2, &norm_test0);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Current norm of rhs=%.5e.\n", norm_test0);
    test end */
    ierr = VecDestroy(&rhs_l);
    ierr = VecDestroy(&rhs_temp);
    ierr = VecDestroy(&neg_u_minus);
    CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode _p_TrescaBP_get_grad_smooth_part(TrescaBP *self, Vec v, Vec rhs, Vec grad_smooth)
{
    PetscErrorCode ierr;
    ierr = MatMult(self->Amat, v, grad_smooth);
    CHKERRQ(ierr);
    ierr = VecAXPBY(grad_smooth, -1.0, self->tau, rhs);
    CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode _p_TrescaBP_get_prox_nonsmooth_part(TrescaBP *self, Vec v, PetscScalar s, Vec prox_nonsmooth)
{
    PetscErrorCode ierr;
    ierr = VecCopy(v, prox_nonsmooth);
    CHKERRQ(ierr);
    PetscScalar ***array;
    double h_HT = self->ctx.h * self->Tresca_bnd, w;
    ierr = DMDAVecGetArrayDOF(self->dmn_dm, prox_nonsmooth, &array);
    CHKERRQ(ierr);
    int nd_ind_x_start, nd_ind_x_len, nd_ind_y_start, nd_ind_y_len, nd_ind_x;
    ierr = DMDAGetCorners(self->dmn_dm, &nd_ind_x_start, &nd_ind_y_start, NULL, &nd_ind_x_len, &nd_ind_y_len, NULL);
    CHKERRQ(ierr);
    if (nd_ind_y_start == 0)
    {
        for (nd_ind_x = nd_ind_x_start; nd_ind_x < nd_ind_x_start + nd_ind_x_len; ++nd_ind_x)
        {
            w = nd_ind_x == 0 ? 0.5 * h_HT : h_HT;
            if (array[0][nd_ind_x][0] >= s * w)
                array[0][nd_ind_x][0] -= s * w;
            else if (array[0][nd_ind_x][0] <= -s * w)
                array[0][nd_ind_x][0] += s * w;
            else
                array[0][nd_ind_x][0] = 0.0;
        }
    }
    ierr = DMDAVecRestoreArrayDOF(self->dmn_dm, prox_nonsmooth, &array);
    CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode _p_TrescaBP_get_val_smooth_part(TrescaBP *self, Vec v, Vec rhs, PetscScalar *val)
{
    PetscErrorCode ierr;
    val[0] = 0.0;
    Vec v_temp; /* Need to destroy. */
    ierr = VecDuplicate(self->u_init, &v_temp);
    CHKERRQ(ierr);
    ierr = MatMult(self->Amat, v, v_temp);
    CHKERRQ(ierr);
    ierr = VecAXPBY(v_temp, -1.0, 0.5 * self->tau, rhs);
    CHKERRQ(ierr);
    ierr = VecTDot(v_temp, v, &val[0]);
    CHKERRQ(ierr);
    ierr = VecDestroy(&v_temp);
    CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode _p_TrescaBP_get_BT_line_search(TrescaBP *self, Vec grad_g, Vec v, PetscScalar s, Vec rhs, PetscScalar *s_plus, Vec v_plus)
{
    PetscErrorCode ierr;
    *s_plus = s;
    Vec v_temp; /* Need to destroy. */
    ierr = VecDuplicate(self->u_init, &v_temp);
    CHKERRQ(ierr);
    PetscScalar current_val, target_val, val_temp, target_val0;
    ierr = _p_TrescaBP_get_val_smooth_part(self, v, rhs, &target_val0);
    CHKERRQ(ierr);
    unsigned int ls_iter_ind;
    for (ls_iter_ind = 0; ls_iter_ind < MAX_ITERS; ++ls_iter_ind)
    {
        ierr = VecWAXPY(v_temp, -*s_plus, grad_g, v);
        CHKERRQ(ierr);
        /* test 
        ierr = VecNorm(grad_g, NORM_2, &norm_test1);
        CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Current norm of grad_g in (get_BT_linesearch) after destroy=%.5e.\n", norm_test1);
        test end */
        ierr = _p_TrescaBP_get_prox_nonsmooth_part(self, v_temp, *s_plus, v_plus);
        CHKERRQ(ierr);
        ierr = _p_TrescaBP_get_val_smooth_part(self, v_plus, rhs, &current_val);
        CHKERRQ(ierr);
        target_val = target_val0;
        ierr = VecWAXPY(v_temp, -1.0, v, v_plus);
        CHKERRQ(ierr);
        ierr = VecTDot(grad_g, v_temp, &val_temp);
        CHKERRQ(ierr);
        target_val += val_temp;
        ierr = VecNorm(v_temp, NORM_2, &val_temp);
        CHKERRQ(ierr);
        target_val += 0.5 / *s_plus * val_temp * val_temp;
        if (current_val > target_val)
            *s_plus = *s_plus * BT_BETA;
        else
            break;
    }
    VecDestroy(&v_temp);
    /* test 
    ierr = PetscPrintf(PETSC_COMM_WORLD, "    Line search iterations=%d, stepsize=%.5f.\n", ls_iter_ind, *s_plus);
    CHKERRQ(ierr);
    test end */

    return ierr;
}

PetscErrorCode TrescaBP_get_next_u(TrescaBP *self, Vec u_minus, Vec rhs, Vec u, int *info)
{
    PetscErrorCode ierr;
    Vec v_temp, grad_g, v_minus, v, v_plus; /* Need to destroy. */

    ierr = VecDuplicate(self->u_init, &v_temp);
    /* ierr = VecZeroEntries(v_temp); */
    ierr = VecDuplicate(self->u_init, &grad_g);
    /* ierr = VecZeroEntries(grad_g); */
    ierr = VecDuplicate(self->u_init, &v_minus);
    /* ierr = VecZeroEntries(v_minus); */
    ierr = VecDuplicate(self->u_init, &v);
    /* ierr = VecZeroEntries(v); */
    ierr = VecDuplicate(self->u_init, &v_plus);
    /* ierr = VecZeroEntries(v_plus); */
    CHKERRQ(ierr);

    PetscScalar s = BT_TZERO, s_plus, v_norm_diff, v_norm;
    unsigned int iter_ind;
    double nestrov_rate;
    for (iter_ind = 1; iter_ind <= MAX_ITERS; ++iter_ind)
    {
        nestrov_rate = ((double)iter_ind - 2.0) / ((double)iter_ind + 1.0);
        ierr = VecWAXPY(v_temp, -nestrov_rate, v_minus, v);
        CHKERRQ(ierr);
        ierr = VecAXPY(v_temp, nestrov_rate, v);
        CHKERRQ(ierr);
        ierr = _p_TrescaBP_get_grad_smooth_part(self, v_temp, rhs, grad_g);
        ierr = _p_TrescaBP_get_BT_line_search(self, grad_g, v_temp, s, rhs, &s_plus, v_plus);
        ierr = VecWAXPY(v_temp, -1.0, v, v_plus);
        CHKERRQ(ierr);
        ierr = VecNorm(v_temp, NORM_2, &v_norm_diff);
        CHKERRQ(ierr);
        ierr = VecNorm(v, NORM_2, &v_norm);
        if (v_norm_diff <= TOL * v_norm)
        {
            *info = iter_ind;
            break;
        }
        else if (iter_ind == MAX_ITERS)
        {
            *info = -1;
        }
        else
        {
            s = s_plus;
            /* v_minus = v; */
            ierr = VecCopy(v, v_minus);
            CHKERRQ(ierr);
            ierr = VecCopy(v_plus, v);
            /* v = v_plus; */
            CHKERRQ(ierr);
        }
    }
    ierr = VecWAXPY(u, self->tau, v_plus, u_minus);
    CHKERRQ(ierr);

    /* test  
    PetscScalar norm_test1;
    ierr = VecNorm(rhs, NORM_2, &norm_test1);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Current norm of rhs in (get_next_u) before destroy=%.5e.\n", norm_test1);
    test end */
    ierr = VecDestroy(&v_temp);
    ierr = VecDestroy(&grad_g);
    ierr = VecDestroy(&v_minus);
    ierr = VecDestroy(&v);
    ierr = VecDestroy(&v_plus);
    CHKERRQ(ierr);
    /* test  
    ierr = VecNorm(rhs, NORM_2, &norm_test1);
    CHKERRQ(ierr);
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Current norm of rhs in (get_next_u) after destroy=%.5e.\n", norm_test1);
    test end */
    /*
    ierr = PetscPrintf(PETSC_COMM_WORLD, "Get next u, current iteration number=%d.\n", *info);
    CHKERRQ(ierr);
    */
    return ierr;
}

PetscErrorCode TrescaBP_solve(TrescaBP *self, Vec u)
{
    PetscErrorCode ierr;
    ierr = VecCopy(self->u_init, u);
    CHKERRQ(ierr);
    Vec rhs, u_minus; /* Need to destroy. */
    ierr = VecDuplicate(self->u_init, &rhs);
    CHKERRQ(ierr);
    ierr = VecDuplicate(self->u_init, &u_minus);
    CHKERRQ(ierr);

    unsigned int time_ind;
    PetscScalar t;
    int info;
    /* test  
    PetscScalar norm_test0, norm_test1;
     test end */
    for (time_ind = 1; time_ind <= self->grids_on_time; ++time_ind)
    {
        t = (double)time_ind * self->tau;
        ierr = VecCopy(u, u_minus);
        CHKERRQ(ierr);
        ierr = VecZeroEntries(rhs);
        CHKERRQ(ierr);
        ierr = TrescaBP_get_rhs(self, u_minus, t, rhs);
        ierr = TrescaBP_get_next_u(self, u_minus, rhs, u, &info);
        /* test 
        ierr = VecNorm(u, NORM_2, &norm_test0);
        CHKERRQ(ierr);
        ierr = VecNorm(rhs, NORM_2, &norm_test1);
        CHKERRQ(ierr);
        ierr = PetscPrintf(PETSC_COMM_WORLD, "Current time t=%.5f, current norm of u=%.5e, curren norm of rhs=%.5e, current iteration number=%d.\n", t, norm_test0, norm_test1, info);
        test end */
    }
    ierr = VecDestroy(&rhs);
    ierr = VecDestroy(&u_minus);
    CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode TrescaBP_final_(TrescaBP *self)
{
    Context_final_(&self->ctx);
    PetscErrorCode ierr;
    ierr = DMDestroy(&self->dmn_dm);
    ierr = MatDestroy(&self->Amat);
    ierr = VecDestroy(&self->u_init);
    CHKERRQ(ierr);
    return ierr;
}

int _p_TrescaBP_main(int argc, char *argv[])
{
    PetscInitialize(&argc, &argv, NULL, NULL);
    PetscErrorCode ierr;
    TrescaBP tbp;
    ierr = TrescaBP_init_(&tbp, 8, 32, default_cell_cff, NULL);
    ierr = TrescaBP_set_conds(&tbp, 64, default_T, default_bdy_f, default_tr_f1, default_tr_f2, default_Tresca_bnd);
    Vec u, rhs;
    ierr = VecDuplicate(tbp.u_init, &u);
    ierr = VecDuplicate(tbp.u_init, &rhs);
    CHKERRQ(ierr);
    PetscScalar t = tbp.tau;
    int info;
    ierr = TrescaBP_get_rhs(&tbp, tbp.u_init, t, rhs);
    ierr = TrescaBP_get_next_u(&tbp, tbp.u_init, rhs, u, &info);
    ierr = VecDestroy(&u);
    ierr = VecDestroy(&rhs);
    CHKERRQ(ierr);
    ierr = TrescaBP_final_(&tbp);
    PetscFinalize();
    return 0;
}