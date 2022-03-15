from Context import *
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg

CORRS_NUM = 3


class Homogenization(Context):
    def __init__(self, prd, grids_on_cell, cell_cff=default_cell_cff):
        super().__init__(prd, grids_on_cell, default_cell_cff)

    def get_loc_inds(self, elem_ind_x, ele_ind_y):
        inds = np.zeros((LOC_FDMS, ), dtype=np.int32)
        nd_ind_x, nd_ind_y = elem_ind_x, ele_ind_y
        nd_ind = nd_ind_y * self.grids_on_cell + nd_ind_x
        inds[0] = nd_ind * XY
        inds[1] = nd_ind * XY + 1

        nd_ind_x, nd_ind_y = (elem_ind_x + 1) % self.grids_on_cell, ele_ind_y
        nd_ind = nd_ind_y * self.grids_on_cell + nd_ind_x
        inds[2] = nd_ind * XY
        inds[3] = nd_ind * XY + 1

        nd_ind_x, nd_ind_y = elem_ind_x, (ele_ind_y + 1) % self.grids_on_cell
        nd_ind = nd_ind_y * self.grids_on_cell + nd_ind_x
        inds[4] = nd_ind * XY
        inds[5] = nd_ind * XY + 1

        nd_ind_x, nd_ind_y = (elem_ind_x + 1) % self.grids_on_cell, (ele_ind_y + 1) % self.grids_on_cell
        nd_ind = nd_ind_y * self.grids_on_cell + nd_ind_x
        inds[6] = nd_ind * XY
        inds[7] = nd_ind * XY + 1

        inds_Amat = np.tile(inds, (LOC_FDMS, 1))
        inds_Amat_col = inds_Amat.flatten()
        inds_Amat_row = inds_Amat.T.flatten()
        return inds, inds_Amat_col, inds_Amat_row

    def get_loc_rhs(self, elem_ind_x, elem_ind_y):
        C = self.cff_data[elem_ind_x, elem_ind_y, :]
        C_ext = np.zeros((DIM, DIM, DIM, DIM))
        C_ext[0, 0, 0, 0] = C[0]  # C_1111 = C11
        C_ext[0, 0, 0, 1] = C[4]  # C_1112 = C13
        C_ext[0, 0, 1, 0] = C[4]  # C_1121 = C_1112 = C13
        C_ext[0, 0, 1, 1] = C[5]  # C_1122 = C12
        C_ext[0, 1, 0, 0] = C[4]  # C_1211 = C_1112 = C13
        C_ext[0, 1, 0, 1] = C[2]  # C_1212 = C33
        C_ext[0, 1, 1, 0] = C[2]  # C_1221 = C_1212 = C33
        C_ext[0, 1, 1, 1] = C[3]  # C_1222 = C2212 = C23

        C_ext[1, 0, 0, 0] = C[4]  # C_2111 = C1112 = C13
        C_ext[1, 0, 0, 1] = C[2]  # C_2112 = C1212 = C33
        C_ext[1, 0, 1, 0] = C[2]  # C_2121 = C1212 = C33
        C_ext[1, 0, 1, 1] = C[3]  # C_2122 = C2212 = C23
        C_ext[1, 1, 0, 0] = C[5]  # C_2211 = C1122 = C12
        C_ext[1, 1, 0, 1] = C[3]  # C_2212 = C23
        C_ext[1, 1, 1, 0] = C[3]  # C_2221 = C_2212 = C23
        C_ext[1, 1, 1, 1] = C[1]  # C_2222 = C22

        loc_rhs = np.zeros((CORRS_NUM, LOC_FDMS))
        for m in range(LOC_NDS):
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
        return -0.5 * self.hh * loc_rhs

    def solve(self):
        max_data_len = self.total_elems_on_cell * LOC_FDMS**2
        Amat_row = np.zeros((max_data_len, ), dtype=np.int32)
        Amat_col = np.zeros((max_data_len, ), dtype=np.int32)
        Amat_data = np.zeros((max_data_len, ))
        rhs = np.zeros((CORRS_NUM, self.total_fdms_on_cell))
        for elem_ind in range(self.total_elems_on_cell):
            elem_ind_y, elem_ind_x = divmod(elem_ind, self.grids_on_cell)
            loc_inds, loc_inds_Amat_col, loc_inds_Amat_row = self.get_loc_inds(elem_ind_x, elem_ind_y)
            loc_Amat = get_loc_Amat(self.cff_data[elem_ind_x, elem_ind_y, :])
            loc_Amat_data = loc_Amat.flatten()
            Amat_col[elem_ind * LOC_FDMS**2:(elem_ind + 1) * LOC_FDMS**2] = loc_inds_Amat_col
            Amat_row[elem_ind * LOC_FDMS**2:(elem_ind + 1) * LOC_FDMS**2] = loc_inds_Amat_row
            Amat_data[elem_ind * LOC_FDMS**2:(elem_ind + 1) * LOC_FDMS**2] = loc_Amat_data
            loc_rhs = self.get_loc_rhs(elem_ind_x, elem_ind_y)
            np.add.at(rhs[0, :], loc_inds, loc_rhs[0, :])
            np.add.at(rhs[1, :], loc_inds, loc_rhs[1, :])
            np.add.at(rhs[2, :], loc_inds, loc_rhs[2, :])
        Amat_coo = coo_matrix((Amat_data, (Amat_row, Amat_col)), shape=(self.total_fdms_on_cell, self.total_fdms_on_cell))
        Amat_csr = Amat_coo.tocsr()

        corrs = np.zeros(rhs.shape)
        x0 = np.zeros((self.total_fdms_on_cell))
        sol, info = cg(Amat_csr, rhs[0, :], x0=x0, tol=TOL)
        assert info == 0
        corrs[0, :] = sol

        sol, info = cg(Amat_csr, rhs[1, :], x0=x0, tol=TOL)
        assert info == 0
        corrs[1, :] = sol

        sol, info = cg(Amat_csr, rhs[2, :], x0=x0, tol=TOL)
        assert info == 0
        corrs[2, :] = sol

        C_eff = np.zeros((CFF_LEN, ))
        C_eff[0] = self.hh**DIM * np.sum(self.cff_data[:, :, 0]) - np.dot(rhs[0, :], corrs[0, :])
        C_eff[1] = self.hh**DIM * np.sum(self.cff_data[:, :, 1]) - np.dot(rhs[1, :], corrs[1, :])
        C_eff[2] = self.hh**DIM * np.sum(self.cff_data[:, :, 2]) - np.dot(rhs[2, :], corrs[2, :])
        C_eff[3] = self.hh**DIM * np.sum(self.cff_data[:, :, 3]) - np.dot(rhs[1, :], corrs[2, :])
        C_eff[4] = self.hh**DIM * np.sum(self.cff_data[:, :, 4]) - np.dot(rhs[0, :], corrs[2, :])
        C_eff[5] = self.hh**DIM * np.sum(self.cff_data[:, :, 5]) - np.dot(rhs[0, :], corrs[1, :])
        return corrs, C_eff


if __name__ == "__main__":
    ctx = Context(4, 32)
    homo_pro = Homogenization(ctx)
    corrs, C_eff = homo_pro.solve()
    print(C_eff)
