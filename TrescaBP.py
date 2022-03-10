from Context import *
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse import csr_matrix

MAX_ITERS = 10000
MU = 1.0


def stepsize_theta(iter_ind: int):
    if iter_ind <= 7:
        return 1.0
    else:
        return 3.0 / (iter_ind + 1.0)


class TrescaBP(Context):
    def set_conds(self, grids_on_time, T=default_T, bdy_f=Zero_s2dto2d, tr_f1=Zero_s1dto2d, tr_f2=default_tr_f2, Tresca_bnd=default_Tresca_bnd, u_init='D'):
        self.grids_on_time = grids_on_time
        self.T = T
        self.tau = T / grids_on_time
        self.bdy_f = bdy_f
        self.tr_f1 = tr_f1
        self.tr_f2 = tr_f2
        self.Tresca_bnd = Tresca_bnd
        if u_init == 'D':
            self.u_init = np.zeros((self.total_fdms))
        else:
            self.u_init = u_init
        self.Amat = None
        self.set_Amat()

    def get_loc_inds(self, elem_ind_x, elem_ind_y):
        inds = np.zeros((LOC_FDMS, ), dtype=np.int32)
        is_clamped_fdm = np.ones((LOC_FDMS, ))
        nd_ind = elem_ind_y * (self.grids_on_dmn + 1) + elem_ind_x
        inds[0] = nd_ind * XY
        inds[1] = nd_ind * XY + 1
        if elem_ind_y == 0:
            is_clamped_fdm[1] = .0

        nd_ind = elem_ind_y * (self.grids_on_dmn + 1) + elem_ind_x + 1
        inds[2] = nd_ind * XY
        if elem_ind_x == self.grids_on_dmn - 1:
            is_clamped_fdm[2] = .0
        inds[3] = nd_ind * XY + 1
        if elem_ind_y == 0 or elem_ind_x == self.grids_on_dmn - 1:
            is_clamped_fdm[3] = .0

        nd_ind = (elem_ind_y + 1) * (self.grids_on_dmn + 1) + elem_ind_x
        inds[4] = nd_ind * XY
        inds[5] = nd_ind * XY + 1

        nd_ind = (elem_ind_y + 1) * (self.grids_on_dmn + 1) + elem_ind_x + 1
        inds[6] = nd_ind * XY
        if elem_ind_x == self.grids_on_dmn - 1:
            is_clamped_fdm[6] = .0
        inds[7] = nd_ind * XY + 1
        if elem_ind_x == self.grids_on_dmn - 1:
            is_clamped_fdm[7] = .0

        P = np.diag(is_clamped_fdm)
        return inds, P

    def set_Amat(self):
        max_data_len = self.total_elems * LOC_FDMS**2
        Amat_row = np.zeros((max_data_len, ), dtype=np.int32)
        Amat_col = np.zeros((max_data_len, ), dtype=np.int32)
        Amat_data = np.zeros((max_data_len, ))
        for elem_ind in range(self.total_elems):
            elem_ind_y, elem_ind_x = divmod(elem_ind, self.grids_on_dmn)
            C = self.cff_data[elem_ind_x % self.grids_on_cell, elem_ind_y % self.grids_on_cell, :]
            loc_Amat = get_loc_Amat(C)
            inds, P = self.get_loc_inds(elem_ind_x, elem_ind_y)
            inds_ext = np.tile(inds, (LOC_FDMS, 1))
            loc_Amat = P @ loc_Amat @ P
            Amat_col[elem_ind * LOC_FDMS**2:(elem_ind + 1) * LOC_FDMS**2] = inds_ext.flatten()
            Amat_row[elem_ind * LOC_FDMS**2:(elem_ind + 1) * LOC_FDMS**2] = inds_ext.T.flatten()
            Amat_data[elem_ind * LOC_FDMS**2:(elem_ind + 1) * LOC_FDMS**2] = loc_Amat.flatten()
        Amat_coo = coo_matrix((Amat_data, (Amat_row, Amat_col)), shape=(self.total_fdms, self.total_fdms))
        Amat = Amat_coo.tocsr()
        self.Amat = Amat

    def get_loc_rhs_bdy_f(self, elem_ind_x, elem_ind_y, t):
        loc_rhs = np.zeros((LOC_FDMS, ))
        bdy_f_at_quad_pnts = np.zeros((XY, QUAD_PNTS))
        x_center, y_center = (elem_ind_x + 0.5) * self.h, (elem_ind_y + 0.5) * self.h
        for quad_pnt_ind in range(QUAD_PNTS):
            quad_pnt_ind_y, quad_pnt_ind_x = divmod(quad_pnt_ind, QUAD_ORDER)
            quad_cord_x, quad_cord_y = 0.5 * self.h * QUAD_CORD[quad_pnt_ind_x] + x_center, 0.5 * self.h * QUAD_CORD[quad_pnt_ind_y] + y_center
            bdy_f_at_quad_pnts[:, quad_pnt_ind] = self.bdy_f(quad_cord_x, quad_cord_y, t)
        loc_rhs[0] = np.dot(bdy_f_at_quad_pnts[0, :] * base_val_at_quad_pnt[0, :], quad_wghts)
        loc_rhs[1] = np.dot(bdy_f_at_quad_pnts[1, :] * base_val_at_quad_pnt[0, :], quad_wghts)
        loc_rhs[2] = np.dot(bdy_f_at_quad_pnts[0, :] * base_val_at_quad_pnt[1, :], quad_wghts)
        loc_rhs[3] = np.dot(bdy_f_at_quad_pnts[1, :] * base_val_at_quad_pnt[1, :], quad_wghts)
        loc_rhs[4] = np.dot(bdy_f_at_quad_pnts[0, :] * base_val_at_quad_pnt[2, :], quad_wghts)
        loc_rhs[5] = np.dot(bdy_f_at_quad_pnts[1, :] * base_val_at_quad_pnt[2, :], quad_wghts)
        loc_rhs[6] = np.dot(bdy_f_at_quad_pnts[0, :] * base_val_at_quad_pnt[3, :], quad_wghts)
        loc_rhs[7] = np.dot(bdy_f_at_quad_pnts[1, :] * base_val_at_quad_pnt[3, :], quad_wghts)
        return 0.25 * self.h**2 * loc_rhs

    def get_loc_rhs_tr_f1(self, elem_ind_x, elem_ind_y, t):
        loc_rhs = np.zeros((LOC_FDMS, ))
        if elem_ind_y != self.grids_on_dmn - 1:
            return loc_rhs
        tr_f1_at_quad_pnts = np.zeros((XY, QUAD_ORDER))
        base_val_at_quad_pnt_1d2 = np.zeros((QUAD_ORDER))
        base_val_at_quad_pnt_1d3 = np.zeros((QUAD_ORDER))
        x_center = (elem_ind_x + 0.5) * self.h
        for quad_pnt_ind_x in range(QUAD_ORDER):
            quad_cord_x = 0.5 * self.h * QUAD_CORD[quad_pnt_ind_x] + x_center
            tr_f1_at_quad_pnts[:, quad_pnt_ind_x] = self.tr_f1(quad_cord_x, t)
            base_val_at_quad_pnt_1d2[quad_pnt_ind_x] = get_locbase_val(2, QUAD_CORD[quad_pnt_ind_x], 1.0)
            base_val_at_quad_pnt_1d3[quad_pnt_ind_x] = get_locbase_val(3, QUAD_CORD[quad_pnt_ind_x], 1.0)
        loc_rhs[4] = np.dot(tr_f1_at_quad_pnts[0, :] * base_val_at_quad_pnt_1d2, QUAD_WGHT)
        loc_rhs[5] = np.dot(tr_f1_at_quad_pnts[1, :] * base_val_at_quad_pnt_1d2, QUAD_WGHT)
        loc_rhs[6] = np.dot(tr_f1_at_quad_pnts[0, :] * base_val_at_quad_pnt_1d3, QUAD_WGHT)
        loc_rhs[7] = np.dot(tr_f1_at_quad_pnts[1, :] * base_val_at_quad_pnt_1d3, QUAD_WGHT)
        return 0.5 * self.h * loc_rhs

    def get_loc_rhs_tr_f2(self, elem_ind_x, ele_ind_y, t):
        loc_rhs = np.zeros((LOC_FDMS, ))
        if elem_ind_x != self.grids_on_dmn - 1:
            return loc_rhs
        tr_f2_at_quad_pnts = np.zeros((XY, QUAD_ORDER))
        base_val_at_quad_pnt_1d1 = np.zeros((QUAD_ORDER))
        base_val_at_quad_pnt_1d3 = np.zeros((QUAD_ORDER))
        y_center = (elem_ind_y + 0.5) * self.h
        for quad_pnt_ind_y in range(QUAD_ORDER):
            quad_cord_y = 0.5 * self.h * QUAD_CORD[quad_pnt_ind_y] + y_center
            tr_f2_at_quad_pnts[:, quad_pnt_ind_y] = self.tr_f2(quad_cord_y, t)
            base_val_at_quad_pnt_1d1[quad_pnt_ind_x] = get_locbase_val(1, 1.0, QUAD_CORD[quad_pnt_ind_y])
            base_val_at_quad_pnt_1d3[quad_pnt_ind_x] = get_locbase_val(3, 1.0, QUAD_CORD[quad_pnt_ind_y])
        loc_rhs[2] = np.dot(tr_f2_at_quad_pnts[0, :] * base_val_at_quad_pnt_1d1, QUAD_WGHT)
        loc_rhs[3] = np.dot(tr_f2_at_quad_pnts[1, :] * base_val_at_quad_pnt_1d1, QUAD_WGHT)
        loc_rhs[6] = np.dot(tr_f2_at_quad_pnts[0, :] * base_val_at_quad_pnt_1d3, QUAD_WGHT)
        loc_rhs[7] = np.dot(tr_f2_at_quad_pnts[1, :] * base_val_at_quad_pnt_1d3, QUAD_WGHT)
        return 0.5 * self.h * loc_rhs

    def get_rhs(self, u_minus, t):
        rhs = np.zeros((self.total_fdms, ))
        for elem_ind in range(self.total_elems):
            elem_ind_y, elem_ind_x = divmod(elem_ind, self.grids_on_dmn)
            loc_inds, P = self.get_loc_inds(elem_ind_x, elem_ind_y)
            loc_rhs = self.get_loc_rhs_bdy_f(elem_ind_x, elem_ind_y, t) \
                + self.get_loc_rhs_tr_f1(elem_ind_x, elem_ind_y, t) \
                + self.get_loc_rhs_tr_f2(elem_ind_x, ele_ind_y, t)
            loc_rhs = P.dot(loc_rhs)
            np.add.at(rhs, loc_inds, loc_rhs)
        rhs -= self.Amat.dot(u_minus)
        return u_minus

    def get_J(self, u):
        data = u[0:2 * self.grids_on_dmn:2]
        assert data.shape == (self.grids_on_dmn, )
        data = self.h * self.Tresca_bnd * np.abs(data)
        data[0] = 0.5 * data[0]
        data[-1] = 0.5 * data[-1]
        return np.sum(data)

    def get_subgrad_J(self, u):
        subgrad_J = np.zeros((self.total_fdms, ))
        h_HT = self.h * self.Tresca_bnd
        for fdm_ind in range(0, 2 * self.grids_on_dmn, 2):
            if fdm_ind == 0 or fdm_ind == 2 * (self.grids_on_cell - 1):
                val = 0.5 * h_HT
            else:
                val = h_HT
            if u[fdm_ind] > 0:
                subgrad_J[fdm_ind] = val
            elif u[fdm_ind] < 0:
                subgrad_J[fdm_ind] = -val
            else:
                subgrad_J[fdm_ind] = 0.0
        return subgrad_J

    def get_obj_val(self, v, rhs):
        val = 0.0
        val += 0.5 * self.tau * np.dot(self.Amat.dot(v), v)
        val += self.get_J(v)
        val -= np.dot(rhs, v)
        return val

    def get_subgrad_obj(self, v, rhs):
        subgrad_obj = np.zeros((self.total_fdms, ))
        subgrad_obj += self.tau * self.Amat.dot(v)
        subgrad_obj += self.get_subgrad_J(v)
        subgrad_obj -= rhs
        return subgrad_obj

    def get_next_u(self, u_minus, t):
        delta_u, info = np.zeros((self.total_fdms)), 0
        rhs = self.get_rhs(u_minus, t)
        mu = MU * self.tau
        w, w_minus = np.zeros((self.total_fdms, )), np.zeros((self.total_fdms, ))
        g = self.get_subgrad_obj(w_minus, rhs)
        w = w_minus - stepsize_theta(0) * g / np.linalg.norm(g)
        for iter_ind in range(1, MAX_ITERS):
            theta, theta_minus = stepsize_theta(iter_ind), stepsize_theta(iter_ind - 1)
            a = 3.0 / (mu * iter_ind * iter_ind)
            y = w + theta * (1.0 / theta_minus - 1) * (w - w_minus)
            g = self.get_subgrad_obj(y, rhs)
            k_y = theta / (theta + a * mu)
            k_w = a * mu / (theta + a * mu)
            k_g = a * theta / (theta + a * mu)
            w_plus = k_y * y + k_w * w - k_g * g
            if (np.linalg.norm(w_plus - w) <= TOL * np.linalg.norm(w)):
                info = iter_ind
                delta_u = w_plus
                break
            elif iter_ind == MAX_ITERS - 1:
                info = -1
                delta_u = w_plus
            else:
                w_minus = w
                w = w_plus
        u = u_minus + self.tau * delta_u
        return u, info

    def solve(self):
        u = self.u_init
        for time_ind in range(1, self.grids_on_time + 1):
            t = time_ind * tau
            u, info = self.get_next_u(u, t)
        return u
