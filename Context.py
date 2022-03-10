""" 
Some abbreviations:
prd = periodicity
bnd = bound
bdy = body
bdry = boundary
tr = traction 
nd = node
ele = element
fdm = freedom
ind = index
pnt = point
grad = graditude
loc = local
cff = coefficient
cond = condition
iter = iteration

Viogt's notation:
11 -> 1
22 -> 2
12 -> 3
Sequence: 11 -> 22 -> 33 -> 23 -> 13 -> 12
1111 -> 2222 -> 1212 -> 2212 -> 1112 -> 1122
"""

import numpy as np

DIM = 2
QUAD_ORDER = 3
XY = 2
QUAD_CORD, QUAD_WGHT = np.polynomial.legendre.leggauss(QUAD_ORDER)
QUAD_PNTS = QUAD_ORDER**DIM
CFF_LEN = 6
LOC_FDMS = 8
LOC_NDS = XY * DIM
TOL = 1.e-5


def get_locbase_val(loc_ind: int, x: float, y: float):
    val = -1.0
    if loc_ind == 0:
        val = 0.25 * (1.0 - x) * (1.0 - y)
    elif loc_ind == 1:
        val = 0.25 * (1.0 + x) * (1.0 - y)
    elif loc_ind == 2:
        val = 0.25 * (1.0 - x) * (1.0 + y)
    elif loc_ind == 3:
        val = 0.25 * (1.0 + x) * (1.0 + y)
    else:
        raise ValueError("Invalid option")
    return val


def get_locbase_grad_val(loc_ind: int, x: float, y: float):
    grad_val_x, grad_val_y = -1.0, -1.0
    if loc_ind == 0:
        grad_val_x = -0.25 * (1.0 - y)
        grad_val_y = -0.25 * (1.0 - x)
    elif loc_ind == 1:
        grad_val_x = 0.25 * (1.0 - y)
        grad_val_y = -0.25 * (1.0 + x)
    elif loc_ind == 2:
        grad_val_x = -0.25 * (1.0 + y)
        grad_val_y = 0.25 * (1.0 - x)
    elif loc_ind == 3:
        grad_val_x = 0.25 * (1.0 + y)
        grad_val_y = 0.25 * (1.0 + x)
    else:
        raise ValueError("Invalid option")
    return grad_val_x, grad_val_y


base_grad_val_at_quad_pnt = np.zeros((LOC_NDS, DIM, QUAD_PNTS))
base_val_at_quad_pnt = np.zeros((LOC_NDS, QUAD_PNTS))
quad_wghts = np.zeros((QUAD_PNTS, ))
for loc_nd_ind in range(LOC_NDS):
    for quad_pnt_ind_x in range(QUAD_ORDER):
        for quad_pnt_ind_y in range(QUAD_ORDER):
            quad_pnt_ind = quad_pnt_ind_y * QUAD_ORDER + quad_pnt_ind_x
            x, y = QUAD_CORD[quad_pnt_ind_x], QUAD_CORD[quad_pnt_ind_y]
            base_grad_val_at_quad_pnt[loc_nd_ind, :, quad_pnt_ind] = get_locbase_grad_val(loc_nd_ind, x, y)
            base_val_at_quad_pnt[loc_nd_ind, quad_pnt_ind] = get_locbase_val(loc_nd_ind, x, y)
            quad_wghts[quad_pnt_ind] = QUAD_WGHT[quad_pnt_ind_x] * QUAD_WGHT[quad_pnt_ind_y]


def get_loc_Amat(C=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])):
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

    loc_Amat = np.zeros((LOC_FDMS, LOC_FDMS))
    for m in range(LOC_NDS):
        for n in range(LOC_NDS):
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
    return loc_Amat


def get_C_from_E_nu(E: float, nu: float):
    '''
    E  : Young's modulus
    nu : Possion's ratio
    '''
    lamb_3d = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
    mu = E / (2.0 * (1.0 + nu))
    lamb_2d = 2 * lamb_3d * mu / (lamb_3d + 2.0 * mu)
    C11 = lamb_2d + 2.0 * mu  # C_1111
    C22 = lamb_2d + 2.0 * mu  # C_2222
    C33 = mu  # C_1212
    C23 = 0.0  # C_2212
    C13 = 0.0  # C_1112
    C12 = lamb_2d  # C_1122
    return C11, C22, C33, C23, C13, C12


def Zero_s1dto2d(x: float, t: float):
    return 0.0, 0.0


def Zero_s2dto2d(x: float, y: float, t: float):
    return 0.0, 0.0


def default_tr_f2(y: float, t: float):
    return 0.08 * (1.25 - y) * t, -0.01 * t


def default_cell_cff(x: float, y: float):
    E0, E1, nu0, nu1 = 150., 250., 0.35, 0.45
    if 1. / 8 <= x <= 7. / 8 and 3. / 8 <= y <= 5. / 8:
        return get_C_from_E_nu(E0, nu0)
    elif 3. / 8 <= x <= 5. / 8 and 1. / 8 <= y <= 7. / 8:
        return get_C_from_E_nu(E0, nu0)
    else:
        return get_C_from_E_nu(E1, nu1)


default_T = 1.0
default_Tresca_bnd = 0.004


class Context:
    def __init__(self, prd: int, grids_on_cell: int, cell_cff=default_cell_cff):
        self.prd = prd
        self.grids_on_cell = grids_on_cell
        self.grids_on_dmn = prd * grids_on_cell
        self.h = 1.0 / (prd * grids_on_cell)
        self.total_nds = (1 + prd * grids_on_cell)**2
        self.total_elems = (prd * grids_on_cell)**2
        self.total_fdms = 2 * (1 + prd * grids_on_cell)**2
        self.hh = 1.0 / grids_on_cell
        self.total_nds_on_cell = (1 + grids_on_cell)**2
        self.total_elems_on_cell = (grids_on_cell)**2
        self.total_fdms_on_cell = 2 * (grids_on_cell)**2
        cff_data = np.zeros((grids_on_cell, grids_on_cell, CFF_LEN))
        for sub_elem_ind_x in range(grids_on_cell):
            for sub_elem_ind_y in range(grids_on_cell):
                x, y = hh * (0.5 + sub_elem_ind_x), hh * (0.5 + sub_elem_ind_y)
                cff_data[sub_elem_ind_x, sub_elem_ind_y, :] = cell_cff(x, y)
        self.cff_data = cff_data
