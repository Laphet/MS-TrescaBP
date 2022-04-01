from Context import *
from Homogenization import *
from TrescaBP import *
import matplotlib.pyplot as plt

GRIDS_ON_CELL = 8
GRIDS_ON_TIME = 128

import sys, os
import logging
from logging import config


def get_errors(u_ms, u_homo, corrs, ms_bvp, homo_bvp):
    l2, l2_ref, h1, h1_corr = 0.0, 0.0, 0.0, 0.0
    grids_on_dmn = ms_bvp.grids_on_dmn
    grids_on_cell = ms_bvp.grids_on_cell
    total_elems = ms_bvp.total_elems
    h = ms_bvp.h
    hh = ms_bvp.hh
    Amat = homo_bvp.Amat
    loc_u_ms, loc_u_homo, loc_corrs = np.zeros((LOC_FDMS,)), np.zeros((LOC_FDMS,)), np.zeros((CORRS_NUM, LOC_FDMS))
    loc_quad_val_l2 = np.zeros((QUAD_PNTS,))
    loc_quad_val_l2_ref = np.zeros((QUAD_PNTS,))
    loc_quad_val_h1 = np.zeros((QUAD_PNTS,))
    loc_quad_val_h1_corr = np.zeros((QUAD_PNTS,))
    for elem_ind in range(total_elems):
        elem_ind_y, elem_ind_x = divmod(elem_ind, grids_on_dmn)
        nd_ind = elem_ind_y * (grids_on_dmn + 1) + elem_ind_x
        loc_u_ms[0] = u_ms[nd_ind * XY]
        loc_u_ms[1] = u_ms[nd_ind * XY + 1]
        loc_u_homo[0] = u_homo[nd_ind * XY]
        loc_u_homo[1] = u_homo[nd_ind * XY + 1]
        nd_ind = elem_ind_y * (grids_on_dmn + 1) + elem_ind_x + 1
        loc_u_ms[2] = u_ms[nd_ind * XY]
        loc_u_ms[3] = u_ms[nd_ind * XY + 1]
        loc_u_homo[2] = u_homo[nd_ind * XY]
        loc_u_homo[3] = u_homo[nd_ind * XY + 1]
        nd_ind = (elem_ind_y + 1) * (grids_on_dmn + 1) + elem_ind_x
        loc_u_ms[4] = u_ms[nd_ind * XY]
        loc_u_ms[5] = u_ms[nd_ind * XY + 1]
        loc_u_homo[4] = u_homo[nd_ind * XY]
        loc_u_homo[5] = u_homo[nd_ind * XY + 1]
        nd_ind = (elem_ind_y + 1) * (grids_on_dmn + 1) + elem_ind_x + 1
        loc_u_ms[6] = u_ms[nd_ind * XY]
        loc_u_ms[7] = u_ms[nd_ind * XY + 1]
        loc_u_homo[6] = u_homo[nd_ind * XY]
        loc_u_homo[7] = u_homo[nd_ind * XY + 1]

        sub_elem_ind_y, sub_elem_ind_x = (elem_ind_y % grids_on_cell), (elem_ind_x % grids_on_cell)
        sub_nd_ind = sub_elem_ind_y * (grids_on_cell) + sub_elem_ind_x
        loc_corrs[:, 0] = corrs[:, sub_nd_ind * XY]
        loc_corrs[:, 1] = corrs[:, sub_nd_ind * XY + 1]
        sub_nd_ind = sub_elem_ind_y * (grids_on_cell) + (sub_elem_ind_x + 1) % grids_on_cell
        loc_corrs[:, 2] = corrs[:, sub_nd_ind * XY]
        loc_corrs[:, 3] = corrs[:, sub_nd_ind * XY + 1]
        sub_nd_ind = (sub_elem_ind_y + 1) % grids_on_cell * (grids_on_cell) + sub_elem_ind_x
        loc_corrs[:, 4] = corrs[:, sub_nd_ind * XY]
        loc_corrs[:, 5] = corrs[:, sub_nd_ind * XY + 1]
        sub_nd_ind = (sub_elem_ind_y + 1) % grids_on_cell * (grids_on_cell) + (sub_elem_ind_x + 1) % grids_on_cell
        loc_corrs[:, 6] = corrs[:, sub_nd_ind * XY]
        loc_corrs[:, 7] = corrs[:, sub_nd_ind * XY + 1]

        # k=1, alpha=1
        loc_quad_val_h1 = loc_u_ms[0::2] @ base_grad_val_at_quad_pnt[:, 0, :]
        loc_quad_val_h1 -= loc_u_homo[0::2] @ base_grad_val_at_quad_pnt[:, 0, :]
        h1 += np.dot(loc_quad_val_h1**2, quad_wghts)
        loc_quad_val_h1_corr = loc_quad_val_h1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[0, 0::2] @ base_grad_val_at_quad_pnt[:, 0, :])\
             * (loc_u_homo[0::2] @ base_grad_val_at_quad_pnt[:, 0, :]) # j=1, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[1, 0::2] @ base_grad_val_at_quad_pnt[:, 0, :])\
             * (loc_u_homo[1::2] @ base_grad_val_at_quad_pnt[:, 1, :]) # j=2, beta=2
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 0::2] @ base_grad_val_at_quad_pnt[:, 0, :])\
             * (loc_u_homo[1::2] @ base_grad_val_at_quad_pnt[:, 0, :]) # j=1, beta=2
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 0::2] @ base_grad_val_at_quad_pnt[:, 0, :])\
             * (loc_u_homo[0::2] @ base_grad_val_at_quad_pnt[:, 1, :]) # j=2, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 0::2] @ base_val_at_quad_pnt)\
             * (np.dot(base_grad2_val_at_quad_pnt, loc_u_homo[0::2])) # j=2, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[1, 0::2] @ base_val_at_quad_pnt)\
             * (np.dot(base_grad2_val_at_quad_pnt, loc_u_homo[1::2])) # j=2, beta=2
        h1_corr += np.dot(loc_quad_val_h1_corr**2, quad_wghts)

        # k=2, alpha=2
        loc_quad_val_h1 = loc_u_ms[1::2] @ base_grad_val_at_quad_pnt[:, 1, :]
        loc_quad_val_h1 -= loc_u_homo[1::2] @ base_grad_val_at_quad_pnt[:, 1, :]
        h1 += np.dot(loc_quad_val_h1**2, quad_wghts)
        loc_quad_val_h1_corr = loc_quad_val_h1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[0, 1::2] @ base_grad_val_at_quad_pnt[:, 1, :])\
             * (loc_u_homo[0::2] @ base_grad_val_at_quad_pnt[:, 0, :]) # j=1, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[1, 1::2] @ base_grad_val_at_quad_pnt[:, 1, :])\
             * (loc_u_homo[1::2] @ base_grad_val_at_quad_pnt[:, 1, :]) # j=2, beta=2
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 1::2] @ base_grad_val_at_quad_pnt[:, 1, :])\
             * (loc_u_homo[1::2] @ base_grad_val_at_quad_pnt[:, 0, :]) # j=1, beta=2
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 1::2] @ base_grad_val_at_quad_pnt[:, 1, :])\
             * (loc_u_homo[0::2] @ base_grad_val_at_quad_pnt[:, 1, :]) # j=2, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[0, 1::2] @ base_val_at_quad_pnt)\
             * (np.dot(base_grad2_val_at_quad_pnt, loc_u_homo[0::2])) # j=1, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 1::2] @ base_val_at_quad_pnt)\
             * (np.dot(base_grad2_val_at_quad_pnt, loc_u_homo[1::2])) # j=1, beta=2
        h1_corr += np.dot(loc_quad_val_h1_corr**2, quad_wghts)

        # k=1, alpha=2
        loc_quad_val_h1 = loc_u_ms[1::2] @ base_grad_val_at_quad_pnt[:, 0, :]
        loc_quad_val_h1 -= loc_u_homo[1::2] @ base_grad_val_at_quad_pnt[:, 0, :]
        h1 += np.dot(loc_quad_val_h1**2, quad_wghts)
        loc_quad_val_h1_corr = loc_quad_val_h1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[0, 1::2] @ base_grad_val_at_quad_pnt[:, 0, :])\
             * (loc_u_homo[0::2] @ base_grad_val_at_quad_pnt[:, 0, :]) # j=1, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[1, 1::2] @ base_grad_val_at_quad_pnt[:, 0, :])\
             * (loc_u_homo[1::2] @ base_grad_val_at_quad_pnt[:, 1, :]) # j=2, beta=2
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 1::2] @ base_grad_val_at_quad_pnt[:, 0, :])\
             * (loc_u_homo[1::2] @ base_grad_val_at_quad_pnt[:, 0, :]) # j=1, beta=2
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 1::2] @ base_grad_val_at_quad_pnt[:, 0, :])\
             * (loc_u_homo[0::2] @ base_grad_val_at_quad_pnt[:, 1, :]) # j=2, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 1::2] @ base_val_at_quad_pnt)\
             * (np.dot(base_grad2_val_at_quad_pnt, loc_u_homo[0::2])) # j=2, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[1, 1::2] @ base_val_at_quad_pnt)\
             * (np.dot(base_grad2_val_at_quad_pnt, loc_u_homo[1::2])) # j=2, beta=2
        h1_corr += np.dot(loc_quad_val_h1_corr**2, quad_wghts)

        # k=2, alpha=1
        loc_quad_val_h1 = loc_u_ms[0::2] @ base_grad_val_at_quad_pnt[:, 1, :]
        loc_quad_val_h1 -= loc_u_homo[0::2] @ base_grad_val_at_quad_pnt[:, 1, :]
        h1 += np.dot(loc_quad_val_h1**2, quad_wghts)
        loc_quad_val_h1_corr = loc_quad_val_h1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[0, 0::2] @ base_grad_val_at_quad_pnt[:, 1, :])\
             * (loc_u_homo[0::2] @ base_grad_val_at_quad_pnt[:, 0, :]) # j=1, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[1, 0::2] @ base_grad_val_at_quad_pnt[:, 1, :])\
             * (loc_u_homo[1::2] @ base_grad_val_at_quad_pnt[:, 1, :]) # j=2, beta=2
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 0::2] @ base_grad_val_at_quad_pnt[:, 1, :])\
             * (loc_u_homo[1::2] @ base_grad_val_at_quad_pnt[:, 0, :]) # j=1, beta=2
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 0::2] @ base_grad_val_at_quad_pnt[:, 1, :])\
             * (loc_u_homo[0::2] @ base_grad_val_at_quad_pnt[:, 1, :]) # j=2, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[0, 0::2] @ base_val_at_quad_pnt)\
             * (np.dot(base_grad2_val_at_quad_pnt, loc_u_homo[0::2])) # j=1, beta=1
        loc_quad_val_h1_corr -= 2.0 / hh * (loc_corrs[2, 0::2] @ base_val_at_quad_pnt)\
             * (np.dot(base_grad2_val_at_quad_pnt, loc_u_homo[1::2])) # j=1, beta=2
        h1_corr += np.dot(loc_quad_val_h1_corr**2, quad_wghts)

        # alpha=1
        loc_quad_val_l2 = (loc_u_ms[0::2] - loc_u_homo[0::2]) @ base_val_at_quad_pnt
        l2 += 0.25 * h**2 * np.dot(loc_quad_val_l2**2, quad_wghts)
        loc_quad_val_l2_ref = loc_u_homo[0::2] @ base_val_at_quad_pnt
        l2_ref += 0.25 * h**2 * np.dot(loc_quad_val_l2_ref**2, quad_wghts)
        #  alpha=2
        loc_quad_val_l2 = (loc_u_ms[1::2] - loc_u_homo[1::2]) @ base_val_at_quad_pnt
        l2 += 0.25 * h**2 * np.dot(loc_quad_val_l2**2, quad_wghts)
        loc_quad_val_l2_ref = loc_u_homo[0::2] @ base_val_at_quad_pnt
        l2_ref += 0.25 * h**2 * np.dot(loc_quad_val_l2_ref**2, quad_wghts)
    h1_ref = np.dot(Amat.dot(u_homo), u_homo)
    return np.sqrt(l2) / np.sqrt(l2_ref), np.sqrt(h1) / np.sqrt(h1_ref), np.sqrt(h1_corr) / np.sqrt(h1_ref)


def test_set(op='a'):
    assert op in ('a', 'b', 'c', 'd', 'e', 'f', 'g', 'h')
    if op == 'a':
        homo = Homogenization(GRIDS_ON_CELL)
        corrs, C_eff = homo.solve()
        prd = 2
        ms_bvp = TrescaBP(prd, GRIDS_ON_CELL)
        ms_bvp.set_conds(GRIDS_ON_TIME)
        homo_bvp = TrescaBP(prd, GRIDS_ON_CELL, const_cell_cff, C_eff)
        homo_bvp.set_conds(GRIDS_ON_TIME)

    if op == 'b':
        homo = Homogenization(GRIDS_ON_CELL)
        corrs, C_eff = homo.solve()
        prd = 4
        ms_bvp = TrescaBP(prd, GRIDS_ON_CELL)
        ms_bvp.set_conds(GRIDS_ON_TIME)
        homo_bvp = TrescaBP(prd, GRIDS_ON_CELL, const_cell_cff, C_eff)
        homo_bvp.set_conds(GRIDS_ON_TIME)

    if op == 'c':
        homo = Homogenization(GRIDS_ON_CELL)
        corrs, C_eff = homo.solve()
        prd = 8
        ms_bvp = TrescaBP(prd, GRIDS_ON_CELL)
        ms_bvp.set_conds(GRIDS_ON_TIME)
        homo_bvp = TrescaBP(prd, GRIDS_ON_CELL, const_cell_cff, C_eff)
        homo_bvp.set_conds(GRIDS_ON_TIME)

    if op == 'd':
        homo = Homogenization(GRIDS_ON_CELL)
        corrs, C_eff = homo.solve()
        prd = 16
        ms_bvp = TrescaBP(prd, GRIDS_ON_CELL)
        ms_bvp.set_conds(GRIDS_ON_TIME)
        homo_bvp = TrescaBP(prd, GRIDS_ON_CELL, const_cell_cff, C_eff)
        homo_bvp.set_conds(GRIDS_ON_TIME)

    if op == 'e':
        homo = Homogenization(GRIDS_ON_CELL, connected_cell_cff)
        corrs, C_eff = homo.solve()
        prd = 2
        ms_bvp = TrescaBP(prd, GRIDS_ON_CELL, connected_cell_cff)
        ms_bvp.set_conds(GRIDS_ON_TIME)
        homo_bvp = TrescaBP(prd, GRIDS_ON_CELL, const_cell_cff, C_eff)
        homo_bvp.set_conds(GRIDS_ON_TIME)

    if op == 'f':
        homo = Homogenization(GRIDS_ON_CELL, connected_cell_cff)
        corrs, C_eff = homo.solve()
        prd = 4
        ms_bvp = TrescaBP(prd, GRIDS_ON_CELL, connected_cell_cff)
        ms_bvp.set_conds(GRIDS_ON_TIME)
        homo_bvp = TrescaBP(prd, GRIDS_ON_CELL, const_cell_cff, C_eff)
        homo_bvp.set_conds(GRIDS_ON_TIME)

    if op == 'g':
        homo = Homogenization(GRIDS_ON_CELL, connected_cell_cff)
        corrs, C_eff = homo.solve()
        prd = 8
        ms_bvp = TrescaBP(prd, GRIDS_ON_CELL, connected_cell_cff)
        ms_bvp.set_conds(GRIDS_ON_TIME)
        homo_bvp = TrescaBP(prd, GRIDS_ON_CELL, const_cell_cff, C_eff)
        homo_bvp.set_conds(GRIDS_ON_TIME)

    if op == 'h':
        homo = Homogenization(GRIDS_ON_CELL, connected_cell_cff)
        corrs, C_eff = homo.solve()
        prd = 16
        ms_bvp = TrescaBP(prd, GRIDS_ON_CELL, connected_cell_cff)
        ms_bvp.set_conds(GRIDS_ON_TIME)
        homo_bvp = TrescaBP(prd, GRIDS_ON_CELL, const_cell_cff, C_eff)
        homo_bvp.set_conds(GRIDS_ON_TIME)

    logging.info("Config: epsilon=1/{0:d}, h=1/{1:d}, tau=1/{2:d}".format(ms_bvp.prd, ms_bvp.grids_on_dmn, ms_bvp.grids_on_time))
    u_ms = ms_bvp.u_init
    u_homo = homo_bvp.u_init
    errors = np.zeros((3, GRIDS_ON_TIME))
    write_log = 0.01 * GRIDS_ON_TIME
    for time_ind in range(1, GRIDS_ON_TIME + 1):
        t = time_ind / GRIDS_ON_TIME
        u_ms, info_ms = ms_bvp.get_next_u(u_ms, t)
        u_homo, info_homo = homo_bvp.get_next_u(u_homo, t)
        if info_ms < 0 or info_homo < 0:
            logging.warning("Optimization solver fails to converge! time-index={0:d}, info-ms={1:d}, info-homo-ms={2:d}.".format(time_ind, info_ms, info_homo))
        errors[:, time_ind - 1] = get_errors(u_ms, u_homo, corrs, ms_bvp, homo_bvp)
        if time_ind >= write_log:
            logging.info("Progress......{0:.2f}%.".format(time_ind / GRIDS_ON_TIME * 100.))
            write_log += 0.01 * GRIDS_ON_TIME


#    plot_final_frame = True
#     if plot_final_frame:
#         data_x = np.linspace(0.0, 1.0, ms_bvp.grids_on_dmn + 1)
#         data_ms = u_ms[0:2 * (ms_bvp.grids_on_dmn + 1):2]
#         data_homo = u_homo[0:2 * (ms_bvp.grids_on_dmn + 1):2]
#         fig, ax = plt.subplots(1, 1)
#         ax.plot(data_x, data_ms, label="ms")
#         ax.plot(data_x, data_homo, label="homo")
#         ax.legend()
#         plt.savefig("data/last-frame-{0:s}.pdf".format(op), dpi=150)
#         plt.savefig("data/last-frame-{0:s}.eps".format(op), dpi=150)
    return errors, u_ms, u_homo

if len(sys.argv) == 1:
    op = 'b'
else:
    op = sys.argv[1]
log_filename = "data/Option-{:s}.log".format(op)

config.fileConfig("Logger.cfg", defaults={'logfilename': log_filename})
logging.info('=' * 80)
logging.info("Start")
errors, u_ms, u_homo = test_set(op)

np.save("data/Errors-op-{:s}.npy".format(op), errors)
np.save("data/last-frame-ms-op-{:s}.npy".format(op), u_ms)
np.save("data/last-frame-homo-op-{:s}.npy".format(op), u_homo)
logging.info("L2={0:.5f}, H1={1:.5f}, H1-correctors={2:.5f}.".format(np.max(errors[0, :]), np.max(errors[1, :]), np.max(errors[2, :])))
logging.info("End")
logging.info('=' * 80)
