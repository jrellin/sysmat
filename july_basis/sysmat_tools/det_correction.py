import numpy as np


def flip_det(proj_array, ind, flip_ud=False, n_rot=1, ndets=(4, 4), det_pxls=(12, 12)):
    """flip_ud flips up/down. n_rot is number of 90 degree rotations, ndets = (row,col), det_pxls = ny, nx
    ind starts at 0 in upper left, to 3 in upper right, left to right up to down when facing front of det.
    proj_array is loaded proj_array. Flip happens before rotation"""
    det_rows, det_cols = ndets
    row = ind //det_rows  # 0 is top row
    col = ind % det_cols  # 0 is on left
    ny, nx = det_pxls

    proj = proj_array.reshape([ny * det_rows, nx * det_cols])
    area = proj[(col * ny):((col + 1) * ny), (row * nx):((row + 1) * nx)]

    if flip_ud:
        area = area[::-1]

    proj[(col * ny):((col+1)*ny), (row * nx):((row+1)*nx)] = np.rot90(area, n_rot)
    return proj


def weights(mid_include=True):

    mid_wgt = 1
    edge_wgts = 1/4 * mid_wgt
    corner_wgts = 1/2 * edge_wgts
    save_name = 'det_correction_mid'

    if not mid_include:
        mid_wgt = 0
        edge_wgts = 1
        corner_wgts = 1/2
        save_name = 'det_correction_no_mid'

    interior_pxl = mid_wgt + (4 * edge_wgts) + (4 * corner_wgts)
    edge_pxls = mid_wgt + (3 * edge_wgts) + (2 * corner_wgts)
    corner_pxls = mid_wgt + (2 * edge_wgts) + (1 * corner_wgts)

    edge_gain_correction = interior_pxl / edge_pxls
    corner_gain_correction = interior_pxl / corner_pxls
    return edge_gain_correction, corner_gain_correction, save_name


def main(**kwargs):
    ndets = np.array((4, 4))
    det_template =  np.ones([12, 12])
    print(weights(**kwargs))
    egc, cgc, save_name = weights(**kwargs)  # edge_gain_correction, corner_gain_correction

    det_template[0] = egc
    det_template[-1] = egc
    det_template[:, 0] = egc
    det_template[:, -1] = egc

    cidx = np.ix_((0, -1), (0, -1))
    det_template[cidx] = cgc

    correction = np.tile(det_template, ndets)
    # print("det_template: ", det_template)
    np.save(save_name, correction)


if __name__ == "__main__":
    main(mid_include=False)
