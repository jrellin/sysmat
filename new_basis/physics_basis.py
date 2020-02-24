import numpy as np
import pickle
from simple_sysmat.simple_system import Detector as det
from simple_sysmat.simple_system import Sources

# +x -> beam
with open("physics_basis.pkl", "rb") as fp:
    basis = pickle.load(fp)


def norm(array):
    arr = np.array(array)
    return arr/np.sqrt(np.dot(arr, arr))

# sysmat = np.load('/Users/justinellin/repos/sysmat/new_basis/2D_matrix_adjusted_11Apr16.npy')


# Physical Constants
det_to_col = 24  # cm, from face
col_thick = 7.5  # cm

# Constant Parameters
source_to_col = 13  # cm, front face
im_pix_sze = 0.1  # cm
k_y = 2  # 2 * k_y + 1 is kernel size in y

# Reproduce John's Code
i_x = 151
i_y = 51
n_x = 48
n_y = 48

detector = det(center=(0, 0, source_to_col+det_to_col+col_thick))  # source plane contains (0, 0, 0)
src = Sources(voxel_size=im_pix_sze, npix_1=i_x, npix_2=i_y)  # cm not mm!!!
end_pts = detector.face_pts()

em_pts = src.source_pts()
tot_em_pts = em_pts.shape[0]
tot_end_pts = end_pts.shape[0]

convolved_sys_mat = np.zeros([tot_em_pts, tot_end_pts])
for em_pt in em_pts:  # i.e. rows
    pass


