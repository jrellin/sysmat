import numpy as np
import pickle
from simple_sysmat.simple_system import Detector as det

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

detector = det(center=(0, 0, source_to_col+det_to_col+col_thick))
end_pts = detector.face_pts()


