import numpy as np
import pickle

with open("kernels.pkl", "rb") as fp:
    kernels = pickle.load(fp)

# sysmat = np.load('/Users/justinellin/repos/sysmat/new_basis/2D_matrix_adjusted_11Apr16.npy')

det_to_col = 24  # cm, from face
det_thick = 7.5  # cm
source_to_col = 13  # cm, front face
PMMA_C = 5  # C5O2H8
PMMA_O = 2
PMMA_H = 8


