# from simple_system import *

import simple_system as ss
import matplotlib.pyplot as plt
import time
import numpy as np


def main():
    start = time.time()
    col_test = ss.Collimator()
    col_test.add_aperture('slit', size=2, loc=col_test.colp)

    slit2 = np.array([5, 10, 0])
    col_test.add_aperture('slit', size=2, slit_ax=(1, 0, 0), loc=col_test.colp + slit2)

    det_test = ss.Detector()

    endpts = det_test.face_pts()

    em_pt = np.array([0, 0, 0])

    projection = col_test.ray_trace(endpts, em_pt)
    # print projection.shape
    # print(np.sum(projection[:]))
    print('It took ' + str(time.time() - start) + ' seconds.')
    # plt.imshow(np.log(projection).T,cmap='Reds')
    plt.imshow(projection.T, cmap='gray')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
