# from simple_system import *
from simple_sysmat import simple_system as ss
# import simple_system as ss
import matplotlib.pyplot as plt
import time
import numpy as np


def main():
    start = time.time()
    col_test = ss.Collimator()
    slit1 = np.array([0, 0, 0])
    col_test.add_aperture('slit', size=2, loc=col_test.colp + slit1)

    slit2 = np.array([0, 10, 0])
    col_test.add_aperture('slit', size=2, slit_ax=(1, 0, 0), loc=col_test.colp + slit2)

    det_test = ss.Detector()

    endpts = det_test.face_pts()
    # print('Endpts (10): ', endpts[:5])

    em_pt = np.array([0, 5, 0])  # TODO: What is GOING ON?

    projection = col_test.ray_trace(endpts, em_pt)
    # print projection.shape
    # print(np.sum(projection[:]))
    print('It took ' + str(time.time() - start) + ' seconds.')
    # plt.imshow(np.log(projection).T,cmap='Reds')
    plt.imshow(projection, cmap='gray')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
