from simple_system import *
import matplotlib.pyplot as plt
import time


def main():
    start = time.time()
    col_test = Collimator()
    col_test.add_aperture('slit', size=2, loc=col_test.colp)

    det_test = Detector()

    endpts = det_test.face_pts()

    em_pt = np.array([0, 0, 0])

    projection = col_test.ray_trace(endpts, em_pt)
    # print projection.shape
    # print(np.sum(projection[:]))
    # print 'It took', time.time() - start, 'seconds.'
    plt.imshow(projection.T,cmap='gray')
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()