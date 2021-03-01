import numpy as np


def generate_detector_centers_and_norms(layout, det_width=50, focal_length=350,
                                        x_dir=np.array([1, 0, 0]),
                                        y_dir=np.array([0, 1, 0]),
                                        focal_dir=np.array([0, 0, 1])):
    """Detector Width and Focal Length in mm. Focal_dir is direction of focal points of detectors"""
    alpha = 2 * np.arctan((det_width/2) / focal_length)
    rows, cols = layout

    scalar_cols = np.arange(-cols / 2 + 0.5, cols / 2 + 0.5)
    scalar_rows = np.arange(-rows / 2 + 0.5, rows / 2 + 0.5)

    x_sc = focal_length * np.sin(np.abs(scalar_rows) * alpha) * np.sign(scalar_rows)  # horizontal
    y_sc = focal_length * np.sin(np.abs(scalar_cols) * alpha) * np.sign(scalar_cols)  # vertical

    # print("x_sc: ", x_sc)
    # print("y_sc: ", y_sc)

    focal_pt = focal_length * focal_dir

    x_vec = np.outer(x_sc, x_dir)  # Start left (near beam port) of beam axis
    y_vec = np.outer(y_sc[::-1], y_dir)  # Start top row relative to ground

    # print("x_vec: ", x_vec)
    # print("y_vec: ", y_vec)

    centers = (y_vec[:, np.newaxis] + x_vec[np.newaxis, :]).reshape(-1, 3)

    # print("Centers: ", centers)

    centers[:, 2] = np.sqrt((focal_length**2) - np.sum(centers[:, :2] ** 2, axis=1)) * (-np.sign(focal_pt[2]))
    # TODO: This is not generic. Fix someday?

    # print("Centers: ", centers + focal_pt)

    directions = norm_vectors_array(-centers, axis=1)
    shifted_centers = centers + focal_pt  # this is now relative to center
    return shifted_centers, directions


def generate_flat_detector_pts(layout, center, mod_spacing_dist):
    rows, cols = layout

    scalar_cols = np.arange(-cols / 2 + 0.5, cols / 2 + 0.5) * mod_spacing_dist
    scalar_rows = np.arange(-rows / 2 + 0.5, rows / 2 + 0.5) * mod_spacing_dist

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])

    # distance_mod_plane = system.collimator.colp + np.array([0, 0, -130]) + (25.4 * x)  # shift of center

    x_vec = np.outer(scalar_cols, x)  # Start left (near beam port) of beam axis
    y_vec = np.outer(scalar_rows[::-1], y)  # Start top row relative to ground

    return (y_vec[:, np.newaxis] + x_vec[np.newaxis, :]).reshape(-1, 3) + center


def norm_vectors_array(mat, axis=1):  # Must be rows of vectors
    return mat/np.sqrt(np.sum(mat**2, axis=axis, keepdims=True))


def test():
    from matplotlib import pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = Axes3D(fig)

    centers, dirs = generate_detector_centers_and_norms(np.array([4, 4]), focal_length=350)
    for det_idx, det_center in enumerate(centers):
        print("Set det_center: ", det_center)
        print("Direction: ", dirs[det_idx])
        pass

    # ax.scatter(centers[:, 0], centers[:, 1], centers[:, 2])
    ax.quiver(centers[:, 0], centers[:, 1], centers[:, 2], dirs[:, 0], dirs[:, 1], dirs[:, 2], length=20)
    ax.set_zlim(0, 130)
    plt.show()


if __name__ == "__main__":
    test()
