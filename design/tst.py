import numpy as np
import tables
# from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from design.utils import compute_mlem


def load_h5file(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def see_projection_together(sysmat_fname, choose_pt=0):
    sysmat_file = load_h5file(sysmat_fname)
    sysmat = sysmat_file.root.sysmat[:]

    plt.figure(figsize=(12, 8))
    plt.imshow(sysmat[choose_pt].reshape([48, 48]), cmap='jet', origin='lower', interpolation='nearest')
    sysmat_file.close()
    plt.show()


def sensitivity_map(sysmat_fname, npix=(150, 50), pxl_sze= 1, dpix=(48, 48), correction=False):
    sysmat_file = load_h5file(sysmat_fname)
    sysmat = sysmat_file.root.sysmat[:]
    print("System Shape: ", sysmat.shape)

    sens = np.sum(sysmat, axis=1).reshape([npix[1], npix[0]])
    if correction:
        sens = np.mean(sens) / sens
    plt.figure(figsize=(12, 8))
    extent_img = np.array([-npix[0]/2, npix[0]/2, -npix[1]/2, npix[1]/2]) * pxl_sze
    img = plt.imshow(sens, cmap='jet', origin='lower', interpolation='nearest', aspect='equal', extent=extent_img)

    if correction:
        plt.title("Sensitivity Correction Map", fontsize=14)
    else:
        plt.title("Sensitivity Map", fontsize=14)
    plt.xlabel('[mm]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('[mm]', fontsize=14)

    plt.colorbar(img, fraction=0.046 * (sysmat.shape[0]/sysmat.shape[1]), pad=0.04)
    sysmat_file.close()
    print("Total Sensitivity: ", np.sum(sysmat))
    print("Average Sensitivity: ", np.sum(sysmat)/np.prod(npix))
    plt.show()


def see_projection_separate(sysmat_fname, choose_pt=0):
    sysmat_file = load_h5file(sysmat_fname)
    sysmat = sysmat_file.root.sysmat[:]

    point_response = sysmat[choose_pt].reshape([48, 48])

    layout = np.array([4, 4])

    fig, ax = plt.subplots(layout[0], layout[1])

    for plt_index in np.arange(layout[1] * layout[0]):
        row = plt_index // layout[1]
        col = plt_index % layout[0]

        data = point_response[(12 * row):(12 * (row+1)), (12 * col):(12 * (col+1))]

        im = ax[row, col].imshow(data, origin='lower')
        # plt.colorbar(im, ax=ax[row, col])
        ax[row, col].set_yticks([])
        ax[row, col].set_xticks([])

    fig.tight_layout()
    plt.show()


def test_mlem(sysmat_filename, check_proj=False, point_check=None, line_source=False, sensitivity_norm=True,
              line_width=1, line_length=50, line_buffer=2, line_sigma=0.5, counts=10**6,
              img_pxl_x=75, img_pxl_y=25, pxl_sze=2, **kwargs):
    # img_pxl_y = 25
    # img_pxl_x = 75
    if point_check is None:
        point_check = img_pxl_x * img_pxl_y // 2

    test_img = np.zeros([img_pxl_y, img_pxl_x])

    if line_source:
        kern_shape = np.array([line_width, line_length + (2 * line_buffer)])  # line_width in y-dir, length in x-dir
        kern = np.zeros(kern_shape)
        kern[:, line_buffer:-line_buffer] = 1

        # print('Kernel: ', kern)

        mid_col = img_pxl_x//2 - ((img_pxl_x+1) % 2)
        mid_row = img_pxl_y//2 - ((img_pxl_y+1) % 2)

        col_offset = mid_col - (kern_shape[1]//2 - ((kern_shape[1]+1) % 2))
        row_offset = mid_row - (kern_shape[0]//2 - ((kern_shape[0]+1) % 2))

        test_img[row_offset:kern_shape[0] + row_offset, col_offset:kern_shape[1] + col_offset] = kern
        test_img = gaussian_filter(test_img, line_sigma, mode='constant')
        test_img = (test_img/test_img.sum() * counts)
           #  \ (kern/kern.sum() * counts)
    else:
        test_img[np.unravel_index(point_check, test_img.shape)] = counts

    plt.imshow(test_img,  cmap='jet', origin='lower')
    plt.show()

    sysmat_file = load_h5file(sysmat_filename)
    sysmat = sysmat_file.root.sysmat[:].T

    test_counts = sysmat.dot(test_img.ravel())
    if sensitivity_norm:
        sens = np.sum(sysmat, axis=0)
    else:
        sens = np.ones(sysmat.shape[1])

    recon = compute_mlem(sysmat, test_counts, img_pxl_x, sensitivity=sens, **kwargs)

    if check_proj:
        fig, ax = plt.subplots(4, 4)

        point_response = test_counts.reshape([48, 48])
        for plt_index in np.arange(4 * 4):
            row = plt_index // 4
            col = plt_index % 4

            data = point_response[(12 * row):(12 * (row+1)), (12 * col):(12 * (col+1))]

            im = ax[row, col].imshow(data, origin='lower')
            # plt.colorbar(im, ax=ax[row, col])
            ax[row, col].set_yticks([])
            ax[row, col].set_xticks([])

        sysmat_file.close()
        fig.tight_layout()
        plt.show()

    plt.figure(figsize=(12, 8))
    extent_img = np.array([-img_pxl_x / 2, img_pxl_x / 2, -img_pxl_y / 2, img_pxl_y / 2]) * pxl_sze
    img = plt.imshow(recon.reshape([img_pxl_y, img_pxl_x]), cmap='jet', origin='lower', interpolation='nearest',
                     extent=extent_img)

    plt.xlabel('[mm]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('[mm]', fontsize=14)
    plt.yticks(fontsize=14)

    print("Total Counts: ", recon.sum())

    plt.colorbar(img, fraction=0.046 * (sysmat.shape[0]/sysmat.shape[1]), pad=0.04)
    plt.show()
    sysmat_file.close()


def main():
    filename = '/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0.h5'
    # filename = '/Users/justinellin/repos/sysmat/design/2021-02-24-0102_SP0.h5'
    # filename = '/Users/justinellin/repos/sysmat/design/2021-02-23-2351_SP0.h5'  # 3 points
    see_projection_together(filename, choose_pt=937)
    # see_projection_separate(filename, choose_pt=937)
    # sensitivity_map(filename, npix=(75, 25), pxl_sze=2, correction=True)


if __name__ == '__main__':
    # main()
    # test_mlem(sigma=0.5)
    test_mlem(sysmat_filename='/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0.h5',
              line_source=True, filt_sigma=0.5, nIterations=100)  # TODO: Sensitivity Term in System_Response
    # TODO: Generate "line" data with poisson noise
