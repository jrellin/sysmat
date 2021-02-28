import numpy as np
import tables
from scipy import ndimage
from matplotlib import pyplot as plt


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


def main():
    filename = '/Users/justinellin/repos/sysmat/design/2021-02-24-0102_SP0.h5'
    # filename = '/Users/justinellin/repos/sysmat/design/2021-02-23-2351_SP0.h5'  # 3 points
    # see_projection_together(filename, choose_pt=2)
    see_projection_separate(filename, choose_pt=937)
    # sensitivity_map(filename, npix=(75, 25), pxl_sze=2, correction=True)


if __name__ == '__main__':
    main()
