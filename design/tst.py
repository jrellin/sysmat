import numpy as np
import tables
# from scipy import ndimage
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from design.utils import compute_mlem
from design.utils import interpolate_system_response
from design.utils import load_h5file


def see_projection_together(sysmat_fname, choose_pt=0):
    sysmat_file = load_h5file(sysmat_fname)
    sysmat = sysmat_file.root.sysmat[:]

    plt.figure(figsize=(12, 8))
    plt.imshow(sysmat[choose_pt].reshape([48, 48]), cmap='jet', origin='lower', interpolation='nearest')
    sysmat_file.close()
    plt.show()


# def sensitivity_map(sysmat_fname, npix=(150, 50), pxl_sze= 1, dpix=(48, 48), correction=False):
def sensitivity_map(sysmat, npix=(150, 50), pxl_sze=1, dpix=(48, 48), correction=False):
    # from matplotlib.ticker import MaxNLocator
    print("System Shape: ", sysmat.shape)

    sens = np.sum(sysmat, axis=1).reshape([npix[1], npix[0]])
    if correction:
        sens = np.mean(sens) / sens
    # plt.figure(figsize=(12, 8))
    extent_img = np.array([-npix[0]/20, npix[0]/20, -npix[1]/20, npix[1]/20]) * pxl_sze  # in cm
    img = plt.imshow(sens, cmap='magma', origin='lower', interpolation='nearest',
                     aspect='equal', extent=extent_img)
    ax = plt.gca()
    ax.yaxis.get_major_locator().set_params(integer=True)
    ax.xaxis.get_major_locator().set_params(integer=True)

    if correction:
        plt.title("Sensitivity Correction Map", fontsize=16)
    else:
        plt.title("Sensitivity Map", fontsize=16)
    plt.xlabel('[cm]', fontsize=14)
    plt.xticks(fontsize=14)
    plt.ylabel('[cm]', fontsize=14)
    plt.yticks(fontsize=14)

    cbar = plt.colorbar(img)
    cbar.set_label('/ mm$^{3}$', rotation=0, y=0, ha='center')
    cbar.ax.tick_params(labelsize=12)
    # plt.colorbar(img, fraction=0.046 * (sysmat.shape[0]/sysmat.shape[1]), pad=0.04)
    # sysmat_file.close()
    print("Total Sensitivity: ", np.sum(sysmat))
    print("Average Sensitivity: ", np.sum(sysmat)/np.prod(npix))
    print("Max Sensitivity: ", sens.max())
    plt.show()
    return sens


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


def system_matrix_interpolate(sysmat_filename, x_dim=75):
    """Kargs: x_img_pixels, save_fname, """
    sysmat_file = load_h5file(sysmat_filename)
    sysmat = sysmat_file.root.sysmat[:]

    save_name = sysmat_filename[:-3] + '_interp'
    interpolate_system_response(sysmat, x_img_pixels=x_dim, save_fname=save_name)
    sysmat_file.close()


def main():
    # filename = '/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0.h5'
    # filename = '/Users/justinellin/repos/sysmat/design/2021-02-23-2351_SP0.h5'  # 3 points
    # see_projection_together(filename, choose_pt=937)
    # see_projection_separate(filename, choose_pt=937)

    # filename = '/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0_interp.npy'
    # filename = '/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0_F1S7.npy'
    filename = '/home/justin/repos/sysmat/design/Apr14_full_F2_0S7.npy'

    # filename = '/home/justin/Desktop/system_responses/Thesis/2021-03-27-1529_SP0.h5'
    sysmat = np.load(filename)
    # sysmat_file = load_h5file(filename)
    # sysmat = sysmat_file.root.sysmat[:]
    # 149, 49 interp size
    sensitivity_map(sysmat, npix=(201, 201), pxl_sze=1, correction=False)


if __name__ == '__main__':
    main()

    # fname = '/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0_F1S7.npy'
    # sysmat = np.load(fname)
    # sens_correct = sensitivity_map(sysmat, npix=(149, 49), pxl_sze=1, correction=True)
    # test_mlem(sysmat_filename=fname,
    #          line_source=True, line_length=100, line_buffer=4, line_sigma=1, line_width=1, filt_sigma=[0.5, 4.5],
    #          img_pxl_x=149, img_pxl_y=49, pxl_sze=1, counts=10**8, slice_plots=True,
    #          nIterations=800, h5file=False)

    # test_mlem(sysmat_filename='/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0.h5',
    #          line_source=True, filt_sigma=[0.25, 1], nIterations=500, counts=10**8, slice_plots=True)

    # [0.25, 1] for nice thin line source, [0.5, 1] wide source
    # test_mlem(sysmat_filename='/Users/justinellin/repos/sysmat/design/2021-02-28-2345_SP0.h5',
    #          line_source=False, filt_sigma=0.5, nIterations=100)  # Flood test

    # system_matrix_interpolate('/home/justin/repos/sysmat/design/2021-03-30-2347_SP0.h5', x_dim=101)

