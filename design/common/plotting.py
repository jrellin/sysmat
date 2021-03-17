import numpy as np
from scipy.ndimage import gaussian_filter
import time
import tables
from design.utils import half_max_x


def fhwm_plotter():
    from matplotlib import pyplot as plt
    img_pxl_x = 75
    img_pxl_y = 25
    y0_plane_idx =  13
    image = np.load("/Users/justinellin/repos/sysmat/design/central_slice.npy")

    plt.figure(figsize=(12, 7))
    extent_img = np.array([-img_pxl_x / 2, img_pxl_x / 2, -img_pxl_y / 2, img_pxl_y / 2]) * 2
    x_vals = np.linspace(extent_img[0], extent_img[1], img_pxl_x)
    slice = image[y0_plane_idx, :]

    # find the two crossing points
    hmx = half_max_x(x_vals, slice)

    # print the answer
    fwhm = hmx[1] - hmx[0]
    print("FWHM:{:.2f}".format(fwhm))

    half = np.max(slice) / 2.0

    plt.plot(x_vals, slice, label='Single Slice',c='k')
    plt.plot(hmx, [half, half], linestyle='--', c='r', linewidth=2)
    plt.xlabel('[mm]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    plt.yticks(fontsize=20)
    # plt.plot(x_vals, np.mean(recon.reshape([img_pxl_y, img_pxl_x])[13-1:13+1+1, :], axis=0), label='3 Row Average')
    plt.title("Central (y=0) Slice", fontsize=28)
    plt.text(-25, 251000, r'$\Delta$ X$_{{FWHM}} = {:.2f}$ mm'.format(fwhm), fontsize=20, c='r')
    plt.show()
####################################
    plt.figure(figsize=(12, 7))

    y_proj = np.sum(image, axis=0)

    # find the two crossing points
    hmx = half_max_x(x_vals, y_proj)

    # print the answer
    fwhm = hmx[1] - hmx[0]
    print("FWHM:{:.2f}".format(fwhm))

    half = np.max(y_proj) / 2.0

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    plt.plot(x_vals, y_proj, label='Cumulative Counts Along Line', c='k')
    plt.plot(hmx, [half, half], linestyle='--', c='r', linewidth=2)
    plt.xlabel('[mm]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    plt.yticks(fontsize=20)
    # plt.plot(x_vals, np.mean(recon.reshape([img_pxl_y, img_pxl_x])[13-1:13+1+1, :], axis=0), label='3 Row Average')
    plt.title("Cumulative Counts Along Line", fontsize=28)
    # plt.text(-20, 1100000, "FWHM - {:.2f} mm".format(fwhm), fontsize=20, c='r')
    plt.text(-25, 1100000, r'$\Delta$ X$_{{FWHM}} = {:.2f}$ mm'.format(fwhm), fontsize=20, c='r')
    # r'$\Delta X_{FWHM} = {:.2f} mm$'.format(fwhm)
    plt.show()

####################################
    plt.figure(figsize=(12, 7))

    x_proj = np.sum(image, axis=1)

    y_vals = np.linspace(extent_img[2], extent_img[3], img_pxl_y)

    # find the two crossing points
    hmx = half_max_x(y_vals, x_proj)

    # print the answer
    fwhm = hmx[1] - hmx[0]
    print("FWHM:{:.2f}".format(fwhm))

    half = np.max(x_proj) / 2.0

    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    plt.plot(y_vals, x_proj, label='Cumulative Counts Along Line', c='k')
    plt.plot(hmx, [half, half], linestyle='--', c='r', linewidth=2)
    plt.xlabel('[mm]', fontsize=20)
    plt.xticks(fontsize=20)
    plt.ylabel('Counts', fontsize=20)
    plt.yticks(fontsize=20)
    # plt.plot(x_vals, np.mean(recon.reshape([img_pxl_y, img_pxl_x])[13-1:13+1+1, :], axis=0), label='3 Row Average')
    plt.title("Cumulative Counts Along Line", fontsize=28)
    # plt.text(-20, 1100000, "FWHM - {:.2f} mm".format(fwhm), fontsize=20, c='r')
    plt.text(-25, 1100000, r'$\Delta$ Y$_{{FWHM}} = {:.2f}$ mm'.format(fwhm), fontsize=20, c='r')
    # r'$\Delta X_{FWHM} = {:.2f} mm$'.format(fwhm)
    plt.show()