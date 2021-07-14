import numpy as np
import tables
import os
from datetime import datetime
import time
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter


def load_sysmat(sysmat_fname):
    if tables.is_hdf5_file(sysmat_fname):
        sysmat_file_obj = load_h5file(sysmat_fname)
        return sysmat_file_obj.root.sysmat[:].T
    return np.load(sysmat_fname).T


def load_h5file(filepath):  # h5file.root.sysmat[:]
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def makedirs(path):
    """ Create directories for `path` (like 'mkdir -p'). """
    if not path:
        return
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)


class psf_generator(object):

    def __init__(self, fname, dims, center=(0, 0), pxl_sze=2):
        self.x_dim, self.y_dim = np.array(dims).astype('int')  # just making sure
        self.center = center  # image center
        self.measurements = 48 * 48  # detector pixels
        self.pixels = self.x_dim * self.y_dim
        self.pxl_size = pxl_sze
        self.sysmat_fname = os.path.splitext(fname)[0]  # Remove extension aka .h5

        # Sysmat load and cleanup min value
        self.sysmat = load_sysmat(fname)  # rows = image points, cols = detector measurements
        self.sysmat[self.sysmat == 0] = np.min(self.sysmat[self.sysmat != 0])

        self.total_counts = 6 * 10 ** 8
        self.image = np.full((self.y_dim, self.x_dim), 1.0 * self.total_counts / self.pixels)

        self.roi = np.zeros_like(self.image, dtype='bool')  # ROI mask used on self.image to calculate CNR
        self.roi_size = 0
        self.bg_size = self.pixels - self.roi_size

        self.projection = np.zeros(self.measurements)
        self.fig, self.ax, self.img_obj, self.cbar = self._initialize_plot()

        self.file = None
        self.table = None
        self.file_open = False

        self.row = 0
        self.col = 0
        self.linind = 0  # allows indexing through system response

        self.current_iteration = 0
        self.total_time = 0  # I.e. Time elapsed for reconstruction

    def _initialize_plot(self):
        fig = plt.figure(figsize=(9, 5), constrained_layout=False)
        ax = fig.add_subplot(111)
        ax.set_xlabel('mm')
        ax.set_ylabel('mm')

        extent_x = np.array([1, 1]) * self.center[0] + (np.array([-1, 1]) * (self.x_dim * self.pxl_size)) / 2
        extent_y = np.array([1, 1]) * self.center[1] + (np.array([-1, 1]) * (self.y_dim * self.pxl_size)) / 2

        # print("Ones like: ", np.ones_like(self.image).shape)
        img = ax.imshow(np.ones_like(self.image), cmap='magma', origin='upper',
                        interpolation='nearest', extent=np.append(extent_x, extent_y))
        cbar = fig.colorbar(img, fraction=0.046, pad=0.04, ax=ax)

        return fig, ax, img, cbar

    def update_plot(self):
        self.img_obj.set_data(self.image)
        self.img_obj.set_clim(vmin=np.min(self.image), vmax=np.max(self.image))
        self.cbar.draw_all()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def show_plot(self):
        # self.fig.show()
        plt.show()

    def _reset_image(self):
        self.image.fill(self.total_counts/self.image.size)

    @property
    def linind(self):
        return self.row * self.x_dim + self.col

    @linind.setter
    def linind(self, value):
        self.row = value // self.x_dim
        self.col = value % self.x_dim

    def open_file(self, **kwargs):
        if self.file_open:
            print("Attempted to open new file when file already open. Continuing...")
            return
        self._open_h5_file(**kwargs)
        self.file_open = True

    def close_file(self):
        try:
            self.file.close()
        except Exception as e:
            print("No file open. Error msg: ", e)

    def _open_h5_file(self, save_fname=None):
        if save_fname is None:
            base_fname = os.path.splitext(os.path.basename(self.sysmat_fname))[0]
            save_fname = os.path.join(os.getcwd(), 'Processed', 'psf_' + base_fname + '_proc_' + datetime.now().strftime("%m-%d") + '.h5')

        makedirs(save_fname)
        self.file = tables.open_file(save_fname, mode="w", title="Point Spread Functions")
        self.table = self.file.create_earray('/', 'psf', atom=tables.atom.Float64Atom(),
                                             shape=(0, self.pixels),
                                             expectedrows=self.pixels)
        #  self.tables.append(self._point_response_function(src_pt, **kwargs).ravel()[None])

    def generate_projection(self, poisson=False):  # TODO: stuck here
        """Generates projection at current linind (position)"""
        self.image = np.zeros([self.y_dim, self.x_dim])
        self.image[self.row, self.col] = 1
        if poisson:
            self.projection = np.floor(np.random.poisson(self.sysmat.dot(self.image.ravel())) + 1)
        else:
            self.projection = np.floor(self.sysmat.dot(self.image.ravel())) + 1

    def reconstruction(self, threshold=0.01):
        self.projection = True  # TODO: Here. Compute_mlem with checks in between. Use self.image as initial guess.
        # Stop if threshold reached

    def generate_psfs(self, rows=None, cols=None):
        pass  # TODO: iterate over rows and columns, all if not provided. Append to open file

    def compute_mlem_full(self, sensitivity=None,
                          det_correction=None,
                          initial_guess=None,
                          nIterations=30,
                          gauss_filter=True,
                          filt_sigma=1/2.355,  # 2.355 sigma = FWHM
                          norm=False,
                          verbose=True,
                          quiet=False,
                          **kwargs):
        """Counts is a projection. Dims is a 2 tuple list of dimensions for each region, sensitivity normalizes iterations
        to detector sensitivity from system response, det_correction is a calibration of det pixel response, initial guess
        is initial guess for image (must be given as a list of image arrays like the output, nIterations is MLEM iterations,
        and filter/filter_sigma apply gaussian filter to ROI space (assumed to be first given region/dim"""
        # self.measurements, self.pixels = sysmat,shape
        if verbose:
            print("Total Measured Counts: ", self.projection.sum())
            print("Check Level (Counts):", self.projection.sum() * 0.001 + 100)
            print("Standard Deviation (Counts):", np.std(self.projection))

        if sensitivity is None:
            sensitivity = np.ones(self.pixels)

        if det_correction is None:
            det_correction = np.ones(self.measurements)

        sensitivity = sensitivity.ravel()
        det_correction = det_correction.ravel()

        sensitivity = sensitivity.ravel()
        det_correction = det_correction.ravel()

        measured = self.projection.ravel() * det_correction

        recon_img = np.ones(self.pixels)

        if initial_guess is None:
            recon_img_previous = np.ones(recon_img.shape)
            self.current_iteration = 0
            self.total_time = 0  # Reset
        else:
            recon_img_previous = initial_guess.flatten()

        itrs = 0
        t1 = time.time()

        sysmat = self.sysmat  # alias
        counts = self.projection  # alias

        diff = np.ones(recon_img.shape) * np.mean(counts)  # NOTE: Added scaling with mean

        while itrs < nIterations and (diff.sum() > (0.001 * counts.sum() + 100)):
            sumKlamb = sysmat.dot(recon_img_previous)
            outSum = (sysmat * measured[:, np.newaxis]).T.dot(1 / sumKlamb)
            # recon_img *= outSum / sensitivity  # TODO: Fix this bug in all code
            recon_img = recon_img_previous * (outSum / sensitivity)

            if self.current_iteration > 5 and gauss_filter:
                recon_img = gaussian_filter(recon_img.reshape([self.y_dim, self.x_dim]), filt_sigma, **kwargs).ravel()
                # gaussian_filter(input, sigma, order=0, output=None, mode='reflect', cval=0.0, truncate=4.0)

            # print('Iteration %d, time: %f sec' % (itrs, time.time() - t1))
            self.total_time += time.time() - t1
            diff = np.abs(recon_img - recon_img_previous)
            if not quiet:
                print('Iteration %d, time: %f sec' % (self.current_iteration, self.total_time))
                print('Diff Sum: ', diff.sum())
            recon_img_previous[:] = recon_img
            itrs += 1
            self.current_iteration += 1

        if verbose:
            print("Total Iterations: ", self.current_iteration)

        if norm:
            self.image = recon_img.reshape([self.y_dim, self.x_dim]) / np.max(recon_img)
            return
        self.image = recon_img.reshape([self.y_dim, self.x_dim])

    def test_point_recon(self, pt=None, poisson=False, **kwargs):
        if pt is None:
            pt = (self.y_dim // 2, self.x_dim // 2)
        self.row = pt[0]
        self.col = pt[1]
        self.generate_projection(poisson=poisson)
        self.compute_mlem_full(**kwargs)

    def set_mask(self, nbs=2):
        self.roi.fill(0)
        self.roi[np.max([0, self.row-nbs]):np.min([self.y_dim, self.row + nbs + 1]), np.max([0, self.col-nbs]):np.min([self.x_dim, self.col + nbs + 1])] = 1
        self.roi_size = np.count_nonzero(self.roi)
        self.bg_size = self.pixels - self.roi_size

    def cnr(self, return_both=False):
        """Calculate CNR of ROI, see set_mask. Return_both returns contrast (top) and noise (bottom) of
        fraction instead of just ratio. All values normalized per pixel"""

        roi_mean = (self.image[self.roi]/self.roi_size).mean()
        bg_mean = (self.image[~self.roi]/self.bg_size).mean()
        bg_std = (self.image[~self.roi]/self.bg_size).std()
        if return_both:
            return np.abs(roi_mean - bg_mean), bg_std
        return np.abs(roi_mean - bg_mean)/bg_std

    def append_pt_response(self, img):
        self.table.append(img.ravel()[None])


def plot_cnr_over_iterations_tst(max_iterations=100):
    sfname = '/Users/justinellin/Desktop/June10sysmat/2021-06-09-1350_SP2.h5'
    psf = psf_generator(sfname, [101, 31])  # sysmat fname, dims, kwargs -> center, pxl_sze

    pt = (16, 25)  # 25, 76
    psf.row = pt[0]
    psf.col = pt[1]
    psf.set_mask()
    psf.generate_projection(poisson=False)
    # current_image = np.copy(psf.image)

    # max_iterations = 100
    contrast = -1 * np.ones(max_iterations)
    std_bg = -1 * np.ones(max_iterations)

    iters = 0
    psf.compute_mlem_full(nIterations=1, initial_guess=None, filt_sigma=0.5 / 2.355, norm=False)  # first iteration

    for idx in np.arange(1, max_iterations):  # convert to while statement
        current_image = np.copy(psf.image)
        psf.compute_mlem_full(nIterations=1, initial_guess=current_image, filt_sigma=0.5 / 2.355, norm=False)
        contrast[idx], std_bg[idx] = psf.cnr(return_both=True)
        iters = idx

    psf.update_plot()
    psf.show_plot()

    x_range = np.arange(max_iterations)
    fig, ax = plt.subplots()
    plt.title("Contrast and Noise per Voxel")
    ax.plot(x_range + 1, contrast, color="red", label="contrast")
    ax.set_xlabel("iteration", fontsize=14)
    ax.set_ylabel("(ROI - BG) counts", color="red", fontsize=14)

    ax2 = ax.twinx()
    ax2.plot(x_range + 1, std_bg, color="blue", label="noise")
    ax2.set_ylabel("STD bkg", color="blue", fontsize=14, rotation=270, va='bottom')
    plt.tight_layout()
    plt.show()

    fig, ax = plt.subplots()
    plt.title("CNR per Voxel")
    ax.plot(x_range + 1, contrast/std_bg, color="red", label="CNR")
    ax.set_xlabel("iteration", fontsize=14)
    ax.set_ylabel("CNR", fontsize=14)
    plt.show()


def plot_cnrs_and_iters(cnr_file, center=(0, -10), x_dim=101, y_dim=31, pxl_size=2):
    # cnr_file = '/Users/justinellin/PycharmProjects/varian/Processed/cnr_data_06-21.npz'
    # data = np.load(cnr_file)
    with np.load(cnr_file) as data:
        max_cnr = data['max_cnr']
        iter_max = data['iter_max_cnr']
    # fig = plt.figure(figsize=(9, 5), constrained_layout=False)
    extent_x = np.array([1, 1]) * center[0] + (np.array([-1, 1]) * (x_dim * pxl_size)) / 2
    extent_y = np.array([1, 1]) * center[1] + (np.array([-1, 1]) * (y_dim * pxl_size)) / 2
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 2))
    im1 = ax1.imshow(max_cnr, cmap='magma', origin='upper', interpolation='nearest', extent=np.append(extent_x, extent_y))
    ax1.set_xlabel('mm')
    ax1.set_ylabel('mm')
    ax1.set_title("Max CNR")
    fig.colorbar(im1, ax=ax1, fraction=0.015, pad=0.04)
    im2 = ax2.imshow(iter_max, cmap='viridis', origin='upper', interpolation='nearest', extent=np.append(extent_x, extent_y))
    ax2.set_xlabel('mm')
    ax2.set_ylabel('mm')
    ax2.set_title("Iteration Max")
    fig.colorbar(im2, ax=ax2, fraction=0.015, pad=0.04)
    fig.suptitle('CNR analysis (first 100 iterations)')
    fig.tight_layout()
    plt.show()


def main():
    sfname = '/Users/justinellin/Desktop/June10sysmat/2021-06-09-1350_SP2.h5'
    psf = psf_generator(sfname, [101, 31])  # sysmat fname, dims, kwargs -> center, pxl_sze

    pt = (16, 25)  # 25, 76
    iterations = 2
    # psf.test_point_recon(pt=pt, nIterations=iterations)
    psf.test_point_recon(pt=pt, nIterations=iterations, filt_sigma=0.5/2.355, norm=True, poisson=False)
    psf.update_plot()
    psf.show_plot()


def generate_psf_file(sfname, max_iterations=100, min_iteration=10, size=(201, 61),
                      cols=None, rows=None,
                      gauss_scale=0.5, save_images=False, save_cnr_data=False, center=(0, 0), pxl_sze=1,
                      nbs=2,
                      **kwargs):
    psf = psf_generator(sfname, size, center=center, pxl_sze=pxl_sze)

    iter_max_CNR = np.zeros_like(psf.image)
    max_CNR = np.zeros_like(psf.image)

    if save_images:
        psf.open_file()

    if cols is None:
        cols = np.arange(psf.x_dim)

    if rows is None:
        rows = np.arange(psf.y_dim)

    # TODO: Make this a variable to choose? Check if adding or removing this is a good idea. Check (1)s
    sensitivity = np.sum(psf.sysmat, axis=0)

    # sensitivity = None, det_correction = None, initial_guess = None, Iterations = 30, gauss_filter = True,
    # filt_sigma = 1 / 2.355,  # 2.355 sigma = FWHM, norm = False, verbose = True, quiet = False

    for row in rows:
        psf.row = row
        for col in cols:
            psf.col = col
            psf.set_mask(nbs=nbs)
            psf.generate_projection(poisson=False)
            psf.compute_mlem_full(nIterations=1, initial_guess=None, filt_sigma=gauss_scale / 2.355,
                                  sensitivity=sensitivity,  # TODO (1): Add or remove
                                  **kwargs)  # first iteration

            best_iteration = 0
            best_cnr = psf.cnr(return_both=False)
            best_image = psf.image.copy()  # in terms of CNR

            for idx in np.arange(1, max_iterations):  # convert to while statement
                current_image = np.copy(psf.image)
                psf.compute_mlem_full(nIterations=1, initial_guess=current_image,
                                      sensitivity=sensitivity,  # TODO (1): Add or remove
                                      filt_sigma=gauss_scale / 2.355, **kwargs)
                new_cnr = psf.cnr(return_both=False)
                if (new_cnr > best_cnr) or (idx == (int(min_iteration)-1)):
                    best_cnr = new_cnr
                    best_iteration = idx
                    best_image = psf.image.copy()

            max_CNR[row, col] = best_cnr
            iter_max_CNR[row, col] = best_iteration
            if psf.file_open:
                psf.append_pt_response(best_image)

        print("Row {r} of {m} complete.".format(r=row + 1, m=psf.y_dim))

    if save_cnr_data:
        fname = os.path.join(os.getcwd(), 'Processed', 'cnr_data_' + datetime.now().strftime("%m-%d"))
        np.savez(fname, max_cnr=max_CNR, iter_max_cnr=iter_max_CNR)

    psf.close_file()

    # plt.imshow(max_CNR, cmap='magma', origin='upper', interpolation='nearest')
    # plt.show()

    # plt.imshow(iter_max_CNR, cmap='magma', origin='upper', interpolation='nearest')
    # plt.show()


def save_psf_file():
    # cols = [49, 50, 51]
    # rows = [15, 16, 17]
    cols = None
    rows = None
    t1 = time.time()
    # sfname = '/home/justin/repos/sysmat_current/sysmat/design/2021-06-17-1746_SP1.h5'
    # June 30, gauss = 1, nbrs = 2 with 6-17-1746

    sfname = '/home/justin/repos/sysmat_current/sysmat/design/2021-07-03-1015_SP1.h5'
    # July 3 (07-03-1015), gauss = 4, nbrs = 4/2.355 * 2 - 0.5 = 3 rounded
    generate_psf_file(sfname, max_iterations=100, min_iteration=10,
                      cols=cols, rows=rows, size=[201, 61],
                      gauss_scale=4, save_images=True, save_cnr_data=True, verbose=False, quiet=True,
                      pxl_sze=1, center=(0, -10), nbs=3)

    print("Total time: ", time.time() - t1)


def psf_fold_sysmat(sysmat_fname, psf_fname, folded_response_fname='folded_response'):
    sysmat = load_sysmat(sysmat_fname).T  # TODO: This transpose is an issue
    folded_sysmat = np.zeros_like(sysmat)
    h5file = tables.open_file(psf_fname, 'r')
    psfs = h5file.root.psf

    tot_rows = psfs.nrows
    print("Total Rows: ", tot_rows)
    for idx, psf in enumerate(psfs):  # tables.iterrows
        folded_sysmat[idx] = np.sum(psf[:, np.newaxis] * sysmat, axis=0)
        if idx % 1000 == 1:
            print("Percent Done: ", idx/tot_rows)
    # sysmat_file_obj.root.sysmat[:].T
    np.save(folded_response_fname, folded_sysmat)


if __name__ == "__main__":
    # main()
    # plot_cnr_over_iterations_tst(max_iterations=100)
    # plot_cnrs_and_iters('/home/justin/repos/sysmat_current/sysmat/design/Processed/cnr_data_07-05.npz'
    #                   , center=(0, -10), x_dim=201, y_dim=61, pxl_size=1)

    # save_psf_file()
    # psf_fold_sysmat('/home/justin/repos/sysmat_current/sysmat/design/2021-06-17-1746_SP1.h5',
    #                '/home/justin/repos/sysmat_current/sysmat/design/Processed/psf_2021-06-17-1746_SP1_proc_06-26.h5',
    #                folded_response_fname='June30_folded_g1')

    psf_fold_sysmat('/home/justin/repos/sysmat_current/sysmat/design/2021-07-03-1015_SP1.h5',
                    '/home/justin/repos/sysmat_current/sysmat/design/Processed/psf_2021-07-03-1015_SP1_proc_07-04.h5',
                    folded_response_fname='July6_folded_g4')
