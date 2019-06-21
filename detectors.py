import numpy as np


#  Global origin (0,0,0) should be in the center of the collimator on the detector-collimator axis
# This function will eventually handle exact detector orientations


class Detector(object):
    def __init__(self, center=(0, 0, 100),  # mm Coordinates of the center face of
                 # the detector (facing the object plane)
                 det_norm=(0, 0, -1),  # unit vector facing collimator
                 det_thickness=20,  # in mm
                 npix_1=12, npix_2=12,
                 pix1_size=4, pix2_size=4,  # in mm
                 ax_1=(1, 0, 0), ax_2=(0, 1, 0),
                 # det_array_size = (4,4),
                 ind=(0, 0),
                 subpix=0):  # subpixels = 2**subpix
        self.c = np.array(center)
        self.norm = np.array(det_norm)
        self.thickness = det_thickness
        self.npix = np.array((npix_1, npix_2))
        self.pix_isodd = self.npix & 0x1
        self.pix_size = np.array(([pix1_size, pix2_size]))
        self.axes = np.array([ax_1, ax_2])
        self.f_plane = np.dot(self.c, self.norm)
        self.b_plane = np.dot(self.c + ((-1) * self.thickness * self.norm), self.norm)
        self.hist_ax0 = (np.arange(-self.npix[0]/2., self.npix[0]/2. + 1)) * self.pix_size[0]
        self.hist_ax1 = (np.arange(-self.npix[1]/2., self.npix[1]/2. + 1)) * self.pix_size[1]
        # self.hist_ax0 = self.npix[0] + 0.5 + 0.5 * int((self.npix[0] & 0x1) ^ 1)) * self.pix_size[0]

    def face_pts(self, back=False):  # back is True means back plane
        ax0_scalars = np.arange((-self.npix[0]/2. + 0.5),
                                (self.npix[0]/2. + 0.5))

        # ax1_scalars = np.arange((-self.npix[1] + 0.5 + 0.5 * int((self.npix[1] & 0x1) ^ 1)) * self.pix_size[1],
        #                         (self.npix[1] + 0.5 + 0.5 * int((self.npix[1] & 0x1) ^ 1)) * self.pix_size[1])[::-1]
        ax1_scalars = np.arange((-self.npix[1]/2. + 0.5),
                                (self.npix[1]/2. + 0.5))[::-1]
        # Reversed ordering

        ax0_vec = np.outer(ax0_scalars, self.axes[0])
        ax1_vec = np.outer(ax1_scalars, self.axes[1])

        centers = (ax0_vec[:, np.newaxis] + ax1_vec[np.newaxis, :]).reshape(-1, 3)

        return centers.reshape(self.npix[0], self.npix[1]) + (back * (-1) * self.thickness * self.norm)

    def crystal_intersection(self, emission_pt, emission_dir, step=0.01, prefactor = 1):  # step in mm

        r_o = (self.f_plane - np.dot(emission_pt, self.norm)) / (np.dot(emission_dir, self.norm) * 1.0)
        omegas = np.arange(
            float(r_o),
            float((self.b_plane - np.dot(emission_pt, self.norm)) / (1.0 * np.dot(emission_dir), self.norm) + step),
            step
        )
        theta = np.abs(np.dot(np.vstack([self.norm, self.axes]), emission_dir))
        # theta_f = np.dot(-emission_dir, self.norm)
        # theta_0 = np.abs(np.dot(emission_dir, self.axes[0]))
        # theta_1 = np.abs(np.dot(emission_dir, self.axes[1]))

        area = np.prod(self.pix_size) * np.cos(theta[0]) + (self.pix_size[1] * np.cos(theta[1]) +
                                                           self.pix_size[0] * np.cos(theta[2])) * self.thickness
        ray = np.outer(omegas, emission_dir) + emission_pt
        intersection = (np.abs(np.dot(ray, self.axes[0])) < self.hist_ax0[-1]) & \
                       (np.abs(np.dot(ray, self.axes[1])) < self.hist_ax1[-1])
        intersection_rays = ray[intersection]
        crystal_ind = np.floor(intersection_rays[:, :2] + 0.5 * self.pix_isodd) + (0.5 * (self.pix_isodd ^ 1))
        # intersection_distances = omegas[intersection] Redundant since step size is constant
        # omegas[intersection][[0, -1]] # First and last value
        # TODO: Next step is histogramming each of these points as they pass through the detector
        # TODO: self.f_plane should be dotted with something.

