import numpy as np
#  Global origin (0,0,0) should be in the center of the collimator on the detector-collimator axis


class Detector(object):
    def __init__(self, center=(0, 0, 100),  # mm Coordinates of the center face of
                 # the detector (facing the object plane)
                 det_norm=(0, 0, -1),  # unit vector facing collimator
                 det_thickness=40,  # in mm
                 npix_1=12, npix_2=12,
                 pix1_size=4, pix2_size=4,  # in mm
                 ax_1=(1, 0, 0), ax_2=(0, 1, 0),
                 subpix=0):  # subpixels = 2**subpix
        self.c = np.array(center)
        self.norm = np.array(det_norm)
        self.thickness = det_thickness
        self.npix = np.array((npix_1, npix_2))
        self.pix_size = np.array(([pix1_size, pix2_size]))
        self.axes = np.array([ax_1, ax_2])

    def face_pts(self, back=False):  # back is True means back plane
        ax0_scalars = np.arange(-self.npix[0] + 0.5 + 0.5 * int((self.npix[0] & 0x1) ^ 1),
                                self.npix[0] + 0.5 + 0.5 * int((self.npix[0] & 0x1) ^ 1))

        ax1_scalars = np.arange(-self.npix[1] + 0.5 + 0.5 * int((self.npix[1] & 0x1) ^ 1),
                                self.npix[1] + 0.5 + 0.5 * int((self.npix[1] & 0x1) ^ 1))[::-1]  # Reversed ordering

        ax0_vec = np.outer(ax0_scalars, self.axes[0])
        ax1_vec = np.outer(ax1_scalars, self.axes[1])

        centers = (ax0_vec[:, np.newaxis] + ax1_vec[np.newaxis, :]).reshape(-1, 3)

        return centers.reshape(self.npix[0], self.npix[1]) + (back * (-1) * self.thickness * self.norm)


