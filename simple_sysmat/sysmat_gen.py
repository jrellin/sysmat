import numpy as np
import pandas as pd
# Really version 2

# Obj| -> Col| -> Image +z ->


class Collimator(object):
    _sup_aper = ('slit', 'pinhole')

    def __init__(self,
                 x_limits=(-100.0, 100.0),  # in mm
                 y_limits=(-100.0, 100.0),  # in mm
                 coll_norm=np.array([0, 0, -1]),
                 coll_plane=np.array([0, 0, 100])  # in mm
                 ):
        self.xlim = np.array(x_limits)
        self.ylim = np.array(y_limits)
        self.norm = norm(coll_norm)
        self.colp = np.array(coll_plane)
        self.apertures = []

    def add_aperture(self, type, **kwargs):
        if type not in self._sup_aper:
            raise ValueError('Aperture type {f} is not supported. '
                             'Supported aperture types: {a}'.format(f=type, a=str(self._sup_aper)[1:-1]))
        if type is 'slit':
            self.apertures.append(Slit(**kwargs))
        if type is 'pinhole':
            self.apertures.append(Pinhole(**kwargs))

    # def ray_trace(self, det1=48, det2=48, *args):
    def ray_trace(self, end_pts, em_pt, *args):
        proj = np.zeros([48, 48])
        dirs, r_sq = ray_params(end_pts, em_pt)
        mag = self.colp[2] / dirs[:, 2]

        for aper in self.apertures:
            # proj *= aper.ray_pass(*args).reshape(det2, det1)
            proj += aper.ray_pass(dirs, mag).reshape(48, 48)

        proj[proj > 1] = 1.0
        return proj
# TODO: Allow for Subpixeling


class Pinhole(object):  # TODO: Check pinhole works
    def __init__(self, size=2.0,  # in mm
                 loc=(0, 0, 0),  # in mm of center relative to collimator center
                 axis=(0, 0, -1),  # normalized axis of pinhole towards object
                 aper_angle=20.0,  # in degrees. Full opening
                 # coll_norm = np.array([1, 0, 0])
                 ):
        self.sze = size
        self.c = np.array(loc)  # Center of pinhole
        self.ax = norm(axis)
        self.h_ang = np.deg2rad(aper_angle / 2.)
        # self.z_ax = coll_norm

    def ray_pass(self, dirs, mag):  # dirs should be normalized as rows, em_pt is point of emission
        hole_hit = (np.sum(((dirs * mag[:, np.newaxis]) - self.c)**2)) < (self.sze ** 2)
        angle_good = np.arccos(np.abs(np.dot(dirs, self.ax))) < self.h_ang
        # return (hole_hit & angle_good).astype(int)
        return hole_hit & angle_good


class Slit(object):
    def __init__(self, size=2., loc=(0., 0., 0.),  # in mm relative to collimator center
                 cen_ax=(0., 0., -1.),  # normal to opening pointing toward object space
                 slit_ax=(0., 1., 0.),  # points along opening
                 aper_angle=20.0,  # Full opening angle in degrees
                 # chan_length=0,
                 x_min=-np.inf,
                 x_max=np.inf,
                 y_min=-np.inf,
                 y_max=np.inf,
                 ):
        # self.z_ax = coll_norm
        self.sze = size / 2.  # Half-width of aperture in mm
        self.c = np.array(loc)  # Some point along center slit plane
        self.f = norm(cen_ax)  # Normal to plane of aperture (focus)
        # self.tube = chan_length  # Total length
        self.x_lim = np.sort([x_min, x_max])
        self.y_lim = np.sort([y_min, y_max])

        self.a = norm(slit_ax)  # axis of slit
        self.h_ang = np.deg2rad(aper_angle / 2.)  # Half of the opening angle of the slit in radians
        self.plane = np.cross(self.a, self.f)  # Vector that is in direction of the slit opening

    def ray_pass(self, dirs, mag):
        rays_coll = dirs * mag[:, np.newaxis]
        # TODO: See where points are on collimator plane
        near_slit = (rays_coll[:, 0] >= self.x_lim[0]) & (rays_coll[:, 0] <= self.x_lim[1]) & \
                    (rays_coll[:, 1] >= self.y_lim[0]) & (rays_coll[:, 1] <= self.y_lim[1])
        hole_hit = np.abs(np.dot(rays_coll - self.c, self.plane)) < self.sze
        # angles = np.arctan(np.abs(np.dot(dirs, self.plane)/np.dot(dirs, self.f)))
        angle_good = np.arctan(np.abs(np.dot(dirs, self.plane)/np.dot(dirs, self.f))) < self.h_ang
        return (near_slit & hole_hit & angle_good).astype(int)
        # True is when it passes through the slit


class Detector(object):

    def __init__(self, center=(0, 0, 200),  # mm Coordinates of the center face of
                 # the detector (facing the object plane)
                 det_norm=(0, 0, -1),  # unit vector facing collimator
                 det_thickness=20.0,  # in mm
                 npix_1=48, npix_2=48,
                 pix1_size=4.0, pix2_size=4.0,  # in mm
                 ax_1=(1, 0, 0),
                 ax_2=(0, 1, 0),
                 subpix=0):  # subpixels = 2**subpix
        self.c = np.array(center)
        self.norm = np.array(det_norm)
        self.thickness = det_thickness
        self.npix = np.array((npix_1, npix_2))
        self.pix_isodd = self.npix & 0x1
        self.pix_size = np.array(([pix1_size, pix2_size]))
        self.axes = np.array([ax_1, ax_2])

        self.f_plane = np.dot(self.c, self.norm)
        # unused for now
        self.hist_ax0 = (np.arange(-self.npix[0]/2., self.npix[0]/2. + 1)) * self.pix_size[0]
        self.hist_ax1 = (np.arange(-self.npix[1]/2., self.npix[1]/2. + 1)) * self.pix_size[1]

    def face_pts(self, back=False):  # back is True means back plane
        ax0_scalars = np.arange((-self.npix[0]/2. + 0.5),
                                (self.npix[0]/2. + 0.5))

        ax1_scalars = np.arange((-self.npix[1]/2. + 0.5),
                                (self.npix[1]/2. + 0.5))[::-1]
        # Reversed ordering

        ax0_vec = np.outer(ax0_scalars, self.axes[0])
        ax1_vec = np.outer(ax1_scalars, self.axes[1])

        return (ax0_vec[:, np.newaxis] + ax1_vec[np.newaxis, :]).reshape(-1, 3) + self.c

        # return centers.reshape(self.npix[0], self.npix[1]) + (back * (-1) * self.thickness * self.norm)
        # return centers.reshape(self.npix[0], self.npix[1], 3) + (back * (-1) * self.thickness * self.norm)


class Sources(object):
    def __init__(self, center=(0, 0, 0),  # mm Coordinates of the center of Source Space
                 voxel_size=1.0,  # in mm, this is pixel size for 2D
                 npix_1=200,
                 npix_2=200,
                 # npix_3=200,  # For 3D source space
                 sax_1=(1, 0, 0),  # (s)ource (ax)is
                 sax_2=(0, 1, 0)
                 # ,sax_3=(0,0,1)  # For 3D source space
                 ):
        self.sc = center
        self.vsze = voxel_size
        self.npix = np.array((npix_1, npix_2))  # np.array((npix_1, npix_2, npix_3))

        self.s_ax = np.array([sax_1, sax_2])  # np.array([sax_1, sax_2, sax_3])
        # self.sax3 = sax_3

    def source_pts(self):  # This could be amended for 3D easily
        ax0_scalars = np.arange((-self.npix[0] / 2. + 0.5),
                                (self.npix[0] / 2. + 0.5))

        ax1_scalars = np.arange((-self.npix[1] / 2. + 0.5),
                                (self.npix[1] / 2. + 0.5))[::-1]

        ax0_vec = np.outer(ax0_scalars, self.s_ax[0])
        ax1_vec = np.outer(ax1_scalars, self.s_ax[1])

        centers = (ax0_vec[:, np.newaxis] + ax1_vec[np.newaxis, :]).reshape(-1, 3)  # TODO: Add source origin
        # c1 = centers[:, np.newaxis]


def ray_params(end_pts, em_pt):
    rays = 1.0 * (end_pts - em_pt)
    dirs = rays / np.sqrt((rays ** 2).sum(axis=1))[:, np.newaxis]
    # return rays, dirs, (rays ** 2).sum(axis=1)
    return dirs, (rays ** 2).sum(axis=1)


def norm(array):
    arr = np.array(array)
    return arr/np.sqrt(np.dot(arr, arr))
