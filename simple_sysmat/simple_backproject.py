import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np

# This is for Davis

# beam -> +x
# skyward -> +y
# Image| -> Col| -> Object +z ->
# ex: Image at z = -200. Col at -100. Object at 0


class Imager(object):
    def __init__(self):
        self.collimator = Collimator()
        self.detector = Detector()
        self.sources = Sources()
        self.subsample = 0

    def _point_response(self, face_pts, em_pt):
        projection = self.collimator.ray_trace(face_pts, em_pt,
                                               subsampling=self.detector.subsamples, det_pixels=self.detector.npix)[0]
        # if not self.detector.subsamples:
        #     return projection

        num_subpixels = 2 ** self.detector.subsamples
        m, n = num_subpixels * self.detector.npix
        return projection.reshape(m // num_subpixels, num_subpixels, n // num_subpixels,
                                  num_subpixels).mean(axis=(1, 3)).ravel()


class Collimator(object):
    _sup_aper = ('slit', 'pinhole')

    def __init__(self, x_limits=(-100.0, 100.0),  # in mm
                 y_limits=(-100.0, 100.0),  # in mm
                 coll_norm=np.array([0, 0, 1]),  # normal facing object
                 coll_plane=np.array([0, 0, -100])  # in mm
                 ):
        self.xlim = np.array(x_limits)
        self.ylim = np.array(y_limits)
        self.norm = norm(coll_norm)
        self.colp = np.array(coll_plane)
        self.apertures = []

    def add_aperture(self, aperture, **kwargs):
        if aperture not in self._sup_aper:
            raise ValueError('Aperture type {f} is not supported. '
                             'Supported aperture types: {a}'.format(f=aperture, a=str(self._sup_aper)[1:-1]))
        if aperture == 'slit':
            self.apertures.append(Slit(**kwargs))
        if aperture == 'pinhole':
            self.apertures.append(Pinhole(**kwargs))

    def ray_trace(self, *args, subsampling=0, det_pixels=np.array([48, 48])):
        proj_size = (2**subsampling) * det_pixels
        proj = np.zeros(np.prod(proj_size))
        ray_dirs, ray_int, r_sq = self.ray_gen(*args)
        for aper in self.apertures:
            proj += aper.ray_pass(ray_dirs, ray_int)
        # return ((proj >= 1)/r_sq).reshape(proj_size), ray_int
        return ((proj >= 1) / r_sq) * np.sin(-ray_dirs[:, 2]), ray_int

    def ray_gen(self, end_pts, em_pt):
        rays = 1.0 * (end_pts - em_pt)
        dirs = rays / np.sqrt((rays ** 2).sum(axis=1))[:, np.newaxis]
        mag = (self.colp[2] - em_pt[2]) / dirs[:, 2]  # fixed probably
        # print("Mag: ", mag[:5])
        intersection = em_pt + dirs * mag[:, np.newaxis]
        return dirs, intersection, (rays ** 2).sum(axis=1)  # Directions, intersection with coll, r^2


class Pinhole(object):
    def __init__(self, size=2.0,  # in mm
                 loc=(0, 0, 0),  # in mm of center relative to collimator center
                 axis=(0, 0, 1),  # normalized axis of pinhole towards object
                 aper_angle=20.0,  # in degrees. Full opening
                 # coll_norm = np.array([1, 0, 0])
                 ):
        self.sze = size
        self.c = np.array(loc)  # Center of pinhole
        self.ax = norm(axis)
        self.h_ang = np.deg2rad(aper_angle / 2.)
        # self.z_ax = coll_norm

    def ray_pass(self, dirs, intersect):  # dirs should be normalized as rows, em_pt is point of emission
        hole_hit = (np.sum((intersect - self.c)**2)) < (self.sze ** 2)
        angle_good = np.arccos(np.abs(np.dot(dirs, self.ax))) < self.h_ang
        return hole_hit & angle_good


class Slit(object):
    def __init__(self, size=2., loc=(0., 0., 0.),  # in mm relative to collimator center
                 cen_ax=(0., 0., 1.),  # normal to opening pointing toward object space
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

    def ray_pass(self, dirs, intersect):
        rays_coll = intersect
        near_slit = (rays_coll[:, 0] >= self.x_lim[0]) & (rays_coll[:, 0] <= self.x_lim[1]) & \
                    (rays_coll[:, 1] >= self.y_lim[0]) & (rays_coll[:, 1] <= self.y_lim[1])
        hole_hit = np.abs(np.dot(rays_coll - self.c, self.plane)) < self.sze
        angle_good = np.arctan(np.abs(np.dot(dirs, self.plane)/np.dot(dirs, self.f))) < self.h_ang
        return near_slit & hole_hit & angle_good
        # True is when it passes through the slit
        # TODO: near_slit must be modified if slits have ends that do not terminate into another slit or the edge


class Detector(object):

    def __init__(self, center=(0, 0, -200),  # mm Coordinates of the center face of
                 # the detector (facing the object plane)
                 det_norm=(0, 0, 1),  # unit vector facing collimator
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
        self.pix_size = np.array(([pix1_size, pix2_size]))
        self.axes = np.array([ax_1, ax_2])

        self.f_plane = np.dot(self.c, self.norm)
        self.subsamples = 0
        # unused for now
        # self.hist_ax0 = (np.arange(-self.npix[0]/2., self.npix[0]/2. + 1)) * self.pix_size[0]
        # self.hist_ax1 = (np.arange(-self.npix[1]/2., self.npix[1]/2. + 1)) * self.pix_size[1]

    def face_pts(self, back=False, subsample=0):  # back is True means back plane
        subpixels = 2 ** subsample
        self.subsamples = subpixels
        ax0_scalars = np.arange((-self.npix[0] * subpixels) / 2 + 0.5,
                                (self.npix[0] * subpixels) / 2 + 0.5)[::-1] * self.pix_size[1] / subpixels
        ax1_scalars = np.arange((-self.npix[1] * subpixels)/2 + 0.5,
                                (self.npix[1] * subpixels)/2 + 0.5)[::-1] * self.pix_size[1] / subpixels
        # Reversed ordering

        ax0_vec = np.outer(ax0_scalars, self.axes[0])
        ax1_vec = np.outer(ax1_scalars, self.axes[1])

        return (ax1_vec[:, np.newaxis] + ax0_vec[np.newaxis, :]).reshape(-1, 3) + self.c, self.npix, subsample

        # return (ax0_vec[:, np.newaxis] + ax1_vec[np.newaxis, :]).reshape(-1, 3) + self.c
        # return centers.reshape(self.npix[0], self.npix[1]) + (back * (-1) * self.thickness * self.norm)
        # return centers.reshape(self.npix[0], self.npix[1], 3) + (back * (-1) * self.thickness * self.norm)


class Sources(object):
    def __init__(self, center=(0, 0, 0),  # mm Coordinates of the center of Source Space
                 voxel_size=1.0,  # in mm, this is pixel size for 2D
                 npix_1=200,
                 npix_2=200,
                 sax_1=(1, 0, 0),  # (s)ource (ax)is
                 sax_2=(0, 1, 0),
                 prepend_n_ax1=0,
                 prepend_n_ax2=0  # In the negative direction, this means down and to the left
                 # , append_n_ax1 = 0,  # In the positive direction, this means up and to the right
                 # , append_n_ax2 = 0
                 ):
        self.sc = center
        self.vsze = voxel_size
        self.npix = np.array((npix_1, npix_2))  # np.array((npix_1, npix_2, npix_3))

        self.s_ax = np.array([sax_1, sax_2])  # np.array([sax_1, sax_2, sax_3])
        # self.sax3 = sax_3
        self.prepend = np.array([prepend_n_ax1, prepend_n_ax2])

    def source_pts(self):  # This could be amended for 3D easily
        ax0_scalars = (np.arange((-self.npix[0] / 2. + 0.5),
                                 (self.npix[0] / 2. + 0.5)) - self.prepend[0]) * self.vsze

        ax1_scalars = (np.arange((-self.npix[1] / 2. + 0.5),
                                 (self.npix[1] / 2. + 0.5))[::-1] - self.prepend[1]) * self.vsze  # [::-1]

        ax0_vec = np.outer(ax0_scalars, self.s_ax[0])
        ax1_vec = np.outer(ax1_scalars[::-1], self.s_ax[1])
        return (ax1_vec[:, np.newaxis] + ax0_vec[np.newaxis, :]) + self.sc


def norm(array):
    arr = np.array(array)
    return arr/np.sqrt(np.dot(arr, arr))


def ang2arr(angle_degrees):  # This means the beam-axis (+x) is the reference point for slit angles
    angle = np.deg2rad(angle_degrees)
    return np.array([np.cos(angle), np.sin(angle)])


def main():
    start = time.time()
    system = Imager()

    # Vertical Slits
    slit1 = np.array([0, 0, 0])
    system.collimator.add_aperture('slit', size=2, loc=system.collimator.colp + slit1)

    h_offset = 48   # horizontal distance between vertical slits
    slit2 = np.array([h_offset, 0, 0])
    system.collimator.add_aperture('slit', loc=system.collimator.colp + slit2)
    system.collimator.add_aperture('slit', loc=system.collimator.colp - slit2)

    # Outside slits
    joint = (203.2/2) - 50.39  # y -coordinate of slit connection, 60 degree slits
    slit3 = np.array([h_offset, joint, 0])  # right side
    slit4 = np.array([h_offset, -joint, 0])  # right side
    system.collimator.add_aperture('slit', slit_ax=ang2arr(60.), x_min=h_offset, loc=system.collimator.colp + slit3)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-60.), x_min=h_offset, loc=system.collimator.colp + slit4)

    # Left side outside slits
    ls3 = np.array([-h_offset, joint, 0])
    ls4 = np.array([-h_offset, -joint, 0])
    system.collimator.add_aperture('slit', slit_ax=ang2arr(120.), x_max=-h_offset, loc=system.collimator.colp + ls3)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-120.), x_max=-h_offset, loc=system.collimator.colp + ls4)

    # y = 0 slits
    rtd_slits = np.array([h_offset, 0, 0])  # (r)ight (t)hirty (d)egree slits
    system.collimator.add_aperture('slit', slit_ax=ang2arr(30), x_min=h_offset, loc=system.collimator.colp + rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-30), x_min=h_offset, loc=system.collimator.colp + rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(150), x_max=-h_offset,
                                   loc=system.collimator.colp - rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-150), x_max=-h_offset,
                                   loc=system.collimator.colp - rtd_slits)

    # Inner Slits
    isv_offset = (203.2/2) - 80.94  # inner slit vertical offsets
    system.collimator.add_aperture('slit', slit_ax=ang2arr(45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=system.collimator.colp + np.array([h_offset, isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=system.collimator.colp + np.array([h_offset, -isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=system.collimator.colp + np.array([-h_offset, isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=system.collimator.colp + np.array([-h_offset, -isv_offset, 0]))

    # system.collimator.add_aperture('slit', size=2, slit_ax=(0, 1, 0), x_max=0, loc=system.collimator.colp - slit2)

    # def __init__(self, size=2., loc=(0., 0., 0.),  # in mm relative to collimator center
    #             cen_ax=(0., 0., 1.),  # normal to opening pointing toward object space
    #             slit_ax=(0., 1., 0.),  # points along opening
    #             aper_angle=20.0,  # Full opening angle in degrees
    #             # chan_length=0,
    #             x_min=-np.inf,
    #             x_max=np.inf,
    #             y_min=-np.inf,
    #             y_max=np.inf,
    #             ):

    endpts, det_pix, subsample = system.detector.face_pts(subsample=4)  # 4 seems best between speed and memory
    # print('Endpts (10): ', endpts[:5])

    em_pt = np.array([0, 0, 900])

    projection, ray_int = system.collimator.ray_trace(endpts, em_pt, subsampling=subsample, det_pixels=det_pix)
    new_projection = projection.reshape((2**subsample) * det_pix)
    m, n = new_projection.shape
    # print("M: ", m, ". N: ", n)
    averaged_projection = new_projection.reshape(m//(2 ** subsample), 2 ** subsample, n//(2 ** subsample),
                                                 2 ** subsample).mean(axis=(1, 3))
    print('It took ' + str(time.time() - start) + ' seconds.')
    # plt.imshow(np.log(projection).T,cmap='Reds')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    print("ray_int shape", ray_int.shape)
    print("ray_int first pt:", ray_int[0])
    print("ray_int last pt:", ray_int[-1])
    ax1.scatter(ray_int[:, 0], ray_int[:, 1])
    ax1.set_xlim((-100, 100))
    ax1.set_ylim((-100, 100))
    ax1.set_ylabel('mm')
    ax1.set_xlabel('mm')
    ax1.set_title('Rays cast from {a} cm'.format(a=(em_pt - system.collimator.colp)/10))
    ax1.set_aspect('equal')
    # im = ax2.imshow(new_projection, cmap='viridis', interpolation='nearest')
    im = ax2.imshow(averaged_projection, cmap='viridis', interpolation='nearest')
    ax2.set_aspect('auto')
    ax2.set_title('Projection')
    ax2.set_xticks([], [])
    ax2.set_yticks([], [])
    fig.colorbar(im, ax=ax2)
    plt.show()
    # plt.imshow(projection, cmap='gray')
    # plt.colorbar()
    # plt.show()


if __name__ == "__main__":
    main()
