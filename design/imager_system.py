import matplotlib.pyplot as plt
import time
import numpy as np
import tables
import os
from datetime import datetime
from design.apertures import Collimator
from design.sources import Sources
from design.detector_system import Detector_System
from design.utils import generate_detector_centers_and_norms


class Imager(object):
    def __init__(self, system_axis=np.array([0, 0, 1])):
        self.system_central_axis = system_axis  # Global axis of system
        self.collimator = Collimator(self)

        # self.detector = Detector()
        # self.detectors = []  # This keeps track of it all
        self.detector_system = Detector_System(self)

        self.sources = Sources()
        self.subsample = 0  # I need to go through and change all of these
        self.sample_area = 0
        self.sample_step = 0.1  # in mm

    def set_collimator(self, **kwargs):
        self.collimator = Collimator(self, **kwargs)

    def create_detector(self, **kwargs):
        self.detector_system.add_detector(**kwargs)
        if not len(self.detector_system.detectors) == 1:
            self.sample_area = self.detector_system.detectors[0].pix_size ** 2

    def set_sources(self, **kwargs):
        self.sources = Sources(**kwargs)

    def generate_ray(self, src_pt, det_pt):  # generates the ray once
        em_dir = norm(det_pt - src_pt)
        max_scalar = (1.01 * self.detector_system.farthest_plane - src_pt[2]) / em_dir[2]  # This is a max length in mm
        return src_pt + em_dir * np.arange(0, max_scalar + self.sample_step, self.sample_step)[:, np.newaxis], em_dir

    def generate_test_response(self, tst_pt, **kwargs):  # TODO: You are here. Don't forget to set mu and rho
        try:
            self.subsample = kwargs['subsample']
            print("Found subsample!")
        except:
            self.subsample = 0

        self.detector_system.initialize_arrays()
        return self._point_response_function(tst_pt, **kwargs)

    def generate_sysmat_response(self, **kwargs):  # subsample = 0, mid = True

        try:
            self.subsample = kwargs['subsample']
        except:
            self.subsample = 0
        print("Subsample: ", 2 ** self.subsample)

        self.detector_system.initialize_arrays()
        tot_img_pxls = self.detector_system.projection.size  # This ensures it is first initialized
        print("Total Image Pixels: ", self.detector_system.projection.size)

        current_datetime = datetime.now().strftime("%Y-%m-%d-%H%M")
        save_fname = os.path.join(os.getcwd(), current_datetime + '_SP' + str(self.subsample) + '.h5')

        src_pts = np.prod(self.sources.npix)
        print("Total point source locations: ", src_pts)

        file = tables.open_file(save_fname, mode="w", title="System Response")
        pt_responses = file.create_earray('/', 'sysmat',
                                          atom=tables.atom.Float64Atom(),
                                          shape=(0, tot_img_pxls),
                                          expectedrows=src_pts)

        prog = 0
        perc = 0

        evts_per_percent = np.floor(0.01 * src_pts)

        for src_pt in self.sources.source_pt_iterator():
            pt_responses.append(self._point_response_function(src_pt, **kwargs).ravel()[None])
            file.flush()
            prog += 1
            if prog > 2 * evts_per_percent:
                perc += prog/src_pts * 100 # total points
                print("Current Source Point: ", src_pt)
                print("Progress (percent): ", perc)
                prog = 0
        file.close()

    def _point_response_function(self, src_pt, **kwargs):  # TODO: Add in angular momentum sampling
        self.detector_system.projection.fill(0.0)  # Clear

        for det_end_pt in self.detector_system.generate_det_sample_pts(**kwargs):
            ray, em_dir = self.generate_ray(src_pt, det_end_pt)
            attenuation_collimator = self.collimator._collimator_ray_trace(ray)
            self.detector_system._ray_projection(ray,
                                                 em_dir,
                                                 coll_att=attenuation_collimator/((2 ** self.subsample) ** 2))
        return self.detector_system.projection  # / ((2 ** self.subsample) ** 2)


def norm(array):
    arr = np.array(array)
    return arr / np.sqrt(np.dot(arr, arr))


def ang2arr(angle_degrees):  # This means the beam-axis (+x) is the reference point for slit angles
    angle = np.deg2rad(angle_degrees)
    return np.array([np.cos(angle), np.sin(angle)])


def test(separate=True):
    start = time.time()
    system = Imager()

    # ==================== Collimator ====================
    system.collimator.colp = np.array([0, 0, -130])  # 100 is the default
    # Vertical Slits
    slit1 = np.array([0, 0, 0])
    system.collimator.add_aperture('slit', size=2, loc=slit1)

    h_offset = 48  # horizontal distance between vertical slits
    slit2 = np.array([h_offset, 0, 0])
    system.collimator.add_aperture('slit', loc=slit2)
    system.collimator.add_aperture('slit', loc=-slit2)

    # Outside slits
    joint = (203.2 / 2) - 50.39  # y -coordinate of slit connection, 60 degree slits
    slit3 = np.array([h_offset, joint, 0])  # right side
    slit4 = np.array([h_offset, -joint, 0])  # right side
    system.collimator.add_aperture('slit', slit_ax=ang2arr(60.), x_min=h_offset, loc=slit3)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-60.), x_min=h_offset, loc=slit4)

    # Left side outside slits
    ls3 = np.array([-h_offset, joint, 0])
    ls4 = np.array([-h_offset, -joint, 0])
    system.collimator.add_aperture('slit', slit_ax=ang2arr(120.), x_max=-h_offset, loc=ls3)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-120.), x_max=-h_offset, loc=ls4)

    # y = 0 slits
    rtd_slits = np.array([h_offset, 0, 0])  # (r)ight (t)hirty (d)egree slits
    system.collimator.add_aperture('slit', slit_ax=ang2arr(30), x_min=h_offset, loc=rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-30), x_min=h_offset, loc=rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(150), x_max=-h_offset,
                                   loc=-1 * rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-150), x_max=-h_offset,
                                   loc=-1 * rtd_slits)

    # Inner Slits
    isv_offset = (203.2 / 2) - 80.94  # inner slit vertical offsets
    system.collimator.add_aperture('slit', slit_ax=ang2arr(45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([h_offset, isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([h_offset, -isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([-h_offset, isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([-h_offset, -isv_offset, 0]))

    # for aper in system.collimator.apertures:
    #    print("Aperture Locations: ", aper.c)

    # ==================== Detectors ====================
    # system.create_detector()  # All that was needed originally for 1 centered mod
    # system.detector_system.layout = np.array([1, 1])  # All that was needed originally for 1 centered mod
    # layout = np.array([2, 2])
    layout = np.array([4, 4])
    system.detector_system.layout = layout  # could just put in __init__ of Detector_System

    # mod_spacing_dist = 50  # Start of flat_detector_array
    # scalars = np.arange(-layout[1]/2 + 0.5, layout[1]/2 + 0.5) * mod_spacing_dist

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    x_displacement = 25.447
    distance_mod_plane = np.array([0, 0, -260]) + (x_displacement * x)

    # x_vec = np.outer(scalars, x)  # Start left (near beam port) of beam axis
    # y_vec = np.outer(scalars[::-1], y)  # Start top row relative to ground

    # mod_centers = (y_vec[:, np.newaxis] + x_vec[np.newaxis, :]).reshape(-1, 3) + distance_mod_plane  # end flat array

    mod_centers, directions = generate_detector_centers_and_norms(layout, det_width=53.2, focal_length=420.9)

    for det_idx, det_center in enumerate(mod_centers + distance_mod_plane):
        print("Set det_center: ", det_center)
        # system.create_detector(det_id=det_idx, center=det_center)  # This is for a flat detector array
        system.create_detector(det_id=det_idx, center=det_center, det_norm=directions[det_idx])
    print("Farthest Plane: ", system.detector_system.farthest_plane)

    # ==================== Sources ====================
    em_pt = np.array([0, 0, 1500])

    # ==================== Attenuation ====================
    # system.collimator.mu = 0.04038 * (10 ** 2)
    # system.collimator.mu = 1000

    # ==================== Run and Display ====================
    system.sample_step = 0.1  # in mm, default
    system.subsample = 0  # powers of 2

    point_response = system.generate_test_response(em_pt, subsample=system.subsample)
    print('It took ' + str(time.time() - start) + ' seconds.')

    if np.prod(layout) == 1:
        ax = plt.axes()
        data = point_response
        im = ax.imshow(data)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im)
        plt.show()
        return

    if not separate:
        ax = plt.axes()
        for plt_index in np.arange(layout[1] * layout[0]):
            row = plt_index // layout[1]
            col = plt_index % layout[0]
            section = point_response[(12 * row):(12 * (row + 1)), (12 * col):(12 * (col + 1))]
            point_response[(12 * row):(12 * (row + 1)), (12 * col):(12 * (col + 1))] = np.flipud(section)

        data = point_response
        im = ax.imshow(data)
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im)
        plt.show()
        return

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
    # NOTE: There is a quirk with hard bins with histogram.
    # Left most edge and right most edge do not have same treatment


def main():
    start = time.time()
    system = Imager()

    # ==================== Collimator ====================
    system.collimator.colp = np.array([0, 0, -130])  # 100 is the default

    # Vertical Slits
    slit1 = np.array([0, 0, 0])
    system.collimator.add_aperture('slit', size=2, loc=slit1)

    h_offset = 48  # horizontal distance between vertical slits
    slit2 = np.array([h_offset, 0, 0])
    system.collimator.add_aperture('slit', loc=slit2)
    system.collimator.add_aperture('slit', loc=-slit2)

    # Outside slits
    joint = (203.2 / 2) - 50.39  # y -coordinate of slit connection, 60 degree slits
    slit3 = np.array([h_offset, joint, 0])  # right side
    slit4 = np.array([h_offset, -joint, 0])  # right side
    system.collimator.add_aperture('slit', slit_ax=ang2arr(60.), x_min=h_offset, loc=slit3)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-60.), x_min=h_offset, loc=slit4)

    # Left side outside slits
    ls3 = np.array([-h_offset, joint, 0])
    ls4 = np.array([-h_offset, -joint, 0])
    system.collimator.add_aperture('slit', slit_ax=ang2arr(120.), x_max=-h_offset, loc=ls3)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-120.), x_max=-h_offset, loc=ls4)

    # y = 0 slits
    rtd_slits = np.array([h_offset, 0, 0])  # (r)ight (t)hirty (d)egree slits
    system.collimator.add_aperture('slit', slit_ax=ang2arr(30), x_min=h_offset, loc=rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-30), x_min=h_offset, loc=rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(150), x_max=-h_offset,
                                   loc=-1 * rtd_slits)
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-150), x_max=-h_offset,
                                   loc=-1 * rtd_slits)

    # Inner Slits
    isv_offset = (203.2 / 2) - 80.94  # inner slit vertical offsets
    system.collimator.add_aperture('slit', slit_ax=ang2arr(45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([h_offset, isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([h_offset, -isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(-45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([-h_offset, isv_offset, 0]))
    system.collimator.add_aperture('slit', slit_ax=ang2arr(45),
                                   x_min=-h_offset, x_max=h_offset,
                                   loc=np.array([-h_offset, -isv_offset, 0]))

    # ==================== Detectors ====================
    # layout = np.array([4, 4])
    # system.detector_system.layout = layout  # could just put in __init__ of Detector_System

    x = np.array([1, 0, 0])
    y = np.array([0, 1, 0])
    x_displacement = 25.447
    distance_mod_plane = np.array([0, 0, -260]) + (x_displacement * x)

    mod_centers, directions = generate_detector_centers_and_norms(system.detector_system.layout,
                                                                  det_width=53.2,
                                                                  focal_length=420.9)

    for det_idx, det_center in enumerate(mod_centers + distance_mod_plane):
        print("Set det_center: ", det_center)
        system.create_detector(det_id=det_idx, center=det_center, det_norm=directions[det_idx])
    print("Farthest Plane: ", system.detector_system.farthest_plane)

    # ==================== Sources ====================
    system.sources.sc = np.array([0, -71, -30])
    system.sources.vsze = 2
    system.sources.npix = np.array([121, 31])

    # ==================== Attenuation ====================
    # system.collimator.mu = 0.04038 * (10 ** 2)
    # system.collimator.mu = 1000

    # ==================== Run and Display ====================
    system.sample_step = 0.1  # in mm, default
    system.subsample = 0  # powers of 2

    system.generate_sysmat_response()  # TODO: Save system specs in generated file
    print('It took ' + str(time.time() - start) + ' seconds.')


if __name__ == "__main__":
    # test(separate=False)
    main()
