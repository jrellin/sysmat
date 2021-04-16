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
# TODO: The purpose of this is to deal with the table mostly in generate system response


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
        except Exception as e:
            print(e)
            print("Going with subsample of 1")
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
            if np.abs(src_pt[2]) > np.abs(self.collimator.colp[2] + self.collimator.col_half_thickness):
                attenuation_collimator = 1  # in between collimator and detectors
            else:
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
    return np.array([np.cos(angle), np.sin(angle), 0])
# TODO: This was always missing the z coordinate. Didn't seem to affect anything but heads up


def main_table():
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

    # ====== Open Sides ======
    cw = 203.2  # collimator width in each dimension
    bot_colly = -cw/2
    open_width = (5.95 + 61.75)  # plate thickness + bottom of tungsten to plate
    table_posy = bot_colly - open_width
    system.collimator.add_aperture('slit', size=open_width, slit_ax=ang2arr(0), aper_angle=0,
                                   chan_length=(system.collimator.col_half_thickness * 2),
                                   y_min=table_posy, y_max=bot_colly,
                                   loc=np.array([0, bot_colly - (open_width/2), 0]))  # bottom opening to table

    # system.collimator.add_aperture('slit', size=cw, aper_angle=0,
    #                               chan_length=(system.collimator.col_half_thickness * 2),
    #                               x_min=bot_colly-cw, x_max=bot_colly,
    #                               loc=np.array([bot_colly - (cw/2), 0, 0]))  # -x side opening

    r_open_wid = 2000  # open to left of collimator
    right_opening = (cw/2) + (r_open_wid/2)
    system.collimator.add_aperture('slit', size=r_open_wid, aper_angle=0,
                                   chan_length=(system.collimator.col_half_thickness * 2),
                                   x_min=cw/2, x_max=(cw/2) + r_open_wid,
                                   loc=np.array([right_opening, 0, 0]))  # +x side opening

    # ==================== Detectors ====================

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
    # ~ Beam stop
    vs = 50
    offset = 1  # 1 mm gap between this beamstop area and imager FoV
    fov_x = 1000
    system.sources.sc = np.array([(cw/2) + offset + (fov_x/2), -10, -20])
    # Center is 20 mm away from collimator (closer to obj)
    system.sources.vsze = vs  # 5 CM steps
    system.sources.npix = np.array([(fov_x//vs) + 1, (200//vs) + 1])
    # ~ Table ~
    # system.sources.sc = np.array([-200, table_posy, -110])  # Center is 20 mm away from collimator (closer to obj)
    # system.sources.s_ax[1] = np.array([0, 0, 1])  # positive z
    # system.sources.vsze = 10  # CM steps
    # system.sources.npix = np.array([19, 23])  # I.E. 200 cm across in beam direction and 230 cm from object to dets
    # Starts at z = 0 (object plane) then goes back to z = -260, sweeps from neg X (near beam port)
    # to positive X (near target) for each z

    # ~ Shielding ~
    # shielding_angle = np.deg2rad(45.)  # radians
    # extent_proj_z = 110.  # mm  (projection along z axis)
    # scx = bot_colly - (np.cos(shielding_angle) * extent_proj_z/2) - 10  # 1 cm from edge to collimator
    # scy = -open_width/2
    # scz = system.collimator.colp[2] - (np.sin(shielding_angle) * extent_proj_z/2)
    # print("Source Center: ({x},{y},{z})".format(x=scx, y=scy, z=scz))

    # system.sources.sc = np.array([scx, scy, scz])  # Center is 20 mm away from collimator (closer to obj)
    # system.sources.s_ax[0] = np.array([np.cos(shielding_angle), 0, np.sin(shielding_angle)])  # along shield
    # system.sources.vsze = 10  # CM steps

    # steps0 = ( extent_proj_z/(2 * np.cos(shielding_angle)) ) // system.sources.vsze
    # vsze is really just a pixel size
    # steps1 = (np.abs(bot_colly) + (open_width/2)) // system.sources.vsze
    # system.sources.npix = np.array([(2 * steps0 + 1), (2 * steps1 + 1)])
    # print("Npix: ", system.sources.npix)

    # ==================== Attenuation ====================
    # system.collimator.mu = 0.04038 * (10 ** 2)
    # system.collimator.mu = 1000

    # ==================== Run and Display ====================
    system.sample_step = 0.1  # in mm, default
    system.subsample = 0  # powers of 2

    system.generate_sysmat_response()
    print('It took ' + str(time.time() - start) + ' seconds.')


if __name__ == "__main__":
    # TODO: Problem with _check_intersection in detector_system if source point past initial plane
    main_table()
