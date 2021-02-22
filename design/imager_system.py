import matplotlib.pyplot as plt
import time
import numpy as np
import tables
import os
from datetime import datetime
from design.apertures import Collimator
from design.sources import Sources
from design.detector_system import Detector, Detector_System


class Imager(object):
    def __init__(self, system_axis=np.array([0, 0, 1])):
        self.system_central_axis = system_axis  # Global axis of system
        self.collimator = Collimator(self)

        self.detector = Detector()
        self.detectors = []  # This keeps track of it all
        self.detector_system = Detector_System(self)

        self.sources = Sources()
        self.subsample = 0  # I need to go through and change all of these
        self.sample_area = 0
        self.sample_step = 0.1  # in mm

        self.projection = 0

    def set_collimator(self, **kwargs):
        self.collimator = Collimator(self, **kwargs)

    def create_detector(self, **kwargs):
        self.detectors.append(Detector(**kwargs))
        if not len(self.detectors) == 1:
            self.sample_area = self.detectors[0].pix_size ** 2
            self.set_detectors = True

    def set_sources(self, **kwargs):
        self.sources = Sources(**kwargs)

    def generate_ray(self, src_pt, det_pt):  # generates the ray once
        em_dir = norm(det_pt - src_pt)
        max_scalar = (self.detector_system.farthest_plane - src_pt[2])/em_dir[2]  # This is a max length in mm
        return src_pt + em_dir * np.arange(0, max_scalar + self.sample_step, self.sample_step)[:, np.newaxis], em_dir

    def generate_test_response(self, tst_pt, **kwargs):  # TODO: You are here. Don't forget to set mu and rho
        try:
            self.subsample = kwargs['subsample']
        except:
            self.subsample = 0

        return self._point_response_function(tst_pt, **kwargs)

    def generate_sysmat_response(self, **kwargs):  # subsample = 0, mid = True

        try:
            self.subsample = kwargs['subsample']
        except:
            self.subsample = 0

        # self.subsample = subsample
        # self.sample_area = np.prod(self.detectors[0].pix_size / (2 ** subsample))

        current_datetime = datetime.now().strftime("%Y-%m-%d-%H%M")
        save_fname = os.path.join(os.getcwd(), current_datetime + '_SP' + str(self.subsample) + '.h5')

        self.projection = np.zeros(self.detector_system.detectors[0].npix[::-1] * self.detector_system.layout)
        src_pts = np.prod(self.sources.npix)

        file = tables.open_file(save_fname, mode="w", title="System Response")
        pt_responses = file.create_earray('/', 'sysmat',
                                          atom=tables.atom.Float64Atom(),
                                          shape=(0, np.prod(self.detectors[0].npix)),
                                          expectedrows=src_pts)

        prog = 0
        perc = 0

        self.detector_system.initialize_arrays()

        for src_pt in self.sources.source_pt_iterator():
            pt_responses.append(self._point_response_function(src_pt, **kwargs).ravel())
            file.flush()
            if (prog + 1) / src_pts > 0.02:
                perc += 2
                print("Progress (percent): ", perc)
                prog = 0
                continue
            prog += 1
        file.close()

    def _point_response_function(self, src_pt, **kwargs):
        attenuation_collimator = 1
        for det_end_pt in self.detector_system.generate_det_sample_pts(**kwargs):  # TODO: hunt for inheritance
            ray, em_dir = self.generate_ray(src_pt, det_end_pt)
            attenuation_collimator = self.collimator._collimator_ray_trace(ray)
            self.detector_system._ray_projection(ray, em_dir)
        return self.projection * attenuation_collimator / ((2 ** self.subsample) ** 2)


def norm(array):
    arr = np.array(array)
    return arr / np.sqrt(np.dot(arr, arr))
