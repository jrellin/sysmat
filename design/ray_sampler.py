import matplotlib.pyplot as plt
import time
import numpy as np
import tables


class Sampler(object):
    z_axis = np.array([0, 0, 1])  # beam in y-hat direction, z-axis is pointing away from ground, x-axis points to right
    # block_sze = 48  # mm
    blk_ind = np.array([-1.5, -0.5, 0.5, 1.5])  # relative to center, in units of block_size
    mod_indexes = np.arange(4)

    def __init__(self,
                 y_fov=(-100.0, 100.0),  # in mm
                 z_fov=(-100.0, 100.0),  # in mm
                 ror=150,  # in mm
                 plane_angle=90,  # angle in degrees of sample plane relative to beam axis
                 mag=1,  # encapsulates ratios of detector and collimator distances
                 sample_step=4/2,  # sample steps in mm
                 beam_step=5,  # in mm, along beam
                 opening_ang=20,  # in degrees
                 block_sze=48  # size of grid unit in mm
                 ):

        self.y_fov = np.array(y_fov)
        self.z_fov = np.array(z_fov)
        self.b_step = beam_step
        self.s_step = sample_step

        plane_ang = np.deg2rad(plane_angle)
        self.norm_plane = np.array([np.sin(plane_ang), np.cos(plane_ang), 0])  # normal to plane
        self.plane_v1 =  np.array([np.cos(plane_ang), -np.sin(plane_ang), 0])  # v1 perp. to v2 and norm
        self.plane_v2 = self.z_axis  # v2 x v1 = norm
        self.plane = self.norm_plane * ror

        self.block = block_sze
        self.blk_centers = self.blk_ind * self.block * mag
        self.hlf_block = self.block/2
        self.slit_ang = np.deg2rad(opening_ang)

    def test(self):
        pass

    def generate_sample_line(self, sample_ang):  # This just gets shifted, sample_ang rel. to z_axis, +ang towards v1
        # Returns a-hat vector and scaling factors for the line. Check 1/19/21 notes for picture
        ang = np.deg2rad(sample_ang)
        # if np.abs(ang) < (np.pi/4):
        #    tmax = self.block//(self.s_step * np.cos(ang))
        # else:  # greater than 45 degrees
        #    tmax = self.block//(self.s_step * np.sin(ang))

        vert_proj = (np.abs(ang) < (np.pi/4))

        tmax = self.block // (self.s_step * ((np.cos(ang) * vert_proj) + (np.sin(ang) * (not vert_proj))))

        a_hat = np.cos(ang) * self.plane_v2 + np.sin(ang) * self.plane_v1
        ts = np.linspace(-tmax, tmax, 2 * tmax + 1)
        return a_hat, ts, vert_proj  # still need shift factor

    def generate_sample_pts(self, v_ind, h_ind, line_params):
        center_pt = (self.blk_centers[v_ind] * self.z_axis) + (self.blk_centers[h_ind] * self.plane_v1) + self.plane

        return center_pt


def norm(array):
    arr = np.array(array)
    return arr/np.sqrt(np.dot(arr, arr))
