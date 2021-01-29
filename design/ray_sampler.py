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
                 sample_step=4/2,  # sample steps in mm along line
                 beam_step=5,  # in mm, along beam
                 line_shift=4/2,  # in mm, distance between lines
                 opening_ang=20,  # in degrees
                 block_sze=48  # size of grid unit in mm
                 ):

        self.y_fov = np.array(y_fov)
        self.z_fov = np.array(z_fov)
        self.b_step = beam_step  # beam axis is in y-hat direction
        self.b_axis = np.array([0, 1, 0])
        self.s_step = sample_step
        self.tot_shift = line_shift
        self.line_params = 0  # i.e. default line angle is along z-axis

        plane_ang = np.deg2rad(plane_angle)
        self.norm_plane = np.array([np.sin(plane_ang), np.cos(plane_ang), 0])  # normal to plane
        self.plane_v1 =  np.array([np.cos(plane_ang), -np.sin(plane_ang), 0])  # v1 perp. to v2 and norm
        self.plane_v2 = self.z_axis  # v2 x v1 = norm
        self.ror = ror
        self.plane = self.norm_plane * self.ror

        self.block = block_sze
        self.blk_centers = self.blk_ind * self.block * mag
        self.hlf_block = self.block/2
        self.slit_ang = np.deg2rad(opening_ang)  # Full opening

    @property
    def line_params(self):
        return self._l_params

    @line_params.setter
    def line_params(self, line_ang):  # default value set in __init__ and is 0. Points along z axis
        # TODO: Possible issue with deleting and recreating dict instead of changing values in keys of _l_params
        ang = np.deg2rad(line_ang)
        ang_mag = np.abs(ang)

        ret_dict = {}

        vert_proj = (np.abs(ang) < (np.pi/4))

        if vert_proj: # i.e. shifts same sized lines shifted in the _horizontal_ direction
            shift = self.tot_shift / np.cos(ang_mag)  # shifts to other lines are in horizontal direction
            max_shift = self.hlf_block * (1 + np.tan(ang_mag))
            max_s_step = self.hlf_block / np.cos(ang_mag)
        else:
            shift = self.tot_shift / np.sin(ang_mag)  # shifts to other lines are in vertical direction
            max_shift = self.hlf_block * (1 + np.tan((np.pi/2)-ang_mag))
            max_s_step = self.hlf_block / np.sin(ang_mag)

        ret_dict['s_hat'] = (vert_proj * self.plane_v1) + (not vert_proj * self.plane_v2)
        ret_dict['shifts'] = mirrored(max_shift, shift)

        # tmax = self.block // (self.s_step * ((np.cos(ang_mag) * vert_proj) + (np.sin(ang_mag) * (not vert_proj))))

        ret_dict['a_hat'] = (np.cos(ang) * self.plane_v2) + (np.sin(ang) * self.plane_v1)
        ret_dict['ts'] = mirrored(max_s_step, self.s_step)

        ret_dict['a_perp'] = norm(np.cross(ret_dict['a_hat'], self.norm_plane))

        self._l_params = ret_dict
        self.template = self._generate_template_pts()

    def _generate_template_pts(self):
        ret_temp = {}
        line = self.line_params['a_hat'] + self.line_params['ts'][:, np.newaxis]
        shift_family = self.line_params['s_hat'] + self.line_params['shifts'][:, np.newaxis]
        # TODO: Calculate mask array for points that are inside the block, use dot products with v1 and v2
        ret_temp['template_pts'] = line + shift_family[:, np.newaxis, :]
        # ret_temp['valid_mask'] = (np.abs(np.dot(template_pts, self.plane_v1)) < self.hlf_block) & \
        #              (np.abs(np.dot(template_pts, self.plane_v2)) < self.hlf_block)
        ret_temp['valid_mask'] = (np.abs(np.dot(ret_temp['template_pts'], self.plane_v1)) < self.hlf_block) & \
                                 (np.abs(np.dot(ret_temp['template_pts'], self.plane_v2)) < self.hlf_block)

        ret_temp['total_pts'] = np.count_nonzero(ret_temp['valid_mask'])
        # t_pts * valid_mask[:,:,np.newaxis]
        return ret_temp

    def _generate_sample_pts(self, v_ind, h_ind):
        center_pt = (self.blk_centers[v_ind] * self.z_axis) + (self.blk_centers[h_ind] * self.plane_v1) + self.plane
        return center_pt + self.template['template_pts']

    def _get_sample_params(self, beam_position, pts):

        r_pos = pts - beam_position
        proj_normal_plane = np.abs(np.dot(r_pos, self.norm_plane))
        proj_a_perp = np.abs(np.dot(r_pos, self.line_params['a_perp']))

        # ang_good = (np.arctan(proj_normal_plane, proj_a_perp) < (self.slit_ang/2))  # TODO: Don't forget
        angle_inc = np.arctan(proj_normal_plane, proj_a_perp)

        r_pos_dot_beam = np.abs(np.dot(r_pos, self.b_axis))  # absolute value to constrain theta
        norm_r_pos = np.sqrt(np.sum(r_pos ** 2, axis=2))  # Normalized

        # return np.arccos(r_pos_dot_beam/norm_r_pos), norm_r_pos  # fits are to cos(th) anyway
        return beam_position, (r_pos_dot_beam / norm_r_pos), norm_r_pos/self.ror, angle_inc

    def _sample(self, sample_pts, beam_positions, **kwargs):
        for one_pos in beam_positions:
            beam_position, cos_th, normalized_r, ang_inc = self._get_sample_params(one_pos, sample_pts)
            values = legendre_project(cos_th, normalized_r, (ang_inc < (self.slit_ang / 2)), **kwargs) * \
                     np.cos(ang_inc) * self.template['valid_mask']
            yield values  # , one_post

    def beam_step_projections(self, v_ind, h_ind, b_start=None, b_end=None, z_offset=0, **kwargs):
        # **kwargs -> angular momentum L projection
        total_pts = self.template['total_pts']

        if b_start is None:
            b_start = self.y_fov[0]
        if b_end is None:
            b_end = self.y_fov[1]

        if b_start > b_end:
            along_beam = np.arange(b_start, b_end - self.b_step, -self.b_step)[:, np.newaxis] * self.b_axis
        else:
            along_beam = np.arange(b_start, b_end + self.b_step, self.b_step)[:, np.newaxis] * self.b_axis

        beam_positions = along_beam + (z_offset * self.z_axis)
        sample_pts = self._generate_sample_pts(v_ind, h_ind)

        efficiencies = np.zeros(beam_positions.shape[0])
        surface_area = np.zeros_like(efficiencies)
        avg_cnrs = np.zeros_like(efficiencies)

        for pos_num, ret in enumerate(self._sample(sample_pts, beam_positions, **kwargs)):
            prev = ret
            efficiencies[pos_num] = np.sum(ret)
            surface_area[pos_num] = np.count_nonzero(prev)/total_pts

            if pos_num:  # i.e. 0th iteration
                avg_cnrs[pos_num] = np.mean(0.5 * np.abs(ret - prev) /
                                            (np.sqrt(ret) + np.sqrt(prev)) + ((ret-prev) == 0))
            # TODO: As this is set up, this only looks at contrast along the beam axis. What about in other lines?

        return efficiencies, surface_area, avg_cnrs


def legendre_project(cos_th, r_mag, ang_mask, l_fit=0):
    # sample_params are cos(thetas), |r| scaled to ror distance, and if incident angle is less than slits
    # cos_th, r_mag, ang_mask = sample_params
    poly = np.polynomial.Legendre.basis(l_fit)

    if l_fit:
        legendre_proj = poly(cos_th) * ang_mask
    else:
        legendre_proj = poly(cos_th) / (r_mag ** 2) * ang_mask
    return legendre_proj


def norm(array):
    arr = np.array(array)
    return arr/np.sqrt(np.dot(arr, arr))


def mirrored(maxval, inc=1.0):
    x = np.arange(inc, maxval + inc, inc)
    # if x[-1] != maxval:  # include max values
    #    x = np.r_[x, maxval]
    return np.r_[-x[::-1], 0, x]


def main():
    # generate sample pts by indexing over row and col indices
    return True


if __name__ == "__main__":
    main()
