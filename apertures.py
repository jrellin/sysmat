import numpy as np


# Obj| -> Col| -> Image +z ->
class Pinhole(object):
    def __init__(self, size=2, loc=(0, 0, 0), axis=(0, 0, 1), chan_length=0, aper_angle=20,
                 # x_min=-np.inf,
                 # x_max=np.inf,
                 # y_min=-np.inf,
                 # y_max=np.inf,
                 coll_lim=(-25, 25)):
        self.sze = size
        self.c = np.array(loc)  # Center of pinhole
        self.ax = np.array(axis)
        self.tube = chan_length
        # self.x_lim = np.sort([x_min, x_max])
        # self.y_lim = np.sort([y_min, y_max])
        self.h_ang = np.deg2rad(aper_angle / 2.)
        self.colz = np.sort(coll_lim)

    def ray_pass(self, ray):
        proj_f = np.abs(np.dot(ray - self.c, self.ax))
        ray_check = np.zeros(proj_f.size)


class Slit(object):
    # Beam ->+x
    __slots__ = ['sze', 'c', 'f', 'tube', 'x_lim', 'y_lim', 'a', 'h_ang', 'colz', 'plane']

    def __init__(self, size=2, loc=(0, 0, 0), cen_ax=(0, 0, 1), slit_ax=(0, 1, 0), aper_angle=20, chan_length=0,
                 x_min=-np.inf,
                 x_max=np.inf,
                 y_min=-np.inf,
                 y_max=np.inf,
                 coll_lim=(-25, 25)):
        self.sze = size/2.  # Minimum half-width of aperture in mm
        self.c = np.array(loc)  # Some point along center slit plane
        self.f = np.array(cen_ax)  # Normal to plane of aperture (focus)
        self.tube = chan_length  # Total length
        self.x_lim = np.sort([x_min, x_max])
        self.y_lim = np.sort([y_min, y_max])

        self.a = np.array(slit_ax)  # axis of slit
        self.h_ang = np.deg2rad(aper_angle/2.)  # Half of the opening angle of the slit in radians
        self.colz = np.sort(coll_lim)

        self.plane = np.cross(self.a, self.f)  # Vector that is in direction of the slit opening

    def ray_pass(self, ray):  # Ray should be an array of 3d points that correspond to the passing ray
        proj_f = np.abs(np.dot(ray - self.c, self.f))
        proj_u = np.abs(np.dot(ray - self.c, self.plane))
        ray_check = np.zeros(proj_f.size)

        near_slit = (ray[:, 0] >= self.x_lim[0]) & (ray[:, 0] <= self.x_lim[1]) & \
                    (ray[:, 1] >= self.y_lim[0]) & (ray[:, 1] <= self.y_lim[1])

        # near_slit checks if the ray is inside the 2D square that encloses the slit or slit segment

        chn_out = ((proj_f - (self.tube / 2.)) > 0)  # Projection onto slit axis is
        # outside channel

        ray_check[chn_out & near_slit] = proj_u[chn_out & near_slit] < \
            (proj_f[chn_out & near_slit] - (self.tube / 2.)) * np.tan(self.h_ang) + self.sze

        ray_check[~chn_out & near_slit] = proj_u[~chn_out & near_slit] < self.sze  # Inside channel

        # I.E. when the ray is inside the slit you get a 1, 0 when outside
        return ray_check
