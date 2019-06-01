import numpy as np


# Obj| -> Col| -> Image +z ->
class pinhole(object):
    def __init__(self, size=1, loc=(0, 0, 0), axis=(0, 0, 1), par_hole_length=0,
                 x_min=-np.inf,
                 x_max=np.inf,
                 y_min=-np.inf,
                 y_max=np.inf,
                 coll_lim = (-25, 25)):
        self.sze = size
        self.c = np.array(loc)  # Center of pinhole
        self.ax = np.array(axis)
        self.tube = par_hole_length
        self.x_lim = np.sort([x_min, x_max])
        self.y_lim = np.sort([y_min, y_max])
        self.colz = np.sort(coll_lim)


class slit(object):
    # Beam ->+x
    def __init__(self, size=1, loc=(0, 0, 0), cen_ax=(0, 0, 1), slit_ax=(0, 1, 0), slit_angle=20, par_hole_length=0,
                 x_min=-np.inf,
                 x_max=np.inf,
                 y_min=-np.inf,
                 y_max=np.inf,
                 coll_lim=(-25, 25)):
        self.sze = size  # Minimum width of aperture
        self.c = np.array(loc)  # Some point along center slit plane
        self.f = np.array(cen_ax)  # Normal to plane of aperture (focus)
        self.tube = par_hole_length  # Total length
        self.x_lim = np.sort([x_min, x_max])
        self.y_lim = np.sort([y_min, y_max])

        self.a = np.array(slit_ax)  # axis of slit
        self.h_ang = np.deg2rad(slit_angle/2.)  # Half of the opening angle of the slit
        self.colz = np.sort(coll_lim)

        self.plane = np.cross(self.a, self.f)  # Vector that in direction of the slit opening

    def ray_pass(self, ray):  # Ray should be an array of 3d points that correspond to the passing ray
        proj_f = np.dot(self.c - ray, self.f)
        channel = np.sign((self.tube/2.)-np.abs(proj_f))

        pass
