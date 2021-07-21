from design.imager_system import Imager
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


# TOP: UTILS
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
            angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            angle_between((1, 0, 0), (1, 0, 0))
            0.0
            angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def normalize_vectors(vectors):
    """ Returns normalized list of vectors if arranged along rows of 2d array """
    # mags = np.sqrt(np.sum(vectors ** 2, axis=1))
    mags = np.linalg.norm(vectors, axis=1)
    return vectors/mags[:, np.newaxis]


def angle_between_vectors_and_beam(face_pts, em_pt, r0=np.array([1, 0, 0])):
    """ Vectors = unnormalized face_pts, r0 is beam axis. Angle between face_pts and r0"""
    # print("Em Pt: ", em_pt)
    vectors_u = normalize_vectors(face_pts-em_pt)
    v2_u = unit_vector(np.array(r0))
    return np.arccos(np.clip(np.dot(vectors_u, v2_u), -1.0, 1.0))


# def full_image(image_list):
#         return np.block([image_list[col:col + 4] for col in np.arange(0, len(image_list), 4)])
def to_image(det_list, idx_row):
    return np.block([det_list[col:col + 4] for col in idx_row])
# BOT: UTILS


def create_system():
    system = Imager()

    # ~=~=~=~=~=~=~=~=~=~=~=~= NOTICE: No Collimator Need be defined =~=~=~=~=~=~=~=~=~=~=~=~
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
        system.create_detector(det_id=det_idx, center=det_center, det_norm=directions[det_idx], pxl_sze=4)
    print("Farthest Plane: ", system.detector_system.farthest_plane)

    # ==================== Sources ====================
    system.sources.sc = np.array([0, -10, -20])
    system.sources.vsze = 1
    system.sources.npix = np.array([201, 61])  # With above, [201, 61]

    # ==================== Run and Display ====================
    # system.sample_step = 0.1  # in mm, default
    # system.subsample = 1  # samples per detector pixel.

    # system.generate_sysmat_response(subsample=system.subsample)
    # print('It took ' + str(time.time() - start) + ' seconds.')
    return system


def single_pt_angles(src_pt=None, legendre_plot=False, l_order=2, stopwatch=False):
    """This is a check that main() will work. legendre_plot plots legendre eval of angles (or just angles in radians
    if set to False). l_order plots that legendre_order"""
    system = create_system()
    nx, ny = system.detector_system.detectors[0].npix

    start = time.time()

    if src_pt is None:
        src_pt = np.array(system.sources.sc)

    det_face_pts = []
    det_angles = [None] * 16  # cycle through these

    for det in system.detector_system.detectors:
        # print("Detector Idx Check. Det ID: ", det.det_id)
        det_face_pts.append(det.face_pts(back=False))

    idx_row = np.arange(0, len(det_angles), 4)  # So not making this every iteration

    for det in system.detector_system.detectors:
        det_angles[det.det_id] = angle_between_vectors_and_beam(det.face_pts(), src_pt).reshape([ny, nx])

    # print("Det angles length: ", len(det_angles))
    # print("Det angles[0]. shape", det_angles[0].shape)
    img = to_image(det_angles, idx_row)
    # print("Img shape: ", img.shape)

    if legendre_plot:
        wgts = np.zeros(7)
        wgts[l_order] = 1
        poly = np.polynomial.Legendre(wgts)

        # TODO: REMOVE, WAS USED FOR TESTING JULY 20
        # print("Poly.coef before: ", poly.coef)
        # leg_norms = (2 * np.arange(7) + 1) / 2.0
        # nc =  np.array([0.5, 0., 1.125, 0., -0.92454545, 0, 0])/leg_norms
        # poly.coef =  nc
        # print("Poly.coef after: ", poly.coef)

        leg_eval = poly(np.cos(img))
        # img = np.max(leg_eval) /  leg_eval
        img = leg_eval
    else:
        img = np.rad2deg(img)

    # plt.imshow(np.rad2deg(img), cmap='magma', origin='upper', interpolation='none')
    plt.imshow(img, cmap='magma', origin='upper', interpolation='none')
    plt.xticks([])
    plt.yticks([])

    base_title = 'Pt {p} , '.format(p=np.array2string(src_pt[:2], separator=','))

    cbar = plt.colorbar()
    if legendre_plot:
        cbar.set_label('amplitude', rotation=90)
        plt.title(base_title + 'L Order={o}'.format(o=l_order))
    else:
        cbar.set_label('degrees', rotation=90)

    if stopwatch:
        print('It took ' + str(time.time() - start) + ' seconds.')
    plt.show()


def main(**kwargs):
    start = time.time()
    system = create_system()
    nx, ny = system.detector_system.detectors[0].npix

    # ==================== H5 File ====================

    current_datetime = datetime.now().strftime("%Y-%m-%d-%H%M")
    save_fname = os.path.join(os.getcwd(), 'angles_' + current_datetime + '.h5')

    src_pts = np.prod(system.sources.npix)
    print("Total point source locations: ", src_pts)

    file = tables.open_file(save_fname, mode="w", title="System Response")
    pt_angles = file.create_earray('/', 'pt_angles',
                                   atom=tables.atom.Float64Atom(),
                                   shape=(0, system.detector_system.projection.size),
                                   expectedrows=src_pts)

    prog = 0
    perc = 0

    evts_per_percent = np.floor(0.01 * src_pts)

    det_face_pts = []
    det_angles = [None] * 16  # cycle through these

    for det in system.detector_system.detectors:
        print("Detector Idx Check. Det ID: ", det.det_id)
        det_face_pts.append(det.face_pts(back=False))

    idx_row = np.arange(0, len(det_angles), 4)  # So not making this every iteration

    for src_pt in system.sources.source_pt_iterator():
        for det in system.detector_system.detectors:
            det_angles[det.det_id] = angle_between_vectors_and_beam(det.face_pts(), src_pt).reshape([ny, nx])
            # TODO: Check that reshape works

        pt_angles.append(to_image(det_angles, idx_row).ravel()[None])
        # pt_angles.append(system._point_response_function(src_pt, **kwargs).ravel()[None])
        file.flush()
        prog += 1
        if prog > 2 * evts_per_percent:
            perc += prog / src_pts * 100  # total points
            print("Current Source Point: ", src_pt)
            print("Progress (percent): ", perc)
            prog = 0

    print("nrows: ", pt_angles.nrows)

    print('It took ' + str(time.time() - start) + ' seconds.')
    file.close()


if __name__ == "__main__":
    main()
    # pt = np.array([-40, -10, -20])
    # single_pt_angles(src_pt=None, legendre_plot=False, stopwatch=True)  # angles, radians
    # single_pt_angles(src_pt=None, legendre_plot=True, l_order=2, stopwatch=True)

