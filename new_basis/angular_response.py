import numpy as np

det_to_col = 24  # cm, from face
det_thick = 7.5  # cm
source_to_col = 13  # cm, front face
det_pxl_size = 0.4  # cm, i.e. mm

# x is beam direction -> +x
n_x = 151
n_y = 51
im_pxl_size = 0.1  # cm, i.e. mm

# PSTAR Ranges
try_names = ['Energy', 'CSDA', 'Projected']
all_ranges = np.genfromtxt('PSTAR.txt', skip_header=3, names=try_names)
# weights = np.ones(range['Energy'].size)

# a20 = a2/a0, a42 = a4/a0
cs_names = ['Energy',
            'Oxy712_sig', 'Oxy712_a20',
            'Oxy692_sig', 'Oxy692_a20', 'Oxy692_a40',
            'Oxy613_sig', 'Oxy613_a20', 'Oxy613_a40', 'Oxy613_a60',
            'C_sig', 'C_a20', 'C_a40',
            'Oxy274_sig'
            ]
excitation = np.genfromtxt('cross_sections.txt', skip_header=2, names=cs_names)

oxygen712 = {'total': excitation['Oxy712_sig'], 'a20': excitation['Oxy712_a20']}
oxygen692 = {'total': excitation['Oxy692_sig'], 'a20': excitation['Oxy692_a20'], 'a40': excitation['Oxy692_a40']}
oxygen613 = {'total': excitation['Oxy613_sig'], 'a20': excitation['Oxy613_a20'], 'a40': excitation['Oxy613_a40'],
             'a60': excitation['Oxy613_a60']}
carbon = {'total': excitation['C_sig'], 'a20': excitation['C_a20'], 'a40': excitation['C_a40']}
kinetic_energy = excitation['Energy']

#
energy, range_ind, _ = np.intersect1d(all_ranges['Energy'], kinetic_energy, return_indices=True)
CSDA_range = all_ranges['CSDA'][range_ind]  # p_water = 1
projected_range = all_ranges['Projected'][range_ind]
diff_dist = np.diff(projected_range,
                    prepend=all_ranges['Projected'][all_ranges['Energy'] == 8])
rel_dist = diff_dist/im_pxl_size  # how much does it contribute to the kernel

PMMA_C = 5  # C5O2H8
PMMA_O = 2

high_E_bin = np.ceil(np.max(projected_range/im_pxl_size))
bins_kernel = high_E_bin - np.floor(np.min(projected_range/im_pxl_size))





