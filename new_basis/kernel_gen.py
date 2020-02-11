import numpy as np
import pickle

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
rel_dist = diff_dist/im_pxl_size  # how much does it contribute to the kernel i.e. weights
wgt = rel_dist

PMMA_C = 5  # C5O2H8
PMMA_O = 2

initial_bin = np.ceil(np.max(projected_range/im_pxl_size))  # High Energy side
bins_kernel = np.int(initial_bin - np.floor(np.min(projected_range/im_pxl_size)))

bin_ind = [None] * bins_kernel

scaled_range = projected_range/im_pxl_size
# print("Initial_bin:", initial_bin)

tmp_bin = 0
for i in np.arange(bins_kernel):  # start from end of range and go backwards
    bin_ind[i] = (scaled_range < (initial_bin - i)) & (scaled_range > ((initial_bin - i) - 1))
    # print("Binning High: ",  (initial_bin - i))
    # print("Binning Low: ", (initial_bin - i - 1))

# print("Bin Indices", bin_ind)


def k_gen(values, list_ind, wgts):  # indices is a LIST of kernel bins
    # print(values)
    # print(list_ind)
    # print(wgts)
    bins = len(list_ind)
    kernel = np.zeros(bins)
    for bin in np.arange(bins):
        idx = list_ind[bin]
        # print("Weights", wgts[idx])
        # print("Values", values[idx])
        kernel[bin] = np.average(values[idx], weights=wgts[idx])
    return kernel


print("Bin Indices: ", bin_ind)
print("Weights: ", wgt)

kernel = {
    'Oxy712':
        {'sig': k_gen(oxygen712['total'], bin_ind, wgt),
         'a20': k_gen(oxygen712['a20'], bin_ind, wgt)
         },
    'Oxy692':
        {'sig': k_gen(oxygen692['total'], bin_ind, wgt),
         'a20': k_gen(oxygen692['a20'], bin_ind, wgt),
         'a40': k_gen(oxygen692['a40'], bin_ind, wgt)
         },
    'Oxy613':
        {'sig': k_gen(oxygen613['total'], bin_ind, wgt),
         'a20': k_gen(oxygen613['a20'], bin_ind, wgt),
         'a40': k_gen(oxygen613['a40'], bin_ind, wgt),
         'a60': k_gen(oxygen613['a60'], bin_ind, wgt)
         },
    'Carbon':
        {'sig': k_gen(carbon['total'], bin_ind, wgt),
         'a20': k_gen(carbon['a20'], bin_ind, wgt),
         'a40': k_gen(carbon['a40'], bin_ind, wgt)
         },
    'Average_Energy': k_gen(energy, bin_ind, wgt)
}

# with open('kernels.pkl', 'wb') as output:
#    # Pickle dictionary using protocol 0.
#    pickle.dump(kernel, output)

# with open("kernels.pkl", "rb") as fp:
#    kernels = pickle.load(fp)


