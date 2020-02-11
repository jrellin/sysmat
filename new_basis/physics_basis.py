import numpy as np

# PSTAR Ranges
try_names = ['Energy', 'CSDA', 'Projected']
range = np.genfromtxt('PSTAR.txt', skip_header=3, names=try_names)
weights = np.ones(range['Energy'].size)

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

sysmat = np.load('/Users/justinellin/repos/sysmat/new_basis/2D_matrix_adjusted_11Apr16.npy')
