import numpy as np
import pickle
# Use kernel_parser before sysmat_physics_basis


def number_density(m_dens=1.18, mol_mass=100.12):  # PMMA, m_dens (g/cm^3), mol_mass (g/mol)
    N_A = 6.02214076e+23  # avogadro
    # mb_convert = 1e-27  # (cm^2/mb)
    N_dens = m_dens / mol_mass * N_A  # * mb_convert
    return N_dens  # (1/cm^3)


def atom_fractions(N_C=5, N_O=2, N_H = 8):  # PMMA defaults
    # PMMA constants
    return {'frac_O': N_O / (N_C + N_O + N_H), 'frac_C': N_C / (N_C + N_O + N_H), 'frac_H': N_H / (N_C + N_O + N_H)}


class kernel_parse(object):  # PMMA
    pstar_names = ['Energy', 'CSDA', 'Projected']  # proton stopping powers

    # a20 = a2/a0, a42 = a4/a0
    cs_names = ['Energy',
                'Oxy712_sig', 'Oxy712_a20',
                'Oxy692_sig', 'Oxy692_a20', 'Oxy692_a40',
                'Oxy613_sig', 'Oxy613_a20', 'Oxy613_a40', 'Oxy613_a60',
                'C_sig', 'C_a20', 'C_a40',
                'Oxy274_sig'
                ]

    # constants
    n_dens = number_density()  # 1/cm^3
    atom_fractions = atom_fractions()

    def __init__(self, cs_fname='cross_sections.txt', pstar_fname='PSTAR_PMMA.txt'):
        self.p_ranges = np.genfromtxt(pstar_fname, skip_header=3, names=self.pstar_names)  # p_ranges
        self.excitation = np.genfromtxt(cs_fname, skip_header=2, names=self.cs_names)  # excitation
        # self.img_pxl_sze = img_pxl_sze

        # find overlap of pstar and cs files of reported energy values
        self.energy, range_ind, _ = np.intersect1d(self.p_ranges['Energy'], self.excitation['Energy'], return_indices=True)
        self.CSDA_range = self.p_ranges['CSDA'][range_ind]  # p_water = 1
        self.projected_range = self.p_ranges['Projected'][range_ind]

    def generate_kern_dict(self, save_name='kernels.pkl'):
        proton_params = {'projected_8MeV': self.p_ranges['Projected'][self.p_ranges['Energy'] == 8],
                         'projected_ranges': self.projected_range}  # CSDA range not saved
        constant_params = {'n_dens': self.n_dens, 'atom_fractions': self.atom_fractions}
        kern_dict = {**self.excitation, **proton_params, **constant_params}
        with open(save_name, 'wb') as output:
           # Pickle dictionary using protocol 0.
            pickle.dump(kern_dict, output)

    def generate_position_kernel(self):
        """Generates list of legendre objects where each entry is one img_pxl averaged emission profile.
        Highest energy first"""
        # TODO: think carefully about desired output
        # oxygen613 = {'total': excitation['Oxy613_sig'], 'a20': excitation['Oxy613_a20'],
        #              'a40': excitation['Oxy613_a40'],
        #              'a60': excitation['Oxy613_a60']}

        wgt =  np.diff(self.projected_range,
                       prepend=self.p_ranges['Projected'][self.p_ranges['Energy'] == 8])
        # / self.img_pxl_sze

        scaled_range = self.projected_range # /self.img_pxl_sze


def main_kern_parse(save_name='kernels.pkl', **kwargs):
    kern_parser = kernel_parse(**kwargs)
    kern_parser.generate_kern_dict(save_name=save_name)


if __name__ == "__main__":
    main_kern_parse()