import numpy as np
import pickle
import tables
# Use kernel_parser before sysmat_physics_basis


def load_response(sysmat_fname, name='sysmat'):  # pt_angles is other
    """Loads all at once"""
    sysmat_file_obj = load_h5file(sysmat_fname)
    data = sysmat_file_obj.get_node('/', name).read()  # TODO: Check if transpose needed
    sysmat_file_obj.close()  # TODO: Check that data is still saved
    return data


def load_response_table(sysmat_fname, name='sysmat'):  # pt_angles is other
    """Returns table object (index as normal)"""
    sysmat_file_obj = load_h5file(sysmat_fname)
    data_table_obj = sysmat_file_obj.get_node('/', name)  # TODO: Check if need to transpose
    return sysmat_file_obj, data_table_obj  # also returns file object to be closed


def load_h5file(filepath):  # h5file.root.sysmat[:]
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


class physics_basis(object):  # PMMA

    leg_norms =  (2 * np.arange(7) + 1)/2.0  # i.e. up to a6 term

    fields = ['Energy',  # from cross section files
              'Oxy712_sig', 'Oxy712_a20',
              'Oxy692_sig', 'Oxy692_a20', 'Oxy692_a40',
              'Oxy613_sig', 'Oxy613_a20', 'Oxy613_a40', 'Oxy613_a60',
              'C_sig', 'C_a20', 'C_a40',
              'Oxy274_sig',
              'n_dens',  # constants, 1/cm^3
              'frac_O', 'frac_C', 'frac_H',  # from  atom_fractions()
              'projected_8MeV',  # from pstar
              'projected_ranges'
              ]

    def __init__(self, kfile_name, im_pxl_sze=0.1):
        self.im_pxl_sze = im_pxl_sze
        with open(kfile_name, "rb") as fp:
            self.params = pickle.load(fp)

    def fold_energy_averaged(self, sysmat_fname, angle_fname, carbon=True, oxygen=False, normalized=True,
                             save=True):
        """Folds in energy averaged cross sections. Can fold in carbon and/or oxygen. Here carbon is 4.4 line,
        and oxygen is only 6.1 MeV. Image pixel size given in cm. Normalized defines probabilities
         relative to a0 term, else calculates absolute given PMMA parameters"""
        both = False
        if not carbon and not oxygen:
            return  # nothing to fold

        if carbon and oxygen:
            both = True

        mb_convert = 1e-27  # (cm^2/mb)
        l_objs = {}
        save_name = 'e_avg'  # folded system response name
        if normalized:
            save_name += 'norm_'

        tot_legendre = np.polynomial.Legendre(np.zeros(7))

        if carbon:
            save_name += 'C'
            wgt_C = np.array([1, 0,
                              self.params['C_a20'].mean(), 0,  # a2, a3
                              self.params['C_a40'].mean(), 0,  # a4, a5
                              0  # a6
                              ])

            if not normalized:  # i.e. absolute probabilities
                wgt_C *= self.params['frac_C'] * self.params['C_sig'].mean() * \
                         self.im_pxl_sze * self.params['n_dens'] * mb_convert
            else:  # relative probabilities
                if both:  # carbon rel. to oxygen
                    wgt_C *= self.params['frac_C']/ (self.params['frac_C'] + self.params['frac_O'])

            bas_C = np.polynomial.Legendre(self.leg_norms * wgt_C)
            tot_legendre += bas_C
            l_objs['Carbon'] = bas_C
            # yld_Oxy613 = bas_Oxy613(costh)

        if oxygen:
            save_name += 'O'
            wgt_613 = np.array([1, 0,
                                self.params['Oxy613_a20'].mean(), 0,  # a2, a3
                                self.params['Oxy613_a40'].mean(), 0,  # a4, a5
                                self.params['Oxy613_a60'].mean()  # a6
                                ])
            if not normalized:  # absolute
                wgt_613 *=  self.params['frac_O'] * self.params['Oxy613_sig'].mean() *\
                            self.im_pxl_sze * self.params['n_dens'] * mb_convert
            else:  # relative probabilities
                if both:  # oxygen rel. to carbon
                    wgt_613 *= self.params['frac_O'] / (self.params['frac_C'] + self.params['frac_O'])

            bas_Oxy613 = np.polynomial.Legendre(self.leg_norms * wgt_613)
            tot_legendre += bas_Oxy613
            l_objs['Oxygen'] = bas_Oxy613

        if not save:
            return tot_legendre, l_objs, save_name

        s_file, s_table = load_response_table(sysmat_fname, name='sysmat')
        a_file, a_table = load_response_table(angle_fname, name='pt_angles')

        assert s_table.nrows == a_table.nrows, \
            "sysmat rows: {s}. angle rows: {a}".format(s=s_table.nrows, a=a_table.nrows)

        new_file = tables.open_file(save_name, mode="w", title="E Avg Folded System Response")
        folded_sysmat = new_file.create_earray('/', 'sysmat',
                                               atom=tables.atom.Float64Atom(),
                                               shape=(0, 48 * 48),
                                               expectedrows=s_table.nrows)
        folded_sysmat.append(s_table.read() * tot_legendre(np.cos(a_table.read())))
        new_file.flush()

        new_file.close()
        s_file.close()
        a_file.close()
