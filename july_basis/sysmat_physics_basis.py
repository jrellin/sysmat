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

    m_dens = 1.18  # g/cm^3 PMMA
    elements = ('C', 'O')  # 4.4 MeV and 6.1 MeV lines

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

    # oxy_fields = ('Oxy613_sig', 'Oxy613_a20', 'Oxy613_a40', 'Oxy613_a60')
    # carbon_fields = ('C_sig', 'C_a20', 'C_a40')

    def __init__(self, kfile_name, im_pxl_sze=0.1):
        self.im_pxl_sze = im_pxl_sze
        with open(kfile_name, "rb") as fp:
            self.params = pickle.load(fp)

    def fold_energy_averaged(self, sysmat_fname, angle_fname,
                             carbon=True, oxygen=False,  # normalized=True,
                             save=True, include_sin_term=False):
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
        # if normalized:
        #    save_name += 'norm_'

        tot_legendre = np.polynomial.Legendre(np.zeros(7))

        if carbon:
            save_name += 'C'
            wgt_C = np.array([1, 0,
                              self.params['C_a20'].mean(), 0,  # a2, a3
                              self.params['C_a40'].mean(), 0,  # a4, a5
                              0  # a6
                              ])

            # if not normalized:  # i.e. absolute probabilities, TODO: include if you want absolute values
            #    wgt_C *= self.params['frac_C'] * self.params['C_sig'].mean() * \
            #             self.im_pxl_sze * self.params['n_dens'] * mb_convert
            # else:  # relative probabilities
            #    if both:  # carbon rel. to oxygen
            #        wgt_C *= self.params['frac_C']/ (self.params['frac_C'] + self.params['frac_O'])

            if both:  # carbon rel. to oxygen
                rel_int_prob = self.params['C_sig']/(self.params['C_sig'] + self.params['Oxy613_sig'])
                wgt_C *= rel_int_prob *  self.params['frac_C'] / (self.params['frac_C'] + self.params['frac_O'])

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
            # if not normalized:  # absolute, TODO: include if you want absolute values
            #     wgt_613 *=  self.params['frac_O'] * self.params['Oxy613_sig'].mean() *\
            #                 self.im_pxl_sze * self.params['n_dens'] * mb_convert
            # else:  # relative probabilities
            #    if both:  # oxygen rel. to carbon
            #        wgt_613 *= self.params['frac_O'] / (self.params['frac_C'] + self.params['frac_O'])

            if both:  # oxygen rel. to carbon
                rel_int_prob = self.params['Oxy613_sig'] / (self.params['C_sig'] + self.params['Oxy613_sig'])
                wgt_613 *= rel_int_prob * self.params['frac_O'] / (self.params['frac_C'] + self.params['frac_O'])

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
        if include_sin_term:
            # This corrects for solid angle subtended at constant polar angle (here, relative to beam axis)
            r_angles = a_table.read()  # response angles
            folded_sysmat.append(s_table.read() * tot_legendre(r_angles)) / np.sin(r_angles)
        else:
            folded_sysmat.append(s_table.read() * tot_legendre(np.cos(a_table.read())))
        new_file.flush()

        new_file.close()
        s_file.close()
        a_file.close()
        return tot_legendre, l_objs, save_name

    def _pos_weights(self, debug=True):
        """Generate weights for each term for position averaging."""
        ranges = self.params['projected_ranges']/self.m_dens
        r0 = self.params['projected_8MeV'] / self.m_dens
        diff_dist = np.diff(ranges, prepend=r0)  # no gamma emission below 8 MeV

        wgts = []
        idxs = []

        pos = 0
        wgt_prev = 0
        n_energies = ranges.size

        while pos < ranges.max():
            idx = np.argwhere((ranges > pos) & (ranges < (pos + self.im_pxl_sze)))
            if not idx.size:
                idxs.append(None)
                wgts.append(None)
                continue
            wgt = diff_dist[idx]
            wgt[0] *= 1 - wgt_prev

            p_idx = np.max(idx)  # last index fully contained below current edge
            # p_idx = idx[-1]  # since the values are sorted, equivalent
            if p_idx < n_energies - 1:
                n_idx = p_idx + 1  # next index
                wgt_prev = (ranges[n_idx] - (pos + self.im_pxl_sze)) / diff_dist[n_idx]
            else:
                n_idx = []
                wgt_prev = []

            idxs.append(np.append(idx, n_idx))
            wgts.append(np.append(wgt, wgt_prev))
            pos += self.im_pxl_sze

        if debug:
            print("Bins generated: ", len(idxs))
        return idxs, wgts

    def _generate_pos_basis(self, element='C', **kwargs):
        """Position folded list of weighted legendre coefficients. First item is closest to end of range"""
        indexes, wgts = self._pos_weights(**kwargs)  # kwargs = Debug
        bins = len(indexes)
        basis = [None] * bins

        a0, a20, a40, a60 = np.zeros(4)

        for i, (bin_idxs, bin_wgts) in enumerate(zip(indexes, wgts)):
            if bin_idxs is None or bin_wgts is None:
                continue  # basis[i] = None
            if element.upper() == 'C':
                a0 = np.average(self.params['C_sig'][bin_idxs], weights=bin_wgts)
                a20 = np.average(self.params['C_a20'][bin_idxs], weights=bin_wgts)
                a40 = np.average(self.params['C_a40'][bin_idxs], weights=bin_wgts)
                a60 = 0
            if element.upper() == 'O':
                a0 = np.average(self.params['Oxy613_sig'][bin_idxs], weights=bin_wgts)
                a20 = np.average(self.params['Oxy613_a20'][bin_idxs], weights=bin_wgts)
                a40 = np.average(self.params['Oxy613_a40'][bin_idxs], weights=bin_wgts)
                a60 = np.average(self.params['Oxy613_a60'][bin_idxs], weights=bin_wgts)

            # Note, normalized to a0 term
            coeff = np.array([1, 0, a20, 0, a40, 0, a60])

            # basis[i] = np.polynomial.Legendre(self.leg_norms * coeff)
            basis[i] = self.leg_norms * coeff

        return basis

    def fold_position_averaged(self, sysmat_fname, angle_fname,
                               x_pixels=201, element='C',
                               include_sin_term=False,
                               **kwargs):
        """Folds in energy averaged cross sections. Can fold in carbon and/or oxygen. Here carbon is 4.4 line,
        and oxygen is only 6.1 MeV. Image pixel size given in cm. Always normalized. *args is fed to folded_response.
        Must be sysmat_fname and angle_fname"""

        if element.upper() not in self.elements:
            ValueError("Element {e} not in allowed elements list: {a}".format(e=element.upper(), a=self.elements))

        l_objs = {}
        save_name = 'p_avg' + element  # folded system response name

        pos_basis = self._generate_pos_basis(element=element, **kwargs)  # returns list of coeff.
        pb_length = len(pos_basis)

        s_file, s_table = load_response_table(sysmat_fname, name='sysmat')
        a_file, a_table = load_response_table(angle_fname, name='pt_angles')

        s = s_table.read()
        a = a_table.read()

        s_file.close()
        a_file.close()

        assert s.shape == a.shape, "Angles table shape, {a}, and sysmat table shape, {s}," \
                                   " not the same".format(a=a.shape, s=s.shape)

        pxls, dets = s.shape

        geom = s.T.reshape([dets, pxls // x_pixels, x_pixels])
        angs = a.T.reshape([dets, pxls // x_pixels, x_pixels])
        tot = np.copy(geom)  # this keeps first pos basis x-axis values the same

        b = np.polynomial.Legendre(np.array([1, 0, 0, 0, 0, 0, 0]))

        for position, coefficients in enumerate(pos_basis):
            # position is relative position of current basis point
            b.coef = coefficients
            if include_sin_term:
                tot[:, :, pb_length:] += geom[:, :, pb_length-position:x_pixels-position] \
                                         * b(angs[:, :, pb_length-position:x_pixels-position]) /\
                                         np.sin(angs[:, :, pb_length-position:x_pixels-position])
            else:
                tot[:, :, pb_length:] += geom[:, :, pb_length - position:x_pixels - position] \
                                         * b(angs[:, :, pb_length - position:x_pixels - position])

        new_file = tables.open_file(save_name, mode="w", title="P Avg Folded System Response")
        folded_sysmat = new_file.create_earray('/', 'sysmat',
                                               atom=tables.atom.Float64Atom(),
                                               shape=(0, 48 * 48),
                                               expectedrows=s_table.nrows)
        folded_sysmat.append(tot.transpose((1, 2, 0)).reshape(s.shape))
        new_file.flush()
        new_file.close()

        
def main():
    pass


if __name__ == "__main__":
    main()