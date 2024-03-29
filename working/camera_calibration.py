import tables
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from working.calibration_values import load_calibration
# from scipy import stats  # This will be necessary to do a per-pixel gain calibration
from scipy.ndimage import uniform_filter1d
from scipy.stats import binned_statistic_2d
# or single pixel spectrum (scipy.binned_statistic2d)


class events_recon(object):
    evt_read_limit = 2 * 10 ** 5  # Number of events to read at a time # TODO: Convert to memory size using col itemsze
    pixels = 12

    def __init__(self, filepath, calib=None, place='Davis', mod_adc_max_bin=18000, mod_adc_bin_size=250,
                 pmt_adc_max_bin=18000, pmt_adc_bin_size=250):
        if calib is None:  # calib should be a dict
            calib = load_calibration(place)

        self._pmts = calib['pmts']
        self.swapped = calib['swapped']
        self._pmt_swapped = calib['swapped_pmts']

        self.h5file = load_signals(filepath)

        # self.histogram_bins = np.arange(0, 180000, 1000)
        self.mod_histogram_bins = np.arange(0, mod_adc_max_bin + mod_adc_bin_size, mod_adc_bin_size)
        self.pmt_histogram_bins = np.arange(0, pmt_adc_max_bin + pmt_adc_bin_size, pmt_adc_bin_size)
        # self.image_size = np.array([100, 100])

        self.gamma_events = 0
        self.filtered_gamma_events = 0

        self.pmt_gains = calib['pmt_gains']
        self.pmt_shifts = calib['pmt_shifts']
        self.module_gains = calib['module_gains']
        self.module_shifts = calib['module_shifts']

        # self.com_bins = np.linspace(-1.0, 1.0, 101)
        # ============ Histogram Objects ============
        self.crude_crystal_cutsX = calib['crystal_x_edges']
        self.crude_crystal_cutsY = calib['crystal_y_edges']
        self.pmt_histograms = [Hist1D(self.pmt_histogram_bins) for _ in np.arange(4)]
        self.E1, self.E2, self.E3, self.E4 = self.pmt_histograms  # aliases
        # 4 PMTs histograms
        self.module_energy_histogram = Hist1D(self.mod_histogram_bins)  # 1 module histograms
        self.mod_image = HistImage(self.crude_crystal_cutsX, self.crude_crystal_cutsY)

    def convert_adc_to_bins(self, ch_start_index, mod_ref_index, dynamic_pmt_gains=np.ones(4), dynamic_mod_gains=1,
                        filter_limits=None, masks=None):  # this is version 2
        """Mask must be an array of linearized indices or a list of 4 linearized indices (PMT weighting)"""
        pmts = np.arange(4)
        tables = [0 for _ in pmts]

        if filter_limits is None:
            filter_limits = 0

        if len(filter_limits) >= 2:
            filter_limits.sort()

        for integer in pmts:
            folder = '/det' + str(int(ch_start_index + integer))
            tables[integer] = self.h5file.get_node('/', folder).EventData

        total_evts = tables[0].nrows
        if total_evts < 1:
            return None, None
        self.gamma_events += total_evts

        read_block_idx = 0
        energy_array = np.zeros([4, np.min([total_evts, self.evt_read_limit])])  # evts are columns
        # print("Hello!")
        if self.swapped[mod_ref_index]:
            pmts_ch_map = self._pmt_swapped
        else:
            pmts_ch_map = self._pmts

        # When facing the front of the detectors. This maps spatial position of PMTs to your cabling
        # ul = pmts_ch_map[0][0], ur = pmts_ch_map[0][1]
        # ll = pmts_ch_map[1][0], lr = pmts_ch_map[1][1]
        (ul, ur), (ll, lr) = pmts_ch_map[:]
        pmt_ch_map = np.array([ul, ur, ll, lr])

        start = 0
        end = 0

        self.mod_image.select(mod_ref_index)
        self.module_energy_histogram.clear()
        self.E1.clear()
        self.E2.clear()
        self.E3.clear()
        self.E4.clear()

        while end < total_evts or not read_block_idx:
            for pmt_id, ch_id in enumerate(pmt_ch_map):
                # pmt_id = 0, 1, 2, 3
                start = read_block_idx * self.evt_read_limit
                end = np.min([start + self.evt_read_limit, total_evts])
                energy_array[pmt_id, :(end-start)] = (self.pmt_gains[mod_ref_index, pmt_id] *
                                                      (tables[ch_id].cols.gate2[start:end] -
                                                       3.0 * tables[ch_id].cols.gate1[start:end]) -
                                                      self.pmt_shifts[mod_ref_index, pmt_id]) * dynamic_pmt_gains[pmt_id]

            all_sum = np.sum(energy_array[:, :(end-start)], axis=0) * dynamic_mod_gains

            e_bound_filter = (all_sum > filter_limits[0])
            if len(filter_limits) is 2:
                e_bound_filter &= (all_sum < filter_limits[1])

            sum_e = all_sum[e_bound_filter]
            valid_pmts = energy_array[:, :(end - start)][:, e_bound_filter]  # TODO: Does this work? It's weird

            Ex = ((valid_pmts[1] - valid_pmts[0]) + (valid_pmts[3] - valid_pmts[2])) / (1.0 * sum_e)
            Ey = ((valid_pmts[1] - valid_pmts[3]) + (valid_pmts[0] - valid_pmts[2])) / (1.0 * sum_e)

            Ex_scaled = (Ex + 1) / 0.02
            Ey_scaled = (Ey + 1) / 0.02

            Ex_scaled[Ex_scaled > 100] = 99.
            Ey_scaled[Ey_scaled > 100] = 99.

            Ex_scaled[Ex_scaled < 0] = 1.
            Ey_scaled[Ey_scaled < 0] = 1.

            if masks is None:
                self.mod_image.fill(Ex_scaled, Ey_scaled)
                self.module_energy_histogram.fill(sum_e)

                self.E1.fill(valid_pmts[0])  # ul
                self.E2.fill(valid_pmts[1])  # ur
                self.E3.fill(valid_pmts[2])  # ll
                self.E4.fill(valid_pmts[3])  # lr
            else:
                if isinstance(masks, list):
                    self.weighted_masks_events(Ex_scaled, Ey_scaled, sum_e, valid_pmts, masks)
                else:
                    self.single_mask_events(Ex_scaled, Ey_scaled, sum_e, valid_pmts, masks)

            read_block_idx += 1

        self.filtered_gamma_events += np.sum(self.module_energy_histogram.hist)
        histograms = [self.module_energy_histogram.data[1], self.E1.data[1], self.E2.data[1], self.E3.data[1], self.E4.data[1]]
        # plt.imshow(self.mod_image.data[2], cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')
        # plt.show()
        return histograms, self.mod_image.data[2]

    def single_mask_events(self, ex, ey, sum_e, valid_pmts, mask):  # NOTE: Problems likely here
        """Mask must be linearized indices that of the  upside down pixel map"""
        xbins, ybins, _ = self.mod_image.data

        ret = binned_statistic_2d(ey, ex, 0, 'count', bins=[ybins, xbins])  # histogram returns along row than col
        self.mod_image.masked_add(ret.statistic, mask, self.pixels)  # MASK and RET.STATISTIC will be upside down

        pxl_filter = np.isin(ret.binnumber, mask)
        self.module_energy_histogram.fill(sum_e[pxl_filter])
        self.E1.fill(valid_pmts[0, pxl_filter])  # ul
        self.E2.fill(valid_pmts[1, pxl_filter])  # ur
        self.E3.fill(valid_pmts[2, pxl_filter])  # ll
        self.E4.fill(valid_pmts[3, pxl_filter])  # lr

    def weighted_masks_events(self, ex, ey, sum_e, valid_pmts, masks):
        """Masks must be a list of linearized indices that of the  upside down pixel maps"""
        xbins, ybins, _ = self.mod_image.data
        ret = binned_statistic_2d(ey, ex, 0, 'count', bins=[ybins, xbins])  # Just needs to be done ONCE

        for idx, (pmt_hist, mask) in enumerate(zip(self.pmt_histograms, masks)):
            #  pmt is an object and masks is now a single mask from the list
            self.mod_image.masked_add(ret.statistic, mask, self.pixels)

            pxl_filter = np.isin(ret.binnumber, mask)
            self.module_energy_histogram.fill(sum_e[pxl_filter])
            pmt_hist.fill(valid_pmts[idx, pxl_filter])


class system_processing(object):

    def __init__(self, filepaths, place="Davis", **kwargs):
        if type(filepaths) == str:
            files = [filepaths]
        else:
            files = filepaths

        self.runs = []
        for file in files:
            self.runs.append(events_recon(file, place=place, **kwargs))

        self.system_id = np.arange(16)  # when facing front of detectors, upper left to upper right, then down
        if place == "Davis":
            self.mod_id = np.array([15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0])  # The order they were
        # plugged in by channel id
            print("Place: Davis")
        else:
            self.mod_id = np.arange(16)
            print("Place: Berkeley")

        self.mod_num_energy_bins = self.runs[0].mod_histogram_bins.size - 1
        self.pmt_num_energy_bins = self.runs[0].pmt_histogram_bins.size - 1
        self.pmt_histograms = np.zeros([64, self.pmt_num_energy_bins])
        self.module_histograms = np.zeros([16, self.mod_num_energy_bins])
        self.dyn_pmt_gains = np.ones(64)
        self.dyn_mod_gains = np.ones(16)

        self.data_generated = False
        self.image_list = [0 for _ in np.arange(16)]  # list of module images by SID
        self.filter_limits = [0]

    def _subprocess_mod_sum_histograms(self, rid, sid, **kwargs):  # raw_id (digitizer), system_id (global labeling)
        total_energy_spectra = np.zeros(self.mod_num_energy_bins)
        tot_E1 = np.zeros(self.pmt_num_energy_bins)
        tot_E2 = np.zeros_like(tot_E1)
        tot_E3 = np.zeros_like(tot_E1)
        tot_E4 = np.zeros_like(tot_E1)

        for run_number, run in enumerate(self.runs):
            eng, img = run.convert_adc_to_bins(rid, sid, **kwargs)
            if not run_number:  # first iteration
                # print("SID ", sid, " img.sum() : ", img.sum())
                # print("SID ", sid, " img.mean() : ", img.mean())
                # plt.imshow(img, cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')
                # plt.show()
                self.image_list[sid] = img
                # plt.imshow(self.image_list[sid], cmap='viridis', origin='lower', interpolation='nearest', aspect='equal')
                # plt.show()
            else:
                self.image_list[sid] += img

            total_energy_spectra += eng[0]
            tot_E1 += eng[1]
            tot_E2 += eng[2]
            tot_E3 += eng[3]
            tot_E4 += eng[4]
        return tot_E1, tot_E2, tot_E3, tot_E4, total_energy_spectra

    def _complete_run_histograms(self, **kwargs):  # kwarg -> energy_filter
        for sid in self.system_id:
            sid_ch_ind = 4 * sid
            dyn_pmt_gain_mod = self.dyn_pmt_gains[sid_ch_ind:(sid_ch_ind + 4)]
            pmt1, pmt2, pmt3, pmt4, mod_eng = \
                self._subprocess_mod_sum_histograms(4 * self.mod_id[sid], sid,
                                                    dynamic_pmt_gains=dyn_pmt_gain_mod,
                                                    dynamic_mod_gains=self.dyn_mod_gains[sid], **kwargs)
            yield sid, [pmt1, pmt2, pmt3, pmt4, mod_eng]

    def generate_spectra(self, **kwargs):  # crystal_binning, energy_filter
        if 'filter_limits' in kwargs:
            self.filter_limits = kwargs['filter_limits']

        for sid, hists in self._complete_run_histograms(**kwargs):
            print("System ID ", str(sid), "(Module ID ", str(self.mod_id[sid]), ") Processed")
            self.pmt_histograms[(4 * sid):((4*sid) + 4)] = np.vstack(hists[:-1])  # pmt1, 2, 3, 4
            self.module_histograms[sid] = hists[-1]
        self.data_generated = True

    def display_spectra_and_image(self, mod_id=None, energy_axis=False, save_fname=None, image_name=None):
        # Display module histograms and pmt histograms
        if not self.data_generated:
            return
        fig = plt.figure(figsize=(12, 9))
        gs = GridSpec(2, 4)
        ax1 = fig.add_subplot(gs[0, :2])  # modules
        ax2 = fig.add_subplot(gs[1, :2])  # pmts
        ax3 = fig.add_subplot(gs[:, 2:])  # projection image

        x_mod = np.linspace(0, self.runs[0].mod_histogram_bins[-1], self.runs[0].mod_histogram_bins.size - 1)  # mods
        x_pmt = np.linspace(0, self.runs[0].pmt_histogram_bins[-1], self.runs[0].pmt_histogram_bins.size - 1)  # PMTs

        if energy_axis:
           x_mod[1:] = 2 * np.log(x_mod[1:]) - 17.

        if mod_id is not None:
            mods = np.array([mod_id])
            image = self.image_list[mod_id]
        else:
            mods = self.system_id  # all of them
            image = self.full_image(self.image_list)

        # print("Mods: ", mods)
        for sid in mods:
            ax1.step(x_mod, self.module_histograms[sid], where='mid', label='mod ' + str(self.mod_id[sid]))
            for pmt_ind in np.arange(4):
                pmt_num = (4 * self.mod_id[sid] + pmt_ind)
                ax2.step(x_pmt, self.pmt_histograms[pmt_num],
                         where='mid',
                         label='m' + str(self.mod_id[sid]) + 'p' + str(pmt_ind))

        # self.image_list[8] = np.zeros([12, 12])  # orient up/down for image

        im = ax3.imshow(image, cmap='viridis', origin='upper', interpolation='nearest', aspect='equal')
        # TODO: image.T is probably not correct. Still need lower origin?
        ax3.axis('off')
        fig.colorbar(im, fraction=0.046, pad=0.04, ax=ax3)
        if type(image_name) is str:
            ax3.set_title(image_name)

        ax1.set_yscale('log')
        ax1.set_title('Module Spectrum')

        ll = np.min(np.argmax(self.module_histograms > 0, axis=1))
        ul = self.module_histograms.shape[1] - np.min(np.argmax(np.fliplr(self.module_histograms) > 0, axis=1)) - 1

        ax1.set_xlim([np.max([0, 0.9 * x_mod[ll]]), np.min([1.01 * x_mod[ul], 1.01 * x_mod[-1]])])
        ax1.legend(loc='upper right')

        pll = np.min(np.argmax(self.pmt_histograms > 0, axis=1))
        pul = self.pmt_histograms.shape[1] - np.min(np.argmax(np.fliplr(self.pmt_histograms) > 0, axis=1)) - 1

        ax2.set_xlim([np.max([0, 0.9 * x_pmt[pll]]), np.min([1.01 * x_pmt[pul], 1.01 * x_pmt[-1]])])
        ax2.set_yscale('log')
        ax2.set_title('PMT Spectrum')

        fig.tight_layout()
        if type(save_fname) is str:
            plt.savefig(save_fname, bbox_inches="tight")

        return fig, [ax1, ax2, ax3]

    @staticmethod
    def full_image(image_list):
        return np.block([image_list[col:col + 4] for col in np.arange(0, len(image_list), 4)])

    def calibrate_pmt_gains(self):  # use gaussian_filter from scipy along axis
        pass

    def calibrate_mod_gains(self, roi_center, roi_window, shape='edge', ma_sze=3):  # , shape_width=5):
        if shape not in ('edge', 'peak') or not self.data_generated:
            return

        bn = np.diff(self.runs[0].mod_histogram_bins[:2])
        cnt_ind = roi_center//bn
        w = roi_window//bn
        region_of_interest = self.module_histograms[:,  np.arange(cnt_ind-w, cnt_ind + w + 1)]
        # issues if cnt_ind -/+ w is near edges

        mode = 'nearest'

        smoothed = uniform_filter1d(1.0 * region_of_interest, ma_sze, axis=1, mode=mode)

        if shape is 'edge':
            logged = np.log(smoothed, out=np.zeros_like(smoothed), where=(smoothed != 0))

            mod_ref_pts = np.argmax(np.diff(logged, axis=1), axis=1) + cnt_ind - w
            mod_standard = np.argmin(mod_ref_pts)
            print("Old Dynamic Mod Gains: ", self.dyn_mod_gains)
            self.dyn_mod_gains = mod_ref_pts[mod_standard] / mod_ref_pts
            print("New Dynamic Mod Gains: ", self.dyn_mod_gains)

    def save_hist_and_calib(self, filename):  # TODO: image_list is not saving or loading correctly
        np.savez(filename, dyn_pmt_gains=self.dyn_pmt_gains,
                 dyn_mod_gains=self.dyn_mod_gains,
                 pmt_histograms=self.pmt_histograms,
                 pmt_histogram_bins=self.runs[0].pmt_histogram_bins,  #
                 filter_limits=self.filter_limits,
                 # energy_filter=self.energy_filter,
                 module_histograms=self.module_histograms,
                 mod_histogram_bins=self.runs[0].mod_histogram_bins,  #
                 image_list=self.full_image(self.image_list))

    def load_hist_and_calib(self, filename):  # TODO: Break up image_list attribute back into a list
        data = np.load(filename)
        for key, value in data.items():
            if key in ('mod_histogram_bins', 'pmt_histogram_bins'):
                for run in self.runs:
                    setattr(run, key, value)
                continue
            setattr(self, key, value)
        self.data_generated = True


class Hist1D(object):
    def __init__(self, xbins):
        self.hist, self.xbins = np.histogram([], bins=xbins)

    def fill(self, arr):
        self.hist += np.histogram(arr, bins=self.xbins)[0]

    def clear(self):
        self.hist.fill(0)

    @property
    def data(self):
        return self.xbins, self.hist.copy()


class HistImage(object):  # TODO: ybins and xbins in right spot?
    def __init__(self, crystal_cutsX, crystal_cutsY):
        self.hist, self.ybins, self.xbins = \
            np.histogram2d([], [], bins=[crystal_cutsY[0].ravel(), crystal_cutsX[0].ravel()])
        self.x_edges_table = crystal_cutsX
        self.y_edges_table = crystal_cutsY
        self.current_module = 0

    def select(self, mod_id):
        self.current_module = mod_id
        self.clear()
        self.xbins = self.x_edges_table[mod_id].ravel()
        self.ybins = self.y_edges_table[mod_id].ravel()

    def fill(self, xarr, yarr):  # These get switched when entered
        self.hist += np.histogram2d(yarr, xarr, bins=[self.ybins, self.xbins])[0][::-1]  # .T
        # The [::-1] index is needed to reverse what histogram does i.e. turn it back rightside up

    def clear(self):
        self.hist.fill(0)

    def masked_add(self, counts, mask, pixels):
        unravel_mask = np.zeros([pixels + 2, pixels + 2])
        unravel_mask[np.unravel_index(mask, unravel_mask.shape)] = 1
        self.hist += (counts * unravel_mask[1:-1, 1:-1])[::-1]  # Must revert to right side up

    @property
    def data(self):
        return self.xbins, self.ybins, self.hist.copy()


def load_signals(filepath):
    if tables.is_hdf5_file(filepath):
        h5file = tables.open_file(filepath, 'r')
        return h5file
    else:
        raise ValueError('{fi} is not a hdf5 file!'.format(fi=filepath))


def create_mask(type='corners', buffer=3, single_pxls=np.array([0, 0]), pixels=12):
    # The purpose of this is for use with multiplying with ret.binnumber
    if type == 'corners':
        mask = np.zeros([pixels, pixels])

        pmt_id = np.arange(4)  # pmt_ch_map = np.array([ul, ur, ll, lr])
        row = pmt_id // 2  # array([0, 0, 1, 1])
        col = pmt_id % 2  # array([0, 1, 0, 1])

        masks = []
        for r, c in zip(row, col):
            # print("Row: ", r)
            # print("Col: ", c)
            mask.fill(0)
            start_row = r * (pixels-buffer)
            start_col = c * (pixels-buffer)
            mask[start_row:(start_row+buffer), start_col:(start_col+buffer)] = 1
            # print(mask)
            masks.append(np.ravel_multi_index(np.where(np.pad(mask[::-1], 1)), np.asarray(mask.shape) + 2))

    else:  # single points
        mask = np.zeros([pixels, pixels])
        mask[single_pxls[..., 0], single_pxls[..., 1]] = 1  # Need to flip this
        masks = np.ravel_multi_index(np.where(np.pad(mask[::-1], 1)), np.asarray(mask.shape)+2)

    return masks


def main_th_measurement():  # one_module_processing for outstanding issues
    # base_path = '/home/proton/repos/python3316/Data/'
    base_path = '/Users/justinellin/Desktop/'
    # files = ['2020-10-31-1704.h5']
    files = ['2020-10-07-1940.h5']
    location = "Davis"  # was berkeley
    filepaths = [base_path + file for file in files]
    full_run = system_processing(filepaths, place=location, mod_adc_max_bin=80000)

    # e_filter = [1]
    # full_run.generate_spectra(filter_limits=e_filter, energy_filter=True, crystal_bin=True)

    e_filter = [20000, 35000]  # Feb 15
    full_run.generate_spectra(filter_limits=e_filter)
    full_run.calibrate_mod_gains(29000, 4000, ma_sze=1)
    # tot_e = [5000, 35000]
    full_run.generate_spectra(filter_limits=e_filter)

    # [0.98347107 0.93700787 1.         0.97540984 0.952      0.90839695
    #  0.90839695 0.9296875  0.91538462 0.92248062 0.92248062 0.91538462
    #  0.99166667 1.         0.94444444 0.96747967]

    # fig, axes = full_run.display_spectra_and_image(save_fname="th_flood_1031_feb_15")  # to allow for changing of axes
    fig, axes = full_run.display_spectra_and_image()
    print("Total Events: ", full_run.module_histograms.sum())
    plt.show()

    for run in full_run.runs:
        run.h5file.close()
    # full_run.save_hist_and_calib(filename="th_flood_1031_feb_15")


def main_step_measurement():  # one_module_processing for outstanding issues
    base_path = '/home/proton/repos/python3316/Data/'
    files56 = ['2020-10-07-1210.h5', '2020-10-07-1219.h5', '2020-10-07-1221.h5', '2020-10-07-1222.h5',
               '2020-10-07-1224.h5', '2020-10-07-1225.h5', '2020-10-07-1226.h5', '2020-10-07-1228.h5',
               '2020-10-07-1229.h5', '2020-10-07-1230.h5']  # step measurement 50-59, 5-6 cm

    # list_of_files = [files12, files23, files34, files45, files56, files67, files78, files89, files9]
    list_of_files = [files56]

    for start in np.arange(len(list_of_files)):
        step = start + 1  # start is at 0 but first run is at 1 cm
        filepaths = [base_path + file for file in list_of_files[start]]

        save_fname = '/home/proton/repos/python3316/processing/images/step_run_' + str(step) + "t" \
                     + str(step + 1) + 'cm_Feb10'

        plot_name = 'Position ' + str(step) + '-' + str(step + 1) + ' cm'

        full_run = system_processing(filepaths, mod_adc_max_bin=160000, pmt_adc_max_bin=80000)
        e_filter = [20000, 100000]
        full_run.generate_spectra(filter_limits=e_filter)
        # full_run.display_spectra_and_image(save_fname=save_fname, image_name=plot_name)
        full_run.display_spectra_and_image()
        plt.show()
        for run in full_run.runs:
            run.h5file.close()

        # full_run.save_hist_and_calib(filename=save_fname)


def fix_loading_issues():
    pass


if __name__ == "__main__":
    # TODO: Check masks work for thorium measurement and image makes sense.
    #  Fix full_image. Double check crystal maps.
    # main()
    main_th_measurement()
    # main_step_measurement()