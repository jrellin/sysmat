import numpy as np
from scipy import constants as sc
# Remember: Stop at 10 MeV since no more gammas


def main():
    m_p = 938  # MeV
    c = sc.speed_of_light  # m/s
    target_energies = np.array([10, 15, 60, 67.5])  # MeV
    # ranges = np.array([1.258, 4.363, 3.172, 3.92]) * (10.0 ** np.array([-1, -1, 0, 0]))  # g/cm^2
    # TODO: Look up
    density = 1.18  # PMMA (g/cm^3)

    stopping_times = np.zeros(target_energies.shape)

    # dist = ranges / density
    # print("Energies (MeV): ", energies)
    # print("Range (in cm): ", dist)

    fields = ['Energy', 'SP', 'CSDA', 'Projected']  # MeV, Stopping Power, CSDA, Projected
    data = np.genfromtxt('pstar.txt', skip_header=6, names=fields)
    dEdX = data['SP'] * density
    energies = data['Energy']
    wgt = np.r_[energies[0], np.ediff1d(energies)]  # really dE to next data point

    avg_dEdX = dEdX * wgt  # unnormalized

    for ind, target_E in enumerate(target_energies):
        upper_ind = np.argmax(energies > target_E)  # next value above target
        print("Closest Energy: ", energies[upper_ind])
        dE_inc = energies[upper_ind] - target_E
        v_o = np.sqrt(2 * target_E/m_p) * (100 * c)  # c in cm/s
        avg_sp = ((dE_inc * dEdX[upper_ind]) + (np.sum(avg_dEdX[:upper_ind])))/target_E
        stopping_times[ind] = target_E / (v_o * avg_sp)

    print("Stopping Times (ps): ", stopping_times * 10 ** 12)
    # print("Ranges (cm): ", data['Projected'] / density)


if __name__ == "__main__":
    main()
