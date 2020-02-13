import numpy as np
import pickle
import matplotlib.pyplot as plt

with open("kernels.pkl", "rb") as fp:
    kernels = pickle.load(fp)  # will give kernel size in x

im_pxl_sze = 0.1

# Physical Constants
N_A = 6.02214076e+23  # avogadro
mb = 1e-27  # in cm^2
MM = 100.12  # Molar mass PMMA (g/mol)
dens = 1.18  # mass dens PMMA (g/cm^3)
N_dens = dens / MM * N_A * mb

# PMMA constants
PMMA_C = 5  # C5O2H8
PMMA_O = 2
PMMA_H = 8
frac_C = PMMA_C/(PMMA_C + PMMA_O + PMMA_H)
frac_O = PMMA_O/(PMMA_C + PMMA_O + PMMA_H)


avg_energies = kernels['Average_Energy']
kern_sze = len(avg_energies)
degrees = np.linspace(0, 180, num=181)
costh = np.cos(np.deg2rad(degrees))


# wgt712 = (kernels['Oxy712']['sig'][0]) * np.array([1, 0, kernels['Oxy712']['a20'][0], 0, 0])
yld_Oxy712 = [None] * kern_sze
yld_Oxy692 = [None] * kern_sze
yld_Oxy613 = [None] * kern_sze

yld_OxyTot = [None] * kern_sze
bas_OxyTot = [None] * kern_sze
bas_OxyTotPoly = [None] * kern_sze

# Oxygen
for i in np.arange(kern_sze):
    wgt_712 = N_dens * im_pxl_sze * frac_O * (kernels['Oxy712']['sig'][i]) \
              * np.array([1, 0, kernels['Oxy712']['a20'][i], 0, 0, 0, 0])
    bas_Oxy712 = np.polynomial.Legendre(wgt_712)
    yld_Oxy712[i] = bas_Oxy712(costh)

    wgt_692 = N_dens * im_pxl_sze * frac_O * (kernels['Oxy692']['sig'][i]) \
              * np.array([1, 0, kernels['Oxy692']['a20'][i], 0, kernels['Oxy692']['a40'][i], 0, 0])
    bas_Oxy692 = np.polynomial.Legendre(wgt_692)
    yld_Oxy692[i] = bas_Oxy692(costh)

    wgt_613 = N_dens * im_pxl_sze * frac_O * (kernels['Oxy613']['sig'][i]) \
              * np.array([1, 0, kernels['Oxy613']['a20'][i], 0, kernels['Oxy613']['a40'][i], 0,
                          kernels['Oxy613']['a60'][i]])
    bas_Oxy613 = np.polynomial.Legendre(wgt_613)
    yld_Oxy613[i] = bas_Oxy613(costh)

    bas_OxyTot = bas_Oxy712 + bas_Oxy692 + bas_Oxy613
    bas_OxyTotPoly[i] = bas_OxyTot
    yld_OxyTot[i] = bas_OxyTot(costh)


# wgt712 = N_dens * frac_O * (kernels['Oxy712']['sig'][0]) * np.array([1, 0, kernels['Oxy712']['a20'][0], 0, 0])

# print("Norm:", np.average(y))

fig, axes = plt.subplots(2, 2, figsize=(12, 7))
ax0 = axes[0, 0]
ax1 = axes[0, 1]
ax2 = axes[1, 0]
ax3 = axes[1, 1]

for i in np.arange(len(yld_Oxy712)):
    ax0.plot(degrees, yld_Oxy613[i], label=str(int(avg_energies[i])) + ' MeV')
    # ax0.plot(degrees, yld_Oxy613[i]/np.max(yld_Oxy613[i]), label=str(int(avg_energies[i])) + ' MeV')
    ax0.set_ylabel('Rel. Emission Prob.')
    ax0.set_xlabel('Degrees')

    ax1.plot(degrees, yld_Oxy692[i], label=str(int(avg_energies[i])) + ' MeV')
    # ax1.plot(degrees,  yld_Oxy692[i] / np.max( yld_Oxy692[i]), label=str(int(avg_energies[i])) + ' MeV')
    ax1.set_ylabel('Rel. Emission Prob.')
    ax1.set_xlabel('Degrees')

    ax2.plot(degrees, yld_Oxy712[i], label=str(int(avg_energies[i])) + ' MeV')
    # ax2.plot(degrees, yld_Oxy712[i] / np.max(yld_Oxy712[i]), label=str(int(avg_energies[i])) + ' MeV')
    ax2.set_ylabel('Rel. Emission Prob.')
    ax2.set_xlabel('Degrees')

    ax3.plot(degrees, yld_OxyTot[i], label=str(int(avg_energies[i])) + ' MeV')
    # ax3.plot(degrees, yld_OxyTot[i] / np.max(yld_OxyTot[i]), label=str(int(avg_energies[i])) + ' MeV')
    ax3.set_ylabel('Rel. Emission Prob.')
    ax3.set_xlabel('Degrees')

ax0.legend(loc='best')
ax0.set_title('Oxygen-6.13 MeV')

ax1.legend(loc='best')
ax1.set_title('Oxygen-6.92 MeV')

ax2.legend(loc='best')
ax2.set_title('Oxygen-7.12 MeV')

ax3.legend(loc='best')
ax3.set_title('Oxygen Total')

plt.tight_layout()
# plt.savefig('images/basis_oxygen.png')

plt.show()
plt.close(fig)

# Carbon
figC, (ax_raw, ax_norm) = plt.subplots(1, 2, figsize=(12, 4))
yld_C = [None] * kern_sze
for i in np.arange(kern_sze):
    wgt_C = N_dens * im_pxl_sze * frac_C * (kernels['Carbon']['sig'][i]) \
              * np.array([1, 0, kernels['Carbon']['a20'][i], 0, kernels['Carbon']['a40'][i], 0, 0])
    bas_C = np.polynomial.Legendre(wgt_C)
    yld_C[i] = bas_C(costh)
    ax_raw.plot(degrees, yld_C[i], label=str(int(avg_energies[i])) + ' MeV')
    ax_norm.plot(degrees, yld_C[i]/np.max(yld_C[i]), label=str(int(avg_energies[i])) + ' MeV')

ax_raw.set_ylabel('Emission Prob.')
ax_raw.set_xlabel('Degrees')
ax_raw.legend(loc='best')
ax_raw.set_title('Carbon-4.4 MeV')

ax_norm.set_ylabel('Rel. Emission Prob.')
ax_norm.set_xlabel('Degrees')
ax_norm.legend(loc='best')
ax_norm.set_title('Carbon-4.4 MeV')

colours = [None] * 3
for ind in np.arange(len(ax_norm.lines)):
    colours[ind] = ax_norm.lines[ind]._color
print(colours)

plt.tight_layout()
# plt.savefig('images/basis_carbon.png')
plt.show()
plt.close(figC)

# Both Together
fig_f, (ax_rawf, ax_normf) = plt.subplots(1, 2, figsize=(12, 6))
for i in np.arange(kern_sze):
    ax_rawf.plot(degrees, yld_C[i], label=str(int(avg_energies[i])) + ' MeV (C)', color=colours[i])
    ax_normf.plot(degrees, yld_C[i] / np.max(yld_C[i]),
                  label=str(int(avg_energies[i])) + ' MeV (C)', color=colours[i])

    ax_rawf.plot(degrees, yld_OxyTot[i], linestyle='--', label=str(int(avg_energies[i])) + ' MeV (O)', color=colours[i])
    ax_normf.plot(degrees, yld_OxyTot[i]/np.max(yld_OxyTot[i]), linestyle='--',
                 label=str(int(avg_energies[i])) + ' MeV (O)', color=colours[i])

ax_rawf.set_ylabel('Emission Prob.')
ax_rawf.set_xlabel('Degrees')
ax_rawf.legend(loc='best')
ax_rawf.set_title('Probability for Oxygen and Carbon in PMMA')

ax_normf.set_ylabel('Relative Emission Prob.')
ax_normf.set_xlabel('Degrees')
ax_normf.legend(loc='best')
ax_normf.set_title('Normalized Probability for Oxygen and Carbon in PMMA')

plt.tight_layout()
plt.savefig('images/prob_emissions.png')
plt.show()
