#/usr/bin/env python

# This code calculates the instantaneous energy deposition due to radioactive
# decay in a SN Ia (called "S_dot" in PHOENIX), using as input a handful of
# parameters as described in Stritzinger,+(2006).

# Starting about 1 month after explosion, the bolometric luminosity (L_bol) of
# a SN Ia at a given time is equal to the energy deposition at that time,
# because the opacity in the ejecta is low enough that almost all of the energy
# released from Ni56 and Co56 decay escapes immediately as radiation.

# This code is therefore useful to estimate L_bol when trying to calulate
# late-time spectra of SNe Ia with PHOENIX. Just provide parameters such as the
# total ejected mass (M_ej), the total Ni56 mass (M_Ni56), and the age of the
# SN, and it will return L_bol.

from astropy import units as u
import math
import numpy as np
from scipy.constants import N_A, golden as golden_ratio
from matplotlib import pyplot as plt, gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

# Decay constants for Ni56 and Co56, of the form dN/dt = -lambda*N.
lambda_Ni = (   8.8 * u.day )**(-1)
lambda_Co = ( 111.3 * u.day )**(-1)

Q_Ni_gamma = 1.75e+6 * u.eV     # Energy released from gamma-rays per Ni56 -> Co56 decay (there are no positrons from Ni56).
Q_Co_gamma = 3.61e+6 * u.eV     # Energy released from gamma-rays per Co56 -> Fe56 decay.
Q_Co_pos   = 0.12e+6 * u.eV     # Energy released from positrons per Co56 -> Fe56 decay.

Ni56_atomic_mass = 55.942 * u.gram      # Molar mass of Ni56.

# Instantaneous energy deposition due to gamma-rays and positrons, a la Eq. 2 of Stritzinger,+(2006).
def E_dep( t, \
          M_Ni56 = 0.5 * u.solMass, \
          M_ej = 1.3 * u.solMass, \
          kappa = 0.025 * u.cm**2 / u.gram, \
          q = 1.0/3.0 * u.dimensionless_unscaled, \
          v_e = 3000.0 * u.km / u.second ):
    N_Ni_0 = M_Ni56.to(u.gram) * N_A / Ni56_atomic_mass     # Convert mass of Ni56 to # of Ni56 atoms.
    M_ej = M_ej.to(u.gram)
    v_e = v_e.to(u.cm/u.second)
    t_0 = np.sqrt(M_ej * kappa * q / (8.0 * math.pi))
    t_0 /= v_e      # This completes Eq. 4 of Stritzinger,+(2006).
    t_0 = t_0.to(u.day)
    tau = np.power(t_0, 2) / np.power(t, 2)
    return ( (lambda_Ni * N_Ni_0 * math.exp(-lambda_Ni * t) * Q_Ni_gamma) \
            + lambda_Co * N_Ni_0 * (lambda_Ni / (lambda_Ni - lambda_Co)) * ( (math.exp(-lambda_Co * t) - math.exp(-lambda_Ni * t)) \
            * (Q_Co_pos + Q_Co_gamma * (1.0 - math.exp(-tau))) ) )

t = np.arange(1.0, 500.0, 1.0) * u.day      # Create grid of days.

# Set parameters.
M_Ni56 = 0.50 * u.solMass
M_ej = 1.38 * u.solMass
kappa = 0.025 * u.cm**2 / u.gram
q = 1.0/3.0 * u.dimensionless_unscaled
v_e = 3000.0 * u.km / u.second

E_dep_array = []
time_array = []
for i, time in enumerate(t):
    time_array.append ( time.value )
    E_dep_array.append( E_dep( time, M_Ni56, M_ej, kappa, q, v_e ).to( u.erg / u.second ).value )

gs = gridspec.GridSpec(1, 1)

fig = plt.figure(figsize=(11.0, 11.0 / golden_ratio), dpi=128)
ax = fig.add_subplot(gs[0, 0])

ax.plot(time_array, E_dep_array)
ax.set_yscale('log')

xmajorLocator = MultipleLocator(100) # Major ticks every 100 days.
xminorLocator = MultipleLocator(20) # Minor ticks every 20 days.
xmajorFormatter = FormatStrFormatter('%d')
ax.xaxis.set_major_locator(xmajorLocator)
ax.xaxis.set_minor_locator(xminorLocator)
ax.xaxis.set_major_formatter(xmajorFormatter)

# Use different grid line styles for major and minor ticks (for clarity).
plt.grid(b=True, which='major', color='b', linestyle='-')
plt.grid(b=True, which='minor', color='r', linestyle='--')

title = 'M_Ni56 = %.2f %s; M_ej = %.2f %s; \n kappa = %.3f %s; q = %.2f; v_e = %.1f %s' % (M_Ni56.value, M_Ni56.unit, M_ej.value, M_ej.unit, kappa.value, kappa.unit, q.value, v_e.value, v_e.unit)

ax.set_xlabel('time (days)')
ax.set_ylabel('energy deposition (erg/s)')
ax.set_title('instantaneous energy deposition due to Ni56 and Co56 decay:\n' + title)

plt.show()

figname = 'edep_M_Ni56_%.2f_M_ej_%.2f_kappa_%.3f_q_%.2f_v_e_%.1f' % (M_Ni56.value, M_ej.value, kappa.value, q.value, v_e.value)
fig.savefig(figname + '.png', dpi=300)
fig.savefig(figname + '.eps', dpi=300)

plt.close()
