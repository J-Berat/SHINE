"""
Physical constants used across the HI pipeline (CGS unless stated otherwise).
"""

# Boltzmann constant [J K^-1] (SI, kept as in the original IDL translation).
const K_PLANCK = 1.3806488e-23
# Hydrogen atom mass [g].
const M_H = 1.6737236e-24
# 21-cm column-density / optical-depth constant [cm^-2 K^-1 (km/s)^-1].
# N_HI = C_TAU * T_spin * tau * dv, with dv in km/s (Draine 2011, eq. 8.7).
const C_TAU = 1.82243e18
# Parsec in centimetres.
const PC_TO_CM = 3.0856775814913673e18
