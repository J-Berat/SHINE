"""
    HIspectrum(n, v, T, velvec, dz; mu = 1.0, therm = 0.0) -> (Tb, Tb_thin, tau)

Solve the 21-cm radiative transfer along a single line of sight and return the
brightness-temperature spectrum.

The line of sight is discretised into `length(n)` cells. Each cell contributes a
Gaussian velocity profile centred on its bulk velocity `v[k]` and broadened by
the thermal dispersion. Cells are integrated **front to back** so that emission
from a cell is attenuated by the optical depth of all foreground cells, which is
what makes the result differ from the optically thin limit.

# Arguments
- `n`      : HI number density along the LOS [cm^-3].
- `v`      : bulk velocity along the LOS [km/s].
- `T`      : kinetic (spin) temperature along the LOS [K].
- `velvec` : output velocity channels [km/s].
- `dz`     : cell depth along the LOS [cm].
- `mu`     : mean molecular weight used for the thermal dispersion (default 1.0).
- `therm`  : if `> 0`, a fixed velocity dispersion [km/s] overriding the thermal
             one (useful to add sub-grid turbulence).

# Returns
- `Tb`      : full (optically thick) brightness temperature [K] per channel.
- `Tb_thin` : optically thin brightness temperature [K] per channel.
- `tau`     : total optical depth per channel (integrated over the LOS).

Translated from IDL (MAMD, 24/10/2023) and cleaned up: constants are named,
the thermal dispersion is guarded against non-positive temperatures, and the
accumulation is allocation-free per channel.
"""
function HIspectrum(n, v, T, velvec, dz; mu::Real = 1.0, therm::Real = 0.0)
    nb = length(n)
    nb == length(v) == length(T) ||
        throw(DimensionMismatch("n, v and T must share the same LOS length."))

    # Thermal velocity dispersion [km/s]. sqrt(k T / (m mu)) with m in g and the
    # 1e3 factor converting the SI Boltzmann constant to a km/s dispersion.
    if therm > 0
        sig_therm = fill(float(therm), nb)
    else
        sig_therm = @. sqrt(K_PLANCK * max(T, zero(eltype(T))) / (1.0e3 * M_H * mu))
    end

    nbvec = length(velvec)
    Tb = zeros(nbvec)
    Tb_thin = zeros(nbvec)
    tau_in_front = zeros(nbvec)

    @inbounds for k in 1:nb
        (n[k] <= 0 || T[k] <= 0 || sig_therm[k] <= 0) && continue

        inv_sig = 1.0 / sig_therm[k]
        norm = inv_sig / sqrt(2π)
        Tk = T[k]

        for j in 1:nbvec
            # Gaussian velocity profile [ (km/s)^-1 ].
            arg = (velvec[j] - v[k]) * inv_sig
            G = exp(-0.5 * arg * arg) * norm
            # Channel column density [cm^-2 (km/s)^-1] -> optical depth.
            N = G * n[k] * dz
            tau_k = N / (C_TAU * Tk)

            Tb[j] += Tk * (1.0 - exp(-tau_k)) * exp(-tau_in_front[j])
            Tb_thin[j] += tau_k * Tk
            tau_in_front[j] += tau_k
        end
    end

    return Tb, Tb_thin, tau_in_front
end
