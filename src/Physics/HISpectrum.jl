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
    dz > 0 || throw(ArgumentError("dz must be positive (got $dz)."))
    mu > 0 || throw(ArgumentError("mu must be positive (got $mu)."))
    therm >= 0 || throw(ArgumentError("therm must be non-negative (got $therm)."))

    nbvec = length(velvec)
    Tb = zeros(nbvec)
    Tb_thin = zeros(nbvec)
    tau = zeros(nbvec)
    trans = ones(nbvec)
    vel = _velocity_channels(velvec)

    _HIspectrum!(Tb, Tb_thin, tau, trans, n, v, T, vel, dz, mu, therm,
                 _uniform_velocity_grid(vel))
    return Tb, Tb_thin, tau
end

# Gaussian line profiles are truncated at this many standard deviations. At 8σ
# the neglected wing is exp(-32) ≈ 1e-14 of the profile peak — below the
# rounding error of the accumulated sum — so the truncated result agrees with
# the untruncated one to ~1e-15 relative while touching far fewer channels.
const GAUSS_TRUNCATION_NSIG = 8.0

"Velocity channels as a plain `Vector{Float64}`, avoiding repeated conversion."
_velocity_channels(velvec) = collect(float.(velvec))
_velocity_channels(velvec::Vector{Float64}) = velvec

"""
Describe `velvec` as `(v0, inv_dv)` when it is uniformly spaced, else `nothing`.

A uniform grid lets the kernel jump straight to the channels a line profile can
reach; on an irregular grid the kernel falls back to scanning every channel.
"""
function _uniform_velocity_grid(velvec)
    nbvec = length(velvec)
    nbvec >= 2 || return nothing
    v0 = float(velvec[1])
    dv = (float(velvec[end]) - v0) / (nbvec - 1)
    (isfinite(dv) && dv != 0) || return nothing
    tol = 1e-6 * abs(dv)
    @inbounds for j in eachindex(velvec)
        abs(float(velvec[j]) - (v0 + (j - 1) * dv)) <= tol || return nothing
    end
    return (v0, 1.0 / dv)
end

# Channels reachable by a Gaussian of width `sig` centred on `vk`, clamped to
# `1:nbvec`. The float comparisons before `ceil`/`floor` keep a very wide
# profile from overflowing the integer conversion.
@inline function _channel_range(grid, vk, sig, nbvec)
    grid === nothing && return 1:nbvec
    v0, inv_dv = grid
    centre = (vk - v0) * inv_dv + 1
    half = GAUSS_TRUNCATION_NSIG * sig * abs(inv_dv)
    lo = centre - half
    hi = centre + half
    jlo = lo <= 1 ? 1 : ceil(Int, lo)
    jhi = hi >= nbvec ? nbvec : floor(Int, hi)
    return jlo:jhi
end

@inline _sigma_los(T_k, mu, therm) =
    therm > 0 ? float(therm) : sqrt(K_PLANCK * T_k / (1.0e3 * M_H * mu))

# Allocation-free kernel shared by the spectrum and cube APIs. `Tb`, `Tb_thin`
# and `tau` must be zero-initialised by the caller and `trans` set to one;
# `trans` carries the foreground transmission exp(-tau_in_front) so the
# attenuation costs a multiplication rather than a second exponential.
function _HIspectrum!(Tb, Tb_thin, tau, trans, n, v, T, velvec, dz, mu, therm, grid)
    nb = length(n)
    nbvec = length(velvec)

    @inbounds for k in 1:nb
        n_k = n[k]
        T_k = T[k]
        (n_k <= 0 || T_k <= 0) && continue

        sig = _sigma_los(T_k, mu, therm)
        inv_sig = 1.0 / sig
        v_k = v[k]
        # Everything constant along the channel loop, including the conversion
        # from Gaussian profile to optical depth, is folded into one factor.
        amp = inv_sig / sqrt(2π) * n_k * dz / (C_TAU * T_k)

        for j in _channel_range(grid, v_k, sig, nbvec)
            arg = (velvec[j] - v_k) * inv_sig
            tau_k = exp(-0.5 * arg * arg) * amp

            # expm1 retains accuracy when tau_k is much smaller than machine
            # precision, which is common in optically thin cells; 1 + em is
            # exp(-tau_k), reused to advance the transmission.
            em = expm1(-tau_k)
            Tb[j] += T_k * (-em) * trans[j]
            Tb_thin[j] += tau_k * T_k
            tau[j] += tau_k
            trans[j] *= 1 + em
        end
    end
    return nothing
end

"""
    HIspectrum_tb(n, v, T, velvec, dz; mu = 1.0, therm = 0.0) -> Tb

Memory-saving radiative-transfer solver returning only the brightness spectrum.
"""
function HIspectrum_tb(n, v, T, velvec, dz; mu::Real = 1.0, therm::Real = 0.0)
    nb = length(n)
    nb == length(v) == length(T) ||
        throw(DimensionMismatch("n, v and T must share the same LOS length."))
    dz > 0 || throw(ArgumentError("dz must be positive (got $dz)."))
    mu > 0 || throw(ArgumentError("mu must be positive (got $mu)."))
    therm >= 0 || throw(ArgumentError("therm must be non-negative (got $therm)."))

    vel = _velocity_channels(velvec)
    Tb = zeros(length(vel))
    trans = ones(length(vel))
    _HIspectrum_tb!(Tb, trans, n, v, T, vel, dz, mu, therm, _uniform_velocity_grid(vel))
    return Tb
end

# As above without the optically thin and optical-depth outputs; `trans` must be
# set to one by the caller.
function _HIspectrum_tb!(Tb, trans, n, v, T, velvec, dz, mu, therm, grid)
    nb = length(n)
    nbvec = length(velvec)

    @inbounds for k in 1:nb
        n_k = n[k]
        T_k = T[k]
        (n_k <= 0 || T_k <= 0) && continue

        sig = _sigma_los(T_k, mu, therm)
        inv_sig = 1.0 / sig
        v_k = v[k]
        amp = inv_sig / sqrt(2π) * n_k * dz / (C_TAU * T_k)

        for j in _channel_range(grid, v_k, sig, nbvec)
            arg = (velvec[j] - v_k) * inv_sig
            tau_k = exp(-0.5 * arg * arg) * amp
            em = expm1(-tau_k)
            Tb[j] += T_k * (-em) * trans[j]
            trans[j] *= 1 + em
        end
    end
    return nothing
end
