"""
Mass- and volume-weighted fractions of the CNM/LNM/WNM phases, both as scalar
diagnostics for a whole cube and as plane-of-sky maps computed along the line of
sight.
"""

function _check_phase_bounds(n, T, TCNM, TWNM)
    axes(n) == axes(T) || throw(DimensionMismatch("n and T must share the same axes."))
    TCNM <= TWNM || throw(ArgumentError("TCNM ($TCNM) must be <= TWNM ($TWNM)."))
    return nothing
end

@inline function _check_mass_cell(ni, Ti)
    isfinite(ni) && isfinite(Ti) || throw(ArgumentError("n and T must contain only finite values."))
    ni >= 0 || throw(ArgumentError("density weights must be non-negative."))
    return nothing
end

"""
    GasFraction(n, T; TCNM = 200, TWNM = 2000) -> (fCNM, fLNM, fWNM)

Mass fraction (in percent) of each phase over the supplied density/temperature
arrays. `n` is used as the mass weight.
"""
function GasFraction(n::AbstractArray, T::AbstractArray; TCNM::Real = 200, TWNM::Real = 2000)
    _check_phase_bounds(n, T, TCNM, TWNM)

    mC = mL = mW = 0.0
    @inbounds for i in eachindex(n, T)
        ni, Ti = n[i], T[i]
        _check_mass_cell(ni, Ti)
        if Ti < TCNM
            mC += ni
        elseif Ti < TWNM
            mL += ni
        else
            mW += ni
        end
    end

    total_mass = mC + mL + mW
    total_mass > 0 || return (0.0, 0.0, 0.0)
    scale = 100 / total_mass
    return mC * scale, mL * scale, mW * scale
end

"""
    VolumeFraction(n, T; TCNM = 200, TWNM = 2000) -> (fCNM, fLNM, fWNM)

Volume filling fraction (in percent) of each phase, i.e. the fraction of cells in
each temperature bin (density-independent).
"""
function VolumeFraction(n::AbstractArray, T::AbstractArray; TCNM::Real = 200, TWNM::Real = 2000)
    _check_phase_bounds(n, T, TCNM, TWNM)
    all(isfinite, T) || throw(ArgumentError("T must contain only finite values."))
    total = length(T)
    total > 0 || return (0.0, 0.0, 0.0)

    nCNM = count(<(TCNM), T)
    nWNM = count(>=(TWNM), T)
    nLNM = total - nCNM - nWNM

    return nCNM / total * 100, nLNM / total * 100, nWNM / total * 100
end

# Shared driver for the per-pixel maps. `weight_of` returns the amount one cell
# adds to its phase — the density for a mass fraction, 1 for a volume fraction.
# `nlos` is the fixed per-pixel total to normalise by, or `nothing` to normalise
# by the accumulated total.
#
# Inputs are validated up front rather than inside the loop, so errors surface
# unwrapped and the threaded sweep carries no per-cell checks. That sweep runs
# `x` innermost (contiguous in memory) and splits rows across tasks, which is
# what keeps the three accumulators race-free.
function _phase_map(weight_of, n::AbstractArray, T::AbstractArray, nlos; TCNM, TWNM)
    ndims(n) == 3 || throw(DimensionMismatch("n and T must be 3D cubes."))
    _check_phase_bounds(n, T, TCNM, TWNM)

    nx, ny = size(n, 1), size(n, 2)
    mapCNM = zeros(nx, ny)
    mapLNM = zeros(nx, ny)
    mapWNM = zeros(nx, ny)

    @sync for rows in _chunk_ranges(ny, Threads.nthreads())
        Threads.@spawn @inbounds for y in rows, k in axes(n, 3), x in axes(n, 1)
            Ti = T[x, y, k]
            w = weight_of(n[x, y, k])
            if Ti < TCNM
                mapCNM[x, y] += w
            elseif Ti < TWNM
                mapLNM[x, y] += w
            else
                mapWNM[x, y] += w
            end
        end
    end

    @inbounds for i in eachindex(mapCNM, mapLNM, mapWNM)
        total = nlos === nothing ? mapCNM[i] + mapLNM[i] + mapWNM[i] : nlos
        if total > 0
            scale = 100 / total
            mapCNM[i] *= scale
            mapLNM[i] *= scale
            mapWNM[i] *= scale
        else
            mapCNM[i] = mapLNM[i] = mapWNM[i] = 0.0
        end
    end
    return mapCNM, mapLNM, mapWNM
end

"""
    GasFractionMap(n, T; TCNM = 200, TWNM = 2000) -> (fCNM, fLNM, fWNM)

Plane-of-sky maps of the mass fraction of each phase, computed independently for
every line of sight (axis 3).
"""
function GasFractionMap(n::AbstractArray, T::AbstractArray; TCNM::Real = 200, TWNM::Real = 2000)
    _check_phase_bounds(n, T, TCNM, TWNM)
    @inbounds for i in eachindex(n, T)
        _check_mass_cell(n[i], T[i])
    end
    return _phase_map(identity, n, T, nothing; TCNM = TCNM, TWNM = TWNM)
end

"""
    VolumeFractionMap(n, T; TCNM = 200, TWNM = 2000) -> (fCNM, fLNM, fWNM)

Plane-of-sky maps of the volume filling fraction of each phase.
"""
function VolumeFractionMap(n::AbstractArray, T::AbstractArray; TCNM::Real = 200, TWNM::Real = 2000)
    _check_phase_bounds(n, T, TCNM, TWNM)
    all(isfinite, T) || throw(ArgumentError("T must contain only finite values."))
    return _phase_map(Returns(1.0), n, T, size(n, 3); TCNM = TCNM, TWNM = TWNM)
end
