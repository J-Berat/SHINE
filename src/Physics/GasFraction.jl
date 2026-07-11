"""
Mass- and volume-weighted fractions of the CNM/LNM/WNM phases, both as scalar
diagnostics for a whole cube and as plane-of-sky maps computed along the line of
sight.
"""

"""
    GasFraction(n, T; TCNM = 200, TWNM = 2000) -> (fCNM, fLNM, fWNM)

Mass fraction (in percent) of each phase over the supplied density/temperature
arrays. `n` is used as the mass weight.
"""
function GasFraction(n::AbstractArray, T::AbstractArray; TCNM::Real = 200, TWNM::Real = 2000)
    total_mass = sum(n)
    total_mass > 0 || return (0.0, 0.0, 0.0)

    maskCNM = T .< TCNM
    maskLNM = (T .>= TCNM) .& (T .< TWNM)
    maskWNM = T .>= TWNM

    fCNM = sum(n[maskCNM]) / total_mass * 100
    fLNM = sum(n[maskLNM]) / total_mass * 100
    fWNM = sum(n[maskWNM]) / total_mass * 100

    return fCNM, fLNM, fWNM
end

"""
    VolumeFraction(n, T; TCNM = 200, TWNM = 2000) -> (fCNM, fLNM, fWNM)

Volume filling fraction (in percent) of each phase, i.e. the fraction of cells in
each temperature bin (density-independent).
"""
function VolumeFraction(n::AbstractArray, T::AbstractArray; TCNM::Real = 200, TWNM::Real = 2000)
    total = length(T)
    total > 0 || return (0.0, 0.0, 0.0)

    nCNM = count(<(TCNM), T)
    nWNM = count(>=(TWNM), T)
    nLNM = total - nCNM - nWNM

    return nCNM / total * 100, nLNM / total * 100, nWNM / total * 100
end

# Shared machinery for the per-pixel maps: `f` returns a 3-tuple for one LOS.
function _phase_map(f, n::AbstractArray, T::AbstractArray; TCNM, TWNM)
    nx, ny = size(n, 1), size(n, 2)
    mapCNM = zeros(nx, ny)
    mapLNM = zeros(nx, ny)
    mapWNM = zeros(nx, ny)

    Threads.@threads for i in 1:nx
        for j in 1:ny
            c, l, w = f(@view(n[i, j, :]), @view(T[i, j, :]); TCNM = TCNM, TWNM = TWNM)
            mapCNM[i, j] = c
            mapLNM[i, j] = l
            mapWNM[i, j] = w
        end
    end

    return mapCNM, mapLNM, mapWNM
end

"""
    GasFractionMap(n, T; TCNM = 200, TWNM = 2000) -> (fCNM, fLNM, fWNM)

Plane-of-sky maps of the mass fraction of each phase, computed independently for
every line of sight (axis 3).
"""
GasFractionMap(n::AbstractArray, T::AbstractArray; TCNM::Real = 200, TWNM::Real = 2000) =
    _phase_map(GasFraction, n, T; TCNM = TCNM, TWNM = TWNM)

"""
    VolumeFractionMap(n, T; TCNM = 200, TWNM = 2000) -> (fCNM, fLNM, fWNM)

Plane-of-sky maps of the volume filling fraction of each phase.
"""
VolumeFractionMap(n::AbstractArray, T::AbstractArray; TCNM::Real = 200, TWNM::Real = 2000) =
    _phase_map(VolumeFraction, n, T; TCNM = TCNM, TWNM = TWNM)
