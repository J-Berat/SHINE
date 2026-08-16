"""
Neutral-hydrogen phase separation by kinetic temperature.

The three-phase decomposition follows the thermal-instability picture of the
neutral ISM (Wolfire et al. 2003): the Cold Neutral Medium (CNM), the thermally
unstable Lukewarm Neutral Medium (LNM), and the Warm Neutral Medium (WNM). The
thresholds `TCNM` and `TWNM` are the temperature boundaries between them and are
run parameters rather than fixed constants.
"""

"""
    HIPhases(n, T; TCNM = 200, TWNM = 2000) -> (nCNM, nLNM, nWNM)

Split a density field into CNM (`T < TCNM`), LNM (`TCNM ≤ T < TWNM`) and WNM
(`T ≥ TWNM`) components. Works for arrays of any shape; the masks are applied
element-wise so `nCNM .+ nLNM .+ nWNM == n` exactly.
"""
function HIPhases(n, T; TCNM::Real = 200, TWNM::Real = 2000)
    axes(n) == axes(T) || throw(DimensionMismatch("n and T must share the same axes."))
    TCNM <= TWNM || throw(ArgumentError("TCNM ($TCNM) must be <= TWNM ($TWNM)."))

    nCNM = similar(n)
    nLNM = similar(n)
    nWNM = similar(n)
    # Validation and splitting share a single sweep; the three boolean masks the
    # broadcast form materialised are never built.
    zero_n = zero(eltype(nCNM))
    @inbounds for i in eachindex(n, T)
        ni, Ti = n[i], T[i]
        isfinite(ni) && isfinite(Ti) ||
            throw(ArgumentError("n and T must contain only finite values."))
        ni >= 0 || throw(ArgumentError("density must be non-negative."))
        Ti > 0 || throw(ArgumentError("temperature must be strictly positive."))

        cold = Ti < TCNM
        warm = Ti >= TWNM
        nCNM[i] = cold ? ni : zero_n
        nLNM[i] = (cold || warm) ? zero_n : ni
        nWNM[i] = warm ? ni : zero_n
    end

    return nCNM, nLNM, nWNM
end

"""
    HIColumnDensity(n, PixelLength_cm) -> Real

Column density of a single line of sight (1D density vector) [cm^-2].
"""
HIColumnDensity(n::AbstractVector, PixelLength_cm::Real) = sum(n) * PixelLength_cm

"""
    HIColumnDensity(n, nCNM, nLNM, nWNM, PixelLength_cm) -> (NHI, NCNM, NLNM, NWNM)

Integrate the total and per-phase density cubes along the line of sight (axis 3)
into column-density maps [cm^-2].
"""
function HIColumnDensity(n, nCNM, nLNM, nWNM, PixelLength_cm::Real)
    NHI = dropdims(sum(n; dims = 3); dims = 3) .* PixelLength_cm
    NCNM = dropdims(sum(nCNM; dims = 3); dims = 3) .* PixelLength_cm
    NLNM = dropdims(sum(nLNM; dims = 3); dims = 3) .* PixelLength_cm
    NWNM = dropdims(sum(nWNM; dims = 3); dims = 3) .* PixelLength_cm

    return NHI, NCNM, NLNM, NWNM
end
