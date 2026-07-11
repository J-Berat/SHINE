"""
Spectral (velocity) moments of a `T_B(v)` cube — the standard first diagnostics
of any 21-cm data set. All routines expect the velocity axis on dimension 3.

- moment 0 : velocity-integrated brightness ∝ column density  [K km/s]
- moment 1 : intensity-weighted mean velocity (centroid)       [km/s]
- moment 2 : intensity-weighted velocity dispersion            [km/s]
"""

"""
    moment0(Tb, velArray) -> Matrix

Zeroth moment: `∫ T_B dv` [K km/s]. Proportional to the HI column density in the
optically thin limit.
"""
function moment0(Tb, velArray)
    _, widths = _moment_inputs(Tb, velArray)
    weights = reshape(widths, 1, 1, :)
    return dropdims(sum(Tb .* weights; dims = 3); dims = 3)
end

"""
    moment1(Tb, velArray) -> Matrix

First moment (velocity centroid) `∫ v T_B dv / ∫ T_B dv` [km/s]. Channels are
clamped to be non-negative before weighting to avoid unphysical centroids from
noise troughs.
"""
function moment1(Tb, velArray)
    vel, widths = _moment_inputs(Tb, velArray)
    v = reshape(vel, 1, 1, :)
    w = max.(Tb, 0) .* reshape(widths, 1, 1, :)
    num = dropdims(sum(w .* v; dims = 3); dims = 3)
    den = dropdims(sum(w; dims = 3); dims = 3)
    return num ./ _safe(den)
end

"""
    moment2(Tb, velArray) -> Matrix

Second moment (velocity dispersion) `sqrt(∫ (v - v̄)² T_B dv / ∫ T_B dv)` [km/s].
"""
function moment2(Tb, velArray)
    vel, widths = _moment_inputs(Tb, velArray)
    v = reshape(vel, 1, 1, :)
    w = max.(Tb, 0) .* reshape(widths, 1, 1, :)
    den = dropdims(sum(w; dims = 3); dims = 3)
    vbar = dropdims(sum(w .* v; dims = 3); dims = 3) ./ _safe(den)
    var = dropdims(sum(w .* (v .- reshape(vbar, size(vbar)..., 1)) .^ 2; dims = 3); dims = 3) ./ _safe(den)
    return sqrt.(max.(var, 0))
end

function _moment_inputs(Tb, velArray)
    ndims(Tb) == 3 || throw(DimensionMismatch("Tb must be a 3D (x, y, velocity) cube."))
    vel = collect(float.(velArray))
    length(vel) == size(Tb, 3) ||
        throw(DimensionMismatch("velocity axis has $(length(vel)) channels, but Tb has $(size(Tb, 3))."))
    isempty(vel) && throw(ArgumentError("velocity axis must not be empty."))
    all(isfinite, vel) || throw(ArgumentError("velocity axis contains NaN or Inf."))

    if length(vel) == 1
        return vel, [1.0]
    end
    delta = diff(vel)
    (all(>(0), delta) || all(<(0), delta)) ||
        throw(ArgumentError("velocity channels must be strictly monotonic."))
    delta = abs.(delta)
    widths = Vector{Float64}(undef, length(vel))
    widths[1] = delta[1]
    widths[end] = delta[end]
    @views widths[2:end-1] .= (delta[1:end-1] .+ delta[2:end]) ./ 2
    return vel, widths
end

_safe(x) = map(v -> v == 0 ? oftype(v, NaN) : v, x)
