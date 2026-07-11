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
    dv = _channel_width(velArray)
    return dropdims(sum(Tb; dims = 3); dims = 3) .* dv
end

"""
    moment1(Tb, velArray) -> Matrix

First moment (velocity centroid) `∫ v T_B dv / ∫ T_B dv` [km/s]. Channels are
clamped to be non-negative before weighting to avoid unphysical centroids from
noise troughs.
"""
function moment1(Tb, velArray)
    v = reshape(collect(float.(velArray)), 1, 1, :)
    w = max.(Tb, 0)
    num = dropdims(sum(w .* v; dims = 3); dims = 3)
    den = dropdims(sum(w; dims = 3); dims = 3)
    return num ./ _safe(den)
end

"""
    moment2(Tb, velArray) -> Matrix

Second moment (velocity dispersion) `sqrt(∫ (v - v̄)² T_B dv / ∫ T_B dv)` [km/s].
"""
function moment2(Tb, velArray)
    v = reshape(collect(float.(velArray)), 1, 1, :)
    w = max.(Tb, 0)
    den = dropdims(sum(w; dims = 3); dims = 3)
    vbar = dropdims(sum(w .* v; dims = 3); dims = 3) ./ _safe(den)
    var = dropdims(sum(w .* (v .- reshape(vbar, size(vbar)..., 1)) .^ 2; dims = 3); dims = 3) ./ _safe(den)
    return sqrt.(max.(var, 0))
end

_channel_width(velArray) = length(velArray) > 1 ? abs(float(velArray[2] - velArray[1])) : 1.0
_safe(x) = map(v -> v == 0 ? oftype(v, NaN) : v, x)
