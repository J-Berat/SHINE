"""
Spectral (velocity) moments of a `T_B(v)` cube — the standard first diagnostics
of any 21-cm data set. All routines expect the velocity axis on dimension 3.

- moment 0 : velocity-integrated brightness ∝ column density  [K km/s]
- moment 1 : intensity-weighted mean velocity (centroid)       [km/s]
- moment 2 : intensity-weighted velocity dispersion            [km/s]

The accumulators sweep the cube channel by channel, so the innermost loop runs
along `x` (contiguous in memory) and no cube-sized temporary is ever built.
"""

"""
    moment0(Tb, velArray) -> Matrix

Zeroth moment: `∫ T_B dv` [K km/s]. Proportional to the HI column density in the
optically thin limit.
"""
function moment0(Tb, velArray)
    _, widths = _moment_inputs(Tb, velArray)
    out = zeros(size(Tb, 1), size(Tb, 2))
    @inbounds for k in axes(Tb, 3)
        w = widths[k]
        for y in axes(Tb, 2), x in axes(Tb, 1)
            out[x, y] += Tb[x, y, k] * w
        end
    end
    return out
end

"""
    moment1(Tb, velArray) -> Matrix

First moment (velocity centroid) `∫ v T_B dv / ∫ T_B dv` [km/s]. Channels are
clamped to be non-negative before weighting to avoid unphysical centroids from
noise troughs.
"""
function moment1(Tb, velArray)
    vel, widths = _moment_inputs(Tb, velArray)
    weight, weighted_v = _weight_sums(Tb, vel, widths)
    return _safe_divide!(weighted_v, weighted_v, weight)
end

"""
    moment2(Tb, velArray) -> Matrix

Second moment (velocity dispersion) `sqrt(∫ (v - v̄)² T_B dv / ∫ T_B dv)` [km/s].
"""
function moment2(Tb, velArray)
    vel, widths = _moment_inputs(Tb, velArray)
    weight, weighted_v = _weight_sums(Tb, vel, widths)
    vbar = _safe_divide!(weighted_v, weighted_v, weight)
    return _dispersion(Tb, vel, widths, weight, vbar)
end

"""
    moments012(Tb, velArray) -> (mom0, mom1, mom2)

All three velocity moments from two sweeps of the cube instead of the five that
computing them separately would cost. The dispersion keeps the two-pass
`⟨(v - v̄)²⟩` form rather than `⟨v²⟩ - v̄²`, which loses precision for narrow
lines far from zero velocity.
"""
function moments012(Tb, velArray)
    vel, widths = _moment_inputs(Tb, velArray)

    mom0 = zeros(size(Tb, 1), size(Tb, 2))
    weight = zeros(size(Tb, 1), size(Tb, 2))
    weighted_v = zeros(size(Tb, 1), size(Tb, 2))
    @inbounds for k in axes(Tb, 3)
        w, v = widths[k], vel[k]
        for y in axes(Tb, 2), x in axes(Tb, 1)
            t = Tb[x, y, k]
            mom0[x, y] += t * w
            wt = (t > 0 ? t : zero(t)) * w
            weight[x, y] += wt
            weighted_v[x, y] += wt * v
        end
    end

    vbar = _safe_divide!(weighted_v, weighted_v, weight)
    return mom0, vbar, _dispersion(Tb, vel, widths, weight, vbar)
end

# Non-negative weight sums: ∫ max(T_B, 0) dv and ∫ v max(T_B, 0) dv.
function _weight_sums(Tb, vel, widths)
    weight = zeros(size(Tb, 1), size(Tb, 2))
    weighted_v = zeros(size(Tb, 1), size(Tb, 2))
    @inbounds for k in axes(Tb, 3)
        w, v = widths[k], vel[k]
        for y in axes(Tb, 2), x in axes(Tb, 1)
            t = Tb[x, y, k]
            wt = (t > 0 ? t : zero(t)) * w
            weight[x, y] += wt
            weighted_v[x, y] += wt * v
        end
    end
    return weight, weighted_v
end

# Second pass: sqrt(∫ (v - v̄)² max(T_B, 0) dv / ∫ max(T_B, 0) dv).
function _dispersion(Tb, vel, widths, weight, vbar)
    var = zeros(size(Tb, 1), size(Tb, 2))
    @inbounds for k in axes(Tb, 3)
        w, v = widths[k], vel[k]
        for y in axes(Tb, 2), x in axes(Tb, 1)
            t = Tb[x, y, k]
            d = v - vbar[x, y]
            var[x, y] += (t > 0 ? t : zero(t)) * w * d * d
        end
    end
    @inbounds for i in eachindex(var, weight)
        q = weight[i] == 0 ? NaN : var[i] / weight[i]
        var[i] = sqrt(max(q, 0))
    end
    return var
end

function _moment_inputs(Tb, velArray)
    ndims(Tb) == 3 || throw(DimensionMismatch("Tb must be a 3D (x, y, velocity) cube."))
    vel = _velocity_channels(velArray)
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

# Elementwise division yielding NaN where the weight vanishes: a spectrum with
# no positive signal has no defined centroid. May alias `out` with `num`.
function _safe_divide!(out, num, den)
    @inbounds for i in eachindex(out, num, den)
        d = den[i]
        out[i] = d == 0 ? NaN : num[i] / d
    end
    return out
end
