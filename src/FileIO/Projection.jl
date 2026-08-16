"""
Viewing geometry: looking at the box from an arbitrary direction.

The three box axes are the only lines of sight that need no work — the cube is
permuted and one velocity component is already the LOS component. Any other
direction requires resampling the box onto a grid aligned with it, and
projecting the velocity *vector* onto the new line of sight.

A direction is given as two angles in degrees: `theta` from the +z axis and
`phi` from the +x axis in the xy plane, so

    n̂ = (sinθ cosφ, sinθ sinφ, cosθ)

is the line of sight and `(ê1, ê2, n̂)` is a right-handed orthonormal frame whose
first two vectors span the plane of the sky. The output cube keeps the input's
shape and is sampled trilinearly, rotating about the box centre.

Boundaries
----------
MHD boxes are usually periodic, and `periodic = true` (the default) samples them
that way, so a rotated cube is filled everywhere instead of having empty corners.
For a non-periodic box pass `periodic = false`, which replicates the edge cells
instead — the corners are then extrapolated and should not be trusted.

Rotations by exact multiples of 90° land back on grid points, so they reproduce
the original values permuted, with no interpolation smoothing.
"""

"""
    los_vectors(theta_deg, phi_deg) -> (e1, e2, nhat)

Right-handed orthonormal frame for the viewing direction `(theta_deg,
phi_deg)`, with `nhat` the line of sight and `e1`, `e2` spanning the sky plane.
"""
function los_vectors(theta_deg::Real, phi_deg::Real)
    isfinite(theta_deg) && isfinite(phi_deg) ||
        throw(ArgumentError("viewing angles must be finite (got $theta_deg, $phi_deg)."))
    st, ct = sincos(deg2rad(float(theta_deg)))
    sp, cp = sincos(deg2rad(float(phi_deg)))
    nhat = (st * cp, st * sp, ct)
    e1 = (ct * cp, ct * sp, -st)
    e2 = (-sp, cp, 0.0)
    return e1, e2, nhat
end

# Index of a (possibly out-of-range) grid coordinate under the chosen boundary.
@inline _wrap(i::Int, n::Int) = mod(i - 1, n) + 1
@inline _edge(i::Int, n::Int) = clamp(i, 1, n)

@inline function _trilinear(f, x, y, z, periodic::Bool)
    nx, ny, nz = size(f)
    x0 = floor(Int, x); y0 = floor(Int, y); z0 = floor(Int, z)
    tx = x - x0; ty = y - y0; tz = z - z0

    bound = periodic ? _wrap : _edge
    i0 = bound(x0, nx); i1 = bound(x0 + 1, nx)
    j0 = bound(y0, ny); j1 = bound(y0 + 1, ny)
    k0 = bound(z0, nz); k1 = bound(z0 + 1, nz)

    @inbounds begin
        c00 = f[i0, j0, k0] * (1 - tx) + f[i1, j0, k0] * tx
        c10 = f[i0, j1, k0] * (1 - tx) + f[i1, j1, k0] * tx
        c01 = f[i0, j0, k1] * (1 - tx) + f[i1, j0, k1] * tx
        c11 = f[i0, j1, k1] * (1 - tx) + f[i1, j1, k1] * tx
    end
    c0 = c00 * (1 - ty) + c10 * ty
    c1 = c01 * (1 - ty) + c11 * ty
    return c0 * (1 - tz) + c1 * tz
end

# Position in the original frame of the output voxel (i, j, k), whose offsets
# from the box centre are (a, b, c) along (e1, e2, nhat).
@inline function _source_point(i, j, k, ctr, e1, e2, nhat)
    a = i - ctr[1]; b = j - ctr[2]; c = k - ctr[3]
    return (ctr[1] + a * e1[1] + b * e2[1] + c * nhat[1],
            ctr[2] + a * e1[2] + b * e2[2] + c * nhat[2],
            ctr[3] + a * e1[3] + b * e2[3] + c * nhat[3])
end

_box_centre(f) = ((size(f, 1) + 1) / 2, (size(f, 2) + 1) / 2, (size(f, 3) + 1) / 2)

# The single resampling loop every projection goes through: `sample(x, y, z)`
# says what to read at the source point, the geometry is decided here and here
# only. Scalars and the velocity vector therefore cannot drift apart — they are
# evaluated at the same points by construction, not by two loops agreeing.
function _resample(sample, dims, ctr, e1, e2, nhat)
    nx, ny, nz = dims
    out = Array{Float64,3}(undef, nx, ny, nz)
    Threads.@threads for k in 1:nz
        for j in 1:ny, i in 1:nx
            x, y, z = _source_point(i, j, k, ctr, e1, e2, nhat)
            @inbounds out[i, j, k] = sample(x, y, z)
        end
    end
    return out
end

"""
    project_cube(field, e1, e2, nhat; periodic = true) -> Array{Float64,3}

Resample a scalar `field` onto a grid whose third axis is `nhat`, keeping the
input's shape and rotating about the box centre.
"""
project_cube(field::AbstractArray{<:Real,3}, e1, e2, nhat; periodic::Bool = true) =
    _resample((x, y, z) -> _trilinear(field, x, y, z, periodic),
              size(field), _box_centre(field), e1, e2, nhat)

"""
    project_los_velocity(Vx, Vy, Vz, e1, e2, nhat; periodic = true) -> Array{Float64,3}

Resample the velocity *vector* field onto the rotated grid and return its
component along the line of sight, `V · nhat`.

The vector is read at the same source points as a scalar would be — the three
components are sampled together and contracted with `nhat` on the spot, so this
is exactly `nhat . (project_cube(Vx), project_cube(Vy), project_cube(Vz))` but
in one pass, allocating one output cube instead of three.
"""
function project_los_velocity(Vx::AbstractArray{<:Real,3}, Vy::AbstractArray{<:Real,3},
                              Vz::AbstractArray{<:Real,3}, e1, e2, nhat; periodic::Bool = true)
    size(Vx) == size(Vy) == size(Vz) ||
        throw(DimensionMismatch("the three velocity components must share the same shape."))
    return _resample(size(Vx), _box_centre(Vx), e1, e2, nhat) do x, y, z
        nhat[1] * _trilinear(Vx, x, y, z, periodic) +
        nhat[2] * _trilinear(Vy, x, y, z, periodic) +
        nhat[3] * _trilinear(Vz, x, y, z, periodic)
    end
end

# Angles are formatted for use in a directory name: no dot, no minus sign.
function _fmt_angle(a::Real)
    r = round(float(a); digits = 1)
    s = isinteger(r) ? string(Int(r)) : replace(string(r), "." => "p")
    return replace(s, "-" => "m")
end

"""
    los_label(LOS) -> String

Directory-safe name for a line of sight: the axis itself for `"x"`, `"y"`, `"z"`,
and `"th<θ>_ph<φ>"` for a `(theta, phi)` pair — e.g. `(45, -30)` becomes
`"th45_phm30"`.
"""
los_label(LOS::AbstractString) = String(LOS)
los_label(LOS::Tuple{<:Real,<:Real}) =
    string("th", _fmt_angle(LOS[1]), "_ph", _fmt_angle(LOS[2]))

"""
    los_metadata(LOS) -> Dict{String,Any}

FITS keywords describing the viewing geometry a product was made with, so a
rotated map records the direction it was observed from.
"""
los_metadata(LOS::AbstractString) = Dict{String,Any}("LOSAXIS" => String(LOS))
los_metadata(LOS::Tuple{<:Real,<:Real}) =
    Dict{String,Any}("LOSTHETA" => float(LOS[1]), "LOSPHI" => float(LOS[2]))
