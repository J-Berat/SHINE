"""
Read the physical fields a 21-cm synthetic observation needs — number density,
kinetic temperature and the line-of-sight velocity — from a simulation
directory, and orient the cubes so the line of sight is the third axis.

Supported inputs
----------------
- One FITS file per field, named after the field (or a common alias), e.g.
  `density.fits`, `temperature.fits`, `Vx.fits` / `Vy.fits` / `Vz.fits`.
- A single HDF5 file (`*.h5` / `*.hdf5`) holding datasets with those names.

Field aliases (case-insensitive) let the reader accept the naming conventions of
several simulation codes:

| canonical    | accepted stems / datasets                     |
|--------------|-----------------------------------------------|
| density      | density, n, nHI, nh, rho                      |
| temperature  | temperature, temp, T                          |
| Vx / Vy / Vz | Vx/velocity_x/vx, Vy/…, Vz/…                   |
"""

using FITSIO: FITS
using HDF5: h5open, ishdf5

const _HDF5_EXTS = (".h5", ".hdf5", ".he5")

const _FIELD_ALIASES = Dict(
    "density"     => ("density", "n", "nhi", "nh", "rho"),
    "temperature" => ("temperature", "temp", "t"),
    "Vx"          => ("vx", "velocity_x", "velx", "v_x"),
    "Vy"          => ("vy", "velocity_y", "vely", "v_y"),
    "Vz"          => ("vz", "velocity_z", "velz", "v_z"),
)

_los_velocity_field(LOS) = LOS == "x" ? "Vx" : LOS == "y" ? "Vy" : "Vz"

# Permute a natively (x, y, z) cube so the requested LOS lands on axis 3.
function _orient_los(cube, LOS)
    LOS == "z" && return cube
    LOS == "x" && return permutedims(cube, (2, 3, 1))   # -> (y, z, x)
    LOS == "y" && return permutedims(cube, (3, 1, 2))   # -> (z, x, y)
    error("Unknown LOS: $(LOS) (expected \"x\", \"y\" or \"z\").")
end

_is_hdf5_name(name) = lowercase(splitext(name)[2]) in _HDF5_EXTS

# Locate the single HDF5 container in `simu`, if any.
function _hdf5_container(simu)
    for name in sort(readdir(simu))
        path = joinpath(simu, name)
        isfile(path) && _is_hdf5_name(name) && ishdf5(path) && return path
    end
    return nothing
end

function _read_hdf5_field(file, field)
    aliases = _FIELD_ALIASES[field]
    return h5open(file, "r") do h5
        for key in keys(h5)
            lowercase(key) in aliases && return Float64.(read(h5[key]))
        end
        return nothing
    end
end

function _read_fits_field(simu, field)
    aliases = _FIELD_ALIASES[field]
    for name in sort(readdir(simu))
        _is_fits(name) || continue
        stem = lowercase(splitext(name)[1])
        if stem in aliases
            return FITS(joinpath(simu, name), "r") do f
                Float64.(read(f[1]))
            end
        end
    end
    return nothing
end

"""
    read_field(simu, field) -> Array{Float64,3}

Read one canonical `field` ("density", "temperature", "Vx", …) from `simu`,
trying a per-field FITS file first and then an HDF5 container. Errors if the
field cannot be found.
"""
function read_field(simu, field)
    haskey(_FIELD_ALIASES, field) || error("Unknown field: $(field).")

    data = _read_fits_field(simu, field)
    if data === nothing
        container = _hdf5_container(simu)
        container !== nothing && (data = _read_hdf5_field(container, field))
    end
    data === nothing && error("Could not find field `$(field)` in $(simu). " *
                              "Expected a FITS file or HDF5 dataset named one of: " *
                              join(_FIELD_ALIASES[field], ", ") * ".")
    ndims(data) == 3 || error("Field `$(field)` in $(simu) is $(ndims(data))D; a 3D cube is required.")
    return data
end

# Shared sanity checks on the raw fields, before any orientation is applied.
function _validate_fields(n, T, V...)
    for v in V
        size(n) == size(T) == size(v) || throw(DimensionMismatch(
            "simulation fields must have the same shape: density=$(size(n)), " *
            "temperature=$(size(T)), velocity=$(size(v))."))
        all(isfinite, v) || throw(ArgumentError("velocity contains NaN or Inf values."))
    end
    all(isfinite, n) || throw(ArgumentError("density contains NaN or Inf values."))
    all(isfinite, T) || throw(ArgumentError("temperature contains NaN or Inf values."))
    all(>=(0), n) || throw(ArgumentError("density contains negative values."))
    all(>(0), T) || throw(ArgumentError("temperature must be strictly positive in every cell."))
    return nothing
end

"""
    ReadSimulation(simu, LOS, conversionn, conversionT, conversionV; periodic = true)
        -> (n, VLOS, T)

Read the density, LOS velocity and temperature cubes for line of sight `LOS`,
apply the unit-conversion factors, and orient every cube so that the line of
sight is the third axis.

`LOS` is either a box axis — `"x"`, `"y"` or `"z"`, handled by a permutation
with no resampling — or a `(theta, phi)` pair of angles in degrees, which
rotates the box onto that viewing direction (see [`project_cube`](@ref)).
`periodic` selects the boundary used by the rotation and is ignored for an axis.

Returned units after conversion: `n` in cm^-3, `VLOS` in km/s, `T` in K.
"""
function ReadSimulation(simu, LOS::AbstractString, conversionn::Real, conversionT::Real,
                        conversionV::Real; periodic::Bool = true)
    LOS in ("x", "y", "z") ||
        error("Invalid LOS `$(LOS)`; expected \"x\", \"y\", \"z\" or a (theta, phi) tuple.")
    all(isfinite, (conversionn, conversionT, conversionV)) ||
        throw(ArgumentError("unit-conversion factors must be finite."))

    n = read_field(simu, "density") .* conversionn
    T = read_field(simu, "temperature") .* conversionT
    V = read_field(simu, _los_velocity_field(LOS)) .* conversionV

    _validate_fields(n, T, V)
    return _orient_los(n, LOS), _orient_los(V, LOS), _orient_los(T, LOS)
end

"""
    ReadSimulation(simu, (theta, phi), conversionn, conversionT, conversionV;
                   periodic = true) -> (n, VLOS, T)

Rotated variant: resample the box so that the viewing direction `(theta, phi)`
in degrees becomes the third axis, and project the velocity vector onto it.

Memory note: this reads five cubes (density, temperature and all three velocity
components) where the axis path reads three, and allocates three more for the
resampled output.
"""
function ReadSimulation(simu, LOS::Tuple{<:Real,<:Real}, conversionn::Real, conversionT::Real,
                        conversionV::Real; periodic::Bool = true)
    all(isfinite, (conversionn, conversionT, conversionV)) ||
        throw(ArgumentError("unit-conversion factors must be finite."))
    e1, e2, nhat = los_vectors(LOS[1], LOS[2])

    n = read_field(simu, "density") .* conversionn
    T = read_field(simu, "temperature") .* conversionT
    Vx = read_field(simu, "Vx") .* conversionV
    Vy = read_field(simu, "Vy") .* conversionV
    Vz = read_field(simu, "Vz") .* conversionV

    _validate_fields(n, T, Vx, Vy, Vz)
    # Wrapping across a boundary only makes sense between faces of equal size,
    # and off-axis views of a non-cubic box mix axes of different length.
    allequal(size(n)) || warn_user(
        "Box is $(join(size(n), "x")) — a rotated view of a non-cubic box mixes axes " *
        "of different lengths; interpret the resampled cube with care.")

    VLOS = project_los_velocity(Vx, Vy, Vz, e1, e2, nhat; periodic = periodic)
    Vx = Vy = Vz = nothing                  # release before the scalar passes
    return (project_cube(n, e1, e2, nhat; periodic = periodic),
            VLOS,
            project_cube(T, e1, e2, nhat; periodic = periodic))
end
