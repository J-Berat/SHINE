"""
FITS writers for the HI data products.

Each product is written atomically (through a temporary sibling) with a header
carrying the physical unit (`BUNIT`) and, for cubes, a fully described velocity
axis (`CTYPE3 = VELO`, `CRVAL3`, `CDELT3`, `CUNIT3 = km/s`). Extra provenance
keywords can be injected through the `metadata` keyword.
"""

using FITSIO: FITS, FITSHeader

# Physical unit of each known product; anything not listed is written unitless.
const _BUNIT = Dict(
    "NHI" => "cm-2", "NCNM" => "cm-2", "NLNM" => "cm-2", "NWNM" => "cm-2",
    "TbHI" => "K", "TbCNM" => "K", "TbLNM" => "K", "TbWNM" => "K",
    "TbthinHI" => "K", "TbthinCNM" => "K", "TbthinLNM" => "K", "TbthinWNM" => "K",
    "tauHI" => "", "tauCNM" => "", "tauLNM" => "", "tauWNM" => "",
    "mom0" => "K.km/s", "mom1" => "km/s", "mom2" => "km/s",
    "QHI" => "K.km/s", "UHI" => "K.km/s", "PHI" => "K.km/s", "polangle" => "rad",
    "fCNMmass" => "percent", "fLNMmass" => "percent", "fWNMmass" => "percent",
    "fCNMvol" => "percent", "fLNMvol" => "percent", "fWNMvol" => "percent",
    "fftcnm" => "",
)

bunit_of(name) = get(_BUNIT, name, "")

function _base_header(name, ndim; metadata = nothing)
    keys = ["BUNIT", "ORIGIN", "PRODUCT"]
    vals = Any[bunit_of(name), "SHINE", name]
    coms = ["Physical unit", "Synthetic HI Neutral Emission", "Data product name"]

    for i in 1:ndim
        push!(keys, "CTYPE$i"); push!(vals, i == 3 ? "VELO" : "PIX")
        push!(coms, i == 3 ? "Velocity axis" : "Pixel axis")
    end

    if metadata isa AbstractDict
        for (k, v) in metadata
            push!(keys, uppercase(String(k))[1:min(8, end)])
            push!(vals, v isa AbstractString ? v : v)
            push!(coms, "")
        end
    end

    return keys, vals, coms
end

"""
    WriteData2D(resultspath, data, DataName; metadata = nothing, filename = nothing)

Write a 2D map to `resultspath/DataName.fits`.
"""
function WriteData2D(resultspath::AbstractString, data::AbstractArray, DataName::AbstractString;
                     metadata = nothing, filename = nothing)
    mkpath(resultspath)
    keys, vals, coms = _base_header(DataName, 2; metadata = metadata)
    header = FITSHeader(keys, vals, coms)

    fits_path = joinpath(resultspath, filename === nothing ? "$DataName.fits" : String(filename))
    atomic_write_path(fits_path) do tmp
        FITS(tmp, "w") do f
            write(f, Array(data); header = header)
        end
    end
    @info "Wrote FITS map" product = DataName path = fits_path
    return fits_path
end

"""
    WriteData3D(resultspath, data, DataName, velArray; metadata = nothing, filename = nothing)

Write a 3D `(nx, ny, nv)` cube to `resultspath/DataName.fits`, describing the
velocity axis from `velArray` [km/s].
"""
function WriteData3D(resultspath::AbstractString, data::AbstractArray, DataName::AbstractString,
                     velArray; metadata = nothing, filename = nothing)
    mkpath(resultspath)
    keys, vals, coms = _base_header(DataName, 3; metadata = metadata)

    v0 = length(velArray) >= 1 ? float(first(velArray)) : 0.0
    dv = length(velArray) >= 2 ? float(velArray[2] - velArray[1]) : 1.0
    append!(keys, ["CRPIX3", "CRVAL3", "CDELT3", "CUNIT3"])
    append!(vals, Any[1.0, v0, dv, "km/s"])
    append!(coms, ["Reference pixel (velocity)", "Velocity at CRPIX3", "Channel width", "Velocity unit"])
    header = FITSHeader(keys, vals, coms)

    fits_path = joinpath(resultspath, filename === nothing ? "$DataName.fits" : String(filename))
    atomic_write_path(fits_path) do tmp
        FITS(tmp, "w") do f
            write(f, Array(data); header = header)
        end
    end
    @info "Wrote FITS cube" product = DataName path = fits_path
    return fits_path
end
