"""
Angular smoothing of the sky-plane maps, emulating a single-dish beam.

`LowPass` convolves a 2D image with a Gaussian kernel; `smooth_cube!` applies it
to every velocity channel of a cube in place.

The beam can be specified two ways:

- **in pixels**, through `kernel_size_hi` (the Gaussian σ) — convenient when
  exploring resolution effects but detached from any real telescope;
- **physically**, through a full width at half maximum in arcminutes plus a
  source distance, which is how an actual instrument is quoted (Arecibo ≈ 3.5′,
  GBT ≈ 9′ at 21 cm). [`resolve_beam`](@ref) turns that pair into the σ in
  pixels the convolution needs.
"""

using ImageFiltering: imfilter, Kernel

# FWHM = 2 sqrt(2 ln 2) σ for a Gaussian.
const FWHM_TO_SIGMA = 1 / (2 * sqrt(2 * log(2)))

"""
    LowPass(img, kernel) -> Matrix

Convolve a 2D image with `kernel` (e.g. `gaussian_beam(σ)`), replicating edge
pixels so the map keeps its shape.
"""
LowPass(img::AbstractMatrix, kernel) = imfilter(img, kernel, "replicate")

"""
    gaussian_beam(sigma_pix) -> kernel

Isotropic Gaussian smoothing kernel of standard deviation `sigma_pix` pixels.
"""
gaussian_beam(sigma_pix::Real) = Kernel.gaussian(sigma_pix)

"""
    pixel_scale_arcmin(dx_pc, distance_pc) -> Float64

Angular size [arcmin] of a sky-plane pixel of physical size `dx_pc` parsecs seen
from `distance_pc` parsecs.

The pipeline is plane-parallel — the line of sight is a box axis and no
projection is applied across the map — so the small-angle relation θ = dx / D is
used uniformly rather than the exact angle subtended by each individual pixel.
"""
function pixel_scale_arcmin(dx_pc::Real, distance_pc::Real)
    dx_pc > 0 || throw(ArgumentError("dx_pc must be positive (got $dx_pc)."))
    distance_pc > 0 || throw(ArgumentError("distance_pc must be positive (got $distance_pc)."))
    return rad2deg(dx_pc / distance_pc) * 60
end

"""
    beam_sigma_pix(fwhm_arcmin, pixel_arcmin) -> Float64

Gaussian σ **in pixels** of a beam of full width at half maximum `fwhm_arcmin`
on a grid sampled at `pixel_arcmin` per pixel.
"""
function beam_sigma_pix(fwhm_arcmin::Real, pixel_arcmin::Real)
    fwhm_arcmin > 0 || throw(ArgumentError("fwhm_arcmin must be positive (got $fwhm_arcmin)."))
    pixel_arcmin > 0 || throw(ArgumentError("pixel_arcmin must be positive (got $pixel_arcmin)."))
    return FWHM_TO_SIGMA * fwhm_arcmin / pixel_arcmin
end

"""
    resolve_beam(PixelLength_cm; kernel_size_hi = 2.0, beam_fwhm_arcmin = 0.0,
                 distance_pc = 0.0)
        -> (; sigma_pix, fwhm_arcmin, pixel_arcmin, distance_pc)

Work out the beam actually applied by the pipeline.

`beam_fwhm_arcmin > 0` selects the physical specification and requires
`distance_pc > 0`; otherwise the beam falls back to `kernel_size_hi` as a σ in
pixels. When a distance is known the physical counterparts are filled in either
way, so a pixel-specified beam still reports the FWHM it corresponds to — those
values are what ends up in the FITS headers.
"""
function resolve_beam(PixelLength_cm::Real; kernel_size_hi::Real = 2.0,
                      beam_fwhm_arcmin::Real = 0.0, distance_pc::Real = 0.0)
    PixelLength_cm > 0 || throw(ArgumentError("PixelLength_cm must be positive (got $PixelLength_cm)."))
    distance_pc >= 0 || throw(ArgumentError("distance_pc must be non-negative (got $distance_pc)."))

    dx_pc = PixelLength_cm / PC_TO_CM
    pixel_arcmin = distance_pc > 0 ? pixel_scale_arcmin(dx_pc, distance_pc) : nothing

    if beam_fwhm_arcmin > 0
        pixel_arcmin === nothing && throw(ArgumentError(
            "A beam FWHM of $(beam_fwhm_arcmin) arcmin needs a source distance: pass distance_pc > 0."))
        sigma_pix = beam_sigma_pix(beam_fwhm_arcmin, pixel_arcmin)
        fwhm_arcmin = float(beam_fwhm_arcmin)
    else
        kernel_size_hi > 0 ||
            throw(ArgumentError("kernel_size_hi must be positive (got $kernel_size_hi)."))
        sigma_pix = float(kernel_size_hi)
        fwhm_arcmin = pixel_arcmin === nothing ? nothing : sigma_pix / FWHM_TO_SIGMA * pixel_arcmin
    end

    # A beam narrower than the grid it is sampled on cannot be represented: warn
    # rather than silently returning a map that looks unsmoothed.
    sigma_pix >= 0.5 || warn_user(
        "Beam σ = $(round(sigma_pix, digits = 3)) pix is under half a pixel — the map " *
        "under-samples this beam and smoothing will barely change it.")

    return (; sigma_pix = sigma_pix, fwhm_arcmin = fwhm_arcmin,
             pixel_arcmin = pixel_arcmin, distance_pc = float(distance_pc))
end

"""
    beam_description(beam) -> String

One-line human-readable summary of a [`resolve_beam`](@ref) result.
"""
function beam_description(beam)
    msg = "σ = $(round(beam.sigma_pix, digits = 3)) pix"
    beam.fwhm_arcmin === nothing ||
        (msg *= ", FWHM = $(round(beam.fwhm_arcmin, digits = 3)) arcmin")
    beam.pixel_arcmin === nothing ||
        (msg *= ", pixel = $(round(beam.pixel_arcmin, digits = 4)) arcmin @ $(round(beam.distance_pc, digits = 1)) pc")
    return msg
end

"""
    beam_metadata(metadata, beam) -> Dict{String,Any}

Merge the beam description of `beam` into `metadata` as FITS keywords, so a
smoothed product carries the resolution it was produced at.

`BMAJ`/`BMIN`/`BPA` follow the FITS convention and are written in **degrees**;
they only appear when a distance makes the beam physical. `BMAJPIX` always does,
since a beam given in pixels still has a well-defined width on the grid.
"""
function beam_metadata(metadata, beam)
    extra = Dict{String,Any}("BMAJPIX" => beam.sigma_pix / FWHM_TO_SIGMA)
    if beam.pixel_arcmin !== nothing
        extra["PIXSCALE"] = beam.pixel_arcmin
        extra["DISTANCE"] = beam.distance_pc
        extra["BMAJ"] = beam.fwhm_arcmin / 60      # arcmin -> deg
        extra["BMIN"] = beam.fwhm_arcmin / 60      # circular beam
        extra["BPA"] = 0.0
    end
    return merge_metadata(metadata, extra)
end

"""
    smooth_cube!(cube, sigma_pix)

Smooth every velocity channel (axis 3) of `cube` in place with a Gaussian beam.
"""
function smooth_cube!(cube::AbstractArray{<:Real,3}, sigma_pix::Real)
    kernel = gaussian_beam(sigma_pix)
    @views for k in axes(cube, 3)
        cube[:, :, k] = LowPass(cube[:, :, k], kernel)
    end
    return cube
end
