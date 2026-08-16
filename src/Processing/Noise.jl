"""
Instrumental noise on the brightness-temperature cubes.

Where the noise enters matters. Receiver noise is added to the signal *before*
the telescope beam and the gridding that follow it, so in a finished map two
pixels closer than a beam width are not independent: the noise field carries the
beam's own correlation length. Drawing an independent Gaussian per pixel of a
smoothed map is therefore too optimistic — it under-states the noise on any
scale larger than a pixel, which is exactly the scale most analyses work at
(moment maps, column densities, power spectra).

[`add_noise!`](@ref) reproduces the right behaviour by drawing white noise,
passing it through the same responses as the signal — the beam across the sky,
the spectral response along velocity — and renormalising so the map still has
the requested standard deviation. With neither, the noise stays white, which is
the correct model for noise added after gridding.
"""

"""
    beam_noise_gain(sigma_pix, kernel = gaussian_beam(sigma_pix)) -> Float64

Factor by which convolution with `kernel` shrinks the standard deviation of a
white-noise field: `σ_out = beam_noise_gain(σ) * σ_in`.

For a kernel normalised to `Σ k = 1` this is `sqrt(Σ k²)`, which is measured
here from the kernel's impulse response rather than from a closed form, so the
value always matches what [`LowPass`](@ref) actually does to a map.
"""
function beam_noise_gain(sigma_pix::Real, kernel = gaussian_beam(sigma_pix))
    sigma_pix > 0 || throw(ArgumentError("sigma_pix must be positive (got $sigma_pix)."))
    # Grid wide enough that the impulse response is fully contained and the
    # replicated boundary never touches it.
    half = max(4, ceil(Int, 4 * sigma_pix))
    delta = zeros(2 * half + 1, 2 * half + 1)
    delta[half + 1, half + 1] = 1.0
    return sqrt(sum(abs2, LowPass(delta, kernel)))
end

"""
    noise_type_label(sigma_pix, sigma_chan) -> String

Short description of the noise model, for the `NOISETYP` FITS keyword.
"""
function noise_type_label(sigma_pix, sigma_chan)
    sigma_pix === nothing && sigma_chan === nothing && return "white"
    parts = String[]
    sigma_pix === nothing || push!(parts, "beam")
    sigma_chan === nothing || push!(parts, "channel")
    return join(parts, "+") * "-correlated"
end

"""
    add_noise!(cube, sigma, rng; sigma_pix = nothing, sigma_chan = nothing) -> cube

Add Gaussian noise of standard deviation `sigma` [K] to `cube` in place.

The two keywords select which instrumental responses the noise passes through
before it lands in the map — the same ones the signal went through:

- both `nothing` — white noise, independent in every pixel and channel.
- `sigma_pix` — **beam-correlated** across the sky, over a beam width.
- `sigma_chan` — **channel-correlated** along velocity, over the spectral
  response width.

Whatever the combination, white noise is drawn loud enough that after filtering
the map is left with a standard deviation of exactly `sigma`, per pixel and per
channel: the responses only shape the noise, they never change its level.

Memory note: correlating along velocity needs the whole noise cube at once, so
that path allocates a cube the size of `cube`. The purely spatial and white
paths still work one channel at a time.
"""
function add_noise!(cube::AbstractArray{<:Real,3}, sigma::Real, rng = Random.default_rng();
                    sigma_pix::Union{Nothing,Real} = nothing,
                    sigma_chan::Union{Nothing,Real} = nothing)
    sigma >= 0 || throw(ArgumentError("sigma must be non-negative (got $sigma)."))
    sigma == 0 && return cube

    nx, ny = size(cube, 1), size(cube, 2)

    # The responses act on independent axes, so their variance reductions
    # multiply and the draw is scaled by the inverse of the product.
    gain = 1.0
    kernel = sigma_pix === nothing ? nothing : gaussian_beam(sigma_pix)
    kernel === nothing || (gain *= beam_noise_gain(sigma_pix, kernel))
    sigma_chan === nothing || (gain *= spectral_noise_gain(sigma_chan))
    draw = float(sigma) / gain

    if sigma_chan !== nothing
        noise = rand(rng, Normal(0.0, draw), nx, ny, size(cube, 3))
        sigma_pix === nothing || smooth_cube!(noise, sigma_pix)
        smooth_spectral!(noise, sigma_chan)
        cube .+= noise
        return cube
    end

    for k in axes(cube, 3)
        white = rand(rng, Normal(0.0, draw), nx, ny)
        @views cube[:, :, k] .+= kernel === nothing ? white : LowPass(white, kernel)
    end
    return cube
end
