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
passing it through the same beam as the signal, and renormalising so the map
still has the requested standard deviation. Without a beam (`sigma_pix =
nothing`) the noise stays white, which is the correct model for noise added
after gridding.
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
    add_noise!(cube, sigma, rng; sigma_pix = nothing) -> cube

Add Gaussian noise of standard deviation `sigma` [K] to every velocity channel
of `cube` in place, with an independent realisation per channel.

`sigma_pix` selects the noise model:

- `nothing` — white noise, independent in every pixel.
- a beam σ in pixels — **beam-correlated** noise: white noise is drawn, filtered
  through `gaussian_beam(sigma_pix)` and scaled up beforehand by
  `1 / beam_noise_gain(sigma_pix)` so the filtered field lands back on `sigma`.

In both cases `sigma` is the standard deviation of the noise in the final map,
per pixel and per channel.
"""
function add_noise!(cube::AbstractArray{<:Real,3}, sigma::Real, rng = Random.default_rng();
                    sigma_pix::Union{Nothing,Real} = nothing)
    sigma >= 0 || throw(ArgumentError("sigma must be non-negative (got $sigma)."))
    sigma == 0 && return cube

    nx, ny = size(cube, 1), size(cube, 2)

    if sigma_pix === nothing
        for k in axes(cube, 3)
            @views cube[:, :, k] .+= rand(rng, Normal(0.0, float(sigma)), nx, ny)
        end
        return cube
    end

    kernel = gaussian_beam(sigma_pix)
    draw = float(sigma) / beam_noise_gain(sigma_pix, kernel)
    for k in axes(cube, 3)
        @views cube[:, :, k] .+= LowPass(rand(rng, Normal(0.0, draw), nx, ny), kernel)
    end
    return cube
end
