"""
Spectral response of the spectrometer, along the velocity axis.

The beam blurs the map across the sky; the spectrometer blurs it across
velocity. A channel does not sample the spectrum at a point — it integrates it
over a response function of finite width, so a line narrower than one channel
comes out broadened to the channel width. Ignoring that makes synthetic spectra
sharper than any instrument could record, which inflates peak brightness and
biases the second moment low.

The response is modelled as a Gaussian of a given FWHM in km/s, converted to a σ
in channels by [`channel_sigma`](@ref) and applied by [`smooth_spectral!`](@ref).
Like the beam it applies to the signal *and* to the noise — see
[`add_noise!`](@ref).
"""

"""
    channel_sigma(fwhm_kms, dvel_kms) -> Float64

Gaussian σ **in channels** of a spectral response of full width at half maximum
`fwhm_kms` on a velocity grid of channel width `dvel_kms`, both in km/s.
"""
channel_sigma(fwhm_kms::Real, dvel_kms::Real) =
    _fwhm_to_sigma(fwhm_kms, dvel_kms, "channel")

"""
    smooth_spectral!(cube, sigma_chan) -> cube

Convolve every spectrum of `cube` (axis 3) in place with a Gaussian of standard
deviation `sigma_chan` channels, replicating the end channels so the cube keeps
its shape.

Sky pixels are independent of one another here — this is a purely spectral
operation, the counterpart of [`smooth_cube!`](@ref) along velocity.
"""
function smooth_spectral!(cube::AbstractArray{<:Real,3}, sigma_chan::Real)
    sigma_chan > 0 || throw(ArgumentError("sigma_chan must be positive (got $sigma_chan)."))
    kernel = Kernel.gaussian((float(sigma_chan),))
    nx, ny = size(cube, 1), size(cube, 2)
    Threads.@threads for i in 1:nx
        for j in 1:ny
            # `cube[i, j, :]` copies the spectrum out, so the filter never reads
            # channels it has already overwritten.
            smoothed = imfilter(cube[i, j, :], kernel, "replicate")
            @views cube[i, j, :] .= smoothed
        end
    end
    return cube
end

"""
    spectral_noise_gain(sigma_chan) -> Float64

Factor by which [`smooth_spectral!`](@ref) shrinks the standard deviation of a
white-noise field, measured from the impulse response exactly as
[`beam_noise_gain`](@ref) does for the beam.
"""
function spectral_noise_gain(sigma_chan::Real)
    sigma_chan > 0 || throw(ArgumentError("sigma_chan must be positive (got $sigma_chan)."))
    half = max(4, ceil(Int, 4 * sigma_chan))
    delta = zeros(1, 1, 2 * half + 1)
    delta[1, 1, half + 1] = 1.0
    smooth_spectral!(delta, sigma_chan)
    return sqrt(sum(abs2, delta))
end
