"""
Fourier-based CNM tracer, following Marchal et al. (2024).
See https://github.com/antoinemarchal/FFT-21cm.

Cold Neutral Medium clouds produce narrow 21-cm lines, which show up as
significant high-frequency power in the Fourier transform of the brightness
spectrum. The ratio of the peak high-frequency power to the zero-frequency
(total-power) component is a cheap, per-pixel CNM indicator that needs no phase
information from the simulation.
"""

using FFTW: fft, fftfreq

"""
    fft_cnm(Tb, dv; klim = 0.12) -> Float64

CNM tracer for a single brightness-temperature spectrum `Tb` sampled at velocity
step `dv` [km/s]. Returns the maximum Fourier amplitude beyond `klim`, normalised
by the zero-frequency amplitude.
"""
function fft_cnm(Tb::AbstractVector, dv::Real; klim::Real = 0.12)
    nv = length(Tb)
    k = fftfreq(nv, dv)
    t = abs.(fft(Tb))
    t[1] == 0 && return 0.0
    pk = t ./ t[1]

    ind = findall(>(klim), k)
    isempty(ind) && return 0.0
    return maximum(pk[ind])
end

"""
    fft_cnm_map(Tb, dv; klim = 0.12) -> Matrix

Apply [`fft_cnm`](@ref) to every line of sight of a `(nx, ny, nv)` cube.
"""
function fft_cnm_map(Tb::AbstractArray{<:Real,3}, dv::Real; klim::Real = 0.12)
    nx, ny = size(Tb, 1), size(Tb, 2)
    out = zeros(nx, ny)
    Threads.@threads for i in 1:nx
        for j in 1:ny
            out[i, j] = fft_cnm(@view(Tb[i, j, :]), dv; klim = klim)
        end
    end
    return out
end
