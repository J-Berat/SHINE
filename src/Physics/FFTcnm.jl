"""
Fourier-based CNM tracer, following Marchal et al. (2024).
See https://github.com/antoinemarchal/FFT-21cm.

Cold Neutral Medium clouds produce narrow 21-cm lines, which show up as
significant high-frequency power in the Fourier transform of the brightness
spectrum. The ratio of the peak high-frequency power to the zero-frequency
(total-power) component is a cheap, per-pixel CNM indicator that needs no phase
information from the simulation.
"""

using FFTW: fftfreq, plan_rfft
using LinearAlgebra: mul!

# Everything that depends only on the spectrum length: the real-input FFT plan,
# its scratch buffers, and the output bins whose frequency exceeds `klim`.
# Brightness spectra are real, so `rfft` computes the half-spectrum — the
# discarded half is the mirror image and never carries a selected frequency.
struct _CNMPlan{P}
    plan::P
    indices::Vector{Int}
    input::Vector{Float64}
    spectrum::Vector{ComplexF64}
end

function _cnm_plan(nv::Integer, dv::Real, klim::Real)
    k = fftfreq(nv, dv)
    nhalf = div(nv, 2) + 1
    indices = [i for i in 1:nhalf if k[i] > klim]
    input = Vector{Float64}(undef, nv)
    return _CNMPlan(plan_rfft(input), indices, input, Vector{ComplexF64}(undef, nhalf))
end

# Tracer for one spectrum, reusing `p`'s buffers. `Tb` is copied into the plan's
# input because FFTW needs a dense, correctly aligned array.
function _fft_cnm!(p::_CNMPlan, Tb::AbstractVector)
    isempty(p.indices) && return 0.0
    copyto!(p.input, Tb)
    mul!(p.spectrum, p.plan, p.input)

    @inbounds zero_amp = abs(p.spectrum[1])
    zero_amp == 0 && return 0.0

    peak = 0.0
    @inbounds for i in p.indices
        amp = abs(p.spectrum[i])
        amp > peak && (peak = amp)
    end
    return peak / zero_amp
end

_validate_fft_cnm(dv, klim) = begin
    dv > 0 && isfinite(dv) || throw(ArgumentError("dv must be finite and positive (got $dv)."))
    klim >= 0 && isfinite(klim) || throw(ArgumentError("klim must be finite and non-negative (got $klim)."))
    nothing
end

"""
    fft_cnm(Tb, dv; klim = 0.12) -> Float64

CNM tracer for a single brightness-temperature spectrum `Tb` sampled at velocity
step `dv` [km/s]. Returns the maximum Fourier amplitude beyond `klim`, normalised
by the zero-frequency amplitude.
"""
function fft_cnm(Tb::AbstractVector, dv::Real; klim::Real = 0.12)
    nv = length(Tb)
    nv > 0 || throw(ArgumentError("Tb spectrum must not be empty."))
    _validate_fft_cnm(dv, klim)
    all(isfinite, Tb) || throw(ArgumentError("Tb spectrum contains NaN or Inf."))
    return _fft_cnm!(_cnm_plan(nv, dv, klim), Tb)
end

"""
    fft_cnm_map(Tb, dv; klim = 0.12) -> Matrix

Apply [`fft_cnm`](@ref) to every line of sight of a `(nx, ny, nv)` cube.
"""
function fft_cnm_map(Tb::AbstractArray{<:Real,3}, dv::Real; klim::Real = 0.12)
    nv = size(Tb, 3)
    nv > 0 || throw(ArgumentError("Tb cube must contain at least one velocity channel."))
    _validate_fft_cnm(dv, klim)
    all(isfinite, Tb) || throw(ArgumentError("Tb spectrum contains NaN or Inf."))

    nx, ny = size(Tb, 1), size(Tb, 2)
    out = zeros(nx, ny)
    # One plan per task rather than per pixel: planning and the frequency
    # selection depend only on `nv`, and FFTW plans must not be shared across
    # concurrently running tasks.
    @sync for rows in _chunk_ranges(ny, Threads.nthreads())
        Threads.@spawn begin
            p = _cnm_plan(nv, dv, klim)
            @inbounds for y in rows, x in 1:nx
                out[x, y] = _fft_cnm!(p, @view Tb[x, y, :])
            end
        end
    end
    return out
end
