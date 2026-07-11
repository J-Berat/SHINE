"""
Spatial (plane-of-sky) statistics of the 2D maps produced by the pipeline —
isotropic power spectrum and second-order structure function. These are the
standard turbulence diagnostics applied to HI column-density and moment maps.

Both estimators assume the periodic boundary conditions of a simulation box and
are computed with FFTs, then azimuthally averaged over `|k|` (power spectrum) or
lag `r` (structure function).
"""

using FFTW: fft, ifft, fftfreq
using Statistics: mean

# Azimuthally average `values` over `radii` into `nbins` equal-width bins,
# ignoring the zero-radius (DC) point. Returns (bin centres, mean per bin).
function _radial_average(values::AbstractArray, radii::AbstractArray, nbins::Integer)
    rmax = maximum(radii)
    edges = range(0, rmax; length = nbins + 1)
    centres = 0.5 .* (edges[1:end-1] .+ edges[2:end])
    sums = zeros(nbins)
    counts = zeros(Int, nbins)
    @inbounds for idx in eachindex(values, radii)
        r = radii[idx]
        r <= 0 && continue
        b = clamp(searchsortedfirst(edges, r) - 1, 1, nbins)
        sums[b] += values[idx]
        counts[b] += 1
    end
    means = [counts[b] > 0 ? sums[b] / counts[b] : NaN for b in 1:nbins]
    return collect(centres), means
end

"""
    power_spectrum(map; dx = 1.0, nbins = 50) -> (k, P)

Isotropic power spectrum of a 2D `map`. `dx` is the pixel size (sets the units of
the wavenumber `k`, in cycles per `dx`-unit). The mean is removed before the
transform. Returns the binned wavenumber `k` and power `P(k)`.
"""
function power_spectrum(map::AbstractMatrix; dx::Real = 1.0, nbins::Integer = 50)
    n, m = size(map)
    F = fft(map .- mean(map))
    P = abs2.(F) ./ (n * m)

    kx = fftfreq(n, 1 / dx)
    ky = fftfreq(m, 1 / dx)
    kr = [hypot(kx[i], ky[j]) for i in 1:n, j in 1:m]

    return _radial_average(P, kr, nbins)
end

"""
    structure_function(map; dx = 1.0, nbins = 50, order = 2) -> (r, SF)

Azimuthally averaged structure function `SF_p(r) = ⟨|f(x+r) − f(x)|^p⟩` of a 2D
`map`, with `p = order`. For `order == 2` the exact FFT autocovariance is used
(fast, periodic); for other orders it falls back to direct lag sampling. `dx`
sets the physical lag units. Returns the binned lag `r` and `SF(r)`.
"""
function structure_function(map::AbstractMatrix; dx::Real = 1.0, nbins::Integer = 50, order::Real = 2)
    n, m = size(map)
    lagx = [min(i - 1, n - (i - 1)) for i in 1:n]
    lagy = [min(j - 1, m - (j - 1)) for j in 1:m]
    r = [hypot(lagx[i], lagy[j]) * dx for i in 1:n, j in 1:m]

    if order == 2
        f0 = map .- mean(map)
        F = fft(f0)
        ac = real.(ifft(abs2.(F))) ./ (n * m)   # autocovariance, ac[1,1] = variance
        sf = 2 .* (ac[1, 1] .- ac)
        return _radial_average(sf, r, nbins)
    end

    # General order: average |f(x+lag) − f(x)|^order over the map via circshift.
    sf = zeros(n, m)
    @inbounds for j in 1:m, i in 1:n
        shifted = circshift(map, (i - 1, j - 1))
        sf[i, j] = mean(abs.(shifted .- map) .^ order)
    end
    return _radial_average(sf, r, nbins)
end

"""
    write_spatial_stats(resultspath, map, name; dx = 1.0, nbins = 50)

Compute the power spectrum and 2nd-order structure function of `map` and write
them as two-column text files `name_powerspectrum.dat` and `name_structfun.dat`
under `resultspath`.
"""
function write_spatial_stats(resultspath::AbstractString, map::AbstractMatrix, name::AbstractString;
                             dx::Real = 1.0, nbins::Integer = 50)
    mkpath(resultspath)
    k, P = power_spectrum(map; dx = dx, nbins = nbins)
    r, S = structure_function(map; dx = dx, nbins = nbins)

    _write_columns(joinpath(resultspath, "$(name)_powerspectrum.dat"), "k", "P(k)", k, P)
    _write_columns(joinpath(resultspath, "$(name)_structfun.dat"), "r", "SF2(r)", r, S)
    return resultspath
end

function _write_columns(path, c1, c2, x, y)
    atomic_write_text(path, string("# $c1\t$c2\n",
        join((string(x[i], "\t", y[i]) for i in eachindex(x, y)), "\n"), "\n"))
end
