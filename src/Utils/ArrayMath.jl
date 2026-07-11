"""
Array-level helpers used throughout the HI pipeline.

All cube helpers assume the line of sight is the **third** axis, following the
convention established by [`ReadSimulation`](@ref): a cube read for a given LOS
is permuted so that `cube[x, y, :]` walks along the line of sight.
"""

using Statistics: std, mean

"""
    intLOS(cube, PixelLength_cm) -> Matrix

Integrate a cube along the line of sight (axis 3), weighting each cell by the
pixel depth `PixelLength_cm`. For a density cube this yields a column density
(cm^-2); for a per-channel quantity it collapses the spectral axis.
"""
intLOS(cube::AbstractArray, PixelLength_cm::Real) =
    dropdims(sum(x -> x * PixelLength_cm, cube; dims = 3), dims = 3)

"""
    maxLOS(cube) -> Matrix

Peak value along the line of sight, e.g. the peak brightness temperature map of
a `T_B(v)` cube.
"""
maxLOS(cube::AbstractArray) = dropdims(maximum(cube, dims = 3), dims = 3)

"""
    sigmaLOS(cube) -> Matrix

Standard deviation along the line of sight.
"""
sigmaLOS(cube::AbstractArray) = dropdims(std(cube, dims = 3), dims = 3)

"""
    logindgen(n, a, b) -> Vector{Float64}

`n` points logarithmically spaced between `a` and `b` (inclusive). Mirrors the
IDL `logindgen` helper used by the original thermal-equilibrium routines.
"""
function logindgen(n::Integer, a::Real, b::Real)
    a > 0 && b > 0 || error("logindgen requires strictly positive bounds (got a=$a, b=$b).")
    return 10 .^ range(log10(a), log10(b); length = n)
end
