"""
Angular smoothing of the sky-plane maps, emulating a single-dish beam.

`LowPass` convolves a 2D image with a Gaussian kernel; `smooth_cube!` applies it
to every velocity channel of a cube in place. The kernel width is a run
parameter (`kernel_size_hi`, the Gaussian σ in pixels).
"""

using ImageFiltering: imfilter, imfilter!, Kernel, KernelFactors

"""
    LowPass(img, kernel) -> Matrix

Convolve a 2D image with `kernel` (e.g. `Kernel.gaussian(σ)`), replicating edge
pixels so the map keeps its shape.
"""
LowPass(img::AbstractMatrix, kernel) = imfilter(img, kernel, "replicate")

"""
    gaussian_beam(sigma_pix) -> kernel

Gaussian smoothing kernel of standard deviation `sigma_pix` pixels. The kernel is
returned in separable (factored) form: convolving with the two 1D factors in turn
costs `O(w)` per pixel instead of the `O(w²)` of the equivalent square kernel.
"""
gaussian_beam(sigma_pix::Real) = KernelFactors.gaussian((sigma_pix, sigma_pix))

"""
    smooth_cube!(cube, sigma_pix)

Smooth every velocity channel (axis 3) of `cube` in place with a Gaussian beam.
"""
function smooth_cube!(cube::AbstractArray{<:Real,3}, sigma_pix::Real)
    kernel = gaussian_beam(sigma_pix)
    nx, ny = size(cube, 1), size(cube, 2)

    # imfilter! cannot write into its own source, so each task keeps one channel
    # -sized pair of buffers and reuses it for every channel it owns.
    @sync for chans in _chunk_ranges(size(cube, 3), Threads.nthreads())
        Threads.@spawn begin
            src = Matrix{eltype(cube)}(undef, nx, ny)
            dest = Matrix{eltype(cube)}(undef, nx, ny)
            for k in chans
                copyto!(src, @view cube[:, :, k])
                imfilter!(dest, src, kernel, "replicate")
                copyto!(@view(cube[:, :, k]), dest)
            end
        end
    end
    return cube
end
