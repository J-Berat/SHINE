"""
Angular smoothing of the sky-plane maps, emulating a single-dish beam.

`LowPass` convolves a 2D image with a Gaussian kernel; `smooth_cube!` applies it
to every velocity channel of a cube in place. The kernel width is a run
parameter (`kernel_size_hi`, the Gaussian σ in pixels).
"""

using ImageFiltering: imfilter, Kernel

"""
    LowPass(img, kernel) -> Matrix

Convolve a 2D image with `kernel` (e.g. `Kernel.gaussian(σ)`), replicating edge
pixels so the map keeps its shape.
"""
LowPass(img::AbstractMatrix, kernel) = imfilter(img, kernel, "replicate")

"""
    gaussian_beam(sigma_pix) -> kernel

Gaussian smoothing kernel of standard deviation `sigma_pix` pixels.
"""
gaussian_beam(sigma_pix::Real) = Kernel.gaussian(sigma_pix)

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
