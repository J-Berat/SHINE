"""
Velocity-coded RGB composites for HI brightness-temperature cubes.

Low, intermediate and high radial velocities are mapped respectively to blue,
green and red.  Each colour plane is the velocity-integrated brightness over
its interval and is normalized independently for display.
"""

"Return channel widths using the same convention as the velocity moments."
function _rgb_velocity_inputs(Tb, velArray)
    vel, widths = _moment_inputs(Tb, velArray)
    length(vel) >= 3 || throw(ArgumentError("an RGB composite needs at least three velocity channels."))
    return vel, widths
end

function _rgb_edges(vel, edges)
    if edges === nothing
        # Split by channel index so every default band is non-empty, including
        # for irregular velocity grids.
        sorted_vel = sort(vel)
        n = length(sorted_vel)
        i1, i2 = fld(n, 3), fld(2n, 3)
        return [first(sorted_vel), (sorted_vel[i1] + sorted_vel[i1 + 1]) / 2,
                (sorted_vel[i2] + sorted_vel[i2 + 1]) / 2, last(sorted_vel)]
    end

    out = collect(float.(edges))
    length(out) == 4 || throw(ArgumentError("rgb_velocity_edges must contain four velocities."))
    all(isfinite, out) || throw(ArgumentError("rgb_velocity_edges contains NaN or Inf."))
    issorted(out) && all(diff(out) .> 0) ||
        throw(ArgumentError("rgb_velocity_edges must be strictly increasing."))
    first(out) <= minimum(vel) && last(out) >= maximum(vel) ||
        throw(ArgumentError("rgb_velocity_edges must cover the complete velocity axis."))
    return out
end

"""
    velocity_rgb_bands(Tb, velArray; edges = nothing)

Integrate a `(x, y, velocity)` HI cube into three velocity bands.  The result is
a named tuple `(red, green, blue, edges)`, where red contains the highest and
blue the lowest velocities. `edges` may provide the four interval boundaries
in km/s; otherwise the channels are split into three equally populated bands.
"""
function velocity_rgb_bands(Tb, velArray; edges = nothing)
    vel, widths = _rgb_velocity_inputs(Tb, velArray)
    bounds = _rgb_edges(vel, edges)
    low = findall(v -> bounds[1] <= v < bounds[2], vel)
    middle = findall(v -> bounds[2] <= v < bounds[3], vel)
    high = findall(v -> bounds[3] <= v <= bounds[4], vel)
    any(isempty, (low, middle, high)) &&
        throw(ArgumentError("each RGB velocity interval must contain at least one channel."))

    # One sweep filling all three planes, rather than one masked copy of the
    # cube per band.
    red = zeros(size(Tb, 1), size(Tb, 2))
    green = zeros(size(Tb, 1), size(Tb, 2))
    blue = zeros(size(Tb, 1), size(Tb, 2))
    for (band, idx) in ((red, high), (green, middle), (blue, low))
        @inbounds for k in idx
            w = widths[k]
            for y in axes(Tb, 2), x in axes(Tb, 1)
                band[x, y] += Tb[x, y, k] * w
            end
        end
    end
    return (; red = red, green = green, blue = blue, edges = bounds)
end

function _display_stretch(map, percentile, stretch)
    positive = filter(x -> isfinite(x) && x > 0, vec(map))
    isempty(positive) && return zeros(Float32, size(map))
    scale = quantile(positive, percentile / 100)
    scale > 0 || return zeros(Float32, size(map))

    inv_asinh = 1 / asinh(stretch)
    out = Array{Float32}(undef, size(map))
    @inbounds for i in eachindex(out, map)
        x = clamp(map[i] / scale, 0, 1)
        out[i] = asinh(stretch * x) * inv_asinh
    end
    return out
end

function _velocity_rgb_image(red, green, blue, percentile, stretch)
    0 < percentile <= 100 || throw(ArgumentError("percentile must be in (0, 100]."))
    stretch > 0 || throw(ArgumentError("stretch must be positive."))
    size(red) == size(green) == size(blue) ||
        throw(DimensionMismatch("the red, green and blue maps must have the same shape."))
    return cat(_display_stretch(red, percentile, stretch),
               _display_stretch(green, percentile, stretch),
               _display_stretch(blue, percentile, stretch); dims = 3)
end

"""
    velocity_rgb(Tb, velArray; edges = nothing, percentile = 99.5, stretch = 3)

Build a normalized `nx × ny × 3` RGB array from an HI cube.  The three colour
planes use an asinh stretch after independent percentile normalization.
"""
function velocity_rgb(Tb, velArray; edges = nothing, percentile::Real = 99.5,
                      stretch::Real = 3.0)
    bands = velocity_rgb_bands(Tb, velArray; edges = edges)
    rgb = _velocity_rgb_image(bands.red, bands.green, bands.blue, percentile, stretch)
    return (; image = rgb, edges = bands.edges)
end

function _write_velocity_rgb_image(resultspath, data, bounds, filename)
    length(bounds) == 4 || throw(ArgumentError("RGB velocity bounds must contain four values."))
    size(data, 3) == 3 || throw(DimensionMismatch("RGB image must have three colour planes."))
    colors = RGBf.(data[:, :, 1], data[:, :, 2], data[:, :, 3])

    fig = Figure(size = (900, 820), backgroundcolor = :black)
    ax = Axis(fig[1, 1]; title = "HI velocity composite", aspect = DataAspect(),
              backgroundcolor = :black, titlecolor = :white, titlesize = 26)
    image!(ax, colors)
    hidedecorations!(ax)
    hidespines!(ax)
    Label(fig[2, 1],
          "BLUE  $(round(bounds[1]; digits=2))–$(round(bounds[2]; digits=2))   " *
          "GREEN  $(round(bounds[2]; digits=2))–$(round(bounds[3]; digits=2))   " *
          "RED  $(round(bounds[3]; digits=2))–$(round(bounds[4]; digits=2)) km/s";
          color = :white, fontsize = 18)
    rowsize!(fig.layout, 2, Auto(0.08))

    output = joinpath(resultspath, filename)
    atomic_write_path(output) do tmp
        save(tmp, fig)
    end
    @info "Wrote HI velocity RGB composite" path = output
    return output
end


"""
    write_velocity_rgb(resultspath, Tb, velArray; kwargs...) -> String

Render `RGBHI.png` with the velocity intervals annotated below the image.
"""
function write_velocity_rgb(resultspath::AbstractString, Tb, velArray;
                            edges = nothing, percentile::Real = 99.5,
                            stretch::Real = 3.0, filename::AbstractString = "RGBHI.png")
    composite = velocity_rgb(Tb, velArray; edges = edges, percentile = percentile, stretch = stretch)
    return _write_velocity_rgb_image(resultspath, composite.image, composite.edges, filename)
end

function _write_velocity_rgb_bands(resultspath, bands, bounds;
                                   percentile = 99.5, stretch = 3.0,
                                   filename = "RGBHI.png")
    data = _velocity_rgb_image(bands[1], bands[2], bands[3], percentile, stretch)
    return _write_velocity_rgb_image(resultspath, data, bounds, filename)
end
