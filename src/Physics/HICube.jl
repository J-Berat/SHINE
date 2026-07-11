"""
    HIcube(n, V, T, velArray, PixelLength_cm; mu = 1.0, therm = 0.0, progress = nothing)
        -> (Tb, Tb_thin, tau)

Build the 21-cm position-position-velocity (PPV) cubes for a full map by solving
[`HIspectrum`](@ref) on every sky pixel.

The three input cubes share the shape `(nx, ny, nz)` with the line of sight on
the **third** axis (guaranteed by [`ReadSimulation`](@ref)). The returned cubes
have shape `(nx, ny, nv)` with `nv = length(velArray)`.

The pixel loop is multithreaded: start Julia with `-t auto` (or set
`JULIA_NUM_THREADS`) to use every core. Pass a `progress` callback taking the
number of completed columns to drive a progress bar.
"""
function HIcube(n, V, T, velArray, PixelLength_cm; mu::Real = 1.0, therm::Real = 0.0,
                progress::Union{Nothing,Function} = nothing)
    size(n) == size(V) == size(T) ||
        throw(DimensionMismatch("n, V and T cubes must share the same shape."))
    ndims(n) == 3 || throw(DimensionMismatch("n, V and T must be 3D cubes."))

    nx, ny = size(n, 1), size(n, 2)
    nv = length(velArray)
    velvec = collect(float.(velArray))

    Tb = zeros(nx, ny, nv)
    Tb_thin = zeros(nx, ny, nv)
    tau = zeros(nx, ny, nv)

    done = Threads.Atomic{Int}(0)
    Threads.@threads for x in 1:nx
        for y in 1:ny
            tb, tbthin, t = HIspectrum(@view(n[x, y, :]), @view(V[x, y, :]), @view(T[x, y, :]),
                                       velvec, PixelLength_cm; mu = mu, therm = therm)
            @views Tb[x, y, :] .= tb
            @views Tb_thin[x, y, :] .= tbthin
            @views tau[x, y, :] .= t
        end
        if progress !== nothing
            Threads.atomic_add!(done, 1)
            progress(done[])
        end
    end

    return Tb, Tb_thin, tau
end


"""
    HIcube_tb(n, V, T, velArray, PixelLength_cm; kwargs...) -> Tb

Build only the brightness-temperature cube, avoiding the two additional output
cubes allocated by [`HIcube`](@ref).
"""
function HIcube_tb(n, V, T, velArray, PixelLength_cm; mu::Real = 1.0, therm::Real = 0.0,
                   progress::Union{Nothing,Function} = nothing)
    size(n) == size(V) == size(T) ||
        throw(DimensionMismatch("n, V and T cubes must share the same shape."))
    ndims(n) == 3 || throw(DimensionMismatch("n, V and T must be 3D cubes."))

    nx, ny = size(n, 1), size(n, 2)
    velvec = collect(float.(velArray))
    Tb = zeros(nx, ny, length(velvec))
    done = Threads.Atomic{Int}(0)
    Threads.@threads for x in 1:nx
        for y in 1:ny
            tb = HIspectrum_tb(@view(n[x, y, :]), @view(V[x, y, :]), @view(T[x, y, :]),
                               velvec, PixelLength_cm; mu = mu, therm = therm)
            @views Tb[x, y, :] .= tb
        end
        if progress !== nothing
            Threads.atomic_add!(done, 1)
            progress(done[])
        end
    end
    return Tb
end
