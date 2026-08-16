"""
    HIcube(n, V, T, velArray, PixelLength_cm; mu = 1.0, therm = 0.0, progress = nothing)
        -> (Tb, Tb_thin, tau)

Build the 21-cm position-position-velocity (PPV) cubes for a full map by solving
[`HIspectrum`](@ref) on every sky pixel.

The three input cubes share the shape `(nx, ny, nz)` with the line of sight on
the **third** axis (guaranteed by [`ReadSimulation`](@ref)). The returned cubes
have shape `(nx, ny, nv)` with `nv = length(velArray)`.

Sky rows are distributed over threads: start Julia with `-t auto` (or set
`JULIA_NUM_THREADS`) to use every core. Pass a `progress` callback taking the
number of completed sky rows to drive a progress bar; it is called from worker
threads and must be thread-safe.
"""
function HIcube(n, V, T, velArray, PixelLength_cm; mu::Real = 1.0, therm::Real = 0.0,
                progress::Union{Nothing,Function} = nothing)
    _check_cube_inputs(n, V, T, PixelLength_cm, mu, therm)

    nx, ny = size(n, 1), size(n, 2)
    velvec = _velocity_channels(velArray)
    grid = _uniform_velocity_grid(velvec)
    nv = length(velvec)

    Tb = zeros(nx, ny, nv)
    Tb_thin = zeros(nx, ny, nv)
    tau = zeros(nx, ny, nv)

    done = Threads.Atomic{Int}(0)
    # Rows are the threaded axis so that the inner loop walks `x`, which is
    # contiguous in memory: consecutive lines of sight then share cache lines.
    Threads.@threads for y in 1:ny
        # Allocated per row rather than per thread: negligible next to the row's
        # work, and correct even if a task migrates between threads.
        trans = Vector{Float64}(undef, nv)
        for x in 1:nx
            fill!(trans, 1.0)
            _HIspectrum!(@view(Tb[x, y, :]), @view(Tb_thin[x, y, :]), @view(tau[x, y, :]),
                         trans, @view(n[x, y, :]), @view(V[x, y, :]), @view(T[x, y, :]),
                         velvec, PixelLength_cm, mu, therm, grid)
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
    _check_cube_inputs(n, V, T, PixelLength_cm, mu, therm)

    nx, ny = size(n, 1), size(n, 2)
    velvec = _velocity_channels(velArray)
    grid = _uniform_velocity_grid(velvec)
    nv = length(velvec)

    Tb = zeros(nx, ny, nv)
    done = Threads.Atomic{Int}(0)
    Threads.@threads for y in 1:ny
        trans = Vector{Float64}(undef, nv)
        for x in 1:nx
            fill!(trans, 1.0)
            _HIspectrum_tb!(@view(Tb[x, y, :]), trans,
                            @view(n[x, y, :]), @view(V[x, y, :]), @view(T[x, y, :]),
                            velvec, PixelLength_cm, mu, therm, grid)
        end
        if progress !== nothing
            Threads.atomic_add!(done, 1)
            progress(done[])
        end
    end
    return Tb
end

function _check_cube_inputs(n, V, T, PixelLength_cm, mu, therm)
    size(n) == size(V) == size(T) ||
        throw(DimensionMismatch("n, V and T cubes must share the same shape."))
    ndims(n) == 3 || throw(DimensionMismatch("n, V and T must be 3D cubes."))
    PixelLength_cm > 0 || throw(ArgumentError("PixelLength_cm must be positive (got $PixelLength_cm)."))
    mu > 0 || throw(ArgumentError("mu must be positive (got $mu)."))
    therm >= 0 || throw(ArgumentError("therm must be non-negative (got $therm)."))
    return nothing
end
