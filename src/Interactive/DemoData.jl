"""
    make_demo_data(dir = "shine_demo"; npix = 16) -> (; base_dir, simu_dir, config_path)

Write a tiny two-phase synthetic simulation (density, temperature and velocity
FITS cubes) plus a ready-to-run configuration, so the whole pipeline can be
exercised without external data:

```julia
using Shine
demo = make_demo_data()
SHINE_from_config(demo.config_path; quiet = false)
```

The box contains a cold, dense CNM clump embedded in a warm, diffuse WNM, with a
smooth line-of-sight velocity gradient — enough to produce non-trivial phase
maps, column densities and brightness-temperature cubes.
"""
function make_demo_data(dir::AbstractString = "shine_demo"; npix::Integer = 16)
    base_dir = abspath(expanduser(String(dir)))
    simu_dir = joinpath(base_dir, "sim1")
    mkpath(simu_dir)

    nx = ny = nz = Int(npix)
    n = fill(0.3, nx, ny, nz)        # WNM floor [cm^-3]
    T = fill(6000.0, nx, ny, nz)     # WNM temperature [K]
    Vx = zeros(nx, ny, nz)
    Vy = zeros(nx, ny, nz)
    Vz = zeros(nx, ny, nz)

    cx, cy, cz = nx ÷ 2, ny ÷ 2, nz ÷ 2
    r2 = (0.2 * nx)^2
    for k in 1:nz, j in 1:ny, i in 1:nx
        d2 = (i - cx)^2 + (j - cy)^2 + (k - cz)^2
        if d2 <= r2                  # cold dense CNM clump
            n[i, j, k] = 30.0
            T[i, j, k] = 80.0
        elseif d2 <= 4 * r2          # unstable LNM shell
            n[i, j, k] = 3.0
            T[i, j, k] = 800.0
        end
        Vz[i, j, k] = 10.0 * (k - cz) / nz    # LOS-z velocity gradient [km/s]
        Vx[i, j, k] = 10.0 * (i - cx) / nx
        Vy[i, j, k] = 10.0 * (j - cy) / ny
    end

    for (field, data) in ("density" => n, "temperature" => T, "Vx" => Vx, "Vy" => Vy, "Vz" => Vz)
        atomic_write_path(joinpath(simu_dir, "$field.fits")) do tmp
            FITS(tmp, "w") do f
                write(f, data)
            end
        end
    end

    config = Dict{String,Any}(
        "base_dir" => base_dir, "simu_choice" => "all", "chosen_LOS_input" => "z",
        "conversionn" => 1.0, "conversionT" => 1.0, "conversionV" => 1.0,
        "BoxLength_pc" => 20.0, "BoxLength_pix" => nz,
        "velstart" => -20.0, "velend" => 20.0, "dvel" => 1.0,
        "TCNM" => 200.0, "TWNM" => 2000.0,
        "phase_cubes" => true, "compute_fractions" => true,
        "compute_moments" => true, "compute_fftcnm" => true,
        "do_filter" => false, "kernel_size_hi" => 2.0,
        "add_noise" => false, "sigma" => 0.0, "rng_seed" => 0,
    )
    config_path = joinpath(base_dir, "shine_config.json")
    atomic_write_text(config_path, JSON.json(config, 2))

    return (; base_dir = base_dir, simu_dir = simu_dir, config_path = config_path)
end
