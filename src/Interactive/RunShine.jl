"""
Interactive command-line workflow for SHINE, mirroring MOOSE's `run_moose`:
a rainbow banner, a guided questionnaire built on [`ask_user`](@ref), a JSON
configuration that can be saved and replayed, and a per-run summary log.
"""

# --- config persistence ----------------------------------------------------

function load_previous_config(config_path = "shine_config.json")
    if isfile(config_path)
        return JSON.parsefile(config_path)
    else
        println("[Info] No existing config found at $(config_path). Starting with defaults.")
        return Dict{String,Any}()
    end
end

save_config(config::AbstractDict, config_path = "shine_config.json") =
    atomic_write_text(config_path, JSON.json(config, 2))

function format_duration(elapsed)
    total_ms = Dates.value(elapsed)
    h = total_ms ÷ 3_600_000
    m = (total_ms % 3_600_000) ÷ 60_000
    s = (total_ms % 60_000) ÷ 1_000
    ms = total_ms % 1_000
    return string(lpad(h, 2, "0"), ":", lpad(m, 2, "0"), ":", lpad(s, 2, "0"), ".", lpad(ms, 3, "0"))
end

function write_summary_log(base_dir, chosen_simu, chosen_LOS, elapsed, cfg; config_saved_path = nothing)
    log_path = joinpath(base_dir, "SHINE_summary.log")
    open(log_path, "a") do io
        println(io, "\nSHINE Summary Log")
        println(io, "=================")
        println(io, "Run completed at: $(Dates.format(now(), "yyyy-mm-dd HH:MM:SS"))")
        println(io, "SHINE version: $(shine_version())  git: $(shine_git_hash())")
        println(io, "Simulations processed:")
        foreach(s -> println(io, "  ", s), chosen_simu)
        println(io, "Lines of sight: $(join(chosen_LOS, ", "))")
        println(io, "Output directory: $base_dir")
        println(io, "Total execution time: $(format_duration(elapsed))")
        config_saved_path !== nothing && println(io, "Config saved: $(config_saved_path)")
        for k in ("conversionn", "conversionT", "conversionV", "BoxLength_pc", "BoxLength_pix",
                  "velstart", "velend", "dvel", "TCNM", "TWNM", "phase_cubes", "compute_fractions",
                  "compute_moments", "compute_fftcnm", "compute_stats",
                  "use_tiles", "tile", "do_filter", "kernel_size_hi",
                  "add_noise", "sigma", "rng_seed", "mu", "therm")
            haskey(cfg, k) && println(io, "$k: $(cfg[k])")
        end
    end
end

# --- helpers ---------------------------------------------------------------

const _VALID_LOS = ("x", "y", "z")

function _parse_los(los_choice)
    uppercase(strip(los_choice)) == "ALL" && return collect(_VALID_LOS)
    out = String[]
    for tok in split(los_choice, ",")
        s = lowercase(strip(tok))
        (s in _VALID_LOS && !(s in out)) && push!(out, s)
    end
    return out
end

function _parse_simu_indices(choice, nmax)
    uppercase(strip(choice)) == "ALL" && return collect(1:nmax)
    out = Int[]
    for tok in split(choice, ",")
        s = strip(tok)
        isempty(s) && continue
        idx = tryparse(Int, s)
        if idx === nothing
            warn_user("Ignoring invalid index: $(s).")
        elseif idx < 1 || idx > nmax
            warn_user("Index $(idx) out of range (1-$(nmax)).")
        elseif !(idx in out)
            push!(out, idx)
        end
    end
    return out
end

yesno(x) = uppercase(strip(String(x))) == "Y"

# --- main entry points -----------------------------------------------------

"""
    run_shine(; quiet = false, reset_config = true, help = false)

Launch the interactive SHINE workflow. With `reset_config = false` a previously
saved `shine_config.json` is offered as the set of defaults.
"""
function run_shine(; quiet::Bool = false, reset_config::Bool = true, help::Bool = false)
    if help
        print("""
SHINE v$(shine_version()) — Synthetic H I Neutral Emission

Usage:
  run_shine(; quiet=false, reset_config=true, help=false)

Options:
  quiet         Skip the startup banner.
  reset_config  Ignore a previous config and prompt again (default true).
  help          Show this message and exit.

Description:
  Interactive tool to turn MHD simulation cubes (density, temperature, velocity)
  into synthetic 21-cm HI observations: phase separation (CNM/LNM/WNM), column
  densities, brightness-temperature cubes via full radiative transfer, velocity
  moments, gas fractions, optional Gaussian beam smoothing and noise.

  Non-interactive:  julia --project src/SHINE_cli.jl --config cfg.json --quiet
  Outputs are written under each simulation's LOS/HI directory; a run summary is
  appended to SHINE_summary.log in the base directory.

  Author: Jack Berat
""")
        return
    end

    cfg, chosen_simu, chosen_LOS, base_dir, saved_path = run_shine_interactive(; quiet = quiet, reset_config = reset_config)
    run_shine_processing(cfg, chosen_simu, chosen_LOS, base_dir; config_saved_path = saved_path)
end

function run_shine_interactive(; quiet::Bool = false, reset_config::Bool = true)
    quiet || print_logo()

    default_config_path = joinpath(pwd(), "shine_config.json")
    if reset_config
        println("\n[Info] Previous configuration ignored (reset_config=true)")
        config = Dict{String,Any}()
    else
        cpath = ask_user("Enter the path to the configuration file to load", default_config_path)
        config = load_previous_config(cpath)
    end

    # 1. Base directory + simulation discovery ------------------------------
    section("Simulations")
    base_dir = ""
    simu_list = String[]
    while true
        candidate = ask_user("Enter the base directory for simulations", get(config, "base_dir", pwd()))
        err = ensure_directory_access(candidate)
        err !== nothing && (warn_user(err); continue)
        list = get_simulation_list(candidate)
        isempty(list) && (warn_user("No simulations with FITS files found in $(candidate)."); continue)
        base_dir = candidate; simu_list = list; break
    end
    config["base_dir"] = base_dir
    display_simulations(simu_list)

    simu_choice = ask_user("Enter 'all' or comma-separated indices (e.g., 1,3,5)",
                           get(config, "simu_choice", "all");
                           validate = c -> !isempty(_parse_simu_indices(c, length(simu_list))),
                           error_message = "Please select at least one valid simulation.")
    chosen_idx = _parse_simu_indices(simu_choice, length(simu_list))
    chosen_simu = simu_list[chosen_idx]
    config["simu_choice"] = simu_choice

    # 2. Unit conversions ---------------------------------------------------
    section("Unit conversions (to cm^-3, K, km/s)")
    config["conversionn"] = ask_user("Conversion factor for number density n to cm^-3", Float64(get(config, "conversionn", 1.0)))
    config["conversionT"] = ask_user("Conversion factor for temperature T to K", Float64(get(config, "conversionT", 1.0)))
    config["conversionV"] = ask_user("Conversion factor for velocity V to km/s", Float64(get(config, "conversionV", 1.0)))

    # 3. Box geometry -------------------------------------------------------
    section("Box geometry")
    config["BoxLength_pc"] = ask_user("Side of the box (pc)", Float64(get(config, "BoxLength_pc", 50.0)))
    config["BoxLength_pix"] = ask_user("Number of pixels along the line of sight", Int(get(config, "BoxLength_pix", 256)))

    # 4. Velocity axis ------------------------------------------------------
    section("Velocity axis")
    config["velstart"] = ask_user("Velocity range start (km/s)", Float64(get(config, "velstart", -30.0)))
    config["velend"] = ask_user("Velocity range end (km/s)", Float64(get(config, "velend", 30.0)))
    config["dvel"] = ask_user("Velocity resolution (km/s)", Float64(get(config, "dvel", 1.0)))

    # 5. Phase thresholds ---------------------------------------------------
    section("Neutral phases")
    config["TCNM"] = ask_user("CNM/LNM temperature threshold TCNM (K)", Float64(get(config, "TCNM", 200.0)))
    config["TWNM"] = ask_user("LNM/WNM temperature threshold TWNM (K)", Float64(get(config, "TWNM", 2000.0)))

    # 6. Products -----------------------------------------------------------
    section("Data products")
    config["phase_cubes"] = yesno(ask_user("Build per-phase T_B cubes (CNM/LNM/WNM)? (Y/N)", get(config, "phase_cubes", true) ? "Y" : "N"; validate = is_yes_no))
    config["compute_fractions"] = yesno(ask_user("Compute mass/volume fraction maps? (Y/N)", get(config, "compute_fractions", true) ? "Y" : "N"; validate = is_yes_no))
    config["compute_moments"] = yesno(ask_user("Compute velocity moment maps (0/1/2)? (Y/N)", get(config, "compute_moments", true) ? "Y" : "N"; validate = is_yes_no))
    config["compute_fftcnm"] = yesno(ask_user("Compute FFT CNM tracer map (Marchal+24)? (Y/N)", get(config, "compute_fftcnm", false) ? "Y" : "N"; validate = is_yes_no))
    config["compute_stats"] = yesno(ask_user("Compute spatial statistics (power spectrum + structure function)? (Y/N)", get(config, "compute_stats", false) ? "Y" : "N"; validate = is_yes_no))

    # 7. Beam smoothing -----------------------------------------------------
    section("Angular smoothing")
    do_filter = yesno(ask_user("Apply Gaussian beam smoothing? (Y/N)", get(config, "do_filter", false) ? "Y" : "N"; validate = is_yes_no))
    config["do_filter"] = do_filter
    config["kernel_size_hi"] = do_filter ? ask_user("Gaussian beam sigma (pixels)", Float64(get(config, "kernel_size_hi", 2.0))) : get(config, "kernel_size_hi", 2.0)

    # 8. Noise --------------------------------------------------------------
    section("Noise")
    add_noise = yesno(ask_user("Add Gaussian noise to brightness cubes? (Y/N)", get(config, "add_noise", false) ? "Y" : "N"; validate = is_yes_no))
    config["add_noise"] = add_noise
    config["sigma"] = add_noise ? ask_user("Noise standard deviation (K)", Float64(get(config, "sigma", 0.1))) : get(config, "sigma", 0.0)
    config["rng_seed"] = add_noise ? ask_user("Random seed (integer; 0 for a random seed)", Int(get(config, "rng_seed", 1234))) : get(config, "rng_seed", 0)

    # 8b. Performance / memory ---------------------------------------------
    section("Performance")
    use_tiles = yesno(ask_user("Use tiled low-memory processing? (2D products only, no PPV cubes) (Y/N)",
                               get(config, "use_tiles", false) ? "Y" : "N"; validate = is_yes_no))
    config["use_tiles"] = use_tiles
    config["tile"] = use_tiles ? ask_user("Sky-plane tile size (pixels)", Int(get(config, "tile", 128))) : Int(get(config, "tile", 128))

    # 9. Lines of sight -----------------------------------------------------
    section("Lines of sight")
    los_choice = ask_user("Enter 'all' or comma-separated lines of sight (e.g., x,z)",
                          get(config, "chosen_LOS_input", "all");
                          validate = c -> !isempty(_parse_los(c)),
                          error_message = "Please enter at least one of x, y, z.")
    chosen_LOS = _parse_los(los_choice)
    config["chosen_LOS_input"] = los_choice

    # 10. Save config -------------------------------------------------------
    section("Save configuration")
    save_path = ask_user("Path where the configuration should be saved", get(config, "config_path", default_config_path))
    saved_path = save_config(config, save_path)
    info_user("Configuration saved to $(saved_path)")

    return config, chosen_simu, chosen_LOS, base_dir, saved_path
end

"""
    run_shine_processing(cfg, chosen_simu, chosen_LOS, base_dir; config_saved_path = nothing)

Run [`ProcessHI`](@ref) over the selected simulations and lines of sight using a
configuration dictionary, then append a summary log. Shared by the interactive
workflow and [`SHINE_from_config`](@ref).
"""
function run_shine_processing(cfg::AbstractDict, chosen_simu, chosen_LOS, base_dir; config_saved_path = nothing)
    velArray = velocity_array(cfg["velstart"], cfg["velend"], cfg["dvel"])
    PixelLength_cm = pixel_length_cm(cfg["BoxLength_pc"], Int(cfg["BoxLength_pix"]))

    seed = Int(get(cfg, "rng_seed", 0))
    rng = (get(cfg, "add_noise", false) && seed != 0) ? MersenneTwister(seed) : Random.default_rng()

    metadata = Dict("SHINEVER" => shine_version(), "SHINEGIT" => shine_git_hash())

    use_tiles = get(cfg, "use_tiles", false)

    t0 = now()
    for simu in chosen_simu
        for LOS in chosen_LOS
            if use_tiles
                ProcessHI_tiled(simu, LOS;
                                TCNM = cfg["TCNM"], TWNM = cfg["TWNM"], velArray = velArray,
                                PixelLength_cm = PixelLength_cm, tile = Int(get(cfg, "tile", 128)),
                                conversionn = cfg["conversionn"], conversionT = cfg["conversionT"],
                                conversionV = cfg["conversionV"],
                                compute_fractions = get(cfg, "compute_fractions", true),
                                compute_moments = get(cfg, "compute_moments", true),
                                compute_fftcnm = get(cfg, "compute_fftcnm", false),
                                compute_stats = get(cfg, "compute_stats", false),
                                mu = get(cfg, "mu", 1.0), therm = get(cfg, "therm", 0.0),
                                metadata = metadata)
            else
                ProcessHI(simu, LOS;
                          TCNM = cfg["TCNM"], TWNM = cfg["TWNM"], velArray = velArray,
                          PixelLength_cm = PixelLength_cm,
                          conversionn = cfg["conversionn"], conversionT = cfg["conversionT"],
                          conversionV = cfg["conversionV"],
                          phase_cubes = get(cfg, "phase_cubes", true),
                          compute_fractions = get(cfg, "compute_fractions", true),
                          compute_moments = get(cfg, "compute_moments", true),
                          compute_fftcnm = get(cfg, "compute_fftcnm", false),
                          compute_stats = get(cfg, "compute_stats", false),
                          do_filter = get(cfg, "do_filter", false),
                          kernel_size_hi = get(cfg, "kernel_size_hi", 2.0),
                          add_noise = get(cfg, "add_noise", false), sigma = get(cfg, "sigma", 0.0),
                          rng = rng, mu = get(cfg, "mu", 1.0), therm = get(cfg, "therm", 0.0),
                          metadata = metadata)
            end
        end
    end
    elapsed = now() - t0

    write_summary_log(base_dir, chosen_simu, chosen_LOS, elapsed, cfg; config_saved_path = config_saved_path)
    printstyled("\n★ SHINE finished in $(format_duration(elapsed)). Summary: $(joinpath(base_dir, "SHINE_summary.log"))\n";
                color = :light_magenta, bold = true)
    return nothing
end
