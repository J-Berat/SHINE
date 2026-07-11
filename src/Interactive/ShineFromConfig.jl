"""
    SHINE_from_config(config_path; quiet = true) -> Nothing

Run the full HI pipeline non-interactively from a JSON configuration file (the
same schema `run_shine` saves). Useful for scripted or reproducible runs and for
the command-line front-end `SHINE_cli.jl`.

The configuration must define at least `base_dir`; `simu_choice` (default
`"all"`) and `chosen_LOS_input` (default `"all"`) select the simulations and
lines of sight, and all processing options fall back to the interactive
defaults when omitted.
"""
function SHINE_from_config(config_path::AbstractString; quiet::Bool = true)
    isfile(config_path) || error("Configuration file not found: $(config_path).")
    cfg = JSON.parsefile(config_path)

    haskey(cfg, "base_dir") || error("Configuration is missing the required `base_dir` key.")
    base_dir = cfg["base_dir"]
    err = ensure_directory_access(base_dir)
    err === nothing || error(err)

    simu_list = get_simulation_list(base_dir)
    isempty(simu_list) && error("No simulations with FITS files found in $(base_dir).")

    chosen_idx = _parse_simu_indices(get(cfg, "simu_choice", "all"), length(simu_list))
    isempty(chosen_idx) && error("`simu_choice` selected no valid simulations.")
    chosen_simu = simu_list[chosen_idx]

    chosen_LOS = _parse_los(get(cfg, "chosen_LOS_input", "all"))
    isempty(chosen_LOS) && error("`chosen_LOS_input` selected no valid lines of sight.")

    # Fill required numeric defaults if absent.
    for (k, v) in ("velstart" => -30.0, "velend" => 30.0, "dvel" => 1.0,
                   "BoxLength_pc" => 50.0, "BoxLength_pix" => 256,
                   "TCNM" => 200.0, "TWNM" => 2000.0,
                   "conversionn" => 1.0, "conversionT" => 1.0, "conversionV" => 1.0)
        haskey(cfg, k) || (cfg[k] = v)
    end

    quiet || println("[SHINE] Running $(length(chosen_simu)) simulation(s) × $(length(chosen_LOS)) LOS from $(config_path)")
    run_shine_processing(cfg, chosen_simu, chosen_LOS, base_dir; config_saved_path = config_path)
    return nothing
end
