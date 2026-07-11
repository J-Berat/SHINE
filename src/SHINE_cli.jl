#!/usr/bin/env julia
#
# Non-interactive command-line front-end for SHINE.
#
#   julia --project src/SHINE_cli.jl --config path/to/shine_config.json [--quiet]
#   julia --project src/SHINE_cli.jl --demo            # generate + run demo data
#   julia --project src/SHINE_cli.jl --help
#
# The `--config` schema is exactly what `run_shine` saves interactively.

import Pkg
Pkg.activate(normpath(joinpath(@__DIR__, "..")); io = devnull)

using Shine

function usage()
    print("""
SHINE command-line interface — Synthetic H I Neutral Emission

Usage:
  julia --project src/SHINE_cli.jl --config <file.json> [--quiet]
  julia --project src/SHINE_cli.jl --demo [--quiet]
  julia --project src/SHINE_cli.jl --help

Options:
  --config <file>   Run the pipeline from a JSON configuration.
  --demo            Generate a tiny demo simulation and process it.
  --quiet           Reduce console output.
  --help            Show this message.
""")
end

function main(args)
    (isempty(args) || "--help" in args || "-h" in args) && (usage(); return)
    quiet = "--quiet" in args

    if "--demo" in args
        demo = make_demo_data()
        quiet || println("[SHINE] Demo data written to $(demo.base_dir)")
        SHINE_from_config(demo.config_path; quiet = quiet)
        return
    end

    ci = findfirst(==("--config"), args)
    if ci === nothing || ci == length(args)
        @error "Missing --config <file>. Use --help for usage."
        exit(2)
    end
    SHINE_from_config(args[ci + 1]; quiet = quiet)
end

main(ARGS)
