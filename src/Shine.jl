module Shine

using Crayons
using Dates
using Distributions
using FITSIO
using JSON
using LinearAlgebra
using Logging
using Random
using SHA
using Statistics
using TOML

# --- Utilities -------------------------------------------------------------
include(joinpath("Utils", "Prompts.jl"))
include(joinpath("Utils", "ArrayMath.jl"))
include(joinpath("Utils", "AtomicWrite.jl"))

# --- File I/O --------------------------------------------------------------
include(joinpath("FileIO", "SimulationDiscovery.jl"))
include(joinpath("FileIO", "ReadSimulation.jl"))
include(joinpath("FileIO", "WriteDataOnDisk.jl"))

# --- Physics ---------------------------------------------------------------
include(joinpath("Physics", "Constants.jl"))
include(joinpath("Physics", "HISpectrum.jl"))
include(joinpath("Physics", "HICube.jl"))
include(joinpath("Physics", "HIPhases.jl"))
include(joinpath("Physics", "GasFraction.jl"))
include(joinpath("Physics", "ThermalEquilibrium.jl"))
include(joinpath("Physics", "StokesHI.jl"))
include(joinpath("Physics", "Moments.jl"))
include(joinpath("Physics", "FFTcnm.jl"))

# --- Processing ------------------------------------------------------------
include(joinpath("Processing", "Filter.jl"))
include(joinpath("Processing", "Velocity.jl"))
include(joinpath("Processing", "ProcessHI.jl"))

# --- Diagnostics (loads CairoMakie; kept last so the numerical core is usable
#     even if a plotting backend is unavailable at include time) ------------
include(joinpath("Diagnostics", "PhaseDiagram.jl"))

# --- Interactive workflow --------------------------------------------------
include(joinpath("Interactive", "Logo.jl"))
include(joinpath("Interactive", "RunShine.jl"))
include(joinpath("Interactive", "ShineFromConfig.jl"))
include(joinpath("Interactive", "DemoData.jl"))

export run_shine, SHINE_from_config, make_demo_data,
       ReadSimulation, read_field, get_simulation_list,
       WriteData2D, WriteData3D,
       HIspectrum, HIcube, HIPhases, HIColumnDensity,
       GasFraction, VolumeFraction, GasFractionMap, VolumeFractionMap,
       ref_bistable, tequilibrium,
       QHI, UHI, thetaHI, PHI, PolarFractionHI,
       moment0, moment1, moment2,
       fft_cnm, fft_cnm_map,
       LowPass, smooth_cube!, velocity_array, pixel_length_cm,
       ProcessHI, phase_diagram, intLOS, maxLOS, sigmaLOS, logindgen

const SHINE_PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))

"""Package version read from `Project.toml`."""
function shine_version()
    try
        return String(get(TOML.parsefile(joinpath(SHINE_PROJECT_ROOT, "Project.toml")), "version", "unknown"))
    catch
        return "unknown"
    end
end

"""Short git hash of the working tree (with a `+dirty` suffix if modified)."""
function shine_git_hash()
    git = Sys.which("git")
    git === nothing && return "unknown"
    try
        rev = readchomp(`$git -C $SHINE_PROJECT_ROOT rev-parse --short=12 HEAD`)
        dirty = success(`$git -C $SHINE_PROJECT_ROOT diff --quiet --ignore-submodules HEAD`) ? "" : "+dirty"
        return rev * dirty
    catch
        return "unknown"
    end
end

end # module
