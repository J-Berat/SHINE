"""
Helpers to locate simulations on disk and validate the base directory.

A "simulation" is any directory that directly contains FITS files. The search is
breadth-first, so nested run directories under a common parent are all found.
"""

const _FITS_EXTS = (".fits", ".fit", ".fts")

_is_fits(name) = lowercase(splitext(name)[2]) in _FITS_EXTS
contains_fits_files(dir) = any(_is_fits, readdir(dir))

function get_simulation_list(base_dir)
    simulation_dirs = String[]
    dirs_to_check = [base_dir]
    while !isempty(dirs_to_check)
        current_dir = popfirst!(dirs_to_check)
        if contains_fits_files(current_dir)
            push!(simulation_dirs, current_dir)
        else
            subdirs = [joinpath(current_dir, d) for d in readdir(current_dir)
                       if isdir(joinpath(current_dir, d))]
            append!(dirs_to_check, subdirs)
        end
    end
    return sort!(simulation_dirs)
end

function display_simulations(simu_list)
    printstyled("\nAvailable simulations:\n"; color = :light_blue, bold = true)
    for (i, simu) in enumerate(simu_list)
        println("  [$i] $simu")
    end
end

"""
    ensure_directory_access(path) -> Union{Nothing, String}

Return `nothing` if `path` is a readable directory, otherwise a human-readable
error message suitable for [`warn_user`](@ref).
"""
function ensure_directory_access(path)
    if !isdir(path)
        return "Directory not found or not a directory: $(path)."
    elseif !isreadable(path)
        return "Directory is not readable: $(path)."
    end
    return nothing
end
