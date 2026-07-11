"""
    ProcessHI(simu, LOS; kwargs...) -> String

Full 21-cm processing of one simulation for one line of sight. Reads the
simulation cubes, separates the neutral phases, solves the radiative transfer,
optionally smooths and adds noise, and writes every product as a FITS file under
`simu/LOS/HI` (or `simu/LOS/HI/filtered` when smoothing is enabled). Returns the
output directory.

# Keyword arguments
- `TCNM`, `TWNM`          : phase temperature thresholds [K].
- `velArray`              : velocity channels [km/s].
- `PixelLength_cm`        : LOS cell depth [cm].
- `conversionn/T/V`       : unit-conversion factors applied on read.
- `phase_cubes`  (true)   : also build CNM/LNM/WNM `T_B(v)` cubes (4× cost).
- `compute_fractions`(true): write mass- and volume-fraction maps.
- `compute_moments` (true): write velocity moment 0/1/2 maps of `TbHI`.
- `compute_fftcnm` (false): write the Marchal FFT CNM tracer map.
- `compute_stats` (false): write power-spectrum + structure-function of NHI/mom0.
- `do_filter` (false)     : Gaussian-smooth every cube (`kernel_size_hi` = σ pix).
- `add_noise` (false)     : add Gaussian noise of std `sigma` [K]; independent
                            realisation per cube, drawn from `rng`.
- `mu`, `therm`           : passed to [`HIspectrum`](@ref).
- `metadata`              : extra FITS header keywords.
"""
function ProcessHI(simu, LOS;
                   TCNM::Real, TWNM::Real, velArray, PixelLength_cm::Real,
                   conversionn::Real = 1.0, conversionT::Real = 1.0, conversionV::Real = 1.0,
                   phase_cubes::Bool = true, compute_fractions::Bool = true,
                   compute_moments::Bool = true, compute_fftcnm::Bool = false,
                   compute_stats::Bool = false,
                   do_filter::Bool = false, kernel_size_hi::Real = 2.0,
                   add_noise::Bool = false, sigma::Real = 0.0, rng = Random.default_rng(),
                   mu::Real = 1.0, therm::Real = 0.0, metadata = nothing)

    printstyled("\n▶ Processing HI for LOS $(LOS): $(simu)\n"; color = :cyan, bold = true)
    resultspath = joinpath(simu, LOS, "HI")
    mkpath(resultspath)

    # --- read fields -------------------------------------------------------
    n, VLOS, T = ReadSimulation(simu, LOS, conversionn, conversionT, conversionV)

    # --- phase separation + column densities -------------------------------
    info_user("Separating neutral phases and integrating column densities")
    nCNM, nLNM, nWNM = HIPhases(n, T; TCNM = TCNM, TWNM = TWNM)
    NHI, NCNM, NLNM, NWNM = HIColumnDensity(n, nCNM, nLNM, nWNM, PixelLength_cm)
    WriteData2D(resultspath, NHI, "NHI"; metadata = metadata)
    WriteData2D(resultspath, NCNM, "NCNM"; metadata = metadata)
    WriteData2D(resultspath, NLNM, "NLNM"; metadata = metadata)
    WriteData2D(resultspath, NWNM, "NWNM"; metadata = metadata)

    if compute_fractions
        info_user("Computing mass- and volume-fraction maps")
        fCm, fLm, fWm = GasFractionMap(n, T; TCNM = TCNM, TWNM = TWNM)
        WriteData2D(resultspath, fCm, "fCNMmass"; metadata = metadata)
        WriteData2D(resultspath, fLm, "fLNMmass"; metadata = metadata)
        WriteData2D(resultspath, fWm, "fWNMmass"; metadata = metadata)
        fCv, fLv, fWv = VolumeFractionMap(n, T; TCNM = TCNM, TWNM = TWNM)
        WriteData2D(resultspath, fCv, "fCNMvol"; metadata = metadata)
        WriteData2D(resultspath, fLv, "fLNMvol"; metadata = metadata)
        WriteData2D(resultspath, fWv, "fWNMvol"; metadata = metadata)
    end

    # --- radiative transfer ------------------------------------------------
    info_user("Solving 21-cm radiative transfer (total HI cube)")
    TbHI, TbthinHI, tauHI = HIcube(n, VLOS, T, velArray, PixelLength_cm; mu = mu, therm = therm)

    cubes = Dict{String,Array{Float64,3}}(
        "TbHI" => TbHI, "TbthinHI" => TbthinHI, "tauHI" => tauHI,
    )

    if phase_cubes
        info_user("Solving radiative transfer per phase (CNM, LNM, WNM)")
        TbC, TbtC, tauC = HIcube(nCNM, VLOS, T, velArray, PixelLength_cm; mu = mu, therm = therm)
        TbL, TbtL, tauL = HIcube(nLNM, VLOS, T, velArray, PixelLength_cm; mu = mu, therm = therm)
        TbW, TbtW, tauW = HIcube(nWNM, VLOS, T, velArray, PixelLength_cm; mu = mu, therm = therm)
        merge!(cubes, Dict(
            "TbCNM" => TbC, "TbthinCNM" => TbtC, "tauCNM" => tauC,
            "TbLNM" => TbL, "TbthinLNM" => TbtL, "tauLNM" => tauL,
            "TbWNM" => TbW, "TbthinWNM" => TbtW, "tauWNM" => tauW,
        ))
    end

    # --- optional angular smoothing ---------------------------------------
    if do_filter
        info_user("Applying Gaussian beam (σ = $(kernel_size_hi) pix)")
        for (name, cube) in cubes
            startswith(name, "tau") || smooth_cube!(cube, kernel_size_hi)  # keep τ un-smoothed
        end
        resultspath = joinpath(resultspath, "filtered")
        mkpath(resultspath)
    end

    # --- optional noise (independent realisation per brightness cube) ------
    if add_noise
        info_user("Adding Gaussian noise (σ = $(sigma) K)")
        for (name, cube) in cubes
            startswith(name, "Tb") || continue      # noise only on brightness cubes
            for k in axes(cube, 3)
                @views cube[:, :, k] .+= rand(rng, Normal(0.0, sigma), size(cube, 1), size(cube, 2))
            end
        end
    end

    # --- write cubes -------------------------------------------------------
    for (name, cube) in cubes
        WriteData3D(resultspath, cube, name, velArray; metadata = metadata)
    end

    # --- velocity moments of the total HI cube -----------------------------
    mom0map = nothing
    if compute_moments
        info_user("Computing velocity moment maps")
        mom0map = moment0(TbHI, velArray)
        WriteData2D(resultspath, mom0map, "mom0"; metadata = metadata)
        WriteData2D(resultspath, moment1(TbHI, velArray), "mom1"; metadata = metadata)
        WriteData2D(resultspath, moment2(TbHI, velArray), "mom2"; metadata = metadata)
    end

    # --- FFT CNM tracer ----------------------------------------------------
    if compute_fftcnm
        info_user("Computing FFT CNM tracer map")
        dv = length(velArray) > 1 ? abs(float(velArray[2] - velArray[1])) : 1.0
        WriteData2D(resultspath, fft_cnm_map(TbHI, dv), "fftcnm"; metadata = metadata)
    end

    # --- spatial statistics ------------------------------------------------
    if compute_stats
        info_user("Computing spatial statistics (power spectrum + structure function)")
        dx_pc = PixelLength_cm / PC_TO_CM
        write_spatial_stats(resultspath, NHI, "NHI"; dx = dx_pc)
        mom0map !== nothing && write_spatial_stats(resultspath, mom0map, "mom0"; dx = dx_pc)
    end

    printstyled("✓ Finished LOS $(LOS): products in $(resultspath)\n"; color = :green, bold = true)
    return resultspath
end
