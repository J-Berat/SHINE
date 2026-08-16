"""
    ProcessHI(simu, LOS; kwargs...) -> String

Full 21-cm processing of one simulation for one line of sight. Reads the
simulation cubes, separates the neutral phases, solves the radiative transfer,
optionally smooths and adds noise, and writes every product as a FITS file under
`simu/<LOS>/HI` (or `.../HI/filtered` when a response is applied). Returns the
output directory.

`LOS` is a box axis (`"x"`, `"y"`, `"z"`) or a `(theta, phi)` pair of angles in
degrees for an arbitrary viewing direction; [`los_label`](@ref) names the output
directory accordingly.

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
- `do_filter` (false)     : Gaussian-smooth every cube with a single-dish beam.
- `kernel_size_hi` (2.0)  : beam σ in **pixels** (used when no physical beam is
                            given).
- `beam_fwhm_arcmin` (0)  : beam FWHM in **arcmin**; overrides `kernel_size_hi`
                            and requires `distance_pc`. This is how a real
                            instrument is quoted (Arecibo ≈ 3.5′, GBT ≈ 9′).
- `distance_pc` (0)       : distance to the simulated region [pc], which sets
                            the angular scale of a pixel.
- `spectral_fwhm_kms`(0)  : FWHM [km/s] of the spectrometer's channel response,
                            convolved along the velocity axis. Independent of
                            `do_filter` — either response can be applied alone.
- `add_noise` (false)     : add Gaussian noise of std `sigma` [K]; independent
                            realisation per cube, drawn from `rng`.
- `correlated_noise`(true): when a beam is applied, filter the noise through it
                            so the map's noise is correlated over a beam width,
                            as in a real observation. Set `false` for white
                            noise (noise added after gridding).
- `periodic_box` (true)   : boundary used when `LOS` is a `(theta, phi)` pair;
                            wraps across faces, as an MHD box usually allows.
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
                   beam_fwhm_arcmin::Real = 0.0, distance_pc::Real = 0.0,
                   spectral_fwhm_kms::Real = 0.0,
                   add_noise::Bool = false, sigma::Real = 0.0, correlated_noise::Bool = true,
                   rng = Random.default_rng(), periodic_box::Bool = true,
                   mu::Real = 1.0, therm::Real = 0.0, metadata = nothing)

    label = los_label(LOS)
    printstyled("\n▶ Processing HI for LOS $(label): $(simu)\n"; color = :cyan, bold = true)
    resultspath = joinpath(simu, label, "HI")
    mkpath(resultspath)

    # --- read fields -------------------------------------------------------
    n, VLOS, T = ReadSimulation(simu, LOS, conversionn, conversionT, conversionV;
                                periodic = periodic_box)
    dv = length(velArray) > 1 ? abs(float(velArray[2] - velArray[1])) : 1.0
    metadata = merge_metadata(metadata, los_metadata(LOS))

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
    nx = size(n, 1)
    info_user("Solving 21-cm radiative transfer (total HI cube)")
    TbHI, TbthinHI, tauHI = HIcube(n, VLOS, T, velArray, PixelLength_cm; mu = mu, therm = therm,
                                   progress = make_progress("total", nx))

    cubes = Dict{String,Array{Float64,3}}(
        "TbHI" => TbHI, "TbthinHI" => TbthinHI, "tauHI" => tauHI,
    )

    if phase_cubes
        info_user("Solving radiative transfer per phase (CNM, LNM, WNM)")
        TbC, TbtC, tauC = HIcube(nCNM, VLOS, T, velArray, PixelLength_cm; mu = mu, therm = therm,
                                 progress = make_progress("CNM", nx))
        TbL, TbtL, tauL = HIcube(nLNM, VLOS, T, velArray, PixelLength_cm; mu = mu, therm = therm,
                                 progress = make_progress("LNM", nx))
        TbW, TbtW, tauW = HIcube(nWNM, VLOS, T, velArray, PixelLength_cm; mu = mu, therm = therm,
                                 progress = make_progress("WNM", nx))
        merge!(cubes, Dict(
            "TbCNM" => TbC, "TbthinCNM" => TbtC, "tauCNM" => tauC,
            "TbLNM" => TbL, "TbthinLNM" => TbtL, "tauLNM" => tauL,
            "TbWNM" => TbW, "TbthinWNM" => TbtW, "tauWNM" => tauW,
        ))
    end

    # --- optional instrumental response (beam and/or spectral) -------------
    beam = nothing
    spectral_sigma = nothing
    if do_filter || spectral_fwhm_kms > 0
        if do_filter
            beam = resolve_beam(PixelLength_cm; kernel_size_hi = kernel_size_hi,
                                beam_fwhm_arcmin = beam_fwhm_arcmin, distance_pc = distance_pc)
            info_user("Applying Gaussian beam ($(beam_description(beam)))")
            for (name, cube) in cubes
                startswith(name, "tau") || smooth_cube!(cube, beam.sigma_pix)  # keep τ un-smoothed
            end
            metadata = beam_metadata(metadata, beam)
        end

        if spectral_fwhm_kms > 0
            spectral_sigma = channel_sigma(spectral_fwhm_kms, dv)
            info_user("Applying spectral response (FWHM = $(spectral_fwhm_kms) km/s, " *
                      "σ = $(round(spectral_sigma, digits = 3)) chan)")
            spectral_sigma >= 0.5 || warn_user(
                "The spectral response is narrower than half a channel — the velocity " *
                "grid under-samples it and the cubes will barely change.")
            for (name, cube) in cubes
                startswith(name, "tau") || smooth_spectral!(cube, spectral_sigma)
            end
            metadata = merge_metadata(metadata, Dict("SPECFWHM" => float(spectral_fwhm_kms)))
        end

        # Everything written from here on is at the instrument's resolution.
        resultspath = joinpath(resultspath, "filtered")
        mkpath(resultspath)
    end

    # --- optional noise (independent realisation per brightness cube) ------
    if add_noise
        # Receiver noise enters ahead of the beam, so once a beam has been
        # applied the noise in the map is correlated over a beam width rather
        # than independent per pixel. `correlated_noise = false` falls back to
        # white noise, the right model for noise added after gridding.
        noise_sigma_pix = (beam !== nothing && correlated_noise) ? beam.sigma_pix : nothing
        noise_sigma_chan = (spectral_sigma !== nothing && correlated_noise) ? spectral_sigma : nothing
        label = noise_type_label(noise_sigma_pix, noise_sigma_chan)
        info_user("Adding $(label) Gaussian noise (σ = $(sigma) K)")
        for (name, cube) in cubes
            startswith(name, "Tb") || continue      # noise only on brightness cubes
            add_noise!(cube, sigma, rng; sigma_pix = noise_sigma_pix,
                       sigma_chan = noise_sigma_chan)
        end
        metadata = merge_metadata(metadata, Dict("NOISE" => float(sigma), "NOISETYP" => label))
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
        WriteData2D(resultspath, fft_cnm_map(TbHI, dv), "fftcnm"; metadata = metadata)
    end

    # --- spatial statistics ------------------------------------------------
    if compute_stats
        info_user("Computing spatial statistics (power spectrum + structure function)")
        dx_pc = PixelLength_cm / PC_TO_CM
        write_spatial_stats(resultspath, NHI, "NHI"; dx = dx_pc)
        mom0map !== nothing && write_spatial_stats(resultspath, mom0map, "mom0"; dx = dx_pc)
    end

    printstyled("✓ Finished LOS $(label): products in $(resultspath)\n"; color = :green, bold = true)
    return resultspath
end
