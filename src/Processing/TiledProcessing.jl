"""
Memory-bounded, tiled variant of the HI pipeline.

`ProcessHI` holds every full `(nx, ny, nv)` cube in memory at once, which becomes
prohibitive for large maps or many velocity channels. `ProcessHI_tiled` instead
sweeps the sky plane in square tiles, materialising only one `(tile, tile, nv)`
brightness cube at a time and reducing it immediately to **2D products**
(column densities, velocity moments, peak brightness, FFT CNM tracer). Peak
memory is therefore the three input cubes plus a single tile cube — independent
of the number of velocity channels held simultaneously.

Use this when you only need maps (not the full PPV cubes); for the cubes
themselves use [`ProcessHI`](@ref).
"""
function ProcessHI_tiled(simu, LOS;
                         TCNM::Real, TWNM::Real, velArray, PixelLength_cm::Real,
                         conversionn::Real = 1.0, conversionT::Real = 1.0, conversionV::Real = 1.0,
                         tile::Integer = 128,
                         compute_fractions::Bool = true, compute_moments::Bool = true,
                         compute_fftcnm::Bool = false, compute_stats::Bool = false,
                         mu::Real = 1.0, therm::Real = 0.0, metadata = nothing)

    tile > 0 || error("tile must be a positive integer (got $tile).")
    printstyled("\n▶ Processing HI (tiled, tile=$(tile)) for LOS $(LOS): $(simu)\n"; color = :cyan, bold = true)
    resultspath = joinpath(simu, LOS, "HI")
    mkpath(resultspath)

    n, VLOS, T = ReadSimulation(simu, LOS, conversionn, conversionT, conversionV)
    nx, ny = size(n, 1), size(n, 2)
    dv = length(velArray) > 1 ? abs(float(velArray[2] - velArray[1])) : 1.0

    # Column densities and fractions come straight from the density/temperature
    # cubes — no PPV cube required.
    info_user("Integrating column densities and phase maps")
    nCNM, nLNM, nWNM = HIPhases(n, T; TCNM = TCNM, TWNM = TWNM)
    NHI, NCNM, NLNM, NWNM = HIColumnDensity(n, nCNM, nLNM, nWNM, PixelLength_cm)
    nCNM = nLNM = nWNM = nothing        # release phase density cubes early
    WriteData2D(resultspath, NHI, "NHI"; metadata = metadata)
    WriteData2D(resultspath, NCNM, "NCNM"; metadata = metadata)
    WriteData2D(resultspath, NLNM, "NLNM"; metadata = metadata)
    WriteData2D(resultspath, NWNM, "NWNM"; metadata = metadata)

    if compute_fractions
        fCm, fLm, fWm = GasFractionMap(n, T; TCNM = TCNM, TWNM = TWNM)
        WriteData2D(resultspath, fCm, "fCNMmass"; metadata = metadata)
        WriteData2D(resultspath, fLm, "fLNMmass"; metadata = metadata)
        WriteData2D(resultspath, fWm, "fWNMmass"; metadata = metadata)
    end

    # 2D accumulators filled tile by tile.
    peakTb = zeros(nx, ny)
    mom0 = compute_moments ? zeros(nx, ny) : nothing
    mom1 = compute_moments ? zeros(nx, ny) : nothing
    mom2 = compute_moments ? zeros(nx, ny) : nothing
    fftc = compute_fftcnm ? zeros(nx, ny) : nothing

    ntiles = cld(nx, tile) * cld(ny, tile)
    info_user("Streaming $(ntiles) sky tile(s) through the radiative transfer")
    for i0 in 1:tile:nx, j0 in 1:tile:ny
        ir = i0:min(i0 + tile - 1, nx)
        jr = j0:min(j0 + tile - 1, ny)
        Tb, _, _ = HIcube(n[ir, jr, :], VLOS[ir, jr, :], T[ir, jr, :], velArray, PixelLength_cm;
                          mu = mu, therm = therm)

        peakTb[ir, jr] .= maxLOS(Tb)
        if compute_moments
            mom0[ir, jr] .= moment0(Tb, velArray)
            mom1[ir, jr] .= moment1(Tb, velArray)
            mom2[ir, jr] .= moment2(Tb, velArray)
        end
        compute_fftcnm && (fftc[ir, jr] .= fft_cnm_map(Tb, dv))
    end

    WriteData2D(resultspath, peakTb, "TbHI_peak"; metadata = metadata)
    if compute_moments
        WriteData2D(resultspath, mom0, "mom0"; metadata = metadata)
        WriteData2D(resultspath, mom1, "mom1"; metadata = metadata)
        WriteData2D(resultspath, mom2, "mom2"; metadata = metadata)
    end
    compute_fftcnm && WriteData2D(resultspath, fftc, "fftcnm"; metadata = metadata)

    if compute_stats
        info_user("Computing spatial statistics (power spectrum + structure function)")
        dx_pc = PixelLength_cm / PC_TO_CM
        write_spatial_stats(resultspath, NHI, "NHI"; dx = dx_pc)
        compute_moments && write_spatial_stats(resultspath, mom0, "mom0"; dx = dx_pc)
    end

    printstyled("✓ Finished LOS $(LOS) (tiled): 2D products in $(resultspath)\n"; color = :green, bold = true)
    return resultspath
end
