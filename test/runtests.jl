using Test
using Shine
using HDF5: h5open
using Statistics: var, std, cor
using Random: MersenneTwister

@testset "SHINE" begin

    @testset "phase separation is exact and complete" begin
        n = [1.0, 2.0, 3.0, 4.0]
        T = [100.0, 500.0, 1500.0, 5000.0]
        nCNM, nLNM, nWNM = HIPhases(n, T; TCNM = 200, TWNM = 2000)
        @test nCNM ≈ [1.0, 0.0, 0.0, 0.0]
        @test nLNM ≈ [0.0, 2.0, 3.0, 0.0]
        @test nWNM ≈ [0.0, 0.0, 0.0, 4.0]
        @test nCNM .+ nLNM .+ nWNM ≈ n            # partition of unity
    end

    @testset "gas & volume fractions sum to 100%" begin
        n = rand(8, 8, 8) .+ 0.1
        T = rand(8, 8, 8) .* 4000
        fC, fL, fW = GasFraction(n, T)
        @test fC + fL + fW ≈ 100 atol = 1e-8
        vC, vL, vW = VolumeFraction(n, T)
        @test vC + vL + vW ≈ 100 atol = 1e-8
    end

    @testset "optically thin limit of the radiative transfer" begin
        # Very low density => tau << 1 => Tb ≈ Tb_thin.
        nz = 20
        n = fill(1e-4, nz); T = fill(100.0, nz); v = zeros(nz)
        vel = range(-10, 10; step = 0.5)
        dz = pixel_length_cm(1.0, nz)
        Tb, Tb_thin, tau = HIspectrum(n, v, T, vel, dz)
        @test maximum(tau) < 1e-2
        @test isapprox(Tb, Tb_thin; rtol = 1e-2)

        # Very small optical depths must not be rounded down to zero.
        tiny, tiny_thin, _ = HIspectrum([1e-20], [0.0], [100.0], [0.0], dz)
        @test tiny[1] > 0
        @test tiny[1] ≈ tiny_thin[1] rtol = 1e-12
        @test_throws ArgumentError HIspectrum(n, v, T, vel, 0.0)
        @test_throws ArgumentError HIspectrum(n, v, T, vel, dz; mu = 0.0)
        @test_throws ArgumentError HIspectrum(n, v, T, vel, dz; therm = -1.0)

        full = HIspectrum(n, v, T, vel, dz)[1]
        @test HIspectrum_tb(n, v, T, vel, dz) ≈ full
    end

    @testset "velocity moments recover an injected Gaussian line" begin
        vel = collect(range(-30, 30; step = 0.5))
        v0, sig = 5.0, 3.0
        prof = @. exp(-0.5 * ((vel - v0) / sig)^2)
        Tb = reshape(prof, 1, 1, :)
        @test moment1(Tb, vel)[1, 1] ≈ v0 atol = 0.1
        @test moment2(Tb, vel)[1, 1] ≈ sig atol = 0.1

        irregular = [0.0, 1.0, 3.0]
        flat = ones(1, 1, 3)
        @test moment0(flat, irregular)[1, 1] ≈ 4.5
        @test_throws DimensionMismatch moment0(Tb, vel[1:end-1])
        @test_throws ArgumentError moment1(flat, [0.0, 2.0, 1.0])
    end

    @testset "thermal equilibrium is a sensible temperature" begin
        Tequ = tequilibrium(1.0)
        @test 10 < Tequ < 10_000
        @test_throws ArgumentError tequilibrium(0.0)
        @test_throws ArgumentError tequilibrium(1.0; npoints = 1)
        @test_throws DomainError tequilibrium(1.0; Tmin = 10, Tmax = 11)
    end

    @testset "geometry helpers" begin
        @test pixel_length_cm(2.0, 2) ≈ Shine.PC_TO_CM
        @test_throws Exception velocity_array(0.0, 10.0, -1.0)
    end

    @testset "invalid shapes are rejected" begin
        @test_throws DimensionMismatch HIPhases(ones(2), ones(3))
        @test_throws DimensionMismatch GasFraction(ones(2), ones(3))
        @test_throws DimensionMismatch HIcube(ones(2, 2), ones(2, 2), ones(2, 2), [0.0], 1.0)
    end

    @testset "spatial statistics" begin
        # White-noise map: flat power spectrum, structure function -> 2·variance.
        img = randn(64, 64)
        k, P = power_spectrum(img)
        @test length(k) == length(P)
        @test all(x -> isnan(x) || x >= 0, P)
        r, S = structure_function(img)
        finite = filter(!isnan, S)
        @test isapprox(maximum(finite), 2 * var(img); rtol = 0.3)
        @test_throws ArgumentError power_spectrum(img; dx = 0)
        @test_throws ArgumentError structure_function(fill(NaN, 2, 2))
        @test_throws ArgumentError structure_function(img; order = 0)
    end

    @testset "structure function of a linear gradient grows with lag" begin
        g = Float64[i for i in 1:32, _ in 1:32]
        r, S = structure_function(g; nbins = 16)
        finite_idx = findall(!isnan, S)
        @test S[finite_idx[end]] > S[finite_idx[1]]
    end

    @testset "tiled processing matches full moment maps" begin
        mktempdir() do dir
            demo = make_demo_data(joinpath(dir, "demo"); npix = 12)
            cfg = Shine.JSON.parsefile(demo.config_path)
            vel = velocity_array(cfg["velstart"], cfg["velend"], cfg["dvel"])
            plen = pixel_length_cm(cfg["BoxLength_pc"], Int(cfg["BoxLength_pix"]))
            kw = (; TCNM = cfg["TCNM"], TWNM = cfg["TWNM"], velArray = vel, PixelLength_cm = plen,
                    compute_fractions = false, compute_fftcnm = false)
            ProcessHI(demo.simu_dir, "z"; phase_cubes = false, kw...)
            full_mom0 = Shine.FITSIO.FITS(joinpath(demo.simu_dir, "z", "HI", "mom0.fits")) do f
                read(f[1])
            end
            full_mom1 = Shine.FITSIO.FITS(joinpath(demo.simu_dir, "z", "HI", "mom1.fits")) do f
                read(f[1])
            end
            full_mom2 = Shine.FITSIO.FITS(joinpath(demo.simu_dir, "z", "HI", "mom2.fits")) do f
                read(f[1])
            end
            ProcessHI_tiled(demo.simu_dir, "z"; tile = 5, kw...)
            tiled_mom0 = Shine.FITSIO.FITS(joinpath(demo.simu_dir, "z", "HI", "mom0.fits")) do f
                read(f[1])
            end
            @test tiled_mom0 ≈ full_mom0
            tiled_mom1 = Shine.FITSIO.FITS(joinpath(demo.simu_dir, "z", "HI", "mom1.fits")) do f
                read(f[1])
            end
            tiled_mom2 = Shine.FITSIO.FITS(joinpath(demo.simu_dir, "z", "HI", "mom2.fits")) do f
                read(f[1])
            end
            @test tiled_mom1 ≈ full_mom1
            @test tiled_mom2 ≈ full_mom2
        end
    end

    @testset "line-of-sight frame" begin
        # theta = 0 looks down +z and leaves the box axes alone.
        e1, e2, nhat = los_vectors(0, 0)
        @test collect(nhat) ≈ [0, 0, 1]
        @test collect(e1) ≈ [1, 0, 0]
        @test collect(e2) ≈ [0, 1, 0]

        # theta = 90 lays the line of sight in the xy plane.
        @test collect(los_vectors(90, 0)[3]) ≈ [1, 0, 0] atol = 1e-12
        @test collect(los_vectors(90, 90)[3]) ≈ [0, 1, 0] atol = 1e-12

        # Orthonormal and right-handed for an arbitrary direction.
        e1, e2, nhat = los_vectors(37, 128)
        for v in (e1, e2, nhat)
            @test sum(abs2, v) ≈ 1
        end
        dot3(a, b) = a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
        @test dot3(e1, e2) ≈ 0 atol = 1e-12
        @test dot3(e1, nhat) ≈ 0 atol = 1e-12
        @test dot3(e2, nhat) ≈ 0 atol = 1e-12
        cross3(a, b) = (a[2] * b[3] - a[3] * b[2],
                        a[3] * b[1] - a[1] * b[3],
                        a[1] * b[2] - a[2] * b[1])
        @test collect(cross3(e1, e2)) ≈ collect(nhat) atol = 1e-12

        @test_throws ArgumentError los_vectors(NaN, 0)
    end

    @testset "rotating the box" begin
        field = reshape(collect(1.0:(6 * 6 * 6)), 6, 6, 6)

        # Looking down +z changes nothing at all.
        @test project_cube(field, los_vectors(0, 0)...) ≈ field

        # Multiples of 90 degrees land back on grid points, so the rotated cube
        # holds exactly the same values, only permuted — no interpolation.
        for (th, ph) in ((90, 0), (90, 90), (180, 0), (90, 180))
            rot = project_cube(field, los_vectors(th, ph)...)
            @test sort(vec(rot)) ≈ sort(vec(field))
            @test sum(rot) ≈ sum(field)
        end

        # A uniform field stays uniform whatever the angle and the boundary.
        flat = fill(2.5, 5, 5, 5)
        for periodic in (true, false)
            @test project_cube(flat, los_vectors(33, 77)...; periodic = periodic) ≈ flat
        end

        # Trilinear interpolation is a convex combination of eight cells, so an
        # arbitrary rotation can never leave the range of the original.
        rot = project_cube(field, los_vectors(33, 77)...)
        @test minimum(rot) >= minimum(field) - 1e-9
        @test maximum(rot) <= maximum(field) + 1e-9
    end

    @testset "velocity projects onto the line of sight" begin
        # A uniform vector field must give V . nhat everywhere, exactly.
        Vx = fill(1.0, 4, 4, 4); Vy = fill(2.0, 4, 4, 4); Vz = fill(3.0, 4, 4, 4)
        for (th, ph, expected) in ((0, 0, 3.0), (90, 0, 1.0), (90, 90, 2.0), (180, 0, -3.0))
            los = project_los_velocity(Vx, Vy, Vz, los_vectors(th, ph)...)
            @test all(v -> isapprox(v, expected; atol = 1e-10), los)
        end

        # theta = 60, phi = 30: nhat = (sin60 cos30, sin60 sin30, cos60).
        los = project_los_velocity(Vx, Vy, Vz, los_vectors(60, 30)...)
        s60, c60 = sincos(deg2rad(60)); s30, c30 = sincos(deg2rad(30))
        @test los[2, 2, 2] ≈ 1 * s60 * c30 + 2 * s60 * s30 + 3 * c60

        @test_throws DimensionMismatch project_los_velocity(Vx, Vy, ones(2, 2, 2),
                                                            los_vectors(0, 0)...)
    end

    @testset "line-of-sight labels and metadata" begin
        @test los_label("z") == "z"
        @test los_label((45, 30)) == "th45_ph30"
        @test los_label((45.5, -30)) == "th45p5_phm30"
        @test los_metadata("y")["LOSAXIS"] == "y"
        @test los_metadata((45, 30))["LOSTHETA"] ≈ 45
        @test los_metadata((45, 30))["LOSPHI"] ≈ 30

        @test Shine._parse_los("all") == ["x", "y", "z"]
        @test Shine._parse_los("z,45:30") == Any["z", (45.0, 30.0)]
        @test Shine._parse_los("x,x,z") == Any["x", "z"]          # de-duplicated
        @test Shine._parse_los("q,z") == Any["z"]                  # junk dropped
        @test Shine._parse_los("1:2:3") == Any[]                   # malformed pair
    end

    @testset "an angled line of sight runs end to end" begin
        mktempdir() do dir
            demo = make_demo_data(joinpath(dir, "demo"); npix = 8)
            cfg = Shine.JSON.parsefile(demo.config_path)
            vel = velocity_array(cfg["velstart"], cfg["velend"], cfg["dvel"])
            plen = pixel_length_cm(cfg["BoxLength_pc"], Int(cfg["BoxLength_pix"]))
            out = ProcessHI(demo.simu_dir, (30.0, 45.0);
                            TCNM = cfg["TCNM"], TWNM = cfg["TWNM"], velArray = vel,
                            PixelLength_cm = plen, phase_cubes = false,
                            compute_fractions = false, compute_fftcnm = false)
            @test out == joinpath(demo.simu_dir, "th30_ph45", "HI")
            @test isfile(joinpath(out, "NHI.fits"))
            hdr = Shine.FITSIO.FITS(joinpath(out, "mom0.fits")) do f
                Shine.FITSIO.read_header(f[1])
            end
            @test hdr["LOSTHETA"] ≈ 30.0
            @test hdr["LOSPHI"] ≈ 45.0

            # Looking down +z through the rotated path must reproduce the plain
            # z-axis run: no resampling happens at theta = 0.
            axis = ReadSimulation(demo.simu_dir, "z", 1.0, 1.0, 1.0)
            rotated = ReadSimulation(demo.simu_dir, (0.0, 0.0), 1.0, 1.0, 1.0)
            @test rotated[1] ≈ axis[1]
            @test rotated[2] ≈ axis[2]
            @test rotated[3] ≈ axis[3]
        end
    end

    @testset "physical beam geometry" begin
        # 0.2 pc pixel at 100 pc: theta = 0.002 rad = 6.8755 arcmin.
        @test pixel_scale_arcmin(0.2, 100.0) ≈ rad2deg(0.002) * 60
        @test pixel_scale_arcmin(0.2, 200.0) ≈ pixel_scale_arcmin(0.1, 100.0)
        @test_throws ArgumentError pixel_scale_arcmin(0.0, 100.0)
        @test_throws ArgumentError pixel_scale_arcmin(0.2, 0.0)

        # A FWHM of exactly one pixel gives sigma = 1/(2 sqrt(2 ln 2)) pixels.
        @test beam_sigma_pix(1.0, 1.0) ≈ 1 / (2 * sqrt(2 * log(2)))
        @test beam_sigma_pix(4.0, 1.0) ≈ 4 * beam_sigma_pix(1.0, 1.0)
        # Only the FWHM/pixel ratio matters.
        @test beam_sigma_pix(6.0, 2.0) ≈ beam_sigma_pix(3.0, 1.0)
        @test_throws ArgumentError beam_sigma_pix(0.0, 1.0)
        @test_throws ArgumentError beam_sigma_pix(1.0, 0.0)
    end

    @testset "resolve_beam picks the right specification" begin
        plen = 0.2 * Shine.PC_TO_CM          # 0.2 pc pixels

        # Physical beam: 20 arcmin at 100 pc, where a pixel is 6.8755 arcmin.
        b = resolve_beam(plen; beam_fwhm_arcmin = 20.0, distance_pc = 100.0)
        @test b.fwhm_arcmin ≈ 20.0
        @test b.pixel_arcmin ≈ pixel_scale_arcmin(0.2, 100.0)
        @test b.sigma_pix ≈ beam_sigma_pix(20.0, b.pixel_arcmin)
        @test b.distance_pc ≈ 100.0

        # Pixel beam without a distance: no physical counterpart is invented.
        p = resolve_beam(plen; kernel_size_hi = 2.5)
        @test p.sigma_pix ≈ 2.5
        @test p.pixel_arcmin === nothing
        @test p.fwhm_arcmin === nothing

        # Pixel beam *with* a distance: the FWHM it corresponds to is filled in,
        # and feeding it back reproduces the same sigma.
        q = resolve_beam(plen; kernel_size_hi = 2.5, distance_pc = 100.0)
        @test q.fwhm_arcmin !== nothing
        @test resolve_beam(plen; beam_fwhm_arcmin = q.fwhm_arcmin,
                           distance_pc = 100.0).sigma_pix ≈ 2.5

        # A FWHM is meaningless without a distance, and the inputs are checked.
        @test_throws ArgumentError resolve_beam(plen; beam_fwhm_arcmin = 9.0)
        @test_throws ArgumentError resolve_beam(plen; kernel_size_hi = 0.0)
        @test_throws ArgumentError resolve_beam(plen; distance_pc = -1.0)
        @test_throws ArgumentError resolve_beam(0.0; kernel_size_hi = 2.0)

        # BMAJ/BMIN are degrees in the FITS convention; BMAJPIX is the FWHM in
        # pixels and survives even when the beam is only known in pixels.
        meta = Shine.beam_metadata(Dict("SHINEVER" => "0.1.0"), b)
        @test meta["SHINEVER"] == "0.1.0"
        @test meta["BMAJ"] ≈ 20.0 / 60
        @test meta["BMIN"] ≈ meta["BMAJ"]
        @test meta["BPA"] == 0.0
        @test meta["DISTANCE"] ≈ 100.0
        @test Shine.beam_metadata(nothing, p)["BMAJPIX"] ≈ 2.5 / (1 / (2 * sqrt(2 * log(2))))
        @test !haskey(Shine.beam_metadata(nothing, p), "BMAJ")
    end

    @testset "the beam smooths both sky axes, isotropically" begin
        img = zeros(65, 65)
        img[33, 33] = 1.0
        sm = LowPass(img, gaussian_beam(3.0))

        # A single-dish beam is a 2D angular response: a point source has to
        # spread along both sky axes, by the same amount.
        @test sm[33, 30] > 0
        @test sm[30, 33] > 0
        @test sm ≈ permutedims(sm)                      # isotropic
        @test sum(sm) ≈ 1.0 rtol = 1e-6                 # flux conserving
        # A wider beam spreads the same flux further, so the peak drops.
        @test maximum(LowPass(img, gaussian_beam(5.0))) <
              maximum(LowPass(img, gaussian_beam(2.0)))
    end

    @testset "a physical beam propagates into the FITS headers" begin
        mktempdir() do dir
            demo = make_demo_data(joinpath(dir, "demo"); npix = 8)
            cfg = Shine.JSON.parsefile(demo.config_path)
            vel = velocity_array(cfg["velstart"], cfg["velend"], cfg["dvel"])
            plen = pixel_length_cm(cfg["BoxLength_pc"], Int(cfg["BoxLength_pix"]))
            out = ProcessHI(demo.simu_dir, "z";
                            TCNM = cfg["TCNM"], TWNM = cfg["TWNM"], velArray = vel,
                            PixelLength_cm = plen, phase_cubes = false,
                            compute_fractions = false, compute_moments = false,
                            compute_fftcnm = false,
                            # 20 pc / 8 pix = 2.5 pc pixels; at 1720 pc that is
                            # ~5 arcmin per pixel, so a 15 arcmin beam is
                            # properly sampled (sigma ~ 1.3 pix).
                            do_filter = true, beam_fwhm_arcmin = 15.0, distance_pc = 1720.0)
            @test basename(out) == "filtered"
            hdr = Shine.FITSIO.FITS(joinpath(out, "TbHI.fits")) do f
                Shine.FITSIO.read_header(f[1])
            end
            @test hdr["BMAJ"] ≈ 15.0 / 60
            @test hdr["DISTANCE"] ≈ 1720.0
            @test hdr["PIXSCALE"] ≈ pixel_scale_arcmin(20.0 / 8, 1720.0)
        end
    end

    @testset "spectral response broadens along velocity only" begin
        @test channel_sigma(1.0, 1.0) ≈ 1 / (2 * sqrt(2 * log(2)))
        @test channel_sigma(4.0, 2.0) ≈ channel_sigma(2.0, 1.0)
        @test_throws ArgumentError channel_sigma(0.0, 1.0)
        @test_throws ArgumentError channel_sigma(1.0, 0.0)
        @test_throws ArgumentError smooth_spectral!(zeros(2, 2, 5), 0.0)

        cube = zeros(4, 5, 21)
        cube[2, 3, 11] = 1.0
        smooth_spectral!(cube, 2.0)
        @test cube[2, 3, 9] > 0                     # spread along velocity
        @test cube[2, 3, 11] < 1.0                  # peak lowered
        @test cube[1, 3, 11] == 0                   # no leak between sky pixels
        @test cube[2, 4, 11] == 0
        @test sum(cube) ≈ 1.0 rtol = 1e-6           # flux conserving
        # A line narrower than the response comes out at the response width:
        # the second moment can only grow.
        narrow = zeros(1, 1, 41); narrow[1, 1, 21] = 1.0
        broad = copy(narrow)
        smooth_spectral!(broad, 3.0)
        chan = collect(1:41)
        secmom(s) = sum(s .* (chan .- 21) .^ 2) / sum(s)
        @test secmom(vec(broad)) > secmom(vec(narrow))

        @test 0 < spectral_noise_gain(2.0) < 1
        @test spectral_noise_gain(4.0) < spectral_noise_gain(2.0)
        @test_throws ArgumentError spectral_noise_gain(0.0)
    end

    @testset "beam noise gain matches what the beam does to white noise" begin
        # Smoothing can only reduce the scatter, and the wider the beam the more.
        @test 0 < beam_noise_gain(2.0) < 1
        @test beam_noise_gain(4.0) < beam_noise_gain(2.0)
        @test_throws ArgumentError beam_noise_gain(0.0)

        # The predicted gain has to agree with the measured one, whatever the
        # kernel's internal representation.
        white = randn(MersenneTwister(7), 256, 256)
        for s in (1.5, 3.0)
            measured = std(LowPass(white, gaussian_beam(s))) / std(white)
            @test measured ≈ beam_noise_gain(s) rtol = 0.15
        end
    end

    @testset "beam-correlated noise" begin
        # Correlation between neighbours along the first axis: ~0 for white
        # noise, close to 1 for a field smoothed over several pixels.
        lag1(m) = cor(vec(m[1:end-1, :]), vec(m[2:end, :]))
        nx = ny = 160

        white = zeros(nx, ny, 2)
        add_noise!(white, 0.5, MersenneTwister(1))
        @test std(white) ≈ 0.5 rtol = 0.1
        @test abs(lag1(white[:, :, 1])) < 0.1

        corr = zeros(nx, ny, 2)
        add_noise!(corr, 0.5, MersenneTwister(1); sigma_pix = 3.0)
        # The requested sigma must survive the renormalisation ...
        @test std(corr) ≈ 0.5 rtol = 0.15
        # ... while neighbouring pixels are now far from independent, which is
        # the whole point: white noise under-states the scatter on beam scales.
        @test lag1(corr[:, :, 1]) > 0.8
        # Channels stay independent realisations of each other. The bound is
        # loose because correlated fields have few independent patches, so the
        # estimator itself is noisy — a genuinely shared realisation would
        # give ~1, not 0.2.
        @test abs(cor(vec(corr[:, :, 1]), vec(corr[:, :, 2]))) < 0.25

        # Same seed, same cube.
        again = zeros(nx, ny, 2)
        add_noise!(again, 0.5, MersenneTwister(1); sigma_pix = 3.0)
        @test again == corr

        # A zero sigma is a no-op, and a negative one is rejected.
        untouched = zeros(4, 4, 2)
        @test add_noise!(untouched, 0.0, MersenneTwister(1)) == zeros(4, 4, 2)
        @test_throws ArgumentError add_noise!(untouched, -1.0, MersenneTwister(1))
    end

    @testset "channel-correlated noise" begin
        lagv(c) = cor(vec(c[:, :, 1:end-1]), vec(c[:, :, 2:end]))
        lag1(m) = cor(vec(m[1:end-1, :]), vec(m[2:end, :]))
        nx = ny = 64

        # Spectral response only: neighbouring channels become correlated while
        # neighbouring sky pixels stay independent.
        spec = zeros(nx, ny, 24)
        add_noise!(spec, 0.4, MersenneTwister(11); sigma_chan = 2.0)
        @test std(spec) ≈ 0.4 rtol = 0.15
        @test lagv(spec) > 0.8
        @test abs(lag1(spec[:, :, 1])) < 0.1

        # Both responses: correlated on the sky *and* along velocity, still at
        # the requested sigma.
        both = zeros(nx, ny, 24)
        add_noise!(both, 0.4, MersenneTwister(11); sigma_pix = 2.0, sigma_chan = 2.0)
        @test std(both) ≈ 0.4 rtol = 0.2
        @test lagv(both) > 0.8
        @test lag1(both[:, :, 1]) > 0.7

        @test Shine.noise_type_label(nothing, nothing) == "white"
        @test Shine.noise_type_label(2.0, nothing) == "beam-correlated"
        @test Shine.noise_type_label(nothing, 2.0) == "channel-correlated"
        @test Shine.noise_type_label(2.0, 2.0) == "beam+channel-correlated"
    end

    @testset "noise model is recorded in the FITS header" begin
        mktempdir() do dir
            demo = make_demo_data(joinpath(dir, "demo"); npix = 8)
            cfg = Shine.JSON.parsefile(demo.config_path)
            vel = velocity_array(cfg["velstart"], cfg["velend"], cfg["dvel"])
            plen = pixel_length_cm(cfg["BoxLength_pc"], Int(cfg["BoxLength_pix"]))
            base = (; TCNM = cfg["TCNM"], TWNM = cfg["TWNM"], velArray = vel,
                      PixelLength_cm = plen, phase_cubes = false,
                      compute_fractions = false, compute_moments = false,
                      compute_fftcnm = false,
                      add_noise = true, sigma = 0.2, rng = MersenneTwister(3))

            noisetype(path) = Shine.FITSIO.FITS(path) do f
                Shine.FITSIO.read_header(f[1])["NOISETYP"]
            end

            out = ProcessHI(demo.simu_dir, "z"; do_filter = true, kernel_size_hi = 2.0, base...)
            @test noisetype(joinpath(out, "TbHI.fits")) == "beam-correlated"

            out = ProcessHI(demo.simu_dir, "z"; do_filter = true, kernel_size_hi = 2.0,
                            correlated_noise = false, base...)
            @test noisetype(joinpath(out, "TbHI.fits")) == "white"

            # No response at all, nothing to correlate with: the noise stays
            # white even though correlated_noise defaults to true.
            out = ProcessHI(demo.simu_dir, "z"; base...)
            @test noisetype(joinpath(out, "TbHI.fits")) == "white"
            @test basename(out) == "HI"

            # A spectral response alone still routes to filtered/ and correlates
            # the noise along velocity only.
            out = ProcessHI(demo.simu_dir, "z"; spectral_fwhm_kms = 3.0, base...)
            @test basename(out) == "filtered"
            hdr = Shine.FITSIO.FITS(joinpath(out, "TbHI.fits")) do f
                Shine.FITSIO.read_header(f[1])
            end
            @test hdr["NOISETYP"] == "channel-correlated"
            @test hdr["SPECFWHM"] ≈ 3.0

            # Both responses together.
            out = ProcessHI(demo.simu_dir, "z"; do_filter = true, kernel_size_hi = 2.0,
                            spectral_fwhm_kms = 3.0, base...)
            @test noisetype(joinpath(out, "TbHI.fits")) == "beam+channel-correlated"
        end
    end

    @testset "tiled run writes mass *and* volume fraction maps" begin
        mktempdir() do dir
            demo = make_demo_data(joinpath(dir, "demo"); npix = 6)
            cfg = Shine.JSON.parsefile(demo.config_path)
            vel = velocity_array(cfg["velstart"], cfg["velend"], cfg["dvel"])
            plen = pixel_length_cm(cfg["BoxLength_pc"], Int(cfg["BoxLength_pix"]))
            ProcessHI_tiled(demo.simu_dir, "z";
                            TCNM = cfg["TCNM"], TWNM = cfg["TWNM"], velArray = vel,
                            PixelLength_cm = plen, tile = 4,
                            compute_fractions = true, compute_moments = false,
                            compute_fftcnm = false)
            hi = joinpath(demo.simu_dir, "z", "HI")
            # The tiled path must emit the same fraction product set as ProcessHI.
            for name in ("fCNMmass", "fLNMmass", "fWNMmass", "fCNMvol", "fLNMvol", "fWNMvol")
                @test isfile(joinpath(hi, name * ".fits"))
            end
        end
    end

    @testset "progress callback" begin
        # Disabled display must yield `nothing` so the kernels skip their counter.
        @test Shine.make_progress("job", 10; enabled = false) === nothing
        @test Shine.make_progress("job", 0; enabled = true) === nothing

        cb = Shine.make_progress("job", 4; enabled = true)
        @test cb isa Function
        mktemp() do path, io
            redirect_stdout(io) do
                for k in 1:4
                    cb(k)
                end
            end
            flush(io)
            text = read(path, String)
            @test occursin("job", text)
            @test occursin("100%", text)
        end
    end

    @testset "elapsed-time formatting" begin
        @test Shine._fmt_elapsed(45) == "45s"
        @test Shine._fmt_elapsed(125) == "2m05s"
        @test Shine._fmt_elapsed(3725) == "1h02m"
    end

    @testset "numeric prompts enforce validation" begin
        # This is what guards the new `mu > 0` / `therm >= 0` questions: the
        # rejected first answer is re-asked, the valid second one is returned.
        mktemp() do path, io
            write(io, "-1.0\n2.5\n")
            flush(io)
            seekstart(io)
            val = redirect_stdout(devnull) do
                redirect_stdin(io) do
                    Shine.ask_user("mean molecular weight", 1.0;
                                   validate = x -> x > 0,
                                   error_message = "must be strictly positive")
                end
            end
            @test val == 2.5
        end

        # An empty line keeps the default without running `validate`.
        mktemp() do path, io
            write(io, "\n")
            flush(io)
            seekstart(io)
            val = redirect_stdout(devnull) do
                redirect_stdin(io) do
                    Shine.ask_user("fixed dispersion", 0.0; validate = x -> x >= 0)
                end
            end
            @test val == 0.0
        end
    end

    @testset "simulation readers validate and orient fields" begin
        mktempdir() do dir
            demo = make_demo_data(joinpath(dir, "fits"); npix = 5)
            for (los, expected) in (("x", (5, 5, 5)), ("y", (5, 5, 5)), ("z", (5, 5, 5)))
                n, v, T = ReadSimulation(demo.simu_dir, los, 1.0, 1.0, 1.0)
                @test size(n) == size(v) == size(T) == expected
            end

            h5dir = joinpath(dir, "hdf5")
            mkpath(h5dir)
            h5open(joinpath(h5dir, "fields.h5"), "w") do h5
                h5["density"] = ones(2, 3, 4)
                h5["temperature"] = fill(100.0, 2, 3, 4)
                h5["velocity_x"] = zeros(2, 3, 4)
            end
            hn, hv, hT = ReadSimulation(h5dir, "x", 1.0, 1.0, 1.0)
            @test size(hn) == size(hv) == size(hT) == (3, 4, 2)

            bad = joinpath(dir, "bad")
            mkpath(bad)
            h5open(joinpath(bad, "fields.h5"), "w") do h5
                h5["density"] = ones(2, 2, 2)
                h5["temperature"] = fill(100.0, 2, 2, 3)
                h5["velocity_z"] = zeros(2, 2, 2)
            end
            @test_throws DimensionMismatch ReadSimulation(bad, "z", 1.0, 1.0, 1.0)
        end
    end

    @testset "non-finite and unphysical inputs are rejected" begin
        @test_throws ArgumentError GasFraction([-1.0, 2.0], [100.0, 3000.0])
        @test_throws ArgumentError VolumeFraction(ones(2), [100.0, NaN])
        @test_throws ArgumentError fft_cnm([1.0, NaN], 1.0)
        @test_throws ArgumentError fft_cnm([1.0], 0.0)
    end

    @testset "end-to-end demo pipeline" begin
        mktempdir() do dir
            demo = make_demo_data(joinpath(dir, "demo"); npix = 8)
            SHINE_from_config(demo.config_path; quiet = true)
            hi = joinpath(demo.simu_dir, "z", "HI")
            @test isfile(joinpath(hi, "NHI.fits"))
            @test isfile(joinpath(hi, "TbHI.fits"))
            @test isfile(joinpath(hi, "mom0.fits"))
            @test isfile(joinpath(dir, "demo", "SHINE_summary.log"))
        end
    end
end
