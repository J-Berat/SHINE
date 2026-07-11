using Test
using Shine
using Statistics: var

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
    end

    @testset "velocity moments recover an injected Gaussian line" begin
        vel = collect(range(-30, 30; step = 0.5))
        v0, sig = 5.0, 3.0
        prof = @. exp(-0.5 * ((vel - v0) / sig)^2)
        Tb = reshape(prof, 1, 1, :)
        @test moment1(Tb, vel)[1, 1] ≈ v0 atol = 0.1
        @test moment2(Tb, vel)[1, 1] ≈ sig atol = 0.1
    end

    @testset "thermal equilibrium is a sensible temperature" begin
        Tequ = tequilibrium(1.0)
        @test 10 < Tequ < 10_000
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
            ProcessHI_tiled(demo.simu_dir, "z"; tile = 5, kw...)
            tiled_mom0 = Shine.FITSIO.FITS(joinpath(demo.simu_dir, "z", "HI", "mom0.fits")) do f
                read(f[1])
            end
            @test tiled_mom0 ≈ full_mom0
        end
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
