using Test
using Shine

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
