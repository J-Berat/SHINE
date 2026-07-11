"""
Thermal balance of the neutral ISM following Wolfire et al. (2003).

`ref_bistable` returns the volumetric heating (Γ) and cooling (Λ) rates for a
given density and temperature; `tequilibrium` inverts Λ(T) = Γ(T) at fixed
density to give the equilibrium temperature, which traces the classic bistable
`P–n` curve overlaid on the phase diagrams.
"""

using Interpolations: linear_interpolation, Line

"""
    ref_bistable(n, T) -> (Gamma, Lambda)

Heating `Gamma` and cooling `Lambda` rates [erg s^-1 cm^-3] for density `n`
[cm^-3] and temperature `T` [K]. Includes C II and O fine-structure cooling,
Lyman-α cooling, photoelectric heating on grains, and recombination cooling on
charged grains (Wolfire et al. 1995, 2003).
"""
function ref_bistable(n, T)
    # Electron density from the C II / hydrogen ionisation balance (Wolfire 2003, C15).
    neq = 2.4e-3 * ((T / 100)^0.25) / 0.5
    x = neq / n
    x = clamp(x, 3.5e-4 * 0.4, 0.1)

    # Fine-structure cooling by singly-ionised carbon (92 K transition).
    froidcII = 92.0 * 1.38e-16 * 2.0 *
               (2.8e-7 * ((T / 100)^(-0.5)) * x + 8.0e-10 * ((T / 100)^0.07)) *
               3.5e-4 * 0.4 * exp(-92 / T)

    # Fine-structure cooling by neutral oxygen (abundance 4.5e-4).
    froido = 1.0e-26 * sqrt(T) * (24.0 * exp(-228 / T) + 7.0 * exp(-326 / T)) * 4.5e-4

    # Lyman-α cooling by neutral hydrogen (Spitzer 1978).
    froidh = 7.3e-19 * x * exp(-118400 / T)

    froid = froidcII + froidh + froido

    # Photoelectric heating on grains, scaled to the Habing/Draine field G0/1.7.
    G0 = 1 / 1.7
    param = G0 * sqrt(T) / (n * x)
    epsilon = 4.9e-2 / (1 + (param / 1925)^0.73)
    epsilon += 3.7e-2 * (T / 1.0e4)^0.7 / (1 + (param / 5e3))
    chaud = 1.0e-24 * epsilon * G0

    # Recombination cooling on positively charged grains.
    bet = 0.74 / (T^0.068)
    froidrec = 4.65e-30 * (T^0.94) * (param^bet) * x

    Gamma = chaud * n
    Lambda = (froid + froidrec) * n^2

    return Gamma, Lambda
end

"""
    tequilibrium(n; Tmin = 10, Tmax = 10000, npoints = 1000) -> Float64

Equilibrium temperature [K] at density `n` [cm^-3], found by root-solving
Λ(T) = Γ(T) via monotone interpolation of the net cooling `Λ − Γ`.
"""
function tequilibrium(n; Tmin::Real = 10, Tmax::Real = 10000, npoints::Integer = 1000)
    T = logindgen(npoints, Tmin, Tmax)
    net = similar(T)
    for i in eachindex(T)
        Gamma, Lambda = ref_bistable(n, T[i])
        net[i] = Lambda - Gamma
    end

    # Interpolation knots must be strictly increasing and unique.
    order = sortperm(net)
    net_sorted = net[order]
    T_sorted = T[order]
    keep = [true; diff(net_sorted) .!= 0]
    itp = linear_interpolation(net_sorted[keep], T_sorted[keep]; extrapolation_bc = Line())

    return itp(0.0)
end
