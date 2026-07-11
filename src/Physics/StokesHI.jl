"""
Toy linear-polarization model for HI emission.

If the HI emission is partially polarized along a plane-of-sky orientation angle
`θ` (for instance the local magnetic-field position angle), the Stokes Q and U
maps follow from integrating `T_B cos 2θ` and `T_B sin 2θ` along the line of
sight. These helpers are intentionally lightweight building blocks; `θ` may be a
scalar, a 2D map, or a full cube matching `TbHI`.
"""

"""
    QHI(TbHI, theta, dvel) -> Matrix

Stokes Q map: velocity integral of `T_B cos 2θ` weighted by the channel width
`dvel` [km/s].
"""
QHI(TbHI, theta, dvel) = intLOS(TbHI .* cos.(2 .* theta), dvel)

"""
    UHI(TbHI, theta, dvel) -> Matrix

Stokes U map: velocity integral of `T_B sin 2θ` weighted by `dvel`.
"""
UHI(TbHI, theta, dvel) = intLOS(TbHI .* sin.(2 .* theta), dvel)

"""    thetaHI(Q, U) -> polarization angle 0.5·atan(U, Q) [rad]."""
thetaHI(Q, U) = 0.5 .* atan.(U, Q)

"""    PHI(Q, U) -> polarized intensity sqrt(Q² + U²)."""
PHI(Q, U) = sqrt.(Q .^ 2 .+ U .^ 2)

"""    PolarFractionHI(Q, U, I) -> polarization fraction P / I."""
PolarFractionHI(Q, U, I) = PHI(Q, U) ./ I
