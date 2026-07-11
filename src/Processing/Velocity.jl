"""
Velocity axis and pixel-geometry helpers.
"""

"""
    velocity_array(velstart, velend, dvel) -> StepRangeLen

Regular velocity grid [km/s] from `velstart` to `velend` with channel width
`dvel`. Errors on a non-positive width.
"""
function velocity_array(velstart::Real, velend::Real, dvel::Real)
    dvel > 0 || error("Velocity resolution must be positive (got $dvel).")
    velend > velstart || error("velend ($velend) must be greater than velstart ($velstart).")
    return range(; start = float(velstart), stop = float(velend), step = float(dvel))
end

"""
    pixel_length_cm(BoxLength_pc, BoxLength_pix) -> Float64

Physical depth of one cell along the line of sight [cm], for a cubic box of side
`BoxLength_pc` parsecs sampled by `BoxLength_pix` pixels.
"""
function pixel_length_cm(BoxLength_pc::Real, BoxLength_pix::Integer)
    BoxLength_pix > 0 || error("BoxLength_pix must be positive (got $BoxLength_pix).")
    return BoxLength_pc / BoxLength_pix * PC_TO_CM
end
