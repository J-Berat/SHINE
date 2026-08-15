# SHINE

[![CI](https://github.com/j-berat/SHINE/actions/workflows/CI.yml/badge.svg)](https://github.com/j-berat/SHINE/actions/workflows/CI.yml)

**SHINE** (*Synthetic H I Neutral Emission*) is a Julia toolkit for turning
magnetohydrodynamic (MHD) simulation cubes into synthetic **21-cm neutral
hydrogen** observations. It is the HI counterpart of
[MOOSE](https://ui.adsabs.harvard.edu/abs/2026A%26A...708A.245B/abstract) (mock
synchrotron emission) and shares the same interactive command-line experience.

## Main features

- Separate the neutral gas into **CNM / LNM / WNM** phases by kinetic temperature.
- Compute total and per-phase **HI column-density** maps.
- Solve the full **21-cm radiative transfer** along the line of sight to produce
  optically-thick and optically-thin brightness-temperature cubes and the
  optical-depth cube.
- Compute **velocity moment** maps (integrated intensity, centroid, dispersion).
- Compute mass- and volume-weighted **gas-fraction** maps.
- Estimate the **CNM fraction from the Fourier spectrum** (Marchal et al. 2024).
- Apply an optional **Gaussian beam** — either as a σ in pixels or as a physical
  **FWHM in arcminutes** at a given source distance, so a real instrument can be
  reproduced (Arecibo ≈ 3.5′, GBT ≈ 9′ at 21 cm) — and add optional
  **Gaussian noise**.
- Overlay the **thermal-equilibrium curve** (Wolfire et al. 2003) on a
  **phase diagram** (n–P 2D histogram).
- Run interactively, from a JSON configuration, or through the CLI.

Inputs are read from per-field FITS files (`density.fits`, `temperature.fits`,
`Vx/Vy/Vz.fits`) or from a single HDF5 container. Products are written as FITS
files under each simulation's `LOS/HI` directory, with the velocity axis fully
described in the header. Every run appends provenance and timing to
`SHINE_summary.log`.

## Installation

SHINE requires [Julia 1.10 or later](https://julialang.org/downloads/). Clone
the repository, then install the dependencies:

```bash
julia --startup-file=no setup.jl          # add --test to also run the tests
# or:
julia --startup-file=no --project -e 'using Pkg; Pkg.instantiate()'
```

## Usage

Interactive workflow:

```julia
using Shine
run_shine()
```

Non-interactive run from a JSON configuration (template in
`config/default_config.json`):

```bash
julia --startup-file=no --project src/SHINE_cli.jl --config /path/to/shine_config.json --quiet
```

Validate the whole pipeline on self-contained demo data:

```julia
using Shine
demo = make_demo_data()                    # writes a tiny 2-phase simulation
SHINE_from_config(demo.config_path; quiet = false)
```

or simply:

```bash
julia --startup-file=no --project src/SHINE_cli.jl --demo
```

The radiative transfer is threaded over sky columns — start Julia with
`-t auto` to use every core — and draws a progress bar when it runs on a
terminal. Set `SHINE_PROGRESS=0` to silence the bar in batch jobs.

### Beam

Set `beam_fwhm_arcmin` together with `distance_pc` to smooth with a real
telescope beam; the distance is what converts the box's parsec-sized pixels into
an angular scale. Leaving `beam_fwhm_arcmin` at `0` falls back to
`kernel_size_hi`, the σ in pixels. Smoothed products are written under
`LOS/HI/filtered/` and carry the resolution in their header (`BMAJ`, `BMIN`,
`BPA` in degrees, plus `PIXSCALE` in arcmin and `DISTANCE` in pc), so they load
into astropy or CASA with the beam already attached.

```julia
ProcessHI(simu, "z"; TCNM = 200, TWNM = 2000, velArray = vel,
          PixelLength_cm = plen,
          do_filter = true, beam_fwhm_arcmin = 3.5, distance_pc = 150)
```

### Noise

Receiver noise enters ahead of the beam, so in a smoothed map it is correlated
over a beam width rather than independent per pixel. When a beam is applied the
noise is therefore drawn white, passed through that same beam and renormalised
so the map keeps the requested `sigma` — set `correlated_noise = false` for
white noise instead (the right model for noise added after gridding). Either
way the choice is recorded as `NOISETYP` in the header.

## Outputs

For each simulation and line of sight, in `<simu>/<LOS>/HI/`:

| Product | Type | Description |
|---------|------|-------------|
| `NHI`, `NCNM`, `NLNM`, `NWNM` | 2D | total / per-phase column density [cm⁻²] |
| `TbHI`, `TbCNM`, `TbLNM`, `TbWNM` | 3D | brightness-temperature cubes [K] |
| `TbthinHI`, … | 3D | optically-thin brightness-temperature cubes [K] |
| `tauHI`, … | 3D | optical-depth cubes |
| `mom0`, `mom1`, `mom2` | 2D | velocity moments [K km/s], [km/s], [km/s] |
| `fCNMmass`, … / `fCNMvol`, … | 2D | mass / volume fraction maps [%] |
| `fftcnm` | 2D | Fourier CNM tracer |

## License

SHINE is distributed under the [MIT License](LICENSE).

## Author

**Jack Berat** — main developer
