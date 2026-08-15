"""
Lightweight progress reporting for the long radiative-transfer loops.

[`HIcube`](@ref) and [`HIcube_tb`](@ref) accept a `progress` callback that is
invoked with the number of completed sky columns. [`make_progress`](@ref) builds
such a callback backed by a single-line bar that redraws in place.

The callback is called from inside a `Threads.@threads` loop, so updates are
serialised by a lock and the bar only redraws when the integer percentage
actually changes — a few hundred writes per run rather than one per column.
"""

# A bar is only useful on an attached terminal: in CI, in a pipe or in a batch
# job it would just fill the log with control characters. `SHINE_PROGRESS=0`
# forces it off even on a TTY.
_progress_enabled() = (stdout isa Base.TTY) && get(ENV, "SHINE_PROGRESS", "1") != "0"

function _fmt_elapsed(seconds::Real)
    s = round(Int, seconds)
    s < 60 && return string(s, "s")
    m, rs = divrem(s, 60)
    m < 60 && return string(m, "m", lpad(rs, 2, "0"), "s")
    h, rm = divrem(m, 60)
    return string(h, "h", lpad(rm, 2, "0"), "m")
end

"""
    make_progress(label, total; width = 28, enabled = _progress_enabled())
        -> Union{Nothing,Function}

Return a thread-safe callback `done -> nothing` drawing a progress bar labelled
`label` for a job of `total` units, or `nothing` when progress display is
disabled (non-interactive output, `SHINE_PROGRESS=0`, or a non-positive
`total`).

Returning `nothing` rather than a no-op closure is deliberate: the compute
kernels skip their atomic counter entirely when no callback is supplied, so a
disabled bar costs nothing at all.
"""
function make_progress(label::AbstractString, total::Integer;
                       width::Int = 28, enabled::Bool = _progress_enabled())
    enabled || return nothing
    ntotal = Int(total)
    ntotal > 0 || return nothing

    lk = ReentrantLock()
    last_pct = Ref(-1)
    t0 = time()
    # Pad so the bars of consecutive stages (total, CNM, LNM, WNM) line up.
    tag = rpad(label, 9)

    return function (done::Integer)
        pct = clamp((100 * Int(done)) ÷ ntotal, 0, 100)
        lock(lk) do
            pct <= last_pct[] && return nothing
            last_pct[] = pct
            filled = (pct * width) ÷ 100
            line = string("\r    ↳ ", tag, " ",
                          "█"^filled, "░"^(width - filled),
                          " ", lpad(pct, 3), "%  ", _fmt_elapsed(time() - t0))
            printstyled(line; color = :light_green)
            pct >= 100 && println()
            flush(stdout)
            return nothing
        end
        return nothing
    end
end
