"""
Thermal phase diagram: a 2D histogram of the gas in the (log n, log P) plane,
optionally overlaid with isotherms and the Wolfire et al. (2003) thermal-
equilibrium curve computed by [`tequilibrium`](@ref).

Requires CairoMakie; kept separate from the numerical core so the pipeline can
run headless without a plotting backend loaded.
"""

using CairoMakie
using LaTeXStrings

const _PHASE_CMAP = cgrad([:white, "#c7e9b4", "#7fcdbb", "#41b6c4", "#253494"],
                          [0.0, 0.25, 0.5, 0.75, 1.0])

function _plot_isotherms!(ax, x_data, T_values, labels, colors; percentage = 0.35)
    for (i, Tiso) in enumerate(sort(collect(T_values)))
        n_range = range(minimum(x_data), maximum(x_data), length = 100)
        P_line = log10.(10 .^ n_range .* Tiso)
        lines!(ax, n_range, P_line; linestyle = :dash, linewidth = 3,
               color = colors[mod1(i, length(colors))], label = "T = $Tiso K")

        pos = clamp(round(Int, percentage * length(n_range)), 8, length(n_range) - 8)
        label = i <= length(labels) ? labels[i] : "T = $Tiso K"
        angle = atan(P_line[pos + 7] - P_line[pos - 7], n_range[pos + 7] - n_range[pos - 7]) - deg2rad(28)
        text!(ax, Point(Float32(n_range[pos]), Float32(P_line[pos - 3])); text = label,
              color = colors[mod1(i, length(colors))], align = (:center, :center),
              rotation = angle, fontsize = 12)
    end
end

"""
    phase_diagram(n, P; T_values = (200, 2000), n_bins = 500, apply_log = true,
                  show_isotherms = true, show_Tequ = true, title = "",
                  xlabel = L"\\log_{10}\\, n\\ [\\mathrm{cm^{-3}}]",
                  ylabel = L"\\log_{10}\\, P/k_B\\ [\\mathrm{K\\, cm^{-3}}]",
                  savepath = nothing) -> Figure

2D histogram of `(n, P)` pairs (flattened). `n` is a density, `P` a pressure
(e.g. `n .* T`). Colours encode `log10(counts)`.
"""
function phase_diagram(n, P;
                       T_values = (200, 2000), n_bins::Integer = 500,
                       apply_log::Bool = true, show_isotherms::Bool = true, show_Tequ::Bool = true,
                       title::AbstractString = "",
                       xlabel = L"\log_{10}\, n\ [\mathrm{cm^{-3}}]",
                       ylabel = L"\log_{10}\, P/k_B\ [\mathrm{K\, cm^{-3}}]",
                       colorbar_label = L"\log_{10}\,\mathrm{counts}",
                       savepath = nothing)

    x = vec(collect(float.(n)))
    y = vec(collect(float.(P)))

    return with_theme(theme_latexfonts()) do
        fig = Figure(backgroundcolor = :white, size = (820, 620))

        if apply_log
            (all(>(0), x) && all(>(0), y)) || error("n and P must be strictly positive for log scaling.")
            x = log10.(x)
            y = log10.(y)
        end

        ax = Axis(fig[1, 1]; xlabel = xlabel, ylabel = ylabel, title = title,
                  xlabelsize = 22, ylabelsize = 22, xticklabelsize = 18, yticklabelsize = 18,
                  xgridvisible = false, ygridvisible = false,
                  xminorticksvisible = true, yminorticksvisible = true)

        x_min, x_max = extrema(x)
        y_min, y_max = extrema(y)
        x_edges = range(x_min, x_max, length = n_bins + 1)
        y_edges = range(y_min, y_max, length = n_bins + 1)

        counts = zeros(Float64, n_bins, n_bins)
        xi = clamp.(searchsortedfirst.(Ref(x_edges), x) .- 1, 1, n_bins)
        yi = clamp.(searchsortedfirst.(Ref(y_edges), y) .- 1, 1, n_bins)
        @inbounds for i in eachindex(xi)
            counts[xi[i], yi[i]] += 1
        end

        hm = heatmap!(ax, x_edges, y_edges, log10.(counts .+ 1); colormap = _PHASE_CMAP)
        xlims!(ax, x_min, x_max); ylims!(ax, y_min, y_max)
        Colorbar(fig[1, 2], hm; label = colorbar_label, ticklabelsize = 18)
        colgap!(fig.layout, 6)

        show_isotherms && _plot_isotherms!(ax, x, T_values, ["", ""], [:royalblue, :firebrick])

        if show_Tequ
            nequ = logindgen(1000, max(1e-3, 10.0^x_min), 10.0^x_max)
            Pequ = nequ .* tequilibrium.(nequ)
            lines!(ax, log10.(nequ), log10.(Pequ); linestyle = :dot, linewidth = 4,
                   color = :black, label = "Thermal equilibrium")
        end

        leg = axislegend(ax; position = :rb); leg.framevisible = false
        savepath !== nothing && save(savepath, fig)
        return fig
    end
end
