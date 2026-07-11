"""
Startup banner for the interactive workflow, matching the MOOSE look-and-feel.
"""

function print_logo()
    cols = displaysize(stdout)[2]
    rainbow = [:red, :yellow, :green, :cyan, :blue, :magenta]

    if cols < 60
        colors = Iterators.cycle(rainbow)
        state = iterate(colors)
        for c in collect("SHINE")
            print(Crayon(foreground = state[1], bold = true)(string(c)))
            state = iterate(colors, state[2])
        end
        println()
        println(Crayon(foreground = :light_green, bold = true)("HI 21-cm Data Tool -- dev. by Jack Berat"))
        return
    end

    println()
    logo_text = raw"""
  ____    _   _   ___   _   _   _____
 / ___|  | | | | |_ _| | \ | | | ____|
 \___ \  | |_| |  | |  |  \| | |  _|
  ___) | |  _  |  | |  | |\  | | |___
 |____/  |_| |_| |___| |_| \_| |_____|
"""
    logo_lines = filter(line -> !isempty(strip(line)), split(logo_text, "\n"))
    max_len = maximum(length.(logo_lines))
    pad_left = max(0, (cols - max_len) ÷ 2)
    border = repeat("─", max_len)

    println(" "^pad_left * "╭" * border * "╮")
    colors = Iterators.cycle(rainbow)
    color_state = iterate(colors)
    for line in logo_lines
        print(" "^pad_left * "│")
        for c in collect(rpad(line, max_len))
            print(Crayon(foreground = color_state[1], bold = true)(string(c)))
            color_state = iterate(colors, color_state[2])
        end
        println("│")
    end
    println(" "^pad_left * "╰" * border * "╯")

    # A little 21-cm sparkle.
    sparkle = [
        (:light_yellow, raw"      .  *  ."),
        (:white,        raw"    *  ((*))  *"),
        (:light_cyan,   raw"      '  *  '"),
    ]
    for (color, line) in sparkle
        println(Crayon(foreground = color)(line))
    end
    println(Crayon(foreground = :light_green, bold = true)("Synthetic H I Neutral Emission -- dev. by Jack Berat"))
    println(Crayon(foreground = :light_red, bold = true)("Version $(shine_version())"))
end
