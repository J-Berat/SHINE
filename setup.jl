#!/usr/bin/env julia
"""
setup.jl

Prepare the SHINE environment: activate the project, install dependencies, and
precompile. Pass `--test` to also run the package tests.

Usage (disable personal startup files that import extra packages):
  julia --startup-file=no setup.jl
  julia --startup-file=no setup.jl --test
"""

using Pkg

function main(; run_tests::Bool = false)
    project_dir = @__DIR__
    println("Activating project at $(abspath(project_dir))…")
    Pkg.activate(project_dir)

    println("Instantiating dependencies (first run may take a while)…")
    Pkg.instantiate()

    println("Precompiling project…")
    Pkg.precompile()

    if run_tests
        println("Running test suite…")
        Pkg.test()
    else
        println("Skipping tests (pass --test to run them).")
    end

    println("SHINE is ready. Launch it with `using Shine; run_shine()`.")
end

main(run_tests = "--test" in ARGS)
