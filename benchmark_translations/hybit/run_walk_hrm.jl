using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools

if length(ARGS) == 0
    bits = 4
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end

DFiP = DistFix{9 + bits, bits}

readings = [82.0, 81.0, 80.0, 83.0, 130.0, 134.0, 130.0, 105.0]

t = @timed_infer pr(@dice begin
    base = bitblast(DFiP, Normal(70.0, 20.0), pieces, 0.0, 128.0)
    incr = bitblast(DFiP, Normal(50.0, 5.0), pieces, 0.0, 64.0)
    p_run  = uniform(DFiP, 0.0, 1.0)
    p_walk = uniform(DFiP, 0.0, 1.0)

    running = flip(0.5)
    for i = 1:length(readings)
        if running
            gaussian_observe(DFiP, pieces, -32.0, 32.0, base + incr, 5.0, DFiP(readings[i]), add=true)
            running = !parametrised_flip(p_walk)
        else
            gaussian_observe(DFiP, pieces, -32.0, 32.0, base, 5.0, DFiP(readings[i]), add=true)
            running = parametrised_flip(p_run)
        end
    end
    (running, base)
end)

dist = t.value

# Marginal over the final running state.
p_running = sum(v for (k, v) in dist if k[1] == true)
bh = sum(k[2] * v for (k, v) in dist)
println("RESULT: ", bh)
@show bits, pieces, p_running, t.time
