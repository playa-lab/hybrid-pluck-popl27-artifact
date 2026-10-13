using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools

if length(ARGS) == 0
    bits = 17
    pieces = 16
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end

t = @timed_infer pr(@dice begin
    DFiP = DistFix{bits + 5, bits}

            alice_skill = bitblast_exponential(DFiP, Normal(5, 1), pieces, -3.0, 13.0)
            bob_skill = bitblast_exponential(DFiP, Normal(5, 1), pieces, -3.0, 13.0)

            alice_skill = bitblast_exponential(DFiP, Normal(5, 1), pieces, -3.0, 13.0)
            bob_skill = bitblast_exponential(DFiP, Normal(5, 1), pieces, -3.0, 13.0)

            lazy_lag = DFiP(2.0)

            match = Vector(undef, 5)

            for i = 1:5
                match[i] = (ifelse(flip(1/3), alice_skill - lazy_lag, alice_skill) < (ifelse(flip(1/3), bob_skill - lazy_lag, bob_skill)))
            end

            observe(match[1])
            observe(!match[2])
            observe(match[3])
            observe(!match[4])
            match[5]
        end)

p = t.value

@show bits, pieces, p[1.0], t.time
