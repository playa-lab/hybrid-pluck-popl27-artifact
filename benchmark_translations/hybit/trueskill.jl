using Revise
using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools

if length(ARGS) == 0
    bits = 4
    pieces = 128
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end

t = @timed_infer pr(@dice begin
    DFiP = DistFix{bits + 6, bits}

            alice_skill = bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0)
            bob_skill = bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0)


            match = Vector(undef, 3)

            for i = 1:3
                match[i] = (bitblast(DFiP, Normal(0, 2), pieces, -16.0, 16.0) + alice_skill) < (bitblast(DFiP, Normal(0, 2), pieces, -16.0, 16.0) + bob_skill)
            end

            observe(match[1])
            observe(!match[2])
            match[3]
        end)

p = t.value

@show bits, pieces, p[1.0], t.time

        