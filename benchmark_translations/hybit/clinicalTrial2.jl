using Revise
using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools


if length(ARGS) == 0
    bits = 6
else
    bits = parse(Int64, ARGS[1])
end

DFiP = DistFix{2+bits, bits}

controlGroup = [false, false, true, false, false]
treatedGroup = [true, false, true, true, true]

t = @timed_infer expectation(@dice begin
            isEffective = flip(0.5)
            probIfTreated = uniform(DFiP, 0.0, 1.0)
            probIfControl = if isEffective uniform(DFiP, 0.0, 1.0) else probIfTreated end

            for i=1:5
                controlFlip = parametrised_flip(probIfControl)
                observe(controlFlip == controlGroup[i])
                treatedFlip = parametrised_flip(probIfTreated)
                observe(treatedFlip == treatedGroup[i])
            end

            observe(isEffective)
            probIfControl
end)


p = t.value
println("RESULT: ", p)   # E[probIfControl] — uniform scalar for the error study

@show bits, p, t.time
