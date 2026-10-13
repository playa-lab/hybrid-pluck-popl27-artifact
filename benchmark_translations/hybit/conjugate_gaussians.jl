using Revise
using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools


if length(ARGS) == 0
    bits = 8
    pieces = 256
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end

DFiP = DistFix{8 + bits, bits}

data = DFiP.([1.0, 2.0])   # observations must match pluck/psi (posterior mean (0+1+2)/3 = 1.0)
add_arg = true

t = @timed_infer pr(@dice begin
                a = bitblast(DFiP, Normal(0, 1), pieces, -16.0, 16.0)
                for datapt in data
                    gaussian_observe(DFiP, pieces, -16.0, 16.0, a, 1.0, datapt, add=add_arg, exp=false)
                end
                a
            end)

p = sum(k*v for (k,v) in t.value)
println("RESULT: ", p)   # E[a] — uniform scalar for the error study
@show bits, pieces, p, t.time



