using Revise
using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools


if length(ARGS) == 0
  precision = 10
  num_pieces = 256
else
  precision = parse(Int64, ARGS[1])
  num_pieces = parse(Int64, ARGS[2])
end

DFiP = DistFix{4+precision, precision}

t = @timed_infer pr(@dice begin

  isWeekend = flip(2/7)
  hour = if isWeekend
            bitblast_exponential(DFiP, Normal(5, 4), num_pieces, 0.0, 8.0)
        else
            bitblast_exponential(DFiP, Normal(2, 4), num_pieces, 0.0, 8.0)
        end
  observe(hour == DFiP(6.0))
  isWeekend
end)

p = t.value

@show precision, num_pieces, p[1.0], t.time
