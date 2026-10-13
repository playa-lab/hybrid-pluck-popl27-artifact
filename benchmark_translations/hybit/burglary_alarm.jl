using Revise
using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools


t = @timed_infer pr(@dice begin
    earthquake = flip(0.0001)
    burglary = flip(0.001)
    alarm = earthquake | burglary
    phoneWorking = ifelse(earthquake, flip(0.7), flip(0.99))
    maryWakes = ifelse(alarm, ifelse(earthquake, flip(0.8), flip(0.6)), flip(0.2))
    called = maryWakes & phoneWorking
    observe(called)
    burglary
end)

p = t.value[true]          # P(burglary = true | called)
println("RESULT: ", p)     # uniform scalar for the cross-PPL error study
@show p, t.time

