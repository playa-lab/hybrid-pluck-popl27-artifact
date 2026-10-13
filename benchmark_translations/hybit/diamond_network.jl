using Revise
using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools


t = @timed_infer pr(@dice begin
    s = true
    for i in 1:100
        route = flip(0.5)
        drop = flip(0.0001)
        s = ifelse(drop, ifelse(route, s, false), s)   # (!drop | route) & s
    end
    s
end)

p = t.value[true]
println("RESULT: ", p)   # P(message gets through) — the cross-PPL error scalar
@show p, t.time

