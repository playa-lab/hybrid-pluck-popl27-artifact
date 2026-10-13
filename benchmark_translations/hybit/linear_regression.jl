using Revise
using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools


if length(ARGS) == 0
    bits = 3
    pieces = 64                           # prior bit-blast pieces
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end

obs_pieces = 256                          # 128/256 = 0.5 spacing over [-64,64]

DFiP = DistFix{bits + 10, bits}          # range +/- 512
PLO, PHI = -32.0, 32.0                    # prior bit-blast range
OLO, OHI = -64.0, 64.0                    # observation grid range (covers the y data)

xs = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
ys = [4.855909670422071, 7.827096399668481, 10.888684138432337, 14.701983725098863, 16.872411716217112, 18.502646585659043, 23.332318344067716, 25.73266252150283, 28.783041315854806, 32.11588478670085]

t = @timed_infer pr(@dice begin
    a = bitblast(DFiP, Normal(0, 10), pieces, PLO, PHI)
    b = bitblast(DFiP, Normal(0, 10), pieces, PLO, PHI)
    for i = 1:length(xs)
        mu = a + b * DFiP(xs[i])          # b * x_i via DistFix*DistFix (b shared)
        gaussian_observe(DFiP, obs_pieces, OLO, OHI, mu, 1.0, DFiP(ys[i]), add=true)
    end
    a
end)

p = sum(k*v for (k,v) in t.value)         # E[a]  (posterior concentrates near 2.0)
println("RESULT: ", p)
@show bits, pieces, p, t.time

