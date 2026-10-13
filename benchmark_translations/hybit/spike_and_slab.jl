using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 2
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end

# wide range: data*coef sums reach a few thousand
DFiP = DistFix{bits + 14, bits}

x1 = [7.0, 1.0, 11.0, 11.0, 7.0, 11.0, 3.0, 1.0, 2.0, 21.0, 1.0, 11.0, 10.0]
x2 = [26.0, 29.0, 56.0, 31.0, 52.0, 55.0, 71.0, 31.0, 54.0, 47.0, 40.0, 66.0, 68.0]
x3 = [6.0, 15.0, 8.0, 8.0, 6.0, 9.0, 17.0, 22.0, 18.0, 4.0, 23.0, 9.0, 8.0]
x4 = [60.0, 52.0, 20.0, 47.0, 33.0, 22.0, 6.0, 44.0, 22.0, 26.0, 34.0, 12.0, 12.0]
ys = [78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7, 72.5, 93.1, 115.9, 83.8, 113.3, 109.4]

obs_pieces = 256
t = @timed_infer pr((@dice begin
    g1 = flip(0.5); g2 = flip(0.5); g3 = flip(0.5); g4 = flip(0.5)
    slab() = bitblast(DFiP, Normal(0, 10), pieces, -32.0, 32.0)
    b1 = ifelse(g1, slab(), DFiP(0.0))
    b2 = ifelse(g2, slab(), DFiP(0.0))
    b3 = ifelse(g3, slab(), DFiP(0.0))
    b4 = ifelse(g4, slab(), DFiP(0.0))
    b0 = bitblast(DFiP, Normal(0, 100), pieces, -512.0, 512.0)
    for n in 1:13
        mu = b0 + b1 * DFiP(x1[n]) + b2 * DFiP(x2[n]) + b3 * DFiP(x3[n]) + b4 * DFiP(x4[n])
        gaussian_observe(DFiP, obs_pieces, -128.0, 128.0, mu, 2.5, DFiP(ys[n]), add=true)
    end
    g1
end), ignore_errors=true)

println("RESULT: ", t.value[true])
@show bits, pieces, t.value, t.time
