using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 3
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = 10

DFiP = DistFix{bits + 11, bits}     # wide integer range for alpha*x + beta
xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]                  # integer predictors (so alpha*x = sum of copies)
ys = [3.0, 5.0, 7.0, 22.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0]

scale_int(g, k) = foldl(+, fill(g, k))

t = @timed_infer expectation(@dice begin
    alpha = bitblast(DFiP, Normal(0, 10.0), pieces, -32.0, 32.0)
    beta  = bitblast(DFiP, Normal(0, 10.0), pieces, -32.0, 32.0)
    pi    = uniform(DFiP, 0.0, 1.0)
    for i in 1:N
        o = parametrised_flip(pi)
        mu = scale_int(alpha, xs[i]) + beta
        noise = ifelse(o, bitblast(DFiP, Normal(0, 10.0), pieces, -32.0, 32.0),
                          bitblast(DFiP, Normal(0, 0.5), pieces, -4.0, 4.0))
        observe(prob_equals(noise + mu, DFiP(ys[i])))
    end
    alpha
end)

println("RESULT: ", t.value)
@show bits, pieces, N, t.value, t.time
