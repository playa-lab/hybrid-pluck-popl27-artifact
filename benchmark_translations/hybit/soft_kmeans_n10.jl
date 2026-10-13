using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 2
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = 10

DFiP = DistFix{bits + 7, bits}
o0 = [-3.0, 3.0, 3.1, -2.9, 2.9, 3.0, -3.1, 3.1, 2.9, -3.0]
o1 = [-2.9, 3.1, -3.0, -3.1, 3.0, -2.9, -3.0, 2.9, -3.1, -3.1]
o2 = [-3.1, 2.9, -2.9, -3.0, 3.1, -3.1, -2.9, 3.0, -3.0, -2.9]

U(k) = DistUInt{2}(k)
pick3(z, a, b, c) = ifelse(prob_equals(z, U(0)), a, ifelse(prob_equals(z, U(1)), b, c))

t = @timed_infer expectation(@dice begin
    m = [bitblast(DFiP, Normal(-3, 3.0), pieces, -32.0, 32.0),
         bitblast(DFiP, Normal(-3, 3.0), pieces, -32.0, 32.0),
         bitblast(DFiP, Normal(-3, 3.0), pieces, -32.0, 32.0),
         bitblast(DFiP, Normal(3, 3.0), pieces, -32.0, 32.0),
         bitblast(DFiP, Normal(3, 3.0), pieces, -32.0, 32.0),
         bitblast(DFiP, Normal(3, 3.0), pieces, -32.0, 32.0),
         bitblast(DFiP, Normal(3, 3.0), pieces, -32.0, 32.0),
         bitblast(DFiP, Normal(-3, 3.0), pieces, -32.0, 32.0),
         bitblast(DFiP, Normal(-3, 3.0), pieces, -32.0, 32.0)]
    for n in 1:N
        z = discrete(DistUInt{2}, [0.34, 0.33, 0.33])
        mu0 = pick3(z, m[1], m[4], m[7])
        mu1 = pick3(z, m[2], m[5], m[8])
        mu2 = pick3(z, m[3], m[6], m[9])
        gaussian_observe(DFiP, pieces, -32.0, 32.0, mu0, 1.0, DFiP(o0[n]), add=true)
        gaussian_observe(DFiP, pieces, -32.0, 32.0, mu1, 1.0, DFiP(o1[n]), add=true)
        gaussian_observe(DFiP, pieces, -32.0, 32.0, mu2, 1.0, DFiP(o2[n]), add=true)
    end
    m[1]
end)

println("RESULT: ", t.value)
@show bits, pieces, N, t.value, t.time
