using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 2
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = 2

DFiP = DistFix{bits + 5, bits}
obs = [-1.71, -1.47]

U(k) = DistUInt{2}(k)

transition(prev) =
    ifelse(prob_equals(prev, U(0)), discrete(DistUInt{2}, [0.6, 0.3, 0.1]),
      ifelse(prob_equals(prev, U(1)), discrete(DistUInt{2}, [0.2, 0.6, 0.2]),
             discrete(DistUInt{2}, [0.1, 0.3, 0.6])))

pick_prod(z, h, m) =
    ifelse(prob_equals(z, U(0)),
        ifelse(prob_equals(h, U(0)), m[1], ifelse(prob_equals(h, U(1)), m[2], m[3])),
      ifelse(prob_equals(z, U(1)),
        ifelse(prob_equals(h, U(0)), m[2], ifelse(prob_equals(h, U(1)), m[4], m[6])),
        ifelse(prob_equals(h, U(0)), m[3], ifelse(prob_equals(h, U(1)), m[6], m[9]))))

t = @timed_infer expectation(@dice begin
    m = [bitblast(DFiP, Normal(-3, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(-1.5, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(-0.5, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(1, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(2, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(3.5, 1), pieces, -8.0, 8.0)]
    z = discrete(DistUInt{2}, [0.5, 0.3, 0.2])
    h = discrete(DistUInt{2}, [0.5, 0.3, 0.2])
    for i in 1:N
        mu = pick_prod(z, h, m)
        gaussian_observe(DFiP, pieces, -8.0, 8.0, mu, 1.0, DFiP(obs[i]), add=true)
        z = transition(z)
        h = transition(h)
    end
    m[1]
end)

println("RESULT: ", t.value)
@show bits, pieces, N, t.value, t.time
