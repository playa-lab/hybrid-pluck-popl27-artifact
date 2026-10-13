using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))


if length(ARGS) == 0
    bits = 2
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = 6

DFiP = DistFix{bits + 5, bits}
obs = [-2.6, -1.39, -1.93, 0.56, 0.84, 0.9]

U(k) = DistUInt{2}(k)

transition(prev) =
    ifelse(prob_equals(prev, U(0)), discrete(DistUInt{2}, [0.6, 0.3, 0.1]),
      ifelse(prob_equals(prev, U(1)), discrete(DistUInt{2}, [0.2, 0.6, 0.2]),
             discrete(DistUInt{2}, [0.1, 0.3, 0.6])))

pick(z, m1, m2, m3) =
    ifelse(prob_equals(z, U(0)), m1, ifelse(prob_equals(z, U(1)), m2, m3))

# native max of two DistUInt states (0..2): max(a,b) = ifelse(a>=b, a, b)
nmax(a, b) = ifelse(a >= b, a, b)

t = @timed_infer expectation(@dice begin
    m1 = bitblast(DFiP, Normal(-2, 1), pieces, -8.0, 8.0)
    m2 = bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0)
    m3 = bitblast(DFiP, Normal(2, 1), pieces, -8.0, 8.0)
    z1 = discrete(DistUInt{2}, [0.5, 0.3, 0.2])
    z2 = transition(z1)
    states = [z1, z2]
    for i in 1:N
        if i <= 2
            z = states[i]
        else
            z = transition(nmax(states[i-2], states[i-1]))
            push!(states, z)
        end
        mu = pick(z, m1, m2, m3)
        gaussian_observe(DFiP, pieces, -8.0, 8.0, mu, 1.0, DFiP(obs[i]), add=true)
    end
    m1
end)

println("RESULT: ", t.value)
@show bits, pieces, N, t.value, t.time
