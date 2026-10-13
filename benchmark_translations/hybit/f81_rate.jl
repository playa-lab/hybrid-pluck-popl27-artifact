using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 5
    pieces = 8
    N = 5
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
    N = length(ARGS) >= 3 ? parse(Int64, ARGS[3]) : 5
end

DFiP = DistFix{bits + 2, bits}

# alignment tip columns (t0..t3), one entry per site, as the F81 nucleotide ids
t0 = [0,1,2,1,0,2,3,3,0,1,0,1,3,2,0]
t1 = [0,1,2,3,1,2,2,3,0,3,0,1,3,2,0]
t2 = [0,3,2,1,3,2,3,3,0,1,1,1,3,2,0]
t3 = [0,3,2,1,3,3,3,1,0,1,3,1,2,2,0]

draw_pi() = discrete(DistUInt{2}, [0.3, 0.2, 0.2, 0.3])

t = @timed_infer expectation(@dice begin
    p = uniform(DFiP, 0.0, 1.0)                 # Beta(1,1) prior on retention prob
    evolve(parent) = ifelse(parametrised_flip(p), parent, draw_pi())
    for s in 1:N
        root = draw_pi()
        L = evolve(root)
        M = evolve(root)
        observe(prob_equals(evolve(L), DistUInt{2}(t0[s])))
        observe(prob_equals(evolve(L), DistUInt{2}(t1[s])))
        observe(prob_equals(evolve(M), DistUInt{2}(t2[s])))
        observe(prob_equals(evolve(M), DistUInt{2}(t3[s])))
    end
    p
end)

println("RESULT: ", t.value)   # uniform scalar for the cross-PPL error study: E[p]
@show bits, pieces, N, t.value, t.time
