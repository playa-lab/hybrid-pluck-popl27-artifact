using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 6
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = 8

DFiP = DistFix{bits + 2, bits}     # [0,1]: 2 int bits so p=1.0 is representable
# (participant idxs (1-based), observed yes-majority) per ballot.
ballots = [([1, 2, 3, 4, 5], true), ([3, 4, 5, 6, 7], true), ([5, 6, 7, 8, 1], false), ([7, 8, 1, 2, 3], false)]

# yes-tally among the members in idxs: each casts an independent Bernoulli(p_m).
function count_yes(ps, idxs)
    c = DistUInt{8}(0)
    for i in idxs
        c = ifelse(parametrised_flip(ps[i]), c + DistUInt{8}(1), c)
    end
    c
end

t = @timed_infer expectation(@dice begin
    ps = [uniform(DFiP, 0.0, 1.0) for _ in 1:N]   # Beta(1,1) per member
    for bal in ballots
        idxs = bal[1]; outcome = bal[2]
        yes = count_yes(ps, idxs)
        # strict yes-majority: 2*yes > |subset|
        maj = DistUInt{8}(2) * yes > DistUInt{8}(length(idxs))
        observe(maj == outcome)
    end
    ps[1]
end)

@show bits, pieces, N, t.value, t.time
