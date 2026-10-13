using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 6
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = 64

DFiP = DistFix{bits + 2, bits}     # [0,1]: 2 int bits so p=1.0 is representable
# (participant idxs (1-based), observed yes-majority) per ballot.
ballots = [([1, 2, 3, 4, 5], true), ([3, 4, 5, 6, 7], false), ([5, 6, 7, 8, 9], false), ([7, 8, 9, 10, 11], false), ([9, 10, 11, 12, 13], false), ([11, 12, 13, 14, 15], true), ([13, 14, 15, 16, 17], false), ([15, 16, 17, 18, 19], false), ([17, 18, 19, 20, 21], false), ([19, 20, 21, 22, 23], true), ([21, 22, 23, 24, 25], true), ([23, 24, 25, 26, 27], false), ([25, 26, 27, 28, 29], false), ([27, 28, 29, 30, 31], true), ([29, 30, 31, 32, 33], true), ([31, 32, 33, 34, 35], true), ([33, 34, 35, 36, 37], false), ([35, 36, 37, 38, 39], true), ([37, 38, 39, 40, 41], true), ([39, 40, 41, 42, 43], true), ([41, 42, 43, 44, 45], true), ([43, 44, 45, 46, 47], false), ([45, 46, 47, 48, 49], false), ([47, 48, 49, 50, 51], false), ([49, 50, 51, 52, 53], false), ([51, 52, 53, 54, 55], false), ([53, 54, 55, 56, 57], false), ([55, 56, 57, 58, 59], true), ([57, 58, 59, 60, 61], true), ([59, 60, 61, 62, 63], false), ([61, 62, 63, 64, 1], false), ([63, 64, 1, 2, 3], false)]

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
