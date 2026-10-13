using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 6
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = 128

DFiP = DistFix{bits + 2, bits}     # [0,1]: 2 int bits so p=1.0 is representable
# (participant idxs (1-based), observed yes-majority) per ballot.
ballots = [([1, 2, 3, 4, 5], true), ([3, 4, 5, 6, 7], false), ([5, 6, 7, 8, 9], false), ([7, 8, 9, 10, 11], false), ([9, 10, 11, 12, 13], false), ([11, 12, 13, 14, 15], false), ([13, 14, 15, 16, 17], true), ([15, 16, 17, 18, 19], true), ([17, 18, 19, 20, 21], true), ([19, 20, 21, 22, 23], false), ([21, 22, 23, 24, 25], false), ([23, 24, 25, 26, 27], true), ([25, 26, 27, 28, 29], false), ([27, 28, 29, 30, 31], false), ([29, 30, 31, 32, 33], false), ([31, 32, 33, 34, 35], true), ([33, 34, 35, 36, 37], false), ([35, 36, 37, 38, 39], true), ([37, 38, 39, 40, 41], true), ([39, 40, 41, 42, 43], false), ([41, 42, 43, 44, 45], false), ([43, 44, 45, 46, 47], false), ([45, 46, 47, 48, 49], true), ([47, 48, 49, 50, 51], true), ([49, 50, 51, 52, 53], false), ([51, 52, 53, 54, 55], true), ([53, 54, 55, 56, 57], true), ([55, 56, 57, 58, 59], true), ([57, 58, 59, 60, 61], true), ([59, 60, 61, 62, 63], true), ([61, 62, 63, 64, 65], true), ([63, 64, 65, 66, 67], true), ([65, 66, 67, 68, 69], false), ([67, 68, 69, 70, 71], false), ([69, 70, 71, 72, 73], true), ([71, 72, 73, 74, 75], true), ([73, 74, 75, 76, 77], true), ([75, 76, 77, 78, 79], false), ([77, 78, 79, 80, 81], true), ([79, 80, 81, 82, 83], true), ([81, 82, 83, 84, 85], false), ([83, 84, 85, 86, 87], true), ([85, 86, 87, 88, 89], true), ([87, 88, 89, 90, 91], false), ([89, 90, 91, 92, 93], true), ([91, 92, 93, 94, 95], true), ([93, 94, 95, 96, 97], false), ([95, 96, 97, 98, 99], false), ([97, 98, 99, 100, 101], true), ([99, 100, 101, 102, 103], false), ([101, 102, 103, 104, 105], false), ([103, 104, 105, 106, 107], true), ([105, 106, 107, 108, 109], true), ([107, 108, 109, 110, 111], true), ([109, 110, 111, 112, 113], true), ([111, 112, 113, 114, 115], false), ([113, 114, 115, 116, 117], true), ([115, 116, 117, 118, 119], true), ([117, 118, 119, 120, 121], true), ([119, 120, 121, 122, 123], false), ([121, 122, 123, 124, 125], false), ([123, 124, 125, 126, 127], true), ([125, 126, 127, 128, 1], false), ([127, 128, 1, 2, 3], true)]

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
