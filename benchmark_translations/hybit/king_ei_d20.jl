using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 6
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
D = 20

DFiP = DistFix{bits + 2, bits}     # [0,1]: 2 int bits so p=1.0 is representable; counts live in DistUInt
precincts = [(1085, 186, 511), (625, 180, 348), (1119, 202, 536), (121, 189, 239), (95, 318, 337), (34, 47, 63), (372, 262, 355), (164, 152, 207), (1079, 278, 575), (272, 59, 146), (357, 304, 397), (492, 251, 406), (381, 68, 198), (112, 50, 81), (800, 280, 530), (716, 272, 496), (93, 53, 65), (825, 291, 575), (118, 148, 199), (270, 147, 219), (114, 150, 194), (47, 53, 67), (93, 75, 112)]    # (Nw, Nb, T) per precinct -- REAL unscaled counts

idiv(x, d) = div(x, d)

# Binomial(n, p): count `true`s among n flips of a shared bit-blasted p.
function count_flips(p, n)
    c = DistUInt{16}(0)
    for _ in 1:n
        c = ifelse(parametrised_flip(p), c + DistUInt{16}(1), c)
    end
    c
end

t = @timed_infer expectation(@dice begin
    bw = uniform(DFiP, 0.0, 1.0)     # Beta(1,1)
    bb = uniform(DFiP, 0.0, 1.0)     # Beta(1,1)
    for pr in precincts
        wc = idiv(pr[1], D)
        bc = idiv(pr[2], D)
        tc = min(idiv(pr[3], D), wc + bc)
        total = count_flips(bw, wc) + count_flips(bb, bc)
        observe(prob_equals(total, DistUInt{16}(tc)))
    end
    bw
end)

println("RESULT: ", t.value)   # E[bw] — White Malloy rate, the cross-PPL error scalar
@show bits, pieces, D, t.value, t.time
