using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits   = 2
    pieces = 32
else
    bits   = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = 40

F = bits
W = 5 + F
DFiP = DistFix{W, F}
stop = 8.0

counts = [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5, 2, 2, 3, 4, 2, 1, 3, 2]

U(x) = DistUInt{2}(x)
b2u(c) = ifelse(c, U(1), U(0))

function poisson_observe(rate::DistFix{W,F}, c::Int, stop::Float64) where {W,F}
    step = 1/2^F; r = 0.0
    while r <= stop + 1e-9
        w = r > 0 ? pdf(Poisson(r), c) : (c == 0 ? 1.0 : 0.0)
        a = prob_equals(rate, DistFix{W,F}(r))
        observe(!a | (a & flip(w)))
        r += step
    end
end

function place_flip(kleft, rem)
    expr = flip(0.0)
    for kl in 1:min(3, rem)
        expr = ifelse(prob_equals(kleft, U(kl)), flip(kl / rem), expr)
    end
    expr
end

t = @timed_infer pr(@dice begin
    h1 = bitblast(DFiP, Gamma(2.0,1.0), pieces, 0.0, stop)
    h2 = bitblast(DFiP, Gamma(2.0,1.0), pieces, 0.0, stop)
    h3 = bitblast(DFiP, Gamma(2.0,1.0), pieces, 0.0, stop)
    h4 = bitblast(DFiP, Gamma(2.0,1.0), pieces, 0.0, stop)
    k     = discrete(DistUInt{2}, [1.0/2.75, 1.0/2.75, 0.5/2.75, 0.25/2.75])
    kleft = k
    seg   = U(0)
    poisson_observe(h1, counts[1], stop)
    for i in 2:N
        rem = N - i + 1
        cut = place_flip(kleft, rem)
        kleft = kleft - b2u(cut)
        seg   = seg + b2u(cut)
        rate = ifelse(prob_equals(seg, U(0)), h1,
               ifelse(prob_equals(seg, U(1)), h2,
               ifelse(prob_equals(seg, U(2)), h3, h4)))
        poisson_observe(rate, counts[i], stop)
    end
    k
end)

p = sum(k*v for (k,v) in t.value)
println("RESULT: ", p)
@show bits, pieces, N, t.value, t.time
