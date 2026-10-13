using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits   = 2        # fractional bits F (rate grid step = 1/2^F)
    pieces = 32       # gamma bit-blast pieces (must be <= (stop)*2^F)
else
    bits   = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = 25

F = bits
W = 5 + F                 
DFiP = DistFix{W, F}
stop = 8.0

counts = [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4, 5, 3, 1, 4, 4]

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
    expr = flip(0.0)                  # kleft==0 -> never cut
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
    k     = discrete(DistUInt{2}, [1.0/2.75, 1.0/2.75, 0.5/2.75, 0.25/2.75])  # trunc Poisson(1)
    kleft = k
    seg   = U(0)
    poisson_observe(h1, counts[1], stop)        # year 1 is always in segment 0
    for i in 2:N
        rem = N - i + 1                          # boundaries remaining (incl. this one)
        cut = place_flip(kleft, rem)             # changepoint at boundary (i-1)?
        kleft = kleft - b2u(cut)
        seg   = seg + b2u(cut)
        rate = ifelse(prob_equals(seg, U(0)), h1,
               ifelse(prob_equals(seg, U(1)), h2,
               ifelse(prob_equals(seg, U(2)), h3, h4)))
        poisson_observe(rate, counts[i], stop)
    end
    k
end)

p = sum(k*v for (k,v) in t.value)   # E[k] -- the cross-PPL error scalar
println("RESULT: ", p)
@show bits, pieces, N, t.value, t.time
