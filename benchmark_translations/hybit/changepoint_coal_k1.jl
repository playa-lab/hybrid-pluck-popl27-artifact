using Revise
using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools

if length(ARGS) == 0
    bits   = 3        # fractional bits F (rate grid step = 1/2^F)
    pieces = 64       # gamma bit-blast pieces (must be <= (stop)*2^F)
    N      = 20       # number of years to fit (first N of 111)
else
    bits   = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
    N      = length(ARGS) >= 3 ? parse(Int64, ARGS[3]) : 20
end

F = bits
W = 5 + F                 
DFiP = DistFix{W, F}
stop = 8.0                
allcounts = [4,5,4,0,1,4,3,4,0,6,3,3,4,0,2,6,3,3,5,4]
counts = allcounts[1:N]

function poisson_observe(rate::DistFix{W,F}, c::Int, stop::Float64) where {W,F}
    step = 1/2^F
    r = 0.0
    while r <= stop + 1e-9
        w = r > 0 ? pdf(Poisson(r), c) : (c == 0 ? 1.0 : 0.0)
        a = prob_equals(rate, DistFix{W,F}(r))
        observe(!a | (a & flip(w)))
        r += step
    end
end

TB = max(1, ceil(Int, log2(N)))   # bits for the changepoint selector (0..N-2)

t = @timed_infer expectation(@dice begin
    e  = bitblast(DFiP, Gamma(2,1), pieces, 0.0, stop)   # early rate
    l  = bitblast(DFiP, Gamma(2,1), pieces, 0.0, stop)   # late rate
    cp = discrete(DistUInt{TB}, [1/(N-1) for _ in 1:(N-1)])   # changepoint after year cp+1
    for i in 1:N
        is_early = DistUInt{TB}(i-1) <= cp
        rate = ifelse(is_early, e, l)
        poisson_observe(rate, counts[i], stop)
    end
    e
end)

p = t.value

println("RESULT: ", p)   # E[e] — early-period disaster rate, the cross-PPL error scalar
@show bits, pieces, p, t.time

