using Turing
using Statistics
using Printf

const PRECINCTS = [(1085,186,511), (625,180,348), (1119,202,536), (121,189,239), (95,318,337), (34,47,63), (372,262,355), (164,152,207), (1079,278,575), (272,59,146), (357,304,397), (492,251,406), (381,68,198), (112,50,81), (800,280,530), (716,272,496), (93,53,65), (825,291,575), (118,148,199), (270,147,219), (114,150,194), (47,53,67), (93,75,112)]
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

@model function king_ei(D, precincts)
    bw ~ Beta(1.0, 1.0)
    bb ~ Beta(1.0, 1.0)
    kw = Vector{Int}(undef, length(precincts))
    for i in 1:length(precincts)
        nw, nb, t = precincts[i]
        wc = nw ÷ D; bc = nb ÷ D
        tc = min(t ÷ D, wc + bc)
        kw[i] ~ Binomial(wc, bw)
        Turing.@addlogprob! logpdf(Binomial(bc, bb), tc - kw[i])
    end
end

function run_pg(D::Int, iters::Int)
    chain = sample(king_ei(D, PRECINCTS), PG(PARTICLES), iters; progress=false)
    return mean(chain[:bw]), mean(chain[:bb])
end

function main()
    D     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 40
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(80, 5)
    t0 = time(); bw, bb = run_pg(D, iters); dt = time() - t0
    println("king_ei (Turing PG)  data-divisor=$D  particles=$PARTICLES  iters=$iters")
    @printf("posterior means: E[bw]=%.3f (White Malloy rate, ~0.3)   E[bb]=%.3f (non-White, ~0.96)\n", bw, bb)
    @printf("RESULT: %.6f\n", bw)   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
