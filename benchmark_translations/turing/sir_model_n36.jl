using Turing
using Statistics
using Printf

const H = [0 0 0 0 0 1 1 1 0 1 0 0; 0 1 0 1 1 1 1 1 0 0 0 0; 0 1 1 1 0 1 0 0 0 1 1 1]   
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

@model function sir_model(H)
    R = size(H, 1); S = size(H, 2)
    rec ~ Beta(1.0, 1.0)   
    inf ~ Beta(1.0, 1.0)   
    bas ~ Beta(0.2, 1.0)   
    sym ~ Beta(1.0, 1.0)   
    z = Vector{Int}(undef, R * (S - 1))   
    zc = 0
    for r in 1:R
        prev = 0                     
        for i in 1:S
            if i == 1
                cur = 0
            else
                ptrans = prev == 1 ? (1.0 - rec) : inf   
                zc += 1
                z[zc] ~ Bernoulli(ptrans)
                cur = z[zc]
            end
            pobs = cur == 1 ? sym : bas
            H[r, i] ~ Bernoulli(pobs)
            prev = cur
        end
    end
end

function run_pg(iters::Int)
    chain = sample(sir_model(H), PG(PARTICLES), iters; progress=false)
    return (mean(chain[:rec]), mean(chain[:inf]), mean(chain[:bas]), mean(chain[:sym]))
end

function main()
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
    run_pg(5)  
    t0 = time(); (rec, inf, bas, sym) = run_pg(iters); dt = time() - t0
    println("sir_model (Turing PG)  rows=$(size(H,1))  steps=$(size(H,2))  particles=$PARTICLES  iters=$iters")
    @printf("posterior means: rec=%.3f inf=%.3f bas=%.3f sym=%.3f\n", rec, inf, bas, sym)
    @printf("RESULT: %.6f\n", inf)   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
