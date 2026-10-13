using Turing
using Statistics
using Printf

const COUNTS = [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4]
const ALPHA = 2.0
const SCALE = 1.0
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

@model function changepoint_k1(N, counts)
    nb = N - 1                                  
    e ~ Gamma(ALPHA, SCALE)                     
    l ~ Gamma(ALPHA, SCALE)                     
    tau ~ Categorical(fill(1.0 / nb, nb))       
    for i in 1:N
        rate = i <= tau ? e : l                 
        counts[i] ~ Poisson(rate)
    end
end

function run_pg(N::Int, iters::Int)
    chain = sample(changepoint_k1(N, COUNTS[1:N]), PG(PARTICLES), iters; progress=false)
    return mean(vec(chain[:e]))                 
end

function main()
    N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) :
            length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(5, 5)                                
    t0 = time(); em = run_pg(N, iters); dt = time() - t0
    println("changepoint_coal_k1 (Turing PG)  N=$N  particles=$PARTICLES  iters=$iters")
    @printf("posterior mean early rate E[e] = %.4f\n", em)
    @printf("RESULT: %.6f\n", em)               
    @printf("inference_seconds: %.3f\n", dt)
end

main()
