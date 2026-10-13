using Turing
using Statistics
using Printf


const OBS = [
 [-3.0, -2.9, -3.1],
 [3.0, 3.1, 2.9],
 [3.1, -3.0, -2.9],
 [-2.9, -3.1, -3.0],
 [2.9, 3.0, 3.1],
 [3.0, -2.9, -3.1],
 [-3.1, -3.0, -2.9],
 [3.1, 2.9, 3.0],
 [2.9, -3.1, -3.0],
 [-3.0, -3.1, -2.9],
]
const PRIOR_MU = [-3.0 -3.0 -3.0; 3.0 3.0 3.0; 3.0 -3.0 -3.0]   
const PRIOR_SD = 3.0
const MIX = [0.34, 0.33, 0.33]
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

@model function soft_kmeans(N, ys)
    mu = Matrix{Float64}(undef, 3, 3)
    for k in 1:3, d in 1:3
        mu[k, d] ~ Normal(PRIOR_MU[k, d], PRIOR_SD)
    end
    z = Vector{Int}(undef, N)
    for n in 1:N
        z[n] ~ Categorical(MIX)
        for d in 1:3
            ys[n][d] ~ Normal(mu[z[n], d], 1.0)
        end
    end
end

function run_pg(N::Int, iters::Int)
    chain = sample(soft_kmeans(N, OBS[1:N]), PG(PARTICLES), iters; progress=false)
    return mean(vec(chain[Symbol("mu[1,1]")]))
end

function main()
    N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 10
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(2, 5)
    t0 = time(); val = run_pg(N, iters); dt = time() - t0
    println("soft_kmeans (Turing PG)  N=$N  particles=$PARTICLES  iters=$iters")
    @printf("RESULT: %.6f\n", val)   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
