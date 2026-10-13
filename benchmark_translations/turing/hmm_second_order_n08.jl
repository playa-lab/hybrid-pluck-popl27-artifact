using Turing
using Statistics
using Printf
const OBS = [3.63, 0.12, 4.05, 2.22, 2.74, -0.0, 1.85, 0.77]
const INIT  = [0.5, 0.3, 0.2]
const TRANS = [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

@model function hmm_second_order(N, ys)
    mu = Vector{Float64}(undef, 3)
    mu[1] ~ Normal(-2.0, 1.0)
    mu[2] ~ Normal(0.0, 1.0)
    mu[3] ~ Normal(2.0, 1.0)
    z = Vector{Int}(undef, N)
    z2 = 0; z1 = 0
    for n in 1:N
        probs = n == 1 ? INIT : n == 2 ? TRANS[z1] : TRANS[max(z2, z1)]
        z[n] ~ Categorical(probs)
        ys[n] ~ Normal(mu[z[n]], 1.0)
        z2 = z1; z1 = z[n]
    end
end

function sorted_means(chain)
    m = [vec(chain[Symbol("mu[$k]")]) for k in 1:3]
    n = length(m[1]); acc = zeros(3)
    for i in 1:n
        acc .+= sort([m[1][i], m[2][i], m[3][i]])
    end
    return acc ./ n
end

function run_pg(N::Int, iters::Int)
    chain = sample(hmm_second_order(N, OBS[1:N]), PG(PARTICLES), iters; progress=false)
    return mean(vec(chain[Symbol("mu[1]")]))
end

function main()
    N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(2, 5)
    t0 = time(); val = run_pg(N, iters); dt = time() - t0
    println("hmm_second_order (Turing PG)  N=$N  particles=$PARTICLES  iters=$iters")
    @printf("RESULT: %.6f\n", val)
    @printf("inference_seconds: %.3f\n", dt)
end

main()
