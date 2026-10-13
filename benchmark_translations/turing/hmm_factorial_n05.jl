using Turing
using Statistics
using Printf

const OBS = [-1.78, -1.98, -0.99, -1.76, 0.64]
const INIT  = [0.5, 0.3, 0.2]
const TRANS = [[0.6, 0.3, 0.1], [0.2, 0.6, 0.2], [0.1, 0.3, 0.6]]
const TRUE_MU = [-3.0, -1.5, -0.5, 1.0, 0.0, 2.0, 0.0, 0.0, 3.5]
const REACHABLE = [0, 1, 2, 3, 5, 8]
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

pick_mu_idx(z::Int, h::Int) =
    z == 0 ? (h == 0 ? 0 : h == 1 ? 1 : 2) :
    z == 1 ? (h == 0 ? 1 : h == 1 ? 3 : 5) :
             (h == 0 ? 2 : h == 1 ? 5 : 8)

@model function hmm_factorial(N, ys)
    mu = Vector{Float64}(undef, 9)
    mu[1] ~ Normal(-3.0, 1.0)
    mu[2] ~ Normal(-1.5, 1.0)
    mu[3] ~ Normal(-0.5, 1.0)
    mu[4] ~ Normal(1.0, 1.0)
    mu[5] ~ Normal(0.0, 1.0)
    mu[6] ~ Normal(2.0, 1.0)
    mu[7] ~ Normal(0.0, 1.0)
    mu[8] ~ Normal(0.0, 1.0)
    mu[9] ~ Normal(3.5, 1.0)
    z = Vector{Int}(undef, N); h = Vector{Int}(undef, N)
    zp = 0; hp = 0
    for n in 1:N
        zprobs = n == 1 ? INIT : TRANS[zp]
        hprobs = n == 1 ? INIT : TRANS[hp]
        z[n] ~ Categorical(zprobs)
        h[n] ~ Categorical(hprobs)
        idx = pick_mu_idx(z[n] - 1, h[n] - 1) + 1
        ys[n] ~ Normal(mu[idx], 1.0)
        zp = z[n]; hp = h[n]
    end
end

function run_pg(N::Int, iters::Int)
    chain = sample(hmm_factorial(N, OBS[1:N]), PG(PARTICLES), iters; progress=false)
    return mean(vec(chain[Symbol("mu[1]")]))
end

function main()
    N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 5
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(2, 5)
    t0 = time(); val = run_pg(N, iters); dt = time() - t0
    println("hmm_factorial (Turing PG)  N=$N  particles=$PARTICLES  iters=$iters")
    @printf("RESULT: %.6f\n", val)
    @printf("inference_seconds: %.3f\n", dt)
end

main()
