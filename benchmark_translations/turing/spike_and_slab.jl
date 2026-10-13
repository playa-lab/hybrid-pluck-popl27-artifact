using Turing
using Statistics
using Printf

const X = [[7.0, 26.0, 6.0, 60.0],
          [1.0, 29.0, 15.0, 52.0],
          [11.0, 56.0, 8.0, 20.0],
          [11.0, 31.0, 8.0, 47.0],
          [7.0, 52.0, 6.0, 33.0],
          [11.0, 55.0, 9.0, 22.0],
          [3.0, 71.0, 17.0, 6.0],
          [1.0, 31.0, 22.0, 44.0],
          [2.0, 54.0, 18.0, 22.0],
          [21.0, 47.0, 4.0, 26.0],
          [1.0, 40.0, 23.0, 34.0],
          [11.0, 66.0, 9.0, 12.0],
          [10.0, 68.0, 8.0, 12.0]]
const YS = [78.5, 74.3, 104.3, 87.6, 95.9, 109.2, 102.7, 72.5, 93.1, 115.9, 83.8, 113.3, 109.4]

const SLAB_SIGMA  = 10.0
const NOISE_SIGMA = 2.5
const NPRED = 4
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

@model function spike_and_slab(X, ys)
    n = length(ys)
    g = Vector{Int}(undef, NPRED)
    b = Vector{Float64}(undef, NPRED)
    for j in 1:NPRED
        g[j] ~ Bernoulli(0.5)
        b[j] ~ Normal(0.0, SLAB_SIGMA)
    end
    b0 ~ Normal(0.0, 100.0)
    for i in 1:n
        mu = b0
        for j in 1:NPRED
            mu += g[j] * b[j] * X[i][j]
        end
        ys[i] ~ Normal(mu, NOISE_SIGMA)
    end
end

function run_pg(iters::Int)
    chain = sample(spike_and_slab(X, YS), PG(PARTICLES), iters; progress=false)
    return [mean(vec(chain[Symbol("g[$j]")])) for j in 1:NPRED]
end

function main()
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
    run_pg(5)  

    t0 = time()
    incl = run_pg(iters)
    dt = time() - t0

    println("spike_and_slab (Turing PG)  particles=$PARTICLES  iters=$iters")
    println("per-predictor inclusion probability P(g_j = 1):")
    for j in 1:NPRED
        @printf("  x%d  P(include)=%.2f\n", j, incl[j])
    end
    @printf("RESULT: %.6f\n", incl[1])
    @printf("inference_seconds: %.3f\n", dt)
end

main()
