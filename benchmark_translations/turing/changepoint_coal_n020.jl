using Turing
using Statistics
using Printf

const COUNTS = [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6, 3, 3, 5, 4]
const ALPHA = 2.0
const SCALE = 1.0
const K_PRIOR = [1.0, 1.0, 0.5, 0.25]
const KMAX = 3
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

@model function changepoint(N, counts)
    nb = N - 1
    k ~ Categorical(K_PRIOR ./ sum(K_PRIOR))
    flags = Vector{Int}(undef, nb)
    kleft = k - 1
    for i in 1:nb
        remaining = nb - i + 1
        pp = remaining > 0 ? clamp(kleft / remaining, 0.0, 1.0) : 0.0
        flags[i] ~ Bernoulli(pp)
        kleft -= flags[i]
    end
    h = Vector{Float64}(undef, KMAX + 1)
    for s in 1:(KMAX + 1)
        h[s] ~ Gamma(ALPHA, SCALE)
    end
    seg = 1
    for g in 1:N
        counts[g] ~ Poisson(h[min(seg, KMAX + 1)])
        if g < N && flags[g] == 1
            seg += 1
        end
    end
end

function run_pg(N::Int, iters::Int)
    chain = sample(changepoint(N, COUNTS[1:N]), PG(PARTICLES), iters; progress=false)
    kv = vec(chain[:k]) .- 1
    return [mean(kv .== k) for k in 0:KMAX]
end

function main()
    N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(5, 5)
    t0 = time(); kp = run_pg(N, iters); dt = time() - t0
    println("changepoint_coal (Turing PG)  N=$N  particles=$PARTICLES  iters=$iters")
    println("posterior over k (number of changepoints):")
    for k in 0:KMAX
        @printf("  k=%d  P=%.3f\n", k, kp[k + 1])
    end
    @printf("RESULT: %.6f\n", sum((0:KMAX) .* kp))   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
