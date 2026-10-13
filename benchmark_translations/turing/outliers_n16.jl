using Turing
using Statistics
using Printf

const XS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0]
const YS = [3.0, 5.0, 7.0, 22.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 5.0, 25.0, 27.0, 29.0, 31.0, 33.0]
const PLANTED = [4, 11]
const SIGMA_INLIER  = 0.5
const SIGMA_OUTLIER = 10.0
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

@model function outliers(N, xs, ys)
    alpha ~ Normal(0.0, 10.0)
    beta  ~ Normal(0.0, 10.0)
    pi    ~ Beta(1.0, 1.0)
    o = Vector{Int}(undef, N)
    for n in 1:N
        o[n] ~ Bernoulli(pi)
        mu = beta + alpha * xs[n]
        sd = o[n] == 1 ? SIGMA_OUTLIER : SIGMA_INLIER
        ys[n] ~ Normal(mu, sd)
    end
end

function run_pg(N::Int, iters::Int)
    chain = sample(outliers(N, XS[1:N], YS[1:N]), PG(PARTICLES), iters; progress=false)
    a = mean(chain[:alpha]); b = mean(chain[:beta])
    po = [mean(vec(chain[Symbol("o[$n]")])) for n in 1:N]
    return a, b, po
end

function main()
    N     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 16
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(4, 5)
    t0 = time(); a, b, po = run_pg(N, iters); dt = time() - t0
    println("outliers (Turing PG)  N=$N  particles=$PARTICLES  iters=$iters")
    @printf("recovered line: alpha=% .2f  beta=% .2f   (true alpha=2, beta=1)\n", a, b)
    @printf("RESULT: %.6f\n", a)
    println("P(outlier) per point (* = planted):")
    for n in 1:N
        mark = n in PLANTED ? "*" : " "
        @printf("  x=%4.1f%s  P(outlier)=%.2f\n", XS[n], mark, po[n])
    end
    @printf("inference_seconds: %.3f\n", dt)
end

main()
