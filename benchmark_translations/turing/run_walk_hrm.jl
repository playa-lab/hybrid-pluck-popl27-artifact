

using Turing
using Statistics
using Printf

const OBS = [82.0, 81.0, 80.0, 83.0, 120.0, 124.0, 120.0, 85.0]
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

@model function run_walk_hrm(obs)
    N = length(obs)
    start_rate ~ Beta(1.0, 1.0)
    stop_rate  ~ Beta(1.0, 1.0)
    base_hr ~ Normal(70.0, 20.0)
    incr    ~ Normal(40.0, 20.0)
    s = Vector{Int}(undef, N)   
    running = false             
    for t in 1:N
        mu = running ? base_hr + incr : base_hr
        obs[t] ~ Normal(mu, 5.0)
        p_next = running ? 1.0 - stop_rate : start_rate   
        s[t] ~ Bernoulli(p_next)
        running = s[t] == 1
    end
end

function run_pg(iters::Int)
    chain = sample(run_walk_hrm(OBS), PG(PARTICLES), iters; progress=false)
    sr = mean(chain[:start_rate]); tr = mean(chain[:stop_rate])
    bh = mean(chain[:base_hr]);    ic = mean(chain[:incr])
    return sr, tr, bh, ic
end

function main()
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
    run_pg(5)  

    t0 = time()
    sr, tr, bh, ic = run_pg(iters)
    dt = time() - t0

    println("run_walk_hrm (Turing PG)  particles=$PARTICLES  iters=$iters")
    @printf("start_rate=%.3f  stop_rate=%.3f  base_hr=%.1f  run_hr=%.1f (base+incr)\n",
            sr, tr, bh, bh + ic)
    println("  (data: walking ~80 bpm, running ~120 bpm)")
    @printf("RESULT: %.6f\n", bh)
    @printf("inference_seconds: %.3f\n", dt)
end

main()
