using Turing
using Statistics
using Printf

const NAMES = ["T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7"]
const TRUE_IDS = [2, 0, 1, 2, 0, 1, 2, 0]   
const GAMES = [
 [0, 1, 1, 1],
 [0, 2, 1, 4],
 [0, 3, 3, 4],
 [0, 4, 2, 1],
 [0, 5, 0, 1],
 [0, 6, 8, 1],
 [0, 7, 3, 2],
 [1, 0, 0, 0],
 [1, 2, 2, 0],
 [1, 3, 1, 1],
 [1, 4, 0, 0],
 [1, 5, 1, 1],
 [1, 6, 1, 2],
 [1, 7, 0, 0],
 [2, 0, 4, 0],
 [2, 1, 2, 0],
 [2, 3, 2, 0],
 [2, 4, 2, 1],
 [2, 5, 1, 0],
 [2, 6, 0, 1],
 [2, 7, 1, 1],
 [3, 0, 6, 2],
 [3, 1, 1, 0],
 [3, 2, 4, 1],
 [3, 4, 2, 0],
 [3, 5, 2, 2],
 [3, 6, 2, 3],
 [3, 7, 3, 1],
 [4, 0, 3, 2],
 [4, 1, 0, 0],
 [4, 2, 1, 1],
 [4, 3, 0, 2],
 [4, 5, 3, 0],
 [4, 6, 0, 0],
 [4, 7, 1, 1],
 [5, 0, 1, 1],
 [5, 1, 3, 1],
 [5, 2, 0, 1],
 [5, 3, 3, 2],
 [5, 4, 1, 1],
 [5, 6, 3, 1],
 [5, 7, 0, 1],
 [6, 0, 4, 3],
 [6, 1, 2, 1],
 [6, 2, 5, 0],
 [6, 3, 5, 3],
 [6, 4, 2, 2],
 [6, 5, 2, 1],
 [6, 7, 0, 4],
 [7, 0, 2, 1],
 [7, 1, 1, 2],
 [7, 2, 0, 0],
 [7, 3, 3, 0],
 [7, 4, 0, 1],
 [7, 5, 1, 1],
 [7, 6, 1, 0],
]
const SCALES = [4.0/9.0, 2.0/3.0, 1.0, 1.5, 2.25, 3.375]
const ID_NAMES = ["defend", "balanced", "attack"]
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

function games_for(T)
    g = [row for row in GAMES if row[1] < T && row[2] < T]
    return [row[1]+1 for row in g], [row[2]+1 for row in g], [row[3] for row in g], [row[4] for row in g]
end

@model function soccer(T, hi, aj, gh, ga)
    theta ~ Dirichlet(ones(3))
    ids   = Vector{Int}(undef, T)
    rates = Vector{Float64}(undef, T)
    for t in 1:T
        ids[t]   ~ Categorical(theta)
        rates[t] ~ Gamma(4.0, 0.25)
    end
    for m in 1:length(hi)
        p = (ids[hi[m]] - 1) + (ids[aj[m]] - 1)
        gh[m] ~ Poisson(rates[hi[m]] * SCALES[p + 2])
        ga[m] ~ Poisson(rates[aj[m]] * SCALES[p + 1])
    end
end

function run_pg(T::Int, iters::Int)
    hi, aj, gh, ga = games_for(T)
    chain = sample(soccer(T, hi, aj, gh, ga), PG(PARTICLES), iters; progress=false)
    probs = zeros(T, 3)
    for t in 1:T
        s = vec(chain[Symbol("ids[$t]")])
        for k in 1:3
            probs[t, k] = mean(s .== k)
        end
    end
    return probs
end

function main()
    T     = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(3, 5)
    t0 = time(); probs = run_pg(T, iters); dt = time() - t0
    @printf("RESULT: %.6f\n", probs[1, 3])
    println("soccer (Turing PG)  T=$T  particles=$PARTICLES  iters=$iters")
    println("per-team identity posterior [defend, balanced, attack]  (MAP vs true):")
    for t in 1:T
        mapid = argmax(probs[t, :]) - 1
        @printf("  %s  [%.2f, %.2f, %.2f]  MAP=%-8s true=%s\n",
                NAMES[t], probs[t, 1], probs[t, 2], probs[t, 3],
                ID_NAMES[mapid + 1], ID_NAMES[TRUE_IDS[t] + 1])
    end
    @printf("inference_seconds: %.3f\n", dt)
end

main()
