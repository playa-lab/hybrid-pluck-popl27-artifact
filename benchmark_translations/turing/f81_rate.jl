using Turing
using Statistics
using Printf


const ALIGN = [
 [1,1,1,1],[2,2,4,4],[3,3,3,3],[2,4,2,2],[1,2,4,4],[3,3,3,4],[4,3,4,4],[4,4,4,2],
 [1,1,1,1],[2,4,2,2],[1,1,2,4],[2,2,2,2],[4,4,4,3],[3,3,3,3],[1,1,1,1],
]
const PI = [0.3, 0.2, 0.2, 0.3]    
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))
function evolve_dist(parent::Int, p)
    v = (1.0 - p) .* PI
    v[parent] += p
    return v ./ sum(v)
end

@model function f81_rate(nsites, align)
    p ~ Beta(1.0, 1.0)
    root = Vector{Int}(undef, nsites)
    L    = Vector{Int}(undef, nsites)
    M    = Vector{Int}(undef, nsites)
    for s in 1:nsites
        root[s] ~ Categorical(PI)
        L[s] ~ Categorical(evolve_dist(root[s], p))
        M[s] ~ Categorical(evolve_dist(root[s], p))
        col = align[s]
        col[1] ~ Categorical(evolve_dist(L[s], p))
        col[2] ~ Categorical(evolve_dist(L[s], p))
        col[3] ~ Categorical(evolve_dist(M[s], p))
        col[4] ~ Categorical(evolve_dist(M[s], p))
    end
end

function run_pg(nsites::Int, iters::Int)
    chain = sample(f81_rate(nsites, ALIGN[1:nsites]), PG(PARTICLES), iters; progress=false)
    return mean(chain[:p])
end

function main()
    nsites = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : length(ALIGN)
    iters  = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(3, 5)  

    t0 = time()
    p = run_pg(nsites, iters)
    dt = time() - t0

    println("f81_rate / F81 (Turing PG)  n_sites=$nsites  particles=$PARTICLES  iters=$iters")
    @printf("retention prob p = %.3f   (true 0.8)\n", p)
    @printf("RESULT: %.6f\n", p)   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
