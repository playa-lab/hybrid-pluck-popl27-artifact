using Turing
using Statistics
using Printf


const ALIGN = [
 [1,1,1,1],[2,2,4,4],[3,3,3,3],[2,4,2,2],[1,2,4,4],[3,3,3,4],[4,3,4,4],[4,4,4,2],
 [1,1,1,1],[2,4,2,2],[1,1,2,4],[2,2,2,2],[4,4,4,3],[3,3,3,3],[1,1,1,1],
]
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))
function evolve_dist(parent::Int, p, pi)
    v = (1.0 - p) .* pi
    v[parent] += p
    return v ./ sum(v)
end

@model function f81_joint(nsites, align)
    p  ~ Beta(1.0, 1.0)
    pi ~ Dirichlet(ones(4))
    root = Vector{Int}(undef, nsites)
    L    = Vector{Int}(undef, nsites)
    M    = Vector{Int}(undef, nsites)
    for s in 1:nsites
        root[s] ~ Categorical(pi)
        L[s] ~ Categorical(evolve_dist(root[s], p, pi))
        M[s] ~ Categorical(evolve_dist(root[s], p, pi))
        col = align[s]
        col[1] ~ Categorical(evolve_dist(L[s], p, pi))
        col[2] ~ Categorical(evolve_dist(L[s], p, pi))
        col[3] ~ Categorical(evolve_dist(M[s], p, pi))
        col[4] ~ Categorical(evolve_dist(M[s], p, pi))
    end
end

function run_pg(nsites::Int, iters::Int)
    chain = sample(f81_joint(nsites, ALIGN[1:nsites]), PG(PARTICLES), iters; progress=false)
    p  = mean(chain[:p])
    pi = [mean(chain[Symbol("pi[$k]")]) for k in 1:4]
    return p, pi
end

function main()
    nsites = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : length(ALIGN)
    iters  = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(3, 5)  

    t0 = time()
    p, pi = run_pg(nsites, iters)
    dt = time() - t0

    println("f81_joint / F81 (Turing PG)  n_sites=$nsites  particles=$PARTICLES  iters=$iters")
    @printf("retention prob p = %.3f   (true 0.8)\n", p)
    @printf("base freqs pi   = [%.2f, %.2f, %.2f, %.2f]   (true [0.3, 0.2, 0.2, 0.3])\n",
            pi[1], pi[2], pi[3], pi[4])
    @printf("RESULT: %.6f\n", p)   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
