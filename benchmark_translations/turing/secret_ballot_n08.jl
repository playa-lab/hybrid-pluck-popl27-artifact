using Turing
using Statistics
using Printf

const BALLOTS = [([0, 1, 2, 3, 4], 1), ([2, 3, 4, 5, 6], 1), ([4, 5, 6, 7, 0], 0), ([6, 7, 0, 1, 2], 0)]
const TEND = [0.2, 0.8, 0.2, 0.8, 0.2, 0.2, 0.8, 0.8]
const EPS = 1e-6
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

function ballots_for(n_members)
    subs = Vector{Vector{Int}}(); outs = Int[]
    for (idxs, out) in BALLOTS
        if all(m -> m < n_members, idxs)
            push!(subs, [m + 1 for m in idxs]); push!(outs, out)
        end
    end
    total = sum(length.(subs); init = 0)
    return subs, outs, total
end

@model function secret_ballot(n_members, subs, outs, total)
    p = Vector{Float64}(undef, n_members)
    for m in 1:n_members
        p[m] ~ Beta(1.0, 1.0)
    end
    v = Vector{Int}(undef, total); vc = 0
    for b in 1:length(subs)
        idxs = subs[b]; yes = 0
        for m in idxs
            vc += 1
            v[vc] ~ Bernoulli(p[m]); yes += v[vc]
        end
        maj = 2 * yes > length(idxs)
        outs[b] ~ Bernoulli(maj ? 1.0 - EPS : EPS)
    end
end

function run_pg(n_members::Int, iters::Int)
    subs, outs, total = ballots_for(n_members)
    chain = sample(secret_ballot(n_members, subs, outs, total), PG(PARTICLES), iters; progress=false)
    return mean(chain[Symbol("p[1]")])
end

function main()
    n_members = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 8
    iters     = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(4, 5)
    t0 = time(); p0 = run_pg(n_members, iters); dt = time() - t0
    println("secret_ballot (Turing PG)  n_members=$n_members  particles=$PARTICLES  iters=$iters")
    @printf("member 0 posterior: E[p_0]=%.3f   (true tendency %.1f)\n", p0, TEND[1])
    @printf("RESULT: %.6f\n", p0)   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
