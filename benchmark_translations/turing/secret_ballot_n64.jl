using Turing
using Statistics
using Printf

const BALLOTS = [([0, 1, 2, 3, 4], 1), ([2, 3, 4, 5, 6], 0), ([4, 5, 6, 7, 8], 0), ([6, 7, 8, 9, 10], 0), ([8, 9, 10, 11, 12], 0), ([10, 11, 12, 13, 14], 1), ([12, 13, 14, 15, 16], 0), ([14, 15, 16, 17, 18], 0), ([16, 17, 18, 19, 20], 0), ([18, 19, 20, 21, 22], 1), ([20, 21, 22, 23, 24], 1), ([22, 23, 24, 25, 26], 0), ([24, 25, 26, 27, 28], 0), ([26, 27, 28, 29, 30], 1), ([28, 29, 30, 31, 32], 1), ([30, 31, 32, 33, 34], 1), ([32, 33, 34, 35, 36], 0), ([34, 35, 36, 37, 38], 1), ([36, 37, 38, 39, 40], 1), ([38, 39, 40, 41, 42], 1), ([40, 41, 42, 43, 44], 1), ([42, 43, 44, 45, 46], 0), ([44, 45, 46, 47, 48], 0), ([46, 47, 48, 49, 50], 0), ([48, 49, 50, 51, 52], 0), ([50, 51, 52, 53, 54], 0), ([52, 53, 54, 55, 56], 0), ([54, 55, 56, 57, 58], 1), ([56, 57, 58, 59, 60], 1), ([58, 59, 60, 61, 62], 0), ([60, 61, 62, 63, 0], 0), ([62, 63, 0, 1, 2], 0)]
const TEND = [0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.2, 0.8, 0.2, 0.2, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.2, 0.8, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.8, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.8, 0.2, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2]
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
    n_members = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 64
    iters     = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(4, 5)
    t0 = time(); p0 = run_pg(n_members, iters); dt = time() - t0
    println("secret_ballot (Turing PG)  n_members=$n_members  particles=$PARTICLES  iters=$iters")
    @printf("member 0 posterior: E[p_0]=%.3f   (true tendency %.1f)\n", p0, TEND[1])
    @printf("RESULT: %.6f\n", p0)   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
