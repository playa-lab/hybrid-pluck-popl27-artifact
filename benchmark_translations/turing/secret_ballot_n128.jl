using Turing
using Statistics
using Printf

const BALLOTS = [([0, 1, 2, 3, 4], 1), ([2, 3, 4, 5, 6], 0), ([4, 5, 6, 7, 8], 0), ([6, 7, 8, 9, 10], 0), ([8, 9, 10, 11, 12], 0), ([10, 11, 12, 13, 14], 0), ([12, 13, 14, 15, 16], 1), ([14, 15, 16, 17, 18], 1), ([16, 17, 18, 19, 20], 1), ([18, 19, 20, 21, 22], 0), ([20, 21, 22, 23, 24], 0), ([22, 23, 24, 25, 26], 1), ([24, 25, 26, 27, 28], 0), ([26, 27, 28, 29, 30], 0), ([28, 29, 30, 31, 32], 0), ([30, 31, 32, 33, 34], 1), ([32, 33, 34, 35, 36], 0), ([34, 35, 36, 37, 38], 1), ([36, 37, 38, 39, 40], 1), ([38, 39, 40, 41, 42], 0), ([40, 41, 42, 43, 44], 0), ([42, 43, 44, 45, 46], 0), ([44, 45, 46, 47, 48], 1), ([46, 47, 48, 49, 50], 1), ([48, 49, 50, 51, 52], 0), ([50, 51, 52, 53, 54], 1), ([52, 53, 54, 55, 56], 1), ([54, 55, 56, 57, 58], 1), ([56, 57, 58, 59, 60], 1), ([58, 59, 60, 61, 62], 1), ([60, 61, 62, 63, 64], 1), ([62, 63, 64, 65, 66], 1), ([64, 65, 66, 67, 68], 0), ([66, 67, 68, 69, 70], 0), ([68, 69, 70, 71, 72], 1), ([70, 71, 72, 73, 74], 1), ([72, 73, 74, 75, 76], 1), ([74, 75, 76, 77, 78], 0), ([76, 77, 78, 79, 80], 1), ([78, 79, 80, 81, 82], 1), ([80, 81, 82, 83, 84], 0), ([82, 83, 84, 85, 86], 1), ([84, 85, 86, 87, 88], 1), ([86, 87, 88, 89, 90], 0), ([88, 89, 90, 91, 92], 1), ([90, 91, 92, 93, 94], 1), ([92, 93, 94, 95, 96], 0), ([94, 95, 96, 97, 98], 0), ([96, 97, 98, 99, 100], 1), ([98, 99, 100, 101, 102], 0), ([100, 101, 102, 103, 104], 0), ([102, 103, 104, 105, 106], 1), ([104, 105, 106, 107, 108], 1), ([106, 107, 108, 109, 110], 1), ([108, 109, 110, 111, 112], 1), ([110, 111, 112, 113, 114], 0), ([112, 113, 114, 115, 116], 1), ([114, 115, 116, 117, 118], 1), ([116, 117, 118, 119, 120], 1), ([118, 119, 120, 121, 122], 0), ([120, 121, 122, 123, 124], 0), ([122, 123, 124, 125, 126], 1), ([124, 125, 126, 127, 0], 0), ([126, 127, 0, 1, 2], 1)]
const TEND = [0.8, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2, 0.2, 0.8, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.8, 0.2, 0.8, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.8, 0.8, 0.8, 0.2, 0.8, 0.2, 0.2, 0.8, 0.2, 0.2, 0.8, 0.2, 0.8, 0.8, 0.2, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.8, 0.2, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8, 0.8, 0.2, 0.2, 0.8, 0.8, 0.8, 0.2, 0.2, 0.2, 0.2]
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
    n_members = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 128
    iters     = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(4, 5)
    t0 = time(); p0 = run_pg(n_members, iters); dt = time() - t0
    println("secret_ballot (Turing PG)  n_members=$n_members  particles=$PARTICLES  iters=$iters")
    @printf("member 0 posterior: E[p_0]=%.3f   (true tendency %.1f)\n", p0, TEND[1])
    @printf("RESULT: %.6f\n", p0)   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
