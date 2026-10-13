using Turing
using Statistics
using Printf

const ANNOTS = [(0,0,0), (0,1,0), (1,1,0), (2,0,1), (2,1,1), (2,2,1), (3,2,1), (4,1,0), (4,2,1), (5,0,0), (5,1,0), (6,0,0), (6,1,0), (8,2,0), (9,0,1), (9,2,1), (10,0,1), (10,1,0), (10,2,0), (11,0,0), (11,1,0), (11,2,0)]
const N_ANNOTATORS = 3
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

function annots_for(nitems, nann)
    sel = [a for a in ANNOTS if a[1] < nitems && a[2] < nann]
    return [a[1]+1 for a in sel], [a[2]+1 for a in sel], [a[3] for a in sel]
end

@model function dawid_skene(nitems, nann, items, anns, labels)
    prevalence ~ Beta(1.0, 1.0)
    tp = Vector{Float64}(undef, nann); fp = Vector{Float64}(undef, nann)
    for j in 1:nann
        tp[j] ~ Beta(2.0, 1.0)
        fp[j] ~ Beta(1.0, 2.0)
    end
    z = Vector{Int}(undef, nitems)
    for i in 1:nitems
        z[i] ~ Bernoulli(prevalence)
    end
    for a in 1:length(items)
        rate = z[items[a]] == 1 ? tp[anns[a]] : fp[anns[a]]
        labels[a] ~ Bernoulli(rate)
    end
end

function run_pg(nitems::Int, iters::Int)
    items, anns, labels = annots_for(nitems, N_ANNOTATORS)
    chain = sample(dawid_skene(nitems, N_ANNOTATORS, items, anns, labels), PG(PARTICLES), iters; progress=false)
    return mean(chain[Symbol("tp[1]")]), mean(chain[Symbol("fp[1]")])
end

function main()
    nitems = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 12
    iters  = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(4, 5)
    t0 = time(); tp1, fp1 = run_pg(nitems, iters); dt = time() - t0
    @printf("RESULT: %.6f\n", tp1)
    println("dawid_skene (Turing PG)  n_items=$nitems  n_annotators=$N_ANNOTATORS  particles=$PARTICLES  iters=$iters")
    @printf("annotator 0 posterior: tp=%.3f (true 0.95)   fp=%.3f (true 0.05)\n", tp1, fp1)
    @printf("inference_seconds: %.3f\n", dt)
end

main()
