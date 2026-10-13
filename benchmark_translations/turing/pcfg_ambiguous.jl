using Turing
using Statistics
using Printf

const CORPUS = [
  ["the", "dog", "saw", "the", "cat"],
  ["the", "cat", "saw", "the", "dog"],
  ["the", "dog", "saw", "the", "cat", "with", "the", "dog"],
  ["the", "cat", "saw", "the", "dog", "with", "the", "cat"],
  ["the", "dog", "saw", "the", "dog"],
  ["the", "cat", "saw", "the", "cat", "with", "the", "dog"],
  ["the", "dog", "saw", "the", "cat", "with", "the", "dog", "with", "the", "cat"],
  ["the", "cat", "saw", "the", "dog", "with", "the", "cat", "with", "the", "dog"],
  ["the", "dog", "saw", "the", "cat", "with", "the", "cat"],
  ["the", "cat", "saw", "the", "dog", "with", "the", "dog", "with", "the", "cat", "with", "the", "dog"],
  ["the", "dog", "saw", "the", "cat"],
  ["the", "cat", "saw", "the", "dog", "with", "the", "cat", "with", "the", "cat"],
  ["the", "dog", "saw", "the", "dog", "with", "the", "cat", "with", "the", "dog", "with", "the", "cat"],
  ["the", "cat", "saw", "the", "cat"],
  ["the", "dog", "saw", "the", "cat", "with", "the", "dog", "with", "the", "cat", "with", "the", "dog", "with", "the", "cat"],
  ["the", "cat", "saw", "the", "dog", "with", "the", "dog"],
  ["the", "dog", "saw", "the", "cat", "with", "the", "cat", "with", "the", "dog"],
  ["the", "cat", "saw", "the", "dog", "with", "the", "cat", "with", "the", "dog", "with", "the", "cat"],
  ["the", "dog", "saw", "the", "dog", "with", "the", "cat"],
  ["the", "cat", "saw", "the", "dog", "with", "the", "cat", "with", "the", "dog", "with", "the", "cat", "with", "the", "dog"],
]

const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))

const EOS, THE, DOG, CAT, SAW, WITH = 0, 1, 2, 3, 4, 5
const WORD_ID = Dict("the" => THE, "dog" => DOG, "cat" => CAT, "saw" => SAW, "with" => WITH)
const SENT, NP, VP, PP, N, DET, V, P = 1, 2, 3, 4, 5, 6, 7, 8

encode(s) = vcat([WORD_ID[w] for w in s], EOS)

@model function pcfg_ambiguous(S)
    pNP ~ Dirichlet(ones(2))
    pVP ~ Dirichlet(ones(2))
    pN  ~ Dirichlet(ones(2))
    c = Vector{Int}(undef, sum(2 * length(s) + 1 for s in S))
    k = 0
    for i in eachindex(S)
        L = length(S[i])
        stack = [SENT]
        pos = 1
        while !isempty(stack) && pos <= L
            sym = pop!(stack)
            if sym == SENT
                push!(stack, VP, NP)
            elseif sym == NP
                k += 1
                c[k] ~ Categorical(pNP)
                c[k] == 1 ? push!(stack, N, DET) : push!(stack, PP, N, DET)
            elseif sym == VP
                k += 1
                c[k] ~ Categorical(pVP)
                c[k] == 1 ? push!(stack, NP, V) : push!(stack, PP, NP, V)
            elseif sym == PP
                push!(stack, NP, P)
            else
                if sym == N
                    k += 1
                    c[k] ~ Categorical(pN)
                    w = c[k] == 1 ? DOG : CAT
                else
                    w = sym == DET ? THE : sym == V ? SAW : WITH
                end
                S[i][pos] ~ Dirac(w)
                pos += 1
            end
        end
        for q in pos:L
            S[i][q] ~ Dirac(EOS)
        end
    end
end

function run_pg(n_sent::Int, iters::Int)
    chain = sample(pcfg_ambiguous(encode.(CORPUS[1:n_sent])), PG(PARTICLES), iters; progress=false)
    getp(name, j) = mean(chain[Symbol("$name[$j]")])
    return getp("pNP", 1), getp("pVP", 1), getp("pN", 1)
end

function main()
    n_sent = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 20
    iters  = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(2, 5)

    t0 = time()
    pNP, pVP, pN = run_pg(n_sent, iters)
    dt = time() - t0

    println("pcfg_ambiguous (Turing PG)  n_sentences=$n_sent  particles=$PARTICLES  iters=$iters")
    println("recovered production probabilities (posterior mean):")
    @printf("  NP  [Det N=%.3f, Det N PP=%.3f]\n", pNP, 1 - pNP)
    @printf("  VP  [V NP=%.3f, V NP PP=%.3f]\n", pVP, 1 - pVP)
    @printf("  N   [dog=%.3f, cat=%.3f]\n", pN, 1 - pN)
    @printf("RESULT: %.6f\n", pVP)
    @printf("inference_seconds: %.3f\n", dt)
end

main()
