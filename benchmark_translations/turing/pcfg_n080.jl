
using Turing
using Statistics
using Printf


const CORPUS = [
  ["the","sad","cat","eats","a","bone","but","a","tasty","bone","eats","a","bone"],
  ["the","cat","eats","the","bone"],
  ["a","tasty","cat","eats","the","bone"],
  ["a","sad","tasty","bone","barks"],
  ["a","sad","dog","eats","a","dog"],
  ["a","cat","eats","the","dog"],
  ["the","dog","eats","a","bone"],
  ["a","tasty","bone","eats","a","cat"],
  ["a","bone","eats"],
  ["a","tasty","happy","bone","eats"],
  ["a","sad","bone","eats","a","dog"],
  ["the","bone","barks"],
  ["the","tasty","bone","barks","and","a","cat","eats"],
  ["a","sad","dog","eats","a","bone"],
  ["a","cat","eats"],
  ["the","cat","eats","the","bone"],
  ["the","dog","barks"],
  ["a","cat","barks"],
  ["a","dog","barks","but","the","sad","cat","eats","the","dog"],
  ["the","cat","barks"],
  ["a","dog","eats","the","bone"],
  ["the","bone","eats","a","cat","but","a","tasty","cat","barks"],
  ["the","bone","eats"],
  ["the","cat","eats","a","dog"],
  ["the","happy","dog","eats","a","tasty","dog"],
  ["a","dog","eats"],
  ["the","happy","bone","eats","the","tasty","sad","dog"],
  ["the","happy","cat","barks"],
  ["the","bone","eats"],
  ["the","dog","barks","but","a","cat","eats","the","dog","and","a","cat","eats"],
  ["the","bone","eats"],
  ["the","bone","eats","a","sad","happy","bone"],
  ["the","cat","eats","a","sad","dog"],
  ["a","cat","barks","but","the","cat","eats","the","happy","bone"],
  ["a","bone","barks","but","the","tasty","cat","barks"],
  ["a","bone","eats","a","dog"],
  ["a","sad","dog","eats","the","dog","but","the","sad","happy","bone","eats","a","sad","happy","bone","and","the","bone","eats","a","cat"],
  ["the","cat","barks"],
  ["the","dog","barks"],
  ["a","dog","eats","and","a","bone","eats"],
  ["a","dog","eats","the","bone","and","a","happy","bone","eats","but","the","tasty","bone","eats","the","sad","dog"],
  ["the","tasty","dog","barks","but","a","dog","barks"],
  ["the","tasty","bone","eats"],
  ["a","bone","barks"],
  ["a","bone","barks"],
  ["the","bone","barks"],
  ["a","dog","eats"],
  ["the","bone","eats","the","dog"],
  ["the","happy","happy","cat","eats","the","dog"],
  ["the","bone","barks"],
  ["the","happy","cat","eats","a","dog"],
  ["the","bone","barks","and","the","dog","barks","and","the","tasty","dog","eats","the","dog"],
  ["a","tasty","tasty","happy","cat","eats","the","tasty","bone","but","the","cat","eats","the","cat"],
  ["a","dog","eats","the","bone"],
  ["the","dog","eats","and","a","cat","barks"],
  ["the","dog","barks"],
  ["a","sad","bone","eats","and","a","dog","barks","but","a","dog","eats","and","the","sad","bone","barks"],
  ["the","bone","eats","and","the","tasty","dog","barks","and","the","dog","eats"],
  ["a","tasty","cat","eats"],
  ["a","bone","eats","a","happy","happy","sad","tasty","dog"],
  ["a","cat","barks","but","the","cat","barks","and","the","tasty","dog","eats"],
  ["a","tasty","sad","dog","eats"],
  ["the","happy","dog","eats","and","a","bone","barks"],
  ["a","bone","barks"],
  ["a","happy","cat","eats","the","tasty","tasty","happy","dog"],
  ["a","bone","eats"],
  ["the","dog","eats","a","sad","dog"],
  ["the","tasty","dog","eats"],
  ["the","bone","eats","a","bone"],
  ["the","dog","eats","a","dog"],
  ["a","cat","barks","and","a","cat","eats","a","sad","dog"],
  ["the","happy","dog","eats","a","dog","and","the","dog","eats","a","happy","sad","happy","bone"],
  ["the","cat","barks"],
  ["a","cat","barks"],
  ["a","cat","eats"],
  ["the","tasty","cat","eats","a","bone"],
  ["a","tasty","dog","barks"],
  ["a","cat","eats","and","the","bone","barks","and","the","dog","barks"],
  ["the","bone","eats","a","tasty","cat"],
  ["the","tasty","tasty","dog","eats","the","cat","and","the","dog","eats","but","a","cat","eats","a","tasty","dog"],
  ["a","cat","eats","the","happy","tasty","bone","and","a","dog","eats","and","the","dog","barks","but","a","happy","cat","eats","a","dog","but","a","bone","eats","a","dog"],
  ["a","cat","eats","the","cat"],
  ["a","dog","eats","and","a","bone","barks","but","a","dog","eats","a","dog"],
  ["a","bone","eats","the","tasty","cat","but","a","sad","bone","eats"],
  ["the","cat","eats"],
  ["the","sad","tasty","bone","eats","a","dog"],
  ["a","cat","barks"],
  ["the","dog","eats","the","dog"],
  ["a","cat","eats","a","happy","dog","and","a","tasty","dog","eats"],
  ["a","bone","eats"],
  ["the","happy","dog","eats"],
  ["the","bone","barks"],
  ["a","cat","eats","the","tasty","tasty","sad","happy","cat"],
  ["a","cat","eats","the","bone"],
  ["a","bone","barks","but","the","sad","sad","dog","barks"],
  ["the","happy","bone","eats","a","tasty","cat"],
  ["the","bone","eats","the","sad","happy","bone","but","a","bone","eats"],
  ["the","happy","happy","bone","eats"],
  ["the","bone","barks"],
  ["the","bone","eats","the","dog","but","the","cat","eats","the","cat"],
]

const ADJS  = Set(["happy", "sad", "tasty"])
const DETS  = Set(["the", "a"])
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))
new_counts() = Dict(:sent=>zeros(Int,2), :vp=>zeros(Int,2), :verb=>zeros(Int,2),
                    :adj=>zeros(Int,3), :det=>zeros(Int,2), :noun=>zeros(Int,2),
                    :nounword=>zeros(Int,3), :conj=>zeros(Int,2))

function p_det(w, i, C)
    C[:det][w[i] == "the" ? 1 : 2] += 1
    return i + 1
end
function p_adj(w, i, C)
    C[:adj][w[i] == "happy" ? 1 : w[i] == "sad" ? 2 : 3] += 1
    return i + 1
end
function p_noun(w, i, C)
    if i <= length(w) && w[i] in ADJS
        C[:noun][1] += 1                       
        i = p_adj(w, i, C)
        return p_noun(w, i, C)
    else
        C[:noun][2] += 1                       
        C[:nounword][w[i] == "cat" ? 1 : w[i] == "dog" ? 2 : 3] += 1
        return i + 1
    end
end
p_np(w, i, C) = p_noun(w, p_det(w, i, C), C)
function p_conj(w, i, C)
    C[:conj][w[i] == "and" ? 1 : 2] += 1
    return i + 1
end
function p_vp(w, i, C)
    if w[i] == "barks"
        C[:vp][1] += 1; C[:verb][1] += 1       
        return i + 1
    else  
        if i + 1 <= length(w) && w[i + 1] in DETS
            C[:vp][2] += 1                      
            return p_np(w, i + 1, C)
        else
            C[:vp][1] += 1; C[:verb][2] += 1    
            return i + 1
        end
    end
end
function p_sent(w, i, C)
    i = p_np(w, i, C)
    i = p_vp(w, i, C)
    if i <= length(w) && (w[i] == "and" || w[i] == "but")
        C[:sent][2] += 1                        
        i = p_conj(w, i, C)
        return p_sent(w, i, C)
    else
        C[:sent][1] += 1                        
        return i
    end
end

function tally(sentences)
    C = new_counts()
    for s in sentences
        p_sent(s, 1, C)
    end
    return C
end


@model function pcfg(counts)
    pSent     ~ Dirichlet(ones(2))
    pVP       ~ Dirichlet(ones(2))
    pVerb     ~ Dirichlet(ones(2))
    pAdj      ~ Dirichlet(ones(3))
    pDet      ~ Dirichlet(ones(2))
    pNoun     ~ Dirichlet(ones(2))
    pNounword ~ Dirichlet(ones(3))
    pConj     ~ Dirichlet(ones(2))
    ps = (pSent, pVP, pVerb, pAdj, pDet, pNoun, pNounword, pConj)
    keys = (:sent, :vp, :verb, :adj, :det, :noun, :nounword, :conj)
    for r in 1:8
        c = counts[keys[r]]
        if sum(c) > 0
            counts[keys[r]] ~ Multinomial(sum(c), ps[r])
        end
    end
end

function run_pg(n_sent::Int, iters::Int)
    C = tally(CORPUS[1:n_sent])
    chain = sample(pcfg(C), PG(PARTICLES), iters; progress=false)
    getp(name, k) = mean(chain[Symbol("$name[$k]")])
    return C,
        [getp("pSent", k) for k in 1:2],
        [getp("pVP", k) for k in 1:2],
        [getp("pVerb", k) for k in 1:2],
        [getp("pAdj", k) for k in 1:3],
        [getp("pDet", k) for k in 1:2],
        [getp("pNoun", k) for k in 1:2],
        [getp("pNounword", k) for k in 1:3],
        [getp("pConj", k) for k in 1:2]
end

function main()
    n_sent = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 80
    iters  = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 500
    run_pg(5, 5)  

    t0 = time()
    C, pSent, pVP, pVerb, pAdj, pDet, pNoun, pNounword, pConj = run_pg(n_sent, iters)
    dt = time() - t0

    println("pcfg (Turing PG)  n_sentences=$n_sent  particles=$PARTICLES  iters=$iters")
    println("recovered production probabilities (posterior mean):")
    @printf("  Sent      [NP VP=%.2f, NP VP Conj Sent=%.2f]\n", pSent[1], pSent[2])
    @printf("  VP        [Verb=%.2f, VT NP=%.2f]\n", pVP[1], pVP[2])
    @printf("  Verb      [barks=%.2f, eats=%.2f]\n", pVerb[1], pVerb[2])
    @printf("  Adj       [happy=%.2f, sad=%.2f, tasty=%.2f]\n", pAdj[1], pAdj[2], pAdj[3])
    @printf("  Det       [the=%.2f, a=%.2f]\n", pDet[1], pDet[2])
    @printf("  Noun      [Adj Noun=%.2f, noun-word=%.2f]\n", pNoun[1], pNoun[2])
    @printf("  Noun-word [cat=%.2f, dog=%.2f, bone=%.2f]\n", pNounword[1], pNounword[2], pNounword[3])
    @printf("  Conj      [and=%.2f, but=%.2f]\n", pConj[1], pConj[2])
    @printf("RESULT: %.6f\n", pSent[1])   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
