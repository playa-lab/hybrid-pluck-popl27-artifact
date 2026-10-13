


using Turing
using Statistics
using Printf

const XS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
const YS = [1.3998703068665284, 6.235613006596935, 14.930938208255952, 27.76621455275837, 44.63428250334452, 65.39877806912918, 90.16345175668341, 118.94703945597995, 151.7909055407857, 188.5411680368732]

const MAXDEG = 5
const GEOM_P = 0.3
const PARTICLES = parse(Int, get(ENV, "TURING_PARTICLES", "32"))


const DEG_PMF = let w = [(1 - GEOM_P)^d * (d < MAXDEG ? GEOM_P : 1.0) for d in 0:MAXDEG]
    w ./ sum(w)
end


function evalpoly_deg(coeffs, deg, x)
    acc = 0.0
    for k in deg:-1:1
        acc = coeffs[k] + x * acc
    end
    return acc
end

@model function polyreg(xs, ys)
    n = length(xs)
    degidx ~ Categorical(DEG_PMF)   
    deg = degidx - 1
    c = Vector{Float64}(undef, MAXDEG)
    for k in 1:MAXDEG
        c[k] ~ Normal(0.0, 100.0)
    end
    oi ~ Categorical(fill(1.0 / n, n))   
    for i in 1:n
        if i == oi
            ys[i] ~ Normal(0.0, 200.0)
        else
            ys[i] ~ Normal(evalpoly_deg(c, deg, xs[i]), 1.0)
        end
    end
end

function run_pg(iters::Int)
    chain = sample(polyreg(XS, YS), PG(PARTICLES), iters; progress=false)
    d = vec(chain[:degidx]) .- 1
    degpost = [mean(d .== k) for k in 0:MAXDEG]
    p_out0 = mean(vec(chain[:oi]) .== 1)   
    return degpost, p_out0
end

function main()
    iters = haskey(ENV, "TURING_ITERS") ? parse(Int, ENV["TURING_ITERS"]) : length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 500
    run_pg(5)  

    t0 = time()
    degpost, p_out0 = run_pg(iters)
    dt = time() - t0

    names = ["0 (const)", "1 (linear)", "2 (quad)", "3 (cubic)", "4 (quartic)", "5 (quintic)"]
    println("polyreg (Turing PG)  particles=$PARTICLES  iters=$iters")
    println("posterior over polynomial degree:")
    for d in 0:MAXDEG
        @printf("  degree %-12s P=%.3f\n", names[d + 1], degpost[d + 1])
    end
    @printf("RESULT: %.6f\n", p_out0)   
    @printf("inference_seconds: %.3f\n", dt)
end

main()
