using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))
using DelimitedFiles
using BenchmarkTools


if length(ARGS) == 0
    bits = 4
    pieces = 16
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end

xs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
ys = [1.3998703068665284, 6.235613006596935, 14.930938208255952, 27.76621455275837,
      44.63428250334452, 65.39877806912918, 90.16345175668341, 118.94703945597995,
      151.7909055407857, 188.5411680368732]

p = 0.3
max_degree = 5   # up to 5 coefficients => degrees 0..4

DFiP = DistFix{20 + bits, bits}

t = @timed_infer pr(@dice begin
    # --- random polynomial via truncated-geometric stopping ---
    # present[k] is true iff we have not stopped at or before coefficient k.
    present = Vector(undef, max_degree)
    cont = !flip(p)          # present[1] (the constant term c0)
    present[1] = cont
    for k = 2:max_degree
        cont = cont & !flip(p)
        present[k] = cont
    end

    coeffs = Vector(undef, max_degree)
    for k = 1:max_degree
        coeffs[k] = ifelse(present[k],
                           bitblast(DFiP, Normal(0.0, 100.0), pieces, -1024.0, 1024.0),
                           DFiP(0.0))
    end

    # --- exactly one outlier at a uniform index 0..9 ---
    outlier_index = uniform(DistUInt{4}, 0, 10)

    # --- evaluate curve at each x and observe against ys ---
    for i = 1:length(xs)
        acc = DFiP(0.0)
        for k = max_degree:-1:1
            acc = coeffs[k] + DFiP(xs[i]) * acc
        end

        is_outlier = prob_equals(outlier_index, DistUInt{4}(i - 1))
        if is_outlier
            gaussian_observe(DFiP, pieces, -1024.0, 1024.0, DFiP(0.0), 200.0, DFiP(ys[i]), add=true)
        else
            gaussian_observe(DFiP, pieces, -256.0, 256.0, acc, 1.0, DFiP(ys[i]), add=true)
        end
    end

    outlier_index
end)

dist = t.value
p0 = sum(v for (k, v) in dist if k == 0)
println("RESULT: ", p0)
@show bits, pieces, t.time
for (k, v) in sort(collect(dist))
    println("outlier_index = $k : $v")
end
