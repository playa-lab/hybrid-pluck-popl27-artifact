using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 5
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
I = 12
J = 3

DFiP = DistFix{bits + 2, bits}     
annotations = [(1, 1, false), (1, 2, false), (2, 2, false), (3, 1, true), (3, 2, true), (3, 3, true), (4, 3, true), (5, 2, false), (5, 3, true), (6, 1, false), (6, 2, false), (7, 1, false), (7, 2, false), (9, 3, false), (10, 1, true), (10, 3, true), (11, 1, true), (11, 2, false), (11, 3, false), (12, 1, false), (12, 2, false), (12, 3, false)]

t = @timed_infer expectation(@dice begin
    prevalence = bitblast_exponential(DFiP, Beta(1.0, 1.0), pieces, 0.0, 1.0)
    tp = [bitblast_exponential(DFiP, Beta(2.0, 1.0), pieces, 0.0, 1.0) for _ in 1:J]
    fp = [bitblast_exponential(DFiP, Beta(1.0, 2.0), pieces, 0.0, 1.0) for _ in 1:J]
    
    z = Any[]
    for _ in 1:I
        push!(z, parametrised_flip(prevalence))
    end
    for a in annotations
        i = a[1]; j = a[2]; label = a[3]
        f = parametrised_flip(ifelse(z[i], tp[j], fp[j]))
        observe(prob_equals(f, label))
    end
    tp[1]
end)

println("RESULT: ", t.value)
@show bits, pieces, I, t.value, t.time
