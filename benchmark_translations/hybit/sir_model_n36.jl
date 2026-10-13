using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

if length(ARGS) == 0
    bits = 4
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = 36

DFiP = DistFix{bits + 2, bits}     # [0,1]: 2 int bits so p=1.0 (Beta endpoint) is representable
H = [[false, false, false, false, false, true, true, true, false, true, false, false],
     [false, true, false, true, true, true, true, true, false, false, false, false],
     [false, true, true, true, false, true, false, false, false, true, true, true]]

t = @timed_infer expectation(@dice begin
    rec = bitblast_exponential(DFiP, Beta(1.0, 1.0), pieces, 0.0, 1.0)
    inf = bitblast_exponential(DFiP, Beta(1.0, 1.0), pieces, 0.0, 1.0)
    bas = bitblast_exponential(DFiP, Beta(0.2, 1.0), pieces, 0.0, 1.0)
    sym = bitblast_exponential(DFiP, Beta(1.0, 1.0), pieces, 0.0, 1.0)
    for r in 1:length(H)
        row = H[r]
        state = false                          # each row restarts not-infected (Julia Bool)
        for i in 1:length(row)
            pobs = ifelse(state, sym, bas)     # Base.ifelse when state is Bool, Dice's when Dist
            observe(parametrised_flip(pobs) == row[i])
            state = ifelse(state, !parametrised_flip(rec), parametrised_flip(inf))
        end
    end
    inf
end)

println("RESULT: ", t.value)   # E[inf] -- infection rate, the cross-PPL error scalar
@show bits, pieces, N, t.value, t.time
