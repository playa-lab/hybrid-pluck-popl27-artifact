# PackageCompiler precompile workload for the Dice sysimage.
#
# Running this during `create_sysimage(["Dice"]; precompile_execution_file=...)`
# traces the methods the benchmarks actually hit (bitblast_exponential,
# parametrised_flip, observe, expectation, @dice, pr) so their native
# specializations get baked into /opt/dice.so. Mirrors the coinBias hot path; it
# must run cleanly (an error here aborts the sysimage build).
using Dice, Distributions

DFiP = DistFix{10, 8}
y = [1, 1, 0, 1, 0]
expectation(@dice begin
    a = bitblast_exponential(DFiP, Beta(2, 5), 16, 0.0, 1.0)
    for i in 1:5
        b = parametrised_flip(a)
        observe((y[i] != 1) | b)
        observe((y[i] == 1) | !b)
    end
    return a
end)
pr(@dice flip(0.3))
