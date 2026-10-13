# PackageCompiler precompile workload for the Turing sysimage.
#
# Running this during `create_sysimage(["Turing"]; precompile_execution_file=...)`
# traces the methods the benchmarks actually hit — @model expansion, the
# Particle Gibbs particle filter (PG), Normal/Categorical observe+assume, and
# the sample() driver — so their native specializations get baked into
# /opt/turing.so. Mirrors the HMM hot path shared by the turing-programs
# (continuous priors + a discrete latent chain observed through a Normal); it
# must run cleanly (an error here aborts the sysimage build).
using Turing

@model function _wl(N, ys)
    mu = Vector{Float64}(undef, 2)
    for k in 1:2
        mu[k] ~ Normal(0.0, 1.0)
    end
    s = Vector{Int}(undef, N)
    for i in 1:N
        s[i] ~ Categorical([0.5, 0.5])
        ys[i] ~ Normal(mu[s[i]], 1.0)
    end
end

sample(_wl(5, [0.1, -0.2, 0.3, 0.0, 0.4]), PG(8), 20; progress=false)
