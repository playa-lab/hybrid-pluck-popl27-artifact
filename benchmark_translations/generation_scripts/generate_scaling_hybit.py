#!/usr/bin/env python3
"""Generate the scaled HyBit (Dice / Alea.jl) benchmark programs.

For every scaling family HyBit can express, emits one program per size of the
Pluck sweep (<family>_<n|d><size>.jl), embedding exactly the data the Pluck
program at that size uses (scaling_data.json) and matching the Pluck model's
priors, standard deviations, transition matrix and mixing weights. Continuous
latents are bit-blasted (`bitblast` / `bitblast_exponential`); `bits pieces` on
the command line override each program's default discretisation.

Not generated (not expressible in HyBit): pcfg, soccer (Dirichlet priors).

HyBit's exact (BDD) inference grows quickly with size for the models with shared
continuous latents (hmm_*, outliers, soft_kmeans); the larger sizes are expected
to time out, which is what the benchmark measures.

Run from anywhere:
    python3 generate_scaling_hybit.py [--out DIR] [--overwrite]
"""

import os

from gen_common import CONST, DATA, TRANSLATIONS, VALUES, Writer, parse_args
from gen_common import fname as _fname

write = None  # set in main()


def fname(fam, v):
    return _fname(fam, v, "jl")


def jl_floats(xs):
    return "[" + ", ".join(f"{x}" for x in xs) + "]"


def jl_bools(bs):
    return "[" + ", ".join("true" if b else "false" for b in bs) + "]"


# ---------------------------------------------------------------------------
# Shared HMM constants (matched to constants["hmm"]).
# ---------------------------------------------------------------------------
HMM = CONST["hmm"]
INIT = HMM["init"]                       # [0.5, 0.3, 0.2]
TRANS = HMM["trans"]                     # 3x3
EMIT_SD = HMM["emission_sd"]             # 1.0


def disc(row):
    return "discrete(DistUInt{2}, [" + ", ".join(str(x) for x in row) + "])"


def hmm_transition_def():
    return (
        "transition(prev) =\n"
        f"    ifelse(prob_equals(prev, U(0)), {disc(TRANS[0])},\n"
        f"      ifelse(prob_equals(prev, U(1)), {disc(TRANS[1])},\n"
        f"             {disc(TRANS[2])}))"
    )


# ---------------------------------------------------------------------------
# hmm_first_order
# ---------------------------------------------------------------------------
def gen_hmm_first_order():
    fam = "hmm_first_order"
    out = []
    for v in VALUES[fam]:
        obs = DATA[fam][str(v)]
        text = f"""using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

# Reference model: pluck-programs/{fam}.pluck (and shared/{fam}.pluck); SCALING N={v}.
# First-order HMM, K=3 shared Gaussian-emission means.
#   mu_k ~ N(0,1);  z_1 ~ cat{INIT};  z_n ~ cat(trans_{{z_n-1}});  y_n ~ N(mu_{{z_n}}, {EMIT_SD})
# Means SHARED across timesteps -> exact (BDD) inference enumerates 3^N paths;
# keep bits/pieces tiny. N and obs are baked in (from scaling_data.json); pass
# `bits pieces` to override. Reports E[mu_1].

if length(ARGS) == 0
    bits = 2
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = {v}

DFiP = DistFix{{bits + 5, bits}}
obs = {jl_floats(obs)}

U(k) = DistUInt{{2}}(k)

{hmm_transition_def()}

pick(z, m1, m2, m3) =
    ifelse(prob_equals(z, U(0)), m1, ifelse(prob_equals(z, U(1)), m2, m3))

t = @timed_infer expectation(@dice begin
    m1 = bitblast(DFiP, Normal(-2, 1), pieces, -8.0, 8.0)
    m2 = bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0)
    m3 = bitblast(DFiP, Normal(2, 1), pieces, -8.0, 8.0)
    z = discrete(DistUInt{{2}}, {INIT})
    for i in 1:N
        mu = pick(z, m1, m2, m3)
        gaussian_observe(DFiP, pieces, -8.0, 8.0, mu, {EMIT_SD}, DFiP(obs[i]), add=true)
        z = transition(z)
    end
    m1
end)

println("RESULT: ", t.value)
@show bits, pieces, N, t.value, t.time
"""
        name = fname(fam, v)
        out.append((name, write(name, text)))
    return fam, out


# ---------------------------------------------------------------------------
# hmm_second_order: z_n ~ cat(trans_{max(z_{n-2}, z_{n-1})}) for n>=3.
# ---------------------------------------------------------------------------
def gen_hmm_second_order():
    fam = "hmm_second_order"
    out = []
    for v in VALUES[fam]:
        obs = DATA[fam][str(v)]
        text = f"""using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

# Reference model: pluck-programs/{fam}.pluck (and shared/{fam}.pluck); SCALING N={v}.
# Second-order HMM: z_n ~ cat(trans_{{max(z_n-2, z_n-1)}}) for n>=3.
#   mu_k ~ N(0,1) shared;  z_1 ~ cat{INIT};  y_n ~ N(mu_{{z_n}}, {EMIT_SD}).
# Exact inference ~3^N -- keep bits/pieces tiny. N and obs baked in; pass
# `bits pieces`. Reports E[mu_1].

if length(ARGS) == 0
    bits = 2
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = {v}

DFiP = DistFix{{bits + 5, bits}}
obs = {jl_floats(obs)}

U(k) = DistUInt{{2}}(k)

{hmm_transition_def()}

pick(z, m1, m2, m3) =
    ifelse(prob_equals(z, U(0)), m1, ifelse(prob_equals(z, U(1)), m2, m3))

# native max of two DistUInt states (0..2): max(a,b) = ifelse(a>=b, a, b)
nmax(a, b) = ifelse(a >= b, a, b)

t = @timed_infer expectation(@dice begin
    m1 = bitblast(DFiP, Normal(-2, 1), pieces, -8.0, 8.0)
    m2 = bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0)
    m3 = bitblast(DFiP, Normal(2, 1), pieces, -8.0, 8.0)
    z1 = discrete(DistUInt{{2}}, {INIT})
    z2 = transition(z1)
    states = [z1, z2]
    for i in 1:N
        if i <= 2
            z = states[i]
        else
            z = transition(nmax(states[i-2], states[i-1]))
            push!(states, z)
        end
        mu = pick(z, m1, m2, m3)
        gaussian_observe(DFiP, pieces, -8.0, 8.0, mu, {EMIT_SD}, DFiP(obs[i]), add=true)
    end
    m1
end)

println("RESULT: ", t.value)
@show bits, pieces, N, t.value, t.time
"""
        name = fname(fam, v)
        out.append((name, write(name, text)))
    return fam, out


# ---------------------------------------------------------------------------
# hmm_factorial: two independent K=3 chains; emission mean index via pick_mu map.
# ---------------------------------------------------------------------------
def gen_hmm_factorial():
    fam = "hmm_factorial"
    pick = CONST["hmm_factorial"]["pick_mu"]   # {"z,h": idx} 0-indexed

    def midx(z, h):                            # -> Julia m[] 1-based index
        return pick[f"{z},{h}"] + 1

    # build the nested ifelse exactly from the pick_mu map
    pick_def = (
        "pick_prod(z, h, m) =\n"
        "    ifelse(prob_equals(z, U(0)),\n"
        f"        ifelse(prob_equals(h, U(0)), m[{midx(0,0)}], ifelse(prob_equals(h, U(1)), m[{midx(0,1)}], m[{midx(0,2)}])),\n"
        "      ifelse(prob_equals(z, U(1)),\n"
        f"        ifelse(prob_equals(h, U(0)), m[{midx(1,0)}], ifelse(prob_equals(h, U(1)), m[{midx(1,1)}], m[{midx(1,2)}])),\n"
        f"        ifelse(prob_equals(h, U(0)), m[{midx(2,0)}], ifelse(prob_equals(h, U(1)), m[{midx(2,1)}], m[{midx(2,2)}]))))"
    )
    out = []
    for v in VALUES[fam]:
        obs = DATA[fam][str(v)]
        text = f"""using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

# Reference model: pluck-programs/{fam}.pluck (and shared/{fam}.pluck); SCALING N={v}.
# Factorial HMM: two independent K=3 chains z_n, h_n; emission mean indexed by the
# pick_mu (z,h)->index map (9 SHARED means).
#   mu_k ~ N(0,1);  z_1,h_1 ~ cat{INIT};  y_n ~ N(mu_{{pick(z_n,h_n)}}, {EMIT_SD})
# Exact inference enumerates BOTH chains: ~9^N -- VERY steep. N and obs baked in;
# pass `bits pieces`. Reports E[mu_1].
#
# CAVEAT (intractability, per base hmm_factorial.jl): the 9 shared bit-blasted
# means are entangled through the emission mux, blowing up the BDD (same
# continuous-shared-mean wall as normal_mixture/soft_kmeans). pluck integrates the
# conjugate means and stays fast. Kept as the faithful structural translation.

if length(ARGS) == 0
    bits = 2
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = {v}

DFiP = DistFix{{bits + 5, bits}}
obs = {jl_floats(obs)}

U(k) = DistUInt{{2}}(k)

{hmm_transition_def()}

# mu index from the pick_mu (z,h) map in scaling_data.json constants.
{pick_def}

t = @timed_infer expectation(@dice begin
    m = [bitblast(DFiP, Normal(-3, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(-1.5, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(-0.5, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(1, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(2, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(0, 1), pieces, -8.0, 8.0),
         bitblast(DFiP, Normal(3.5, 1), pieces, -8.0, 8.0)]
    z = discrete(DistUInt{{2}}, {INIT})
    h = discrete(DistUInt{{2}}, {INIT})
    for i in 1:N
        mu = pick_prod(z, h, m)
        gaussian_observe(DFiP, pieces, -8.0, 8.0, mu, {EMIT_SD}, DFiP(obs[i]), add=true)
        z = transition(z)
        h = transition(h)
    end
    m[1]
end)

println("RESULT: ", t.value)
@show bits, pieces, N, t.value, t.time
"""
        name = fname(fam, v)
        out.append((name, write(name, text)))
    return fam, out


# ---------------------------------------------------------------------------
# outliers: robust linear regression, per-point inlier/outlier flips.
# ---------------------------------------------------------------------------
def gen_outliers():
    fam = "outliers"
    o = CONST["outliers"]
    ab_sd = o["alpha_beta_prior"][1]          # N(0,10) -> sd 10
    sig_in = o["sigma_inlier"]                # 0.5
    sig_out = o["sigma_outlier"]              # 10.0
    out = []
    for v in VALUES[fam]:
        payload = DATA[fam][str(v)]
        xs = [int(x) for x in payload["xs"]]  # integer predictors (scale by add)
        ys = payload["ys"]
        planted = payload["planted"]
        text = f"""using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

# Reference model: pluck-programs/{fam}.pluck; SCALING N={v}.
# Robust linear regression with per-point outlier indicators.
#   alpha, beta ~ N(0,{int(ab_sd)});  pi ~ Beta(1,1);  o_n ~ Bernoulli(pi)
#   y_n ~ N(alpha*x_n + beta, sigma_{{o_n}})   sigma_inlier={sig_in}, sigma_outlier={sig_out}
#   True line y = 2x+1; planted outliers at x in {planted}.
# xs/ys baked in from scaling_data.json. Reports E[alpha] (true 2.0).
#
# CAVEAT (intractability, per base outliers.jl): Dice has no DistFix*Float, so
# alpha*x is built by REPEATED ADDITION (exact, integer x only). Even so, the mean
# alpha*x+beta is a wide-range linear combination of bit-blasted Gaussians -- the
# fixed-point range blow-up that makes polyreg/spike_and_slab intractable in Dice.
# pluck stays in the conjugate linear-Gaussian path and solves it exactly.
# Wide DistFix range for alpha*x+beta; pass `bits pieces`.

if length(ARGS) == 0
    bits = 3
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = {v}

DFiP = DistFix{{bits + 11, bits}}     # wide integer range for alpha*x + beta
xs = {jl_floats(xs)}                  # integer predictors (so alpha*x = sum of copies)
ys = {jl_floats(ys)}

# alpha * k for integer k, via repeated addition (Dice has no DistFix * Float)
scale_int(g, k) = foldl(+, fill(g, k))

t = @timed_infer expectation(@dice begin
    alpha = bitblast(DFiP, Normal(0, {ab_sd}), pieces, -32.0, 32.0)
    beta  = bitblast(DFiP, Normal(0, {ab_sd}), pieces, -32.0, 32.0)
    pi    = uniform(DFiP, 0.0, 1.0)
    for i in 1:N
        o = parametrised_flip(pi)
        mu = scale_int(alpha, xs[i]) + beta
        noise = ifelse(o, bitblast(DFiP, Normal(0, {sig_out}), pieces, -32.0, 32.0),
                          bitblast(DFiP, Normal(0, {sig_in}), pieces, -4.0, 4.0))
        observe(prob_equals(noise + mu, DFiP(ys[i])))
    end
    alpha
end)

println("RESULT: ", t.value)
@show bits, pieces, N, t.value, t.time
"""
        name = fname(fam, v)
        out.append((name, write(name, text)))
    return fam, out


# ---------------------------------------------------------------------------
# soft_kmeans: K=3, D=3, mu ~ N(0,3), mixing [.34,.33,.33], emission sd 1.
# ---------------------------------------------------------------------------
def gen_soft_kmeans():
    fam = "soft_kmeans"
    sk = CONST["soft_kmeans"]
    mu_sd = sk["mu_prior"][1]                 # 3.0  (pluck uses N(0,3), NOT N(0,1))
    mixing = sk["mixing"]                     # [0.34, 0.33, 0.33]
    emit = sk["emission_sd"]                  # 1.0
    RANGE = 32.0                              # widened from +/-8 to cover N(0,3)
    out = []
    for v in VALUES[fam]:
        obs = DATA[fam][str(v)]["obs"]        # list of [d0,d1,d2]
        o0 = [row[0] for row in obs]
        o1 = [row[1] for row in obs]
        o2 = [row[2] for row in obs]
        text = f"""using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

# Reference model: pluck-programs/{fam}.pluck; SCALING N={v}.
# Soft K-means / Gaussian mixture: K=3 clusters, D=3 dims.
#   mu_{{k,d}} ~ N(0,{int(mu_sd)})  (9 SHARED means);  z_n ~ cat{mixing};
#   y_{{n,d}} ~ N(mu_{{z_n,d}}, {emit})
# Exact inference enumerates 3^N cluster assignments -- keep N small.
# obs are the 12-point 3-D corpus prefix from scaling_data.json; pass `bits pieces`.
# Reports E[mu_{{0,0}}] (cluster 0, dim 0); under label switching averages toward 0.
#
# PRIOR FIX vs the hand-written base: pluck uses cluster-mean prior N(0,{int(mu_sd)})
# (the old base used N(0,1)) and this 12-point corpus (old base used a 4-point set).
# The bit-blast range is widened to +/-{int(RANGE)} (was +/-8) to cover N(0,{int(mu_sd)}).
#
# CAVEAT (intractability, per base soft_kmeans.jl): the 9 shared bit-blasted means
# are entangled through the per-point emission mux, so the BDD blows up -- the same
# continuous-shared-mean wall that limits normal_mixture, here with 9 means. pluck
# integrates the conjugate means and does the full corpus in well under a second.

if length(ARGS) == 0
    bits = 2
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = {v}

DFiP = DistFix{{bits + 7, bits}}     # wider integer range to cover N(0,{int(mu_sd)})
o0 = {jl_floats(o0)}
o1 = {jl_floats(o1)}
o2 = {jl_floats(o2)}

U(k) = DistUInt{{2}}(k)
pick3(z, a, b, c) = ifelse(prob_equals(z, U(0)), a, ifelse(prob_equals(z, U(1)), b, c))

t = @timed_infer expectation(@dice begin
    # m[3c+d+1] is cluster c (0..2), dim d (0..2); prior N(0,{int(mu_sd)}).
    m = [bitblast(DFiP, Normal(-3, {mu_sd}), pieces, -{RANGE}, {RANGE}),
         bitblast(DFiP, Normal(-3, {mu_sd}), pieces, -{RANGE}, {RANGE}),
         bitblast(DFiP, Normal(-3, {mu_sd}), pieces, -{RANGE}, {RANGE}),
         bitblast(DFiP, Normal(3, {mu_sd}), pieces, -{RANGE}, {RANGE}),
         bitblast(DFiP, Normal(3, {mu_sd}), pieces, -{RANGE}, {RANGE}),
         bitblast(DFiP, Normal(3, {mu_sd}), pieces, -{RANGE}, {RANGE}),
         bitblast(DFiP, Normal(3, {mu_sd}), pieces, -{RANGE}, {RANGE}),
         bitblast(DFiP, Normal(-3, {mu_sd}), pieces, -{RANGE}, {RANGE}),
         bitblast(DFiP, Normal(-3, {mu_sd}), pieces, -{RANGE}, {RANGE})]
    for n in 1:N
        z = discrete(DistUInt{{2}}, {mixing})
        mu0 = pick3(z, m[1], m[4], m[7])
        mu1 = pick3(z, m[2], m[5], m[8])
        mu2 = pick3(z, m[3], m[6], m[9])
        gaussian_observe(DFiP, pieces, -{RANGE}, {RANGE}, mu0, {emit}, DFiP(o0[n]), add=true)
        gaussian_observe(DFiP, pieces, -{RANGE}, {RANGE}, mu1, {emit}, DFiP(o1[n]), add=true)
        gaussian_observe(DFiP, pieces, -{RANGE}, {RANGE}, mu2, {emit}, DFiP(o2[n]), add=true)
    end
    m[1]
end)

println("RESULT: ", t.value)
@show bits, pieces, N, t.value, t.time
"""
        name = fname(fam, v)
        out.append((name, write(name, text)))
    return fam, out


# ---------------------------------------------------------------------------
# changepoint_coal (VARIABLE-k): scaled copies of the hand-written base port
# hybit-programs/changepoint_coal.jl (which stays untouched at its reduced N=4).
# Only N and the baked `counts` prefix change with the size; deliberate deltas
# vs the base: no benchmarks/.../results.txt write (run_hybit mkdirs a scratch
# dir PER VARIANT NAME, so the base's relative path would fail here) and a
# RESULT: line printing E[k] (compute_error reduces pluck's posterior over k
# to its mean).
# ---------------------------------------------------------------------------
def gen_changepoint_coal():
    fam = "changepoint_coal"
    cc = CONST[fam]
    alpha, beta = cc["alpha"], cc["beta"]      # segment rates h_g ~ Gamma(2,1)
    kp = cc["k_prior"]                         # truncated Poisson(1) on {0..3}
    ktot = sum(float(x) for x in kp.values())  # 2.75
    kw = ", ".join(f"{float(kp[str(i)])}/{ktot}" for i in range(4))
    out = []
    for v in VALUES[fam]:
        counts = DATA[fam][str(v)]["counts"]
        text = f"""using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

# Reference model: pluck-programs/changepoint_coal_n{v:03d}.pluck (variable-k);
# base port (verified at N=4 vs pluck/PSI): hybit-programs/changepoint_coal.jl.
# SCALING N={v} years.
#   k ~ truncated Poisson(1) on {{0,1,2,3}}; changepoints = uniform random
#   k-subset of the N-1 boundaries (sequential flips); up to 4 segment rates
#   h_g ~ Gamma({alpha},{beta}); counts[i] ~ Poisson(h_seg(i)).
# WHY EXPRESSIBLE: k is BOUNDED, so the transdimensional model is a finite
# mixture -- pad to 4 rates (unused ones marginalize out), sequential placement
# flips, and the enumerate-the-grid poisson_observe (see the base header).
# MEASURED (run_20260708_141708 + _164113, 600s budget, 15/15 reps each): at
# this coarse grid n010/n015/n020 COMPLETE (infer ~127/276/472s) but the rate
# grid biases the posterior hard as N grows (E[k]=0.52/0.40/0.21 vs pluck's
# exact 0.86/0.82/0.50, rel. err 40-57% -- the table's error gate shows r/e);
# n025 is the first TIMEOUT size. The finer grid that matches pluck at the
# base's N=4 (F=3/pieces=64) is OOM-killed by N=6. So Hybit's variable-k sweep
# is either inaccurate (coarse grid) or intractable (fine grid); pluck runs the
# full-N sweep exactly.
# Reports E[k] (RESULT) and the full posterior over k (@show).

if length(ARGS) == 0
    bits   = 2        # fractional bits F (rate grid step = 1/2^F)
    pieces = 32       # gamma bit-blast pieces (must be <= (stop)*2^F)
else
    bits   = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = {v}

F = bits
W = 5 + F                 # signed range [-16,16): covers the rate support [0,8]
DFiP = DistFix{{W, F}}
stop = 8.0

counts = {list(counts)}

U(x) = DistUInt{{2}}(x)             # k, kleft, seg all live in {{0,1,2,3}}
b2u(c) = ifelse(c, U(1), U(0))    # Bool -> {{0,1}}

# Poisson observation of integer `c` against a bit-blasted continuous rate
# (enumerate the rate grid, inject the pmf as a flip weight; cf. the base port).
function poisson_observe(rate::DistFix{{W,F}}, c::Int, stop::Float64) where {{W,F}}
    step = 1/2^F; r = 0.0
    while r <= stop + 1e-9
        w = r > 0 ? pdf(Poisson(r), c) : (c == 0 ? 1.0 : 0.0)
        a = prob_equals(rate, DistFix{{W,F}}(r))
        observe(!a | (a & flip(w)))
        r += step
    end
end

# cut indicator at a boundary with `rem` boundaries remaining (constant), given a
# random `kleft`: cut w.p. kleft/rem. Enumerated over kleft in 1..min(3,rem) so
# every constructed flip probability is <= 1 (invariant kleft <= rem).
function place_flip(kleft, rem)
    expr = flip(0.0)                  # kleft==0 -> never cut
    for kl in 1:min(3, rem)
        expr = ifelse(prob_equals(kleft, U(kl)), flip(kl / rem), expr)
    end
    expr
end

t = @timed_infer pr(@dice begin
    h1 = bitblast(DFiP, Gamma({alpha},{beta}), pieces, 0.0, stop)
    h2 = bitblast(DFiP, Gamma({alpha},{beta}), pieces, 0.0, stop)
    h3 = bitblast(DFiP, Gamma({alpha},{beta}), pieces, 0.0, stop)
    h4 = bitblast(DFiP, Gamma({alpha},{beta}), pieces, 0.0, stop)
    k     = discrete(DistUInt{{2}}, [{kw}])  # trunc Poisson(1)
    kleft = k
    seg   = U(0)
    poisson_observe(h1, counts[1], stop)        # year 1 is always in segment 0
    for i in 2:N
        rem = N - i + 1                          # boundaries remaining (incl. this one)
        cut = place_flip(kleft, rem)             # changepoint at boundary (i-1)?
        kleft = kleft - b2u(cut)
        seg   = seg + b2u(cut)
        rate = ifelse(prob_equals(seg, U(0)), h1,
               ifelse(prob_equals(seg, U(1)), h2,
               ifelse(prob_equals(seg, U(2)), h3, h4)))
        poisson_observe(rate, counts[i], stop)
    end
    k
end)

p = sum(k*v for (k,v) in t.value)   # E[k] -- the cross-PPL error scalar
println("RESULT: ", p)
@show bits, pieces, N, t.value, t.time
"""
        name = fname(fam, v)
        out.append((name, write(name, text)))
    return fam, out


# ---------------------------------------------------------------------------
# king_ei (pluck-only port): beta-binomial ecological inference.
# Beta(1,1) rates = uniform(DFiP,0,1); Binomial(n,p) = sum of n parametrised_flips;
# observe T = countflips(Nw/D, bw) + countflips(Nb/D, bb), clamped to Nw/D+Nb/D.
# ---------------------------------------------------------------------------
def gen_king_ei():
    fam = "king_ei"
    out = []
    for v in VALUES[fam]:                      # v is the data-divisor D
        precincts = DATA[fam][str(v)]["precincts"]   # [[Nw, Nb, T], ...]
        pj = "[" + ", ".join(f"({nw}, {nb}, {t})" for nw, nb, t in precincts) + "]"
        text = f"""using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

# Reference model: pluck-programs/shared/king_ei.pluck + king_ei.template.pluck;
# SCALING data-divisor D={v} (SMALLER D = MORE data = slower; cost ~ trial count).
# King (1999) beta-binomial ecological inference (Waterbury, CT precincts).
#   bw, bb ~ Beta(1,1)                         district-wide White / non-White rates
#   kw_i ~ Binomial(Nw_i/D, bw), kb_i ~ Binomial(Nb_i/D, bb)   LATENT cell counts
#   T_i  = kw_i + kb_i                         OBSERVED margin (clamped to Nw/D+Nb/D)
# Beta(1,1) = Uniform[0,1], so the rates are uniform(DFiP,0,1); a Binomial(n,p) is
# the count of `true`s among n parametrised_flip(p) draws (Beta-Bernoulli, exactly
# the coinBias.jl idiom). Reports E[bw] (the White Malloy rate). Counts are
# integer-divided by D before inference (the full ~1100-trial counts are
# intractable); this is the faithful Dice port of the pluck exact model.

if length(ARGS) == 0
    bits = 6
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
D = {v}

DFiP = DistFix{{bits + 2, bits}}     # [0,1]: 2 int bits so p=1.0 is representable; counts live in DistUInt
precincts = {pj}    # (Nw, Nb, T) per precinct -- REAL unscaled counts

idiv(x, d) = div(x, d)

# Binomial(n, p): count `true`s among n flips of a shared bit-blasted p.
function count_flips(p, n)
    c = DistUInt{{16}}(0)
    for _ in 1:n
        c = ifelse(parametrised_flip(p), c + DistUInt{{16}}(1), c)
    end
    c
end

t = @timed_infer expectation(@dice begin
    bw = uniform(DFiP, 0.0, 1.0)     # Beta(1,1)
    bb = uniform(DFiP, 0.0, 1.0)     # Beta(1,1)
    for pr in precincts
        wc = idiv(pr[1], D)
        bc = idiv(pr[2], D)
        tc = min(idiv(pr[3], D), wc + bc)
        total = count_flips(bw, wc) + count_flips(bb, bc)
        observe(prob_equals(total, DistUInt{{16}}(tc)))
    end
    bw
end)

println("RESULT: ", t.value)   # E[bw] — White Malloy rate, the cross-PPL error scalar
@show bits, pieces, D, t.value, t.time
"""
        name = fname(fam, v)
        out.append((name, write(name, text)))
    return fam, out


# ---------------------------------------------------------------------------
# dawid_skene (pluck-only port): crowd-annotation, Beta priors on confusion rates.
# z_i ~ Bern(prevalence); annotation iff parametrised_flip(z?tp_j:fp_j) == label.
# ---------------------------------------------------------------------------
def gen_dawid_skene():
    fam = "dawid_skene"
    ds = CONST["dawid_skene"]
    a_pre, b_pre = ds["prevalence_prior"]      # [1,1]
    a_tp, b_tp = ds["tp_prior"]                # [2,1]
    a_fp, b_fp = ds["fp_prior"]                # [1,2]
    out = []
    for v in VALUES[fam]:                      # v = n-items I
        payload = DATA[fam][str(v)]
        J = payload["n_annotators"]
        annots = payload["annotations"]        # [[i, j, bool], ...]
        # Julia is 1-based: store (i+1, j+1, label).
        aj = "[" + ", ".join(
            f"({i + 1}, {j + 1}, {'true' if lab else 'false'})" for i, j, lab in annots
        ) + "]"
        text = f"""using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

# Reference model: pluck-programs/generation_scripts/dawid_skene.template.pluck;
# SCALING n-items I={v} (J={J} annotators). cost ~ 2^I (enumerate true labels).
# Dawid & Skene (1979) crowd-annotation model with Beta priors on confusion rates.
#   prevalence ~ Beta{[a_pre, b_pre]}                  P(true label = 1), shared
#   tp_j ~ Beta{[a_tp, b_tp]}, fp_j ~ Beta{[a_fp, b_fp]}    per-annotator rates
#   z_i  ~ Bernoulli(prevalence)               item i's TRUE label  (DISCRETE latent)
#   r_ij ~ Bernoulli(z_i ? tp_j : fp_j)        annotation, OBSERVED when present
# Continuous Beta rates via bitblast_exponential(Beta,...); Bernoullis via
# parametrised_flip (the coinBias.jl idiom). Reports E[tp_0] (first annotator's
# sensitivity). Annotations are the I x J prefix from scaling_data.json.

if length(ARGS) == 0
    bits = 5
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
I = {v}
J = {J}

DFiP = DistFix{{bits + 2, bits}}     # [0,1]: 2 int bits so p=1.0 (Beta endpoint) is representable
# annotations as (item, annotator, label) -- 1-based item/annotator indices.
annotations = {aj}

t = @timed_infer expectation(@dice begin
    prevalence = bitblast_exponential(DFiP, Beta({a_pre}, {b_pre}), pieces, 0.0, 1.0)
    tp = [bitblast_exponential(DFiP, Beta({a_tp}, {b_tp}), pieces, 0.0, 1.0) for _ in 1:J]
    fp = [bitblast_exponential(DFiP, Beta({a_fp}, {b_fp}), pieces, 0.0, 1.0) for _ in 1:J]
    # z in an EXPLICIT loop: a comprehension body compiles outside the @dice
    # IRTools dynamo, so parametrised_flip's guard evaluates a DistBool in Julia
    # Bool context (TypeError at Alea util.jl:67). Pure-constructor comprehensions
    # (the bitblasts above) are safe.
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
"""
        name = fname(fam, v)
        out.append((name, write(name, text)))
    return fam, out


# ---------------------------------------------------------------------------
# secret_ballot (pluck-only port): committee inference from public majorities.
# p_m ~ Beta(1,1) = uniform; per-ballot yes-tally = sum of parametrised_flip(p_m)
# over the (repeat-allowed) participant idxs; observe 2*yes > |subset|.
# ---------------------------------------------------------------------------
def gen_secret_ballot():
    fam = "secret_ballot"
    out = []
    for v in VALUES[fam]:                      # v = n-members N
        payload = DATA[fam][str(v)]
        ballots = payload["ballots"]           # [[idxs, outcome], ...]
        # idxs are 0-based member indices; Julia 1-based.
        bj = "[" + ", ".join(
            "([" + ", ".join(str(i + 1) for i in idxs) + "], "
            + ("true" if outcome else "false") + ")"
            for idxs, outcome in ballots
        ) + "]"
        text = f"""using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

# Reference model: pluck-programs/secret_ballot_n*.pluck; SCALING n-members N={v}.
# Secret-ballot committee inference: we observe ONLY each ballot's yes/no majority,
# never the individual votes (subset size 5, window stride 2).
#   p_m ~ Beta(1,1)                            each member's yes-tendency
#   each ballot: members in `idxs` cast secret yes-votes ~ Bernoulli(p_m);
#   OBSERVED = strict yes-majority (2*yes > |subset|).
# Beta(1,1) = Uniform[0,1] -> uniform(DFiP,0,1); votes via parametrised_flip.
# Reports E[p_0] (first member's tendency), marginalising the secret ballots and
# every other member. Ballots/idxs from scaling_data.json (idxs are 1-based here).

if length(ARGS) == 0
    bits = 6
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = {v}

DFiP = DistFix{{bits + 2, bits}}     # [0,1]: 2 int bits so p=1.0 is representable
# (participant idxs (1-based), observed yes-majority) per ballot.
ballots = {bj}

# yes-tally among the members in idxs: each casts an independent Bernoulli(p_m).
function count_yes(ps, idxs)
    c = DistUInt{{8}}(0)
    for i in idxs
        c = ifelse(parametrised_flip(ps[i]), c + DistUInt{{8}}(1), c)
    end
    c
end

t = @timed_infer expectation(@dice begin
    ps = [uniform(DFiP, 0.0, 1.0) for _ in 1:N]   # Beta(1,1) per member
    for bal in ballots
        idxs = bal[1]; outcome = bal[2]
        yes = count_yes(ps, idxs)
        # strict yes-majority: 2*yes > |subset|
        maj = DistUInt{{8}}(2) * yes > DistUInt{{8}}(length(idxs))
        observe(maj == outcome)
    end
    ps[1]
end)

@show bits, pieces, N, t.value, t.time
"""
        name = fname(fam, v)
        out.append((name, write(name, text)))
    return fam, out


# ---------------------------------------------------------------------------
# sir_model (SIR epidemic HMM; dial N = TOTAL observations = 12 * rows). Nested-loop
# port of the hand-written hybit-programs/sir_model.jl: each row is an independent
# 12-step infection chain restarting at not-infected; all rows share the 4
# bit-blasted Beta rates. Measured (600 s budget): infer n12 ~85s, n24 ~185s,
# n36 ~246s, n48 ~379s, n60 ~562s; n72 times out. From n24 up the RESULT value is
# unreliable -- an Alea/Dice engine artifact, not a port bug: swapping the two
# n24 data rows (a no-op for the likelihood) reproducibly changes E[inf]
# 0.2569 <-> 0.3493, while at n12 two isomorphic programs agree to 16 digits.
# ---------------------------------------------------------------------------
def gen_sir_model():
    fam = "sir_model"
    out = []
    for v in VALUES[fam]:
        rows = DATA[fam][str(v)]["histories"]
        R = len(rows)
        hrows = "[" + ",\n     ".join(jl_bools(r) for r in rows) + "]"
        text = f"""using Dice, Distributions
include(joinpath(@__DIR__, "timing", "bench_util.jl"))

# Reference model: hybit-programs/sir_model.jl (base = 1 row x 12 steps); SCALING
# N={v} ({R} rows x 12 steps). SIR epidemic HMM: each row is an independent 12-step
# infection chain restarting at not-infected; all rows share 4 continuous Beta rates
#   rec ~ Beta[1.0, 1.0]   recovery rate     P(recover | infected)
#   inf ~ Beta[1.0, 1.0]   infection rate    P(infect  | susceptible)
#   bas ~ Beta[0.2, 1.0]   baseline symptom rate (when not infected)
#   sym ~ Beta[1.0, 1.0]   infected symptom rate
# via bitblast_exponential; Bernoullis via parametrised_flip; emission from the
# CURRENT state, then transition. Reports E[inf] (the cross-PPL error scalar).
# The transition rates are CONTINUOUS latents, so the exact BDD grows ~4.5x per
# WITHIN-CHAIN step; across rows the hidden chain resets and growth is roughly
# additive (measured infer: n12 85s, n24 185s, n36 246s, n48 379s, n60 562s;
# n72 = first 600s TIMEOUT -- the honest wall). WARNING: from n24 up the RESULT
# value is unreliable (Alea engine artifact, not a port bug): swapping the data
# rows -- mathematically a no-op -- reproducibly changes E[inf] (0.2569 vs 0.3493
# at n24), while at n12 isomorphic programs agree to 16 digits. Timings stay
# meaningful; treat the n24+ result column with suspicion.

if length(ARGS) == 0
    bits = 4
    pieces = 8
else
    bits = parse(Int64, ARGS[1])
    pieces = parse(Int64, ARGS[2])
end
N = {v}

DFiP = DistFix{{bits + 2, bits}}     # [0,1]: 2 int bits so p=1.0 (Beta endpoint) is representable
H = {hrows}

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
"""
        name = fname(fam, v)
        out.append((name, write(name, text)))
    return fam, out


# ---------------------------------------------------------------------------
def main():
    global write
    args = parse_args(__doc__, os.path.join(TRANSLATIONS, "hybit"))
    write = Writer(args.out, args.overwrite, "#")
    for gen in (gen_hmm_first_order, gen_hmm_second_order, gen_hmm_factorial,
                gen_outliers, gen_soft_kmeans, gen_changepoint_coal, gen_king_ei,
                gen_dawid_skene, gen_secret_ballot, gen_sir_model):
        gen()
    write.report()


if __name__ == "__main__":
    main()
