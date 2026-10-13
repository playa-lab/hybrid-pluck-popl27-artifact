#!/usr/bin/env python3
"""Generate the scaled PSI (.psi) benchmark programs.

For every scaling family PSI can express, emits one program per size of the
Pluck sweep (<family>_<n|d><size>.psi), embedding exactly the data the Pluck
program at that size uses (scaling_data.json). Model hyperparameters match the
Pluck programs.

PSI note: gauss(mean, VARIANCE) takes the variance (sd^2), not the sd.

Not generated (not expressible in PSI): pcfg (unbounded recursion).

Run from anywhere:
    python3 generate_scaling_psi.py [--out DIR] [--overwrite]
"""

import os

from gen_common import CONST, DATA, TRANSLATIONS, VALUES, Writer, parse_args
from gen_common import fname as _fname

write = None  # set in main()


def fname(family, v):
    return _fname(family, v, "psi")


def reals(xs):
    return "[" + ", ".join(repr(float(x)) for x in xs) + "]"


def cat(weights):
    return "categorical([" + ", ".join(str(w) for w in weights) + "])"


INIT = CONST["hmm"]["init"]
TRANS = CONST["hmm"]["trans"]


# hmm_first_order: K=3-state HMM with Gaussian emissions; distinct priors on the
# three emission means break label symmetry. Returns E[m1].
def hmm_first_order(family):
    for v in VALUES[family]:
        obs = DATA[family][str(v)]
        trans_branches = (f"if z == 0 {{ {cat(TRANS[0])} }}\n"
                          f"            else {{ if z == 1 {{ {cat(TRANS[1])} }} else {{ {cat(TRANS[2])} }} }}")
        text = f"""def main() {{
    obs := {reals(obs)};
    m1 := gauss(-2,1); m2 := gauss(0,1); m3 := gauss(2,1);

    z := {cat(INIT)};
    for n in [0..{len(obs)}) {{
        mu := if z == 0 {{ m1 }} else {{ if z == 1 {{ m2 }} else {{ m3 }} }};
        cobserve(gauss(mu, 1), obs[n]);
        z = {trans_branches};
    }}
    return Expectation(m1);
}}
"""
        yield fname(family, v), write(fname(family, v), text), len(obs)


# hmm_second_order: like hmm_first_order, but z_n ~ trans[max(z_{n-2}, z_{n-1})].
def hmm_second_order(family):
    for v in VALUES[family]:
        obs = DATA[family][str(v)]
        n = len(obs)
        text = f"""def trans(prev) {{
    return if prev == 0 {{ {cat(TRANS[0])} }}
           else {{ if prev == 1 {{ {cat(TRANS[1])} }} else {{ {cat(TRANS[2])} }} }};
}}

def main() {{
    obs := {reals(obs)};
    m1 := gauss(-2,1); m2 := gauss(0,1); m3 := gauss(2,1);

    z1 := {cat(INIT)};
"""
        if n == 1:
            text += """    cobserve(gauss(if z1 == 0 { m1 } else { if z1 == 1 { m2 } else { m3 } }, 1), obs[0]);
"""
        else:
            text += """    z2 := trans(z1);
    cobserve(gauss(if z1 == 0 { m1 } else { if z1 == 1 { m2 } else { m3 } }, 1), obs[0]);
    cobserve(gauss(if z2 == 0 { m1 } else { if z2 == 1 { m2 } else { m3 } }, 1), obs[1]);

    zp2 := z1;
    zp1 := z2;
    for n in [2..%d) {
        mx := if zp2 >= zp1 { zp2 } else { zp1 };
        z := trans(mx);
        cobserve(gauss(if z == 0 { m1 } else { if z == 1 { m2 } else { m3 } }, 1), obs[n]);
        zp2 = zp1;
        zp1 = z;
    }
""" % n
        text += """    return Expectation(m1);
}
"""
        yield fname(family, v), write(fname(family, v), text), n


# hmm_factorial (Multi-Chain HMM): two independent 3-state chains z, h; each
# emission's mean is picked from 9 means by (z, h) (pick_mu map). The means have
# distinct priors N(true_mu_k, 1), as in the Pluck program. Returns E[m1].
def hmm_factorial(family):
    mus = CONST["hmm_factorial"]["true_mu"]
    prior_lines = "\n    ".join(
        " ".join(f"m{k + 1} := gauss({mus[k]:g},1);" for k in range(row, row + 3))
        for row in (0, 3, 6))
    for v in VALUES[family]:
        obs = DATA[family][str(v)]
        step = "    cobserve(gauss(pickMu(z, h, m1, m2, m3, m4, m5, m6, m7, m8, m9), 1), obs[%d]);\n"
        steps = step % 0 + "".join(
            "\n    z = trans(z); h = trans(h);\n" + step % n for n in range(1, len(obs)))
        text = f"""def trans(prev) {{
    return if prev == 0 {{ {cat(TRANS[0])} }}
           else {{ if prev == 1 {{ {cat(TRANS[1])} }} else {{ {cat(TRANS[2])} }} }};
}}

def pickMu(z, h, m1, m2, m3, m4, m5, m6, m7, m8, m9) {{
    return if z == 0 {{
             if h == 0 {{ m1 }} else {{ if h == 1 {{ m2 }} else {{ m3 }} }}
           }} else {{ if z == 1 {{
             if h == 0 {{ m2 }} else {{ if h == 1 {{ m4 }} else {{ m6 }} }}
           }} else {{
             if h == 0 {{ m3 }} else {{ if h == 1 {{ m6 }} else {{ m9 }} }}
           }} }};
}}

def main() {{
    obs := {reals(obs)};
    {prior_lines}

    z := {cat(INIT)};
    h := {cat(INIT)};
{steps}
    return Expectation(m1);
}}
"""
        yield fname(family, v), write(fname(family, v), text), len(obs)


# outliers: robust linear regression with a per-point outlier flag.
#   alpha, intercept ~ N(0,10); pOut ~ Beta(1,1); o_n ~ Bernoulli(pOut);
#   y_n ~ N(alpha*x_n + intercept, 0.5 if inlier else 10).
def outliers(family):
    ab = CONST["outliers"]["alpha_beta_prior"]
    prior_var = ab[1] ** 2
    s_in = CONST["outliers"]["sigma_inlier"] ** 2
    s_out = CONST["outliers"]["sigma_outlier"] ** 2
    pi = CONST["outliers"]["pi_prior"]
    for v in VALUES[family]:
        d = DATA[family][str(v)]
        xs, ys = d["xs"], d["ys"]
        n = len(xs)
        text = f"""def main() {{
    xs := {reals(xs)};
    ys := {reals(ys)};

    alpha     := gauss(0, {prior_var:g});
    intercept := gauss(0, {prior_var:g});
    pOut      := beta({pi[0]:g}, {pi[1]:g});

    for i in [0..{n}) {{
        o := flip(pOut);
        mu := alpha*xs[i] + intercept;
        if o {{
            cobserve(gauss(mu, {s_out:g}), ys[i]);
        }} else {{
            cobserve(gauss(mu, {s_in:g}), ys[i]);
        }}
    }}
    return (alpha, intercept);
}}
"""
        yield fname(family, v), write(fname(family, v), text), n


# soft_kmeans: Gaussian mixture, K=3 clusters in D=3 dimensions; cluster means
# have distinct N(+-3, 3) priors; observations are given per dimension.
def soft_kmeans(family):
    sk = CONST["soft_kmeans"]
    D = sk["D"]
    prior_var = sk["mu_prior"][1] ** 2
    emit_var = sk["emission_sd"] ** 2
    mixing = sk["mixing"]
    for v in VALUES[family]:
        obs = DATA[family][str(v)]["obs"]
        n = len(obs)
        cols = [[row[d] for row in obs] for d in range(D)]
        col_lines = "\n    ".join(f"o{d} := {reals(cols[d])};" for d in range(D))
        text = f"""def main() {{
    {col_lines}

    m00 := gauss(-3,{prior_var:g}); m01 := gauss(-3,{prior_var:g}); m02 := gauss(-3,{prior_var:g});
    m10 := gauss(3,{prior_var:g}); m11 := gauss(3,{prior_var:g}); m12 := gauss(3,{prior_var:g});
    m20 := gauss(3,{prior_var:g}); m21 := gauss(-3,{prior_var:g}); m22 := gauss(-3,{prior_var:g});

    for n in [0..{n}) {{
        z := {cat(mixing)};
        mu0 := if z == 0 {{ m00 }} else {{ if z == 1 {{ m10 }} else {{ m20 }} }};
        mu1 := if z == 0 {{ m01 }} else {{ if z == 1 {{ m11 }} else {{ m21 }} }};
        mu2 := if z == 0 {{ m02 }} else {{ if z == 1 {{ m12 }} else {{ m22 }} }};
        cobserve(gauss(mu0, {emit_var:g}), o0[n]);
        cobserve(gauss(mu1, {emit_var:g}), o1[n]);
        cobserve(gauss(mu2, {emit_var:g}), o2[n]);
    }}
    return (m00, m01, m02, m10, m11, m12, m20, m21, m22);
}}
"""
        yield fname(family, v), write(fname(family, v), text), n


# king_ei: King (1999) beta-binomial ecological inference. bw, bb ~ Beta(1,1);
# per precinct (counts integer-divided by the dial D), observe
# Binomial(nw/D, bw) + Binomial(nb/D, bb) == min(t/D, nw/D + nb/D).
def king_ei(family):
    bp = CONST["king_ei"]["beta_prior"]
    for v in VALUES[family]:
        D = int(v)
        prec = DATA[family][str(v)]["precincts"]
        rows = []
        for (nw, nb, t) in prec:
            wc = nw // D
            bc = nb // D
            tt = min(t // D, wc + bc)
            rows.append((wc, bc, tt))
        prec_lines = "\n    ".join(
            f"observePrecinct(bw, bb, {wc}, {bc}, {tt});" for (wc, bc, tt) in rows)
        text = f"""def observePrecinct(bw, bb, wc, bc, tt) {{
    observe(binomial(wc, bw) + binomial(bc, bb) == tt);
}}

def main() {{
    bw := beta({bp[0]:g}, {bp[1]:g});
    bb := beta({bp[0]:g}, {bp[1]:g});

    {prec_lines}

    return bw;
}}
"""
        yield fname(family, v), write(fname(family, v), text), len(prec)


# dawid_skene: Dawid & Skene (1979) crowd annotation. prevalence ~ Beta(1,1);
# tp_j ~ Beta(2,1), fp_j ~ Beta(1,2) per annotator; z_i ~ Bernoulli(prevalence);
# each observed label r_ij ~ Bernoulli(z_i ? tp_j : fp_j). Returns tp0.
def dawid_skene(family):
    prev = CONST["dawid_skene"]["prevalence_prior"]
    tp = CONST["dawid_skene"]["tp_prior"]
    fp = CONST["dawid_skene"]["fp_prior"]
    for v in VALUES[family]:
        d = DATA[family][str(v)]
        J = d["n_annotators"]
        ann = d["annotations"]
        n_items = max(a[0] for a in ann) + 1
        z_lines = "\n    ".join(f"z{i} := flip(prevalence);" for i in range(n_items))
        tp_lines = "\n    ".join(f"tp{j} := beta({tp[0]:g}, {tp[1]:g});" for j in range(J))
        fp_lines = "\n    ".join(f"fp{j} := beta({fp[0]:g}, {fp[1]:g});" for j in range(J))
        obs_block = "\n    ".join(
            f"observe(flip((if z{i} {{ tp{j} }} else {{ fp{j} }})) == {str(label).lower()});"
            for (i, j, label) in ann)
        text = f"""def main() {{
    prevalence := beta({prev[0]:g}, {prev[1]:g});

    {tp_lines}
    {fp_lines}

    {z_lines}

    {obs_block}

    return tp0;
}}
"""
        yield fname(family, v), write(fname(family, v), text), len(ann)


# secret_ballot: p_m ~ Beta(1,1) per member; for each ballot only the strict
# yes-majority of the participating members' secret votes is observed. Returns p0.
def secret_ballot(family):
    bp = CONST["secret_ballot"]["beta_prior"]
    for v in VALUES[family]:
        ballots = DATA[family][str(v)]["ballots"]
        p_lines = "\n    ".join(f"p{m} := beta({bp[0]:g}, {bp[1]:g});" for m in range(v))
        obs_lines = []
        for (idxs, outcome) in ballots:
            terms = " + ".join(f"(if flip(p{i}) {{1}} else {{0}})" for i in idxs)
            cond = f"(2 * ({terms})) > {len(idxs)}"
            obs_lines.append(f"observe(({cond}) == {str(outcome).lower()});")
        obs_block = "\n    ".join(obs_lines)
        text = f"""def main() {{
    {p_lines}

    {obs_block}

    return p0;
}}
"""
        yield fname(family, v), write(fname(family, v), text), len(ballots)


# sir_model: SIR epidemic HMM. Each row of the data is an independent 12-step
# chain starting not-infected; all rows share 4 Beta rates. The symptom is emitted
# from the current state, then the state transitions. Returns E[inf].
def sir_model(family):
    pri = CONST["sir_model"]["beta_priors"]
    steps = CONST["sir_model"]["steps_per_row"]
    rates = "\n    ".join(f"{r} := beta({pri[r][0]:g}, {pri[r][1]:g});"
                          for r in ("rec", "inf", "bas", "sym"))
    for v in VALUES[family]:
        rows = DATA[family][str(v)]["histories"]
        flat = [int(b) for row in rows for b in row]
        H = "[" + ", ".join(str(b) for b in flat) + "]"
        def body(idx, ind):
            return (f"{ind}pobs := if state == 1 {{ sym }} else {{ bas }};\n"
                    f"{ind}observe(flip(pobs) == (H[{idx}] == 1));\n"
                    f"{ind}state = if state == 1 {{ (if flip(rec) {{ 0 }} else {{ 1 }}) }}\n"
                    f"{ind}        else          {{ (if flip(inf) {{ 1 }} else {{ 0 }}) }};")
        if len(rows) == 1:
            chains = f"""    state := 0;
    for n in [0..{steps}) {{
{body("n", " " * 8)}
    }}"""
        else:
            chains = f"""    for r in [0..{len(rows)}) {{
        state := 0;
        for n in [0..{steps}) {{
{body(f"{steps}*r + n", " " * 12)}
        }}
    }}"""
        text = f"""def main() {{
    H := {H};

    {rates}

{chains}

    return Expectation(inf);
}}
"""
        yield fname(family, v), write(fname(family, v), text), len(flat)


# changepoint_coal: coal-mining disasters with an unknown number k <= 3 of
# changepoints. Segment rates h_s ~ Gamma(2,1); k ~ truncated Poisson(1); each
# year boundary is a changepoint with probability (changepoints left)/(boundaries
# left). Returns the posterior over k.
def changepoint_coal(family):
    c = CONST["changepoint_coal"]
    kp = [c["k_prior"][str(k)] for k in range(4)]
    tot = sum(kp)
    k_cat = "categorical([" + ", ".join(f"{w:g}/{tot:g}" for w in kp) + "])"
    for v in VALUES[family]:
        counts = DATA[family][str(v)]["counts"]
        text = f"""def main(){{
    counts := [{",".join(str(x) for x in counts)}];
    N := {len(counts)};
    h0 := gamma({c["alpha"]:g},{c["beta"]:g});
    h1 := gamma({c["alpha"]:g},{c["beta"]:g});
    h2 := gamma({c["alpha"]:g},{c["beta"]:g});
    h3 := gamma({c["alpha"]:g},{c["beta"]:g});
    k := {k_cat};
    kleft := k;
    seg := 0;
    observe(poisson(h0) == counts[0]);
    for i in [1..N){{
        rem := N - i;
        cut := bernoulli(kleft / rem);
        kleft = kleft - cut;
        seg = seg + cut;
        rate := if seg == 0 {{ h0 }} else {{ if seg == 1 {{ h1 }} else {{ if seg == 2 {{ h2 }} else {{ h3 }} }} }};
        observe(poisson(rate) == counts[i]);
    }}
    return k;
}}
"""
        yield fname(family, v), write(fname(family, v), text), len(counts)


# soccer (Sports Analysis): each team has an identity (defend / balanced / attack)
# drawn from theta ~ Dirichlet(1,1,1) (via normalised Gamma(1,1) draws) and a
# goal rate r_t ~ Gamma(4,4). A game's goals are Poisson(rate * scale), where the
# scale depends on the two identities (home side boosted). Returns id0.
def soccer(family):
    for v in VALUES[family]:
        d = DATA[family][str(v)]
        T = len(d["names"])
        id_lines = "\n    ".join(f"id{t}:=categorical([th0,th1,th2]);" for t in range(T))
        r_line = " ".join(f"r{t}:=gamma(4,4);" for t in range(T))
        game_lines = []
        for g, (i, j, gi, gj) in enumerate(d["games"]):
            p = "p" if g == 0 else f"p{g + 1}"
            decl = f"{p}:=id{i}+id{j};"
            home = f"observe(poisson(r{i}*scaleOf({p}+1))=={gi});"
            away = f"observe(poisson(r{j}*scaleOf({p}))=={gj});"
            game_lines.append(f"{decl:<12} {home:<38} {away}")
        games = "\n    ".join(game_lines)
        text = f"""def scaleOf(k:ℕ){{
    return if k==0 {{4.0/9.0}} else {{ if k==1 {{2.0/3.0}} else {{ if k==2 {{1.0}}
           else {{ if k==3 {{1.5}} else {{ if k==4 {{2.25}} else {{3.375}} }} }} }} }};
}}

def main(){{
    a0:=gamma(1,1); a1:=gamma(1,1); a2:=gamma(1,1); sa:=a0+a1+a2;
    th0:=a0/sa; th1:=a1/sa; th2:=a2/sa;
    {id_lines}
    {r_line}

    {games}

    return id0;
}}
"""
        yield fname(family, v), write(fname(family, v), text), len(d["games"])


GENERATORS = [
    ("hmm_first_order", hmm_first_order),
    ("hmm_second_order", hmm_second_order),
    ("hmm_factorial", hmm_factorial),
    ("outliers", outliers),
    ("soft_kmeans", soft_kmeans),
    ("king_ei", king_ei),
    ("dawid_skene", dawid_skene),
    ("secret_ballot", secret_ballot),
    ("sir_model", sir_model),
    ("changepoint_coal", changepoint_coal),
    ("soccer", soccer),
]


def main():
    global write
    args = parse_args(__doc__, os.path.join(TRANSLATIONS, "psi"))
    write = Writer(args.out, args.overwrite, "//")
    for fam, gen in GENERATORS:
        for _ in gen(fam):
            pass
    write.report()


if __name__ == "__main__":
    main()
