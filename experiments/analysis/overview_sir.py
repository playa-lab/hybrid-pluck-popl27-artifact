# Simplified overview example: SIR-temp WITHOUT the fever layer, T=2.
#   rho = beta(2,4) recovery rate; iota = beta(1,6) infection rate
#   emit s y = { mu = if s then 101.5 else 98.6; observe(close?(normal(mu,0.7), y)) }
#   s1 = flip(iota); emit s1 99.8
#   s2 = if s1 then !flip(rho) else flip(iota); emit s2 100.4
#   return (s2, [rho, iota])
# Prints: BDD, exact Z/posteriors by enumeration, the WMC circuit, sum weights.
import math
from math import exp, sqrt, pi, lgamma

PRIOR = {'rho': (2, 4), 'iota': (1, 6)}
sig = 0.7; muF, muN = 101.5, 98.6; y1, y2 = 99.8, 100.4
def npdf(y, m): return exp(-(y - m) ** 2 / (2 * sig * sig)) / (sig * sqrt(2 * pi))
D = {'z1F': npdf(y1, muF), 'z1N': npdf(y1, muN), 'z2F': npdf(y2, muF), 'z2N': npdf(y2, muN)}
STAT = {k: (y - m) / sig for k, (y, m) in
        {'z1F': (y1, muF), 'z1N': (y1, muN), 'z2F': (y2, muF), 'z2N': (y2, muN)}.items()}

# ---------------- BDD ----------------
class BDD:
    def __init__(s): s.T = ('T',); s.F = ('F',); s.mk = {}; s.itec = {}
    def node(s, v, lo, hi):
        if lo is hi: return lo
        k = (v, id(lo), id(hi))
        if k in s.mk: return s.mk[k]
        n = (v, lo, hi); s.mk[k] = n; return n
    def var(s, v): return s.node(v, s.F, s.T)
    def ite(s, f, g, h, o):
        if f is s.T: return g
        if f is s.F: return h
        if g is s.T and h is s.F: return f
        k = (id(f), id(g), id(h))
        if k in s.itec: return s.itec[k]
        vs = [x[0] for x in (f, g, h) if x[0] not in ('T', 'F')]
        v = min(vs, key=lambda x: o.index(x))
        co = lambda n, val: n if (n[0] in ('T', 'F') or n[0] != v) else (n[2] if val else n[1])
        hi = s.ite(co(f, 1), co(g, 1), co(h, 1), o); lo = s.ite(co(f, 0), co(g, 0), co(h, 0), o)
        r = s.node(v, lo, hi); s.itec[k] = r; return r

order = ['f0', 'o1F', 'o1N', 'rec', 'inf', 'o2F', 'o2N']
B = BDD(); V = {v: B.var(v) for v in order}
AND = lambda a, b: B.ite(a, b, B.F, order)
OR = lambda a, b: B.ite(a, B.T, b, order)
NOT = lambda a: B.ite(a, B.F, B.T, order)
f0, o1F, o1N, rec, inf, o2F, o2N = [V[v] for v in order]
obs1 = OR(AND(f0, o1F), AND(NOT(f0), o1N))
s2 = OR(AND(f0, NOT(rec)), AND(NOT(f0), inf))
obs2 = OR(AND(s2, o2F), AND(NOT(s2), o2N))
psi = AND(obs1, obs2)
seen = set(); nodes = []
def walk(n):
    if n[0] in ('T', 'F') or id(n) in seen: return
    seen.add(id(n)); nodes.append(n); walk(n[1]); walk(n[2])
walk(psi)
lab = {}; cnt = {}
for n in nodes: cnt[n[0]] = cnt.get(n[0], 0) + 1; lab[id(n)] = "%s%d" % (n[0], cnt[n[0]])
nm = lambda n: 'T' if n[0] == 'T' else ('_' if n[0] == 'F' else lab[id(n)])
print("BDD decision nodes = %d (order %s)" % (len(nodes), order))
for n in nodes: print("  %-6s lo->%-6s hi->%-6s" % (nm(n), nm(n[1]), nm(n[2])))

# ---------------- exact posterior by enumeration ----------------
def logB(x, y): return lgamma(x) + lgamma(y) - lgamma(x + y)
def bn(rate, H, T):
    a, b = PRIOR[rate]; return exp(logB(a + H, b + T) - logB(a, b))
worlds = []
for s1v in (0, 1):
    for s2v in (0, 1):
        c = {'rho': [0, 0], 'iota': [0, 0]}
        c['iota'][0 if s1v else 1] += 1              # s1 = flip(iota)
        if s1v: c['rho'][0 if (1 - s2v) else 1] += 1  # s2 = !flip(rho)
        else:   c['iota'][0 if s2v else 1] += 1       # s2 = flip(iota)
        z1 = D['z1F'] if s1v else D['z1N']
        z2 = D['z2F'] if s2v else D['z2N']
        coeff = z1 * z2
        for r in c: coeff *= bn(r, c[r][0], c[r][1])
        worlds.append((s1v, s2v, coeff, {r: tuple(c[r]) for r in c}))
Z = sum(w[2] for w in worlds)
print("\nZ = %.4e * eps^2   (%d worlds)" % (Z, len(worlds)))
print("P(s2 = infected) = %.4f ; P(s1 = infected) = %.4f"
      % (sum(w[2] for w in worlds if w[1]) / Z, sum(w[2] for w in worlds if w[0]) / Z))
print("world weights (s1,s2):")
for w in sorted(worlds, key=lambda w: -w[2]):
    print("  s1=%d s2=%d  w=%.4f  counts=%s" % (w[0], w[1], w[2] / Z, w[3]))
from collections import defaultdict
for r in PRIOR:
    mix = defaultdict(float)
    for w in worlds: mix[w[3][r]] += w[2]
    a, b = PRIOR[r]
    mean = sum(c / Z * (a + HT[0]) / (a + b + HT[0] + HT[1]) for HT, c in mix.items())
    print("%-5s prior mean %.3f -> post mean %.3f ; comps %s"
          % (r, a / (a + b), mean,
             {('B(%g,%g)' % (a + HT[0], b + HT[1])): round(c / Z, 3) for HT, c in mix.items()}))

# ---------------- WMC circuit (same machinery as sir4_spn) ----------------
ZERO = ('zero',); CONS = {}
def hc(n):
    k = ('leaf', n[1]) if n[0] == 'leaf' else (n[0],) + tuple(id(x) for x in n[1])
    if k in CONS: return CONS[k]
    CONS[k] = n; return n
def leaf(L): return hc(('leaf', L))
ONE = leaf(('c', 1.0))
def supp(n):
    if n is ZERO: return frozenset()
    if n[0] == 'leaf':
        L = n[1]
        if L[0] == 'c': return frozenset()
        return frozenset({L[1]})
    return frozenset().union(*(supp(k) for k in n[1])) if n[1] else frozenset()
chp = lambda n: list(n[1]) if n[0] == 'prod' else [n]
chs = lambda n: list(n[1]) if n[0] == 'sum' else [n]
def mkprod(kids):
    fl = []
    for k in kids: fl += chp(k)
    fl = [k for k in fl if not (k[0] == 'leaf' and k[1] == ('c', 1.0))]
    return ONE if not fl else (fl[0] if len(fl) == 1 else hc(('prod', tuple(fl))))
def mksum(kids):
    fl = []
    for k in kids:
        if k is ZERO: continue
        fl += chs(k)
    return ZERO if not fl else (fl[0] if len(fl) == 1 else hc(('sum', tuple(fl))))
def odot(L1, L2):
    if L1[0] == 'beta' and L2[0] == 'beta' and L1[1] == L2[1]:
        return ('beta', L1[1], L1[2] + L2[2], L1[3] + L2[3])
    raise Exception('bad odot %s %s' % (L1, L2))
MEMO = {}
def boxtimes(n1, n2):
    if n1 is ZERO or n2 is ZERO: return ZERO
    k = ('x', id(n1), id(n2))
    if k in MEMO: return MEMO[k]
    r = _bt(n1, n2); MEMO[k] = r; return r
def _bt(n1, n2):
    if not (supp(n1) & supp(n2)): return mkprod([n1, n2])
    if n1[0] == 'leaf' and n2[0] == 'leaf': return leaf(odot(n1[1], n2[1]))
    if n1[0] == 'sum': return mksum([boxtimes(m, n2) for m in n1[1]])
    if n2[0] == 'sum': return mksum([boxtimes(n1, m) for m in n2[1]])
    if n1[0] == 'prod': return _pt(n1, n2)
    return _pt(n2, n1)
def _pt(prod, other):
    kids = list(prod[1]); So = supp(other)
    O = [i for i, m in enumerate(kids) if supp(m) & So]
    acc = other
    for i in O: acc = boxtimes(kids[i], acc)
    return mkprod([kids[i] for i in range(len(kids)) if i not in O] + [acc])
def wt(v):
    if v == 'f0':  return leaf(('beta', 'iota', 1, 0))
    if v == 'inf': return leaf(('beta', 'iota', 1, 0))
    if v == 'rec': return leaf(('beta', 'rho', 1, 0))
    kk = {'o1F': 'z1F', 'o1N': 'z1N', 'o2F': 'z2F', 'o2N': 'z2N'}[v]
    return leaf(('z', 'z1' if v.startswith('o1') else 'z2', kk))
def wf(v):
    m = {'f0': ('iota', 0, 1), 'inf': ('iota', 0, 1), 'rec': ('rho', 0, 1)}
    if v in m: return leaf(('beta',) + m[v])
    return ONE
WM = {}
def wmc(n):
    if n[0] == 'T': return ONE
    if n[0] == 'F': return ZERO
    if id(n) in WM: return WM[id(n)]
    r = mksum([boxtimes(wf(n[0]), wmc(n[1])), boxtimes(wt(n[0]), wmc(n[2]))])
    WM[id(n)] = r; return r
root = wmc(psi)

NN = {}; ncnt = {}
def L2s(L):
    if L[0] == 'c': return 'c%.3g' % L[1]
    if L[0] == 'beta': return '%s[%d,%d]' % (L[1], L[2], L[3])
    return L[2]
def lb(n):
    if n is ZERO: return '0'
    if id(n) in NN: return NN[id(n)]
    if n[0] == 'leaf': s = L2s(n[1])
    else:
        t = '+' if n[0] == 'sum' else 'x'; ncnt[t] = ncnt.get(t, 0) + 1; s = '%s%d' % (t, ncnt[t])
    NN[id(n)] = s; return s
vseen = set(); topo = []
def vis(n):
    if n is ZERO or id(n) in vseen: return
    vseen.add(id(n))
    if n[0] in ('sum', 'prod'):
        for k in n[1]: vis(k)
    topo.append(n)
vis(root)
ni = sum(1 for n in topo if n[0] in ('sum', 'prod'))
print("\nCIRCUIT: %d internal, %d leaves" % (ni, sum(1 for n in topo if n[0] == 'leaf')))
for n in topo:
    if n[0] in ('sum', 'prod'):
        print("  %-4s %-4s -> [%s]" % (lb(n), n[0], ', '.join(lb(k) for k in n[1])))
print("  root =", lb(root))
from collections import Counter
usec = Counter()
for n in topo:
    if n[0] in ('sum', 'prod'):
        for k in n[1]: usec[lb(k)] += 1
print("  shared(used>1):", {k: c for k, c in usec.items() if c > 1})

def summ(n):
    if n is ZERO: return None
    if n[0] == 'leaf':
        L = n[1]
        if L[0] == 'c': return (0, L[1]) if L[1] > 0 else None
        if L[0] == 'beta': return (0, bn(L[1], L[2], L[3]))
        return (1, D[L[2]])
    subs = [summ(k) for k in n[1]]
    if n[0] == 'prod':
        if any(s is None for s in subs): return None
        k = sum(s[0] for s in subs); c = 1.0
        for s in subs: c *= s[1]
        return (k, c)
    subs = [s for s in subs if s]
    km = min(s[0] for s in subs)
    return (km, sum(s[1] for s in subs if s[0] == km))
print("\nsum-node mixture weights:")
for n in topo:
    if n[0] == 'sum':
        ss = [(lb(k), summ(k)) for k in n[1]]
        km = min(s[0] for _, s in ss if s); tot = sum(s[1] for _, s in ss if s and s[0] == km)
        print("  %-4s %s" % (lb(n), '  '.join(
            "%s:w=%.3f" % (l, (s[1] / tot if s[0] == km else 0)) for l, s in ss if s)))
rk, rc = summ(root)
print("\nZ (circuit) = %.4e * eps^%d" % (rc, rk))
print("delta pins:", {k: round(v, 3) for k, v in STAT.items()})
