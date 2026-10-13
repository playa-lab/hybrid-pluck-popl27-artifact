#!/usr/bin/env python3
# fix_stan.py — modernize SlicStan's generated Stan, fixing several codegen gaps
# that make CmdStan (>= 2.31) reject or fail to initialize SlicStan's output.
#
# 1) Declaration syntax (SlicStan is 2018-era):
#      int[5] y;            -> array[5] int y;
#      real[N][K] x;        -> array[N,K] real x;
#      vector[3][3] trans;  -> array[3] vector[3] trans;   (array of vectors)
#      matrix[3,3][P] R;    -> array[P] matrix[3,3] R;     (array of matrices)
#    For int/real every [..] is an array dim. For vector/matrix the FIRST [..] is
#    the element size; the rest become a leading `array[..]`.
# 2) Marginalization-message accumulators: SlicStan declares them `real[K] m;` but
#    assigns `m = rep_vector(0,K);` and indexes `m[i]` — i.e. uses them as vectors.
#    A single-dim `real[K] NAME` that is assigned a rep_vector is emitted as
#    `vector[K] NAME` (a real[] cannot be assigned a rep_vector).
# 3) `poisson_lpdf` -> `poisson_lpmf` (Poisson is integer-valued; SlicStan emits the
#    _lpdf form inside its enumeration accumulators).
# 4) Constraints: SlicStan emits all parameters UNCONSTRAINED, so probability/rate
#    parameters fail HMC initialization (their prior is -inf off-support). We infer
#    the support from each parameter's prior — `~ beta`/`~ uniform` => <lower=0,
#    upper=1>; `~ gamma`/`inv_gamma`/`exponential`/`lognormal`/`chi_square` =>
#    <lower=0> — and add it to that parameter's `real NAME;` declaration. The
#    constraint is implied by the model, so this is correctness-preserving.
#
# Usage: fix_stan.py < model.stan > model.fixed.stan
import re, sys

DECL = re.compile(
    r'^(\s*)(int|real|vector|matrix)((?:\[[^\]]*\])+)(\s+)([A-Za-z][A-Za-z0-9_.]*)')
REPV = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_.]*)\s*=\s*rep_vector\b', re.M)
SAMPLE = re.compile(r'^\s*([A-Za-z][A-Za-z0-9_.]*)\s*~\s*([a-z_]+)\s*\(', re.M)
SCALAR = re.compile(r'^(\s*)real(\s+)([A-Za-z][A-Za-z0-9_.]*)\s*;\s*$')

UNIT = {'beta', 'uniform'}
POS = {'gamma', 'inv_gamma', 'exponential', 'lognormal', 'chi_square'}

text = sys.stdin.read()
text = text.replace('poisson_log_lpdf', 'poisson_log_lpmf').replace(
    'poisson_lpdf', 'poisson_lpmf')

vec_names = set(m.group(1) for m in REPV.finditer(text))

# infer a support constraint per parameter from its prior
constraint = {}
for m in SAMPLE.finditer(text):
    name, dist = m.group(1), m.group(2)
    if dist in UNIT:
        constraint[name] = '<lower=0, upper=1>'
    elif dist in POS:
        constraint.setdefault(name, '<lower=0>')

def repl(m):
    indent, base, dims_s, sp, name = m.groups()
    dims = [d[1:-1] for d in re.findall(r'\[[^\]]*\]', dims_s)]
    if base in ('int', 'real'):
        if base == 'real' and len(dims) == 1 and name in vec_names:
            return f'{indent}vector[{dims[0]}]{sp}{name}'
        return f'{indent}array[{",".join(dims)}] {base}{sp}{name}'
    size, arr = dims[0], dims[1:]
    elem = f'{base}[{size}]'
    if arr:
        return f'{indent}array[{",".join(arr)}] {elem}{sp}{name}'
    return f'{indent}{elem}{sp}{name}'

out = []
for line in text.split('\n'):
    sm = SCALAR.match(line)
    if sm and sm.group(3) in constraint:
        ind, sp, nm = sm.groups()
        out.append(f'{ind}real{constraint[nm]}{sp}{nm};')
        continue
    out.append(DECL.sub(repl, line))
sys.stdout.write('\n'.join(out))
