# The Hybrid Pluck Language

Hybrid Pluck is a lazy, functional probabilistic programming language with Lisp-style
syntax. A program is a list of definitions and queries. Each query is answered by exact
inference: discrete choices are compiled into a binary decision diagram, and continuous
variables are integrated in closed form using conjugacy.

This guide covers what you need to write and run your own models.

## 1. Running a program

Build with `cargo build --release`, then run from the repository root:

```sh
target/release/pluck FILE.pluck            # or: cargo run --release -- FILE.pluck
target/release/pluck FILE.pluck --json     # machine-readable output
```

Run from the root because `include` paths are relative to the working directory. In the
Docker image the root is `/artifact` (see [README.md](../README.md)). The standard
library, `stdlib.pluck`, is built into the binary and loaded before every program.

```scheme
; Two coins. What is the chance the first is heads, given at least one is?
(define (model)
  (let ((a (flip 0.5))
        (b (flip 0.5)))
    (Posterior a (or a b))))

(query first-given-either (model))
(query both (Marginal (and (flip 0.5) (flip 0.5))))
```

```
first-given-either:
  True  0.6666666666666666
  False  0.33333333333333337

both:
  False  0.7499999999999999
  True  0.24999999999999994
```

### Reading the output

Each query prints its name, then one row per distinct value, most probable first, with its
probability. Probabilities are printed in full; the last digits, and the order of rows with
equal probability, can change from run to run.

When the value contains continuous random variables, the row shows them by name and the
lines below give their posterior distribution:

| Printed form | Meaning |
|---|---|
| `x ~ N(1.0000, 0.5774)` | Gaussian with mean 1 and **standard deviation** 0.5774 |
| `x, y ~ Joint Gaussian:` with `mean` and `cov` | correlated Gaussians: mean vector and covariance matrix |
| `p ~ Beta(3.0000, 2.0000)` | Beta(a, b) |
| `p ~ Dirichlet(2.0000, 2.0000, 4.0000)` | Dirichlet(alpha_1, ..., alpha_n) |
| `λ ~ Gamma(2.0000, 2.5000)` | Gamma(shape, rate) |
| `λ ~ Mix[+1.1517·Gamma(...) -0.1517·Gamma(...)]` | a mixture of Gammas; weights can be negative |
| `p = 0.3000` | a point mass |

A variable stored in a constructor field whose declared name has at most five characters is
shown under that name (`bias`, or `rates₀`, `rates₁` for a list). Otherwise names are
generated: `x, y, z, ...` for Gaussians, `p, q, r, ...` for Beta and Dirichlet variables,
`λ, μ, ν, ...` for Gamma rates.

If one value has several posterior components, they are listed as a tree, each with its share
of the row's probability in percent (see Section 8). After the ten most probable rows, rows
below probability 1e-6 are folded into `... and N more components (combined prob: P)`. A
query that prints no rows either had evidence of probability zero or reported an error on
stderr.

`--json` must come after the file name. It prints a JSON array with one object per query,
`{"query": NAME, "kind": {...}}`. `kind.result_kind` is `"distribution"`, with `items` that
each have `display`, `log_probability` (natural log) and `posteriors` (lists `beta`,
`dirichlet`, `gamma`, `gaussian`), or `"samples"`, with a list of `values`. JSON items are
not grouped or truncated, and Gaussian entries give the covariance, not the standard
deviation.

### Environment variables

| Variable | Effect |
|---|---|
| `PLUCK_STATS` | If set, each exact query prints statistics to stderr: number of worlds, BDD and sum-product-network sizes, number of posterior components, time per stage, and a final `inference_seconds:` line. |
| `PLUCK_DISPLAY_PRECISION` | Decimals for distribution parameters (default 4) and for percentage weights (default 2). Probabilities are always printed in full. |
| `PLUCK_EQ_TOL` | Relative tolerance (default `1e-6`) the sampling queries use to check that sampled Gaussian values satisfy an equality observation. It does not affect `Marginal` or `Posterior`. |

Most errors stop the program with a Rust panic (`thread 'main' ... panicked at ...`) and a
message such as `unknown token: 'g'`. Section 6 shows one. A few problems only print a
message to stderr, and the query then prints no rows.

## 2. Syntax

Comments start with `;` and run to the end of the line. Commas are optional separators in
lists, vectors and parameter lists.

| Literal | Value |
|---|---|
| `3` | natural number, built from `O` and `S`. Used for counts, list indices, `range`, `take`. |
| `@3` | native integer. Used for `categorical` outcomes, `vector_index`, Poisson counts, bit widths. |
| `2.5`, `.5`, `-1` | float. Negative numbers are always floats. There is no exponent notation (`1e-3`). |
| `true`, `false` | the constructors `True` and `False`, also written `True` or `(True)` |
| `nothing` | `Unit` |
| `"hi"`, `0ca` | a string (a list of 8-bit integers); the character `a` |
| `'red` | a symbol; compare with `native_eq` |
| `[a, b, c]` | list built from `Cons` and `Nil` |
| `{1.0, 2.0}`, `{{1.0, 0.0}, {0.0, 1.0}}` | vector and matrix of floats or Gaussians |

| Top-level form | Meaning |
|---|---|
| `(define name expr)` | a named value, re-evaluated at every use (see below) |
| `(define (f a b) body)` | a function. It may call itself. Functions are curried. |
| `(define (f) body)` | a function of no arguments, called as `(f)` |
| `(define-type t (C1 field ...) (C2) ...)` | a data type with constructors `C1`, `C2`. Field names are labels, not types. |
| `(include "path")` | read the forms of another file; the path is relative to the working directory |
| `(query name expr)`, `(query expr)` | answer a query (Section 5) |

| Expression | Meaning |
|---|---|
| `(f a b)` | application. `(f)` applies `f` to `nothing`. |
| `(fn x y -> body)` | function (`lambda`, `lam` and `λ` also work). `(fn -> body)` takes no argument. |
| `(let ((x e1) (y e2)) body)` | bindings, each visible in the next. `(let (x e1 y e2) body)` is the same. |
| `(if c a b)` | conditional on a boolean |
| `(match e C1 x y => e1 C2 => e2)` | case analysis on the outer constructor. `case`, `of` and `\|` are accepted. |
| `(C a b)` | constructor application. A constructor with no fields can be written `C` or `(C)`. |
| `(Y (fn self x -> ...))` | anonymous recursion; rarely needed, since `define` can recurse |

Built-in types are `bool` (`True`, `False`), `nat` (`O`, `S n`), `list` (`Nil`, `Cons head
tail`) and `unit` (`Unit`). The standard library adds `Pair a b`.

A name must be defined before it is used, so two top-level functions cannot call each other.
Redefining a name replaces it everywhere, including in earlier functions and the standard
library. A primitive such as `beta` can be bound with `let` and used as a variable, but
`(beta ...)` still calls the primitive unless a top-level `define` shadows it.

Patterns are one constructor deep, and there is no catch-all pattern: `_ => e` is a case for
a constructor named `_`. If no case matches, that execution path is discarded, exactly as if
it had failed an observation. `(error "msg")` does the same.

```scheme
; Deterministic code: every query below has a single outcome.
(define-type shape (Circle r) (Rect w h))

(define (area s)
  (match s
    Circle r => (*. 3.14 (*. r r))
    Rect w h => (*. w h)))

(define (add3 a b c) (+ a (+ b c)))
(define add-ten (add3 4 6))              ; partial application
(define (ones) (Cons 1 (ones)))          ; an infinite list (evaluation is lazy)

(query areas (Marginal (map area [(Circle 1.0), (Rect 2.0 3.5)])))
(query lazy (Marginal (Pair (add-ten 1) (take 3 (ones)))))
(query let-if
  (Marginal
    (let ((xs [1, 2, 3])
          (total (fold (fn acc x -> (+ acc x)) 0 xs)))
      (if (> total 5) "big" "small"))))
```

```
areas:
  [3.14, 7]  1

lazy:
  (11, [1, 1, 1])  1

let-if:
  "big"  1
```

**Laziness and sharing.** Arguments and constructor fields are evaluated only when needed,
so infinite structures are fine as long as the query looks at a finite part. A `let`-bound
variable or a function argument is evaluated once and shared. A top-level
`(define name expr)` is not: each use makes fresh random choices.

```scheme
; A top-level value is re-evaluated at each use; a let-bound one is shared.
(define c (flip 0.5))

(query top-level (Marginal (Pair c c)))
(query let-bound (Marginal (let ((x (flip 0.5))) (Pair x x))))
```

```
top-level:
  (False, False)  0.24999999999999994
  (False, True)  0.24999999999999994
  (True, False)  0.24999999999999994
  (True, True)  0.24999999999999994

let-bound:
  (False, False)  0.5
  (True, True)  0.5
```

**Debugging with `print`.** `(print e)` returns `e`, and writes to stderr the values `e` can
take on the current path, separated by `|`. Inference is symbolic, so this is every possible
value, not one sample. Values are shown only as far as they have been evaluated: lazy parts
print as `<thunk>`. A `print` runs each time its expression is evaluated. All queries run
before any result is printed, so its lines come before the results.

```scheme
; print writes the values its argument can take to stderr, and returns it.
(define (count-heads n)
  (if (== n 0)
      0
      (+ (if (print (flip 0.5)) 1 0) (count-heads (dec n)))))

(query heads (Marginal (count-heads 2)))
(query list (Marginal (length (print [1, 2, 3]))))
```

```
print: True | False
print: True | False
print: (Cons <thunk> <thunk>)
heads:
  1  0.5
  0  0.25
  2  0.25

list:
  3  1
```

## 3. Random choices

| Discrete | Distribution |
|---|---|
| `(flip p)` | `True` with probability `p`: a float, a Beta draw, or an element of a Dirichlet draw |
| `(discrete (e1 w1) (e2 w2) ...)` | `e_i` with probability `w_i / sum(w)`; weights must be number literals |
| `(uniform e1 e2 ...)` | one of the expressions, uniformly |
| `(categorical p0 p1 ...)` | index `@0`, `@1`, ... with the given probabilities, which must sum to 1 |
| `(categorical d)` | an index drawn from `d`, a Dirichlet draw |
| `(geom p)` | natural number `k` with probability `p (1-p)^k` (standard library) |
| `(uniform_int_range @bits @lo @hi)` | uniform integer in `[lo, hi]`, as a `bits`-wide bit vector |
| `(uniform_int @bits)` | uniform over all `2^bits` bit vectors |
| `(poisson rate)` | Poisson count; `rate` is a positive float or a Gamma draw |

Bit vectors support `int_eq`, `int_add`, `int_sub`, `int_lt` and `(mk_int @bits @value)`.

| Continuous | Distribution |
|---|---|
| `(beta a b)` | Beta(a, b). Write `1.0`, not `1`. |
| `(normal mu sd)` | Gaussian with mean `mu` and standard deviation `sd` (standard library). `mu` may be a Gaussian expression. |
| `(gaussian mu sd)` | the same, but `mu` must be a number |
| `(gaussian {m1, m2} {{c11, c12}, {c21, c22}})` | multivariate Gaussian with constant mean vector and covariance matrix; returns a vector |
| `(dirichlet a1 ... an)` | Dirichlet: a random probability vector of length `n` |
| `(gamma shape rate)` | Gamma(shape, rate), used as the rate of `poisson` or `exponential` |
| `(exponential rate)` | Exponential draw; `rate` is a positive float or a Gamma draw |

The parameters of `beta`, `dirichlet` and `gamma`, and the standard deviation of `normal` and
`gaussian`, must evaluate to numbers. They may depend on discrete choices, such as
`(if b 1.0 2.0)`, but not on continuous draws.

A continuous draw can be returned by a query, and otherwise used only in these ways:

| Draw | Allowed uses |
|---|---|
| Beta `p` | `(flip p)`, `(prob_close p v)` |
| Dirichlet `d` | `(categorical d)`; `(vector_index d @i)`, a probability for `flip`; `(dirichlet_eq d {...})` |
| Gaussian `x` | `+.` and `-.` with numbers or Gaussians; `*.` and `/.` by numbers; `sum`, `dot`, `@` and `vector_index` on vectors; `(real=? e v)` with Gaussian `e` and number `v` |
| Gamma `g` | `*.` or `/.` by a positive number; rate of `poisson` or `exponential`; `(close g v)` |
| Poisson draw `k` | `(native_eq k @n)`, `(native_leq k @n)`, `(native_geq k @n)`; `n` may also be a natural |
| Exponential draw `t` | `(close t v)`, `(real_lt t v)`, `(real_geq t v)` |

`+. -. *. /.` also work elementwise on vectors and matrices, with a number on either side.
`(@ A x)` is the matrix product, `(sum m)` adds all entries, and `(vector_index m @i)` or
`(vector_index m @i @j)` reads an entry.

```scheme
; Discrete random choices.
(query discrete (Marginal (discrete ("low" 1) ("mid" 2) ("high" 1))))
(query categorical (Marginal (categorical 0.2 0.5 0.3)))
(query two-dice
  (let ((a (uniform_int_range @4 @1 @6))
        (b (uniform_int_range @4 @1 @6)))
    (Posterior a (int_eq (int_add a b) (mk_int @4 @10)))))
```

```
discrete:
  "mid"  0.49999999999999994
  "high"  0.25000000000000006
  "low"  0.25

categorical:
  1  0.49999999999999994
  2  0.30000000000000004
  0  0.19999999999999998

two-dice:
  @6  0.33333333333333354
  @4  0.33333333333333337
  @5  0.33333333333333337
```

```scheme
; Continuous priors feeding discrete choices.
(query cat-dirichlet (Marginal (categorical (dirichlet 1.0 2.0 3.0))))
(query poisson     (Marginal (native_leq (poisson (gamma 2.0 1.0)) @1)))
(query exponential (Marginal (real_lt (exponential 2.0) 0.5)))
```

```
cat-dirichlet:
  2  0.4999999999999995
  1  0.33333333333333365
  0  0.16666666666666682

poisson:
  True  0.5000000000000006
  False  0.4999999999999995

exponential:
  True  0.6321205588285577
  False  0.36787944117144233
```

## 4. Observations

There is no `observe` statement. You condition by giving a boolean to a query:
`(Posterior v c)` is the distribution of `v` given that `c` is `True`. Inside an expression,
`(given c v)` from the standard library does the same. Paths where a `match` fails or
`error` is called are also removed. Build `c` with `and`, `or`, `not`, `iff` and the equality
functions of Section 7. Continuous values are observed with these primitives:

| Form | Event | Posterior |
|---|---|---|
| `(real=? e v)`, same as `(close e v)` | `abs(e - v) <= eps/2`, with `e` an affine combination of Gaussians | Gaussian update (density) |
| `(close t v)` | an Exponential draw equals `v` | Gamma update of its rate |
| `(real_lt t v)`, `(real_geq t v)` | an Exponential draw is below, or at least, `v` | Gamma update or mixture |
| `(native_eq k @n)`, `native_leq`, `native_geq` | a Poisson count equals, is at most, or is at least `n` | Gamma update or mixture |
| `(prob_close p v)` | a Beta draw equals `v` | point mass |
| `(dirichlet_eq d {...})` | a Dirichlet draw equals a vector | point mass |
| `(close g v)` | a Gamma draw equals `v` | point mass |
| `(close a b)` with numbers `a`, `b` | exact equality | no continuous variable |

**Tolerance.** In the semantics, `close` tests whether two reals are within a tolerance
`eps` (`abs(a - b) <= eps/2`), and the reported answer is the limit as `eps` goes to 0. You
never choose `eps`. For a Gaussian observed at a value this gives the usual density-based
posterior. Events are also ranked by their order in `eps`: an event whose probability is
proportional to `eps` (a density) vanishes next to one of positive probability (a point
mass), as the second query shows.

```scheme
; close compares within a tolerance eps; results are the limit eps -> 0.
(query density-event (Marginal (real=? (normal 0.0 1.0) 0.5)))
(query point-mass-wins
  (let ((x (if (flip 0.5) (normal 0.0 1.0) 0.5)))
    (Posterior x (real=? x 0.5))))
(query beta-pinned
  (let ((p (beta 2.0 2.0)))
    (Posterior p (prob_close p 0.3))))
```

```
density-event:
  False  1

point-mass-wins:
  0.5  1

beta-pinned:
  p  1
        p = 0.3000
```

The conjugate observations, each with a closed-form posterior:

```scheme
; Conjugate observations. Each query returns the posterior of a latent.

; A Gaussian observed twice (density; the eps -> 0 limit).
(query gaussian-obs
  (let ((mu (normal 0.0 1.0)))
    (Posterior mu (and (real=? (normal mu 1.0) 1.0)
                       (real=? (normal mu 1.0) 2.0)))))

; An affine combination of Gaussians observed at a value.
(query difference
  (let ((x (normal 0.0 1.0))
        (y (normal 0.0 1.0)))
    (Posterior x (real=? (-. x y) 1.0))))

; Bernoulli outcomes of a Beta-distributed coin.
(query coin
  (let ((p (beta 1.0 1.0)))
    (Posterior p (and (flip p) (and (flip p) (not (flip p)))))))

; Two Poisson counts with a Gamma rate.
(query counts
  (let ((rate (gamma 2.0 1.0)))
    (Posterior rate (and (native_eq (poisson rate) @3)
                         (native_geq (poisson rate) @1)))))
```

```
gaussian-obs:
  x  1
        x ~ N(1.0000, 0.5774)

difference:
  x  1
        x ~ N(0.5000, 0.7071)

coin:
  p  1
        p ~ Beta(3.0000, 2.0000)

counts:
  λ  1
        λ ~ Mix[+1.1517·Gamma(5.0000, 2.0000) -0.1517·Gamma(5.0000, 3.0000)]
```

Each `(normal ...)`, `(flip p)` or `(poisson rate)` call is a fresh draw: `counts` observes two
independent counts with the same rate. To compare two Gaussians, observe their difference,
`(real=? (-. x y) 0.0)`. The Gaussian must be the first argument of `real=?`.

## 5. Queries

A query expression must evaluate to one of four constructors. Random choices made in `let`
bindings around them are fine, because `let` is lazy.

| Form | Prints |
|---|---|
| `(Marginal v)` | the exact distribution of `v` |
| `(Posterior v c)` | the exact distribution of `v` given that `c` is true |
| `(PosteriorSamples v c n)` | `n` independent exact samples of `v` given `c`, one per line |
| `(Gibbs v c n blocks init)` | `n` samples of `v` given `c` from a blocked Gibbs chain, one per line |

`n` is a natural number or a native integer. In samples, continuous draws appear as numbers.
The random generator is not seeded, so samples differ between runs.

`blocks` is a list of expressions, usually latent variables. Each Gibbs step holds the
current block at its last value, redraws every other random choice exactly from its
conditional distribution given `c`, and moves to the next block. `init` sets the first block:
`(WithConstant e)` uses the value of `e`; `(WithPriorSample e)` draws `e` from the prior and
samples the first block given it and `c`. `(block-gibbs [x, y, z])` makes blocks that each
hold all variables but one, which gives single-site Gibbs sampling.

```scheme
; Exact and sampling queries on one model.
(define-type result (Result heads bias))

(define (model)
  (let ((bias (beta 1.0 1.0))
        (a (flip bias))
        (b (flip bias)))
    (Pair (Result a bias) (or a b))))

(query posterior
  (let ((m (model))) (Posterior (fst m) (snd m))))

(query samples
  (let ((m (model))) (PosteriorSamples (fst m) (snd m) 4)))

(query gibbs
  (let ((x (flip 0.5)) (y (flip 0.5)))
    (Gibbs (Pair x y) (or x y) 4 [x, y] (WithConstant true))))
```

```
posterior:
  (Result True bias)  0.7499999999999997
        bias ~ Beta(2.0000, 1.0000)
  (Result False bias)  0.2500000000000004
        bias ~ Beta(2.0000, 2.0000)

samples:
  (Result False 0.5763771535244979)
  (Result False 0.2311532634845726)
  (Result False 0.4414043090644294)
  (Result False 0.3465590306676637)

gibbs:
  (True, True)
  (True, True)
  (True, True)
  (False, True)
```

Returning the evidence alongside the value in a `Pair`, as `model` does, is a common
pattern. Each query runs independently, from a fresh inference state.

## 6. What can be inferred exactly

Exact inference works when every continuous variable is either observed or used only by
operations conjugate to its prior, which are the uses listed in Section 3:

| Prior | Conjugate observations | Posterior |
|---|---|---|
| `beta` | `flip` outcomes | Beta, or a mixture of Betas |
| `dirichlet` | `categorical` outcomes; `flip` of an element | Dirichlet, or a mixture |
| `normal`, `gaussian` | `real=?` on affine combinations | Gaussian or joint Gaussian, or a mixture |
| `gamma` | `poisson` counts (`=`, `<=`, `>=`) and `exponential` times (`=`, `<`, `>=`), with rates scaled by constants | Gamma, or a mixture with possibly negative weights |
| any of these | `prob_close`, `dirichlet_eq`, `close` on a Gamma | point mass |

Discrete choices may decide which prior, observation or parameter is used. The posterior is
then an exact mixture over the discrete outcomes.

**Not supported.** These stop the program with an error:

- Inequalities on Gaussians. `real_lt` and `real_geq` accept only Exponential draws and
  numbers, so `x < 0`, `x < y` and `max` or `min` of Gaussians cannot be written. This is
  why the paper lists some benchmarks as "Not Expressible in Hybrid Pluck".
- Multiplying two Gaussians, dividing by a Gaussian, or using a Gaussian as a `flip`
  probability.
- A random standard deviation, or random Beta, Dirichlet or Gamma parameters.
- Arithmetic on Beta or Dirichlet draws, such as `(*. 0.5 p)`.
- Arithmetic on Poisson or Exponential draws, and adding Gamma draws.

There is no static type checker. A violation is reported when the expression is evaluated.
For example, `(Posterior x (real_geq x 0.0))` with `x` drawn by `normal` stops with
`real_geq: range observations are only supported on Exponential draws (or plain floats),
got GaussianExpr { ... }`.

**Keeping inference fast.** Cost grows with the size of the decision diagram and the number
of posterior components. These habits, taken from the examples, help:

- Observe each data point in the same recursion step that draws its latent variables,
  instead of building all the data first and comparing at the end. This keeps the decision
  diagram small (see the comments in `examples/outliers_n08.pluck`).
- Return only what you need. Each extra latent in the query value can multiply the number of
  rows and components (`examples/secret_ballot_n04.pluck`).
- Bound the recursion that the query forces. `(Marginal (geom 0.5))` has infinitely many
  outcomes and does not finish; `(Marginal (< (geom 0.5) 3))` does.
- Natural-number literals are unary. Prefer small naturals, or native integers (`@n`).
- When exact inference is exponential in the data size, `Gibbs` with good blocks can still be
  practical (`examples/shared/soft_kmeans.pluck` explains this trade-off).

## 7. Standard library

`stdlib.pluck` is loaded automatically. "Natural" means the unary `nat` type.

| Function | Description |
|---|---|
| `and a b`, `or a b`, `not a`, `iff a b` | boolean operations; `and` and `or` evaluate `b` only if needed |
| `real=? x y` | `(close x y)`, the continuous observation of Section 4 |
| `normal mu sd` | Gaussian with mean `mu` and standard deviation `sd` |
| `inc n`, `dec n` | successor; predecessor (`dec 0` is `0`) |
| `+`, `-`, `*` | arithmetic on naturals; `-` stops at 0 |
| `==`, `nat=?`, `<`, `>`, `<=`, `>=` | comparison of naturals |
| `iseven n`, `mod x y` | parity; remainder |
| `geom p` | geometric: `k` with probability `p (1-p)^k` |
| `randnat`, `randnatlist` | `(geom 0.5)`; a random-length list of random naturals |
| `car xs`, `cdr xs`, `cdr_safe xs` | head; tail; tail that maps `Nil` to `Nil` |
| `isempty xs`, `head-or xs x` | emptiness test; head, or `x` if empty |
| `length xs`, `range n` | length; the list `[0, ..., n-1]` |
| `index n xs`, `take n xs` | element `n` (from 0); first `n` elements |
| `append xs ys`, `append_one xs y` | concatenation; add one element at the end |
| `map f xs`, `mapi f xs` | apply `f` to each element; `mapi` calls `(f x i)` |
| `filter f xs`, `filteri f xs` | keep elements where `f` is true; `filteri` calls `(f x i)` |
| `any f xs`, `all f xs` | some or every element satisfies `f` |
| `fold f init xs` | right fold: `(f (fold f init rest) x)` when `xs` is `(Cons x rest)` |
| `zip_with f xs ys` | pairwise `f`, up to the shorter list |
| `list=? eq xs ys` | list equality with element equality `eq`, as in `(list=? real=? data ys)` |
| `char=?`, `string=?` | equality of characters and of strings |
| `=? a b` | structural equality of data values |
| `constructor=? a b`, `constructors_equal a b` | same outer constructor |
| `Pair a b`, `fst p`, `snd p` | pairs |
| `given c v` | `v` conditioned on `c` |
| `float-of-nat n` | a natural as a float |
| `dot a b` | dot product of two vectors, `(sum (*. a b))` |
| `make-uniform xs` | the list of `(Pair x 1/n)` |
| `normalize-seq pairs`, `sample-seq dist` | rescale a list of `(Pair value weight)` so `sample-seq` can draw from it in order |
| `int1` to `int4`, `int1=?` to `int4=?` | binary integers made of booleans, and their equality |
| `subsets-n-minus-1 xs`, `block-gibbs xs` | all sublists missing one element; Gibbs blocks built from them |
| `Marginal`, `Posterior`, `PosteriorSamples`, `Gibbs` | the query constructors (Section 5) |
| `WithConstant`, `WithPriorSample` | Gibbs initializations |

Other primitives: `native_eq` (exact equality of native integers, symbols and floats),
`native_leq` and `native_geq` (on native integers), `real_lt` and `real_geq` (also on two
floats), `get_args` and `get_constructor` (inspect a data value), and `print` (Section 2).

## 8. A worked example

This is the model of the paper's overview. A patient is infected or healthy on each of two
days. The infection rate `iota` and the recovery rate `rho` are unknown. Each day we read one
temperature, near 101.5 if infected and near 98.6 if healthy. We observe 99.8, then 100.4,
and ask whether the patient is infected on day 2, and what the rates are.

```scheme
(define temps [99.8, 100.4])

(define-type result (Result day2 rates))

(define (reading infected)
  (normal (if infected 101.5 98.6) 0.7))

;; Walk the days: draw today's status, observe today's reading, recurse.
;; Returns (Pair last-status evidence).
(define (days status rho iota ts)
  (match ts
    Nil => (Pair status true)
    Cons t rest =>
      (let ((next (if status (not (flip rho)) (flip iota)))
            (ok (real=? (reading next) t))
            (out (days next rho iota rest)))
        (Pair (fst out) (and ok (snd out))))))

(query sir
  (let ((rho  (beta 2.0 4.0))      ; recovery rate
        (iota (beta 1.0 6.0))      ; infection rate
        (out  (days false rho iota temps)))
    (Posterior (Result (fst out) [rho, iota]) (snd out))))
```

```
sir:
  (Result True [rates₀, rates₁])  0.5733433271343528
    ├ 83.17%
    │       rates₀ ~ Beta(2.0000, 4.0000)
    │       rates₁ ~ Beta(2.0000, 7.0000)
    └ 16.83%
            rates₀ ~ Beta(2.0000, 5.0000)
            rates₁ ~ Beta(2.0000, 6.0000)
  (Result False [rates₀, rates₁])  0.4266566728656472
    ├ 98.57%
    │       rates₀ ~ Beta(2.0000, 4.0000)
    │       rates₁ ~ Beta(1.0000, 8.0000)
    └ 1.43%
            rates₀ ~ Beta(3.0000, 4.0000)
            rates₁ ~ Beta(2.0000, 6.0000)
```

The pieces: the data is a list of floats. The priors are drawn in the query's `let`; `rho`
and `iota` are Beta draws used only through `flip`, so they stay conjugate. `days` draws each
day's status, observes that day's reading with `real=?`, and recurses, returning the last
status and the conjunction of all observations. The field name `rates` names the two Beta
variables `rates₀` (`rho`) and `rates₁` (`iota`).

The patient is infected on day 2 with probability 0.573. In that case the rates form a
two-component mixture. The 83.17% component is the history "healthy, then infected": `iota`
saw one non-infection and one infection, so Beta(1, 6) became Beta(2, 7). The temperatures
were integrated out exactly. Returning `iota` instead of the `Result` gives a
three-component mixture: 47.68% Beta(2, 7), 42.06% Beta(1, 8) and 10.26% Beta(2, 6).

## 9. Examples to learn from

All files are in `examples/`. Each one listed here runs in a few seconds or less.

| Topic | Files |
|---|---|
| Discrete only | `burglary_alarm`, `hmm` (lazy infinite chain), `sorted_list`, `typos` and `strings` (strings), `pcfg` and `bst` (data types, `PosteriorSamples`) |
| Beta-Bernoulli | `beta_bernoulli`, `coinBias`, `clickGraph`, `clinicalTrial1`, `secret_ballot_n04`, `fig_dawid_skene_listing`, `fig_sir_listing` |
| Gaussian | `conjugate_gaussians`, `linear_regression`, `bivariate_gaussian`, `mv_dot_regression`, `linear_transform`, `ssm` |
| Discrete and Gaussian | `gaussian-mixture`, `normal_mixture`, `outliers_n08`, `robust-linreg`, `run_walk_hrm`, `spike_and_slab`, `fig_mixture_listing`, `fig_polyreg_listing` |
| Dirichlet | `f81_joint`, `pcfg_n010` |
| Gamma-Poisson | `fig_changepoint_listing`, `soccer_n03` |
| Point masses | `GPA` (`prob_close`, and `close` on numbers) |
| `include` | `soft_kmeans_n04`, `soccer_n03`, `pcfg_n010`: each sets a size, then includes a model from `examples/shared/` |

The five `fig_*_listing.pluck` files are the programs of Figure 1 in the paper. The programs
in `tests/programs/` cover the remaining primitives, such as `exponential`, `real_lt`,
`dirichlet_eq` and `Gibbs`.

## 10. Paper notation

The paper presents a core calculus. Its constructs map to this language as follows:

| Paper | Hybrid Pluck |
|---|---|
| `observe(e)` | `(Posterior v e)` or `(given e v)` |
| `close?_G(a, b)` | `(real=? a b)` or `(close a b)` |
| `close?_P(p, v)` | `(prob_close p v)` |
| `normal(mu, sigma)`, `beta(a, b)`, `flip(p)` | `(normal mu sigma)`, `(beta a b)`, `(flip p)` |
| `⊕`, `⊙` on Gaussian reals | `+.`, `*.` |
| `asProb`, `asGReal` | not needed; numbers convert automatically |
| types `Prob`, `GReal` | checked while the program runs, not before |

The implementation adds Dirichlet, Gamma, Poisson and Exponential distributions,
multivariate Gaussians, and the sampling queries `PosteriorSamples` and `Gibbs`.
