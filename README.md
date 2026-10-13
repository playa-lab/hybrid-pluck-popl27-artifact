# Hybrid Pluck: artifact guide

This is the artifact for the paper *Hybrid Pluck*. It provides instructions to load and run 
a Docker image that contains:

- **Hybrid Pluck**: the implementation of the Hybrid Pluck probabilistic programming language, in Rust.
- **Benchmark programs**: the probabilistic programs used to evaluate Hybrid Pluck vs. other systems.
- **Baseline systems**: versions of PSI, HyBit, Turing.jl, SlicStan and
  Siren, together with translations of the benchmark programs into each baseline language.
- **Evaluation harness**: scripts to run the benchmarks, regenerate the paper's tables and figures,
  and generate a report comparing the run with the results from the paper.

This guide has four parts:

1. [Claims](#1-claims): a list of claims from the paper.
2. [Getting started](#2-getting-started-about-10-minutes): a quick smoke test to ensure that things are installed properly.
3. [Evaluation](#3-evaluation): instructions for running the benchmarks and comparing them with the paper.
4. [Using and extending the artifact](#4-using-and-extending-the-artifact): guidance on where to find different parts of the implementation, and how to write your own programs using Hybrid Pluck.

Nothing in the artifact needs network access at run time. All commands below use `docker run --network none`.

## AI Disclosure

We used Claude Code while implementing Hybrid Pluck. The inference algorithm that Hybrid Pluck uses and the code's architecture were primarily designed by the human authors of this submission. The benchmark program translations, benchmark harness scripts, and artifact evaluation scripts were also written with the assistance of Claude Code. All generated code was reviewed by the authors for correctness.

---

## 1. Claims

We now list the empirical claims made by the Hybrid Pluck paper, organized by section of the paper.

For each claim, we give a description of the claim, a pointer to what part of the artifact can be used to check the claim, and a forward reference to further details (later in this README). 

The `artifact ...` commands (run inside the docker container using `docker run`) are what generates various parts of the artifact (see sections 2 and 3 for details). These commands will write various logs and artifacts to the `analysis` folder, which you can then inspect. Note that these artifacts have multiple forms: some are .tex files for tables which appear in the paper while some are PDFs for generated plots.

### Section 1 (Introduction)

1. Figure 1: Hybrid Pluck computes exact posteriors for five hybrid models in about a second.

These are the programs in `examples/fig_*_listing.pluck`. The command `artifact fig1` writes the 15 panels, the lines of code and the median times. See section E4 of evaluation for more details.

2. Existing symbolic approaches time out on all five Figure 1 models.

The PSI ports are in `experiments/figure1/fig_*.psi`. The command `artifact fig1` runs them with the paper's 600 s timeout.  See section E4 of evaluation for more details.

3. For the SIR model, exact inference is polynomial ($O(T^5)$) and feasible for $T \approx 50$–$80$.

The various SIR models are listed as `examples/sir_model_*.pluck`. The `artifact run <tier>` command runs them all, and produces multiple tables in the `analysis` folder of the results. To see the SIR model runtimes, look at the `pluck_stats_full.tex` table. See section E5 of evaluation for more details.

### Section 6 (Evaluation)

4. §6: each configuration runs 15 times with a 600 s wall-clock timeout, on one core.

The paper's settings are in `experiments/tiers/paper.env` (`N=15`, `TIMEOUT=600`), and every job is single-threaded. The command `artifact run paper` runs the benchmarks with these settings. See sections E1 and E2 of evaluation for more details.

5. §6.1, Table 1: a suite of 16 applied hybrid benchmarks.

The Hybrid Pluck programs are in `examples/`, and `experiments/benchmarks.txt` lists the ones that are benchmarks. Their translations for the five baseline systems are in `benchmark_translations/{psi,hybit,turing,slicstan,siren}/`. See section E5 of evaluation for more details.

6. §6.2, Table 2: Hybrid Pluck is the fastest deterministic method on almost all standard benchmarks.

The command `artifact run <tier>` writes Table 2 to `analysis/inference_standard.tex`, in the same format as the paper, and the claims table in `analysis/comparison.md` checks this claim. See sections E3 and E5 of evaluation for more details.

7. §6.2, Table 3: Hybrid Pluck is significantly faster than PSI and HyBit on every Table 3 benchmark.

The command `artifact run <tier>` writes Table 3 to `analysis/inference_applied.tex`, and the claims table in `analysis/comparison.md` checks this claim row by row. The submitted paper's own Table 3 has one exception, `f81_rate` (see [Other notes](#other-notes)). See sections E3 and E5 of evaluation for more details.

8. §6.2: discrete inference stays fast for Hybrid Pluck and HyBit.

See the *Fully Discrete* rows of Table 2 (`analysis/inference_standard.tex`). See section E5 of evaluation for more details.

9. §6.2 and abstract: on 14 problems where existing exact systems time out after 10 minutes, Hybrid Pluck computes exact posteriors in milliseconds to seconds.

The claims table in `analysis/comparison.md` counts the models where Hybrid Pluck finishes and PSI does not. A run that skips some programs counts fewer, and `comparison.md` gives both counts. See section E3 of evaluation for more details.

10. §6.3: each approximate cell is the shortest run whose 75th-percentile relative error is below 5%.

This rule is implemented in `experiments/analysis/make_table.py`, which chooses each approximate cell of Tables 2 and 3. The error of every run is in `analysis/error.csv`, measured against the exact answers in `experiments/reference/exact_answers.csv`. See sections E5 and E9 of evaluation for more details.

11. §6.3: for small data, exact inference is as fast as or faster than many approximate methods.

Compare Hybrid Pluck's times with the approximate systems' times in Table 3 (`analysis/inference_applied.tex`), at the smaller sizes of each model. See section E5 of evaluation for more details.

12. §6.3: Siren is consistently the best approximate method.

The claims table in `analysis/comparison.md` checks this claim. See section E3 of evaluation for more details.

13. §6.3, Figure 9: approximate inference varies widely from run to run.

The command `artifact run <tier>` writes the two panels of Figure 9 to `analysis/*_approx_inf_comparison.pdf`, where each point is one run. See section E6 of evaluation for more details, including a command that makes Figure 9 on its own.

14. §6.3, Figure 9b: Turing fails to converge on Ecological Inference.

The claims table in `analysis/comparison.md` checks this claim, and Figure 9b (`analysis/king_ei_d20_approx_inf_comparison.pdf`) shows it. See sections E3 and E6 of evaluation for more details.

15. §6.4, Tables 4 and 5: BDD and SPN sizes, and the time of each stage.

SPNs and BDDs are internal data structures used by Hybrid Pluck. The command `artifact run <tier>` writes Tables 4 and 5 to `analysis/pluck_stats.tex` and `analysis/pluck_stats_full.tex`. The claims table in `analysis/comparison.md` checks every BDD and SPN count against the paper; the SPN sizes of seven programs vary between runs (see [Other notes](#other-notes)). See sections E3 and E5 of evaluation for more details.

16. §6.4: interning shrinks SPNs by orders of magnitude (Secret Ballot); the bottleneck stage differs between programs; SIR and Secret Ballot scale polynomially, the first-order HMM exponentially.

Read Table 5 (`analysis/pluck_stats_full.tex`) across the sizes of one model: the interned and tree SPN sizes of Secret Ballot, the stage times of each program, and how the times grow for SIR, Secret Ballot and the first-order HMM. See section E5 of evaluation for more details.

17. Appendix C, Table 6: the scalar each benchmark uses for error.

Each program reports this scalar as its last query. [The table in Section 4](#the-scalar-each-benchmark-reports) lists the scalar of every benchmark.

18. Appendix E: the source of the Figure 1 programs.

These are the programs in `examples/fig_*_listing.pluck`. See section E4 of evaluation for more details.

### Other notes

The submitted version of the paper (Section 6.2) claims that Hybrid Pluck is “faster than HyBit and PSI across every benchmark in Table 3.” In fact, there is one benchmark (f81_rate) for which HyBit is slightly faster than Hybrid Pluck. This is already reflected in the table, and we are planning to update the prose accordingly.

Table 3 incorrectly shows `t/o` for Turing on `pcfg_ambiguous`. In our reduced run, its two smallest budgets finished in 36 to 104 s but with error above 5%, and the two larger ones timed out, so the cell reads `r/e`. We are planning to update the table with correct results.

Tables 2 and 3 report results for a single query per program. By contrast, Tables 4 and 5 sometimes add up times for multiple queries (e.g. polyreg). 

Fig. 9b’s caption should say “1/20” instead of “1/15”.

The model called “Activity Detection” in Tables 1 and 6 is called “Run-Walk HRM” in Tables 3 and 5.

The caption of Tables 4 and 5 states that "counts are deterministic", but the SPN sizes
(interned, tree and compression) of seven programs change from run to run or between
machines. They are listed in `experiments/lists/spn_varies.txt`. The reason is that SPN interning
compares the normalized coefficients of a node's children exactly, as floating-point
numbers. When the operations run in a different order, those coefficients can round
slightly differently, and two nodes that should be identical are not interned as one
(see [How close the results should be](#how-close-the-results-should-be)).

---

## 2. Getting started (about 10 minutes)

### What you need

**Docker.** Docker Desktop on macOS or Windows, or Docker Engine on Linux. Tested with
Docker Desktop (engine 29.8) on macOS 26 on an Apple M1, with both images (amd64 under
emulation).

To check that Docker is installed and running:

```sh
docker --version
docker info --format '{{.ServerVersion}}'
```

The first should print something like `Docker version 29.8.2, build 7fc2dff`, and the
second the engine's version, such as `29.8.2`. If the second says it cannot connect to
the Docker daemon, Docker is not running: start Docker Desktop. Slightly older or newer
versions of Docker should not make a difference, although significantly outdated
versions of Docker may potentially show differences.

**Memory.** Give Docker at least 12 GB (Docker Desktop: *Settings → Resources →
Memory*). The paper's machine had 64 GB. Six of the largest programs need more than
10 GB on their own, and the commands below skip them unless you have the memory (see
[Memory](#memory-and-the-largest-programs)).

You can check how much memory Docker has with `docker info --format '{{.MemTotal}}'`.
It prints bytes: with Docker Desktop set to 12 GB, it printed `12262920192`.

**Disk.** About 20 GB free: the image is 6 to 7 GB once loaded, and a reduced run writes
about 450 MB.

You can check available disk space using `df -h`. Docker Desktop also has its own disk
limit (*Settings → Resources*), and `docker system df` shows how much of it Docker
already uses.

**CPU.** Any x86-64 or arm64 machine. Every job is single-threaded. More cores let you
run more jobs at once (`JOBS`).

### Load the image

The image is on Docker Hub, and `docker pull` picks the one for your
machine: arm64 on Apple Silicon Macs and arm64 Linux, amd64 elsewhere.

```sh
docker pull theodorelong98/hybrid-pluck:popl
docker tag theodorelong98/hybrid-pluck:popl hybrid-pluck:popl   # the name this guide uses
docker image ls hybrid-pluck                                    # shows hybrid-pluck:popl
```

`docker pull` prints `Digest: sha256:3e1d6f3c…`, which should match the first digest in
`SHA256SUMS`; that file lists the digests of both images.

The source is this GitHub repository: paths in this guide are
relative to its top folder, which is `/artifact` in the image (the image has everything
except `sample-results/`).

### Run your first Hybrid Pluck program

To test that the image has loaded correctly and can run Hybrid Pluck programs, run the following command:
```sh
docker run --rm --network none hybrid-pluck:popl pluck examples/coinBias.pluck
```
This runs the `examples/coinBias.pluck` program, which models a biased coin with a `Beta(2, 5)`
prior on the probability of `True`, and flips it 5 times with outcomes `y = [1,1,0,1,0]`. The result
should be
```
coinBias:
  p  1
        p ~ Beta(5.0000, 7.0000)
```
The row `p  1` says the query's value is the variable `p` with probability 1, and the line
below it gives `p`'s posterior: Beta(2 + 3, 5 + 2), after three heads and two tails.
`docs/LANGUAGE.md` explains the output format.

To see verbose statistics about Hybrid Pluck's performance 
(the sizes of internal data structures, breakdowns of runtime across stages of inference, etc.), 
set the environment variable `PLUCK_STATS=1`:
```sh
docker run --rm --network none -e PLUCK_STATS=1 hybrid-pluck:popl pluck examples/coinBias.pluck
```
This should give output that looks like this (the times will differ):
```
=== PLUCK_STATS ===
worlds (value,BDD-guard pairs)        : 1
continuous latents (Beta/Gaussian/…) : 1
BDD vars allocated (discrete bits)    : 5
BDD recursive calls (compile work)    : 467
BDD guard nodes  : sum=5  max=5  mean=5
SPN nodes (TREE) : sum=1  max=1  mean=1
SPN nodes (DAG)  : per-world-max=1  all-worlds-unique=1  intern-table=7
SPN leaf nodes   : sum=1  (max sum-arity=0, max prod-arity=0)
posterior mixture components (total)  : 1
time (s): parse=0.0000  symbolic-exec=0.0029  wmc=0.0000  posterior=0.0000  flatten=0.0000  display/naming=0.0000  total=0.0030
===================
inference_seconds: 0.00299104
coinBias:
  p  1
        p ~ Beta(5.0000, 7.0000)
```

### Run the smoke test (about 10 minutes)

Make an empty folder for the results and run the smoke test from it:

```sh
mkdir hybrid-pluck-results && cd hybrid-pluck-results
docker run --rm --network none -v "$PWD/results:/results" hybrid-pluck:popl artifact smoke
```

All the results will show up under `hybrid-pluck-results/results` (or `my-dir/results` if you 
choose a different name.)

The smoke test runs:

- Hybrid Pluck and five baseline systems on two small programs (`coinBias` and `changepoint_coal_k1`)
- every analysis script (although they will only pick up data from the two small programs)
- Figure 1, with PSI's (one of the baseline systems) timeout cut to 60 s

In our testing this took ~10 minutes on an Apple M1 with the arm64 image. 
Most of that is PSI timing out on the five Figure 1 models. It ends like this:

```
== Figure 1 (median of 1 runs; PSI timeout 60 s)
== error
== tables
== pluck_stats
== fig9
== overview
analysis: /results/smoke_20261009_224317/analysis
- Claims: 8 of 8 checked claims agree.
- Tables 2 and 3: 10 cells ran; 10 have the paper's status, 0 timed out here but finished in the paper, 0 were limited by this machine's memory, 0 are on the other side of the 5% error threshold, and 0 differ otherwise. 203 cells were not run.
- Tables 4 and 5: 0 counts differ from the paper, and 0 SPN sizes that vary between runs; 95 programs have no counts in this run.
comparison: /results/smoke_20261009_224317/analysis/comparison.md
```

Several of the baselines use Monte Carlo inference, which is nondeterministic. The smoke test uses relatively small sampling budgets, and so variance may be somewhat high. As a result, some of the baseline methods may fail to clear our 5%-error threshold. In this case, the generated report may say, e.g., "N are on the other side of the 5% error threshold" (for N > 0); this does not mean the smoke test failed.

The results are in `results/smoke_<date>_<time>/` on your machine. Check the generated files:

- `analysis/comparison.md`: the comparison with the paper
- `fig1/*.pdf`: the Figure 1 panels

`sample-results/` in the source has our smoke runs on both images, for comparison.

If `/results` is not mounted, `artifact` refuses to run (the smoke test only warns),
because results written inside the container are lost when it exits. Each run writes a
new folder and never overwrites an earlier one, so running a command twice is safe.

### Apptainer instead of Docker

On a cluster without Docker, the amd64 image runs under Apptainer. We tested Apptainer
1.5.3 on an x86-64 SLURM cluster with the smoke test, the unit tests and the six largest
programs (`sample-results/smoke_20261010_204902/` and `sample-results/cluster_large/`).

Build the Apptainer image once, straight from Docker Hub. This needs network access and
about 7 GB of temporary space (`APPTAINER_TMPDIR`); in our test, converting the image took
about an hour on network storage:

```sh
apptainer build hybrid-pluck-popl-amd64.sif docker://theodorelong98/hybrid-pluck:popl-amd64
```

Then use `apptainer run` where this guide uses `docker run`, binding the results folder the
same way:

```sh
mkdir -p results
apptainer run --cleanenv --bind "$PWD/results:/results" hybrid-pluck-popl-amd64.sif artifact smoke
```

- `--cleanenv` keeps your shell's environment, such as loaded modules and `PYTHONPATH`,
  out of the container.
- A setting given to `docker run` as `-e NAME=value` becomes an environment variable
  `APPTAINERENV_NAME`, for example `export APPTAINERENV_FILTER=sir_model_n72`.
- The image is read-only under Apptainer. Runs write only to `/results`, but
  `artifact test` needs `--writable-tmpfs` after `apptainer run`.
- Every job is single-threaded, so a SLURM job needs one CPU, and the memory in
  [Memory](#memory-and-the-largest-programs).

---

## 3. Evaluation

### E1. Choose a tier

A *tier* is a set of benchmark settings, in `experiments/tiers/`:

| Tier | What it runs | Time |
|---|---|---|
| `smoke` | Every system on 2 programs, 1 repeat | 10 min |
| `reduced` | Every program behind Tables 2 to 5 and Figure 1: 3 repeats, a 300 s timeout for the baselines (600 s for Pluck), and 4 Turing, 2 SlicStan and 3 Siren budgets | Tested at ~6h on an M1 Mac (16 GB RAM) with `JOBS=3`, skipping the 6 largest programs |
| `paper` | The paper's settings: 15 repeats, a 600 s timeout, and the full sweeps (11 Turing, 7 SlicStan and 11 Siren budgets) | About 180 CPU-hours |
| `extra` | The baselines' programs that no paper table uses (mostly in-between sizes), with the paper's settings | About 320 CPU-hours |

We suggest the `reduced` tier for easy evaluation as it can run in a few hours on a standard M1 Mac. Note that the reason for running Hybrid Pluck with a 600s timeout in the reduced tier is so that we can generate data for all Hybrid Pluck programs when running the reduced tier on a laptop (tested on Mac M1). The timings will be visible in the results, and most Hybrid Pluck programs take at most a few seconds to run (but some larger variants included in Table 5 need a longer timeout).

### E2. Run the reduced tier (about 6 hours)

From the folder you made for the smoke test (`hybrid-pluck-results`):

```sh
docker run --rm --network none -e JOBS=3 -e FILTER_PLUCK='!@large' \
  -v "$PWD/results:/results" hybrid-pluck:popl artifact run reduced
```

- **`JOBS=3`** runs three jobs at once. Each job is one system at one budget. On a machine
  with more cores and memory, more jobs finish sooner. Running several jobs at once makes
  the timings noisier (see [what to expect](#how-close-the-results-should-be)).
- **`FILTER_PLUCK='!@large'`** skips the six Pluck programs that need more than 10 GB
  (`experiments/lists/large.txt`). The baselines still run those programs. Leave it out
  only if Docker has about 48 GB; see [Memory](#memory-and-the-largest-programs) for what
  each one needs.
- **Keep the machine awake** for the whole run: on power, lid open. On macOS, put
  `caffeinate -i` in front of the command. If the machine sleeps, the programs running at
  the time get wrong timings and can time out.

The run writes `results/reduced_<date>_<time>/`:

| Path | Contents |
|---|---|
| `results_<system>[_<budget>].csv` | One row per repeat: status, wall time, inference time, answer |
| `results_*.txt` | The same, summarized per program |
| `logs/` | Every program's output |
| `fig1/`, `fig1.out` | Figure 1 panels, `fig1_metrics.tex` (lines of code and times) and the raw posteriors |
| `analysis/inference_standard.tex`, `inference_applied.tex` | Tables 2 and 3, in the paper's format |
| `analysis/pluck_stats.tex`, `pluck_stats_full.tex` | Tables 4 and 5 |
| `analysis/*_approx_inf_comparison.pdf` | Figure 9a and 9b |
| `analysis/error.csv` | Every approximate answer's error against the exact answer |
| `analysis/comparison.md` | The comparison with the paper (E3) |
| `analysis/*.out` | Each analysis script's log |

The paper's versions are in `experiments/reference/`. Our run of this command is in
`sample-results/reduced_20261010_004546/`.

You can redo the analysis or the comparison of any run without rerunning the benchmarks:

```sh
docker run --rm --network none -v "$PWD/results:/results" hybrid-pluck:popl artifact analyze reduced_<date>_<time>
docker run --rm --network none -v "$PWD/results:/results" hybrid-pluck:popl artifact compare reduced_<date>_<time>
```

All the scripts used for analysis and comparison to paper results are in `experiments/analysis/`,
including the script which automatically generates `comparison.md`.

### E3. Read `comparison.md`

The report has these sections:

1. **Summary**: three lines, also printed at the end of the run.
2. **The paper's claims**: the claims that can be checked automatically, 
    each with the paper's value, this run's value, and whether they agree. 
    A claim is only judged on the rows this run has. The abstract's 14
   models appear as fewer when rows are skipped (13 with `secret_ballot_n128` skipped)
3. **Differences from the paper**: each Table 2/3 cell whose status differs, sorted by
   cause:
   - *Timed out here, finished in the paper*: the machine or the tier's timeout. The note
     gives the slowdown this implies.
   - *Limited by this machine*: a Pluck program in `lists/large.txt` that timed out or
     ran out of memory.
   - *Other side of the 5% error threshold*: an approximate cell that is `r/e` in one and
     a time in the other. Sampling is random, and the reduced tier sweeps fewer budgets.
   - *Other differences*: anything else.
   - The Table 4/5 counts that differ, and the SPN sizes that vary between runs.
4. **Timing**: the ratio of this run's times to the paper's, per table and system.
5. **Figure 1**: lines of code, times and PSI status for the five models.
6. **Every cell of Tables 2 and 3**: the paper's value and this run's, side by side.

In the tables, `t/o` is a measured timeout and `r/e` means every budget had a relative
error above 5%. `__` means the system cannot express the model, so there is no program.
`n/r` means the program exists but this run did not run it (e.g. the reduced tier does
not run every program).

### E4. Figure 1

Compare the panels in `<run>/fig1/` with the paper's in `experiments/reference/fig1/`.
They have the same file names. The fonts differ, because the image has no Linux
Libertine. `comparison.md` checks the lines of code (exact) and PSI's timeouts, and
gives the times. Our reduced run took 0.02 to 1.5 s per model, against the paper's 0.016
to 1.4 s.

To run only Figure 1: `artifact fig1 [TIER]`. It takes about 50 minutes with the reduced
tier's 600 s PSI timeout. Other tables in the reduced tier use a 300s timeout for PSI,
however we use a 600 s timeout here to be able to fully reproduce the results of Figure 1
from the paper.

### E5. Tables 2 to 5

Compare the `.tex` files in `<run>/analysis/` with those in `experiments/reference/`, or
read the cell-by-cell table at the end of `comparison.md`. In Tables 2 and 3, bold marks
the fastest exact system in each row and underline the fastest approximate one, as in the
paper.

For claims 3 and 16, read the stage times and SPN sizes across the sizes of one model in
Table 5 (`pluck_stats_full.tex`), for example SIR from 12 to 72 observations, or the
first-order HMM from 2 to 14 steps.

### E6. Figure 9

`<run>/analysis/hmm_second_order_n14_approx_inf_comparison.pdf` and
`king_ei_d20_approx_inf_comparison.pdf` are Figure 9a and 9b. Compare them with
`experiments/reference/fig9/`. The reduced tier plots 4 Turing and 3 Siren budgets, so the
curves have fewer points than the paper's 11 each.

If the run skipped the large programs (`FILTER_PLUCK='!@large'`, as in E2), Figure 9b has
no Hybrid Pluck points, because `king_ei_d20` is one of them: it needs more than 10 GB. The
Turing and Siren points are still all there, since their errors are measured against the
shipped exact answers (see E9).

To make Figure 9 on its own, run only its two programs and the three systems it plots.
With the reduced tier's settings (about 35 minutes on an M1 Mac in our testing).
Docker needs about 12 GB for `king_ei_d20`.

```sh
docker run --rm --network none -e FILTER='hmm_second_order_n14 king_ei_d20' \
  -v "$PWD/results:/results" hybrid-pluck:popl artifact run reduced pluck turing siren
```

With the paper's settings instead (15 runs of each of 11 Turing and 11 Siren budgets, as
in the paper), replace `reduced` with `paper`. We estimate this takes about 10 hours,
most of it Turing's largest budgets and Hybrid Pluck's 15 runs of `king_ei_d20`; adding
`-e N=5` cuts it to about a third, with fewer points per budget. Either run writes the
two PDFs to `<run>/analysis/`; its tables mark everything else `n/r`.

```sh
docker run --rm --network none -e FILTER='hmm_second_order_n14 king_ei_d20' \
  -v "$PWD/results:/results" hybrid-pluck:popl artifact run paper pluck turing siren
```

### E7. The overview example (§2.4)

```sh
docker run --rm --network none hybrid-pluck:popl artifact overview
```

It prints the four numbers of §2.4 in a few seconds. Every `artifact run` also checks them
in `comparison.md`.

### E8. The engine's tests

```sh
docker run --rm --network none hybrid-pluck:popl artifact test
```

This runs `cargo test --release`. It takes about 25 seconds and should report 425 passed
tests and none failed.

The tests in `tests/gibbs_tests.rs` check the sampling queries (`Gibbs`,
`PosteriorSamples`) against statistical tolerances, with new random numbers on every run.
This means they can occasionally fail. Run the command
again, and it should pass. No benchmark uses these queries, and other tests should not fail.

### E9. The exact answers (optional, about 1 hour)

`experiments/reference/exact_answers.csv` has the exact answer of every benchmark, which
the approximate systems' errors (Tables 2 and 3, Figure 9) are measured against. Each is
Pluck's posterior reduced to the benchmark's scalar ([Table 6](#the-scalar-each-benchmark-reports)),
except for five Table 2 models that Pluck cannot express, whose answers are computed with
SciPy. To regenerate the file, run Pluck once on every benchmark, then write the file from
that run's logs:

```sh
docker run --rm --network none -e N=1 -v "$PWD/results:/results" hybrid-pluck:popl artifact run paper pluck
docker run --rm --network none -v "$PWD/results:/results" hybrid-pluck:popl artifact exact paper_<date>_<time>
```

The second command writes `results/paper_<date>_<time>/exact_answers.csv`, to compare
with `experiments/reference/exact_answers.csv`. The run's `comparison.md` already compares
its answers with that file, in the last row of the claims table. The last digits can
differ between machines, so expect a largest relative difference of about 1e-14.

The first command took about an hour on an M1 with 11.4 GB for Docker. Four of the six
programs in `experiments/lists/large.txt` need more memory than that, so they time out or
run out of memory, and their rows come out as `nan,dummy`. With less than 48 GB for
Docker, add `-e FILTER='!@large'` to skip all six, which takes about 15 minutes and
leaves all six rows as dummies. Our answers for those six come from the runs in
`sample-results/cluster_large/`.

### How close the results should be

From our reduced run on an Apple M1 with 16 GB (Docker had 11.4 GB), with `JOBS=3` and the
large programs skipped (`sample-results/reduced_20261010_004546/`):

- **Claims:** all 10 that can be checked agreed.
- **Table 4/5 counts:** all 505 checked counts matched exactly. 19 SPN sizes from
  `spn_varies.txt` differed, by 0.03% to 14%.
- **Table 2/3 statuses:** 201 of 219 cells matched the paper's.
  - 11 timed out here but finished in the paper. That's the 300 s timeout and a slower
    machine: HyBit was 2.1 to 2.4× slower than the paper, and PSI 1.2 to 2×. The timeout covers
    wall time, which includes HyBit's Julia start-up of up to about 200 s.
  - 6 approximate cells were on the other side of the 5% threshold. Sampling is random,
    and the reduced tier sweeps fewer and smaller budgets than the paper.
  - 1 differed otherwise: Turing on `pcfg_ambiguous` is incorrectly labeled as `t/o` in the paper
    (see [Other notes](#other-notes)) while it is `r/e` here. Its two smallest budgets finished, but with too much error.
- **Times:** Hybrid Pluck was 1.1× slower than the paper on Table 2 and 1.8× on Table 3
  (geometric means), and up to 13× on single programs. The worst cases come from memory
  pressure while three jobs share 11 GB. `soccer_n07` took 91 s here, against 9.4 s when
  it ran alone.

Timings will be impacted by virtualization overhead, memory pressure, and different hardware.
You should also expect more timeouts (especially if you are running with a lower timeout threshold). 
Memory pressure seems to cause the largest slowdowns seen for the largest program sizes.
If you would like to get more accurate timings for those programs, you can run with `JOBS=1`
(although this will take longer to run).

SPN sizes vary across runs because SPN interning uses floating-point comparison for equality testing.
In particular, sum and product nodes are interned based on the *normalized coefficients* of their children.
Different orders of floating point operations lead to slightly different rounding in the normalized
coefficients, so that two identical SPN sum nodes may fail to be interned to the same node
as their normalized coefficients differ very slightly, leading to fewer SPN nodes being interned.

### Memory and the largest programs

Peak memory and wall time of the largest Pluck programs, each run alone. We ran them in two
places: in the arm64 image on an M1 with 11.4 GB for Docker, and in the amd64 image under
Apptainer on one core of an Intel Xeon Platinum 8562Y+ with 64 GB. The Xeon is the
Emerald Rapids type the paper used.

| Program | M1, 11.4 GB | Xeon, 64 GB | Paper |
|---|---|---|---|
| `changepoint_coal_n040` | 6.2 GB, 50 s | | 31 s |
| `hmm_first_order_n14` | 6.1 GB, 175 s | | 44 s |
| `hmm_second_order_n14` | 6.3 GB, 164 s | | 47 s |
| `sir_model_n60` | 6.1 GB, 305 s | | 39 s |
| `king_ei_d20` † | 10.4 GB, 268 s | 10.4 GB, 24 s | 23 s |
| `secret_ballot_n128` † | 10.9 GB, 470 s | 12.3 GB, 77 s | 71 s |
| `sir_model_n72` † | timed out at 600 s | 14.7 GB, 111 s | 121 s |
| `changepoint_coal_n050` † | timed out at 600 s | 15.4 GB, 136 s | 178 s |
| `soccer_n08` † | ran out of memory | 21.3 GB, 42 s | 48 s |
| `king_ei_d15` † | ran out of memory | 40.8 GB, 67 s | 92 s |

† In `experiments/lists/large.txt`. Smaller programs need less memory.

The paper's time is from Table 3, or the sum of Table 5's stages for the rows only in
Table 5. The M1 times are long because the 16 GB machine swaps Docker's memory. Docker
needs about 16 GB for all but the last four programs, 24 GB for all but `king_ei_d15`,
and 48 GB for all of them.

To run some of them on a bigger machine, name them with `FILTER` (next section).
`sample-results/cluster_large/` has the Xeon runs.

### Running part of a tier

Any tier setting can be overridden with `-e NAME=value`:

| Setting | Meaning |
|---|---|
| `JOBS` | Jobs run at once (default 1) |
| `N`, `TIMEOUT` | Repeats and timeout (s), for every system |
| `N_<SYS>`, `TIMEOUT_<SYS>` | The same for one system, e.g. `TIMEOUT_PSI=600` |
| `FILTER`, `FILTER_<SYS>` | Only programs matching these names or globs. `@NAME` stands for `experiments/lists/NAME.txt`, and `!` excludes |
| `TURING_SWEEP`, `SLICSTAN_SWEEP`, `SIREN_SWEEP` | The budgets swept |

Naming systems after the tier runs only those systems. For example, to rerun one cell
with the paper's settings (15 repeats, so about 30 minutes, with 16 GB or more for Docker):

```sh
docker run --rm --network none -e FILTER=sir_model_n72 \
  -v "$PWD/results:/results" hybrid-pluck:popl artifact run paper pluck
```

The tables and `comparison.md` of a partial run mark everything it did not run as `n/r`.

---

## 4. Using and extending the artifact

### Tutorial

`docs/TUTORIAL.ipynb` walks through creating and testing a few different regression models in Hybrid Pluck,
as well as building some of your own.

Note that the notebook is intended to run *locally* on your machine, and uses `docker run` calls to execute
Hybrid Pluck in a docker container. Before starting it, load the image as `hybrid-pluck:popl`
(see [Load the image](#load-the-image)); the notebook explains how to use a binary built from
source instead. You also need [`uv`](https://docs.astral.sh/uv/) on your machine, which we
recommend for running the tutorial:

```sh
uv run --with numpy,scipy,matplotlib,notebook jupyter notebook docs/TUTORIAL.ipynb
```

### File map

| Path | Contents |
|---|---|
| `src/` | The Hybrid Pluck engine (Rust) |
| `stdlib.pluck` | The standard library |
| `vendor/rsdd/` | The rsdd decision-diagram library (third party, MIT) |
| `tests/` | Unit and program tests (`tests/programs/` has programs with expected answers) |
| `examples/` | Example programs: the 97 benchmarks, the Figure 1 listings (`fig_*_listing.pluck`), and models shared through `include` (`examples/shared/`) |
| `docs/LANGUAGE.md` | The Hybrid Pluck language |
| `docs/TUTORIAL.ipynb` | The tutorial notebook (see [Tutorial](#tutorial)) |
| `benchmark_translations/{psi,hybit,turing,slicstan,siren}/` | The benchmarks in each baseline's language |
| `benchmark_translations/generation_scripts/` | Generators for the PSI and HyBit scaling variants |
| `experiments/artifact` | The `artifact` command |
| `experiments/run_suite.sh`, `run_bench.sh` | Run a tier, and run one system on a set of programs |
| `experiments/tiers/`, `lists/` | Tier settings and named program lists |
| `experiments/benchmarks.txt` | The Pluck programs that are benchmarks |
| `experiments/analysis/` | Scripts behind each table, figure and the comparison (below) |
| `experiments/figure1/` | Figure 1's script, the PSI ports of its models and the SIR panel queries |
| `experiments/reference/` | The paper's tables and figures, `paper_reference.csv` (every cell, extracted from the tables) and `exact_answers.csv` |
| `experiments/drivers/`, `slicstan-data/`, `docker/` | Glue for SlicStan and Siren, SlicStan's data, and the image's build files |
| `Dockerfile` | The image, for both architectures |
| `SHA256SUMS` | The digests of the images on Docker Hub |
| `sample-results/` | Our smoke, reduced and cluster runs (see its README) |

Which script makes what (all in `experiments/analysis/`):

| Output | Script |
|---|---|
| Tables 2 and 3 | `make_table.py` |
| Tables 4 and 5 | `make_pluck_stats_table.py` |
| Figure 9 | `make_approx_plots.py` |
| `error.csv` | `compute_error.py` |
| Figure 1 | `../figure1/make_figure1_panels.py` |
| §2.4 numbers | `overview_sir.py` |
| `comparison.md` | `compare_to_paper.py` (paper values from `extract_reference.py`) |
| `reference/exact_answers.csv` | `make_exact_answers.py` (`artifact exact`) |

### What is new

| Part | Where | Status |
|---|---|---|
| Hybrid extension: conjugate families, symbolic likelihoods, sum-product networks, exact posteriors | `src/inference/conjugate_pairs/`, `src/inference/spn/`, and the continuous parts of `src/language/` | New in this paper |
| Pluck core: language and lazy knowledge compilation | `src/language/`, `src/inference/lazy_kc/` | A Rust port of Pluck (Bowers et al., POPL 2025), made for this work |
| rsdd | `vendor/rsdd/` | Third party |
| Benchmarks, translations, harness and analysis | `examples/`, `benchmark_translations/`, `experiments/` | New |

The baselines (PSI, HyBit, Turing.jl, SlicStan, Siren) are third-party systems, which the
image builds from fixed versions:

| System | Version |
|---|---|
| PSI | commit `ef8a9e52c94089a0b4e547ac3ff99db10f3517dd` (the paper does not record the commit it used) |
| HyBit | Alea.jl commit `cd9a09e75e74707b523fcfc9df9ef432ab0d475f`, Julia 1.8.5 |
| Turing.jl | 0.29.3, Julia 1.8.5 (`benchmark_translations/turing/Manifest.toml`) |
| SlicStan | commit `bbe12087d9c4286e1f54d7edb4dfee60782839c7`, CmdStan 2.31.0 |
| Siren | commit `d3ca59cf335c2d4596fa778a987e976e9741decb` |
| Rust | 1.96.0 |

`docker inspect hybrid-pluck:popl` shows the same versions as image labels.

### Building the image yourself

The images were built from the source with its `Dockerfile`. To build one yourself:

```sh
docker build -t hybrid-pluck:popl .      # from the top folder of the source
```

This builds the image for your machine. Add `--platform linux/amd64` or
`--platform linux/arm64` to build the other one, under emulation. On an M1 the build took
31 minutes for arm64 and 37 minutes for amd64. It needs network access and about 20 GB of
Docker's disk for the image and the build cache (`docker builder prune` frees the cache
afterwards).

Every external dependency is pinned: the base image's digest, the toolchain versions and
upstream commits (the `ARG`s in the `Dockerfile`), and the Julia and Python environments
in `experiments/docker/pins/`. Only Ubuntu's packages come from the 22.04 archive at build
time. So a new build has the same systems as ours, but is not byte-for-byte identical.

### Writing and running your own programs

`docs/LANGUAGE.md` describes the language. To run a program from your machine, mount its
folder:

```sh
docker run --rm -v "$PWD/my-models:/artifact/my-models" hybrid-pluck:popl pluck my-models/model.pluck
```

`include` paths are relative to `/artifact`, so `(include "examples/shared/...")` works.
`PLUCK_STATS=1` (`docker run -e PLUCK_STATS=1 ...`) adds the stage times and the BDD and
SPN sizes. For an interactive shell in the image: `docker run --rm -it hybrid-pluck:popl`.

To build the engine without Docker, install Rust (https://rustup.rs) and run, from the
repository root:

```sh
cargo build --release
target/release/pluck examples/coinBias.pluck
```

The baselines are only available in the image. The `Dockerfile` shows how each one is
built.

### Adding a benchmark

1. Write `examples/<name>.pluck`. Its last query must be the scalar used for error: the
   probability of a Boolean, the mean of an integer, or the mean of a continuous variable.
2. Add `<name>` to `experiments/benchmarks.txt`.
3. Add a translation for each baseline that can express it, in
   `benchmark_translations/<system>/`, printing `RESULT: <value>` with the same scalar. Look
   at an existing benchmark for each system's conventions. SlicStan also needs
   `experiments/slicstan-data/<name>.json` and `<name>.query`.
4. Run it, and add its exact answer to `experiments/reference/exact_answers.csv` as a row
   `<name>,<value>,pluck`. The value is Pluck's answer reduced to the scalar, which this
   prints (inside the image, from `/artifact/experiments/analysis`):

   ```sh
   ../.venv/bin/python -c 'import sys, compute_error as c; print(c.reduce_pluck(open(sys.argv[1]).read()))' \
     /results/<run>/logs/pluck_<name>.log
   ```

   `artifact exact` regenerates the whole file from the Pluck logs of complete runs
   ([E9](#e9-the-exact-answers-optional-about-1-hour)).
5. To show it in Table 2 or 3, add a row to `TABLE1_SECTIONS` (Table 2) or
   `TABLE2_GROUPS` (Table 3) in `make_table.py`.

### The scalar each benchmark reports

This is the paper's Table 6. Approximate systems' errors are measured on these quantities,
against `experiments/reference/exact_answers.csv`.

| Programs | Quantity |
|---|---|
| `addFun_sum` | E[x₁+x₂+x₃] |
| `conjugate_gaussians` | E[a] |
| `linear_regression` | E[a], the intercept |
| `clickGraph` | E[similarity] |
| `clinicalTrial1` | P(isEffective) |
| `clinicalTrial2` | E[probIfControl] |
| `coinBias` | E[bias] |
| `GPA` | P(school 1) |
| `normal_mixture` | E[θ], the mixture weight |
| `spike_and_slab` | P(g₁), the first inclusion indicator |
| `polyreg` | P(point 0 is an outlier) |
| `run_walk_hrm` | E[baseHr], the baseline heart rate |
| `sir_model_*` | E[inf], the infection rate |
| `f81_rate`, `f81_joint` | E[p], the retention probability |
| `changepoint_coal_k1` | E[e], the early disaster rate |
| `changepoint_coal_n*` | E[k], the number of changepoints |
| `dawid_skene_*` | E[tp₀], the first annotator's sensitivity |
| `hmm_first_order_*`, `hmm_second_order_*`, `hmm_factorial_*` | E[μ₁], the first emission mean |
| `king_ei_*` | E[b_w], the White-voter support rate |
| `outliers_*` | E[α], the slope |
| `soft_kmeans_*` | E[μ₁], the first cluster's first coordinate |
| `secret_ballot_*` | E[p₀], the first voter's yes-tendency |
| `soccer_*` | P(team 0 is "attack") |
| `pcfg_*` | The first production's probability |
