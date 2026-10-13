# Sample results

Outputs of runs of the artifact, for comparing with your own. Each folder is a
complete run directory as `artifact` writes it; start with its
`analysis/comparison.md`.

| Folder | Command | Machine | Time |
|---|---|---|---|
| `smoke_20261009_224317` | `artifact smoke` | M1, arm64 image | 9 min |
| `smoke_20261009_225246` | `artifact smoke` | M1, amd64 image under Rosetta emulation | 11 min |
| `reduced_20261010_004546` | `artifact run reduced` with `-e JOBS=3 -e FILTER_PLUCK='!@large'` | M1, arm64 image | 5 h 40 min |
| `smoke_20261010_204902` | `artifact smoke` | Xeon, amd64 image under Apptainer | 9 min |
| `cluster_large/large_<program>` | `artifact run paper pluck` with `FILTER=<program>` and `N=1` | Xeon, amd64 image under Apptainer, 64 GB | 1 to 3 min each |

The M1 runs used an Apple M1 with 16 GB of memory, Docker Desktop having 11.4 GB
of it, with `docker run --network none`, and the images from commit `128815f`.
The Xeon runs used one core of an Intel Xeon Platinum 8562Y+ (Emerald Rapids, as
in the paper) in a SLURM job, with Apptainer 1.5.3 and the amd64 image from
commit `9fe7c81`. The `cluster_large` runs are the six programs in
`experiments/lists/large.txt`, one job each with 64 GB.

All of them were re-analysed (`artifact analyze` and `artifact compare`) with the
final `experiments/reference/exact_answers.csv`, whose last three exact answers
come from the `cluster_large` runs.

Differences from a run of your own:

- Each run's `scratch/` folder (SlicStan's compiled models) is left out.
- Siren's logs are gzipped (`logs/siren_*.log.gz`); read them with `zcat`. The
  analysis does not read them, so `artifact analyze` works on these folders as
  they are.
