#!/usr/bin/env bash
#
# run_siren.sh <program.si>
#
# Thin adapter so Siren fits the cross-PPL harness (container/run_bench.sh).
# Siren prints its posterior scalar under a `===== Result =====` marker (the next
# line) and its inference time under `===== Evaluation Time =====`. run_bench.sh
# instead greps for `RESULT: <scalar>` and `inference_seconds: <float>` (like
# hybit/turing/mappl), so this wrapper runs Siren, echoes its full output (so the
# raw log is preserved), then re-emits those two lines in the harness format.
#
# Inference plan (per the study): continuous latents are SYMBOLIC, discrete
# latents are SMC samples, run under semi-symbolic inference (-m ssi). The knobs
# come from the environment (set by run_bench.sh); defaults below let the wrapper
# be run standalone.
set -uo pipefail

SIREN_BIN="${SIREN_BIN:-siren}"
SIREN_METHOD="${SIREN_METHOD:-ssi}"     # ssi = semi-symbolic (the hybrid method)
SIREN_HANDLER="${SIREN_HANDLER:-smc}"   # smc handler (particles) for the sampled latents
SIREN_PARTICLES="${SIREN_PARTICLES:-100}"

f="${1:?usage: run_siren.sh <program.si>}"

# --verbose makes Siren print the `===== Mixture =====` block (each particle's
# symbolic family + weight). The mean pipeline below is unchanged; the moment
# analysis (compute_moment_error.py via moments.py) parses that block from the
# preserved raw log to score higher-order moments (variance/skew/kurtosis).
out=$("$SIREN_BIN" "$f" -m "$SIREN_METHOD" -l "$SIREN_HANDLER" -p "$SIREN_PARTICLES" --verbose 2>&1)
rc=$?

# preserve Siren's raw output for the log.
printf '%s\n' "$out"

# re-emit the scalar + time in the harness's grepped format. The value/time each
# sit on the line AFTER their marker.
t=$(printf '%s\n'   "$out" | awk '/^===== Evaluation Time =====/{getline; print; exit}')
val=$(printf '%s\n' "$out" | awk '/^===== Result =====/{getline; print; exit}')
[ -n "$t" ]   && printf 'inference_seconds: %s\n' "$t"
[ -n "$val" ] && printf 'RESULT: %s\n' "$val"

exit "$rc"
