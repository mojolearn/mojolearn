#!/bin/bash
# Run ONLY as a root-managed serial M3 CMD job, never alongside scored work.
# STAGE TAG previous_A.npz previous_B.npz [reviewed_baseline_A.npz]
# ee-small <- passed MCD cap3000; mcd-full <- passed EE cap3000;
# ee-full <- passed full MCD. Each stage is its own queue entry.
set -euo pipefail
stage=$1 tag=$2 gate_a=$3 gate_b=$4 baseline=${5:-}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
py=$HOME/board-0834/cache/venv/bin/python
source_sha=77520069efe12b704444ef453d119b8c015b6cfa
case "$stage" in
  ee-small) lane=elliptic-envelope; rows=3000; prior=min-cov-det; prior_rows=3000 ;;
  mcd-full) lane=min-cov-det; rows=; prior=elliptic-envelope; prior_rows=3000 ;;
  ee-full) lane=elliptic-envelope; rows=; prior=min-cov-det; prior_rows=full ;;
  *) echo "MCDQ invalid stage $stage" >&2; exit 2 ;;
esac
out=$HOME/afc-def/$tag
[ ! -e "$out/A.npz" ] && [ ! -e "$out/B.npz" ] && [ ! -e "$out/PASS" ] || {
  echo "MCDQ refusing repeated fit tag $tag" >&2; exit 2;
}
# Verify the prerequisite really belongs to the required stage, then rerun
# only its artifact comparison (no model fit, no scored measurement).
"$py" - "$gate_a" "$gate_b" "$prior" "$prior_rows" <<'PY'
import sys
import numpy as np
for path in sys.argv[1:3]:
    with np.load(path) as z:
        assert str(z['lane']) == sys.argv[3], 'wrong prerequisite lane'
        assert str(z['dataset']) == 'taxi', 'wrong prerequisite dataset'
        n = int(z['shape'][0])
        assert n > 3000 if sys.argv[4] == 'full' else n == int(sys.argv[4]), 'wrong prerequisite size'
PY
"$py" tools/mcd_compat_quality.py compare "$gate_a" "$gate_b"
# Validates exact source scope + manifest hashes; all copies occur inside
# this serial job, before timing. Helper-only revisions reuse compiled arms.
"$py" "$HOME/mq/verified_arms.py" "$source_sha" x_decomp MOJOLEARN_MCD_BATCH_MMA "$tag" --stage-only
export MCD_COMPILED_SOURCE="$source_sha" MCD_DEFINE=MOJOLEARN_MCD_BATCH_MMA
export MCD_EXECUTION_POLICY=serial-queue-required
export MCD_A_SO="$out/A.so" MCD_B_SO="$out/B.so"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
if [ -n "$baseline" ]; then export MCD_BASELINE_NPZ="$baseline"; else unset MCD_BASELINE_NPZ; fi
# Quality and fit_ms come from the SAME single fit per arm, never a second
# timing pass after quality. A valid reused baseline is not fitted again.
bash tools/mcd_compat_ab.sh "$tag" taxi "$lane" "$rows"
