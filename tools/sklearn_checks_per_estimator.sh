#!/bin/bash
# sklearn_checks_per_estimator.sh <python> <tests-dir> <outdir> [limit_s]
# scikit-learn's estimator checks, ONE pytest process per estimator under a hard
# wall-clock limit, so a native hang in one estimator cannot hide the rest of the
# list. Plus the non-parametrized tests of test_sklearn_compat.py once.
# Test tooling (release gate).
set -u
PY=$1; T=$2; OUT=$3; LIM=${4:-900}
mkdir -p "$OUT/per"
export MOJOLEARN_RUN_SKLEARN_CHECKS=1
F=$T/test_sklearn_compat.py
"$PY" -c "import sys; sys.path.insert(0, '$T'); import test_sklearn_compat as C; print('\\n'.join(sorted(n for n in C.SHARED_NAMES if n not in C.FLOAT32_ONLY)))" > "$OUT/estimators.txt"
run() {  # name, pytest -k expr, only
    local name=$1 expr=$2 log="$OUT/per/$1.log"
    MOJOLEARN_SKLEARN_CHECKS_ONLY=${3:-} "$PY" -m pytest -q -rfE -p no:cacheprovider --timeout=120 --timeout-method=signal -k "$expr" "$F" > "$log" 2>&1 &
    local pid=$! t=0
    while kill -0 $pid 2>/dev/null; do sleep 5; t=$((t+5)); [ $t -ge "$LIM" ] && { kill -9 $pid; echo "HUNG over ${LIM}s" >> "$log"; break; }; done
    wait $pid 2>/dev/null; local rc=$?
    printf '%s\trc=%s\t%s\n' "$name" "$rc" "$(grep -E 'passed|failed|HUNG|error' "$log" | tail -n 1)" >> "$OUT/summary.tsv"
}
: > "$OUT/summary.tsv"
run _non_parametrized "not test_scikit_learn_estimator_checks"
while read -r e; do run "$e" "test_scikit_learn_estimator_checks" "$e"; done < "$OUT/estimators.txt"
echo "DONE $(wc -l < "$OUT/summary.tsv") rows; not clean: $(grep -vcE 'rc=0' "$OUT/summary.tsv")"
