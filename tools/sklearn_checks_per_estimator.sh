#!/bin/bash
# sklearn_checks_per_estimator.sh <python> <tests-dir> <outdir> [limit_s] [parallel]
# scikit-learn's estimator checks, ONE pytest process per estimator under a hard
# wall-clock limit, so a native hang in one estimator cannot hide the rest of the
# list, <parallel> estimators at a time. Plus the non-parametrized tests of
# test_sklearn_compat.py once. Test tooling (release gate).
set -u
PY=$1; T=$2; OUT=$3; LIM=${4:-900}; PAR=${5:-1}
mkdir -p "$OUT/per"
export MOJOLEARN_RUN_SKLEARN_CHECKS=1 PY T OUT LIM
"$PY" -c "import sys; sys.path.insert(0, '$T'); import test_sklearn_compat as C; print('\n'.join(sorted(n for n in C.SHARED_NAMES if n not in C.FLOAT32_ONLY)))" > "$OUT/estimators.txt"
run() {  # name ("_non_parametrized" runs every other test once)
    local name=$1 log="$OUT/per/$1.log" expr="test_scikit_learn_estimator_checks" only=$1
    [ "$name" = _non_parametrized ] && { expr="not test_scikit_learn_estimator_checks"; only=""; }
    MOJOLEARN_SKLEARN_CHECKS_ONLY=$only "$PY" -m pytest -q -rfE -p no:cacheprovider --timeout=120 \
        --timeout-method=signal -k "$expr" "$T/test_sklearn_compat.py" > "$log" 2>&1 &
    local pid=$! t=0
    while kill -0 $pid 2>/dev/null; do sleep 5; t=$((t+5)); [ $t -ge "$LIM" ] && { kill -9 $pid; echo "HUNG over ${LIM}s" >> "$log"; break; }; done
    wait $pid 2>/dev/null; local rc=$?
    printf '%s\trc=%s\t%s\n' "$name" "$rc" "$(grep -E 'passed|failed|HUNG|error' "$log" | tail -n 1)" >> "$OUT/summary.tsv"
}
export -f run
: > "$OUT/summary.tsv"
{ echo _non_parametrized; cat "$OUT/estimators.txt"; } | xargs -P "$PAR" -I{} bash -c 'run "$@"' _ {}
echo "DONE $(wc -l < "$OUT/summary.tsv") rows; not clean: $(grep -vcE 'rc=0' "$OUT/summary.tsv")"
