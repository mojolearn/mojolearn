#!/bin/bash
# sklearn_checks_per_estimator.sh <python> <tests-dir> <outdir> [limit_s] [parallel]
# scikit-learn's estimator checks (tests/api/test_sklearn_estimator_checks.py, over
# every twin of docs/api/sklearn_twins.json), ONE pytest process per estimator under a
# hard wall-clock limit, so a native hang in one estimator cannot hide the rest of the
# list, <parallel> estimators at a time (default: every core; lower it only for a
# measured memory limit). Plus the non-parametrized tests of <tests-dir>/
# test_sklearn_compat.py and of the checks file once. Test tooling (release gate).
set -u
PY=$1; T=$2; OUT=$3; LIM=${4:-900}; PAR=${5:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || nproc)}
CHECKS="$(cd "$(dirname "$0")/.." && pwd)/tests/api/test_sklearn_estimator_checks.py"
mkdir -p "$OUT/per"
export MOJOLEARN_RUN_SKLEARN_CHECKS=1 PY T OUT LIM CHECKS
"$PY" -c "import sys; sys.path.insert(0, '$(dirname "$CHECKS")'); import test_sklearn_estimator_checks as C; print('\n'.join(n for n in C.TWINS if n not in C.NOT_CHECKED))" > "$OUT/estimators.txt"
run() {  # name ("_non_parametrized" runs every other test once)
    local name=$1 log="$OUT/per/$1.log" expr="test_scikit_learn_estimator_checks" only=$1 files="$CHECKS"
    [ "$name" = _non_parametrized ] && { expr="not test_scikit_learn_estimator_checks"; only=""; files="$T/test_sklearn_compat.py $CHECKS"; }
    MOJOLEARN_SKLEARN_CHECKS_ONLY=$only "$PY" -m pytest -q -rfEX -p no:cacheprovider --timeout=120 \
        --timeout-method=signal -k "$expr" $files > "$log" 2>&1 &
    local pid=$! t=0
    while kill -0 $pid 2>/dev/null; do sleep 5; t=$((t+5)); [ $t -ge "$LIM" ] && { kill -9 $pid; echo "HUNG over ${LIM}s" >> "$log"; break; }; done
    wait $pid 2>/dev/null; local rc=$?
    printf '%s\trc=%s\t%s\n' "$name" "$rc" "$(grep -E 'passed|failed|HUNG|error' "$log" | tail -n 1)" >> "$OUT/summary.tsv"
}
export -f run
: > "$OUT/summary.tsv"
{ echo _non_parametrized; cat "$OUT/estimators.txt"; } | xargs -P "$PAR" -I{} bash -c 'run "$@"' _ {}
echo "DONE $(wc -l < "$OUT/summary.tsv") rows; not clean: $(grep -vcE 'rc=0' "$OUT/summary.tsv")"
