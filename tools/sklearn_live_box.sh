#!/bin/bash
# sklearn_live_box.sh <nvidia|amd|apple> <outdir> [version]
#
# scikit-learn integration on a live install: a fresh venv, the PUBLISHED
# wheels from PyPI (mojolearn + the vendor plugin), this tree's Python files
# laid over the installed package (the fixes not yet released ride on the
# released binaries), then tools/sklearn_practice.py (every estimator in real
# scikit-learn workflows) and the full test_sklearn_compat.py with fits on.
# Test tooling: run by the orchestrator on a box, never by a lane.
set -u
VENDOR=$1; OUT=$2; VER=${3:-0.8.46}
ROOT=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
PY=$(command -v python3.12 || command -v python3)
case $VENDOR in
    nvidia) PLUGIN="mojolearn-nvidia==$VER" ;;
    amd) PLUGIN="mojolearn-amd==$VER" ;;
    apple) PLUGIN="" ;;
    *) echo "vendor must be nvidia, amd or apple"; exit 2 ;;
esac
V=$OUT/venv
{ $PY -m venv "$V" || { $PY -m pip install -q virtualenv && $PY -m virtualenv -q "$V"; }; } > "$OUT/setup.log" 2>&1
"$V/bin/pip" install -q --upgrade pip >> "$OUT/setup.log" 2>&1
"$V/bin/pip" install -q --no-cache-dir "mojolearn==$VER" $PLUGIN scikit-learn pandas pytest pytest-timeout >> "$OUT/setup.log" 2>&1
echo "setup rc=$?" | tee -a "$OUT/setup.log"
SITE=$("$V/bin/python" -c "import mojolearn,os;print(os.path.dirname(mojolearn.__file__))")
# Python files only: the binaries stay the published wheel's.
(cd "$ROOT/python/mojolearn" && find . -name '*.py' -o -name '*.tsv' -o -name '*.json' | grep -v '/verify_reference/' \
    | while read -r f; do mkdir -p "$SITE/$(dirname "$f")"; cp "$f" "$SITE/$f"; done)
# A box checkout can be sparse without the tests: take them from the commit itself.
git -C "$ROOT" archive HEAD python/mojolearn/tests | tar -x --strip-components=1 -C "$(dirname "$SITE")"
"$V/bin/python" -c "import mojolearn,sklearn;print('mojolearn',mojolearn.__version__,'sklearn',sklearn.__version__)" | tee "$OUT/versions.txt"
cd "$OUT"
"$V/bin/python" "$ROOT/tools/sklearn_practice.py" "$OUT/practice.tsv" > "$OUT/practice.log" 2>&1
echo "practice rc=$?" | tee "$OUT/practice.exit"
tail -n 1 "$OUT/practice.log"
MOJOLEARN_RUN_SKLEARN_CHECKS=1 "$V/bin/python" -m pytest -q -rfE -p no:cacheprovider --timeout=120 \
    --timeout-method=signal "$SITE/tests/test_sklearn_compat.py" > "$OUT/checks.log" 2>&1
echo "checks rc=$?" | tee "$OUT/checks.exit"
tail -n 1 "$OUT/checks.log"
