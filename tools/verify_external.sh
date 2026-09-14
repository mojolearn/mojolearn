#!/bin/sh
# tools/verify_external.sh: reproduce one column of mojolearn's three-vendor
# identity record on YOUR box, from the PyPI wheel, and diff it against the
# committed columns. No rental plumbing, no dataset store, no secrets.
#
#   sh tools/verify_external.sh <box-label> [record-dir] [version]
#
#   box-label   names your machine, lowercase: nvidia-rtx4090-sm_89,
#               amd-mi300x-gfx942, apple-m4. It is refused if it is a
#               placeholder (the tool wants to know what ran).
#   record-dir  the committed record to compare against; default the newest
#               bench/results/identity_break/*-lanes directory.
#   version     the wheel to install; default the version the record ran.
#
# What it does: installs the wheel, clones this repository at the commit the
# record's columns were made from (so the lane list is the record's), runs
# every lane on nine fixtures twice under MOJOLEARN_NUMERIC_MODE=identical,
# writes <label>.json beside this script's output directory, and diffs it
# cell by cell against the committed Apple, NVIDIA and AMD columns. Exit 0
# means every cell of your column equals the three committed columns. A
# DIVERGENT cell is a finding: keep the JSON and open an issue with it.
# Needs: python3 with pip and numpy, git, and one supported GPU (a Metal
# Apple silicon Mac, CUDA sm_89 or sm_90a, HIP gfx942; other architectures
# are not in the wheel and the import refuses by name).
set -eu
LABEL=${1:?box label, e.g. nvidia-rtx4090-sm_89}
HERE=$(cd "$(dirname "$0")/.." && pwd)
RECORD=${2:-$(ls -d "$HERE"/bench/results/identity_break/*-lanes 2>/dev/null | sort | tail -1)}
[ -d "$RECORD" ] || { echo "no record directory at $RECORD" >&2; exit 2; }
COMMIT=$(python3 -c "import json,glob,sys; print(json.load(open(sorted(glob.glob('$RECORD/*.json'))[0]))['commit'])")
VERSION=${3:-$(python3 -c "import json,glob; d=json.load(open(sorted(glob.glob('$RECORD/*.json'))[0])); print(d.get('package',{}).get('version','0.8.5'))")}
OUT="$HERE/verify_external_out"; mkdir -p "$OUT"
echo "record=$RECORD commit=$COMMIT wheel=mojolearn==$VERSION label=$LABEL"
python3 -m pip install --quiet "mojolearn==$VERSION"
# the harness at the record's commit, so the lanes and fixtures are the record's;
# the ESTIMATORS are the installed wheel's, which is the point
if [ ! -d "$OUT/src/.git" ]; then git clone --quiet https://github.com/mojolearn/mojolearn.git "$OUT/src"; fi
git -C "$OUT/src" checkout --quiet "$COMMIT"
python3 -c "import mojolearn, sys; print('loaded', mojolearn.__version__, 'vendor', mojolearn.vendor(), 'mode', mojolearn.numeric_mode(), 'from', mojolearn.__file__)"
# provenance: the harness reads the commit from the clone; every loaded binding's sha256 lands in the JSON
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMMIT="$COMMIT" \
    python3 "$OUT/src/tools/identity_break.py" --vendor "$LABEL" --json "$OUT/$LABEL.json" | tee "$OUT/$LABEL.txt"
set +e
python3 "$OUT/src/tools/identity_break.py" --diff "$RECORD"/*.json "$OUT/$LABEL.json" | tee "$OUT/diff.$LABEL.txt"
RC=$?
set -e
echo "diff exit $RC (0 = every cell of $LABEL equals the committed columns); column at $OUT/$LABEL.json"
exit $RC
