#!/bin/bash
# Compile-only closure for lane decode-resident-20261028 (both arms of each define).
# Usage: bash compile.sh <evidence_dir> [extra mojo flags...]
#   ROOTS (default "compile_decode_resident compile_train_one_wait"), DEFINES_FOR_<root> pick the arm define.
#   COLUMN_DEFS (default "-D MOJOLEARN_COLUMN_CPU"); on a GPU-target leg pass COLUMN_DEFS="" and
#   --target-accelerator sm_89 as an extra flag. MOJO overrides the compiler path; MAIN the pixi manifest checkout.
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
EV=${1:?evidence dir}; shift
MAIN=${MAIN:-/Users/andrewhendel/CascadeProjects/mojolearn}
COMPILER=${MOJO:-$MAIN/.pixi/envs/default/bin/mojo}
JOBS=${JOBS:-10}
if [ -n "${NO_PIXI:-}" ]; then
  MOJO_RUN=("$COMPILER")
else
  MOJO_RUN=(pixi run --manifest-path "$MAIN/pixi.toml" --environment default --locked --no-install --executable "$COMPILER")
fi
ROOTS=${ROOTS:-"compile_decode_resident compile_train_one_wait"}
mkdir -p "$EV"
cd "$WT" || exit 2
git rev-parse HEAD > "$EV/source-commit.txt"
"${MOJO_RUN[@]}" --version > "$EV/compiler.txt" 2>&1
overall=0
for root in $ROOTS; do
  case $root in
    compile_decode_resident) define=MOJOLEARN_IDN_DECODE_RESIDENT_IO ;;
    compile_train_one_wait) define=MOJOLEARN_IDN_TRAIN_ONE_WAIT ;;
    *) echo "unknown root $root"; exit 2 ;;
  esac
  [ -f "experiments/decode_resident_20261028/$root.mojo" ] || { echo "ROOT $root missing"; overall=1; continue; }
  for arm in off on; do
    defs=(-D MOJOLEARN_NUMERIC_IDENTICAL=1 ${COLUMN_DEFS--D MOJOLEARN_COLUMN_CPU})
    [ "$arm" = on ] && defs+=(-D "$define=1")
    tag=$root-$arm
    t0=$(date +%s)
    "${MOJO_RUN[@]}" build --optimization-level 0 -j "$JOBS" --emit shared-lib "${defs[@]}" "$@" -I "$WT" -I "$WT/bindings" \
      experiments/decode_resident_20261028/$root.mojo -o "$EV/$tag.so" > "$EV/$tag.log" 2>&1
    rc=$?
    t1=$(date +%s)
    sha=-
    if [ $rc = 0 ]; then
      sha=$( (shasum -a 256 "$EV/$tag.so" 2>/dev/null || sha256sum "$EV/$tag.so") | cut -d' ' -f1)
    fi
    echo "ROOT $tag rc=$rc seconds=$((t1-t0)) sha256=$sha" | tee -a "$EV/summary.txt"
    [ $rc = 0 ] || overall=1
  done
done
echo "OVERALL rc=$overall" | tee -a "$EV/summary.txt"
exit $overall
