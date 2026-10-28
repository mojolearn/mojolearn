#!/bin/bash
# Compile-only closure for lane lm-resident-generate-20261028 (CPU column, both arms).
# Usage: bash compile.sh <evidence_dir> [extra mojo flags...]
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
EV=${1:?evidence dir}; shift
MAIN=${MAIN:-/Users/andrewhendel/CascadeProjects/mojolearn}
COMPILER=${MOJO:-$MAIN/.pixi/envs/default/bin/mojo}
# The main checkout's locked pixi environment provides std; the worktree has none.
MOJO_RUN=(pixi run --manifest-path "$MAIN/pixi.toml" --environment default --locked --no-install --executable "$COMPILER")
mkdir -p "$EV"
cd "$WT" || exit 2
git rev-parse HEAD > "$EV/source-commit.txt"
"${MOJO_RUN[@]}" --version > "$EV/compiler.txt" 2>&1
overall=0
for root in compile_causal_public compile_resident_session; do
  for arm in off on; do
    defs=(-D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU)
    [ "$arm" = on ] && defs+=(-D MOJOLEARN_IDN_LM_RESIDENT_GENERATE=1)
    tag=$root-$arm
    t0=$(date +%s)
    "${MOJO_RUN[@]}" build --optimization-level 0 -j 10 --emit shared-lib "${defs[@]}" "$@" -I "$WT" -I "$WT/bindings" \
      experiments/lm_resident_generate_20261028/$root.mojo -o "$EV/$tag.so" > "$EV/$tag.log" 2>&1
    rc=$?
    t1=$(date +%s)
    sha=-; [ $rc = 0 ] && sha=$(shasum -a 256 "$EV/$tag.so" | cut -d' ' -f1)
    echo "ROOT $tag rc=$rc seconds=$((t1-t0)) sha256=$sha" | tee -a "$EV/summary.txt"
    [ $rc = 0 ] || overall=1
  done
done
echo "OVERALL rc=$overall" | tee -a "$EV/summary.txt"
exit $overall
