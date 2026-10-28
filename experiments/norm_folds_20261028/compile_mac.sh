#!/bin/bash
# norm-folds Mac CPU-column compiles (MOJOLEARN_COLUMN_CPU, IDENTICAL), arm B (define off) and arm on.
# usage: compile_mac.sh <outdir> <entry-family>:<on|off> ...
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
OUT=$1; shift
MOJO=${MOJO:-/Users/andrewhendel/CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo}
SLOT=$WT/tools/compile_slot.sh
cd "$WT" || exit 2
# the main checkout's pixi env (std, MAX) for this worktree
eval "$(pixi shell-hook --manifest-path ${PIXI_MANIFEST:-/Users/andrewhendel/CascadeProjects/mojolearn/pixi.toml} 2>/dev/null)"
MOJO=$(command -v mojo)
mkdir -p "$OUT"
git rev-parse HEAD > "$OUT/source-commit.txt"
"$MOJO" --version > "$OUT/compiler.txt" 2>&1
SDK=$(xcrun --sdk macosx --show-sdk-version)
for spec in "$@"; do
  # spec: <family>:<on|off>[:<accelerator>]; a *_host family is the CPU column,
  # any other family the device binding (Apple Metal on this Mac, or the named
  # --target-accelerator)
  IFS=: read -r fam arm accel <<< "$spec"
  DEF=""; [ "$arm" = on ] && DEF="-D MOJOLEARN_IDN_NORM_FIXED_LANES=1"
  COL=""; case $fam in *_host|*_host_compile) COL="-D MOJOLEARN_COLUMN_CPU";; esac
  ENTRY=bindings/entry/entry__mojolearn_${fam}.mojo; [ -f "$ENTRY" ] || ENTRY=experiments/norm_folds_20261028/compile_${fam}.mojo
  ACC=""; tag=$arm; [ -n "${accel:-}" ] && { ACC="--target-accelerator $accel"; tag=$arm.$accel; }
  log="$OUT/$fam.$tag.log"; so="$OUT/_mojolearn_${fam}.$tag.so"
  cmd="$MOJO build --optimization-level 0 -j 10 --emit shared-lib --target-cpu apple-m1 $ACC -Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker $SDK -D MOJOLEARN_NUMERIC_IDENTICAL=1 $COL $DEF -I . -I bindings $ENTRY -o $so"
  t0=$(date +%s)
  bash "$SLOT" $cmd > "$log" 2>&1; rc=$?
  t1=$(date +%s)
  sha=$( [ -f "$so" ] && shasum -a 256 "$so" | cut -d' ' -f1 || echo none)
  echo "RESULT fam=$fam arm=$tag exit=$rc seconds=$((t1-t0)) sha256=$sha log=$log cmd=$cmd" | tee -a "$OUT/results.txt"
done
