#!/bin/bash
# Compile-only closure for lane mamba-folds (2026-10-28): both defines, each arm.
# Usage: bash compile.sh <evidence_dir> <roots...>   (env DEFSETS, COLUMN_DEFS, EXTRA, MOJO_RUN_MODE)
#   roots: host (bindings/_mojolearn_mamba_host.mojo, shared-lib)
#          device (bindings/_mojolearn_mamba.mojo, shared-lib)
#          oracle (mamba/checks/mamba_backward_host_oracle_check.mojo, executable)
#          training (bindings/_mojolearn_training.mojo, shared-lib; the Samba train step)
#   DEFSETS: space-separated arms; "off" or a comma list of defines (default: off m1 bwd both)
set -u
WT=$(cd "$(dirname "$0")/../.." && pwd)
EV=${1:?evidence dir}; shift
ROOTS=${*:-host}
MAIN=${MAIN:-/Users/andrewhendel/CascadeProjects/mojolearn}
if [ -n "${MOJO_BIN:-}" ]; then
  MOJO_RUN=("$MOJO_BIN")
else
  COMPILER=${MOJO:-$MAIN/.pixi/envs/default/bin/mojo}
  MOJO_RUN=(pixi run --manifest-path "$MAIN/pixi.toml" --environment default --locked --no-install --executable "$COMPILER")
fi
JOBS=${JOBS:-10}
DEFSETS=${DEFSETS:-"off m1 bwd both"}
mkdir -p "$EV"
cd "$WT" || exit 2
git rev-parse HEAD > "$EV/source-commit.txt"
"${MOJO_RUN[@]}" --version > "$EV/compiler.txt" 2>&1
overall=0
for root in $ROOTS; do
  case $root in
    host) entry=bindings/_mojolearn_mamba_host.mojo; emit=(--emit shared-lib); ext=so ;;
    device) entry=bindings/_mojolearn_mamba.mojo; emit=(--emit shared-lib); ext=so ;;
    oracle) entry=mamba/checks/mamba_backward_host_oracle_check.mojo; emit=(); ext=bin ;;
    training) entry=bindings/_mojolearn_training.mojo; emit=(--emit shared-lib); ext=so ;;
    *) echo "unknown root $root"; exit 2 ;;
  esac
  for arm in $DEFSETS; do
    defs=(-D MOJOLEARN_NUMERIC_IDENTICAL=1 ${COLUMN_DEFS--D MOJOLEARN_COLUMN_CPU})
    case $arm in
      off) ;;
      m1) defs+=(-D MOJOLEARN_IDN_MAMBA1_CHUNKED_SCAN=1) ;;
      bwd) defs+=(-D MOJOLEARN_IDN_MAMBA_BWD_FOLDS=1) ;;
      both) defs+=(-D MOJOLEARN_IDN_MAMBA1_CHUNKED_SCAN=1 -D MOJOLEARN_IDN_MAMBA_BWD_FOLDS=1) ;;
    esac
    tag=$root-$arm
    t0=$(date +%s)
    "${MOJO_RUN[@]}" build --optimization-level 0 -j "$JOBS" "${emit[@]}" "${defs[@]}" ${EXTRA:-} -I "$WT" -I "$WT/bindings" \
      "$entry" -o "$EV/$tag.$ext" > "$EV/$tag.log" 2>&1
    rc=$?
    t1=$(date +%s)
    sha=-; [ $rc = 0 ] && sha=$( (sha256sum "$EV/$tag.$ext" 2>/dev/null || shasum -a 256 "$EV/$tag.$ext") | cut -d' ' -f1)
    echo "ROOT $tag entry=$entry rc=$rc seconds=$((t1-t0)) sha256=$sha" | tee -a "$EV/summary.txt"
    [ $rc = 0 ] || overall=1
  done
done
echo "OVERALL rc=$overall" | tee -a "$EV/summary.txt"
exit $overall
