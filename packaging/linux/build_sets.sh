#!/usr/bin/env bash
# Build ONE vendor's three binary sets on the box that has that vendor's GPU,
# stage the MAX runtime libraries beside them, read the vendor back out of
# every binary, and record every size. RUNS ON A RENTED BOX, never on the Mac.
#
#   bash packaging/linux/build_sets.sh <dest-dir>
#
# Output, under <dest-dir>:
#   sets/<vendor>/_mojolearn*.so                 the FAST set
#   sets/<vendor>/deterministic/_mojolearn*.so   the DETERMINISTIC set
#   sets/<vendor>/identical/_mojolearn*.so       the IDENTICAL set
#   sets/<vendor>/.libs/*.so                     the MAX runtime closure
#   sets/<vendor>/manifest.json                  sizes, hashes, read-backs
#   sets/<vendor>.tar.gz                         the same, for the fetch
#   build_logs/<tier>_<binding>.log              one per build
#   SIZES.txt                                    the numbers, human readable
#
# <vendor> is NOT an argument. It is READ BACK from the first binary built,
# through the same `mojolearn_vendor()` export the selector checks, and every
# other binary must agree. A box whose binaries answer `none` (no accelerator
# target) or disagree with each other fails this script rather than producing
# a set under a label somebody typed.
#
# THIRTY BUILDS, IN PARALLEL ACROSS TIERS. Measured 2026-08-29: ten bindings
# times three tiers took about fifty minutes SERIALLY on a rented RTX 4090 and
# blew a 54-minute work bound. The three tiers write to three different
# directories and share nothing but the compiler cache, so they run as three
# background jobs (MOJOLEARN_BUILD_JOBS, default 3). Whether that fits in a
# lease on a given box is a measurement this file has not made; the per-build
# logs carry timestamps so the next leg can read it.
#
# THE BUILD SCRIPTS ARE THE EXISTING ONES. bindings/build_*.sh already know
# the tier define and the tier directory; this file runs them with the gates
# off (the gates import the whole package, which does not exist on a fresh
# box) exactly as tools/e1_bootstrap.sh phase 9 does, then MOVES the outputs
# out of python/mojolearn/ into the set directory so the checkout is left
# flat and clean for anything else the leg runs afterwards.
set -uo pipefail

DEST="${1:?dest dir}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
mkdir -p "$DEST/build_logs" "$DEST/sets"
PIXI_ENV="${MOJOLEARN_BUILD_PIXI_ENV:-gbmbench}"
JOBS="${MOJOLEARN_BUILD_JOBS:-3}"
TIERS="${MOJOLEARN_BUILD_TIERS:-fast deterministic identical}"
SCRIPTS="${MOJOLEARN_BUILD_SCRIPTS:-build.sh build_gbdt.sh build_estimators.sh build_rf.sh build_trees.sh build_svm.sh build_solver.sh build_metrics.sh build_tsa.sh build_linalg.sh}"
EXT_NAMES="_mojolearn _mojolearn_gbdt _mojolearn_estimators _mojolearn_rf _mojolearn_trees _mojolearn_svm _mojolearn_solver _mojolearn_metrics _mojolearn_tsa _mojolearn_linalg"
say() { echo "[$(date +%T) build_sets] $*"; }

say "repo $REPO, dest $DEST, tiers: $TIERS, jobs: $JOBS"
say "pixi env: $PIXI_ENV"
export PATH="$HOME/.pixi/bin:$PATH"
command -v pixi >/dev/null || { say "no pixi on PATH"; exit 2; }

# A clean slate: a stale .so from another leg's build in python/mojolearn/
# would be moved into the set as if it were this build's. The macOS release
# script refuses stale files by mtime; here they are removed first.
for t in $TIERS; do
  case "$t" in fast) d=python/mojolearn ;; *) d=python/mojolearn/$t ;; esac
  rm -f "$d"/_mojolearn*.so
done

# ---------------------------------------------------------------- builds
build_tier() {
  local tier="$1" rc=0
  for s in $SCRIPTS; do
    local log="$DEST/build_logs/${tier}_${s%.sh}.log"
    { echo "start $(date -u +%FT%TZ)"; } > "$log"
    if MOJOLEARN_NUMERIC_MODE=$tier MOJOLEARN_SKIP_BUILD_GATE=1 \
         pixi run -e "$PIXI_ENV" bash "bindings/$s" >> "$log" 2>&1; then
      echo "end $(date -u +%FT%TZ) OK" >> "$log"
    else
      echo "end $(date -u +%FT%TZ) FAILED" >> "$log"
      say "FINDING: $tier bindings/$s did not build; first error:"
      grep -m2 -E 'error:|constraint failed' "$log" | cut -c1-200 | sed 's/^/      /'
      rc=1
    fi
  done
  return $rc
}

T0=$(date +%s)
BUILD_RC=0
if [ "$JOBS" -le 1 ]; then
  for tier in $TIERS; do build_tier "$tier" || BUILD_RC=1; done
else
  # One background job per tier (three), each writing its own directory.
  pids=""
  for tier in $TIERS; do
    build_tier "$tier" &
    pids="$pids $!"
  done
  for p in $pids; do wait "$p" || BUILD_RC=1; done
fi
say "builds finished in $(( $(date +%s) - T0 ))s (rc=$BUILD_RC)"

# ---------------------------------------------------------------- read-back
# THE VENDOR COMES OUT OF THE BINARY. A bare ExtensionFileLoader import, no
# package, no GPU: `<prefix>_vendor()` is a folded constant and PyInit never
# opens a device. Every binary is asked; they must all agree.
READBACK="$DEST/readback.txt"
: > "$READBACK"
for t in $TIERS; do
  case "$t" in fast) d=python/mojolearn ;; *) d=python/mojolearn/$t ;; esac
  for n in $EXT_NAMES; do
    so="$d/$n.so"
    [ -f "$so" ] || { echo "$t $n MISSING" >> "$READBACK"; continue; }
    v=$(pixi run -e "$PIXI_ENV" python3 - "$so" "$n" <<'PY' 2>&1 | tail -1
import importlib.machinery, importlib.util, sys
so, name = sys.argv[1], sys.argv[2]
fn = "mojolearn_vendor" if name == "_mojolearn" else name[len("_mojolearn_"):] + "_vendor"
loader = importlib.machinery.ExtensionFileLoader(name, so)
spec = importlib.util.spec_from_loader(name, loader, origin=so)
m = importlib.util.module_from_spec(spec)
loader.exec_module(m)
f = getattr(m, fn, None)
print("NO-READBACK" if f is None else str(f()))
PY
)
    echo "$t $n $v" >> "$READBACK"
  done
done
say "vendor read-back per binary:"
sed 's/^/    /' "$READBACK"
VENDORS=$(awk '$3!="MISSING"{print $3}' "$READBACK" | sort -u | tr '\n' ' ')
VENDOR=$(echo "$VENDORS" | awk '{print $1}')
case "$VENDORS" in
  "cuda "|"hip "|"metal ") ;;
  *)
    say "REFUSING: the built binaries do not agree on ONE vendor: '$VENDORS'"
    say "  (a binary that answers 'none' had no accelerator target; one that"
    say "   answers NO-READBACK predates mojo_only/vendor.mojo; anything that"
    say "   looks like a Python traceback is a load failure, read it above)"
    exit 3 ;;
esac
say "vendor: $VENDOR"

# ---------------------------------------------------------------- move
SET="$DEST/sets/$VENDOR"
rm -rf "$SET"; mkdir -p "$SET/deterministic" "$SET/identical"
for t in $TIERS; do
  case "$t" in fast) src=python/mojolearn; dst="$SET" ;; *) src=python/mojolearn/$t; dst="$SET/$t" ;; esac
  for n in $EXT_NAMES; do
    [ -f "$src/$n.so" ] && mv "$src/$n.so" "$dst/$n.so"
  done
done
cp "$READBACK" "$SET/readback.txt"

# ---------------------------------------------------------------- stage
# patchelf from PyPI into a throwaway venv: the images do not ship it and
# the pixi environments cannot be re-solved from here.
TOOLS="$DEST/tools-venv"
if [ ! -x "$TOOLS/bin/patchelf" ]; then
  python3 -m venv "$TOOLS" > "$DEST/build_logs/tools_venv.log" 2>&1 \
    && "$TOOLS/bin/pip" install -q patchelf >> "$DEST/build_logs/tools_venv.log" 2>&1 \
    || { say "could not install patchelf (see build_logs/tools_venv.log)"; }
fi
PATCHELF="$TOOLS/bin/patchelf"
[ -x "$PATCHELF" ] || PATCHELF="$(command -v patchelf || true)"
[ -n "$PATCHELF" ] || { say "no patchelf; the set is built but NOT staged"; exit 4; }
ENV_LIB="$(pixi run -e "$PIXI_ENV" python3 -c 'import sys,os; print(os.path.join(sys.prefix,"lib"))' | tail -1)"
say "pixi env lib: $ENV_LIB"
python3 packaging/linux/stage_libs.py --set "$SET" --env-lib "$ENV_LIB" \
  --manifest "$SET/manifest.json" --patchelf "$PATCHELF" \
  2>&1 | tee "$DEST/build_logs/stage.log"
STAGE_RC=${PIPESTATUS[0]}

# ---------------------------------------------------------------- sizes
( cd "$DEST/sets" && tar czf "$VENDOR.tar.gz" "$VENDOR" )
{
  echo "vendor=$VENDOR"
  echo "build_rc=$BUILD_RC stage_rc=$STAGE_RC"
  echo "build_seconds=$(( $(date +%s) - T0 ))"
  for t in $TIERS; do
    case "$t" in fast) d="$SET" ;; *) d="$SET/$t" ;; esac
    echo "set_${t}_count=$(ls "$d"/_mojolearn*.so 2>/dev/null | wc -l | tr -d ' ')"
    echo "set_${t}_bytes=$(cat "$d"/_mojolearn*.so 2>/dev/null | wc -c | tr -d ' ')"
  done
  echo "runtime_libs_bytes=$(cat "$SET"/.libs/* 2>/dev/null | wc -c | tr -d ' ')"
  echo "runtime_libs=$(ls "$SET"/.libs 2>/dev/null | tr '\n' ' ')"
  echo "driver_libs_not_staged=$(python3 -c 'import json,sys; print(" ".join(json.load(open(sys.argv[1]))["driver_libs_not_staged"]))' "$SET/manifest.json" 2>/dev/null)"
  echo "set_tar_gz_bytes=$(wc -c < "$DEST/sets/$VENDOR.tar.gz" | tr -d ' ')"
} | tee "$DEST/SIZES.txt"
[ "$BUILD_RC" = 0 ] && [ "$STAGE_RC" = 0 ]
