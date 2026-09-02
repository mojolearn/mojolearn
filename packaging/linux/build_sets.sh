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
# THE TWO LISTS BELOW ARE THE LINUX WHEEL'S CONTENTS AND THEY GO STALE
# SILENTLY. A binding missing from them is not a build error -- it is a wheel
# that ships without that extension and imports fine until the user touches
# the missing surface. Fifteen bindings as of 2026-09-02: `build_mamba.sh`
# (fourteenth) and `build_transformer.sh` (fifteenth) were added here the day
# the macOS release script was found to have the same gap, one commit after
# both of those scripts turned out to be non-executable. When a binding is
# added, THREE lists move together: this one, EXT_NAMES below, and
# `packaging/macos/build_release_wheel.sh`'s pair.
SCRIPTS="${MOJOLEARN_BUILD_SCRIPTS:-build.sh build_gbdt.sh build_estimators.sh build_rf.sh build_trees.sh build_svm.sh build_solver.sh build_metrics.sh build_tsa.sh build_linalg.sh build_arima.sh build_training.sh build_gp.sh build_mamba.sh build_transformer.sh}"
EXT_NAMES="_mojolearn _mojolearn_gbdt _mojolearn_estimators _mojolearn_rf _mojolearn_trees _mojolearn_svm _mojolearn_solver _mojolearn_metrics _mojolearn_tsa _mojolearn_linalg _mojolearn_arima _mojolearn_training _mojolearn_gp _mojolearn_mamba _mojolearn_transformer"
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

# ------------------------------------------------- ARCHITECTURE READ-BACK
# WHICH GPU ARCHITECTURES ARE ACTUALLY IN THESE BINARIES.
#
# The vendor read-back above proves a binary was compiled FOR NVIDIA. It says
# nothing about WHICH NVIDIA, and on 2026-08-30 that gap shipped: a set built
# on an H100 carried `sm_90a` and nothing else, installed cleanly on an A40,
# and then failed 27 of 29 lanes with CUDA_ERROR_NO_BINARY_FOR_GPU. Nothing
# in this script noticed, because every check it had was green.
#
# The architectures are read straight out of the binary with `strings`, which
# needs no CUDA or ROCm tool and works for both vendors. An EMPTY answer is
# the loudest result of the three: it means the binary has no device code at
# all, which is exactly what `--target-accelerator` does on Metal (measured
# 2026-08-21) and what MOJOLEARN_GPU_ARCHS might therefore do here.
ARCHBACK="$DEST/arch_readback.txt"
: > "$ARCHBACK"
arch_of() {
  { strings -a "$1" 2>/dev/null || tr -c '[:print:]' '\n' < "$1"; } \
    | grep -oE '\b(sm_[0-9]+[a-z]*|compute_[0-9]+[a-z]*|gfx[0-9a-f]+)\b' \
    | sort -u | tr '\n' ',' | sed 's/,$//'
}
for t in $TIERS; do
  case "$t" in fast) d=python/mojolearn ;; *) d=python/mojolearn/$t ;; esac
  for n in $EXT_NAMES; do
    so="$d/$n.so"
    [ -f "$so" ] || { echo "$t $n MISSING" >> "$ARCHBACK"; continue; }
    a=$(arch_of "$so")
    echo "$t $n ${a:-NONE}" >> "$ARCHBACK"
  done
done
say "GPU architectures embedded, per binary:"
awk '{print $3}' "$ARCHBACK" | sort | uniq -c | sort -rn | sed 's/^/    /'
ARCH_SET=$(awk '$3!="MISSING"{print $3}' "$ARCHBACK" | sort -u | tr '\n' ' ')
if awk '$3=="NONE"{found=1} END{exit !found}' "$ARCHBACK"; then
  say "REFUSING: at least one binary names NO GPU architecture, so it carries"
  say "  no device code. If MOJOLEARN_GPU_ARCHS is set, this is the Metal"
  say "  behaviour reproducing on this vendor: passing --target-accelerator"
  say "  suppressed ahead-of-time compilation. Unset it and rebuild."
  awk '$3=="NONE"{print "    " $1 " " $2}' "$ARCHBACK" | head -5
  exit 4
fi
say "architecture set: $ARCH_SET"


# ONE ARCHITECTURE PER SET, AND TYPED MUST EQUAL BUILT. Two findings of
# 2026-08-30 meet here. (1) `--target-accelerator` takes EXACTLY ONE
# architecture -- the comma list parses and the compiler rejects it -- so a
# set is one architecture by construction and a mixed read-back means some
# builds and not others received the flag. (2) That mix actually happened:
# only bindings/build.sh read MOJOLEARN_GPU_ARCHS until 2026-08-30, so an
# sm_80-asking leg on an A40 got twenty-seven sm_86 binaries from the nine
# scripts that never saw the flag. The read-back, not the flag, names the
# set's directory.
N_ARCH=$(awk '$3!="MISSING"{print $3}' "$ARCHBACK" | sort -u | wc -l | tr -d ' ')
if [ "$N_ARCH" != 1 ]; then
  say "REFUSING: the binaries do not agree on ONE architecture: $ARCH_SET"
  say "  A set is one architecture. A mixed read-back means some builds"
  say "  received --target-accelerator and others did not; per-binary:"
  awk '{print "    " $1 " " $2 " " $3}' "$ARCHBACK" | head -8
  exit 5
fi
ARCH=$(awk '$3!="MISSING"{print $3}' "$ARCHBACK" | sort -u)
case "$ARCH" in
  *,*)
    say "REFUSING: a single binary names several architectures ($ARCH);"
    say "  no measured build has ever produced that, so this read-back is"
    say "  telling us something new. Read arch_readback.txt before trusting it."
    exit 5 ;;
esac
if [ -n "${MOJOLEARN_GPU_ARCHS:-}" ] && [ "$ARCH" != "$MOJOLEARN_GPU_ARCHS" ]; then
  say "REFUSING: MOJOLEARN_GPU_ARCHS=$MOJOLEARN_GPU_ARCHS was asked for but"
  say "  every binary carries $ARCH. A set must never ship under an"
  say "  architecture it was not verified to carry."
  exit 5
fi
say "architecture: $ARCH (requested: ${MOJOLEARN_GPU_ARCHS:-the box GPU itself})"
VENDORS=$(awk '$3!="MISSING"{print $3}' "$READBACK" | sort -u | tr '\n' ' ')
VENDOR=$(echo "$VENDORS" | awk '{print $1}')
case "$VENDORS" in
  "cuda "|"hip "|"metal ") ;;
  *)
    say "REFUSING: the built binaries do not agree on ONE vendor: '$VENDORS'"
    say "  (a binary that answers 'none' had no accelerator target; one that"
    say "   answers NO-READBACK predates checks/vendor.mojo; anything that"
    say "   looks like a Python traceback is a load failure, read it above)"
    exit 3 ;;
esac
say "vendor: $VENDOR"

# ---------------------------------------------------------------- move
# THE ARCHITECTURE IS A DIRECTORY LEVEL, named by the read-back and never
# typed: sets/<vendor>/<arch>/{,deterministic,identical}. The selector and
# pack_wheel.py mirror this layout (docs/LINUX_WHEEL.md).
SET="$DEST/sets/$VENDOR/$ARCH"
rm -rf "$DEST/sets/$VENDOR"; mkdir -p "$SET/deterministic" "$SET/identical"
for t in $TIERS; do
  case "$t" in fast) src=python/mojolearn; dst="$SET" ;; *) src=python/mojolearn/$t; dst="$SET/$t" ;; esac
  for n in $EXT_NAMES; do
    [ -f "$src/$n.so" ] && mv "$src/$n.so" "$dst/$n.so"
  done
done
cp "$READBACK" "$SET/readback.txt"
cp "$ARCHBACK" "$SET/arch_readback.txt"

# ------------------------------------------------- CPU ISA BASELINE
# RUNS AFTER THE MOVE, not before it. Placed before it, this block named
# $SET sixty lines before that variable was assigned, and `set -u` ended
# the leg one line after a clean read-back with
#     build_sets.sh: line 180: SET: unbound variable
# on the first AMD rebuild for 0.3.1. A gate's first real run is exactly
# where that class of mistake surfaces, which is an argument for running
# it rather than reading it.
# THE HOST CPU IS A TARGET TOO, and until 2026-08-30 nothing here checked it.
# `mojo build` defaults --target-cpu to the chip that ran the compiler, so the
# 0.3.0 Linux wheel shipped with AVX-512 in its host code and died with SIGILL
# on the first box whose CPU lacked it, an AMD EPYC 7773X. Every one of the
# thirty binaries carried it, unguarded: our extensions contain no `cpuid` at
# all. macOS had pinned a CPU and gated the result since 0.1.0; Linux had
# neither. The build scripts now pin x86-64-v3 and this refuses the set if
# anything above that baseline survives.
if command -v objdump >/dev/null 2>&1; then
  if ! pixi run -e "$PIXI_ENV" python3 "$REPO/packaging/linux/isa_baseline_linux.py" \
        "$SET" --json "$DEST/isa_baseline.json" > "$DEST/isa_baseline.txt" 2>&1; then
    say "REFUSING: this set is above the x86-64-v3 CPU baseline."
    grep -E "^  FAIL|first at|REFUSING|binaries," "$DEST/isa_baseline.txt" | head -12 | sed 's/^/    /'
    exit 5
  fi
  say "CPU ISA baseline: clean at x86-64-v3 ($(grep -c '^  ok' "$DEST/isa_baseline.txt") binaries)"
else
  say "FINDING: no objdump on this box, the CPU ISA baseline was NOT checked."
  echo "objdump absent; baseline NOT checked" > "$DEST/isa_baseline.txt"
fi


# ---------------------------------------------------------------- stage
# PATCHELF, FOUR WAYS, BECAUSE ONE WAY LOST A WHOLE AMD LEASE.
#
# 2026-08-30: the first end-to-end AMD leg built all thirty extensions in
# 174 s, read `hip` back from every one of them, and then died here, because
# this block had exactly one strategy and the DigitalOcean gpu-mi325x1 image
# does not ship `ensurepip`:
#
#   The virtual environment was not created successfully because ensurepip
#   is not available. On Debian/Ubuntu systems, you need to install the
#   python3-venv package
#
# Thirty good binaries were thrown away for a missing apt package. The
# routes below are tried in order and every outcome is recorded, so a
# failure names what was attempted rather than just what was missing.
TOOLS="$DEST/tools-venv"
PELOG="$DEST/build_logs/tools_venv.log"
: > "$PELOG"
PATCHELF=""

# 1. Already on the box. Costs nothing to ask first.
if command -v patchelf > /dev/null 2>&1; then
  PATCHELF="$(command -v patchelf)"
  echo "route 1: patchelf already on PATH at $PATCHELF" >> "$PELOG"
fi

# 2. The pixi environment's own interpreter. It HAS pip and needs no
#    ensurepip, which is exactly what the image was missing.
if [ -z "$PATCHELF" ]; then
  echo "route 2: pip install patchelf into the pixi env ($PIXI_ENV)" >> "$PELOG"
  if pixi run -e "$PIXI_ENV" python3 -m pip install -q patchelf >> "$PELOG" 2>&1; then
    CAND="$(pixi run -e "$PIXI_ENV" python3 -c \
      'import shutil,sys; print(shutil.which("patchelf") or "")' 2>> "$PELOG" | tail -1)"
    [ -n "$CAND" ] && [ -x "$CAND" ] && PATCHELF="$CAND"
    if [ -z "$PATCHELF" ]; then
      CAND="$(pixi run -e "$PIXI_ENV" python3 -c \
        'import os,sys; print(os.path.join(sys.prefix,"bin","patchelf"))' 2>> "$PELOG" | tail -1)"
      [ -x "$CAND" ] && PATCHELF="$CAND"
    fi
  fi
  [ -n "$PATCHELF" ] && echo "route 2 gave $PATCHELF" >> "$PELOG"
fi

# 3. A throwaway venv, the original route. Works wherever ensurepip exists.
if [ -z "$PATCHELF" ]; then
  echo "route 3: throwaway venv at $TOOLS" >> "$PELOG"
  if python3 -m venv "$TOOLS" >> "$PELOG" 2>&1 \
     && "$TOOLS/bin/pip" install -q patchelf >> "$PELOG" 2>&1; then
    [ -x "$TOOLS/bin/patchelf" ] && PATCHELF="$TOOLS/bin/patchelf"
  fi
  [ -n "$PATCHELF" ] && echo "route 3 gave $PATCHELF" >> "$PELOG"
fi

# 4. The distribution's own package. Last because it is the least pinned,
#    and it is the one that would have saved the 2026-08-30 lease.
if [ -z "$PATCHELF" ] && command -v apt-get > /dev/null 2>&1; then
  echo "route 4: apt-get install patchelf" >> "$PELOG"
  ( apt-get update -qq && apt-get install -y -qq patchelf ) >> "$PELOG" 2>&1
  command -v patchelf > /dev/null 2>&1 && PATCHELF="$(command -v patchelf)"
  [ -n "$PATCHELF" ] && echo "route 4 gave $PATCHELF" >> "$PELOG"
fi

if [ -z "$PATCHELF" ]; then
  say "NO PATCHELF. All four routes failed; the thirty binaries ARE built"
  say "  and readback.txt is valid, but nothing is staged. Routes tried:"
  say "  1 PATH, 2 pixi env pip, 3 throwaway venv, 4 apt-get."
  say "  Full transcript: build_logs/$(basename "$PELOG")"
  exit 4
fi
say "patchelf: $PATCHELF"
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
  echo "arch=$ARCH"
  echo "gpu_archs_requested=${MOJOLEARN_GPU_ARCHS:-}"
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
