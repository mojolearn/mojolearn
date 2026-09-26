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
# MOJOLEARN_BUILD_JOBS extension builds run at a time (default 4; DEVIATION
# 2501). Concurrent compilers once multiplied worker counts and memory
# pressure, so the caps are per build, not per box: every build keeps two
# compiler workers and one BLAS/OpenMP thread, and the affinity is 2 x jobs
# CPUs out of the inherited set (never widened beyond it). Each build writes
# its own log; a failure names the script and its first error as before.
# A partial timed-out build is retained as partial, never release-qualified.
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
JOBS="${MOJOLEARN_BUILD_JOBS:-4}"
[[ "$JOBS" =~ ^[1-9][0-9]?$ && "$JOBS" -le 16 ]] || { echo 'MOJOLEARN_BUILD_JOBS must be 1..16' >&2; exit 2; }
BUILD_CORES=$((2 * JOBS))
export MOJOLEARN_COMPILE_JOBS=2 MAX_JOBS=2 CMAKE_BUILD_PARALLEL_LEVEL=2
# AN AMD GPU BINDING COMPILES WITH ONE WORKER, AND IS STILL NOT REPRODUCIBLE
# (2026-09-22, lane/release-cpu-build-box). The Mojo compiler's gfx942 output
# varies from run to run with a cold cache: a few register numbers or two
# swapped instructions inside some embedded AMDGPU code objects
# (gemm_identical, holtwinters). Measured at d181d9792 on RunPod CPU pods,
# the Mojo cache wiped before every build:
#   build_mixture.sh -j 2: 3 distinct binaries in 5 builds; -j 1: 1 in 5
#   build_tsa.sh     -j 2: 2 in 4;  -j 1: 2 in 12 (8 of one, 4 of the other)
# The variance is inside one compiler process (idle or loaded box alike), so
# -j 1 narrows it and does not remove it; disabling ASLR to test the usual
# cause (pointer-ordered containers) is refused inside a RunPod container.
# A warm Mojo cache replays the first compile, which is why every single box
# always looked reproducible. What makes a released AMD binary reproducible is
# the binding cache: once built, the same key serves the same bytes to every
# later build (tools/bincache.py). NVIDIA sets and the host bindings (CPU
# codegen) keep two workers: every pair of cold builds compared matched (the
# H100 and L40S legs against the CPU box at d181d9792, two cold CPU boxes at
# 4756f57a9: 264 of 264 NVIDIA binaries, the host bindings on every box). The
# host bindings in an AMD set also keep two, so they stay byte-identical to
# the NVIDIA sets' copies (pack_wheel.py compares them).
GPU_COMPILE_JOBS=2
case "${MOJOLEARN_GPU_ARCHS:-}" in gfx*) GPU_COMPILE_JOBS=1 ;; esac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
if [[ "$(uname -s)" != Linux ]]; then
    echo 'Linux vendor builds must run on the remote Linux GPU host' >&2
    exit 2
fi
command -v taskset >/dev/null || { echo 'taskset required for CPU cap' >&2; exit 2; }
BUILD_CPUS=$(python3 -c 'import os, sys; print(",".join(map(str, sorted(os.sched_getaffinity(0))[:int(sys.argv[1])])))' "$BUILD_CORES") || exit 2
[[ -n "$BUILD_CPUS" ]] || { echo 'Empty CPU affinity' >&2; exit 2; }
taskset -pc "$BUILD_CPUS" $$ || exit 2
# ONE BINDING, BOUNDED (packaging/linux/binding_timeout.sh, 2026-09-25): a
# build past RELEASE_BINDING_TIMEOUT_SECONDS (default 1200) is stopped and
# named as a FINDING while the rest of the pool carries on.
. "$REPO/packaging/linux/binding_timeout.sh"
BINDING_TIMEOUT=$(binding_timeout_seconds) || exit 2
python3 -c 'import os, sys; assert 1 <= len(os.sched_getaffinity(0)) <= int(sys.argv[1]), "build affinity exceeds 2 x jobs"' "$BUILD_CORES" || exit 2
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
SCRIPTS="${MOJOLEARN_BUILD_SCRIPTS:-build_gbdt.sh build_rf.sh build_trees.sh}"
EXT_NAMES="_mojolearn_gbdt _mojolearn_rf _mojolearn_trees"

# ONE TIER RULE (DEVIATION 2490, 2026-09-10): the three TREE lanes above
# build in every tier TIERS names. EVERY OTHER BINDING is identical only and
# lives in the two lists below, built and gated for the identical tier alone
# the way the byte LM always was. Cross-vendor bitwise identity is the
# product; a fast tier ships only where it has a measured win over the
# opponent's own CPU, and outside trees it has none (the reasoning and the
# M4 numbers are on `_TIERED` in python/mojolearn/_backend.py). The
# identical-only build scripts exit 2 on any other MOJOLEARN_NUMERIC_MODE.
# Before this the split was neural-only (three lanes, 2026-09-10 morning);
# the reason for THOSE was different (their fused kernels were gated on the
# identical contract, so the lower tiers were slower) and no longer matters.
# CLASSICAL ML (2026-09-25): fast and identical, never deterministic.
FAST_CLASSICAL_SCRIPTS="build.sh build_estimators.sh build_svm.sh build_solver.sh build_metrics.sh build_preprocessing.sh build_tsa.sh build_linalg.sh build_arima.sh build_gp.sh build_kernel_methods.sh build_mixture.sh build_hdbscan.sh build_resample.sh build_ivf.sh"
FAST_CLASSICAL_NAMES="_mojolearn _mojolearn_estimators _mojolearn_svm _mojolearn_solver _mojolearn_metrics _mojolearn_preprocessing _mojolearn_tsa _mojolearn_linalg _mojolearn_arima _mojolearn_gp _mojolearn_kernel_methods _mojolearn_mixture _mojolearn_hdbscan _mojolearn_resample _mojolearn_ivf"
IDENTICAL_ONLY_SCRIPTS="build_training.sh build_mamba.sh build_transformer.sh build_embedding.sh"
IDENTICAL_ONLY_NAMES="_mojolearn_training _mojolearn_mamba _mojolearn_transformer _mojolearn_embedding"
PACKAGE_BYTE_LM=${MOJOLEARN_PACKAGE_BYTE_LM:-0}
case "$PACKAGE_BYTE_LM" in 0|1) ;; *) echo 'MOJOLEARN_PACKAGE_BYTE_LM must be 0 or 1' >&2; exit 2 ;; esac
unset MOJOLEARN_BYTE_LM_OUTDIR
tier_names() {
  printf '%s' "$EXT_NAMES"
  if [[ "$1" = identical ]]; then printf ' %s' "$IDENTICAL_ONLY_NAMES"; fi
  if [[ "$1" = identical || "$1" = fast ]]; then printf ' %s' "$FAST_CLASSICAL_NAMES"; fi
  if [[ "$PACKAGE_BYTE_LM" = 1 && "$1" = identical ]]; then printf ' _mojolearn_byte_lm'; fi
  printf '\n'
}
tier_scripts() {
  printf '%s' "$SCRIPTS"
  if [[ "$1" = identical ]]; then printf ' %s' "$IDENTICAL_ONLY_SCRIPTS"; fi
  if [[ "$1" = identical || "$1" = fast ]]; then printf ' %s' "$FAST_CLASSICAL_SCRIPTS"; fi
  if [[ "$PACKAGE_BYTE_LM" = 1 && "$1" = identical ]]; then printf ' build_byte_lm.sh'; fi
  # THE HOST (CPU) BINDINGS ARE BUILT HERE, in the identical tier's pass,
  # because identical is the only tier they support. They appear in
  # tier_SCRIPTS and deliberately NOT in tier_NAMES: tier_names drives the
  # per-tier read-back loop and the staging move, and these binaries are
  # neither tier members nor vendor members -- each answers 'cpu', carries
  # no GPU code, and is staged once beside the tiers in <set>/host/.
  #
  # Omitting this line is what failed the first 0.8.4 gfx942 leg: everything
  # downstream (read-back, arch read-back, staging, the build-provenance
  # host_extension accounting) was in place around a build that never ran, so
  # the leg reached its own assertion with nothing to count and refused.
  if [[ "$PACKAGE_BYTE_LM" = 1 && "$1" = identical ]]; then
    for f in $HOST_FAMILIES; do printf ' build_%s_host.sh' "$f"; done
  fi
  printf '\n'
}
# THE HOST BINDINGS ARE NOT TIER MEMBERS AND NOT VENDOR MEMBERS. Each has
# no GPU code, answers 'cpu' when asked its vendor, and the runtime loads it
# from <package>/host/ (by path, or through _backend.load_host_module) rather
# than through _backend.binding(), so they belong in neither tier_names nor
# the per-vendor directories. They are carried once, beside them, and every
# check below that assumes "vendor binary in a tier" is given an explicit
# exception rather than being loosened for everything.
#
# WHICH FAMILIES is not written here. Since 0.8.6 (the packaging lane,
# 2026-09-14) every host family the manifest declares ships in the wheel, and
# the list is READ from python/mojolearn/host_surface.py, the one declaration
# of the CPU surface; packaging/check_ext_lists.py fails this file if it ever
# carries a host list of its own. Until 0.8.5 the byte LM's was the only one.
HOST_FAMILIES=$(python3 python/mojolearn/host_surface.py --wheel-families) || exit 2
HOST_NAMES=$(python3 python/mojolearn/host_surface.py --wheel-bindings) || exit 2
[[ -n "$HOST_FAMILIES" && -n "$HOST_NAMES" ]] || { echo 'the manifest names no wheel host family' >&2; exit 2; }
host_so() { printf 'python/mojolearn/host/%s.so' "$1"; }
say() { echo "[$(date +%T) build_sets] $*"; }

say "repo $REPO, dest $DEST, tiers: $TIERS, jobs: $JOBS, gpu compile workers: $GPU_COMPILE_JOBS, per-binding bound: ${BINDING_TIMEOUT}s"
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
# The host bindings sit BESIDE the tiers, so the loop above never reaches
# them, and bindings/build_host_family.sh REFUSES to overwrite an existing
# output rather than silently replacing it. A leftover from an earlier leg
# would therefore fail this build instead of being reused.
for n in $HOST_NAMES; do rm -f "$(host_so "$n")"; done

# ---------------------------------------------------------------- the cache
# THE BINDING CACHE (tools/bincache.py, 2026-09-22). ON only where a runner
# staged a URL map on this box (tools/release_linux_build.sh through
# tools/runpod_cpu_leg.sh --release-bincache) and MOJOLEARN_BINCACHE is not 0;
# everywhere else, the GPU legs included, every build below is the plain
# `pixi run bash bindings/<script>` it always was. Each build DECLARES its one
# output (several builds share this tree at once), a hit is placed only after
# the archive's key, fields and every file's sha256 verify, and the host
# bindings built for one set are served to the next set from a box-local hot
# directory. $DEST/bincache/ records every build's outcome and key, and every
# hit's origin commit; build-provenance.json carries it per binary.
BINCACHE_MAP=${MOJOLEARN_BINCACHE_MAP:-/root/.mojolearn_bincache/urls.tsv}
USE_BINCACHE=0
if [[ "${MOJOLEARN_BINCACHE:-}" != 0 && -f "$BINCACHE_MAP" ]]; then USE_BINCACHE=1; fi
say "binding cache: $([[ $USE_BINCACHE = 1 ]] && echo "ON ($BINCACHE_MAP)" || echo off)"
declared_output() {   # tier script -> the one .so it writes, repo-relative
  local tier="$1" s="$2" name dir
  if [[ "$s" = build_*_host.sh ]]; then
    name="${s#build_}"; name="${name%.sh}"
    printf 'python/mojolearn/host/_mojolearn_%s.so' "$name"; return
  fi
  if [[ "$s" = build.sh ]]; then name=_mojolearn; else name="${s#build_}"; name="_mojolearn_${name%.sh}"; fi
  case "$tier" in fast) dir=python/mojolearn ;; *) dir=python/mojolearn/$tier ;; esac
  printf '%s/%s.so' "$dir" "$name"
}
run_binding() {   # tier script env-prefix...: the build, through the cache when it is on
  local tier="$1" s="$2"
  shift 2
  if [[ "$USE_BINCACHE" = 1 ]]; then
    "$@" MOJOLEARN_BINCACHE_MAP="$BINCACHE_MAP" MOJOLEARN_BINCACHE_OUT="$DEST/bincache" \
      MOJOLEARN_BINCACHE_HOT_DIR="${MOJOLEARN_BINCACHE_HOT_DIR:-/root/.mojolearn_bincache/hot}" \
      MOJOLEARN_BINCACHE_SHELL=bash MOJOLEARN_BINCACHE_OUTPUTS="$(declared_output "$tier" "$s")" \
      pixi run -e "$PIXI_ENV" python3 tools/bincache.py build "bindings/$s"
  else
    "$@" pixi run -e "$PIXI_ENV" bash "bindings/$s"
  fi
}

# ---------------------------------------------------------------- builds
build_one() {
  local tier="$1" s="$2"
  local log="$DEST/build_logs/${tier}_${s%.sh}.log"
  local rc=0
  { echo "start $(date -u +%FT%TZ)"; } > "$log"
  if [[ "$s" = build_*_host.sh ]]; then
    # A HOST BUILD MUST NOT SEE AN ACCELERATOR TARGET. Every release leg
    # exports MOJOLEARN_GPU_ARCHS (gfx942, sm_89, sm_90a) and a host binding
    # has no device code, so bindings/build_host_family.sh refuses that
    # variable BY NAME on Linux. Its output directories (the family's own and
    # the shared MOJOLEARN_HOST_OUTDIR) are unset for the same reason the
    # byte LM build unsets its own, so the binary lands at
    # python/mojolearn/host/ where the staging move below looks for it.
    # It compiles the CPU column (COLUMN_CPU, the CPU training lane
    # 2026-09-13) and refuses any other MOJOLEARN_TARGET_COLUMN by name, while
    # tools/release061_remote_build.sh exports the leg's GPU column to every
    # build; the 0.8.5 H100 leg failed here with "MOJOLEARN_TARGET_COLUMN=
    # nvidia is refused", so the column is pinned to cpu for every host build.
    local fam="${s#build_}"; fam="${fam%_host.sh}"
    local FAM; FAM=$(printf '%s' "$fam" | tr 'a-z' 'A-Z')
    MOJOLEARN_NUMERIC_MODE=$tier MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_COMPILE_JOBS=2 \
      with_binding_timeout "$BINDING_TIMEOUT" "$log" \
      run_binding "$tier" "$s" env -u MOJOLEARN_GPU_ARCHS -u MOJOLEARN_HOST_OUTDIR -u "MOJOLEARN_${FAM}_HOST_OUTDIR" \
      || rc=$?
  else
    MOJOLEARN_NUMERIC_MODE=$tier MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=$GPU_COMPILE_JOBS \
      with_binding_timeout "$BINDING_TIMEOUT" "$log" run_binding "$tier" "$s" env || rc=$?
  fi
  if [[ "$rc" = 0 ]]; then
    echo "end $(date -u +%FT%TZ) OK" >> "$log"
    say "built $tier bindings/$s"
  else
    echo "end $(date -u +%FT%TZ) FAILED" >> "$log"
    if [[ "$rc" = 124 ]]; then
      say "FINDING: $tier bindings/$s TIMED OUT after ${BINDING_TIMEOUT}s (RELEASE_BINDING_TIMEOUT_SECONDS); log $log"
      return 1
    fi
    say "FINDING: $tier bindings/$s did not build; first error:"
    grep -m2 -E 'error:|constraint failed' "$log" | cut -c1-200 | sed 's/^/      /'
    return 1
  fi
}

# The pool: every (tier, script) pair is one background build; at most JOBS
# run at once. Each build has its own output directory (python/mojolearn or
# python/mojolearn/<tier>) and its own mktemp scratch, so the pairs are
# independent. The identical tier is queued first because it holds the most
# scripts; the tail of the schedule is then the shortest.
T0=$(date +%s)
BUILD_RC=0
ordered_tiers=""
for tier in $TIERS; do [[ "$tier" = identical ]] && ordered_tiers="identical"; done
for tier in $TIERS; do [[ "$tier" = identical ]] || ordered_tiers="$ordered_tiers $tier"; done
running=0
for tier in $ordered_tiers; do
  for s in $(tier_scripts "$tier"); do
    build_one "$tier" "$s" &
    running=$((running + 1))
    if (( running >= JOBS )); then
      wait -n || BUILD_RC=1
      running=$((running - 1))
    fi
  done
done
while (( running > 0 )); do
  wait -n || BUILD_RC=1
  running=$((running - 1))
done
say "builds finished in $(( $(date +%s) - T0 ))s (rc=$BUILD_RC, jobs=$JOBS, cpus=$BUILD_CPUS)"

# ---------------------------------------------------------------- read-back
# THE VENDOR COMES OUT OF THE BINARY. A bare ExtensionFileLoader import, no
# package, no GPU: `<prefix>_vendor()` is a folded constant and PyInit never
# opens a device. Every binary is asked; they must all agree.
READBACK="$DEST/readback.txt"
: > "$READBACK"
for t in $TIERS; do
  case "$t" in fast) d=python/mojolearn ;; *) d=python/mojolearn/$t ;; esac
  for n in $(tier_names "$t"); do
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
if name == '_mojolearn_byte_lm':
    assert m.byte_lm_numeric_mode() == 1, 'Byte LM must be IDENTICAL'
    assert m.byte_lm_profile() == 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1', 'Wrong byte LM profile'
print("NO-READBACK" if f is None else str(f()))
PY
)
    echo "$t $n $v" >> "$READBACK"
  done
done
# Every host binding is asked the same question by the same mechanism, and
# each must answer 'cpu'. A GPU vendor here would mean the CPU-only build
# picked up an accelerator target, which is the one way such a binary could
# be wrong in a way that still loads. The read-back trio is the manifest's
# (`<prefix>_numeric_mode` 1, `<prefix>_vendor` cpu, `<prefix>_column` cpu);
# the column is the build-time witness that the binding compiled as the
# kernel matrix's CPU column and never as the leg's GPU column.
if [[ "$PACKAGE_BYTE_LM" = 1 ]]; then
  for n in $HOST_NAMES; do
    so=$(host_so "$n")
    if [[ -f "$so" ]]; then
      hv=$(pixi run -e "$PIXI_ENV" python3 - "$so" "$n" <<'PY' 2>&1 | tail -1
import importlib.machinery, importlib.util, sys
so, name = sys.argv[1], sys.argv[2]
prefix = name[len("_mojolearn_"):]
loader = importlib.machinery.ExtensionFileLoader(name, so)
spec = importlib.util.spec_from_loader(name, loader, origin=so)
m = importlib.util.module_from_spec(spec)
loader.exec_module(m)
assert getattr(m, prefix + "_numeric_mode")() == 1, name + ' must be IDENTICAL'
assert str(getattr(m, prefix + "_column")()) == 'cpu', name + ' must compile as the CPU column'
print(str(getattr(m, prefix + "_vendor")()))
PY
)
      echo "host $n $hv" >> "$READBACK"
    else
      echo "host $n MISSING" >> "$READBACK"
    fi
  done
fi
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
  for n in $(tier_names "$t"); do
    so="$d/$n.so"
    [ -f "$so" ] || { echo "$t $n MISSING" >> "$ARCHBACK"; continue; }
    a=$(arch_of "$so")
    echo "$t $n ${a:-NONE}" >> "$ARCHBACK"
  done
done
# NONE IS THE CORRECT ANSWER FOR A HOST BINDING, and the only one. Every
# check below reads $3, and an empty architecture is refused there by design
# because for a GPU binary it means the device code was suppressed. A host
# binary has no device code to suppress, so it is recorded with a value that
# says so in words and is excluded from the architecture agreement checks by
# name rather than by being allowed to look like a GPU set.
if [[ "$PACKAGE_BYTE_LM" = 1 ]]; then
  for n in $HOST_NAMES; do
    so=$(host_so "$n")
    if [[ -f "$so" ]]; then
      ha=$(arch_of "$so")
      if [[ -n "$ha" ]]; then
        say "REFUSING: the host binding $n names GPU architectures ($ha)."
        say "  It is built with no accelerator target, so device code in it means"
        say "  MOJOLEARN_GPU_ARCHS reached a build that must never see it."
        exit 4
      fi
      echo "host $n NONE-BY-DESIGN" >> "$ARCHBACK"
    else
      echo "host $n MISSING" >> "$ARCHBACK"
    fi
  done
fi
say "GPU architectures embedded, per binary:"
awk '{print $3}' "$ARCHBACK" | sort | uniq -c | sort -rn | sed 's/^/    /'
ARCH_SET=$(awk '$1!="host" && $3!="MISSING"{print $3}' "$ARCHBACK" | sort -u | tr '\n' ' ')
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
N_ARCH=$(awk '$1!="host" && $3!="MISSING"{print $3}' "$ARCHBACK" | sort -u | wc -l | tr -d ' ')
if [ "$N_ARCH" != 1 ]; then
  say "REFUSING: the binaries do not agree on ONE architecture: $ARCH_SET"
  say "  A set is one architecture. A mixed read-back means some builds"
  say "  received --target-accelerator and others did not; per-binary:"
  awk '{print "    " $1 " " $2 " " $3}' "$ARCHBACK" | head -8
  exit 5
fi
ARCH=$(awk '$1!="host" && $3!="MISSING"{print $3}' "$ARCHBACK" | sort -u)
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
# THE HOST ROW IS EXCLUDED HERE TOO, and for the same reason as the
# architecture checks above: it answers 'cpu', which is not a GPU API, so a
# set carrying it would read as two vendors and be refused as a mixed build.
# Every check that reduces a read-back file to one value must skip $1=="host";
# the ones that merely PRINT the file (the per-binary summaries) must not, or
# the evidence stops showing the binary it is evidence for.
VENDORS=$(awk '$1!="host" && $3!="MISSING"{print $3}' "$READBACK" | sort -u | tr '\n' ' ')
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
  for n in $(tier_names "$t"); do
    [ -f "$src/$n.so" ] && mv "$src/$n.so" "$dst/$n.so"
  done
done
# host/ sits beside the tiers, not inside one, mirroring where the runtime
# looks for them in an installed package.
if [[ "$PACKAGE_BYTE_LM" = 1 ]]; then
  mkdir -p "$SET/host"
  for n in $HOST_NAMES; do
    [[ -f "$(host_so "$n")" ]] && mv "$(host_so "$n")" "$SET/host/$n.so"
  done
fi
cp "$READBACK" "$SET/readback.txt"
cp "$ARCHBACK" "$SET/arch_readback.txt"

# ------------------------------------------- IDENTICAL PTX, ROUNDING PINNED
# The CUDA sets ship PTX only and the driver JIT compiles it with fmad on, so
# a plain `mul.f32` feeding a plain `add.f32` may become one FFMA on the
# user's box. packaging/linux/ptx_contract.py gives every plain float
# mul/add/sub in the IDENTICAL set its `.rn` spelling, in place and
# length-preserving, which ptxas never contracts; the wheel audit
# (packaging/portable_math/wheel.py) refuses a set that still carries one.
# HERE, before the manifest hashes the set, so every recorded sha256 (the
# manifest, LINUX_PAYLOAD.json, release_reuse) is of the shipped bytes.
if [[ "$VENDOR" = cuda ]] && ls "$SET"/identical/*.so > /dev/null 2>&1; then
  python3 "$REPO/packaging/linux/ptx_contract.py" patch "$SET"/identical/*.so > "$DEST/ptx_rn.jsonl" \
    || { say "REFUSING: the IDENTICAL PTX .rn pass failed (read $DEST/ptx_rn.jsonl)"; exit 5; }
  if ! python3 "$REPO/packaging/linux/ptx_contract.py" audit "$SET"/identical/*.so > "$DEST/ptx_audit.jsonl"; then
    say "REFUSING: IDENTICAL PTX still carries a float op without a rounding modifier:"
    grep -v '"plain_float_ops": 0' "$DEST/ptx_audit.jsonl" | head -5 | cut -c1-240 | sed 's/^/    /'
    exit 5
  fi
  say "IDENTICAL PTX: $(python3 -c 'import json,sys; print(sum(json.loads(l)["rewritten"] for l in open(sys.argv[1])))' "$DEST/ptx_rn.jsonl") float ops given .rn"

  # ----------------------------------- IDENTICAL MACHINE CODE, NO DRIVER JIT
  # The .rn PTX above is still compiled by the USER's driver (the Mojo runtime
  # hands the embedded bytes to cuModuleLoadDataEx). packaging/linux/cubin_contract.py
  # compiles every module HERE with a pinned ptxas, contraction off, and
  # writes a compressed fatbin over the PTX in place (or, for a tiny
  # module whose fatbin is longer than its PTX, into freed padding with its
  # RIP-relative lea references repointed; objdump cross-checks each); the
  # driver then only loads it.
  # THE TOOLKIT IS 12.5, ON PURPOSE: the oldest ptxas that accepts our PTX
  # (sm_90a is PTX ISA 8.5). A CUDA 13 cubin (ELF ABI 8) and a zstd fatbin load
  # only on a 580+ driver; 12.5 cubins with LZ4 compression load on the same
  # drivers the PTX wheel does (measured on driver 570, 2026-09-26,
  # docs/lanes/NVIDIA_CUBIN_RESUME.md), so the driver floor does not rise.
  # ptxas and fatbinary come from NVIDIA's cuda_nvcc redist archive, pinned by
  # sha256 (the pip wheel nvidia-cuda-nvcc-cu12 carries no fatbinary).
  CUDA_TOOLS_VERSION=12.5.82
  CUDA_TOOLS_SHA256=ded05fe3c8d075c6c1bf892005d3c50bde3eceaa049b879fcdff6158e068e3be
  CUDA_TOOLS_URL="https://developer.download.nvidia.com/compute/cuda/redist/cuda_nvcc/linux-x86_64/cuda_nvcc-linux-x86_64-$CUDA_TOOLS_VERSION-archive.tar.xz"
  CUDA_TOOLS="${MOJOLEARN_CUDA_TOOLS_DIR:-$DEST/cuda-tools-$CUDA_TOOLS_VERSION}"
  if [[ ! -x "$CUDA_TOOLS/bin/ptxas" || ! -x "$CUDA_TOOLS/bin/fatbinary" ]]; then
    mkdir -p "$CUDA_TOOLS"
    curl -fsSL --retry 3 -o "$CUDA_TOOLS/nvcc.tar.xz" "$CUDA_TOOLS_URL" \
      || { say "REFUSING: could not download the pinned CUDA tools ($CUDA_TOOLS_URL)"; exit 5; }
    echo "$CUDA_TOOLS_SHA256  $CUDA_TOOLS/nvcc.tar.xz" | sha256sum -c --quiet \
      || { say "REFUSING: $CUDA_TOOLS_URL does not have the pinned sha256"; exit 5; }
    tar -xJf "$CUDA_TOOLS/nvcc.tar.xz" -C "$CUDA_TOOLS" --strip-components=1 --wildcards '*/bin/ptxas' '*/bin/fatbinary' \
      || { say "REFUSING: the CUDA tools archive has no bin/ptxas + bin/fatbinary"; exit 5; }
    rm -f "$CUDA_TOOLS/nvcc.tar.xz"
  fi
  PTXAS="$CUDA_TOOLS/bin/ptxas"; FATBINARY="$CUDA_TOOLS/bin/fatbinary"
  if ! "$PTXAS" --version 2>/dev/null | grep -q "V$CUDA_TOOLS_VERSION"; then
    say "REFUSING: $PTXAS is not ptxas V$CUDA_TOOLS_VERSION"; exit 5
  fi
  if ! python3 "$REPO/packaging/linux/cubin_contract.py" patch --arch "$ARCH" --ptxas "$PTXAS" --fatbinary "$FATBINARY" \
      "$SET"/identical/*.so > "$DEST/cubin.jsonl"; then
    say "REFUSING: an IDENTICAL module could not be given its fatbin (\"unplaced\" in $DEST/cubin.jsonl)"
    exit 5
  fi
  if ! python3 "$REPO/packaging/linux/cubin_contract.py" audit --arch "$ARCH" "$SET"/identical/*.so > "$DEST/cubin_audit.jsonl"; then
    say "REFUSING: IDENTICAL CUDA binaries the driver would still JIT:"
    grep '"errors": \[".' "$DEST/cubin_audit.jsonl" | head -5 | cut -c1-240 | sed 's/^/    /'
    exit 5
  fi
  say "IDENTICAL machine code: $(python3 -c 'import json,sys; r=[json.loads(l) for l in open(sys.argv[1])][1:]; print(sum(x["in_place"] for x in r), "modules fatbin in place,", sum(x["moved"] for x in r), "moved (lea repointed), ptxas --fmad=false")' "$DEST/cubin.jsonl")"
fi

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
pixi run -e pkg python packaging/linux/stage_libs.py --set "$SET" --env-lib "$ENV_LIB" \
  --manifest "$SET/manifest.json" --patchelf "$PATCHELF" \
  2>&1 | tee "$DEST/build_logs/stage.log"
STAGE_RC=${PIPESTATUS[0]}
if [[ "$STAGE_RC" = 0 && "$PACKAGE_BYTE_LM" = 1 && -f "$SET/identical/_mojolearn_byte_lm.so" ]]; then
  python3 - "$SET/manifest.json" <<'PYBYTE'
import json, pathlib, sys
path = pathlib.Path(sys.argv[1])
record = json.loads(path.read_text())
record.setdefault('optional_native', {})['_mojolearn_byte_lm'] = dict(
    included=True, supported_modes=['identical'], unsupported_modes=['fast', 'deterministic'],
    profile='mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1')
path.write_text(json.dumps(record, indent=2) + '\n')
PYBYTE
  [[ $? = 0 ]] || STAGE_RC=1
fi

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
