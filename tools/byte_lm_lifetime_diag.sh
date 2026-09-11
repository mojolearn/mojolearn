#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# DEVIATION 2494: Byte-LM trainer lifetime diagnostic, as a gemm-leg EXTRA.
#
# Shipped by tools/gemm_remote_leg.sh (MOJOLEARN_GEMM_LEG_EXTRA) and run on
# the rented box AFTER the device check and the card, from /root/mojolearn
# with pixi on PATH. POSIX sh only: RunPod images link /bin/sh to dash.
#
#   1. builds the base binding (host helpers), the trees binding (the
#      DEVIATION 2513 ExtraTrees control) and the byte LM binding, all
#      IDENTICAL, for this box's GPU,
#   2. runs tools/byte_lm_lifetime_diag.py (each case in its own
#      subprocess with a per-case deadline, stacks retained on timeout);
#      17 cases since DEVIATION 2513 (3 controls through other bindings,
#      2 MOJOLEARN_BYTE_LM_KEEP_CONTEXT=1 variants), worst case 17 x 120 s,
#   3. leaves everything under /root/gemm_leg_out/byte-lm-lifetime/, which
#      the leg fetches home as remote/byte-lm-lifetime/.
#
# Knobs (all optional):
#   MOJOLEARN_GPU_ARCHS            sm_NN; otherwise read from nvidia-smi
#   MOJOLEARN_TARGET_COLUMN        passed to the build script when set
#   BYTE_LM_LIFETIME_DEADLINE      seconds per case (default 120)
#   BYTE_LM_LIFETIME_CASES         comma-separated subset (default: all)
#   BYTE_LM_LIFETIME_OUT           output directory (default below)
#
# No claim follows from this file; it records what the box did.
set -u
ROOT=${MOJOLEARN_ROOT:-/root/mojolearn}
OUT=${BYTE_LM_LIFETIME_OUT:-/root/gemm_leg_out/byte-lm-lifetime}
DEADLINE=${BYTE_LM_LIFETIME_DEADLINE:-120}
cd "$ROOT" || { echo "byte-lm-lifetime: no $ROOT" >&2; exit 9; }
mkdir -p "$OUT" || exit 9
say() { printf '%s %s\n' "$(date -u +%H:%M:%S)" "$*"; }
started=$(date +%s)
say "byte-lm-lifetime: start, root=$ROOT out=$OUT deadline=${DEADLINE}s"

# ---------------------------------------------------------------- environment
if ! command -v pixi > /dev/null 2>&1; then
    PATH="$HOME/.pixi/bin:$PATH"
    export PATH
fi
command -v pixi > "$OUT/pixi_which.txt" 2>&1 || say "byte-lm-lifetime: NO PIXI on PATH"
uname -a > "$OUT/uname.txt" 2>&1
if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi > "$OUT/nvidia-smi.txt" 2>&1
    nvidia-smi --query-gpu=name,driver_version,compute_cap,memory.total --format=csv,noheader \
        > "$OUT/gpu.csv" 2>&1
fi
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
pixi run python3 -c 'import sys; print(sys.version); print(sys.executable)' > "$OUT/python_version.txt" 2>&1

# GPU architecture: explicit wins; otherwise the compute capability of the
# first NVIDIA device. sm_89 (L40S, RTX 4090), sm_90a (H100: the only
# spelling the compiler produces there, DEVIATION 2293), sm_86, sm_80.
ARCH=${MOJOLEARN_GPU_ARCHS:-}
if [ -z "$ARCH" ]; then
    cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ')
    case "$cap" in
        9.0) ARCH=sm_90a ;;
        [0-9].[0-9]) ARCH="sm_$(printf '%s' "$cap" | tr -d '.')" ;;
        *) say "byte-lm-lifetime: cannot read compute capability ('$cap'); set MOJOLEARN_GPU_ARCHS"; echo "arch_exit=9" >> "$OUT/status.txt"; exit 9 ;;
    esac
fi
say "byte-lm-lifetime: GPU arch $ARCH"
{
  echo "root=$ROOT"
  echo "arch=$ARCH"
  echo "target_column=${MOJOLEARN_TARGET_COLUMN:-}"
  echo "deadline_s=$DEADLINE"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
} > "$OUT/status.txt"

# ---------------------------------------------------------------- build
# bindings/build_byte_lm.sh refuses an existing destination. A leg archive
# carries no built .so, but a reused box might; move any prior file aside
# and record its hash so the run cannot silently test an older binary.
DEST_DIR="$ROOT/python/mojolearn/identical"
DEST="$DEST_DIR/_mojolearn_byte_lm.so"
mkdir -p "$DEST_DIR"
if [ -e "$DEST" ] || [ -L "$DEST" ]; then
    mkdir -p "$OUT/prior_binding"
    sha256sum "$DEST" > "$OUT/prior_binding/sha256.txt" 2>&1 || true
    mv "$DEST" "$OUT/prior_binding/_mojolearn_byte_lm.so.prior"
    say "byte-lm-lifetime: moved a prior binding aside"
fi
build_started=$(date +%s)
# The NumPy-free Python layer resolves its host helpers (all_finite_f32 and
# the buffer converters) from the IDENTICAL base binding, so that binding
# is built first; without it every case fails at input validation in 0.3 s
# (first L40S run, 2026-09-11 03:06Z).
rm -f python/mojolearn/identical/_mojolearn.so
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS="$ARCH" MOJOLEARN_SKIP_BUILD_GATE=1 \
    timeout -k 30 1500 sh bindings/build.sh > "$OUT/build_base.log" 2>&1
base_rc=$?
echo "build_base_exit=$base_rc" >> "$OUT/status.txt"
if [ "$base_rc" != 0 ] || [ ! -f python/mojolearn/identical/_mojolearn.so ]; then
    say "byte-lm-lifetime: BASE BUILD FAILED; see $OUT/build_base.log"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/status.txt"
    exit 1
fi
sha256sum python/mojolearn/identical/_mojolearn.so > "$OUT/binding_base_sha256.txt" 2>&1
# DEVIATION 2513: the ExtraTrees control (control_extratrees_x2) goes
# through the trees binding; same rm-then-build pattern, IDENTICAL tier
# (bindings/build_trees.sh lands it under python/mojolearn/identical/ and
# skips its own gate for that tier). A failed trees build is recorded and
# only the ET control fails; the KMeans controls and the keeper cases still
# run through the base and byte LM bindings.
rm -f python/mojolearn/identical/_mojolearn_trees.so
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS="$ARCH" MOJOLEARN_SKIP_BUILD_GATE=1 \
    timeout -k 30 1500 sh bindings/build_trees.sh > "$OUT/build_trees.log" 2>&1
trees_rc=$?
echo "build_trees_exit=$trees_rc" >> "$OUT/status.txt"
if [ "$trees_rc" != 0 ] || [ ! -f python/mojolearn/identical/_mojolearn_trees.so ]; then
    say "byte-lm-lifetime: TREES BUILD FAILED (exit $trees_rc); control_extratrees_x2 will fail at import; see $OUT/build_trees.log"
else
    sha256sum python/mojolearn/identical/_mojolearn_trees.so > "$OUT/binding_trees_sha256.txt" 2>&1
fi
MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS="$ARCH" \
MOJOLEARN_BYTE_LM_OUTDIR="$DEST_DIR" \
    timeout -k 30 1500 sh bindings/build_byte_lm.sh > "$OUT/build.log" 2>&1
build_rc=$?
echo "build_exit=$build_rc build_seconds=$(( $(date +%s) - build_started ))" >> "$OUT/status.txt"
say "byte-lm-lifetime: build exit $build_rc after $(( $(date +%s) - build_started ))s"
if [ "$build_rc" != 0 ] || [ ! -f "$DEST" ]; then
    say "byte-lm-lifetime: BUILD FAILED; see $OUT/build.log"
    echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/status.txt"
    exit 1
fi
sha256sum "$DEST" > "$OUT/binding_sha256.txt" 2>&1
# Read the mode/vendor/profile BACK from the binary, never from the command.
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" pixi run python3 - > "$OUT/binding_readback.txt" 2>&1 <<'PYEOF'
from mojolearn import _backend
b = _backend.binding('_mojolearn_byte_lm', 'identical')
print('numeric_mode', int(b.byte_lm_numeric_mode()))
print('vendor', str(b.byte_lm_vendor()))
print('profile', str(b.byte_lm_profile()))
print('file', b.__file__)
# DEVIATION 2513: the keeper read-back must exist on THIS binary, and must
# be False before any call (the switch is off and nothing ran).
print('keeper_readback', hasattr(b, 'byte_lm_context_keeper_active'))
print('keeper_active_at_import', bool(b.byte_lm_context_keeper_active()) if hasattr(b, 'byte_lm_context_keeper_active') else None)
PYEOF
echo "readback_exit=$?" >> "$OUT/status.txt"
cat "$OUT/binding_readback.txt"

# ---------------------------------------------------------------- run
# Each case is its own process; the harness bounds every one of them. The
# outer timeout only protects the lease if the harness itself wedges.
cases_arg=""
[ -z "${BYTE_LM_LIFETIME_CASES:-}" ] || cases_arg="--cases $BYTE_LM_LIFETIME_CASES"
# A native stack of a hung child is the datum the Python stacks cannot give
# (run 3 and 4 on the RTX 4090 had neither tool). Best effort, root pod only:
# gdb from apt, py-spy from pip into the pixi interpreter.
if ! command -v gdb >/dev/null 2>&1 && command -v apt-get >/dev/null 2>&1; then
    (apt-get update -qq && apt-get install -y -qq gdb) > "$OUT/gdb_install.log" 2>&1 || true
fi
pixi run python3 -m pip install -q py-spy > "$OUT/pyspy_install.log" 2>&1 || true
export PATH="$ROOT/.pixi/envs/default/bin:$PATH"
{ echo "gdb=$(command -v gdb || echo none)"; echo "py-spy=$(command -v py-spy || echo none)"; } >> "$OUT/status.txt"
run_started=$(date +%s)
MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" PYTHONUNBUFFERED=1 \
    timeout -k 30 3000 pixi run python3 tools/byte_lm_lifetime_diag.py \
        --out "$OUT/cases" --deadline "$DEADLINE" $cases_arg > "$OUT/harness.log" 2>&1
run_rc=$?
echo "harness_exit=$run_rc run_seconds=$(( $(date +%s) - run_started ))" >> "$OUT/status.txt"
say "byte-lm-lifetime: harness exit $run_rc after $(( $(date +%s) - run_started ))s"
cat "$OUT/harness.log"
if [ -f "$OUT/cases/summary.json" ]; then
    pixi run python3 - "$OUT/cases/summary.json" <<'PYEOF'
import json, sys
s = json.load(open(sys.argv[1]))
for name, c in s['cases'].items():
    print('%-32s %-8s exit=%s wall=%.1fs' % (name, c['status'], c['exit_code'], c['wall_s']))
for key in ('first_step_equality', 'second_step_equality'):
    eq = s[key]
    bad = [f for f, v in eq['fields'].items() if not v['bit_equal']]
    print(key, 'cases:', eq['cases_compared'], 'NOT bit-equal fields:', bad or 'none')
for case, eq in s.get('control_fit_equality', {}).items():
    print(case, 'fit1 == fit2:', eq.get('all_bit_equal') if eq.get('compared') else 'not compared')
for case, k in s.get('keeper_reached', {}).items():
    print(case, 'keeper_active:', k.get('keeper_active'), 'env:', k.get('keep_context_env'))
print('hung:', s['hung'], 'failed:', s['failed'])
PYEOF
fi
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ) total_seconds=$(( $(date +%s) - started ))" >> "$OUT/status.txt"
say "byte-lm-lifetime: done, total $(( $(date +%s) - started ))s"
exit "$run_rc"
