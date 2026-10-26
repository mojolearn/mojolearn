#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Which HIP / HSA runtime copies load when mojolearn and PyTorch (ROCm) share a process on AMD.
# (2026-10-26, lane amd-torch-interop-0859; python/mojolearn/_rocm_runtime.py, docs/plans/MODULAR_ASK_AMD_HIP_RUNTIME.md)
#
#   bash tools/amd_torch_interop_diag.sh <venv> [out-dir] [rocm-lib-dir]
#
# Writes every full output under <out-dir> (default ~/mojolearn-evidence/amd-torch-interop/diag-<utc>)
# and prints only DIAG lines: one per scenario (status, exit code, HIP/HSA copies mapped at exit).
# Scenarios: two import orders x two LD_LIBRARY_PATH settings, each with one KMeans fit and one
# torch GPU op, then the two autograd smoke entries (import-torch-smoke, real-model-smoke).
set -u
VENV=${1:?usage: amd_torch_interop_diag.sh <venv> [out-dir] [rocm-lib-dir]}
OUT=${2:-$HOME/mojolearn-evidence/amd-torch-interop/diag-$(date -u +%Y%m%dT%H%M%SZ)}
ROCM=${3:-/opt/rocm-7.2.3/lib}
PY="$VENV/bin/python"
REPO=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$OUT"
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
TORCH_LIB="$SP/torch/lib"
MLIBS=$(ls -d "$SP/mojolearn/.libs" 2>/dev/null)
echo "DIAG venv=$VENV site=$SP torch_lib=$TORCH_LIB rocm=$ROCM out=$OUT"
# The branch's loader over the installed package, without touching the venv: a symlink farm of the
# installed mojolearn/ (binaries, plugin sets, .libs) with this tree's loader files copied in.
# OVERLAY=0 runs the installed package as it is (the "before" arm).
if [ "${OVERLAY:-1}" = 1 ]; then
  rm -rf "$OUT/overlay"; mkdir -p "$OUT/overlay"
  cp -rs "$SP/mojolearn" "$OUT/overlay/mojolearn"
  for f in _rocm_runtime.py _backend.py _gpu_witness.py; do
    rm -f "$OUT/overlay/mojolearn/$f"; cp "$REPO/python/mojolearn/$f" "$OUT/overlay/mojolearn/$f"
  done
  find "$OUT/overlay" -name '__pycache__' -prune -exec rm -rf {} + 2>/dev/null
  export PYTHONPATH="$OUT/overlay${PYTHONPATH:+:$PYTHONPATH}"
  echo "DIAG overlay=$OUT/overlay commit=$(git -C "$REPO" rev-parse --short HEAD)"
fi

# 1. Static facts: sonames, NEEDED, RUNPATH of every library on the path.
{
  for f in "$TORCH_LIB"/libamdhip64.so* "$TORCH_LIB"/libhsa-runtime64.so* "$TORCH_LIB"/librocprofiler-register.so* \
           "$TORCH_LIB"/libtorch_hip.so "$SP"/torchvision/_C.so "$ROCM"/libamdhip64.so* "$ROCM"/libhsa-runtime64.so* \
           $(ls "$SP"/torch/lib/libc10_hip.so 2>/dev/null) $(ls "$SP"/torchvision.libs/*.so* 2>/dev/null) \
           $(find "$SP/mojolearn" -name '_mojolearn_native.so' 2>/dev/null | head -1) \
           $(ls "$MLIBS"/*.so* 2>/dev/null) $(find "$SP/max" "$SP/modular" -name '*.so' 2>/dev/null | head -40); do
    [ -e "$f" ] || continue
    echo "== $f -> $(readlink -f "$f") inode=$(stat -Lc %i "$f")"
    readelf -d "$f" 2>/dev/null | grep -E 'NEEDED|SONAME|RPATH|RUNPATH'
  done
} > "$OUT/readelf.txt" 2>&1
# Bare names the MAX runtime may dlopen, per library.
{
  for f in $(ls "$MLIBS"/*.so* 2>/dev/null) $(find "$SP/max" "$SP/modular" "$SP/mojo" -name '*.so' 2>/dev/null); do
    s=$(strings -n 8 "$f" 2>/dev/null | grep -E '^(lib)?(amdhip64|hsa-runtime64|rocprofiler)[^ ]*$|HIP_PATH|ROCM_PATH|MODULAR_[A-Z_]*(HIP|ROCM|AMD)' | sort -u | tr '\n' ' ')
    [ -n "$s" ] && echo "$f: $s"
  done
} > "$OUT/modular-dlopen-names.txt" 2>&1
echo "DIAG static readelf=$OUT/readelf.txt names=$OUT/modular-dlopen-names.txt"

MAPS='import re
c={}
for l in open("/proc/self/maps"):
    p=l.split(None,5)
    if len(p)==6 and re.search(r"/lib(amdhip64|hsa-runtime64|rocprofiler-register)[^/]*$", p[5].strip()):
        c.setdefault(re.search(r"lib(amdhip64|hsa-runtime64|rocprofiler-register)", p[5]).group(1), set()).add(p[5].strip())
print("COPIES " + " ".join("%s=%d:%s" % (k, len(v), ",".join(sorted(v))) for k, v in sorted(c.items())), flush=True)'
FIT='import numpy as np, mojolearn
X=np.random.default_rng(0).random((4096,8),dtype=np.float32)
mojolearn.KMeans(n_clusters=4).fit(X); print("MOJOLEARN_FIT_OK", mojolearn.vendor(), mojolearn.__file__, flush=True)'
TORCHOP='import torch
x=torch.ones(1024, device="cuda"); print("TORCH_GPU_OK", float((x*2).sum()), flush=True)
import torchvision; print("TORCHVISION_OK", torchvision.__version__, flush=True)'

scenario() {  # name ld order
  local name=$1 ld=$2 order=$3 code
  if [ "$order" = mojolearn-first ]; then code="$FIT"$'\n'"$TORCHOP"$'\n'"$FIT"$'\n'"$MAPS"
  else code="$TORCHOP"$'\n'"$FIT"$'\n'"$MAPS"; fi
  LD_LIBRARY_PATH="$ld" LD_DEBUG=libs LD_DEBUG_OUTPUT="$OUT/$name.lddebug" timeout 600 "$PY" -c "$code" > "$OUT/$name.log" 2>&1
  local rc=$?
  local status=FAIL
  grep -q '^COPIES' "$OUT/$name.log" && status=PASS
  # the branch loader's import-time refusal (torch first, MAX would map a second runtime)
  grep -q 'that the MAX runtime would not reuse' "$OUT/$name.log" && status=REFUSED
  echo "DIAG scenario=$name rc=$rc status=$status $(grep -m1 '^COPIES' "$OUT/$name.log" | cut -c1-400)"
  [ "$status" = PASS ] || grep -m 3 -E 'Error|error|abort|hip.cpp|HSA' "$OUT/$name.log" | cut -c1-240 | sed 's/^/DIAG   /'
}
scenario mojolearn-first.rocm   "$ROCM"                 mojolearn-first
scenario torch-first.rocm       "$ROCM"                 torch-first
scenario mojolearn-first.torchlib "$TORCH_LIB:$ROCM"    mojolearn-first
scenario torch-first.torchlib   "$TORCH_LIB:$ROCM"      torch-first

# 2. The two autograd smoke entries, as the gate runs them (system ROCm only).
for probe in import-torch real-model; do
  LD_LIBRARY_PATH="$MLIBS:$ROCM" timeout 900 "$PY" "$REPO/tools/autograd_smoke.py" probe "$probe" > "$OUT/smoke-$probe.log" 2>&1
  rc=$?
  echo "DIAG smoke=$probe rc=$rc $(grep -m1 -E '^(SMOKE_DIGEST|SMOKE_FAIL|SMOKE_REFUSED|SKIPPED)' "$OUT/smoke-$probe.log" | cut -c1-240)"
  [ $rc -eq 0 ] || grep -m 3 -E 'Error|error|abort|hip.cpp|HSA' "$OUT/smoke-$probe.log" | cut -c1-240 | sed 's/^/DIAG   /'
done
echo "DIAG done out=$OUT"
