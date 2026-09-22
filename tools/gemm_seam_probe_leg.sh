#!/bin/sh
# tools/gemm_seam_probe_leg.sh -- DEVIATION 2701, the on-box body of the seam
# probe. Builds and runs
# gemm/checks/gemm_seam_probe.mojo under IDENTICAL and leaves seam_probe.log in
# the leg's out dir. Needs NO dataset: the probe generates its 64 words itself.
#
# VENDOR-AGNOSTIC. Runs as MOJOLEARN_GEMM_LEG_EXTRA of tools/gemm_remote_leg.sh
# (RunPod NVIDIA) and of tools/hotaisle_leg.sh (Hot Aisle AMD); both export
# MOJOLEARN_GPU_ARCHS and copy /root/gemm_leg_out home. The M4 column is run
# by hand: pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
#   gemm/checks/gemm_seam_probe.mojo -o /tmp/seam && /tmp/seam > seam_probe.log
# and every log is read by tools/gemm_seam_probe_reference.py.
# POSIX sh only: RunPod's Ubuntu images link /bin/sh to dash.
set -u
ROOT=${MOJOLEARN_GEMM_STEP_ROOT:-/root/mojolearn}
OUT=${MOJOLEARN_GEMM_STEP_LEG_OUT:-/root/gemm_leg_out/seam-probe}
mkdir -p "$OUT"
export PATH="$HOME/.pixi/bin:$PATH"
{
    echo "deviations=2701"
    echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "root=$ROOT gpu_archs=${MOJOLEARN_GPU_ARCHS:-unset} column=${MOJOLEARN_TARGET_COLUMN:-unset}"
} > "$OUT/probe.txt"
cd "$ROOT" || { echo "no $ROOT" >> "$OUT/probe.txt"; exit 9; }
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
if pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 \
    gemm/checks/gemm_seam_probe.mojo -o "$OUT/seam_probe" > "$OUT/build.log" 2>&1; then
    echo "build=0" >> "$OUT/probe.txt"
else
    echo "build=$?" >> "$OUT/probe.txt"; exit 9
fi
"$OUT/seam_probe" > "$OUT/seam_probe.log" 2>&1
echo "run=$?" >> "$OUT/probe.txt"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/probe.txt"
grep -h '^SEAM_PROBE\|^SEAM_HASH\|^SEAM_BOUNDARY\|^SEAM_MISMATCH' "$OUT/seam_probe.log" >> "$OUT/probe.txt"
