#!/bin/bash
# On-pod bootstrap for the Samba training lane: pixi, the identical builds.
set -u
ROOT=/root/mojolearn
OUT=/root/samba_out
mkdir -p "$OUT"
cd "$ROOT" || exit 9
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export MAX_JOBS=4 MOJOLEARN_COMPILE_JOBS=4
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader > "$OUT/gpu.txt" 2>&1
if [ ! -x "$HOME/.pixi/bin/pixi" ] && ! command -v pixi >/dev/null 2>&1; then
    curl -fsSL --max-time 300 https://pixi.sh/install.sh | sh > "$OUT/pixi_install.log" 2>&1
fi
export PATH="$HOME/.pixi/bin:$PATH"
( time pixi install --locked ) > "$OUT/pixi_env.log" 2>&1
echo "pixi_install_exit=$?" | tee -a "$OUT/leg.txt"
pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1
cat "$OUT/mojo_version.txt"
# byte_lm FIRST: its script refuses an existing python/mojolearn/identical.
for b in byte_lm training mamba transformer; do
    ( time MOJOLEARN_NUMERIC_MODE=identical bash bindings/build_$b.sh ) > "$OUT/build_$b.log" 2>&1
    echo "build_${b}_exit=$?" | tee -a "$OUT/leg.txt"
    tail -3 "$OUT/build_$b.log"
done
ls -l python/mojolearn/identical/
