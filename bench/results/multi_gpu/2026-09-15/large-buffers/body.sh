# Large-buffer runs of the merged GaussianMixture and resampling native gates (lane/multigpu-large-buffers).
set -u
cd /root/mojolearn || exit 9
echo 8de0b43fe219fb3b7a547968d0aee9e4140b5a64 > commit.txt
OUT=/root/gemm_leg_out/large
mkdir -p "$OUT/traces"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-runpod-leg}" MOJOLEARN_NUMERIC_MODE=identical
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then ARCH=sm_90a; COL=NVIDIA; else ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+'); COL=AMD; fi
echo "arch=$ARCH" > "$OUT/gate.txt"
FLAGS="--target-accelerator $ARCH -D MOJOLEARN_COLUMN_$COL -D MOJOLEARN_NUMERIC_IDENTICAL=1"
env MOJOLEARN_GMM_CHECK_LARGE=1 MOJOLEARN_GMM_CHECK_DIR="$OUT/traces" pixi run mojo run $FLAGS -I . training/checks/gmm_parallel_check.mojo > "$OUT/gmm-large.log" 2>&1
echo "gmm-large exit=$?" >> "$OUT/gate.txt"
env MOJOLEARN_RESAMPLE_CHECK_LARGE=1 MOJOLEARN_RESAMPLE_CHECK_DIR="$OUT/traces" pixi run mojo run $FLAGS -I . training/checks/resample_parallel_check.mojo > "$OUT/resample-large.log" 2>&1
echo "resample-large exit=$?" >> "$OUT/gate.txt"
grep -hE "^PASS|Unhandled" "$OUT/gmm-large.log" "$OUT/resample-large.log" >> "$OUT/gate.txt"
( cd "$OUT/traces" && for f in *.trace; do echo "$(sha256sum "$f" | cut -c1-64) $f"; done ) > "$OUT/digests.txt"
rm -rf "$OUT/traces"
exit 0
