# peer_copy_check.mojo (PEERCOPY + PEERSOLVE variants) on a two-GPU box.
set -u
cd /root/mojolearn || exit 9
echo 32a39bd6196de9f4b4b08f3fa71d106df864c44f > commit.txt
OUT=/root/gemm_leg_out/peer
mkdir -p "$OUT"
export RUNPOD_POD_ID="${RUNPOD_POD_ID:-runpod-leg}" MOJOLEARN_NUMERIC_MODE=identical
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then ARCH=sm_90a; COL=NVIDIA; else ARCH=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+'); COL=AMD; fi
echo "arch=$ARCH" > "$OUT/gate.txt"
cp commit.txt "$OUT/commit.txt"
pixi run mojo build -D MOJOLEARN_COLUMN_$COL -D MOJOLEARN_NUMERIC_IDENTICAL=1 --target-accelerator $ARCH -I . training/checks/peer_copy_check.mojo -o /root/peer_copy_check > "$OUT/build.log" 2>&1
echo "build exit=$?" >> "$OUT/gate.txt"
env MOJOLEARN_PEERCOPY_SKIP_BARE=1 MOJOLEARN_PEERCOPY_TRIALS=4 MOJOLEARN_PEERCOPY_ENABLE_PEER=1 pixi run /root/peer_copy_check > "$OUT/peer.log" 2>&1
echo "peer exit=$?" >> "$OUT/gate.txt"
env MOJOLEARN_PEERCOPY_SKIP_BARE=1 MOJOLEARN_PEERCOPY_SKIP_SOLVE=1 MOJOLEARN_PEERCOPY_TRIALS=4 pixi run /root/peer_copy_check > "$OUT/peer-repeat.log" 2>&1
echo "peer-repeat exit=$?" >> "$OUT/gate.txt"
grep -E "^PEER|Unhandled|rror" "$OUT/peer.log" "$OUT/peer-repeat.log" >> "$OUT/gate.txt"
exit 0
