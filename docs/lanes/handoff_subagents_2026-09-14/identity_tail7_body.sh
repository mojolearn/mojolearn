# the seven lanes the 60-minute MI300X leg did not reach (iforest, iforest-tuned and five par-* lanes), same commit, every binding rebuilt
set -u
cd /root/mojolearn
[ -n "${MOJOLEARN_COMMIT:-}" ] && echo "$MOJOLEARN_COMMIT" > commit.txt
OUT=/root/gemm_leg_out; mkdir -p "$OUT"
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+'); export MOJOLEARN_GPU_ARCHS; fi
echo "archs=$MOJOLEARN_GPU_ARCHS"
built=0; failed=""
for s in bindings/build*.sh; do
    n=$(basename "$s" .sh); [ "$n" = build_host_family ] && continue
    case "$n" in build_*_host) E="env -u MOJOLEARN_GPU_ARCHS MOJOLEARN_TARGET_COLUMN=cpu";; *) E="env";; esac
    if $E MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_COMPILE_JOBS=8 sh "$s" > "$OUT/build_$n.log" 2>&1; then built=$((built+1)); else failed="$failed $n"; fi
done
echo "bindings_built=$built failed=${failed:-none}"
LANES=iforest,iforest-tuned,par-arima,par-byte-lm,par-iforest,par-mlp,par-samba
env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMMIT="$(cat commit.txt)" PYTHONPATH=/root/mojolearn/python pixi run python tools/identity_break.py --lanes "$LANES" --json "$OUT/identity_break.tail7.json" > "$OUT/identity_tail7.log" 2>&1
echo "identity exit $?"; grep -E "^cells=|^REFUSED" "$OUT/identity_tail7.log" | head -8
exit 0
