#!/bin/sh
# tools/multinomial_checks_leg.sh: the on-box body of the QN softmax and
# six-loss gate legs (MOJOLEARN_GEMM_LEG_EXTRA for tools/gemm_remote_leg.sh,
# tools/do_extra_leg.sh and tools/hotaisle_leg.sh). Same shape as
# tools/identity_three_columns_leg.sh: cd /root/mojolearn, resolve the GPU
# architecture from the device when unset, run, write logs and gate.txt into
# /root/gemm_leg_out/, which the runner brings home.
#
# Three runs, the two pixi tasks' exact command lines plus the negative
# control (lane/logistic-multiclass, 2026-09-14; Apple M4 evidence in the
# commit that added this file):
#   multinomial   glm/checks/multinomial_check.mojo, nine arms, must exit 0
#   qn_losses     glm/checks/qn_losses_check.mojo, seven arms, must exit 0
#   sabotage      multinomial_check with -D MOJOLEARN_SOFTMAX_SABOTAGE=1 (the
#                 log-sum-exp fold walked descending), must exit NONZERO at
#                 check_softmax_device_equals_host; a pass here is the finding
# gate.txt ends with one VERDICT line: GREEN when all three are as required.
set -u
cd /root/mojolearn || exit 9
OUT=/root/gemm_leg_out/multinomial_checks
mkdir -p "$OUT/logs"
G="$OUT/gate.txt"
say() { echo "$@" >> "$G"; }
run() {
    _n=$1; shift
    _t0=$(date +%s)
    "$@" > "$OUT/logs/$_n.log" 2>&1
    _e=$?
    echo "$_n	$_e	$(( $(date +%s) - _t0 ))" >> "$OUT/status.tsv"
    return "$_e"
}
say "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
say "vendor=${MOJOLEARN_TARGET_COLUMN:-unset} archs=${MOJOLEARN_GPU_ARCHS:-unset}"
if [ -n "${MOJOLEARN_COMMIT:-}" ] && [ ! -s /root/mojolearn/commit.txt ]; then echo "$MOJOLEARN_COMMIT" > /root/mojolearn/commit.txt; fi
say "commit=$(cat /root/mojolearn/commit.txt /root/mojolearn/COMMIT 2>/dev/null | head -1 || git -C /root/mojolearn rev-parse HEAD 2>/dev/null || echo unknown)"
(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null; rocminfo 2>/dev/null | grep -m1 -oE "gfx[0-9a-z]+") > "$OUT/logs/device.txt" 2>&1
say "device=$(tr '\n' ' ' < "$OUT/logs/device.txt")"
# The RunPod runner passes no environment to the body; derive the
# architecture from the device when unset, as identity_three_columns_leg.sh
# does, so `mojo run` and any build it triggers target this box.
if [ -z "${MOJOLEARN_GPU_ARCHS:-}" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        _cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d ' ')
        case "$_cc" in 9.0) MOJOLEARN_GPU_ARCHS=sm_90a ;; 8.9) MOJOLEARN_GPU_ARCHS=sm_89 ;; 8.6) MOJOLEARN_GPU_ARCHS=sm_86 ;; 8.0) MOJOLEARN_GPU_ARCHS=sm_80 ;; 12.0) MOJOLEARN_GPU_ARCHS=sm_120a ;; *) MOJOLEARN_GPU_ARCHS="sm_$(echo "$_cc" | tr -d .)" ;; esac
    elif command -v rocminfo >/dev/null 2>&1; then
        MOJOLEARN_GPU_ARCHS=$(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-z]+')
    fi
    export MOJOLEARN_GPU_ARCHS
fi
say "gpu_archs_resolved=${MOJOLEARN_GPU_ARCHS:-unset}"
run multinomial pixi run mojo run -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 glm/checks/multinomial_check.mojo
run qn_losses pixi run mojo run -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 glm/checks/qn_losses_check.mojo
run sabotage pixi run mojo run -j 2 -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_SOFTMAX_SABOTAGE=1 glm/checks/multinomial_check.mojo
for n in multinomial qn_losses sabotage; do
    say "== $n exit=$(awk -F'\t' -v n="$n" '$1==n{print $2}' "$OUT/status.tsv") seconds=$(awk -F'\t' -v n="$n" '$1==n{print $3}' "$OUT/status.tsv")"
    # The verdict lines only: the check header, every arm's OK or REPORT
    # line, and the failure sentence when one arm raised.
    grep -E "^== glm/checks|^check_|^Unhandled exception" "$OUT/logs/$n.log" >> "$G" 2>/dev/null
done
e_m=$(awk -F'\t' '$1=="multinomial"{print $2}' "$OUT/status.tsv")
e_q=$(awk -F'\t' '$1=="qn_losses"{print $2}' "$OUT/status.tsv")
e_s=$(awk -F'\t' '$1=="sabotage"{print $2}' "$OUT/status.tsv")
if [ "$e_m" = 0 ] && [ "$e_q" = 0 ] && [ "$e_s" != 0 ] && grep -q "check_softmax_device_equals_host \[IDENTICAL\]:.*differ" "$OUT/logs/sabotage.log"; then
    say "VERDICT=GREEN multinomial=0 qn_losses=0 sabotage=$e_s (failed at device_equals_host as required)"
else
    say "VERDICT=RED multinomial=$e_m qn_losses=$e_q sabotage=$e_s (a 0 for sabotage, or a nonzero elsewhere, is the finding)"
fi
say "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
