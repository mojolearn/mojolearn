#!/bin/sh
# tools/amd_mi325x_perflevel_probe.sh -- lane/amd-step-time-2 (2026-09-25).
# PREPARED, NOT RUN. For ONE DigitalOcean MI325X droplet, AFTER the T3 run's
# segment A/4 has landed and its droplet is gone (the account has one MI325X
# and tools/do_extra_leg.sh refuses while another droplet exists):
#
#   MOJOLEARN_GPU_ARCHS=gfx942 \
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/amd_mi325x_perflevel_probe.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/amd_step_time_2026-09-24/legs/$(date -u +%Y-%m-%d_%H%M%S)-do-mi325x-perflevel \
#   bash tools/do_extra_leg.sh amd --size gpu-mi325x1-256gb --region nyc2 --skip-gates
#
# (about 25 minutes of droplet, about $1.60 at $3.80/h). The question: are the
# ~100 ms stalls about every 250 ms that the MI325X step showed (branch
# lane/amd-step-time-mi325x, README section MI325X) a power-state effect
# that a performance level removes, or something the GPU settings cannot
# reach (host, hypervisor, driver)? The loop is ONE small kernel launch plus
# synchronize (tools/amd_codegen/stall_probe.mojo), timed for 60 s, twice
# (an almost empty kernel, and one of about 1 ms), under:
#   1. auto            the droplet as delivered (T1 to T3 ran this way)
#   2. determinism     rocm-smi --setperfdeterminism 1900 (sclk cap, MHz)
#   3. high            rocm-smi --setperflevel high
#   4. auto again      after --resetperfdeterminism and --setperflevel auto
# with rocm-smi sampled every second beside each run. Read the STALL lines in
# session.txt: over20ms and slow_gap_median_ms near 250 ms under auto and
# near zero under 2 or 3 means the power state; unchanged under all four
# means it is not a GPU clock setting. The settings are restored at the end
# whatever happens. No floating-point operation changes under any level
# (clocks only).
set -u
OUT=/root/gemm_leg_out/perflevel
mkdir -p "$OUT"
cd /root/mojolearn || exit 9
PATH="$HOME/.pixi/bin:$PATH"; export PATH
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=amd
: "${MOJOLEARN_GPU_ARCHS:=gfx942}"; export MOJOLEARN_GPU_ARCHS
export PYTHONPATH="/root/mojolearn/python:/root/mojolearn"
SMI=$(command -v rocm-smi || echo /opt/rocm/bin/rocm-smi)
say() { echo "$(date -u +%H:%M:%S) $*" | tee -a "$OUT/session.txt"; }
restore() {
    "$SMI" --resetperfdeterminism > /dev/null 2>&1
    "$SMI" --setperflevel auto > /dev/null 2>&1
}
trap restore EXIT INT TERM
say "perflevel probe started"
"$SMI" --showproductname --showuniqueid --showperflevel --showclocks --showpower > "$OUT/smi_before.txt" 2>&1
cat /sys/class/drm/card*/device/power_dpm_force_performance_level > "$OUT/dpm_level_before.txt" 2>&1
pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_AMD --target-accelerator "$MOJOLEARN_GPU_ARCHS" -I . \
    tools/amd_codegen/stall_probe.mojo -o /root/stall_probe > "$OUT/build.log" 2>&1
say "build exit=$?"
[ -x /root/stall_probe ] || exit 3

runset() {  # label
    for spin in 0 20000; do
        ( while :; do date -u +%H:%M:%S; "$SMI" --showclocks --showpower --showuse 2>/dev/null | grep -E 'sclk|mclk|Power|GPU use'; sleep 1; done ) \
            > "$OUT/smi-$1-$spin.txt" 2>&1 &
        sp=$!
        STALL_LABEL=$1 STALL_SECONDS=60 STALL_SPIN=$spin /root/stall_probe > "$OUT/stall-$1-$spin.log" 2>&1
        rc=$?
        kill $sp 2> /dev/null; wait $sp 2> /dev/null
        say "$1 spin=$spin exit=$rc: $(grep '^STALL ' "$OUT/stall-$1-$spin.log" | cut -c1-400)"
    done
}
runset auto
"$SMI" --setperfdeterminism 1900 > "$OUT/set_determinism.txt" 2>&1; say "setperfdeterminism 1900 exit=$?"
"$SMI" --showperflevel >> "$OUT/set_determinism.txt" 2>&1
runset determinism
"$SMI" --resetperfdeterminism > "$OUT/reset_determinism.txt" 2>&1; say "resetperfdeterminism exit=$?"
"$SMI" --setperflevel high > "$OUT/set_high.txt" 2>&1; say "setperflevel high exit=$?"
"$SMI" --showperflevel >> "$OUT/set_high.txt" 2>&1
runset high
restore
say "restored: $("$SMI" --showperflevel 2>&1 | grep -i level | tr '\n' ' ')"
runset auto2
"$SMI" --showperflevel --showclocks --showpower > "$OUT/smi_after.txt" 2>&1
say "perflevel probe done"
exit 0
