#!/bin/bash
# tools/ops/box_job_lib.sh -- the shared functions of the box measurement jobs (tools/ops/paper_compare_job.sh,
# tools/ops/attribution_job.sh), moved here unchanged from paper_compare_job.sh (lane attribution-job, 2026-11-16) so
# a second job reuses them instead of copying them. Sourced, never run. The caller sets the globals they read: RES
# (results dir), J (RES/job), PY, V (vendor), TOKEN and NGPU (gpu_other), WORST=0, PHASE, P_T0, JOB_NAME (say's tag)
# and TREE (cwd: host_probe_phase runs tools/ops/host_probe.py relative to it).
# Tests: tools/test_paper_compare.py (the functions run on their own), tools/test_attribution_job.py.

detect_vendor() {
    if command -v nvidia-smi > /dev/null 2>&1; then echo nvidia
    elif command -v rocm-smi > /dev/null 2>&1 || [ -x /opt/rocm/bin/rocm-smi ]; then echo amd
    else return 2; fi
}

say() { echo "[$(date -u +%H:%M:%S) ${JOB_NAME:-box-job}] $*"; }

phase() { PHASE=$1; P_T0=$(date +%s); echo "$1" > "$J/.phase"; say "phase $PHASE start"; }
phase_end() {   # rc
    local rc=$1 full
    full=$(disk_full_in_phase)
    [ -n "$full" ] && rc=$EXIT_DISK
    printf '%s\t%s\t%s\t%s\n' "$PHASE" "$rc" "$(( $(date +%s) - P_T0 ))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$J/phases.tsv"
    say "phase $PHASE rc=$rc $(( $(date +%s) - P_T0 )) s"
    [ "$rc" -gt "$WORST" ] && WORST=$rc
    if [ -n "$full" ]; then
        # fail soft stops here: the next phase would run into the same full disk (2026-11-14: the owed prior and all
        # 653 smoke races of the full-board screen failed one after the other on ENOSPC)
        say "DISK FULL in phase $PHASE: $full; $(free_gb) GB free on $(df -P "$RES" 2> /dev/null | awk 'NR==2 {print $6}'); stopping (exit $EXIT_DISK)"
        exit "$EXIT_DISK"
    fi
    return 0
}

# ---- disk (2026-11-14) ---------------------------------------------------------------------------------------------
EXIT_DISK=28    # ENOSPC: the job stopped on a full disk, or refused to start a phase whose evidence would not fit
nlines() { local c; c=$(grep -c . "$1" 2> /dev/null); echo "${c:-0}"; }
free_gb() { df -Pk "$RES" 2> /dev/null | awk 'NR==2 {printf "%.1f", $4 / 1048576}'; }
# non-empty (the evidence) when this phase hit a full disk: a log written during it that says ENOSPC, or less than
# PC_DISK_MIN_FREE_MB (512) left at its end
disk_full_in_phase() {
    local hit kb
    # job/.phase is written when the phase starts (phase()): a log modified after it was written during this phase.
    # job/, command.log and persist.log carry this job's own messages (a DISK FULL line of an earlier submit): not evidence
    hit=$(find "$RES" -maxdepth 3 -path "$J" -prune -o -type f -name '*.log' ! -name command.log ! -name persist.log \
        -newer "$J/.phase" -exec grep -l -m 1 -E 'No space left on device|Errno 28|ENOSPC' {} + 2> /dev/null | head -n 3 | tr '\n' ' ')
    if [ -n "$hit" ]; then echo "ENOSPC in ${hit% }"; return; fi
    kb=$(df -Pk "$RES" 2> /dev/null | awk 'NR==2 {print $4}')
    [ -n "$kb" ] && [ "$kb" -lt $(( ${PC_DISK_MIN_FREE_MB:-512} * 1024 )) ] && echo "only $(( kb / 1024 )) MB free at the phase end"
}
# refuse to START a phase that writes race evidence when the free disk is below the prediction:
# races x PC_EVIDENCE_MB_PER_RACE (smoke: PC_SMOKE_EVIDENCE_MB_PER_RACE) + boards x PC_LOADED_MB (the loaded native
# modules, retained once per board dir: tools/bench_board_evidence.py loaded_module_ref) + PC_DISK_RESERVE_GB.
# The per-race default is the retained receipts, logs and records (the .so bytes are no longer in each race,
# 2026-11-14); raise it for a board of large-output races.
disk_gate() {   # <what> <races> <board processes> <MB per race>
    local what=$1 races=$2 boards=$3 per=$4 need_mb free_kb
    need_mb=$(( races * per + boards * ${PC_LOADED_MB:-400} + ${PC_DISK_RESERVE_GB:-5} * 1024 ))
    free_kb=$(df -Pk "$RES" 2> /dev/null | awk 'NR==2 {print $4}')
    [ -n "$free_kb" ] || { say "DISK $what: cannot read the free space of $RES; not gated"; return 0; }
    say "DISK $what: $races races x $per MB + $boards boards x ${PC_LOADED_MB:-400} MB + reserve ${PC_DISK_RESERVE_GB:-5} GB = $need_mb MB needed; $(( free_kb / 1024 )) MB free"
    if [ $(( free_kb / 1024 )) -lt "$need_mb" ]; then
        say "DISK $what refused: free space is below the predicted evidence; free disk or lower the estimate (exit $EXIT_DISK)"
        printf '%s\t%s\t%s\t%s\n' "$what-disk-refused" "$EXIT_DISK" 0 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$J/phases.tsv"
        WORST=$EXIT_DISK
        exit "$EXIT_DISK"
    fi
}

# busy cores of total every 60 s (job/cores.log) and a BUSY line in job.log every 5 minutes.
# BUSY-NEGATIVE FIX (2026-11-14, L40S pod nvc1: `busy=-18.3 total=128`): the old sampler took (all - idle - iowait)
# over the /proc/stat deltas and scaled by nproc. iowait is not monotonic (the kernel documents that it can go down),
# and a container's /proc/stat and nproc describe the host, not the pod. Now: the pod's own CPU time from its cgroup
# (cgroup v2 cpu.stat usage_usec, else v1 cpuacct.usage), busy = delta CPU time / delta wall time; total = the
# allocation (cgroup v2 cpu.max quota/period, else nproc, which honours affinity). Without a cgroup counter it falls
# back to /proc/stat with busy = delta(user+nice+system+irq+softirq+steal) over delta(those + idle + iowait), scaled
# by the cpu lines /proc/stat lists. A delta that goes backwards (counter reset) is not a sample: it is logged as
# `busy=NA` and the next sample starts from it; a value outside 0..total is clamped and marked `clamped`.
alloc_cores() {
    local q per n
    n=$(nproc 2> /dev/null || echo 1)
    if read -r q per < /sys/fs/cgroup/cpu.max 2> /dev/null && [ "$q" != max ] && [ "${per:-0}" -gt 0 ] 2> /dev/null; then
        q=$(( (q + per - 1) / per ))
        [ "$q" -ge 1 ] && [ "$q" -lt "$n" ] && n=$q
    elif read -r q < /sys/fs/cgroup/cpu/cpu.cfs_quota_us 2> /dev/null && read -r per < /sys/fs/cgroup/cpu/cpu.cfs_period_us 2> /dev/null \
            && [ "$q" -gt 0 ] 2> /dev/null && [ "${per:-0}" -gt 0 ] 2> /dev/null; then
        q=$(( (q + per - 1) / per ))     # cgroup v1 (the RunPod L40S pods)
        [ "$q" -ge 1 ] && [ "$q" -lt "$n" ] && n=$q
    fi
    echo "$n"
}
cpu_sample() {   # "<wall_usec> <cpu_usec> cgroup" or "<total ticks> <busy ticks> <cpus> procstat"
    local u
    u=$(awk '$1 == "usage_usec" {print $2}' /sys/fs/cgroup/cpu.stat 2> /dev/null)
    # %.0f, not %d: mawk's %d is 32-bit, so cpuacct.usage (ns since the container started) / 1000 stuck at
    # 2147483647 after about 36 CPU-minutes and every later sample read busy=0.0 (L40S nvc1, 2026-11-14).
    [ -n "$u" ] || u=$(awk '{printf "%.0f", $1 / 1000}' /sys/fs/cgroup/cpuacct/cpuacct.usage 2> /dev/null)
    if [ -n "$u" ]; then echo "$(( $(date +%s%N) / 1000 )) $u cgroup"; return; fi
    awk '/^cpu / {b=$2+$3+$4+$7+$8+$9; t=b+$5+$6} /^cpu[0-9]/ {c++} END {printf "%.0f %.0f %d procstat\n", t, b, c}' /proc/stat
}
busy_line() {   # <allocated cores> <phase> <previous cpu_sample> <current cpu_sample>: one cores.log line
    echo "$3 $4" | awk -v n="$1" -v t="$(date -u +%H:%M:%S)" -v ph="$2" '
        { if ($3 == "cgroup" && $6 == "cgroup") { dw = $4 - $1; dc = $5 - $2; ok = (dw > 0 && dc >= 0); b = ok ? dc / dw : 0; src = "cgroup" }
          else { dt = $5 - $1; db = $6 - $2; ok = (dt > 0 && db >= 0 && db <= dt); b = ok ? db / dt * $7 : 0; src = "procstat" }
          if (!ok) { printf "%s busy=NA total=%d phase=%s source=%s (counter went backwards; next sample from here)", t, n, ph, src; exit }
          c = ""; if (b < 0) { b = 0; c = " clamped" } if (b > n) { b = n; c = " clamped" }
          printf "%s busy=%.1f total=%d phase=%s source=%s%s", t, b, n, ph, src, c }'
}
cores_sampler() {
    local n prev cur i=0 line
    n=$(alloc_cores)
    prev=$(cpu_sample)
    while sleep 60; do
        cur=$(cpu_sample)
        line=$(busy_line "$n" "$(cat "$J/.phase" 2>/dev/null)" "$prev" "$cur")
        echo "$line" >> "$J/cores.log"
        i=$(( i + 1 )); [ $(( i % 5 )) -eq 0 ] && say "BUSY $line"
        prev=$cur
    done
}

# NVIDIA: every GPU must be the paper's (L40S) unless PC_ALLOW_HW_DIFF=1 (2026-11-13: an out-of-stock L40S got a
# 2x RTX 6000 Ada rented; compare on the same device). AMD: HW-DIFF is recorded and the job goes on unless
# PC_REQUIRE_SAME_GPU=1 (the MI325X run of 2026-11-13). tools/paper_compare.py flags every row HW-DIFF either way.
# (nvidia-smi prints one name per GPU, so every line must carry the token; rocm-smi prints headers, so any line)
gpu_other() {
    if [ "$V" = nvidia ]; then [ "$NGPU" -eq 0 ] || grep -v -q "$TOKEN" "$J/gpu.txt"
    else ! grep -q "$TOKEN" "$J/gpu.txt"; fi
}

# upload leftovers of finished box-persisted jobs: a killed upload leaves its piece (at most max_bytes since
# 2026-11-14) or, from an older runner, its whole staging tarball (36 GB on the L40S pod, 2026-11-13); this job's
# own and running jobs' are kept
prune_persist_staging() {
    local root f tag pid
    root=$(dirname "$(dirname "${MOJOLEARN_PERSIST_RUNNER:-/root/box-persist/bin/x}")")
    for f in "$root"/state/*/upload.tgz "$root"/state/*/readback.tgz "$root"/state/*/piece "$root"/state/*/piece.back; do
        [ -f "$f" ] || continue
        tag=$(basename "$(dirname "$f")")
        [ "$tag" = "${MOJOLEARN_PERSIST_TAG:-}" ] && continue
        pid=$(sed -n 's/^PID=//p' "$root/active/$tag.env" 2> /dev/null | head -1)
        [ -n "$pid" ] && kill -0 "$pid" 2> /dev/null && continue
        say "pruning $(du -sh "$f" 2> /dev/null | cut -f1) upload leftover of finished job $tag: $f"
        rm -f "$f" "$(dirname "$f")/upload.sha256" "$f.sha256"
    done
}

# ---- the host probe (every submit, before anything else uses the GPU) -----------------------------------------
# AGENTS.md "Times are compared only on the same box" (2026-11-14): what this box is (tools/ops/host_probe.py) in
# job/host-probe.json and job/box.txt; every board stores it (bench_board reads MOJOLEARN_HOST_PROBE: box.host_probe
# and each race's host.host_id) and tools/paper_compare.py refuses to compare boards of two host ids.
host_probe_phase() {
    local rc PREV_HOST
    phase probe
    PREV_HOST=$(sed -n 's/^ *"host_id": "\(.*\)",*$/\1/p' "$J/host-probe.json" 2> /dev/null | head -n 1)
    $PY tools/ops/host_probe.py --vendor "$V" --out "$J/host-probe.json.new" > "$RES/logs/host-probe.log" 2>&1
    rc=$?
    HOST_ID=$(sed -n 's/^ *"host_id": "\(.*\)",*$/\1/p' "$J/host-probe.json.new" 2> /dev/null | head -n 1)
    if [ $rc -eq 0 ] && [ -n "$HOST_ID" ]; then
        if [ -n "$PREV_HOST" ] && [ "$PREV_HOST" != "$HOST_ID" ]; then
            # a split submit on another box: its boards carry the new host id and the compare refuses to mix them
            say "HOST CHANGED: an earlier submit of this results dir ran on $PREV_HOST, this one on $HOST_ID; boards" \
                "resumed here mix hosts and the compare will refuse them (start a new results dir)"
            cp "$J/host-probe.json" "$J/host-probe-earlier.json" 2> /dev/null
        fi
        mv "$J/host-probe.json.new" "$J/host-probe.json"
        { echo "== host probe (tools/ops/host_probe.py)"; cat "$J/host-probe.json"; } >> "$J/box.txt"
        export MOJOLEARN_HOST_PROBE="$J/host-probe.json"
        say "host probe: host_id $HOST_ID; $(grep -E '"(memcpy_gbps|h2d_pageable_gbps|h2d_pinned_gbps|cpu_model|numa_nodes)"' "$J/host-probe.json" | tr -d ' \n')"
    else
        # no probe: the boards carry no host id and the compare gives no verdict (NOT-COMPARED), never a cross-host flag
        say "host probe FAILED (logs/host-probe.log): this job's boards carry no host id; the compare gives no verdict"
        [ $rc -eq 0 ] && rc=7
    fi
    phase_end $rc
}
