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

# job/phases.tsv: <phase> <rc> <seconds> <end UTC> <state>. State (fb-harness, 2026-11-17: a 240-minute cap cut every
# full-board arm and the next submit resumed it with no line saying so): `done` (phase_end), `refused` (the disk
# gate), `partial <done>/<total> <board>` (the job ended inside the phase: the cap's TERM, a crash; phase_partial from
# the job's EXIT trap; the next submit resumes it from board.json), `restart` (host_probe_phase: the boot id changed
# since the last submit). A phase that starts after its own `partial` line says "resuming".
PHASE_OPEN=0
phase() {
    local last
    PHASE=$1; P_T0=$(date +%s); PHASE_OPEN=1; echo "$1" > "$J/.phase"
    last=$(awk -F'\t' -v p="$1" '$1 == p {s = $5} END {print s}' "$J/phases.tsv" 2> /dev/null)
    case "$last" in
        partial*) say "phase $PHASE start: resuming (an earlier submit ended inside it: $last)" ;;
        *) say "phase $PHASE start" ;;
    esac
}
phase_end() {   # rc
    local rc=$1 full
    full=$(disk_full_in_phase)
    [ -n "$full" ] && rc=$EXIT_DISK
    printf '%s\t%s\t%s\t%s\tdone\n' "$PHASE" "$rc" "$(( $(date +%s) - P_T0 ))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$J/phases.tsv"
    PHASE_OPEN=0
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
# the job's EXIT trap calls this: a phase still open when the job ends (the cap's TERM, a crash) gets its `partial`
# line, with the timed board's done/total when one was running (CUR_IDS, CUR_BOARD: board_progress)
phase_partial() {   # rc
    [ "${PHASE_OPEN:-0}" = 1 ] || return 0
    PHASE_OPEN=0
    printf '%s\t%s\t%s\t%s\tpartial%s\n' "$PHASE" "$1" "$(( $(date +%s) - P_T0 ))" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        "$(board_progress | sed 's/^./ &/')" >> "$J/phases.tsv"
    say "phase $PHASE ended partial (rc=$1)$(board_progress | sed 's/^./: &/'); the next submit resumes it"
}
# "<done>/<total> <board dir>" of the timed board that is running (CUR_IDS: its race ids; CUR_BOARD: its out dir), or
# nothing. Done is counted from its board.json (tools/paper_compare.py left).
board_progress() {
    local total left
    [ -n "${CUR_IDS:-}" ] && [ -s "${CUR_IDS:-}" ] || return 0
    total=$(nlines "$CUR_IDS")
    left=$(races_left "$CUR_IDS" "$CUR_BOARD/board.json")
    echo "$(( total - left ))/$total $(basename "$CUR_BOARD")"
}
# how many of the race ids in <ids> are not yet done in <board.json> (all of them without a readable board): what a
# resumed timed board still runs, so what its disk gate counts
races_left() {   # ids board.json
    $PY tools/paper_compare.py left --ids "$1" --board "$2" 2> /dev/null || nlines "$1"
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
        printf '%s\t%s\t%s\t%s\trefused\n' "$what-disk-refused" "$EXIT_DISK" 0 "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$J/phases.tsv"
        PHASE_OPEN=0
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
probe_field() {   # <probe json> <top-level string field>: its value, or nothing
    sed -n 's/^ *"'"$2"'": "\(.*\)",*$/\1/p' "$1" 2> /dev/null | head -n 1
}
# THE ARITHMETIC GATE (lane fb13-identity-audit, 2026-11-17): checks/ieee_arith_check.mojo on this box's GPU before
# any timing. It judges, bit for bit over 2^20 patterns, the operations IDENTICAL runs (portable_sqrtf, portable_divf,
# raw division, the signed-zero table) against exact references; a box whose backend diverges stops here (exit 3)
# instead of timing a column whose bits are not the contract's. Cost: one `mojo build` of the check alone (it imports
# only checks/numerics; the binary is cached per commit under CACHE_ROOT, so a resumed submit rebuilds nothing) plus
# its run, 2 s on the L40S (native-checks run of d652a4b0a1, 2026-11-17). Both are in phases.tsv as phase `arith`.
# No Mojo toolchain on the box: recorded as rc 2 (skipped) and the job goes on; it never passes silently.
arith_gate() {
    local rc bin commit
    phase arith
    commit=$(git rev-parse --short=12 HEAD 2> /dev/null || echo unknown)
    bin="${CACHE_ROOT:-$RES}/ieee-arith-$commit"
    if [ ! -x "$bin" ]; then
        if ! timeout 120 pixi run mojo --version > "$RES/logs/arith-build.log" 2>&1; then
            say "arith gate SKIPPED: no Mojo toolchain on this box (logs/arith-build.log)"
            phase_end 2
            return 0
        fi
        timeout 1200 pixi run mojo build -D MOJOLEARN_NUMERIC_IDENTICAL=1 -I . -o "$bin" checks/ieee_arith_main.mojo \
            >> "$RES/logs/arith-build.log" 2>&1
        rc=$?
        if [ "$rc" != 0 ]; then
            say "arith gate build failed rc=$rc (logs/arith-build.log); STOP before timing"
            phase_end 3
            exit 3
        fi
    fi
    timeout 600 pixi run "$bin" > "$RES/logs/arith.log" 2>&1
    rc=$?
    if [ "$rc" != 0 ]; then
        say "arith gate FAILED rc=$rc: this box diverges on an identical-path operation (logs/arith.log:"
        grep -m 3 -i 'fail\|diverg\|mismatch\|abort' "$RES/logs/arith.log" | cut -c1-200 | sed 's/^/    /'
        say "STOP before timing"
        phase_end 3
        exit 3
    fi
    say "arith gate passed ($(grep -c . "$RES/logs/arith.log") log lines)"
    phase_end 0
}
# THE FAULT GATE (lane fx-coverage-smokes, 2026-11-20): once per job, on this box's GPU, that the checks can still
# FAIL: `verify --tamper-test` from the candidate's installed wheel and the cheap registered sabotage checks
# (tools/ops/gpu_fault_gate.py: each must report its injected fault DETECTED; one that passes silently fails the
# unit). Untimed; it runs after the candidate's install and before any smoke or board, so the GPU is idle. Its line is
# FAULTS_LINE (job.log and status.json `faults`), its file $RES/FAULTS-$V.txt, its logs $RES/faults/. Fail soft: a
# failure is phase `faults` rc 1 in phases.tsv (the job's worst rc) and the job goes on. A resubmit into the same
# results dir does not run it again (faults/faults.done). RELEASE ONLY (AGENTS.md hard rule "RELEASE-ONLY WORK",
# 2026-11-26: sabotage arms are built and run once, at the release): off unless PC_FAULTS=1; PC_FAULT_ARMS names other pixi
# sabotage tasks. Cost: ESTIMATE ~6-9 pod minutes the first time per commit (builds cached under CACHE_ROOT).
# Usage: fault_gate <candidate venv python>
FAULTS_LINE=
fault_gate() {
    local rc
    [ "${PC_FAULTS:-0}" = 1 ] || { FAULTS_LINE="FAULTS: off (release only; PC_FAULTS=1 runs it)"; say "$FAULTS_LINE"; return 0; }
    if [ -s "$RES/faults/faults.done" ]; then
        FAULTS_LINE=$(head -n 1 "$RES/FAULTS-$V.txt" 2> /dev/null)
        say "fault gate: ran earlier in this results dir: ${FAULTS_LINE:-no FAULTS-$V.txt}"
        return 0
    fi
    phase faults
    mkdir -p "$RES/faults"
    ${PY:-python3} tools/ops/gpu_fault_gate.py --python "$1" --out "$RES/faults" --summary "$RES/FAULTS-$V.txt" \
        --vendor "$V" --tree "$(pwd)" --cache "${CACHE_ROOT:-$RES}/fault-bins" ${PC_FAULT_ARMS:+--arms "$PC_FAULT_ARMS"} \
        > "$RES/logs/faults.log" 2>&1
    rc=$?
    FAULTS_LINE=$(head -n 1 "$RES/FAULTS-$V.txt" 2> /dev/null)
    [ -n "$FAULTS_LINE" ] || FAULTS_LINE="FAULTS $V: the unit wrote no summary (rc=$rc, logs/faults.log)"
    say "$FAULTS_LINE"
    date -u +%Y-%m-%dT%H:%M:%SZ > "$RES/faults/faults.done"
    phase_end $rc
}
host_probe_phase() {
    local rc PREV_HOST PREV_BOOT BOOT_ID
    phase probe
    PREV_HOST=$(probe_field "$J/host-probe.json" host_id)
    PREV_BOOT=$(probe_field "$J/host-probe.json" boot_id)
    $PY tools/ops/host_probe.py --vendor "$V" --out "$J/host-probe.json.new" > "$RES/logs/host-probe.log" 2>&1
    rc=$?
    HOST_ID=$(probe_field "$J/host-probe.json.new" host_id)
    if [ $rc -eq 0 ] && [ -n "$HOST_ID" ]; then
        if [ -n "$PREV_HOST" ] && [ "$PREV_HOST" != "$HOST_ID" ]; then
            # a split submit on another box: its boards carry the new host id and the compare refuses to mix them
            say "HOST CHANGED: an earlier submit of this results dir ran on $PREV_HOST, this one on $HOST_ID; boards" \
                "resumed here mix hosts and the compare will refuse them (start a new results dir)"
            cp "$J/host-probe.json" "$J/host-probe-earlier.json" 2> /dev/null
        fi
        BOOT_ID=$(probe_field "$J/host-probe.json.new" boot_id)
        if [ -n "$PREV_BOOT" ] && [ -n "$BOOT_ID" ] && [ "$PREV_BOOT" != "$BOOT_ID" ]; then
            # RESTART (fb-harness, 2026-11-17): the d95f805cce pod restarted between its arms (NVIDIA driver 595.91.07
            # -> 595.112.07, kernel 6.8.0-147 -> 149) and the compare gave verdicts across it. The earlier probe is
            # kept (host-probe-boot-<id8>.json), phases.tsv says so, and each race's record carries its boot id
            # (bench_board race_host), so the rows timed before and after are told apart; tools/paper_compare.py
            # gives no verdict across two system states
            cp "$J/host-probe.json" "$J/host-probe-boot-${PREV_BOOT:0:8}.json" 2> /dev/null
            printf '%s\t0\t0\t%s\trestart boot %s -> %s; driver %s -> %s; kernel %s -> %s\n' restart \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$PREV_BOOT" "$BOOT_ID" \
                "$(grep -m 1 '"driver"' "$J/host-probe.json" | sed 's/.*: "\(.*\)".*/\1/')" \
                "$(grep -m 1 '"driver"' "$J/host-probe.json.new" | sed 's/.*: "\(.*\)".*/\1/')" \
                "$(probe_field "$J/host-probe.json" kernel)" "$(probe_field "$J/host-probe.json.new" kernel)" >> "$J/phases.tsv"
            say "RESTART: the box rebooted since the last submit (boot id $PREV_BOOT -> $BOOT_ID; see phases.tsv);" \
                "boards resumed here carry both boot ids and the compare gives no verdict across them"
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

# ---- NVIDIA profilers (tools/ops/attribution_job.sh, tools/ops/call_profile_job.sh) ----------------------------------
find_tool() {   # name: the first executable of that name on PATH or in the usual NVIDIA install dirs
    local c
    for c in "$(command -v "$1" 2> /dev/null)" /usr/local/cuda/bin/"$1" /opt/nvidia/nsight-systems/*/bin/"$1" \
            /opt/nvidia/nsight-compute/*/"$1" /usr/local/cuda/nsight-systems-*/bin/"$1" /usr/local/cuda/nsight-compute-*/"$1" \
            /usr/lib/nsight-systems/bin/"$1" /usr/lib/nsight-compute/"$1"; do
        [ -n "$c" ] && [ -x "$c" ] && { echo "$c"; return 0; }
    done
    return 1
}
apt_install_first() {   # log, package patterns...: install the first that apt knows (exact name, else the newest match)
    local log=$1 p name; shift
    for p in "$@"; do
        if [[ "$p" == *'*'* ]]; then
            name=$(apt-cache search --names-only "^${p//\*/.*}\$" 2> /dev/null | cut -d' ' -f1 | sort -V | tail -n 1)
        else name=$p; fi
        [ -n "$name" ] || continue
        DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "$name" >> "$log" 2>&1 && { echo "$name"; return 0; }
    done
    return 1
}
install_nsight() {   # nsys and ncu by apt when the pod has neither; the CUDA keyring/repository added when apt knows none
    # (uses $RES/logs and records how each came in $J/tools-how.txt)
    local log="$RES/logs/apt-nsight.log" distro got
    if find_tool nsys > /dev/null && find_tool ncu > /dev/null; then return 0; fi
    DEBIAN_FRONTEND=noninteractive apt-get update -qq >> "$log" 2>&1
    if ! apt-cache policy 2> /dev/null | grep -q developer.download.nvidia.com; then
        distro=$(. /etc/os-release 2> /dev/null; echo "ubuntu${VERSION_ID//./}")
        curl -fsSL -o /root/cuda-keyring.deb \
            "https://developer.download.nvidia.com/compute/cuda/repos/$distro/x86_64/cuda-keyring_1.1-1_all.deb" >> "$log" 2>&1 \
            && dpkg -i /root/cuda-keyring.deb >> "$log" 2>&1 && DEBIAN_FRONTEND=noninteractive apt-get update -qq >> "$log" 2>&1
    fi
    if ! find_tool nsys > /dev/null; then
        got=$(apt_install_first "$log" nsight-systems-cli 'nsight-systems-[0-9]*' 'cuda-nsight-systems-*') && echo "nsys apt $got" >> "$J/tools-how.txt"
    fi
    if ! find_tool ncu > /dev/null; then
        got=$(apt_install_first "$log" 'nsight-compute-[0-9]*' 'cuda-nsight-compute-*') && echo "ncu apt $got" >> "$J/tools-how.txt"
    fi
    return 0
}
