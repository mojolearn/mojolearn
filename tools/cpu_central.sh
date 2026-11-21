#!/bin/bash
# tools/cpu_central.sh -- THE CPU CHECK BOX (lane fx-cpu-box-checks, 2026-11-19). Andrew: "why are we waiting on
# things? we use runpod?": with six lanes checking at once a native-checks run waited 40 minutes for free GitHub
# runners. One rented RunPod CPU pod (no GPU, key cpc1) runs the same checks for any pushed commit in minutes, every
# group at once across its cores (tools/ops/cpu_box_checks_job.sh).
#
#   tools/cpu_central.sh up [minutes]          ORCHESTRATOR ONLY (MOJOLEARN_ORCHESTRATOR=1). RENTS one CPU pod: the
#                                              instance with the most vCPUs for the price from RunPod's CPU catalog (at
#                                              least CPC_MIN_VCPU=32 vCPU and CPC_MIN_RAM_PER_VCPU=8 GB a vCPU; model and
#                                              price in the log; a pinned MOJOLEARN_CPC_VCPU outside a flavor's catalog
#                                              range is refused before any provider call), refused when tools/rent_guard.sh (the account cap the
#                                              other tools use) refuses or the balance is under CPC_MIN_BALANCE. The
#                                              fixed watchdog covers [minutes] (default 120) until the queue's lease
#                                              takes over: the pod deletes itself CPC_IDLE_MIN (120) idle minutes after
#                                              its last job; the Mac watcher deletes it when the pod stops answering.
#                                              Installs pixi and seeds the tools tree (/root/mojolearn) at CPC_TOOLS_REF
#                                              (default: this checkout's HEAD; it must be pushed).
#   tools/cpu_central.sh check <sha> [parse|native|tools|all] [--filter GLOBS] [--cap MIN] [--timeout S]
#                                              seeds <sha> on the box (from GitHub; shipped from this checkout when
#                                              GitHub refuses the box) and queues tools/ops/cpu_box_checks_job.sh under
#                                              tools/box_persist_run.sh; prints the job id (cpc1-NNNN) and its R2 tag.
#                                              Several checks run at once (slots from the box's memory, below; the units
#                                              of all of them share one memory ledger on the box and WAIT for memory,
#                                              never skipped). Lanes may call this on a box that is up. With two boxes
#                                              up it goes to the one with fewer active checks per slot (CPC_BOX pins).
#   tools/cpu_central.sh status [<id>]         the pods, their leases and queues; or one job
#   tools/cpu_central.sh log <id> [N]          last N lines (default 40) of a job's log
#   tools/cpu_central.sh summary <id>          the job's SUMMARY.md on the box (the same lines tools/box_persist.py fetch gets),
#                                              then each part's verdict on its own line, so a compile result reads
#                                              apart from red tool tests (compile = build-cpu and the parse groups):
#                                                COMPILE <id> <sha> <suite>: green (N) | red: ... | not in this suite
#                                                NATIVE  <id> <sha> <suite>: ...
#                                                TOOLS   <id> <sha> <suite>: ...
#                                              then ONE line a lane decides on:
#                                                RESULT <id> <sha> <suite>: green | red: ... | pending (<status>)
#                                              exit 0 only when green (every check ran and passed), 1 red, 2 pending or
#                                              no summary
#   tools/cpu_central.sh cancel <id>           cancel a queued or running check
#   tools/cpu_central.sh sh <command>          a command on the box (keeps the lease busy while it runs)
#   tools/cpu_central.sh tools <sha>           ORCHESTRATOR ONLY: move the box's tools tree to <sha>
#   tools/cpu_central.sh down [--force]        ORCHESTRATOR ONLY: archive the box home and to R2, then delete it
#                                              (refuses while checks are queued or running unless --force)
#
# ONE COPY OF THE MACHINERY. This file SOURCES tools/nvidia_central.sh (its library mode) for the queue install and
# lease, the Mac watcher, reconcile, the balance read and submit, and calls tools/dev_pod.sh (up --vendor cpu, seed,
# down) for the dead-man, the create, the on-pod watchdog, the git seeding, the archive and the verified delete.
# NEVER AN APPLE MACHINE (AGENTS.md hard rule): only RunPod Linux CPU flavors are accepted (runpod_gpu_request.py cpu).
#
# TWO BOXES. CPC_BOX=cpc1 (default) or cpc2 names the box a command acts on (`up`, `down`, `tools`, `status`); a job id
# cpcN-NNNN names its own box. Each box has its own state dir ($MOJOLEARN_DEVPOD_STATE/cpcN), dead-man, Mac watcher,
# queue lease and lane tree (cpuchk on cpc1, cpuchk2 on cpc2). `up` never rents a second box unless CPC_BOX=cpc2.
#
# Parallel plan: one box of >= 32 vCPU at CPC_MIN_RAM_PER_VCPU=8 GB a vCPU (2026-11-20: on 32 vCPU / 64 GB the
# memory-admitted parse groups, 5-15 GiB each, kept 3.5 of 32 cores busy). Check jobs at once (queue slots) = the box's
# memory budget (85 percent) / CPC_GIB_PER_SLOT (64 GiB, one job's build plus a few parse groups), at least 1; CPC_SLOTS
# pins it. Inside a job every parse group, the checks build, the check slices and the tool tests start as memory allows
# (measured peaks per group, $CPU_CHECKS_HOME/peaks.tsv on the box), each multi-threaded on every core; units of all
# jobs share one ledger and wait for memory. Each job's log states busy cores of total every minute with the reason
# when below 80 percent, and its SUMMARY.md the busy cores of total.
#
# Tests (no rental): tools/test_cpu_central.sh.
set -euo pipefail
_CPC_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MOJOLEARN_NVC_HOME="${MOJOLEARN_CPC_HOME:-$HOME/mojolearn-evidence/cpu_central}"
export MOJOLEARN_NVC_CONF="${MOJOLEARN_CPC_CONF:-/dev/null}"   # the NVIDIA pods' settings file does not apply here
CPC_IDLE_MIN=${CPC_IDLE_MIN:-120}
export NVC_IDLE_MIN=$CPC_IDLE_MIN                     # the queue's lease: deleted this many idle minutes after its last job
export NVC_CAP_MIN=${CPC_CAP_MIN:-180}                # a check job's default and maximum wall-clock cap
# shellcheck source=tools/nvidia_central.sh
. "$_CPC_ROOT/tools/nvidia_central.sh"
ROOT=$_CPC_ROOT
# every RunPod call for the CPU box is API v2 (api.runpod.io/v2): dev_pod.sh creates through it, and reconcile and the
# Mac watcher read and delete through it (v1's host rest.runpod.io stopped resolving 2026-11-20, lane fx-cpu-pod-v2)
rp_select_api 2
die() { echo "cpu_central: $*" >&2; exit 1; }
say() { echo "cpu_central: $*"; }
CPC_MAX_BOXES=2
set_box() {  # <cpcN>: CPC_KEY and its queue lane (cpc1: cpuchk, tree /root/mojolearn-cpuchk -> /root/mojolearn; cpcN: cpuchkN)
    [[ "$1" =~ ^cpc([1-9])$ ]] && [ "${BASH_REMATCH[1]}" -le "$CPC_MAX_BOXES" ] || die "a CPU check box is cpc1..cpc$CPC_MAX_BOXES, not '$1'"
    CPC_KEY=$1; CPC_LANE=cpuchk; [ "$1" = cpc1 ] || CPC_LANE=cpuchk${1#cpc}
}
set_box "${CPC_BOX:-cpc1}"
CPC_SLOTS=${CPC_SLOTS:-}                              # empty: from the box's memory (slots_for_kb)
CPC_GIB_PER_SLOT=${CPC_GIB_PER_SLOT:-64}
CPC_MIN_VCPU=${CPC_MIN_VCPU:-32}
CPC_MIN_RAM_PER_VCPU=${CPC_MIN_RAM_PER_VCPU:-8}
CPC_MIN_BALANCE=${CPC_MIN_BALANCE:-10}
CPC_DISK_GB=${CPC_DISK_GB:-150}
CPC_HOME_BOX=/root/cpu-checks
pod_keys() { local i; for i in $(seq 1 "$CPC_MAX_BOXES"); do echo "cpc$i"; done; }
QUEUE_VENDOR=cpu
slots_for_kb() {  # <MemTotal kB>: check jobs at once = 85 percent of the memory / CPC_GIB_PER_SLOT, at least 1
    [ -z "$CPC_SLOTS" ] || { echo "$CPC_SLOTS"; return; }
    awk -v kb="$1" -v g="$CPC_GIB_PER_SLOT" 'BEGIN {n = int(kb * 0.85 / 1048576 / g); if (n < 1) n = 1; if (n > 16) n = 16; print n}'
}
queue_slots() {  # <key>: from the box's memory (the same allocation the job reads: MemTotal capped by the cgroup)
    local kb
    kb=$(BOX_TIMEOUT=120 box "kb=\$(awk '/MemTotal/ {print \$2}' /proc/meminfo); m=\$(cat /sys/fs/cgroup/memory.max 2> /dev/null || echo max); [ \"\$m\" = max ] || [ \$(( m / 1024 )) -ge \$kb ] || kb=\$(( m / 1024 )); echo \$kb" < /dev/null | tr -dc 0-9)
    [ -n "$kb" ] || [ -n "$CPC_SLOTS" ] || { echo 1; return; }
    slots_for_kb "${kb:-0}"
}

# Nothing Apple, from any input (the provider is RunPod Linux only; runpod_gpu_request.py cpu refuses again).
refuse_apple() {
    local v
    for v in "$@"; do
        if grep -qiE 'apple|macos|darwin|osx' <<< "$v"; then
            die "refused: '$v' names an Apple machine. Never an Apple machine without Andrew's express permission (AGENTS.md hard rule); nothing rented"
        fi
    done
}
refuse_apple "${MOJOLEARN_CPC_FLAVORS:-}" "${MOJOLEARN_DEVPOD_CPU_IMAGE:-}" "${CPC_PROVIDER:-runpod}"
[ "${CPC_PROVIDER:-runpod}" = runpod ] || die "CPC_PROVIDER must be runpod (the only CPU box provider); nothing rented"
[ -z "$CPC_SLOTS" ] || [[ "$CPC_SLOTS" =~ ^[1-9][0-9]?$ ]] || die "CPC_SLOTS must be 1..99"
[[ "$CPC_GIB_PER_SLOT" =~ ^[1-9][0-9]*$ ]] || die "CPC_GIB_PER_SLOT must be a whole number of GiB"

need_box() {  # the box is up, its lane file names it
    reconcile
    [ -f "$STATE_ROOT/$CPC_KEY/state.env" ] || die "the CPU check box is not up ($CPC_KEY). Only the orchestrator brings it up (tools/cpu_central.sh up); a lane reports that and uses the GitHub runners"
    mkdir -p "$NVC_HOME/lanes/$CPC_LANE"; echo "$CPC_KEY" > "$NVC_HOME/lanes/$CPC_LANE/pod"
    pod_env "$CPC_KEY"
}
job_id() {  # <id>: cpcN-NNNN (that box) or NNNN (CPC_BOX, default cpc1) -> JOB_N, CPC_KEY
    case "$1" in
        cpc[1-9]-[0-9]*)
            [ -z "${CPC_BOX:-}" ] || [ "$CPC_BOX" = "${1%%-*}" ] || die "job $1 is on ${1%%-*}, not CPC_BOX=$CPC_BOX"
            set_box "${1%%-*}"; JOB_N=${1#*-} ;;
        [0-9]*) JOB_N=$1 ;;
        *) die "a job id is $CPC_KEY-<n>, e.g. $CPC_KEY-0003" ;;
    esac
    [[ "$JOB_N" =~ ^[0-9]{4,}$ ]] || die "a job id is $CPC_KEY-<n>, e.g. $CPC_KEY-0003"
}
choose_box() {  # `check` without CPC_BOX: the live box with the fewest queued and running checks per slot
    local k best='' bestv='' v
    [ -z "${CPC_BOX:-}" ] || return 0
    reconcile
    for k in $(live_pods); do
        pod_env "$k"
        v=$(BOX_TIMEOUT=60 box "a=\$(grep -lE '^(queued|starting|running)\$' $QD/[0-9]*/status 2> /dev/null | wc -l); s=\$(cat $QD/slots 2> /dev/null || echo 1); echo \$(( a * 1000 / (s > 0 ? s : 1) ))" < /dev/null | tr -dc 0-9) || v=''
        [ -n "$v" ] || continue
        if [ -z "$best" ] || [ "$v" -lt "$bestv" ]; then best=$k; bestv=$v; fi
    done
    [ -z "$best" ] || set_box "$best"
}
pick_instance() {  # -> PICK_FLAVORS PICK_VCPU, from the catalog (GraphQL cpuFlavors) or the fallback order
    local cat="$TMPD/cpu_catalog.json"
    if [ -n "${MOJOLEARN_CPC_CATALOG:-}" ]; then cp "$MOJOLEARN_CPC_CATALOG" "$cat"
    elif [ "$NO_API" != 1 ]; then
        curl -s -m 30 -K "$CURLRC" -H 'Content-Type: application/json' https://api.runpod.io/graphql \
            -d '{"query":"query { cpuFlavors { id groupId displayName minVcpu maxVcpu ramMultiplier } }"}' > "$cat" 2> /dev/null || : > "$cat"
    else : > "$cat"; fi
    cp "$cat" "$NVC_HOME/cpu_catalog-$(date -u +%Y%m%dT%H%M%SZ).json" 2> /dev/null || true
    # the pins (MOJOLEARN_CPC_FLAVORS, MOJOLEARN_CPC_VCPU) replace the pick and are refused here, before any provider
    # call, when a flavor's catalog range does not take the vCPU count (cpu5m: 2..32)
    local out; out=$(python3 "$ROOT/tools/runpod_gpu_request.py" cpu-pick "$cat" "$CPC_MIN_VCPU" "$CPC_MIN_RAM_PER_VCPU" \
        "${MOJOLEARN_CPC_FLAVORS:--}" "${MOJOLEARN_CPC_VCPU:--}") || die "nothing rented (above)"
    PICK_FLAVORS=${out% *}; PICK_VCPU=${out#* }
    [[ "$PICK_VCPU" =~ ^[0-9]+$ ]] && [ "$PICK_VCPU" -ge "$CPC_MIN_VCPU" ] || die "vCPU count '$PICK_VCPU' is under CPC_MIN_VCPU=$CPC_MIN_VCPU; nothing rented"
    if [ -s "$cat" ] && grep -q cpuFlavors "$cat"; then PICK_FROM="RunPod's CPU catalog"; else PICK_FROM="the fallback order (no catalog)"; fi
}

summary_verdict() {  # <id> <summary output>: prints the RESULT line; exit 0 green, 1 red, 2 pending or no summary
    local id=$1 f=$2 st head res
    st=$(sed -n 's/^JOBSTATUS //p' "$f" | tail -n 1)
    head=$(sed -n 's/^# CPU box checks \([0-9a-f]*\) suite \([a-z]*\).*/\1 \2/p' "$f" | head -n 1)
    res=$(sed -n 's/^result: //p' "$f" | head -n 1)
    case "${st:-unknown}" in
        queued|starting|running|unknown|'') echo "RESULT $id ${head:-?}: pending (${st:-no answer from the box})"; exit 2 ;;
    esac
    local part label v
    for part in compile native tools; do
        case "$part" in compile) label="COMPILE" ;; native) label="NATIVE " ;; tools) label="TOOLS  " ;; esac
        v=$(sed -n "s/^verdict $part: //p" "$f" | head -n 1)
        echo "$label $id ${head:-?}: ${v:-not in this summary (a job script from before the per-part verdicts; read the parts above)}"
    done
    if [ -z "$res" ]; then
        # a summary from before the not-run line (or none): green only if line 3 says 0 failed and 0 missing/not run
        if grep -qE '^checks: [1-9][0-9]* +passed: [0-9]+ +failed: 0( +not run: 0)? *$' "$f" && [ "$st" = done ]; then res=green
        elif grep -q '^checks: ' "$f"; then res="red: $(grep -m 1 '^checks: ' "$f")"
        else echo "RESULT $id ${head:-?}: no summary (job $st)"; exit 2; fi
    fi
    case "$res" in
        green) [ "$st" = done ] || res="red: job $st although its summary says green" ;;
    esac
    echo "RESULT $id ${head:-?}: $res (job $st)"
    [ "$res" = green ]
    exit $?
}

cmd="${1:-}"; shift || true
case "$cmd" in up|tools|down)
    [ "${MOJOLEARN_ORCHESTRATOR:-0}" = 1 ] || die "only the orchestrator provisions or changes the box: '$cmd' needs MOJOLEARN_ORCHESTRATOR=1. A lane may only 'check' on a box that is up." ;;
esac
case "$cmd" in
up)
    minutes=${1:-120}; [ $# -le 1 ] || die "up [minutes]: unexpected '$*'"
    [[ "$minutes" =~ ^[1-9][0-9]*$ ]] && [ "$minutes" -le 240 ] || die "up [minutes]: 1..240 (the fixed watchdog until the queue's lease takes over)"
    tools_sha=$(git -C "$ROOT" rev-parse --verify "${CPC_TOOLS_REF:-HEAD}^{commit}") || die "no commit ${CPC_TOOLS_REF:-HEAD} for the tools tree"
    [ -n "$(git -C "$ROOT" branch -r --contains "$tools_sha" 2> /dev/null | head -n 1)" ] || [ "${CPC_ALLOW_UNPUSHED:-0}" = 1 ] \
        || die "the tools tree $tools_sha is not on any pushed branch (push it, or CPC_TOOLS_REF=<pushed sha>); nothing rented"
    [ "$NO_API" = 1 ] || load_key || die "no RunPod key (~/.mojolearn_runpod_key, mode 600)"
    # the account cap (tools/rent_guard.sh, every pod on the shared account) before any provider call; dev_pod.sh
    # checks it again before each create attempt
    if [ "$NO_API" != 1 ] || [ -n "${MOJOLEARN_RENT_GUARD_TEST_PODS:-}" ]; then
        bash "$ROOT/tools/rent_guard.sh" || die "tools/rent_guard.sh refused (above); nothing rented"
    fi
    up_lock
    reconcile
    [ ! -f "$STATE_ROOT/$CPC_KEY/state.env" ] || { pod_env "$CPC_KEY"; say "the CPU check box is already up ($POD_ID, $GPU, \$$COST_HR/hr); nothing rented"; exit 0; }
    pick_instance
    say "instance: $PICK_VCPU vCPU of flavors $PICK_FLAVORS (first is preferred), from $PICK_FROM; at least $CPC_MIN_VCPU vCPU and $CPC_MIN_RAM_PER_VCPU GB a vCPU"
    if [ "${MOJOLEARN_CPC_PRINT_REQUEST:-0}" = 1 ]; then
        python3 "$ROOT/tools/runpod_gpu_request.py" cpu "$TMPD/req.json" "mojolearn-dev-$CPC_KEY-print" \
            "${MOJOLEARN_DEVPOD_CPU_IMAGE:-runpod/base:1.3.1-ubuntu2204}" "$PICK_FLAVORS" "$PICK_VCPU" "$CPC_DISK_GB" || exit 1
        echo "request: $(tr -d '\n ' < "$TMPD/req.json")"; exit 0
    fi
    bal=$(balance); [ -n "$bal" ] || die "could not read the RunPod balance; nothing rented"
    awk -v b="$bal" -v m="$CPC_MIN_BALANCE" 'BEGIN{exit !(b >= m)}' \
        || die "RunPod balance is \$$bal, under \$$CPC_MIN_BALANCE (CPC_MIN_BALANCE); nothing rented"
    say "RunPod balance \$$bal; CREATE: MOJOLEARN_DEVPOD_VIA_CENTRAL=1 MOJOLEARN_DEVPOD_CPU_FLAVORS=$PICK_FLAVORS MOJOLEARN_DEVPOD_CPU_VCPU=$PICK_VCPU" \
        "MOJOLEARN_DEVPOD_DISK_GB=$CPC_DISK_GB $DEVPOD up $CPC_KEY $minutes --vendor cpu --base ${tools_sha:0:10}"
    MOJOLEARN_DEVPOD_VIA_CENTRAL=1 MOJOLEARN_DEVPOD_CPU_FLAVORS="$PICK_FLAVORS" MOJOLEARN_DEVPOD_CPU_VCPU="$PICK_VCPU" \
        MOJOLEARN_DEVPOD_DISK_GB="$CPC_DISK_GB" "$DEVPOD" up "$CPC_KEY" "$minutes" --vendor cpu --base "$tools_sha" \
        || die "dev_pod up --vendor cpu failed (above)"
    # the queue (CPC_SLOTS concurrent checks) and its idle lease, the Mac watcher: nvidia_central.sh's install
    ( install_pod "$CPC_KEY" ) || { say "install failed; deleting the box"; MOJOLEARN_NVC_DOWN=1 "$DEVPOD" down "$CPC_KEY"; exit 1; }
    pod_env "$CPC_KEY"
    # dev_pod up seeds the tools tree's index only (a sync checks a lane's tree out); the box runs the job from files
    "$DEVPOD" seed "$CPC_KEY" "$tools_sha" /root/mojolearn --checkout || die "could not check the tools tree out at $tools_sha"
    BOX_TIMEOUT=900 box "set -e; mkdir -p $CPC_HOME_BOX; ln -sfn /root/mojolearn /root/mojolearn-$CPC_LANE
command -v /usr/bin/time > /dev/null || { export DEBIAN_FRONTEND=noninteractive; apt-get update -qq > /dev/null && apt-get install -y -qq time > /dev/null; }
git config --global --add safe.directory '*'
echo BOX nproc=\$(nproc) mem_gib=\$(awk '/MemTotal/ {print int(\$2/1048576)}' /proc/meminfo) cpu=\"\$(grep -m 1 'model name' /proc/cpuinfo | cut -d: -f2 | sed 's/^ *//')\"" \
        < /dev/null > "$TMPD/box.txt" 2>&1 || { cat "$TMPD/box.txt" >&2; die "box setup failed; the box is up (tools/cpu_central.sh down to delete it)"; }
    mkdir -p "$NVC_HOME/lanes/$CPC_LANE"; echo "$CPC_KEY" > "$NVC_HOME/lanes/$CPC_LANE/pod"
    boxline=$(grep '^BOX ' "$TMPD/box.txt" | tail -n 1)
    per=$(awk -v c="$COST_HR" -v n="$(printf '%s' "$boxline" | sed -n 's/.*nproc=\([0-9]*\).*/\1/p')" 'BEGIN{if (c+0 > 0 && n+0 > 0) printf "%.4f", c/n; else print "?"}')
    slots=$(BOX_TIMEOUT=60 box "cat $QD/slots" < /dev/null 2> /dev/null | tr -dc 0-9 || true)
    say "UP: $CPC_KEY pod $POD_ID: $GPU, \$$COST_HR/hr (\$$per per vCPU-hour); ${boxline#BOX }; ${slots:-?} check slots (85 percent of memory / $CPC_GIB_PER_SLOT GiB${CPC_SLOTS:+; pinned CPC_SLOTS=$CPC_SLOTS}); deletes itself $CPC_IDLE_MIN idle minutes after its last check"
    say "check a commit: tools/cpu_central.sh check <sha> [parse|native|tools|all]; then tools/cpu_central.sh summary <id> (exit 0 = green)"
    ;;
check)
    sha_in=${1:?check <sha> [parse|native|tools|all] [--filter GLOBS] [--cap MIN] [--timeout S]}; shift
    suite=all; case "${1:-}" in parse|native|tools|all) suite=$1; shift ;; esac
    jenv=(); cap=$NVC_CAP_MIN
    while [ $# -gt 0 ]; do
        case "$1" in
            --filter)
                # comma separated globs over check names; `*` travels as `%` (tools/ops/native_checks_report.py
                # `select` reads it back). Checked here with the submit's own rule, before any box is touched.
                f="${2:?--filter GLOBS}"; f="NC_FILTER=${f//\*/%}"
                env_kv_ok "$f" || die "--filter ${2}: globs of letters, digits and _ . , : / @ + = - * only (no spaces or quotes)"
                jenv+=(--env "$f"); shift 2 ;;
            --timeout) [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || die "--timeout S"; jenv+=(--env "NC_TIMEOUT=$2"); shift 2 ;;
            --cap) [[ "${2:-}" =~ ^[1-9][0-9]*$ ]] || die "--cap MIN"; cap=$2; shift 2 ;;
            *) die "check <sha> [parse|native|tools|all] [--filter GLOBS] [--cap MIN] [--timeout S]: unexpected '$1'" ;;
        esac
    done
    sha=$(git -C "$ROOT" rev-parse --verify -q "$sha_in^{commit}") || die "no commit $sha_in in this checkout (fetch it first)"
    choose_box
    need_box
    # the commit into the box's source repo (dev_pod.sh seed: GitHub first, else shipped from this checkout); one seed
    # at a time, the repo's index is shared
    t0=$(date +%s)
    until mkdir "$NVC_HOME/seed.lock" 2> /dev/null; do
        [ $(( $(date +%s) - t0 )) -lt 600 ] || die "another check holds $NVC_HOME/seed.lock for 10 minutes"
        sleep 2
    done
    trap 'rmdir "$NVC_HOME/seed.lock" 2> /dev/null; rm -rf "$TMPD"' EXIT
    "$DEVPOD" seed "$CPC_KEY" "$sha" "$CPC_HOME_BOX/src" > "$TMPD/seed.log" 2>&1 || { cat "$TMPD/seed.log" >&2; die "could not seed $sha on the box"; }
    rmdir "$NVC_HOME/seed.lock" 2> /dev/null || true
    id=$(central_submit "$CPC_LANE" --cap "$cap" --note "check ${sha:0:10} $suite" --env "CPU_CHECK_SHA=$sha" \
        --env "CPU_CHECK_SUITE=$suite" ${jenv[@]+"${jenv[@]}"} tools/ops/cpu_box_checks_job.sh)
    # the job's R2 tag: the newest persisted job record naming this id (box_persist_lib.sh bp_note_job)
    tag=''
    hits=$(grep -lF "$id	$CPC_LANE	" "${MOJOLEARN_PERSIST_HOME:-$HOME/mojolearn-evidence/box-persist}"/*/job.txt 2> /dev/null || true)
    # shellcheck disable=SC2086
    [ -z "$hits" ] || tag=$(basename "$(dirname "$(ls -t $hits | head -n 1)")")
    echo "$id"
    say "queued $id on $CPC_KEY: ${sha:0:10} $suite${tag:+; results tag $tag (tools/box_persist.py fetch --tag $tag)}; decide on: tools/cpu_central.sh summary $id (its last line RESULT ...; exit 0 only when green)"
    ;;
status)
    if [ -n "${1:-}" ]; then job_id "$1"; need_box; box "$NVQ status $(sq "$JOB_N")" < /dev/null; exit; fi
    boxes=$CPC_KEY
    if [ -z "${CPC_BOX:-}" ]; then reconcile; boxes=$(live_pods | tr '\n' ' '); [ -n "${boxes// /}" ] || boxes=$CPC_KEY; fi
    for k in $boxes; do
        set_box "$k"; need_box
        echo "== $CPC_KEY: pod $POD_ID, $GPU, \$$COST_HR/hr, ssh $SSH_TARGET"
        echo "watcher: $(tail -1 "$(ls -d "$STATE_ROOT/$CPC_KEY"/watch-* 2>/dev/null | tail -1)/watch.log" 2>/dev/null || echo none)"
        BOX_TIMEOUT=120 box "$NVQ lease-check; echo nproc=\$(nproc) load=\$(cut -d' ' -f1-3 /proc/loadavg) mem_avail_gib=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo) slots=\$(cat $QD/slots 2> /dev/null); $NVQ queue" < /dev/null || echo "(no answer from $CPC_KEY)"
    done
    ;;
log)
    job_id "${1:?log <id> [N]}"; nl="${2:-40}"; [[ "$nl" =~ ^[0-9]+$ ]] || die "N is a line count"
    need_box; box "tail -n $nl $QD/$JOB_N/log" < /dev/null
    ;;
summary)
    job_id "${1:?summary <id>}"; need_box
    # the job's results dir is named in its note ([persist <tag>: <script> -> <dir>], job.env written with %q)
    cat > "$TMPD/summary.sh" <<'SUMMARY_EOF'
j=$1/$2
d=$( (. "$j/job.env"; printf '%s\n' "$NOTE") | sed -n 's/.*-> \(.*\)\]$/\1/p' | head -n 1)
st=$(cat "$j/status" 2> /dev/null)
if [ -n "$d" ] && [ -f "$d/SUMMARY.md" ]; then cat "$d/SUMMARY.md"
else echo "no SUMMARY.md yet in ${d:-?} (job ${st:-?})"; tail -n 5 "$j/log"; fi
echo "JOBSTATUS ${st:-unknown}"
SUMMARY_EOF
    box "bash -s -- $(sq "$QD") $(sq "$JOB_N")" < "$TMPD/summary.sh" > "$TMPD/summary.out" 2>&1 || true
    grep -v '^JOBSTATUS ' "$TMPD/summary.out" || true
    summary_verdict "$CPC_KEY-$JOB_N" "$TMPD/summary.out"
    ;;
cancel)
    job_id "${1:?cancel <id>}"; need_box; box "$NVQ cancel $(sq "$CPC_LANE") $(sq "$JOB_N")" < /dev/null
    ;;
sh)
    [ $# -gt 0 ] || die "sh needs a command"
    need_box
    BOX_TIMEOUT=14400 box "$NVQ hold $(sq "export PATH=/root/.pixi/bin:\$PATH; cd /root/mojolearn && bash -c $(sq "$*")")"
    ;;
tools)
    sha=$(git -C "$ROOT" rev-parse --verify -q "${1:?tools <sha>}^{commit}") || die "no commit $1 in this checkout"
    need_box
    "$DEVPOD" seed "$CPC_KEY" "$sha" /root/mojolearn --checkout
    ;;
down)
    force=0; for a in "$@"; do case "$a" in --force) force=1 ;; *) die "down [--force]" ;; esac; done
    [ "$NO_API" = 1 ] || load_key || die "no RunPod key"
    need_box
    busy=$(BOX_TIMEOUT=60 box "grep -lx queued $QD/[0-9]*/status 2>/dev/null | wc -l; grep -lE '^(starting|running)\$' $QD/[0-9]*/status 2>/dev/null | wc -l" < /dev/null | tr '\n' ' ' || true)
    set -- $busy
    if [ "${1:-0}" != 0 ] || [ "${2:-0}" != 0 ]; then
        [ $force = 1 ] || die "$CPC_KEY has ${1:-0} queued and ${2:-0} running check(s); it goes down by itself $CPC_IDLE_MIN min after they finish (or pass --force)"
    fi
    # dev_pod.sh down: the logs and results home and to R2 first (tools/box_archive.sh), then the verified delete
    MOJOLEARN_NVC_DOWN=1 "$DEVPOD" down "$CPC_KEY"
    rm -f "$NVC_HOME/lanes/$CPC_LANE/pod"
    ;;
*) sed -n 2,37p "$0"; exit 2 ;;
esac
