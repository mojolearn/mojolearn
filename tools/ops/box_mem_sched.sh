# shellcheck shell=bash
# tools/ops/box_mem_sched.sh -- sourced (bash), never run. The memory-admitted unit scheduler of a rented box job,
# shared by every job on the box through one ledger (lane fx-cpu-box-wait, 2026-11-20).
#
# WHY. On 2026-11-20 the CPU check box (32 vCPU / 64 GB) ran job cpc1-0001 while six more checks were queued. Each of
# them ended within a minute with every unit "no result" and `failed 1`: the scheduler loop died on its first minute
# report (`${!PID[*]}` of an empty associative array under `set -u`, cpu_box_checks_job.sh:219 at fde1016eb2) while its
# units were still waiting for memory that cpc1-0001 held. A unit that does not fit now WAITS; the job ends only when
# every unit has run, or was refused as "cannot fit: needs N GiB of M" (its estimate is above the whole allocation).
#
# ADMISSION. Every job on the box shares $SCHED_HOME/ledger (one file per running unit: "<unit pid> <estimate kB>";
# a dead pid's entry is dropped) under $SCHED_HOME/ledger.lock. A unit starts when
#     reserved (every live ledger entry) + held (reservations ahead of it) + its estimate <= BUDGET_KB
# A unit larger than BUDGET_KB (but within TOTAL_KB) starts only when nothing is reserved and nothing is held ahead.
# FAIRNESS, so a large unit is never starved by smaller ones backfilling from other jobs: a job's oldest waiting unit
# that has waited SCHED_RESERVE_S (300 s; at once for a unit larger than the budget) writes a reservation
# $SCHED_HOME/reserve/<tag>-<name> ("<job pid> <estimate kB> <epoch>"). Reservations count against every unit except
# those of OLDER reservations, so the oldest one always progresses as running units finish (no deadlock): one per job.
#
# The caller sets SCHED_HOME, SCHED_TAG (unique per job), RES, BUDGET_KB, TOTAL_KB, CORES, BUSY_FILE (the busy_sampler
# file of tools/ops/box_cores_lib.sh) and QUEUE (array of "name<TAB>class<TAB>estimate kB" in start order), and defines
#   say <text>                 a log line
#   run_unit <name>            starts the unit in the background FROM THIS SHELL with fds 8 and 9 closed, sets UNIT_PID
#                              (empty when it cannot start); the unit writes $RES/rc/<name> when it is over
#   sched_on_done <name> <rc>  optional: called once per finished unit; may queue more units with sched_front
#   sched_pending_note         optional: prints why more work is still to come (e.g. "check slices wait for the build")
# Optional: SCHED_TICK (2 s), SCHED_REPORT_S (60 s), SCHED_RESERVE_S (300 s).
# Results: $RES/rc/<name> (by the units), $RES/notrun/<name> (the reason a unit never started).

SCHED_GIB=1048576
declare -gA SCHED_PID=() SCHED_SINCE=()
SCHED_MY_RESERVE=''

sched_front() { QUEUE=("$@" ${QUEUE[@]+"${QUEUE[@]}"}); }
sched_gib() { awk -v k="$1" 'BEGIN {printf "%.1f", k / 1048576}'; }

sched_ledger_sum() {  # kB reserved by every running unit of every job on the box (dead entries dropped)
    local s=0 f p kb
    for f in "$SCHED_HOME"/ledger/*; do
        [ -f "$f" ] || continue
        read -r p kb < "$f" 2> /dev/null || continue
        if [[ "$p" =~ ^[0-9]+$ ]] && [[ "$kb" =~ ^[0-9]+$ ]] && kill -0 "$p" 2> /dev/null; then s=$(( s + kb )); else rm -f "$f"; fi
    done
    echo "$s"
}

sched_held_ahead() {  # <my reservation file or ''>: kB held by live reservations ahead of me (all of them if I hold none)
    local mine=$1 s=0 f p kb t mt=''
    if [ -n "$mine" ] && [ -f "$mine" ]; then read -r p kb mt < "$mine" 2> /dev/null || mt=''; fi
    for f in "$SCHED_HOME"/reserve/*; do
        [ -f "$f" ] && [ "$f" != "$mine" ] || continue
        read -r p kb t < "$f" 2> /dev/null || continue
        if ! [[ "$p" =~ ^[0-9]+$ ]] || ! kill -0 "$p" 2> /dev/null; then rm -f "$f"; continue; fi
        if [[ "$mt" =~ ^[0-9]+$ ]] && [[ "$t" =~ ^[0-9]+$ ]]; then
            # only reservations older than mine (ties by name) hold memory ahead of me
            if [ "$t" -gt "$mt" ] || { [ "$t" -eq "$mt" ] && [[ "$f" > "$mine" ]]; }; then continue; fi
        fi
        [[ "$kb" =~ ^[0-9]+$ ]] && s=$(( s + kb ))
    done
    echo "$s"
}

sched_running_names() { local n out=''; for n in "${!SCHED_PID[@]}"; do out="$out $n"; done; echo "${out# }"; }

sched_loop() {
    local name class est line rc now used held ok rest started_any min_need last_report busy mine note
    local tick=${SCHED_TICK:-2} every=${SCHED_REPORT_S:-60} reserve_after=${SCHED_RESERVE_S:-300}
    mkdir -p "$SCHED_HOME/ledger" "$SCHED_HOME/reserve" "$RES/rc" "$RES/notrun"
    last_report=$(date +%s); used=0; held=0; min_need=0; started_any=0
    while [ ${#QUEUE[@]} -gt 0 ] || [ ${#SCHED_PID[@]} -gt 0 ]; do
        # reap
        for name in "${!SCHED_PID[@]}"; do
            if [ -f "$RES/rc/$name" ] || ! kill -0 "${SCHED_PID[$name]}" 2> /dev/null; then
                wait "${SCHED_PID[$name]}" 2> /dev/null
                rm -f "$SCHED_HOME/ledger/$SCHED_TAG-$name"
                rc=$(cat "$RES/rc/$name" 2> /dev/null || echo none)
                unset "SCHED_PID[$name]"
                say "done $name rc=$rc"
                if declare -F sched_on_done > /dev/null; then sched_on_done "$name" "$rc"; fi
            fi
        done
        # admit every queued unit that fits now (a smaller unit may pass a larger one, within the reservations)
        rest=(); started_any=0; min_need=0; now=$(date +%s)
        exec 8> "$SCHED_HOME/ledger.lock"; flock 8
        used=$(sched_ledger_sum)
        for line in ${QUEUE[@]+"${QUEUE[@]}"}; do
            IFS=$'\t' read -r name class est <<< "$line"
            [ -n "${SCHED_SINCE[$name]:-}" ] || SCHED_SINCE[$name]=$now
            if [ "$est" -gt "$TOTAL_KB" ]; then
                echo "cannot fit: needs $(sched_gib "$est") GiB of $(sched_gib "$TOTAL_KB") (the whole allocation; estimate from $class)" > "$RES/notrun/$name"
                say "NOT RUN $name: $(cat "$RES/notrun/$name")"
                continue
            fi
            mine="$SCHED_HOME/reserve/$SCHED_TAG-$name"
            held=$(sched_held_ahead "$mine")
            ok=0
            if [ "$est" -le "$BUDGET_KB" ]; then [ $(( used + held + est )) -le "$BUDGET_KB" ] && ok=1
            else [ "$used" -eq 0 ] && [ "$held" -eq 0 ] && ok=1; fi
            if [ "$ok" = 1 ]; then
                UNIT_PID=''
                run_unit "$name" || UNIT_PID=''
                rm -f "$mine"; [ "$SCHED_MY_RESERVE" != "$mine" ] || SCHED_MY_RESERVE=''
                if [ -z "$UNIT_PID" ]; then
                    echo "could not start (see the job log)" > "$RES/notrun/$name"; say "NOT RUN $name: could not start"
                    continue
                fi
                echo "$UNIT_PID $est" > "$SCHED_HOME/ledger/$SCHED_TAG-$name"
                SCHED_PID[$name]=$UNIT_PID; started_any=1
                used=$(( used + est ))
                say "start $name (estimate $(sched_gib "$est") GiB, waited $(( now - ${SCHED_SINCE[$name]} )) s; reserved now $(sched_gib "$used") of $(sched_gib "$BUDGET_KB") GiB)"
                continue
            fi
            rest+=("$line")
            [ "$min_need" -ne 0 ] && [ "$min_need" -le "$est" ] || min_need=$est
            # this job's oldest waiting unit reserves after SCHED_RESERVE_S (a unit above the budget at once)
            if [ -z "$SCHED_MY_RESERVE" ] && { [ $(( now - ${SCHED_SINCE[$name]} )) -ge "$reserve_after" ] || [ "$est" -gt "$BUDGET_KB" ]; }; then
                echo "$$ $est $now" > "$mine"; SCHED_MY_RESERVE=$mine
                say "reserve $(sched_gib "$est") GiB for $name (waiting $(( now - ${SCHED_SINCE[$name]} )) s on memory; later units on the box wait behind it)"
            fi
        done
        flock -u 8; exec 8>&-
        QUEUE=(${rest[@]+"${rest[@]}"})
        now=$(date +%s)
        if [ $(( now - last_report )) -ge "$every" ]; then
            last_report=$now
            busy=$(tail -n 1 "$BUSY_FILE" 2> /dev/null); [ -n "$busy" ] || busy='?'
            say "BUSY $busy of $CORES cores; running ${#SCHED_PID[@]} ($(sched_running_names)); queued ${#QUEUE[@]}; reserved $(sched_gib "$used") of $(sched_gib "$BUDGET_KB") GiB on the box"
            if [[ "$busy" =~ ^[0-9.]+$ ]] && awk -v b="$busy" -v c="$CORES" 'BEGIN {exit !(b < 0.8 * c)}'; then
                if [ ${#QUEUE[@]} -gt 0 ] && [ "$started_any" = 0 ]; then
                    say "BELOW 80%: waiting on memory: ${#QUEUE[@]} unit(s) queued, the smallest needs $(sched_gib "$min_need") GiB; $(sched_gib "$used") GiB reserved by running units on the box, $(sched_gib "$(sched_held_ahead '')") GiB held for waiting units, budget $(sched_gib "$BUDGET_KB") GiB"
                elif [ ${#QUEUE[@]} -eq 0 ]; then
                    note=''; declare -F sched_pending_note > /dev/null && note=$(sched_pending_note)
                    say "BELOW 80%: nothing left to start in this job${note:+ ($note)}; its ${#SCHED_PID[@]} running unit(s) use fewer cores than the box"
                else
                    say "BELOW 80%: units were just admitted; ${#QUEUE[@]} still queued"
                fi
            fi
        fi
        sleep "$tick"
    done
    [ -z "$SCHED_MY_RESERVE" ] || rm -f "$SCHED_MY_RESERVE"
    SCHED_MY_RESERVE=''
}
