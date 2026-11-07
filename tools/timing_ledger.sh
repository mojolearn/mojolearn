#!/bin/bash
# tools/timing_ledger.sh -- the timing ledger for bash wrappers (tools/timing_ledger.py is the reader/report).
#
#   timing_ledger.sh start <kind> <name>            note the start clock; a background sampler records busy
#                                                   cores every TL_SAMPLE_SEC (default 60; 0 = none) until end
#   timing_ledger.sh end <kind> <name> [rc] [k=v]... append one mojolearn.timing.v1 line to the ledger
#   timing_ledger.sh sample                         print busy cores now (1 s sample) and the core count
#   . timing_ledger.sh                              (sourced) defines tl_start / tl_end / tl_busy / tl_cores
#
# It only records clocks (date +%s) and busy-core samples; no other work. Ledger path, the same rule as the
# Python side: $MOJOLEARN_TIMING_LEDGER; else <MOJOLEARN_RESULTS_DIR>/timing-ledger.jsonl inside a box job
# (MOJOLEARN_BOX_PERSIST=1); else ~/mojolearn-evidence/timing/ledger.jsonl. Open units live under
# $TL_STATE_DIR (default <ledger dir>/.open), keyed by kind+name (+ $TL_KEY when one name runs twice at once).
# The commit is $MOJOLEARN_TIMING_COMMIT, $MOJOLEARN_COMMIT, or `git rev-parse HEAD` in the current directory.
# Needs bash, date, mkdir; awk only for the macOS busy sample. Never fails the caller: every error is ignored.

tl_ledger() {
    if [ -n "${MOJOLEARN_TIMING_LEDGER:-}" ]; then echo "$MOJOLEARN_TIMING_LEDGER"
    elif [ "${MOJOLEARN_BOX_PERSIST:-}" = 1 ] && [ -n "${MOJOLEARN_RESULTS_DIR:-}" ]; then
        echo "$MOJOLEARN_RESULTS_DIR/timing-ledger.jsonl"
    else echo "$HOME/mojolearn-evidence/timing/ledger.jsonl"; fi
}

tl_cores() {
    nproc 2> /dev/null || sysctl -n hw.ncpu 2> /dev/null || getconf _NPROCESSORS_ONLN 2> /dev/null || echo 0
}

# busy cores x10 (an integer: one decimal without floating point)
tl_busy10() {
    if [ -r /proc/stat ]; then
        local a b t0=0 i0 t1=0 i1 x n
        read -r _ a < /proc/stat; sleep 1; read -r _ b < /proc/stat
        set -- $a; i0=$(( $4 + ${5:-0} )); for x in $a; do t0=$(( t0 + x )); done
        set -- $b; i1=$(( $4 + ${5:-0} )); for x in $b; do t1=$(( t1 + x )); done
        n=$(tl_cores)
        [ $(( t1 - t0 )) -gt 0 ] || { echo ""; return; }
        echo $(( n * 10 * ( (t1 - t0) - (i1 - i0) ) / (t1 - t0) ))
    else
        ps -A -o %cpu= 2> /dev/null | awk '{s += $1} END {printf "%d\n", s / 10}'
    fi
}

tl_busy() { local b; b=$(tl_busy10); [ -n "$b" ] && echo "$(( b / 10 )).$(( b % 10 ))"; }

tl_state() {  # <kind> <name> -> the state file path
    local dir key
    dir=${TL_STATE_DIR:-$(dirname "$(tl_ledger)")/.open}
    key=$(printf '%s__%s%s' "$1" "$2" "${TL_KEY:+__$TL_KEY}" | tr -c 'A-Za-z0-9._-' '_' | cut -c1-200)
    mkdir -p "$dir" 2> /dev/null
    echo "$dir/$key"
}

tl_json_str() { local s=${1//\\/\\\\}; s=${s//\"/\\\"}; s=${s//$'\n'/ }; s=${s//$'\t'/ }; printf '"%s"' "$s"; }

tl_start() {  # <kind> <name>
    local st every first
    st=$(tl_state "$1" "$2") || return 0
    date +%s > "$st" 2> /dev/null || return 0
    : > "$st.samples"
    every=${TL_SAMPLE_SEC:-60}; first=${TL_SAMPLE_FIRST_SEC:-30}
    case "$every" in ''|*[!0-9]*|0) return 0 ;; esac
    # the sampler stops when the unit ends (state file gone) or after TL_SAMPLE_MAX_SEC (default 7 days)
    (
        trap '' HUP
        exec > /dev/null 2>&1 < /dev/null
        deadline=$(( $(date +%s) + ${TL_SAMPLE_MAX_SEC:-604800} )); wait_s=$first
        while [ -f "$st" ] && [ "$(date +%s)" -lt "$deadline" ]; do
            sleep "$wait_s"; [ -f "$st" ] || break
            b=$(tl_busy10); [ -n "$b" ] && echo "$b" >> "$st.samples"
            wait_s=$every
        done
    ) &
    disown $! 2> /dev/null || true
    return 0
}

tl_end() {  # <kind> <name> [rc] [key=value ...]
    local kind=$1 name=$2 rc=${3:-null} st start end ledger commit busy="null" n=0 sum=0 max=0 s extra="" kv
    shift 2; [ $# -gt 0 ] && shift
    case "$rc" in ''|*[!0-9]*) rc=null ;; esac
    st=$(tl_state "$kind" "$name") || return 0
    end=$(date +%s)
    start=$(cat "$st" 2> /dev/null); case "$start" in ''|*[!0-9]*) start=$end ;; esac
    if [ -s "$st.samples" ]; then
        while read -r s; do case "$s" in ''|*[!0-9]*) continue ;; esac
            n=$(( n + 1 )); sum=$(( sum + s )); [ "$s" -gt "$max" ] && max=$s; done < "$st.samples"
    fi
    if [ "$n" -gt 0 ]; then
        busy="{\"mean\": $(( sum / n / 10 )).$(( sum / n % 10 )), \"max\": $(( max / 10 )).$(( max % 10 )), \"samples\": $n}"
    elif [ "${TL_SAMPLE_SEC:-60}" != 0 ]; then
        s=$(tl_busy10)   # a unit shorter than the first sample: one sample now, marked at_end
        [ -n "$s" ] && { busy="{\"mean\": $(( s / 10 )).$(( s % 10 )), \"max\": $(( s / 10 )).$(( s % 10 )), \"samples\": 1}"; extra="\"busy_at_end\": true"; }
    fi
    rm -f "$st" "$st.samples" 2> /dev/null
    for kv in "$@"; do
        case "$kv" in *=*) ;; *) continue ;; esac
        extra="${extra:+$extra, }$(tl_json_str "${kv%%=*}"): $(tl_json_str "${kv#*=}")"
    done
    commit=${MOJOLEARN_TIMING_COMMIT:-${MOJOLEARN_COMMIT:-$(git rev-parse HEAD 2> /dev/null || echo unknown)}}
    ledger=$(tl_ledger); mkdir -p "$(dirname "$ledger")" 2> /dev/null
    printf '{"schema": "mojolearn.timing.v1", "id": "%s", "kind": %s, "name": %s, "start": %s, "end": %s, "seconds": %s, "host": %s, "cores": %s, "busy": %s, "commit": %s, "rc": %s, "extra": {%s}}\n' \
        "$(printf '%s-%s-%s-%s' "$(hostname 2> /dev/null)" "$end" "$$" "$RANDOM$RANDOM" | tr -c 'A-Za-z0-9-' '-')" \
        "$(tl_json_str "$kind")" "$(tl_json_str "$name")" "$start" "$end" "$(( end - start ))" \
        "$(tl_json_str "$(hostname 2> /dev/null || echo unknown)")" "$(tl_cores)" "$busy" \
        "$(tl_json_str "$commit")" "$rc" "$extra" >> "$ledger" 2> /dev/null
    return 0
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    case "${1:-}" in
        start) [ $# -ge 3 ] || { sed -n 2,16p "$0"; exit 2; }; tl_start "$2" "$3" ;;
        end) [ $# -ge 3 ] || { sed -n 2,16p "$0"; exit 2; }; shift; tl_end "$@" ;;
        sample) echo "busy $(tl_busy) of $(tl_cores)" ;;
        *) sed -n 2,16p "$0"; exit 2 ;;
    esac
    exit 0
fi
