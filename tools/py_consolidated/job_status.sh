#!/bin/bash
# Sourced by the two job drivers. Keep collecting evidence after a failed phase.
# A file ledger also captures errors inside subshells/functions/pipelines.
set -Euo pipefail
mkdir -p "$EV"
JOB_FAILURES="$EV/job-failures.tsv"
: > "$JOB_FAILURES"
job_record_failure() {
    local rc=$1 line=$2 command=$3
    printf '%s\t%s\t%s\n' "$rc" "$line" "$command" >> "$JOB_FAILURES"
    printf 'PHASE FAILED rc=%s line=%s: %s\n' "$rc" "$line" "$command" >&2
}
trap 'job_record_failure "$?" "$LINENO" "$BASH_COMMAND"' ERR
job_finish() {
    if [ -s "$JOB_FAILURES" ]; then
        echo "JOB FAILED/INCOMPLETE: see $JOB_FAILURES"
        return 1
    fi
    echo 'JOB PASS: all selected required commands completed'
}
