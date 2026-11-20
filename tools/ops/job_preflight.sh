#!/bin/bash
# tools/ops/job_preflight.sh -- the SUBMIT-TIME check of a box job (lane fx-pod-job, 2026-11-18). On 2026-11-18 a
# call-profile job waited 2 h 37 min in the queue and then failed in 90 s on a condition that was already true when it
# was submitted (the venv held another wheel; then no manifest; then an install call that could not install), three
# times. tools/nvidia_central.sh submit streams this file to the lane's pod and runs it there before anything is
# queued; a FAIL line refuses the submit (exit 4), nothing is queued.
#
#   bash -s -- <lane tree> <script on the box> <results dir or ""> [K=V ...]   (the submit's --env values)
#
# Checks, in order, all reported (none stops the others):
#   1. the script is on the box and parses (bash -n);
#   2. every --env value that is an absolute path exists on the box (a staged wheel dir, a rows file, a manifest);
#   3. a script that declares `# job-preflight: <how>` runs its own input check, `<script> --preflight`, with the
#      submit's --env values and MOJOLEARN_RESULTS_DIR set as the job will see them. That mode reads and never writes
#      (tools/ops/call_profile_job.sh: the venv, the wheel and the manifest it needs agree; tools/ops/paper_compare_job.sh:
#      the candidate and B wheel sets, the rows file and the stored board are where the job reads them).
# Prints PREFLIGHT lines; exit 0 when nothing failed, 4 otherwise. CPU, seconds; nothing here is worth a second core.
# Tests: tools/test_job_preflight.py.
set -u
tree=${1:?lane tree}; script=${2:?script}; res=${3:-}
shift 3 2> /dev/null || shift $#
fails=0
fail() { echo "PREFLIGHT FAIL $*"; fails=$(( fails + 1 )); }
ok() { echo "PREFLIGHT ok $*"; }

if [ ! -f "$script" ]; then
    fail "no script $script on the box (sync the lane tree first)"
else
    if err=$(bash -n "$script" 2>&1); then ok "parses: $script"
    else fail "$script does not parse: $(printf '%s' "$err" | head -n 3 | tr '\n' ' ')"; fi
fi
for kv in "$@"; do
    k=${kv%%=*}; v=${kv#*=}
    case "$v" in
        /*) if [ -e "$v" ]; then ok "--env $k: $v"; else fail "--env $k=$v: no such file or dir on the box"; fi ;;
    esac
done
if [ -f "$script" ] && grep -q '^# job-preflight:' "$script"; then
    out=$(cd "$tree" 2> /dev/null && env "$@" MOJOLEARN_RESULTS_DIR="${res:-$tree/box-out/preflight}" \
          MOJOLEARN_BOX_PERSIST=1 bash "$script" --preflight 2>&1)
    rc=$?
    printf '%s\n' "$out" | grep '^PREFLIGHT' | head -n 40
    [ "$rc" -eq 0 ] || fail "$(basename "$script") --preflight exit $rc: $(printf '%s' "$out" | grep -v '^PREFLIGHT ok' | tail -n 2 | tr '\n' ' ')"
fi
if [ "$fails" -gt 0 ]; then
    echo "PREFLIGHT REFUSED $fails check(s) failed: nothing queued"
    exit 4
fi
echo "PREFLIGHT PASSED"
exit 0
