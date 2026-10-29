#!/bin/bash
# A single finite run, then verified final retention and early self-delete.
# No lease renewal, service, watcher, polling collector or new rental.
set -u
root=/root/measure-final-neural-bc11389
persist=/root/box-persist/jobs/neural-final-bc11389-20261029/persist.json
trap '' HUP PIPE
bash /root/box-persist/bin/box_persist_run.sh "$persist" "$root" -- \
    /root/neural-venv/bin/python "$root/finite_continuation.py"
run_rc=$?
printf '%s finite command exit=%s\n' "$(date -u +%FT%TZ)" "$run_rc"
# The verified earlier archive holds both model checkpoint inputs, the source,
# numerical binaries and previous completed scores. Never delete on truncation.
if [ ! -e "$root/PERSIST_TRUNCATED.txt" ] && \
   grep -q 'READBACK-OK name=final.tgz' "$root/persist.log" && \
   grep -q 'UPLOADED name=final.tgz' "$root/persist.log" && \
   /root/neural-venv/bin/python -c 'import json; assert json.load(open("/root/measure-final-neural-bc11389/prior-retention.json"))["checkpoint_bytes_retained"]'; then
    # Same existing guarded RunPod endpoints and credential file as the lease;
    # no credential is copied to evidence or exposed through process arguments.
    for url in https://api.runpod.io/v2/pods/9fjva638f5j3tu https://rest.runpod.io/v1/pods/9fjva638f5j3tu; do
        code=$(curl -s -o /tmp/neural-final-delete-response -w '%{http_code}' --max-time 30 \
            -X DELETE "$url" -K /tmp/mojolearn-lease.curlrc)
        printf '%s early delete http=%s\n' "$(date -u +%FT%TZ)" "$code"
        case "$code" in 2*) exit 0;; esac
    done
fi
printf '%s early deletion not confirmed; existing hard lease remains armed\n' "$(date -u +%FT%TZ)"
exit 1
