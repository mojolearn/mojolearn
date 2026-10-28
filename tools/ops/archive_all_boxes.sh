#!/bin/bash
# launchd (com.mojolearn.box-archive, every 3 h): archive every LIVE rented box to the Mac + R2, independent of any Claude session
# (Andrew 2026-10-10: nothing may be lost if the session dies; boxes self-destruct at lease end, so archive often).
# 2026-10-18: boxes are found from their state files (devpods/<key>/state.env, do-amd-steward/state.env), never from a
# hand-kept name list: the old list pointed at released boxes while two live ones went unarchived. Teardown archives too
# (tools/dev_pod.sh down, tools/do_amd_steward.sh down call tools/box_archive.sh before the delete).
E=~/mojolearn-evidence; R=~/CascadeProjects/mojolearn; A=$R/tools/box_archive.sh
[ -f "$A" ] || { echo "$(date -u +%H:%MZ) archiver: $A missing" >> "$E/lease_steward.log"; exit 1; }
failures=0
mkdir -p "$E/box-archive"
for s in "$E"/devpods/*/state.env; do
    [ -f "$s" ] || continue
    # a released box's state dir is renamed <key>.down-<stamp>; its address may now belong to someone else's machine
    case "$s" in *.down*) continue ;; esac
    (
        # shellcheck disable=SC1090
        . "$s"
        case "${PROVIDER:-}" in runpod|hotaisle) ;; *) exit 0 ;; esac
        [ -n "${SSH_TARGET:-}" ] || exit 0
        # shellcheck disable=SC2086
        bash "$A" "$(basename "$(dirname "$s")")" $SSH_TARGET
    ) >> "$E/box-archive/launchd.log" 2>&1 || failures=$((failures + 1))
done
if [ -f "$E/do-amd-steward/state.env" ]; then
    ( . "$E/do-amd-steward/state.env"; bash "$A" "do-amd-$DROPLET_ID" "root@$IP" ) >> "$E/box-archive/launchd.log" 2>&1 || failures=$((failures + 1))
fi
# Only prune copies with a successful R2 receipt. Failed uploads and incomplete
# archives stay on disk for recovery; an unsuccessful run cannot evict evidence.
cd "$E/box-archive" || exit 1
python3 - <<'PRUNE'
from pathlib import Path
import re, shutil
by_box = {}
for p in Path('.').iterdir():
    match = re.fullmatch(r'(.*)-20[0-9]{6}T[0-9]{4}Z', p.name)
    if match and p.is_dir() and (p/'R2_OK').is_file() and (p/'archive-lq.tgz').is_file():
        by_box.setdefault(match[1], []).append(p)
for copies in by_box.values():
    for p in sorted(copies, key=lambda x: x.name, reverse=True)[3:]:
        shutil.rmtree(p)
PRUNE
prune_rc=$?
[ "$prune_rc" -eq 0 ] && [ "$failures" -eq 0 ]
