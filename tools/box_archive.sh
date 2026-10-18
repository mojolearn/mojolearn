#!/bin/bash
# box_archive.sh <name> <ssh args...> : copy a rented box's logs and results home and to R2 BEFORE it can disappear.
#   e.g. tools/box_archive.sh nvlive -p 14530 root@202.181.159.235
#        tools/box_archive.sh do-amd root@107.170.35.161
# Writes ~/mojolearn-evidence/box-archive/<name>-<UTC stamp>/{box-info.txt,archive-lq.tgz,files.count,r2.log} and uploads
# the tarball to R2 as measurements/box-archive/<name>/<name>-<stamp>.tgz. Exit 0: archived and in R2; 3: box unreachable
# (nothing could be saved); 1: reachable but the archive or the upload failed.
# Called by tools/dev_pod.sh down and tools/do_amd_steward.sh down before the delete, and every hour by the launchd archiver
# (~/mojolearn-evidence/archive_all_boxes.sh) for every live box in the state dirs (Andrew 2026-10-10: nothing may be lost if a
# session dies; 2026-10-18: every run's results go off the box, to the Mac and R2, without anyone remembering to do it).
set -u
name=${1:?usage: box_archive.sh <name> <ssh args...>}; shift; [ $# -gt 0 ] || { echo "box_archive: no ssh target" >&2; exit 2; }
REPO=$(cd "$(dirname "$0")/.." && pwd)
SO="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=8"
stamp=$(date -u +%Y%m%dT%H%MZ); D=~/mojolearn-evidence/box-archive/$name-$stamp; mkdir -p "$D"
LOG=~/mojolearn-evidence/lease_steward.log
# shellcheck disable=SC2086
if ! ssh $SO "$@" true < /dev/null 2> "$D/remote.txt"; then
    echo "$(date -u +%H:%MZ) archive $name: box unreachable ($(head -c 120 "$D/remote.txt"))" | tee -a "$LOG"; exit 3
fi
# the box's hardware + software snapshot, so a result can always be traced to what it ran on (Andrew 2026-10-10)
# shellcheck disable=SC2086
ssh $SO "$@" 'cd /root && { date -u; uname -a; nproc; free -g | head -n 2; nvidia-smi -q 2>/dev/null | head -n 40; rocm-smi --showproductname --showdriverversion 2>/dev/null; cat /opt/rocm/.info/version 2>/dev/null; nvcc --version 2>/dev/null | tail -n 2; for d in /root/mojolearn-ab2 /root/mojolearn; do [ -d $d ] && (cd $d && git log -1 --format="%H %cs" 2>/dev/null; .pixi/envs/default/bin/mojo --version 2>/dev/null); done; } > /root/box-info.txt 2>&1' < /dev/null
# streamed straight to the Mac: no copy on the box (a 7 GB copy once filled nv2's disk, 2026-10-11). Every run directory's
# logs and results, without its build caches, venvs or files over 50 MB (wheels, binaries and datasets are rebuilt or in R2).
# shellcheck disable=SC2086
ssh $SO "$@" 'cd /root && find box-info.txt *.log lq gpu-queue/dispatcher.log ref-* *-fixtures* ols-* live-* measure-* race-* results* mojolearn-evidence -type f -size -50M ! -path "*/cache/*" ! -path "*/venv/*" ! -path "*/.venv/*" ! -path "*/finish-venv/*" ! -name "*.so" ! -name "*.whl" 2>/dev/null | tar czf - -T -' < /dev/null > "$D/archive-lq.tgz" 2>> "$D/remote.txt"
if ! gzip -t "$D/archive-lq.tgz" 2>/dev/null || [ ! -s "$D/archive-lq.tgz" ]; then
    echo "$(date -u +%H:%MZ) archive $name: FAILED, tarball incomplete ($(tail -c 160 "$D/remote.txt" | tr '\n' ' '))" | tee -a "$LOG"; exit 1
fi
tar tzf "$D/archive-lq.tgz" | wc -l | tr -d ' ' > "$D/files.count"
echo "$(date -u +%H:%MZ) archived $name -> $D ($(cat "$D/files.count") files, $(du -sh "$D" | cut -f1))" | tee -a "$LOG"
K=measurements/box-archive/$name/$(basename "$D").tgz
if (cd "$REPO" && sh tools/dataset_store.sh put-file "$D/archive-lq.tgz" "$K") > "$D/r2.log" 2>&1; then
    echo "$(date -u +%H:%MZ) r2 ok $K" | tee -a "$LOG"; touch "$D/R2_OK"
else
    echo "$(date -u +%H:%MZ) r2 FAILED $K ($(tail -n 1 "$D/r2.log" | cut -c1-160))" | tee -a "$LOG"; exit 1
fi
