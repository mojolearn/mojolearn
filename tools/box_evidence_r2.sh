#!/bin/sh
# Ship a measurement directory from a remote box STRAIGHT to R2; the bytes never pass through the laptop.
#
#   sh tools/box_evidence_r2.sh "<ssh flags+target>" <remote dir> <r2 key> [tar --exclude pattern...]
#   e.g. sh tools/box_evidence_r2.sh "-p 18074 root@64.247.206.212" /root/pr44-nvidia \
#          measurements/2026-10-01/pr44-nvidia.tar.gz venv 'venv-*' data work
#
# The laptop only mints two short-lived presigned URLs (PUT and GET; credentials stay in ~/.mojolearn_r2) and pipes
# them to the box over stdin, never argv. The box tars the directory, uploads it, downloads it back and refuses
# unless the sha256 matches. On success it appends one row to bench/results/r2-index.tsv (commit it with the
# summary). Small evidence (SUMMARY.md, races.txt, digests) still goes in git; this is for the bulk: full logs,
# board roots, raw outputs (see docs/MEASUREMENT_STORAGE.md).
set -eu
target="${1:?usage: box_evidence_r2.sh \"<ssh flags+target>\" <remote dir> <r2 key> [exclude...]}"
dir="${2:?remote dir}"; key="${3:?r2 key}"; shift 3
here=$(cd "$(dirname "$0")" && pwd)
put=$(sh "$here/dataset_store.sh" presign-put "$key" 7200)
get=$(sh "$here/dataset_store.sh" presign "$key" 7200)
excl=""
for e in "$@"; do excl="$excl --exclude=$e"; done
# shellcheck disable=SC2086
res=$({ printf "PUT='%s'\nGET='%s'\nDIR='%s'\nEXCL='%s'\n" "$put" "$get" "$dir" "$excl"; cat <<'EOS'
set -eu
sum() { if command -v sha256sum >/dev/null 2>&1; then sha256sum "$1" | cut -d' ' -f1; else shasum -a 256 "$1" | cut -d' ' -f1; fi; }
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT; umask 077
tar czf "$W/e.tgz" $EXCL -C "$DIR" .
S=$(sum "$W/e.tgz"); N=$(wc -c < "$W/e.tgz" | tr -d ' ')
printf 'url = "%s"\n' "$PUT" > "$W/put"; printf 'url = "%s"\n' "$GET" > "$W/get"
curl -sS -f -K "$W/put" -T "$W/e.tgz"
curl -sS -f -K "$W/get" -o "$W/back.tgz"
B=$(sum "$W/back.tgz")
[ "$S" = "$B" ] || { echo "R2-REFUSED read-back sha $B != $S" >&2; exit 1; }
echo "R2-OK $S $N $(hostname)"
EOS
} | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $target sh -s)
echo "$res"
line=$(echo "$res" | grep '^R2-OK ' | tail -1)
[ -n "$line" ] || { echo "upload FAILED for $key" >&2; exit 1; }
set -- $line
idx="$here/../bench/results/r2-index.tsv"
[ -f "$idx" ] || printf 'utc\tr2_key\tsha256\tbytes\tbox\tsource_dir\n' > "$idx"
printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$key" "$2" "$3" "$4" "$dir" >> "$idx"
echo "indexed $key in bench/results/r2-index.tsv"
