# stage_full.sh (box-run-2): FULL_URL line prepended; rows-full for the DART quality gate.
set -euo pipefail
exec >>/root/lq/stage-full.log 2>&1
echo "== $(date -u +%FT%TZ) start"
umask 077; D=/root/board-0833/cache/algos-data; cd $D
cfg=$(mktemp); trap 'rm -f "$cfg" /root/.stage_full.in' EXIT
printf 'url = "%s"\n' "$FULL_URL" > "$cfg"; unset FULL_URL
curl -fsS --retry 3 -K "$cfg" -o rows-full.tar.gz.partial; mv rows-full.tar.gz.partial rows-full.tar.gz
sha256sum rows-full.tar.gz > /root/lq/rows-full-archive.sha256
tar -xzf rows-full.tar.gz; ls rows-full | wc -l
echo "== $(date -u +%FT%TZ) FULL_READY"
