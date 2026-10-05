# stage_box.sh (box-run-2): run on a box via `ssh ... bash -s`, with SMALL_URL and NEURAL_URL lines prepended.
# Toolchain + source at the integration SHA + canonical medium data + neural fixtures. Logs to /root/lq/stage.log.
set -euo pipefail
mkdir -p /root/lq /root/mojolearn-evidence /root/board-0833/cache/algos-data
exec >>/root/lq/stage.log 2>&1
echo "== $(date -u +%FT%TZ) stage start"
export PATH=/root/.pixi/bin:/opt/rocm/bin:$PATH
command -v git >/dev/null || (apt-get update -qq && apt-get install -y -qq git)
command -v rsync >/dev/null || (apt-get update -qq && apt-get install -y -qq rsync) || true
[ -x /root/.pixi/bin/pixi ] || (curl -fsSL https://pixi.sh/install.sh | bash)
SHA=d795d2ade
if [ ! -d /root/mojolearn/.git ]; then git clone -q https://github.com/mojolearn/mojolearn.git /root/mojolearn; fi
cd /root/mojolearn
git fetch -q origin lane/box-run-2
git checkout -q --detach FETCH_HEAD
git rev-parse HEAD
pixi install --locked
pixi run mojo --version
umask 077
cd /root/board-0833/cache/algos-data
cfg=$(mktemp); trap 'rm -f "$cfg"' EXIT
printf 'url = "%s"\n' "$SMALL_URL" > "$cfg"; curl -fsS --retry 3 -K "$cfg" -o rows-small-canonical.tar.gz
sha256sum rows-small-canonical.tar.gz > /root/lq/rows-small-archive.sha256
tar -xzf rows-small-canonical.tar.gz
sha256sum -c rows-small-files.sha256 > /root/lq/rows-small-verify.log
cp rows-small-files.sha256 /root/lq/rows-small-files.sha256
printf 'url = "%s"\n' "$NEURAL_URL" > "$cfg"; curl -fsS --retry 3 -K "$cfg" -o neural-canonical-v2.tar.gz
sha256sum neural-canonical-v2.tar.gz > /root/lq/neural-archive.sha256
tar -tzf neural-canonical-v2.tar.gz > /root/lq/neural-tar-list.txt
mkdir -p wave-neural; tar -xzf neural-canonical-v2.tar.gz -C wave-neural
unset SMALL_URL NEURAL_URL
echo "== $(date -u +%FT%TZ) STAGE_READY"
