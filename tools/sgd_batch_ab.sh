#!/usr/bin/env bash
# SGD-family batch_size A/B on a GPU box against the published wheel (tools/sgd_batch_ab.py).
#   bash tools/sgd_batch_ab.sh <nvidia|amd>
# Creates /root/sgdab/<vendor>/venv, installs mojolearn==$SGDAB_VERSION (default 0.8.49) plus the vendor
# plugin with tools/pypi_wheel_install.py (byte provenance against PyPI), runs the driver and writes
# /root/sgdab/<vendor>/{results.jsonl,run.log,summary.txt,wheel-manifest.json}. GPU route only.
set -u
vendor=${1:?usage: sgd_batch_ab.sh <nvidia|amd>}
case "$vendor" in nvidia|amd) ;; *) echo "sgd_batch_ab: vendor must be nvidia or amd" >&2; exit 2;; esac
version=${SGDAB_VERSION:-0.8.49}
here=$(cd "$(dirname "$0")/.." && pwd)
O=/root/sgdab/$vendor
mkdir -p "$O"
unset MOJOLEARN_VENDOR
log=$O/run.log
: > "$log"
# The system interpreter is the venv base, never the checkout's pixi env, and the install runs from $O with no
# PYTHONPATH: the installer's import check must load mojolearn from the venv's site-packages, not from the tree
# (n0001 on 2026-10-17 imported tree/python/mojolearn and failed on its missing identical binaries).
unset PYTHONPATH
if [ -x /usr/bin/python3 ]; then base=/usr/bin/python3; else base=$(command -v python3); fi
cd "$O" || exit 3
echo "sgd_batch_ab: vendor=$vendor version=$version base=$base out=$O" | tee -a "$log"
[ -x "$O/venv/bin/python" ] || "$base" -m venv "$O/venv" >>"$log" 2>&1 || { echo "sgd_batch_ab: venv failed rc=$?" | tee -a "$log"; exit 3; }
P=$O/venv/bin/python
"$P" -m pip install --no-input --disable-pip-version-check -q numpy >>"$log" 2>&1 || { echo "sgd_batch_ab: numpy install failed" | tee -a "$log"; exit 3; }
wheels=$O/wheels-$(date -u +%Y%m%dT%H%M%SZ)
"$P" "$here/tools/pypi_wheel_install.py" install --version "$version" --vendor "$vendor" --python "$P" \
  --dir "$wheels" --manifest "$O/wheel-manifest.json" --log "$O/pypi-wheel-install.log" >>"$log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then echo "sgd_batch_ab: wheel install failed rc=$rc (see $O/pypi-wheel-install.log)" | tee -a "$log"; exit $rc; fi
rm -f "$O/results.jsonl"
cd "$O" || exit 3   # import mojolearn from site-packages, never the checkout
"$P" "$here/tools/sgd_batch_ab.py" --vendor "$vendor" --manifest "$O/wheel-manifest.json" --out "$O" >>"$log" 2>&1
rc=$?
grep -E '^SGDAB' "$log" > "$O/summary.txt"
echo "rc=$rc" >> "$O/summary.txt"
cat "$O/summary.txt"
exit $rc
