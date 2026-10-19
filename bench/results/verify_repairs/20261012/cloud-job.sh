#!/bin/bash
set -uo pipefail
cd /root/ml-verify
A=/root/ml-verify/artifacts
finish() { rc=$?; printf '%s\n' "$rc" > "$A/job.exit"; }
trap finish EXIT
mkdir -p "$A/wheels"
export DEBIAN_FRONTEND=noninteractive
if ! python3 -m venv venv > "$A/venv.log" 2>&1; then
 apt-get update > "$A/apt.log" 2>&1 && apt-get install -y python3-venv >> "$A/apt.log" 2>&1 || exit 20
 python3 -m venv venv >> "$A/venv.log" 2>&1 || exit 21
fi
venv/bin/python -m pip install --disable-pip-version-check 'mojolearn[verify]==0.8.42' > "$A/install.log" 2>&1
rc=$?; echo "$rc" > "$A/install.exit"; [ "$rc" = 0 ] || exit 22
venv/bin/python -m pip download --no-deps --dest "$A/wheels" mojolearn==0.8.42 mojolearn-nvidia==0.8.42 mojolearn-amd==0.8.42 > "$A/wheels.log" 2>&1 || exit 23
venv/bin/python -m pip freeze > "$A/freeze.txt"
nvidia-smi -q > "$A/nvidia-smi.txt" 2>&1 || true
rocminfo > "$A/rocminfo.txt" 2>&1 || true
export MOJOLEARN_NUMERIC_MODE=identical
# A bounded test controller; one GPU process at a time. Full profile, all fixtures.
timeout -k 30 5400 venv/bin/python -I -u -m mojolearn verify --neural-training --all --json-out "$A/neural.json" > "$A/neural.log" 2>&1
rc=$?; echo "$rc" > "$A/neural.exit"
printf 'NEURAL_EXIT=%s\n' "$rc"
exit "$rc"
