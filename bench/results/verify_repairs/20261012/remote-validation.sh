#!/bin/bash
set -uo pipefail
cd /root/ml-verify
A=/root/ml-verify/artifacts
export MOJOLEARN_NUMERIC_MODE=identical
venv/bin/python -I -u -m mojolearn verify --neural-training --json-out "$A/repaired-neural-base.json" > "$A/repaired-neural-base.log" 2>&1
rc=$?; echo "$rc" > "$A/repaired-neural-base.exit"
venv/bin/python -I -u -m mojolearn verify --cross-check all --lanes radius,radius-chebyshev,radius-manhattan,radius-minkowski-p3 > "$A/radius-cross-check.log" 2>&1
rc=$?; echo "$rc" > "$A/radius-cross-check.exit"
# Keep every patched package byte together with the original downloaded wheels.
PKG=$(venv/bin/python -I -c 'import mojolearn,pathlib;print(pathlib.Path(mojolearn.__file__).parent)')
tar czf "$A/patched-python-sources.tar.gz" -C "$PKG" --exclude='*.so' --exclude='__pycache__' --exclude='*.dylib' . > "$A/retain-sources.log" 2>&1
rc=$?; echo "$rc" > "$A/retain-sources.exit"
printf '%s\n' "$rc" > "$A/validation.exit"
