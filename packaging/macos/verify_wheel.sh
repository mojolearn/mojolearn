#!/bin/sh
# Install the built wheel into a CLEAN venv under every python3.N on this
# machine and run a real fit.
#
# THE VENV MUST BE CLEAN AND MUST NOT BE THE BUILD ENVIRONMENT. On the build
# machine the extension's original @rpath still resolves to the pixi
# environment, so a wheel with NO staged dylibs imports perfectly here and
# fails on every other Mac. Testing in the environment that built it proves
# nothing. That is what this script exists to avoid.
#
# It also decides the `py3` interpreter tag in setup.py. That tag claims one
# artifact serves several CPython minors, which is true only because the
# extension links no libpython -- and "true in principle" is not the standard.
# Every interpreter this passes on is a version the tag may claim; the
# classifiers in pyproject.toml list exactly those and no others.
set -eu

# --no-gpu: verify build, install and API on every interpreter, but do not
# attempt a fit. For environments with no usable GPU, which on this project
# means GitHub's virtualized runners. See packaging/macos/smoke.py.
SMOKE_ARGS=""
MODE="full (device fits)"
if [ "${1:-}" = "--no-gpu" ]; then
    SMOKE_ARGS="--no-gpu"
    MODE="--no-gpu (import and API only, DEVICE NOT TESTED)"
fi

here=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
WHEEL=$(ls "$here"/python/dist/mojolearn-*.whl 2>/dev/null | head -1)
[ -n "$WHEEL" ] || { echo "no wheel in python/dist; run build_release_wheel.sh" >&2; exit 1; }
echo "wheel: $(basename "$WHEEL")"
echo "mode:  $MODE"

fails=0
# Every interpreter at or above the declared floor. Ones that are not
# installed are SKIPPED and reported as skipped, never silently passed.
for py in python3.10 python3.11 python3.12 python3.13 python3.14; do
    command -v "$py" >/dev/null 2>&1 || { echo "SKIP $py (not installed)"; continue; }
    tmp=$(mktemp -d)
    if ! "$py" -m venv "$tmp/venv" >/dev/null 2>&1; then
        echo "SKIP $py (venv creation failed)"; rm -rf "$tmp"; continue
    fi
    # --no-cache-dir so a previously built wheel cannot be silently reused.
    if ! "$tmp/venv/bin/pip" install --quiet --no-cache-dir "$WHEEL" >/dev/null 2>&1; then
        echo "FAIL $py: pip install refused the wheel"; fails=$((fails+1)); rm -rf "$tmp"; continue
    fi
    # cd to /tmp so the repository's ./python/mojolearn cannot shadow the
    # installed package. Without this the test can pass on source that is not
    # in the wheel at all.
    if out=$(cd "$tmp" && "$tmp/venv/bin/python" "$here/packaging/macos/smoke.py" $SMOKE_ARGS 2>&1); then
        echo "PASS $py  $out"
    else
        echo "FAIL $py"; echo "$out" | tail -5; fails=$((fails+1))
    fi
    rm -rf "$tmp"
done

[ "$fails" -eq 0 ] || { echo "$fails interpreter(s) failed"; exit 1; }
echo "all interpreters passed"
