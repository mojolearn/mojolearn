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
# All interpreter/mode jobs remain serial; bound CPU math inside each job.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1

# --no-gpu: verify build, install and API on every interpreter, but do not
# attempt a fit. For environments with no usable GPU, which on this project
# means GitHub's virtualized runners. See packaging/macos/smoke.py.
SMOKE_ARGS=""
MODE="full (device fits)"
if [ "${1:-}" = "--no-gpu" ]; then
    SMOKE_ARGS="--no-gpu"
    MODE="--no-gpu (import and API only, DEVICE NOT TESTED)"
    shift
fi
[ "$#" -eq 0 ] || { echo "usage: $0 [--no-gpu]" >&2; exit 2; }

here=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
set -- "$here"/python/dist/mojolearn-*.whl
[ -f "$1" ] || { echo "no wheel in python/dist; run build_release_wheel.sh" >&2; exit 1; }
[ "$#" -eq 1 ] || { echo "multiple wheels in python/dist; qualification requires exactly one candidate" >&2; exit 1; }
WHEEL=$1
echo "wheel: $(basename "$WHEEL")"
echo "mode:  $MODE"

MODES="${MOJOLEARN_RELEASE_MODES:-fast deterministic identical}"
set -f
set -- $MODES
[ "$#" -gt 0 ] || { echo "no numeric modes requested" >&2; exit 2; }
for mode do
    case "$mode" in
        fast|deterministic|identical) ;;
        *) echo "unsupported numeric mode: $mode" >&2; exit 2 ;;
    esac
done

fails=0
skips=0
passed=0
# Every interpreter at or above the declared floor. Ones that are not
# installed are reported as SKIP, and incomplete coverage fails this gate.
for py in python3.10 python3.11 python3.12 python3.13 python3.14; do
    command -v "$py" >/dev/null 2>&1 || { echo "SKIP $py (not installed)"; skips=$((skips+1)); continue; }
    tmp=$(mktemp -d)
    if ! "$py" -m venv "$tmp/venv" >/dev/null 2>&1; then
        echo "SKIP $py (venv creation failed)"; skips=$((skips+1)); rm -rf "$tmp"; continue
    fi
    # --no-cache-dir so a previously built wheel cannot be silently reused.
    if ! "$tmp/venv/bin/pip" install --quiet --no-cache-dir "$WHEEL" >/dev/null 2>&1; then
        echo "FAIL $py: pip install refused the wheel"; fails=$((fails+1)); rm -rf "$tmp"; continue
    fi
    # cd to /tmp so the repository's ./python/mojolearn cannot shadow the
    # installed package. Without this the test can pass on source that is not
    # in the wheel at all.
    # EVERY SHIPPED NUMERIC MODE, from the same installed wheel. This said
    # BOTH NUMERIC MODES while there were two; there are three as of
    # 2026-08-29 and the word is wrong rather than merely dated. Each upper
    # tier is selected by the env var at import (python/mojolearn/_backend.py)
    # and smoke.py asserts that the mode it reads BACK OUT OF THE BINARY is
    # the one that was asked for, so a wheel that silently shipped only the
    # fast set fails here rather than on a user's machine.
    okmode=1
    if [ "${MOJOLEARN_PACKAGE_BYTE_LM:-0}" = 1 ]; then
        # Availability plus bounded generalized/resident native execution.
        if (cd "$tmp" && env -u PYTHONPATH -u PYTHONHOME MOJOLEARN_NUMERIC_MODE=identical \
                "$tmp/venv/bin/python" - <<'PYBYTE'
import pathlib, sys
import mojolearn
from mojolearn import _mojolearn_byte_lm as native
package = pathlib.Path(mojolearn.__file__).resolve().parent
assert package.is_relative_to(pathlib.Path(sys.prefix).resolve())
assert pathlib.Path(native.__file__).resolve() == package / 'identical/_mojolearn_byte_lm.so'
assert not (package / '_mojolearn_byte_lm.so').exists()
assert not (package / 'deterministic/_mojolearn_byte_lm.so').exists()
assert native.byte_lm_numeric_mode() == 1 and native.byte_lm_vendor() == 'metal'
assert native.byte_lm_profile() == 'mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1'
print('Byte LM IDENTICAL-only native available; numerical training qualification separate')
PYBYTE
        ); then :; else
            echo "FAIL $py: installed byte LM native availability"; okmode=0
        fi
        if (cd "$tmp" && env -u PYTHONPATH -u PYTHONHOME MOJOLEARN_NUMERIC_MODE=identical \
                "$tmp/venv/bin/python" "$here/packaging/language_model_smoke.py"); then :; else
            echo "FAIL $py: installed generalized/resident language model"; okmode=0
        fi
    fi
    # The same tier list the wheel was built with, so a deliberately two-tier
    # wheel is not failed for lacking a third. MOJOLEARN_RELEASE_MODES is what
    # build_release_wheel.sh reads; keep them set the same for one release.
    for mode in $MODES; do
        if out=$(cd "$tmp" && MOJOLEARN_NUMERIC_MODE=$mode "$tmp/venv/bin/python" "$here/packaging/macos/smoke.py" $SMOKE_ARGS 2>&1); then
            printf 'PASS %s [%s]  %s\n' "$py" "$mode" "$out"
        else
            printf 'FAIL %s [%s]\n%s\n' "$py" "$mode" "$out"; okmode=0
        fi
    done
    if [ "$okmode" -eq 1 ]; then
        passed=$((passed+1))
    else
        fails=$((fails+1))
    fi
    rm -rf "$tmp"
done

[ "$fails" -eq 0 ] && [ "$skips" -eq 0 ] && [ "$passed" -eq 5 ] || {
    echo "incomplete qualification: $passed interpreter(s) passed, $fails failed, $skips skipped"
    exit 1
}
echo "all 5 interpreters passed ($MODE)"
