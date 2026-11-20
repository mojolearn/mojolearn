#!/bin/bash
# tools/ops/apple_fast_check_install.sh <built artifacts dir> <out dir> -- the smoke job of
# .github/workflows/apple-fast-check.yml (lane fx-apple-fast-ci, 2026-11-20). Run from the checkout under test, on the
# macOS runner, with the pixi environments `default` and `pkg` installed.
#
#   1. place   every built <stem>.so of the build jobs into python/mojolearn/{,identical/,host/} with its merged index
#   2. stage   the runtime libraries with the release's own stagers (packaging/macos/stage_dylibs.py,
#              packaging/portable_math/stage.py). A stager that refuses is recorded and the smoke goes on: on this
#              machine the files' original rpath into .pixi still resolves.
#   3. pack    `python -m build --wheel` of python/ (the command packaging/macos/build_release_wheel.sh packs with).
#              NOT the release packer: no readback gates, no ISA or GPU-code gate, no identity payload, no shrink.
#   4. install the wheel into a clean venv; the smoke runs from empty directories, so the INSTALLED package imports.
#              If the pack or the install fails, the smoke runs on the source tree (PYTHONPATH=python) and the table
#              says so: a source-tree run is not an installed-wheel run.
#   5. smoke   tools/ops/apple_fast_check.py smoke (one process per cell, a timeout each, every cell at once).
#
# Every step's full output goes to <out>/<step>.log; a line per step goes to <out>/NOTE-<n>-<step>.txt, which the
# summary prints. Exit 0 unless the smoke itself could not be started: this reports, it gates nothing.
set -u
built=${1:?built artifacts dir}
out=${2:?out dir}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$out"
out=$(cd "$out" && pwd)
pkg=python/mojolearn
note() { printf 'smoke %s: %s\n' "$1" "$2" | tee "$out/NOTE-$1.txt"; }

{ sw_vers; sysctl -n machdep.cpu.brand_string hw.ncpu hw.memsize; } > "$out/machine-smoke.txt" 2>&1

# 1. place
python3 "$here/apple_fast_check.py" place --artifacts "$built" --pkg "$pkg" > "$out/place.log" 2>&1
note 1-place "rc=$? $(tail -n 1 "$out/place.log")"
grep -h '^MISSING' "$out/place.log" | head -n 40

# 2. stage (every file of every set in ONE call: the stager wipes and rebuilds the one .dylibs directory)
sos=$(find "$pkg" -maxdepth 2 -name '_mojolearn*.so' | sort)
if [ -n "$sos" ]; then
    # shellcheck disable=SC2086  # the file list is deliberately word-split
    pixi run -e pkg python packaging/macos/stage_dylibs.py $sos "$PWD/.pixi/envs/default/lib" \
        --codec-lib "$PWD/.pixi/envs/pkg/lib" > "$out/stage-dylibs.log" 2>&1
    note 2-stage-dylibs "rc=$? $(tail -n 1 "$out/stage-dylibs.log" | cut -c1-200)"
    pixi run -e pkg python packaging/portable_math/stage.py "$pkg/.dylibs" --receipt portable-math-build.json \
        > "$out/stage-math.log" 2>&1
    note 3-stage-math "rc=$? $(tail -n 1 "$out/stage-math.log" | cut -c1-200)"
else
    note 2-stage-dylibs "skipped: no built file was placed"
fi

# 3. pack
( cd python && rm -rf dist build ./*.egg-info && pixi run -e pkg python -m build --wheel --no-isolation ) \
    > "$out/pack.log" 2>&1
pack_rc=$?
wheel=$(ls python/dist/*.whl 2>/dev/null | head -n 1)
note 4-pack "rc=$pack_rc wheel=${wheel:-none} $( [ -n "$wheel" ] && wc -c < "$wheel" | tr -d ' ' ) bytes"

# 4. install
venv="${RUNNER_TEMP:-$out}/apple-smoke-venv"
pixi run -e pkg python -m venv "$venv" > "$out/install.log" 2>&1
py="$venv/bin/python"
installed=0
if [ -n "$wheel" ] && "$py" -m pip install --quiet "$wheel" >> "$out/install.log" 2>&1; then
    installed=1
    note 5-install "installed $wheel into a clean venv (unsigned-audit, ungated wheel of this runner; not a release wheel)"
else
    "$py" -m pip install --quiet numpy >> "$out/install.log" 2>&1
    export PYTHONPATH="$PWD/python"
    note 5-install "WHEEL NOT INSTALLED (pack rc=$pack_rc; see pack.log, install.log): the smoke below ran on the SOURCE TREE (PYTHONPATH=python), not an installed wheel"
fi
[ -x "$py" ] || { note 6-smoke "no interpreter: the venv was not created (install.log)"; exit 1; }

# 5. smoke
"$py" "$here/apple_fast_check.py" smoke --out "$out" --timeout 300 > "$out/smoke.log" 2>&1
note 6-smoke "rc=$? installed_wheel=$installed $(grep -m 1 '^cells:' "$out/smoke.log")"
cat "$out/SMOKE.md" 2>/dev/null >> "${GITHUB_STEP_SUMMARY:-/dev/null}"
tail -n 30 "$out/smoke.log"
exit 0
