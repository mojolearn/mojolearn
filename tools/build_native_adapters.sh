#!/usr/bin/env bash
# Optional adapter build; run on an authorized build/test machine only.
set -euo pipefail
out=${1:?absolute output directory}
arch=${2:-gfx942}
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root"
mkdir -p "$out"
git rev-parse HEAD > "$out/source-head.txt"
git diff --binary HEAD > "$out/source-changes.patch"
cp .git/devpod_synced "$out/synced-source.txt" 2>/dev/null || true
tar -czf "$out/adapter-source.tar.gz" adapters tools/build_native_adapters.sh tools/check_native_adapters.py
pixi run mojo --version > "$out/compiler.txt" 2>&1
host_flags=()
target_flags=()
column_flags=()
if [[ -n "${MOJOLEARN_TARGET_COLUMN:-}" ]]; then
    [[ "$MOJOLEARN_TARGET_COLUMN" =~ ^[a-z]+$ ]] || { echo 'invalid target column'; exit 2; }
    column_flags=(-D "MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')")
fi
case "$(uname -s)" in
    Linux) host_flags=(--target-cpu x86-64-v3); target_flags=(--target-accelerator "$arch") ;;
    Darwin)
        [ "$arch" = apple-native ] || { echo 'Darwin requires apple-native on a real Apple GPU build host' >&2; exit 2; }
        # Match the accepted Apple build policy in bindings/build.sh. On the
        # pinned compiler, forcing --target-accelerator can omit embedded AIR.
        # Use native GPU detection; packaging must inspect Metal payloads and
        # execute real kernels before admitting the resulting artifact.
        host_flags=(--target-cpu apple-m1)
        system_profiler SPDisplaysDataType > "$out/apple-hardware.txt"
        ;;
    *) echo 'unsupported build host' >&2; exit 2 ;;
esac
# Python is an external build controller. This process-group deadline works on
# Linux and macOS without requiring a GNU timeout executable on the Apple host.
python3 - "$out" pixi run mojo build -j "${MOJOLEARN_COMPILE_JOBS:-2}" \
    --emit shared-lib "${host_flags[@]}" "${target_flags[@]}" \
    -D MOJOLEARN_NUMERIC_IDENTICAL=1 "${column_flags[@]}" -I . -I bindings \
    adapters/native/c_api.mojo -o "$out/libmojolearn_adapter.so" <<'PY'
import json, os, signal, subprocess, sys
from pathlib import Path
out, command = Path(sys.argv[1]), sys.argv[2:]
(out / 'build-command.json').write_text(json.dumps(command, indent=2) + '\n')
with (out / 'build.log').open('w') as log:
    process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    expired = False
    try:
        code = process.wait(timeout=1200)
    except subprocess.TimeoutExpired:
        expired = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        code = 124
(out / 'build.rc').write_text(str(code) + '\n')
(out / 'build-status.json').write_text(json.dumps({'exit_code': code, 'timed_out': expired}) + '\n')
raise SystemExit(code)
PY
if [ "$(uname -s)" = Darwin ]; then
    # Standard Mach-O packaging metadata, not a compiler/kernel modification.
    install_name_tool -id @rpath/libmojolearn_adapter.so "$out/libmojolearn_adapter.so"
    # Changing a load command invalidates the arm64 ad-hoc signature, matching
    # the existing packaging/macos/stage_dylibs.py retention/loading policy.
    codesign --force --sign - "$out/libmojolearn_adapter.so"
    otool -L "$out/libmojolearn_adapter.so" > "$out/native-libraries.txt"
    nm -g "$out/libmojolearn_adapter.so" > "$out/native-symbols.txt"
else
    ldd "$out/libmojolearn_adapter.so" > "$out/native-libraries.txt"
    nm -D "$out/libmojolearn_adapter.so" > "$out/native-symbols.txt"
fi
shasum -a 256 "$out/libmojolearn_adapter.so" > "$out/native.sha256"
# An interpreter dependency is not admitted for this standalone native bridge.
if grep -Eq 'libpython|[[:space:]]U _?Py[A-Za-z_]' "$out/native-libraries.txt" "$out/native-symbols.txt"; then
    echo 'adapter unexpectedly depends on CPython' >&2
    exit 1
fi
