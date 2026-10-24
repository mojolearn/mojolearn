#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# THE FRAMEWORK ADAPTERS OF ONE RELEASE SET, built beside its bindings.
#
#   bash packaging/adapters/build.sh OUT LOGDIR VENDOR ARCH
#
#   VENDOR ARCH   cuda sm_NN | hip gfxNNN   on a Linux release leg (packaging/linux/build_sets.sh)
#                 metal apple               on the macOS release build (packaging/macos/build_release_wheel.sh)
#
# OUT receives exactly the shipped files and one build record:
#   libmojolearn_adapter.so   the native adapter library (adapters/native/c_api.mojo), compiled
#                             with the IDENTICAL tier's flags for this set's target, the same
#                             kernels the set's bindings run
#   mojolearn_torch.so        the PyTorch dispatcher registration (adapters/torch/registration.cpp)
#                             linked against the pinned Torch wheel of this vendor (pins.json)
#   mojolearn_ops.mojoc       the MAX custom-operation package (adapters/max_ops)
#   adapters-build.json       commands, toolchain, Torch and source provenance; turned into the
#                             shipped adapters.json by `manifest.py finalize` once the files are
#                             staged (RUNPATH / install names rewritten, signatures redone)
# LOGDIR keeps the compiler logs, the exact commands and the extracted Torch build tree.
#
# Build glue only. The caller bounds the time (build_sets.sh: with_binding_timeout).
# Nothing here runs a GPU, imports torch or max, or installs anything.
set -euo pipefail
OUT=${1:?OUT}
LOGDIR=${2:?LOGDIR}
VENDOR=${3:?VENDOR cuda|hip|metal}
ARCH=${4:?ARCH}
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT"
PY=${MOJOLEARN_ADAPTER_PYTHON:-python3}
say() { echo "[$(date +%T) adapters] $*"; }
die() { echo "[$(date +%T) adapters] REFUSED: $*" >&2; exit 2; }

# The forced host-staging route is a qualification artifact only; a shipped
# adapter never carries it (adapters/PACKAGING_HANDOFF_20261023.txt).
[[ -z "${MOJOLEARN_ADAPTER_TEST_HOST_STAGING:-}" ]] || die 'MOJOLEARN_ADAPTER_TEST_HOST_STAGING is a test-only route'
case "$(uname -s):$VENDOR" in
  Linux:cuda) [[ "$ARCH" =~ ^sm_[0-9]+a?$ ]] || die "cuda needs an sm_NN architecture, got $ARCH"; PLATFORM=linux ;;
  Linux:hip)  [[ "$ARCH" =~ ^gfx[0-9a-f]+$ ]] || die "hip needs a gfxNNN architecture, got $ARCH"; PLATFORM=linux ;;
  Darwin:metal) [[ "$ARCH" = apple ]] || die "metal builds use ARCH apple"; PLATFORM=macos ;;
  *) die "unsupported host/vendor $(uname -s):$VENDOR" ;;
esac
mkdir -p "$OUT" "$LOGDIR"
OUT=$(cd "$OUT" && pwd)
LOGDIR=$(cd "$LOGDIR" && pwd)
[[ -z $(ls -A "$OUT") ]] || die "$OUT is not empty"
NATIVE="$OUT/libmojolearn_adapter.so"
TORCH_SO="$OUT/mojolearn_torch.so"
MOJOC="$OUT/mojolearn_ops.mojoc"

# ------------------------------------------------ 1. the native adapter library
# THE IDENTICAL TIER'S FLAGS, as bindings/merged_build.sh composes them: the
# pinned host CPU, ONE accelerator target on Linux, none on macOS (a forced
# --target-accelerator there omits the embedded Metal AIR; bindings/build.sh),
# the identical define and the leg's kernel column.
mojo_flags=(--emit shared-lib -j "${MOJOLEARN_COMPILE_JOBS:-2}" -D MOJOLEARN_NUMERIC_IDENTICAL=1)
link_flags=()
if [[ "$PLATFORM" = linux ]]; then
  mojo_flags+=(--target-cpu "${MOJOLEARN_LINUX_CPU:-x86-64-v3}" --target-accelerator "$ARCH")
else
  unset MACOSX_DEPLOYMENT_TARGET
  sdk=$(xcrun --sdk macosx --show-sdk-version)
  mojo_flags+=(--target-cpu apple-m1)
  # the floor every macOS binding links with (bindings/build.sh MACOS_FLOOR)
  link_flags=(-Xlinker -platform_version -Xlinker macos -Xlinker 11.0 -Xlinker "$sdk")
fi
if [[ -n "${MOJOLEARN_TARGET_COLUMN:-}" ]]; then
  [[ "$MOJOLEARN_TARGET_COLUMN" =~ ^[a-z]+$ ]] || die "MOJOLEARN_TARGET_COLUMN=$MOJOLEARN_TARGET_COLUMN"
  mojo_flags+=(-D "MOJOLEARN_COLUMN_$(printf %s "$MOJOLEARN_TARGET_COLUMN" | tr '[:lower:]' '[:upper:]')")
fi
native_cmd=(pixi run mojo build "${mojo_flags[@]}" "${link_flags[@]}" -I . -I bindings
            adapters/native/c_api.mojo -o "$NATIVE")
"$PY" packaging/adapters/manifest.py command "$LOGDIR/native-command.json" -- "${native_cmd[@]}"
say "native library: ${native_cmd[*]}"
"${native_cmd[@]}" > "$LOGDIR/native-build.log" 2>&1 || {
  grep -m 3 -B 2 -A 8 'error' "$LOGDIR/native-build.log" | cut -c1-240 >&2 || true
  die "the native adapter library did not build (log $LOGDIR/native-build.log)"; }
if [[ "$PLATFORM" = macos ]]; then
  # The Torch registration records this install name as its load command, so
  # it resolves beside itself in the installed package. A load-command change
  # voids the ad-hoc signature; sign again (packaging/macos/stage_dylibs.py).
  install_name_tool -id @loader_path/libmojolearn_adapter.so "$NATIVE"
  codesign --force --sign - "$NATIVE"
  otool -L "$NATIVE" > "$LOGDIR/native-libraries.txt"
  nm -g "$NATIVE" > "$LOGDIR/native-symbols.txt"
else
  readelf -d "$NATIVE" > "$LOGDIR/native-libraries.txt"
  nm -D "$NATIVE" > "$LOGDIR/native-symbols.txt"
  # The architecture read back from the bytes, the rule build_sets.sh applies
  # to every binding: exactly this set's target.
  got=$({ strings -a "$NATIVE" 2>/dev/null || true; } | grep -oE '\b(sm_[0-9]+[a-z]*|gfx[0-9a-f]+)\b' | sort -u | tr '\n' ',' | sed 's/,$//')
  [[ "$got" = "$ARCH" ]] || die "the native adapter library carries '$got', the set is $ARCH"
fi
for sym in mojolearn_adapter_abi mojolearn_adapter_operation mojolearn_adapter_operation_host_buffers mojolearn_adapter_train; do
  grep -Eq "[[:space:]]T _?$sym\$" "$LOGDIR/native-symbols.txt" || die "the native adapter library does not export $sym"
done
# A standalone C ABI library: no interpreter dependency.
if grep -Eq 'libpython|[[:space:]]U _?Py[A-Za-z_]' "$LOGDIR/native-libraries.txt" "$LOGDIR/native-symbols.txt"; then
  die 'the native adapter library depends on CPython'
fi

# ------------------------------------------------ 2. the MAX custom-operation package
# A .mojoc holds non-elaborated Mojo: MAX compiles it for the device at graph
# load, against the max release the `max` extra pins (pins.json).
max_cmd=(pixi run mojo precompile adapters/max_ops -o "$MOJOC")
"$PY" packaging/adapters/manifest.py command "$LOGDIR/max-command.json" -- "${max_cmd[@]}"
say "MAX package: ${max_cmd[*]}"
"${max_cmd[@]}" > "$LOGDIR/max-build.log" 2>&1 || {
  grep -m 3 -B 2 -A 8 'error' "$LOGDIR/max-build.log" | cut -c1-240 >&2 || true
  die "the MAX package did not build (log $LOGDIR/max-build.log)"; }

# ------------------------------------------------ 3. the PyTorch registration
# Against the exact Torch wheel this vendor's users install (pins.json): its
# headers, its link libraries and its C++ ABI. Fetched and verified by
# fetch_torch.py; torch is never imported here.
"$PY" packaging/adapters/fetch_torch.py --vendor "$VENDOR" --dest "$LOGDIR/torch-build" \
  --cache "${MOJOLEARN_ADAPTER_TORCH_CACHE:-$HOME/.cache/mojolearn-adapters}" > "$LOGDIR/torch-fetch.log" 2>&1 || {
  tail -n 5 "$LOGDIR/torch-fetch.log" >&2; die "the pinned Torch wheel could not be fetched (log $LOGDIR/torch-fetch.log)"; }
TB="$LOGDIR/torch-build"
field() { "$PY" -c 'import json,sys; v=json.load(open(sys.argv[1]))[sys.argv[2]]; print("" if v is None else v)' "$TB/torch-build.json" "$1"; }
TORCH_INC=$(field include_dir)
TORCH_API_INC=$(field api_include_dir)
TORCH_LIB=$(field library_dir)
CXX11_ABI=$(field cxx11_abi)
CXX=${CXX:-c++}
command -v "$CXX" > /dev/null || die "no C++ compiler ($CXX); the Torch registration needs a C++20 compiler on the build host"
cxx_flags=(-std=c++20 -O2 -fPIC -I adapters/native -I "$TB/$TORCH_INC" -I "$TB/$TORCH_API_INC"
           # TORCH_CHECK embeds __FILE__: no build-host path in the shipped bytes
           "-ffile-prefix-map=$ROOT/=" "-ffile-prefix-map=$TB/=")
if [[ "$PLATFORM" = linux ]]; then
  [[ "$CXX11_ABI" =~ ^[01]$ ]] || die "the Torch wheel names no C++ ABI (TorchConfig.cmake)"
  # -z defs: every symbol resolves now, against this Torch and our library.
  # --as-needed before the libraries: no DT_NEEDED on a library nothing uses
  # (the wheel's platform-math audit refuses a stray libm). No RUNPATH here:
  # packaging/linux/stage_libs.py writes the package-relative one.
  torch_cmd=("$CXX" "${cxx_flags[@]}" -shared "-D_GLIBCXX_USE_CXX11_ABI=$CXX11_ABI" adapters/torch/registration.cpp
             -o "$TORCH_SO" -L "$TB/$TORCH_LIB" -L "$OUT" -Wl,--as-needed
             -lmojolearn_adapter -ltorch -ltorch_cpu -lc10 -Wl,-z,defs)
else
  # The adapter library by path: ld64 records its install name
  # (@loader_path/libmojolearn_adapter.so), never this path. Torch is found
  # through the installed torch package beside mojolearn in site-packages.
  torch_cmd=("$CXX" "${cxx_flags[@]}" -dynamiclib -mmacosx-version-min=11.0 adapters/torch/registration.cpp
             -o "$TORCH_SO" -Wl,-install_name,@loader_path/mojolearn_torch.so
             "$NATIVE" -L "$TB/$TORCH_LIB" -ltorch -ltorch_cpu -lc10
             -Wl,-undefined,error -Wl,-rpath,@loader_path/../../torch/lib)
fi
"$PY" packaging/adapters/manifest.py command "$LOGDIR/torch-command.json" -- "${torch_cmd[@]}"
say "Torch registration: ${torch_cmd[*]}"
"${torch_cmd[@]}" > "$LOGDIR/torch-build.log" 2>&1 || {
  grep -m 3 -B 2 -A 8 'error' "$LOGDIR/torch-build.log" | cut -c1-240 >&2 || true
  die "the Torch registration did not build (log $LOGDIR/torch-build.log)"; }
if [[ "$PLATFORM" = macos ]]; then
  codesign --force --sign - "$TORCH_SO"
  otool -L "$TORCH_SO" > "$LOGDIR/torch-libraries.txt"
  otool -l "$TORCH_SO" > "$LOGDIR/torch-load-commands.txt"
  nm -g "$TORCH_SO" > "$LOGDIR/torch-symbols.txt"
  grep -q '@loader_path/libmojolearn_adapter.so' "$LOGDIR/torch-libraries.txt" \
    || die 'the Torch registration does not load the adapter library from beside itself'
else
  readelf -d "$TORCH_SO" > "$LOGDIR/torch-libraries.txt"
  nm -D "$TORCH_SO" > "$LOGDIR/torch-symbols.txt"
  grep -q 'Shared library: \[libmojolearn_adapter.so\]' "$LOGDIR/torch-libraries.txt" \
    || die 'the Torch registration does not need libmojolearn_adapter.so by name'
  if grep -Eq '\((RPATH|RUNPATH)\)' "$LOGDIR/torch-libraries.txt"; then die 'the Torch registration carries a build-host search path'; fi
fi
# The dispatcher registration links libtorch, never Torch's Python binding.
if grep -Eq 'libtorch_python|libpython|[[:space:]]U _?Py[A-Za-z_]' "$LOGDIR/torch-libraries.txt" "$LOGDIR/torch-symbols.txt"; then
  die 'the Torch registration depends on CPython or libtorch_python'
fi

# ------------------------------------------------ 4. the build record
"$PY" packaging/adapters/manifest.py record "$OUT" --vendor "$VENDOR" --arch "$ARCH" --platform "$PLATFORM" \
  --logs "$LOGDIR" --torch-build "$TB/torch-build.json" \
  --pixi-prefix "$ROOT/.pixi/envs/${MOJOLEARN_BUILD_PIXI_ENV:-default}" \
  --mojo-version "$(pixi run mojo --version 2>/dev/null | tail -1)" \
  --source-commit "${MOJOLEARN_COMMIT:-$(git rev-parse HEAD 2>/dev/null || cat commit.txt)}"
say "built $(ls "$OUT" | tr '\n' ' ')"
