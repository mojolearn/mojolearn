#!/bin/sh
# Set the local Mojo runtime immediately before Python: protected macOS shells
# may discard an inherited DYLD_LIBRARY_PATH.
set -eu
cd "$(dirname "$0")/.."
if [ "$(uname)" = Darwin ]; then
    DYLD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export DYLD_LIBRARY_PATH
fi
exec "${MOJOLEARN_PIPELINE_PYTHON:-python3}" "$@"
