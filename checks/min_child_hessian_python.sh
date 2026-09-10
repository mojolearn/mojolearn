#!/bin/sh
# macOS protected shells discard inherited DYLD_ variables. Set the local
# runtime path immediately before exec, including the build's copied-package smoke.
set -eu
cd "$(dirname "$0")/.."
if [ "$(uname)" = Darwin ]; then
    DYLD_LIBRARY_PATH="$PWD/.pixi/envs/default/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
    export DYLD_LIBRARY_PATH
fi
exec "${MOJOLEARN_HESSIAN_PYTHON:-python3}" "$@"
