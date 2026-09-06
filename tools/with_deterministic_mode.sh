#!/bin/sh
# Run a command in NUMERIC_DETERMINISTIC mode -- BY BUILD DEFINE, touching no
# file. The sibling of tools/with_identical_mode.sh, same mechanism and same
# reasons; read that file's header for why this is a define and not a flip.
#
#   tools/with_deterministic_mode.sh pixi run mojo run -I . <driver>.mojo
#
# READ THE MODE BACK. Every gate that labels its output prints the mode it was
# COMPILED with. `numeric_mode_name()` returns DETERMINISTIC; the 63 older
# `_mode_name` helpers predate the tier and still say FAST, so a driver that
# has not been moved onto `numeric_mode_name` will MISLABEL itself here.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
DEFINE="-D MOJOLEARN_NUMERIC_DETERMINISTIC=1"
if [ "${MOJOLEARN_DET_LOCK_HELD:-}" != "1" ]; then
    MOJOLEARN_DET_LOCK_HELD=1
    export MOJOLEARN_DET_LOCK_HELD
    exec "$HERE/with_build_lock.sh" "$0" "$@"
fi
export MOJOLEARN_NUMERIC_MODE=deterministic
export MOJOLEARN_MOJO_DEFINES="$DEFINE"
out=""; prev=""; injected=0
for a in "$@"; do
    out="$out \"$(printf %s "$a" | sed 's/"/\\"/g')\""
    if [ $injected = 0 ] && [ "$prev" = "mojo" ] && { [ "$a" = "run" ] || [ "$a" = "build" ]; }; then
        out="$out $DEFINE"; injected=1
    fi
    prev="$a"
done
eval "exec $out"
