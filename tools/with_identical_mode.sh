#!/bin/sh
# Run a command in NUMERIC_IDENTICAL mode -- BY BUILD DEFINE, touching no file.
#
#   tools/with_identical_mode.sh pixi run mojo run -I . cluster/kmeans_main.mojo
#   tools/with_identical_mode.sh pixi run check-kmeans-identity
#   tools/with_identical_mode.sh bash extratrees/tools/check.sh
#
# THREE FORMS, one mechanism: `-D MOJOLEARN_NUMERIC_IDENTICAL=1`, which
# `mojo_only/numerics.mojo` reads through `is_defined` (the same shape as the
# column defines).
#   1. The argv contains `mojo run` / `mojo build`: the define is inserted
#      right after it.
#   2. The argv is `pixi run <task>`: the task's command is read from
#      pixi.toml; if it is a `mojo run`/`mojo build` the define is inserted
#      and the command runs through `pixi run -- ...`; otherwise the task
#      runs as-is with the environment below exported.
#   3. Anything else (a script): runs as-is with the environment exported.
# In every form the environment carries MOJOLEARN_NUMERIC_MODE=identical and
# MOJOLEARN_MOJO_DEFINES="-D MOJOLEARN_NUMERIC_IDENTICAL=1", which the gate
# scripts (extratrees/tools/check.sh, tools/check_linalg_*.sh,
# bindings/build_*.sh, tools/e2_mojo_cards.sh) splice into their own
# `mojo run` calls. Nested calls are harmless.
#
# HISTORY, and why this is the shape it is. Until 2026-08-23 this script
# FLIPPED `comptime GLOBAL_NUMERIC_MODE = NUMERIC_FAST` in numerics.mojo with
# sed, ran the command, and reverted on exit (first via a copy-and-move-back,
# which silently discarded any edit another session made inside the window
# -- one was lost that day -- then via a sed of the one line). A flip left
# behind by a crash mislabelled every number the next session measured, and
# the measurement-contamination guards in tools/check_linalg_*.sh exist
# because that happened three times in one day. DEVIATION 514 added the build
# lock around the window. The define removes the window: there is nothing to
# revert, nothing to lock for correctness, and two sessions can build the
# two modes at once. The build lock is still taken, because these gates are
# GPU-exclusive and serializing them keeps the timings honest.
#
# READ THE MODE BACK. Every gate that labels its output prints the mode it
# was compiled with (`_mode_name()` from the comptime constant) -- keep
# checking that line rather than trusting the flag that was passed.

set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
DEFINE="-D MOJOLEARN_NUMERIC_IDENTICAL=1"

if [ "${MOJOLEARN_IDENT_LOCK_HELD:-}" != "1" ]; then
    MOJOLEARN_IDENT_LOCK_HELD=1
    export MOJOLEARN_IDENT_LOCK_HELD
    exec "$HERE/with_build_lock.sh" "$0" "$@"
fi

export MOJOLEARN_NUMERIC_MODE=identical
export MOJOLEARN_MOJO_DEFINES="$DEFINE"

# form 2: `pixi run <task>` -> resolve the task command from pixi.toml
if [ "${1:-}" = "pixi" ] && [ "${2:-}" = "run" ] && [ -n "${3:-}" ] \
   && [ "${3}" != "mojo" ] && [ "${3}" != "--" ]; then
    task="$3"; shift 3
    # dependency-free lookup (the boxes' system python predates tomllib):
    # pixi.toml tasks are `name = "command"` lines; the last definition wins
    cmd=$(cd "$REPO" && awk -v t="$task" '
        $0 ~ "^" t " = \"" { line=$0; sub("^" t " = \"", "", line); sub("\"[[:space:]]*$", "", line); found=line }
        END { if (found == "") exit 3; print found }' pixi.toml
) || { echo "with_identical_mode: no pixi task '$task'" >&2; exit 2; }
    case "$cmd" in
        "mojo run "*)   rest=${cmd#mojo run };   exec pixi run -- mojo run $DEFINE $rest "$@" ;;
        "mojo build "*) rest=${cmd#mojo build }; exec pixi run -- mojo build $DEFINE $rest "$@" ;;
        *)              exec pixi run "$task" "$@" ;;
    esac
fi

# form 1: the define goes right after `mojo run` / `mojo build` in argv
out=""; prev=""; injected=0
for a in "$@"; do
    out="$out \"$(printf %s "$a" | sed 's/"/\\"/g')\""
    if [ $injected = 0 ] && [ "$prev" = "mojo" ] && { [ "$a" = "run" ] || [ "$a" = "build" ]; }; then
        out="$out $DEFINE"; injected=1
    fi
    prev="$a"
done
# form 3 falls through: nothing injected, environment exported
eval exec "$out"
