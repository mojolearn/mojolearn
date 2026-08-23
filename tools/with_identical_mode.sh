#!/bin/sh
# Run a command against a NUMERIC_IDENTICAL build, and put the tree back.
#
#   tools/with_identical_mode.sh pixi run mojo run -I . cluster/kmeans_main.mojo
#
# `mojo_only/numerics.mojo:GLOBAL_NUMERIC_MODE` is a COMPTIME constant, which
# is deliberate (`numerics.mojo`: "the two modes are different code") and
# means the only way to build the other arm is to edit that line. Every
# identity gate therefore needs the same three steps -- flip, run, revert --
# and `tools/e1_bootstrap.sh` already wrote them once for the cross-vendor
# run. This is that flip, factored out, so a LOCAL identity check does not
# have to bring the whole E1 remote bootstrap with it.
#
# THE REVERT IS A TRAP, NOT A TRAILING LINE. A check that fails, a Ctrl-C, or
# a compile error must not leave the working tree carrying an edit nobody
# made deliberately: the flip must never be committed (E1_RUNBOOK
# preconditions), and a stray IDENTICAL build silently changes every number
# the next session measures.
#
# ------------------------------------------------------------------------
# THE LOCK, ADDED 2026-08-23 (DEVIATION 514), AND WHY IT IS NOT OPTIONAL
# ------------------------------------------------------------------------
# This script MUTATES A SHARED FILE for the duration of a command that can
# run for minutes. `tools/with_build_lock.sh` exists because this repository
# is worked by parallel sessions in ONE checkout, and until today this
# script did not take that lock. The failure it allows is not a merge
# conflict -- it is a SILENTLY MISLABELLED MEASUREMENT:
#
#   session A: flip -> IDENTICAL ......... run (2 min) ......... revert
#   session B:              `pixi run mojo run ... kmeans_main.mojo`
#                           compiles HERE, gets an IDENTICAL binary,
#                           and prints "FAST" because `_mode_name()`
#                           reads the same constant it was compiled
#                           against. Every number in that run is the
#                           other arm's, correctly labelled as this one.
#
# It was not hypothetical. This run hit the other half of the same race:
# the refusal below fired on a tree that `grep` showed as FAST one second
# later, because a concurrent session was inside its own flip window.
#
# So the whole flip-run-revert window is held under `/tmp/cbsym-build.lock`,
# the same lock `with_build_lock.sh` takes. A session that wants a FAST
# build while this one holds the lock will block rather than compile the
# wrong arm. `MOJOLEARN_IDENT_LOCK_HELD` makes re-entry a no-op so a gate
# that nests these (`check_unsupervised_identity.sh` re-enters itself) does
# not deadlock against itself.
set -e

HERE="$(dirname "$0")"
NUMERICS="$HERE/../mojo_only/numerics.mojo"
FAST_LINE='comptime GLOBAL_NUMERIC_MODE = NUMERIC_FAST'
IDENT_LINE='comptime GLOBAL_NUMERIC_MODE = NUMERIC_IDENTICAL'

# Re-exec under the shared build lock, once. Everything below this point
# runs with the checkout's mode line owned by this process.
if [ "${MOJOLEARN_IDENT_LOCK_HELD:-}" != "1" ]; then
    MOJOLEARN_IDENT_LOCK_HELD=1
    export MOJOLEARN_IDENT_LOCK_HELD
    exec "$HERE/with_build_lock.sh" "$0" "$@"
fi

if ! grep -q "^$FAST_LINE\$" "$NUMERICS"; then
    if grep -q "^$IDENT_LINE\$" "$NUMERICS"; then
        # Reaching this WITH the lock held means a human (or a crashed run)
        # left the edit in the tree, not that another session is mid-flip.
        echo "with_identical_mode: the tree is ALREADY in IDENTICAL mode,"
        echo "  and this process holds the build lock -- so it is not a"
        echo "  concurrent session's flip window, it is an edit that was"
        echo "  left behind. Refusing: this script reverts to FAST on exit"
        echo "  and would discard it. Put numerics.mojo back to"
        echo "  NUMERIC_FAST, or run the command directly."
        exit 2
    fi
    echo "with_identical_mode: cannot find the mode line in $NUMERICS" >&2
    exit 2
fi

# ------------------------------------------------------------------------
# THE REVERT PUTS BACK THE LINE, NOT THE FILE (2026-08-23)
# ------------------------------------------------------------------------
# This used to `cp` the whole file and `mv` it back on exit. That silently
# DESTROYS any edit made to numerics.mojo during the window, and the window
# is minutes long in a checkout worked by parallel sessions. It is not
# hypothetical: the E2 lane lost an edit to a flip/revert cycle at 06:44
# today and had to re-derive it.
#
# The flip is ONE LINE, so the revert is one line. `sed` the mode line back
# and leave everything else in the file exactly as whoever else was editing
# it left it. The backup is still taken, but only as evidence for the
# refusal below -- it is never restored over the working file.
#
# The refusal at the end catches the remaining case: if the line is not
# where we left it when we exit, someone else changed the MODE underneath a
# lock-holding run, and that is worth a loud message rather than a silent
# overwrite in either direction.
cp "$NUMERICS" "$NUMERICS.identbak"
revert() {
    if grep -q "^$IDENT_LINE\$" "$NUMERICS" 2>/dev/null; then
        sed -i.tmp "s/^$IDENT_LINE\$/$FAST_LINE/" "$NUMERICS"
        rm -f "$NUMERICS.tmp"
    elif ! grep -q "^$FAST_LINE\$" "$NUMERICS" 2>/dev/null; then
        echo "with_identical_mode: WARNING -- on exit the mode line is" >&2
        echo "  neither FAST nor IDENTICAL. Not touching the file; the" >&2
        echo "  copy taken at flip time is at $NUMERICS.identbak." >&2
        return
    fi
    rm -f "$NUMERICS.identbak"
}
trap revert EXIT INT TERM

sed -i.tmp "s/^$FAST_LINE\$/$IDENT_LINE/" "$NUMERICS"
rm -f "$NUMERICS.tmp"

"$@"
