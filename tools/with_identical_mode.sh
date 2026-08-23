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
set -e

NUMERICS="$(dirname "$0")/../mojo_only/numerics.mojo"
FAST_LINE='comptime GLOBAL_NUMERIC_MODE = NUMERIC_FAST'
IDENT_LINE='comptime GLOBAL_NUMERIC_MODE = NUMERIC_IDENTICAL'

if ! grep -q "^$FAST_LINE\$" "$NUMERICS"; then
    if grep -q "^$IDENT_LINE\$" "$NUMERICS"; then
        echo "with_identical_mode: the tree is ALREADY in IDENTICAL mode."
        echo "  Refusing to flip: this script reverts to FAST on exit and"
        echo "  would leave your deliberate edit reverted. Run the command"
        echo "  directly, or put numerics.mojo back to NUMERIC_FAST first."
        exit 2
    fi
    echo "with_identical_mode: cannot find the mode line in $NUMERICS" >&2
    exit 2
fi

cp "$NUMERICS" "$NUMERICS.identbak"
revert() { mv -f "$NUMERICS.identbak" "$NUMERICS" 2>/dev/null || true; }
trap revert EXIT INT TERM

sed -i.tmp "s/^$FAST_LINE\$/$IDENT_LINE/" "$NUMERICS"
rm -f "$NUMERICS.tmp"

"$@"
