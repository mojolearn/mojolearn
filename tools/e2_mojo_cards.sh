#!/bin/sh
# E2 identity cards for the training paths Python cannot reach.
#
#   tools/e2_mojo_cards.sh <out_dir>
#
# Runs `mojo_only/e2_growth_cards.mojo` (pixi task `e2-growth-cards`) ONCE
# into <out_dir>, producing one card per Mojo-only training path:
#
#   <out_dir>/gbdt_depthwise.card         depthwise growth, depth 4
#   <out_dir>/gbdt_lossguide.card         lossguide growth, depth 6 / 9 leaves
#   <out_dir>/gbdt_multiclass_ova.card    train(loss="MultiClassOneVsAll")
#   <out_dir>/gbdt_feature_parallel.card  rung-2 searcher, OUTPUT-LEVEL
#
# each ONE fit per file (tools/identity_trace_diff.py refuses more), then
# writes <out_dir>/e2_mojo_cards.json: name -> {card, record_count,
# description, control_match, numeric_mode}. The fixtures are pure
# functions of constants (hashed bins, fixed sm_count where a searcher
# takes one), so the same rows reach every vendor and the cards join the
# cross-vendor diff directly:
#
#   python3 tools/identity_trace_diff.py <mac>/gbdt_depthwise.card <amd>/gbdt_depthwise.card
#
# THE RUN-TO-RUN CONTROL. The set is emitted a SECOND time into
# <out_dir>/control/ and each card is compared byte for byte; the verdict
# is `control_match` in the JSON and a line on stdout. Under FAST on a
# backend with live float atomics a card can legitimately disagree with
# itself, and such a card must be read only under IDENTICAL
# (`tools/with_identical_mode.sh`). The control is what tells you which.
# Set E2_MOJO_CARDS_CONTROL=0 to skip it (half the time, no verdict).
#
# NOT A MEASUREMENT: every record drains and copies (identity_trace rule 4).
# Does not flip GLOBAL_NUMERIC_MODE; it reports the mode the tree is in.
set -e

if [ $# -lt 1 ]; then
    echo "usage: tools/e2_mojo_cards.sh <out_dir>" >&2
    exit 2
fi

REPO="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$1"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"
CONTROL="${E2_MOJO_CARDS_CONTROL:-1}"

CARDS="gbdt_depthwise gbdt_lossguide gbdt_multiclass_ova gbdt_feature_parallel"

# THE MODE IS A BUILD DEFINE (2026-08-23): MOJOLEARN_NUMERIC_MODE=identical in
# the environment (tools/with_identical_mode.sh exports it) runs the probe
# with -D MOJOLEARN_NUMERIC_IDENTICAL=1 through the injector; otherwise FAST.
# The probe prints the mode it was COMPILED with; that line is the truth.
if [ "${MOJOLEARN_NUMERIC_MODE:-fast}" = "identical" ]; then
    MODE=IDENTICAL
    MOJOLEARN_RUN_PREFIX="$REPO/tools/with_identical_mode.sh"
else
    MODE=FAST
    MOJOLEARN_RUN_PREFIX=""
fi

echo "e2_mojo_cards: numeric mode $MODE, out_dir $OUT"
cd "$REPO"

# THE EMISSION. One process, every card; the probe raises if any card's
# record count is under its floor, and `set -e` carries that out.
${MOJOLEARN_RUN_PREFIX:-} pixi run e2-growth-cards "$OUT"

for name in $CARDS; do
    [ -s "$OUT/$name.card" ] || { echo "e2_mojo_cards: $name.card missing or empty" >&2; exit 1; }
done

# THE CONTROL: the same set again, byte-compared. A differing card under
# FAST is reported, not hidden -- that is the finding.
if [ "$CONTROL" != "0" ]; then
    mkdir -p "$OUT/control"
    ${MOJOLEARN_RUN_PREFIX:-} pixi run e2-growth-cards "$OUT/control"
    echo "-- run-to-run control --"
    for name in $CARDS; do
        if cmp -s "$OUT/$name.card" "$OUT/control/$name.card"; then
            echo "   $name: REPRODUCIBLE [$MODE]"
        else
            echo "   $name: NOT REPRODUCIBLE run to run [$MODE] -- read this card only under IDENTICAL"
            python3 tools/identity_trace_diff.py "$OUT/$name.card" "$OUT/control/$name.card" 2>/dev/null | head -8 || true
        fi
    done
fi

# THE INDEX. Record count = non-comment, non-blank lines, which is the
# differ's own definition of a record.
python3 - "$OUT" "$MODE" "$CONTROL" $CARDS <<'EOF'
import json, os, sys
out, mode, control = sys.argv[1], sys.argv[2], sys.argv[3]
names = sys.argv[4:]
desc = {
    "gbdt_depthwise": (
        "fit_depthwise_tree on depthwise_check.Fixture (4096 rows, 8 binary"
        " + 4 half-byte + 4 one-byte features, hashed bins, divergent"
        " target), depth 4; every stage of the depthwise ladder"
    ),
    "gbdt_lossguide": (
        "fit_non_symmetric_tree under GROW_LOSSGUIDE on the same fixture,"
        " depth 6, max_leaves 9; the merged non-symmetric driver's Lossguide"
        " branches, every stage"
    ),
    "gbdt_multiclass_ova": (
        "train(loss='MultiClassOneVsAll') on multiclass_train_check's"
        " splitmix fixture (4096 rows, 5 features, 5 learnable classes),"
        " border_count 32, 10 trees, depth 4, lr 0.3; borders + per-tree"
        " hist/pstats/winners/leaves through the env-read trace"
    ),
    "gbdt_feature_parallel": (
        "fit_feature_parallel_oblivious_tree_structure (rung 2) on"
        " feature_parallel_identity_check's fixture at 16434 rows (three"
        " compression blocks), depth 4, sm_count 10; OUTPUT-LEVEL card:"
        " splits + per-document docBins, no in-searcher stages"
    ),
}
def records(path):
    n = 0
    with open(path) as fh:
        for line in fh:
            s = line.strip()
            if s and not s.startswith("#"):
                n += 1
    return n
index = {}
for name in names:
    card = os.path.join(out, name + ".card")
    entry = {
        "card": name + ".card",
        "record_count": records(card),
        "description": desc[name],
        "numeric_mode": mode,
    }
    ctl = os.path.join(out, "control", name + ".card")
    if control != "0" and os.path.exists(ctl):
        entry["control_match"] = open(card, "rb").read() == open(ctl, "rb").read()
    else:
        entry["control_match"] = None
    index[name] = entry
with open(os.path.join(out, "e2_mojo_cards.json"), "w") as fh:
    json.dump(index, fh, indent=2, sort_keys=True)
    fh.write("\n")
for name in names:
    e = index[name]
    print(f"   {name}: {e['record_count']} records, control_match={e['control_match']}")
EOF

echo "e2_mojo_cards: wrote $OUT/e2_mojo_cards.json"
