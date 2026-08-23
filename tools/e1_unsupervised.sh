#!/bin/sh
# E1 Phase 3u: the unsupervised cards, one leg of the cross-vendor run.
#
#   sh tools/e1_unsupervised.sh [out_dir]
#
# Run this ON each machine. It produces one card per arm plus the console
# hashes and an environment record, under
# `bench/results/e1u/<stamp>-<host>/`. Ship one machine's directory to the
# other and diff arm by arm:
#
#   python3 tools/identity_trace_diff.py mac/kmeans.card amd/kmeans.card
#
# WHY THIS IS NOT PART OF `tools/e1_bootstrap.sh`. That script drives the
# GBDT/ET/RF families through the Python bindings and spends most of its
# runtime building five `.so` files. `cluster/`, `neighbors/` and `dbscan/`
# have NO bindings -- which is why rows 19-26 had never been to E1 at all --
# so this leg needs none of that and can run in a couple of minutes on a
# box that has nothing but pixi. Keeping it separate also keeps two lanes
# out of one file.
#
# THE ORDER OF THE CHECKS BELOW IS THE ORDER OF THE ARGUMENT, and each step
# is worthless without the one above it:
#
#   0  the column this build resolved to, and its lane width. An AMD box
#      that silently built the apple column would agree with the Mac for
#      the least interesting reason there is.
#   1  `check-ieee-arith`. IDENTITY_PATHS row 10's precondition: what this
#      backend actually does with denormals and with `a*b+c`. RUN IT AND
#      READ IT before believing any card. The 2026-08-22 AMD leg recorded
#      "a*b+c is UNFUSED on this backend" from a counting arm later shown
#      to be an artifact (row 9's correction), so AMD's real contraction
#      behaviour is UNMEASURED and it is precisely what `identical_mul_add`
#      exists for.
#   2  `check-portable-translog`. Its printed device hash must be the SAME
#      NUMBER on every vendor (Apple: 8705486125800438413).
#   3  the local gates, IDENTICAL. A machine that fails its own gates
#      teaches nothing when diffed against another.
#   4  the three cards.
#
# THE MODE FLIP is session-local, holds the shared build lock, and reverts
# on exit (`tools/with_identical_mode.sh`, DEVIATION 514). It must never be
# committed. On a 64-lane column the FAST build of some sections is a
# compile error BY DESIGN, so IDENTICAL is not optional there.
set -u

cd "$(dirname "$0")/.."
REPO="$(pwd)"

if [ "${MOJOLEARN_E1U_INNER:-}" != "1" ]; then
    STAMP="$(date +%Y-%m-%d_%H%M%S)-$(hostname -s)"
    OUT="${1:-$REPO/bench/results/e1u/$STAMP}"
    mkdir -p "$OUT"
    echo "== E1 Phase 3u: unsupervised cards =="
    echo "   out: $OUT"
    git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || echo "unknown" > "$OUT/commit.txt"
    # THE SOURCE HASH, AND IT IS NOT REDUNDANT WITH THE COMMIT.
    #
    # A rented box usually has no `.git`: the repository is private, so the
    # source arrives as a tarball and `git rev-parse` writes "unknown".
    # E1's first precondition is "same commit on both sides", and on the
    # 2026-08-23 AMD leg that precondition was unverifiable from the
    # artifacts -- the Apple card said 3d0a842 and the AMD card said
    # "unknown", which is not a comparison, it is a belief about what was
    # shipped.
    #
    # So hash the SOURCE instead of trusting the transport. Every `.mojo`
    # file, sorted, hashed, and the hashes hashed. Two legs whose
    # `source_sha256.txt` agree ran the same program whatever their `.git`
    # says, and that is the property the claim actually needs. (Measured on
    # the AMD leg after the fact: 665d04ebefbca0f4... on both sides.)
    ( find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
        | LC_ALL=C sort \
        | xargs shasum -a 256 2>/dev/null || \
      find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
        | LC_ALL=C sort | xargs sha256sum ) \
        | { shasum -a 256 2>/dev/null || sha256sum; } \
        | awk '{print $1}' > "$OUT/source_sha256.txt"
    echo "   source sha256: $(cat "$OUT/source_sha256.txt" | cut -c1-32)"
    uname -a > "$OUT/uname.txt"
    sh bench/external/record_environment.sh > "$OUT/environment.txt" 2>&1 || true
    MOJOLEARN_E1U_INNER=1 MOJOLEARN_E1U_OUT="$OUT" \
        exec tools/with_identical_mode.sh "$0"
fi

OUT="$MOJOLEARN_E1U_OUT"
LOG="$OUT/e1u.log"
exec > "$LOG" 2>&1
step() { echo; echo "=== $* ==="; }

step "phase 0: which column did this build resolve to"
# Any arm prints it; kmeans is the cheapest. A card is written and thrown
# away -- the driver refuses to run without a trace path, deliberately.
MOJOLEARN_IDENTITY_TRACE="$OUT/.probe.card" MOJOLEARN_UNSUP_ARM=kmeans \
    pixi run mojo run -I . bench/unsupervised_trace_main.mojo 2>&1 \
    | grep -E "^arm|^mode|^column" | tee "$OUT/column.txt"
rm -f "$OUT/.probe.card"

step "phase 1: vendor characterization (row 10's precondition)"
pixi run check-ieee-arith || echo "PHASE1-FINDING: ieee-arith"

step "phase 1b: the portable transcendental certificate"
pixi run check-portable-translog || echo "PHASE1-FINDING: portable-translog"

step "phase 2: the local gates, IDENTICAL"
# Already inside the flip, so the files run directly rather than through
# check_unsupervised_identity.sh (which would try to flip again).
for f in cluster/mojo_only/kmeans_identity_check.mojo \
         neighbors/mojo_only/knn_identity_check.mojo \
         dbscan/mojo_only/dbscan_identity_check.mojo \
         cluster/kmeans_main.mojo \
         neighbors/knn_main.mojo \
         dbscan/dbscan_main.mojo; do
    echo "--- $f"
    pixi run mojo run -I . "$f" 2>&1 | grep -E "^check_|^ball_cover|Unhandled|error:" \
        || echo "PHASE2-FINDING: $f produced no check lines"
done

step "phase 3u: the cards"
for arm in kmeans knn dbscan; do
    echo "--- $arm"
    : > "$OUT/$arm.card"
    if MOJOLEARN_IDENTITY_TRACE="$OUT/$arm.card" MOJOLEARN_UNSUP_ARM="$arm" \
        pixi run mojo run -I . bench/unsupervised_trace_main.mojo 2>&1 \
        | grep -E "^arm|^mode|^column|^input\.|^output\.|^query_tile|^batches|^done|Unhandled|error:" \
        | tee "$OUT/$arm.hashes"; then
        :
    fi
    # THE MODE IS READ BACK, not assumed from the flip: this checkout is
    # worked by parallel sessions and a build landing in someone else's
    # window compiles the other arm and labels it correctly (DEVIATION 514).
    got=$(grep "^mode " "$OUT/$arm.hashes" | head -1 | awk '{print $2}')
    if [ "$got" != "IDENTICAL" ]; then
        echo "E1U-FINDING: $arm compiled as ${got:-<none>}, not IDENTICAL."
        echo "  This card is NOT usable for a cross-vendor diff."
    fi
    echo "    $(grep -c '	' "$OUT/$arm.card") stages"
done

step "done"
echo "artifacts in $OUT"
echo
echo "NEXT, on the machine that has BOTH directories:"
echo "  for a in kmeans knn dbscan; do"
echo "    diff <(grep '^input\\.' A/\$a.hashes) <(grep '^input\\.' B/\$a.hashes) \\"
echo "      || echo \"\$a: INPUTS DIFFER -- stop, the fixtures are not the same bytes\""
echo "    python3 tools/identity_trace_diff.py A/\$a.card B/\$a.card"
echo "  done"
echo
echo "Read the k-NN card knowing knn.out_dist/out_idx are PRE-SORT (an arm's"
echo "internal order, which the two arms genuinely disagree about) and"
echo "knn.sorted_dist/sorted_idx are what the caller gets. Sorted distances"
echo "agreeing over sorted indices differing = the selector chose different"
echo "members of an equidistant class, and nothing else diverged."
