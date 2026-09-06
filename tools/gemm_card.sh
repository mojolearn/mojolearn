#!/bin/sh
# The `gemm.fp32.v1` identity card: emit one arm, or emit two and PROVE they
# differ.
#
#   tools/gemm_card.sh oracle                       one card, default path
#   tools/gemm_card.sh serial /tmp/s.card           one card, chosen path
#   tools/gemm_card.sh device                       Phase 2b's kernel
#   tools/gemm_card.sh compare                      BOTH host arms + the diff
#
# This drives `bench/gemm_card_main.mojo`, whose docstring is the authority on
# what a card contains. This script's job is the three things a driver has to
# do that the emitter cannot do for itself: run it under the right numeric
# mode, VERIFY the mode it actually got, and turn two cards into a verdict.
#
# WHY `compare` ASSERTS DIVERGENCE AND NOT AGREEMENT
# --------------------------------------------------
# Every other identity gate in this repo is green when two runs agree. This
# one is green when they DISAGREE, and that inversion is the point rather
# than a quirk.
#
# The `oracle` and `serial` arms compute the same product with the same leaf
# rule and differ in exactly one thing: `oracle` folds the P leaf partials
# with contract section 7.2's fixed balanced tree, `serial` folds them with
# the superseded serial ascending chain recorded in 7.5. So:
#
#     P == 1  -> section 7.3 says NO fold addition happens at all, so the two
#                arms are the same arithmetic and MUST agree, bit for bit.
#     P >  1  -> the tree and the chain are different summation orders, so
#                they MUST differ at some cell.
#
# Both halves are checked, per shape, by name. Checking only "they diverge
# somewhere" would pass on a build where the balanced tree had stopped being
# reached and something else had broken instead. **If the two arms agree
# everywhere, the balanced tree has stopped being reached and the whole
# profile is a no-op -- a green "identical" there is a FAILURE, and this
# script says so in those words.**
#
# THE FIXTURE IS CHECKED BEFORE THE PRODUCT. The card's first two stages per
# shape are hashes of the raw input bytes. If those ever differ between the
# two arms, the `.out` divergence is a diff of the FIXTURES and not of the
# folds, and every conclusion below it is void -- so that is a hard error
# with its own message, not a line in the same table.
#
# THE MODE IS READ BACK FROM THE RUN, NEVER ASSUMED FROM THE FLAG. The card
# is a claim about NUMERIC_IDENTICAL, and `bench/gemm_card_main.mojo` prints
# the mode it was COMPILED with in its `== ... [IDENTICAL] ==` banner. A
# build that lands in another session's window compiles the other arm and
# labels itself consistently all the way down, so the flag proves nothing.
# That guard has caught three real mislabelled measurements in this repo in
# one day (DEVIATION 514, and see tools/with_identical_mode.sh's history
# note). It stays even now that the mode is a `-D` define with no window to
# land in, because a mis-plumbed define is exactly as invisible as a lost
# flip and this line is the only thing that sees either.
#
# THE LOCK IS TAKEN PER RUN, NOT ONCE AROUND THE WHOLE SCRIPT. Four sessions
# share this checkout. `tools/with_identical_mode.sh` takes the build lock
# itself, and wrapping each `mojo run` individually rather than re-exec'ing
# this whole script under one lock means a `compare` (two builds) or a column
# sweep (six) releases the lock between runs instead of holding it for five
# minutes. Nothing here is a timing measurement, so there is nothing to keep
# serialized beyond the build itself.
#
# ENVIRONMENT
#   MOJOLEARN_GEMM_CARD_ARM       arm when argv gives none  (default oracle)
#   MOJOLEARN_GEMM_CARD_OUT       output dir for cards+logs
#   MOJOLEARN_GEMM_CARD_DEFINES   extra `-D` flags spliced into the build.
#                                 tools/gemm_column_invariance.sh drives the
#                                 vendor columns through this, so the mode
#                                 read-back above is written once and both
#                                 gates get it.
#   MOJOLEARN_GEMM_CARD_FULL      passed through: 1 removes the host cap
#   MOJOLEARN_GEMM_CARD_FAST      1 = run under FAST instead of IDENTICAL.
#                                 The read-back then REQUIRES the FAST label,
#                                 so this exercises the guard in both
#                                 directions rather than disabling it.
#   MOJOLEARN_GEMM_CARD_COMPARE_ARMS
#                                 the pair `compare` emits, default
#                                 "oracle serial". THIS IS THE SABOTAGE
#                                 HANDLE, and it is here because the
#                                 agree-everywhere branch is the rarest and
#                                 most important one in this file: a branch
#                                 that has never been made to fire is not a
#                                 branch anyone should trust. Set it to
#                                 "oracle oracle" and the run MUST go red
#                                 with the no-op message. If it does not, the
#                                 check is inert.
#   MOJOLEARN_GEMM_CARD_SABOTAGE_INPUT
#                                 1 = corrupt one input-stage hash, to prove
#                                 the fixture gate below is reached. Same
#                                 reasoning as above; see its comment.
set -e

cd "$(dirname "$0")/.."

CARD_MAIN="bench/gemm_card_main.mojo"
DIFFER="tools/identity_trace_diff.py"

OUT="${MOJOLEARN_GEMM_CARD_OUT:-bench/results/gemm_card/$(date +%Y-%m-%d_%H%M%S)}"
EXTRA_DEFINES="${MOJOLEARN_GEMM_CARD_DEFINES:-}"

if [ "${MOJOLEARN_GEMM_CARD_FAST:-}" = "1" ]; then
    WANT_MODE="FAST"
else
    WANT_MODE="IDENTICAL"
fi

usage() {
    echo "usage: tools/gemm_card.sh <oracle|serial|device> [out.card]"
    echo "       tools/gemm_card.sh compare [outdir]"
}

# ---------------------------------------------------------------------------
# emit_card <arm> <card path> <log path>
#
# Returns 0 only when the run succeeded AND the binary was compiled in the
# mode this card claims. Everything downstream reads the card, so a card that
# exists is treated as a card that is true -- which is exactly why the two
# failure paths here have to be the ones that delete nothing and report loud.
# ---------------------------------------------------------------------------
emit_card() {
    _arm="$1"
    _card="$2"
    _log="$3"

    : > "$_card"

    if [ "$WANT_MODE" = "FAST" ]; then
        _runner="tools/with_build_lock.sh"
    else
        _runner="tools/with_identical_mode.sh"
    fi

    if MOJOLEARN_GEMM_CARD_ARM="$_arm" MOJOLEARN_IDENTITY_TRACE="$_card" \
        "$_runner" pixi run mojo run \
        ${MOJOLEARN_MOJO_DEFINES:-} $EXTRA_DEFINES \
        -I . "$CARD_MAIN" > "$_log" 2>&1; then
        :
    else
        echo "  arm $_arm: RUN FAILED (log $_log)"
        # The device arm raises a long, deliberate explanation of why it is
        # not landed yet; showing 12 lines rather than 5 keeps that message
        # intact instead of truncating it into something that reads like a
        # compiler error.
        tail -12 "$_log" | sed 's/^/      /'
        return 1
    fi

    _got=$(sed -n 's/^== bench\/gemm_card_main\.mojo \[\([A-Z]*\)\] ==$/\1/p' \
           "$_log" | head -1)
    if [ "$_got" != "$WANT_MODE" ]; then
        echo "  arm $_arm: CONTAMINATED -- the binary reports mode"
        echo "      [${_got:-<no banner>}], not [$WANT_MODE]. The card is a"
        echo "      claim about $WANT_MODE and this run cannot support it."
        echo "      Check that -D MOJOLEARN_NUMERIC_IDENTICAL reached the"
        echo "      build; log $_log"
        return 1
    fi

    # The emitter reports shapes it could not run at all. A card that quietly
    # omits a shape looks exactly like a card whose shape agreed, so the skip
    # count is surfaced here rather than left in the log.
    _skipped=$(sed -n 's/^stages: .*; \([0-9][0-9]*\) skipped$/\1/p' "$_log" \
               | head -1)
    _stages=$(sed -n 's/^stages: \([0-9][0-9]*\) over .*/\1/p' "$_log" | head -1)
    echo "  arm $_arm: [$_got] ${_stages:-?} stages, ${_skipped:-?} shapes skipped -> $_card"
    return 0
}

# ---------------------------------------------------------------------------
# compare_cards <oracle log> <oracle card> <serial card>
#
# The per-shape verdict. `P` comes from the RUN'S OWN LOG, not from a copy of
# `contract_leaf_size` reimplemented here in awk: a gate that recomputes the
# quantity it is gating agrees with itself by construction and would go on
# passing after the profile constants changed underneath it.
# ---------------------------------------------------------------------------
compare_cards() {
    awk -v L="$1" -v A="$2" -v B="$3" '
    FILENAME == L {
        # "   name m=.. (capped from ..) n=.. k=.. P=32" -- the middle fields
        # are variable width, so key off the first field and the last.
        if ($NF ~ /^P=/) { order[++n] = $1; pv[$1] = substr($NF, 3) }
        next
    }
    FILENAME == A { if ($2 ~ /\.out$/) { t = $2; sub(/\.out$/, "", t); ah[t] = $5 } next }
    FILENAME == B { if ($2 ~ /\.out$/) { t = $2; sub(/\.out$/, "", t); bh[t] = $5 } next }
    END {
        bad = 0; agreed = 0; diverged = 0; first = ""
        for (i = 1; i <= n; i++) {
            t = order[i]
            if (!(t in ah) || !(t in bh)) {
                printf "  MISSING  %-34s not in both cards\n", t
                bad++; continue
            }
            if (pv[t] + 0 == 1) {
                if (ah[t] == bh[t]) {
                    printf "  agree    %-34s P=1 -- section 7.3: no fold addition\n", t
                    agreed++
                } else {
                    printf "  BROKEN   %-34s P=1 but the arms DIFFER; with no\n", t
                    printf "           fold addition to disagree about, the leaf\n"
                    printf "           chain itself has diverged (section 7.1)\n"
                    bad++
                }
            } else {
                if (ah[t] != bh[t]) {
                    printf "  DIVERGE  %-34s P=%s -- tree != chain, as required\n", t, pv[t]
                    diverged++
                    if (first == "") first = t
                } else {
                    printf "  AGREED   %-34s P=%s but the arms MATCH\n", t, pv[t]
                    bad++
                }
            }
        }
        printf "\n  %d shapes diverged (P>1), %d agreed (P==1), %d wrong\n",
               diverged, agreed, bad
        if (first != "") printf "  first diverging shape: %s\n", first
        if (diverged == 0) {
            print ""
            print "  THE BALANCED TREE HAS STOPPED BEING REACHED AND THE WHOLE"
            print "  PROFILE IS A NO-OP. The two arms agree everywhere, so the"
            print "  fold topology is not load-bearing in this build. A green"
            print "  \"identical\" here is a FAILURE, not a pass."
            bad++
        }
        exit (bad == 0 ? 0 : 1)
    }' "$1" "$2" "$3"
}

# ---------------------------------------------------------------------------

ARM="${1:-${MOJOLEARN_GEMM_CARD_ARM:-oracle}}"

case "$ARM" in
    -h|--help|help)
        usage
        exit 0
        ;;
    compare)
        # NOT `[ -n "$2" ] && OUT="$2"`: under `set -e` a false test in that
        # form is the script's last command status and kills the run.
        if [ -n "${2:-}" ]; then OUT="$2"; fi
        mkdir -p "$OUT"
        pair="${MOJOLEARN_GEMM_CARD_COMPARE_ARMS:-oracle serial}"
        arm_a=$(echo "$pair" | awk '{print $1}')
        arm_b=$(echo "$pair" | awk '{print $2}')
        echo "== gemm.fp32.v1 card compare: $arm_a vs $arm_b, $WANT_MODE =="
        echo "   out: $OUT"
        echo
        ocard="$OUT/a_${arm_a}.card"; olog="$OUT/a_${arm_a}.log"
        scard="$OUT/b_${arm_b}.card"; slog="$OUT/b_${arm_b}.log"
        emit_card "$arm_a" "$ocard" "$olog"
        emit_card "$arm_b" "$scard" "$slog"
        echo

        # THE FIXTURE GATE, BEFORE ANY PRODUCT IS COMPARED. Both arms build
        # their inputs from the same bit-assembled generator, so the `.in.*`
        # stages must be identical. If they are not, the `.out` differences
        # below are a diff of the inputs and mean nothing about the fold.
        # Extracted with awk rather than the differ, because the differ
        # requires a whole card with contiguous sequence numbers and a
        # filtered subset is not one -- feeding it a subset would report a
        # parse error and look like the guard firing.
        #
        # MOJOLEARN_GEMM_CARD_SABOTAGE_INPUT=1 corrupts one input hash in the
        # second card. Without it this branch would never once have executed
        # -- both arms share one generator, so the fixtures agree by
        # construction and a guard that has never fired is a guard nobody has
        # any reason to believe.
        if [ "${MOJOLEARN_GEMM_CARD_SABOTAGE_INPUT:-}" = "1" ]; then
            awk 'NR==2 && NF==5 { $5 = ($5 ~ /^0/ ? "1" : "0") substr($5,2) }
                 { print }' OFS='\t' "$scard" > "$scard.sab"
            mv "$scard.sab" "$scard"
            echo "  (fixture sabotage active: one input hash flipped in $arm_b)"
        fi
        awk '$2 ~ /\.in\.[ab]$/ { print $2, $3, $4, $5 }' "$ocard" \
            > "$OUT/a.inputs"
        awk '$2 ~ /\.in\.[ab]$/ { print $2, $3, $4, $5 }' "$scard" \
            > "$OUT/b.inputs"
        if ! cmp -s "$OUT/a.inputs" "$OUT/b.inputs"; then
            echo "gemm card compare: RED -- THE TWO ARMS SAW DIFFERENT INPUTS."
            echo "The input-stage hashes differ, so any product divergence"
            echo "below would be a diff of the FIXTURES, not of the folds."
            echo "Nothing about the fold topology can be concluded. First"
            echo "differing input stage:"
            diff "$OUT/a.inputs" "$OUT/b.inputs" \
                | sed 's/^/    /' | head -10 || true
            exit 1
        fi
        echo "  inputs: identical across both arms (fixture is not the variable)"
        echo

        rc=0
        compare_cards "$olog" "$ocard" "$scard" || rc=$?

        echo
        echo "  the differ's own view of the first divergence:"
        /usr/bin/python3 "$DIFFER" "$ocard" "$scard" \
            --labels "$(echo "$arm_a" | tr a-z A-Z)_A","$(echo "$arm_b" | tr a-z A-Z)_B" \
            2>&1 | sed 's/^/    /' | head -30 || true

        echo
        if [ "$rc" -ne 0 ]; then
            echo "gemm card compare: RED. Cards in $OUT"
            exit 1
        fi
        echo "gemm card compare: GREEN -- and green here means the two folds"
        echo "DISAGREE wherever P > 1 and AGREE wherever P == 1, which is"
        echo "contract sections 7.2 and 7.3 demonstrated rather than asserted."
        echo "Cards in $OUT"
        echo
        echo "This says the fold topology reaches the answer. It says nothing"
        echo "about whether the answer is RIGHT: both arms are the same host"
        echo "oracle code and a bug in the leaf loop is invisible to a"
        echo "comparison between two of its callers."
        ;;
    oracle|serial|device)
        if [ -n "${2:-}" ]; then
            card="$2"
            OUT="$(dirname "$card")"
            mkdir -p "$OUT"
            log="${card}.log"
        else
            mkdir -p "$OUT"
            card="$OUT/${ARM}.card"
            log="$OUT/${ARM}.log"
        fi
        echo "== gemm.fp32.v1 card: arm $ARM, $WANT_MODE =="
        emit_card "$ARM" "$card" "$log"
        ;;
    *)
        echo "gemm_card.sh: unknown arm '$ARM'." >&2
        usage >&2
        exit 2
        ;;
esac
