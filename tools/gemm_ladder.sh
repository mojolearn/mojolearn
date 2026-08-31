#!/bin/sh
# The `gemm.fp32.v1` FOLD-LADDER card: emit one, or emit two and say WHICH
# LEVEL of the reduction tree moved first.
#
#   tools/gemm_ladder.sh emit                     one card, timestamped dir
#   tools/gemm_ladder.sh emit /tmp/mine           one card, chosen dir
#   tools/gemm_ladder.sh diff A.card B.card       localize a divergence
#   tools/gemm_ladder.sh sabotage FOLD_STRIDE     clean vs broken, localized
#
# DEVIATION 533. This drives `bench/gemm_ladder_main.mojo`, whose docstring is
# the authority on what a ladder card contains and how the workspace is laid
# out. This script does the four things the emitter cannot do for itself: run
# it under the right numeric mode, VERIFY the mode and the sabotage it
# actually got, put the cards somewhere with the commit beside them, and turn
# two cards into a LEVEL.
#
# WHAT THIS TOOL IS FOR
# ---------------------
# `tools/gemm_card.sh` emits a card that hashes a shape's inputs and its
# output. When a cross-vendor run diverges, that card says
#
#     llama8b.mlp_down.t1.out   DIFFERS
#
# and stops. Three unrelated investigations produce that exact line: a leaf
# ARITHMETIC defect (a contraction or a denormal policy that did not pin), a
# fold TOPOLOGY defect (the leaves agree and the tree pairs them differently),
# and a CARRY defect (everything agrees but the odd tail). This tool hashes
# every level of the tree, so the answer is instead
#
#     level 0 agrees, level 1 moved  -> the leaves are fine, the FOLD is not
#
# which names the file to open. The GBDT lane's cross-vendor run found an
# NVIDIA divergence at `tree001.winners.scores` while Apple and AMD agreed;
# it found it because there was a stage there, and a final-token comparison
# cannot produce that class of finding.
#
# WHAT THIS TOOL CANNOT TELL YOU
# -------------------------------
#  * **It runs ONE execution plan**, `PLAN_SPLITK_STAGED`, because that is the
#    only plan that materializes every node of the tree in memory where a host
#    can read it. The plan the dispatcher actually ships folds in registers and
#    materializes nothing. If those two plans ever disagreed, this card would
#    show the staged one and would not notice -- that is
#    `check_device_is_launch_invariant`'s job in
#    `gemm/original/gemm_device_check.mojo`, and it is a different gate.
#    **This is a localizer, not an invariance gate.**
#  * **Two cards agreeing does not prove the two runs computed identically.**
#    It proves they agree at these checkpoints, on this fixture. The mechanism
#    ledger is `IDENTITY_PATHS.md`; the instrument only localizes what the
#    ledger missed.
#  * **It says nothing about speed.** Every record drains the queue and copies
#    a buffer to the host, and the driver additionally reads the entire
#    workspace back. A timing taken under this is fiction.
#  * **A level that agrees is not a level that was exercised.** At `P == 1`
#    there is one level and no fold addition at all (contract 7.3); at
#    `P == 2` a stride-paired fold and an adjacent-paired fold are the same
#    pairing. `sabotage` below is how you find out which levels this fixture
#    can actually separate, and the answer for some defects is "none" -- see
#    EXIT CODES.
#
# THE MODE AND THE SABOTAGE ARE BOTH READ BACK FROM THE RUN
# ----------------------------------------------------------
# The card is a claim about `NUMERIC_IDENTICAL`, and a build that compiled the
# other arm labels itself consistently all the way down, so the flag proves
# nothing. `bench/gemm_ladder_main.mojo` prints the mode it COMPILED with in
# its banner and this script requires that line to match. That guard has
# caught three real mislabelled measurements in this repository in one day
# (DEVIATION 514, and `tools/with_identical_mode.sh`'s history note).
#
# THE SAME GUARD IS APPLIED TO THE SABOTAGE, and it matters more here than the
# mode does. A `-D MOJOLEARN_GEMM_SABOTAGE_*` that is passed and silently
# dropped produces a card IDENTICAL to the clean one -- which reads exactly
# like "the ladder cannot see this defect" when the truth is "the defect was
# never built". So the driver prints `sabotage <NAME>` from the comptime
# constant the kernel compiled against, and this script refuses a run whose
# witness is not the sabotage that was asked for.
#
# THE LOCK IS TAKEN PER RUN, NOT ONCE AROUND THE SCRIPT. Four sessions share
# this checkout. `tools/with_identical_mode.sh` takes the build lock itself
# (see its `MOJOLEARN_IDENT_LOCK_HELD` re-exec), so wrapping each `mojo run`
# individually rather than re-exec'ing this whole script under one lock means
# a `sabotage` run (two builds) releases the lock between them instead of
# holding it across both. Nothing here is a timing, so there is nothing to
# keep serialized beyond the build itself. **Do not add a second
# `with_build_lock.sh` around `with_identical_mode.sh`** -- it is not a
# deadlock, the lock is re-entrant by env flag, but it is a lie about what is
# being serialized.
#
# ENVIRONMENT
#   MOJOLEARN_GEMM_LADDER_OUT       output dir (default: a timestamped dir
#                                   under bench/results/gemm_ladder/ named
#                                   with the commit SHA)
#   MOJOLEARN_GEMM_LADDER_SHAPES    passed through: comma-separated table row
#                                   indices, e.g. "0,3,8". Default is the
#                                   driver's own list (P = 1, 32, 112, 512,
#                                   1024).
#   MOJOLEARN_GEMM_LADDER_DEFINES   extra `-D` flags spliced into the build.
#   MOJOLEARN_GEMM_LADDER_DUMP      substring; every record whose tag contains
#                                   it also writes a raw `.bin` beside the
#                                   card, so the differ can go per NODE.
#                                   "fold" is the useful value.
#   MOJOLEARN_GEMM_LADDER_FAST      1 = run under FAST instead of IDENTICAL.
#                                   The read-back then REQUIRES the FAST
#                                   label, so this exercises the guard in both
#                                   directions rather than disabling it.
#
# EXIT CODES
#   0  what was asked for succeeded (and for `sabotage`: the broken build
#      moved something this card can see)
#   1  a run failed, a witness was wrong, or `diff` found a divergence
#   2  usage error
#   3  `sabotage` only: THE SABOTAGE WAS BUILT AND THE CARD DID NOT MOVE.
#      Not an error in this script -- a finding about the FIXTURE. It means
#      this ladder cannot see that defect class, which is the failure mode
#      row 9's correction came from: 2^20 patterns that scored a contracting
#      backend as unfused because not one of them separated the two
#      spellings. Report it; do not tune the fixture until it goes green.
set -e

cd "$(dirname "$0")/.."

LADDER_MAIN="bench/gemm_ladder_main.mojo"
DIFFER="tools/identity_trace_diff.py"

if [ "${MOJOLEARN_GEMM_LADDER_FAST:-}" = "1" ]; then
    WANT_MODE="FAST"
else
    WANT_MODE="IDENTICAL"
fi

# The commit the cards were produced at, beside the timestamp. `-dirty` is
# not decoration: four sessions share this checkout and a card emitted over
# an uncommitted edit is not a card for that SHA.
SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    SHA="${SHA}-dirty"
fi
export MOJOLEARN_GEMM_LADDER_SHA="$SHA"

OUT="${MOJOLEARN_GEMM_LADDER_OUT:-bench/results/gemm_ladder/$(date +%Y-%m-%d_%H%M%S)_${SHA}}"

usage() {
    echo "usage: tools/gemm_ladder.sh emit [outdir]"
    echo "       tools/gemm_ladder.sh diff <a.card> <b.card>"
    echo "       tools/gemm_ladder.sh sabotage <NAME> [outdir]"
    echo "       NAME is one of LEAF_READS_LAUNCH FOLD_STRIDE PAD_PLUS_ZERO"
    echo "                      FOLD_SERIAL NODE_ORDER LEAF_ROTATE"
}

# ---------------------------------------------------------------------------
# emit_card <card path> <log path> <want sabotage name> [extra defines...]
#
# Returns 0 only when the binary compiled in the mode AND with the sabotage
# this card claims. The RUN's own exit status is reported but is NOT the
# verdict: a sabotage build is supposed to fail its own proofs, and treating
# that as an emit failure would throw away the card that shows why.
# ---------------------------------------------------------------------------
emit_card() {
    _card="$1"; shift
    _log="$1"; shift
    _want_sab="$1"; shift

    # The env constructor of IdentityTrace APPENDS (two fits in one process
    # share a file deliberately), so a stale card here would be read back as
    # this run's records concatenated with the last one's.
    : > "$_card"

    if [ "$WANT_MODE" = "FAST" ]; then
        _runner="tools/with_build_lock.sh"
    else
        _runner="tools/with_identical_mode.sh"
    fi

    _rc=0
    MOJOLEARN_IDENTITY_TRACE="$_card" \
    MOJOLEARN_IDENTITY_TRACE_DUMP="${MOJOLEARN_GEMM_LADDER_DUMP:-}" \
        "$_runner" pixi run mojo run \
        ${MOJOLEARN_MOJO_DEFINES:-} ${MOJOLEARN_GEMM_LADDER_DEFINES:-} "$@" \
        -I . "$LADDER_MAIN" > "$_log" 2>&1 || _rc=$?

    _got=$(sed -n 's/^== bench\/gemm_ladder_main\.mojo \[\([A-Z]*\)\] ==$/\1/p' \
           "$_log" | head -1)
    if [ "$_got" != "$WANT_MODE" ]; then
        echo "  CONTAMINATED -- the binary reports mode [${_got:-<no banner>}],"
        echo "      not [$WANT_MODE]. The card is a claim about $WANT_MODE and"
        echo "      this run cannot support it. Check that the define reached"
        echo "      the build; log $_log"
        tail -15 "$_log" | sed 's/^/      /'
        return 1
    fi

    _sab=$(sed -n 's/^sabotage \(.*\)$/\1/p' "$_log" | head -1)
    if [ "$_sab" != "$_want_sab" ]; then
        echo "  SABOTAGE WITNESS WRONG -- the binary reports sabotage"
        echo "      [${_sab:-<no line>}], not [$_want_sab]. A define that was"
        echo "      passed and dropped produces a card identical to the clean"
        echo "      one, which reads as 'the ladder cannot see this defect'"
        echo "      when the truth is 'the defect was never built'. log $_log"
        return 1
    fi

    if [ ! -s "$_card" ]; then
        echo "  NO CARD -- the run wrote nothing to $_card (log $_log)"
        tail -15 "$_log" | sed 's/^/      /'
        return 1
    fi

    _recs=$(grep -c '	' "$_card" || true)
    echo "  [$_got] sabotage=$_sab, $_recs records, driver exit $_rc -> $_card"
    return 0
}

# ---------------------------------------------------------------------------
# localize <a.card> <b.card> <label a> <label b>
#
# THE POINT OF THE FILE. Per shape: do the INTEGER stages agree, do the INPUTS
# agree, which LEVEL first moved, and did the levels below it stay put.
#
# The order of the questions is the order of the ladder and it is not
# negotiable. A shape whose `dims` moved has a different partition, so its
# level records are not the same records and comparing them is meaningless. A
# shape whose inputs moved is a diff of the FIXTURE and every float below it
# is void. Only then does a level verdict mean anything.
# ---------------------------------------------------------------------------
localize() {
    awk -v LA="$3" -v LB="$4" '
    function shape_of(t,   s) {
        s = t
        if (sub(/\.dims$/, "", s)) return s
        s = t; if (sub(/\.in\.[ab]$/, "", s)) return s
        s = t; if (sub(/\.fold\.L[0-9]+$/, "", s)) return s
        s = t; if (sub(/\.out$/, "", s)) return s
        return ""
    }
    # `FNR == NR` IS THE FIRST-FILE TEST AND NOT `FILENAME == A`. Comparing
    # filenames looks equivalent and is not: `diff x.card x.card` -- the
    # smoke test anyone runs first, and the one that must report NO
    # divergence -- makes both names equal, so both blocks fire on the first
    # pass, the second map stays empty, and every record is reported as
    # STRUCTURALLY MISSING. That is a false red on the one input whose answer
    # is known in advance, and it is how this was caught.
    FNR == NR && /^[0-9]/ && NF == 5 {
        ah[$2] = $5; ac[$2] = $4; ord[++nord] = $2; next
    }
    FNR == NR { next }
    /^[0-9]/ && NF == 5 { bh[$2] = $5; bc[$2] = $4; next }
    END {
        bad = 0; moved_any = 0
        # ---- record 0: did the two runs even run the same rows? ----
        if (("ladder.shapes" in ah) && ("ladder.shapes" in bh)) {
            if (ah["ladder.shapes"] != bh["ladder.shapes"]) {
                print "  THE TWO RUNS SELECTED DIFFERENT TABLE ROWS."
                print "  ladder.shapes differs at record 0, so nothing below"
                print "  is a comparison of the same shapes. Set"
                print "  MOJOLEARN_GEMM_LADDER_SHAPES the same on both legs."
                exit 1
            }
        }
        for (i = 1; i <= nord; i++) {
            t = ord[i]
            s = shape_of(t)
            if (s == "") continue
            if (!(s in seen)) { seen[s] = 1; shp[++ns] = s }
            if (!(t in bh)) { missing[s] = missing[s] " " t; continue }
            same = (ah[t] == bh[t] && ac[t] == bc[t])
            if (t == s ".dims")   dims[s] = same
            else if (t == s ".in.a" || t == s ".in.b") {
                if (!same) fixture[s] = 1
                inputs[s] = (s in inputs) ? (inputs[s] && same) : same
            }
            else if (t == s ".out") outp[s] = same
            else {
                d = t; sub(/^.*\.fold\.L/, "", d); d = d + 0
                lv[s, d] = same
                if (!(s in nlv) || d + 1 > nlv[s]) nlv[s] = d + 1
                if (!same) { moved_any = 1
                    if (!(s in firstmv)) firstmv[s] = d }
            }
        }
        for (i = 1; i <= ns; i++) {
            s = shp[i]
            printf "\n  %s\n", s
            if (missing[s] != "") {
                printf "     STRUCTURAL: records present in %s and absent in %s:%s\n", LA, LB, missing[s]
                bad = 1
                continue
            }
            printf "     dims (i32)   %s\n", (dims[s] ? "same" : "DIFFER  <-- the PARTITION moved: L, P or the level count. Integer stage, before any float.")
            if (!dims[s]) {
                print  "     Nothing below is a comparison of the same tree."
                bad = 1; continue
            }
            printf "     inputs       %s\n", (inputs[s] ? "same" : "DIFFER  <-- a diff of the FIXTURE, not of the fold.")
            if (fixture[s]) {
                print  "     Every float below is void."
                bad = 1; continue
            }
            line = ""
            for (d = 0; d < nlv[s]; d++)
                line = line sprintf(" L%02d:%s", d, (lv[s, d] ? "=" : "X"))
            printf "     levels      %s\n", line
            printf "     out          %s\n", (outp[s] ? "same" : "DIFFER")
            if (!(s in firstmv) && outp[s]) {
                printf "     verdict      no divergence at any stage\n"
                continue
            }
            bad = 1
            if (!(s in firstmv) && !outp[s]) {
                printf "     verdict      EVERY FOLD LEVEL IS IDENTICAL AND THE OUTPUT MOVED.\n"
                printf "                  The tree agrees node for node; the defect is at the\n"
                printf "                  EMIT SEAM, downstream of the whole fold. Look at\n"
                printf "                  identical_gemm_emit_kernel, not at the fold.\n"
                continue
            }
            f = firstmv[s]
            if (f == 0) {
                printf "     verdict      LEVEL 0 MOVED: the LEAF PARTIALS already differ.\n"
                printf "                  Not a fold defect. This is the leaf loop -- an FMA\n"
                printf "                  contraction, a denormal policy, or the addressing\n"
                printf "                  a partial was written at.\n"
            } else {
                ok = 1
                for (d = 0; d < f; d++) if (!lv[s, d]) ok = 0
                printf "     verdict      LEVEL %02d MOVED AND LEVELS 00..%02d ARE IDENTICAL%s\n", f, f - 1, (ok ? "." : " (?!)")
                printf "                  The leaves agree bit for bit and the FOLD changed the\n"
                printf "                  answer: a reduction-ORDER defect at level %02d. That is\n", f
                printf "                  the finding an input/output card cannot produce.\n"
            }
        }
        printf "\n  %d shape(s) compared; %s\n", ns, (bad ? "DIVERGENCE LOCALIZED ABOVE" : "no divergence")
        exit (bad ? 1 : 0)
    }' "$1" "$2"
}

# ---------------------------------------------------------------------------

ACTION="${1:-emit}"

case "$ACTION" in
    -h|--help|help)
        usage
        exit 0
        ;;

    emit)
        if [ -n "${2:-}" ]; then OUT="$2"; fi
        mkdir -p "$OUT"
        echo "== gemm.fp32.v1 fold ladder: emit, $WANT_MODE, $SHA =="
        echo "   out: $OUT"
        rc=0
        emit_card "$OUT/ladder.card" "$OUT/ladder.log" "none" || rc=1
        if [ "$rc" -ne 0 ]; then
            echo "gemm ladder emit: RED. Log in $OUT"
            exit 1
        fi
        sed -n '/^  /p' "$OUT/ladder.log" | head -60
        echo
        echo "gemm ladder emit: card in $OUT/ladder.card"
        echo "Diff it against another leg with:"
        echo "  tools/gemm_ladder.sh diff $OUT/ladder.card OTHER.card"
        ;;

    diff)
        a="${2:-}"; b="${3:-}"
        if [ -z "$a" ] || [ -z "$b" ]; then usage >&2; exit 2; fi
        if [ ! -s "$a" ]; then echo "gemm_ladder.sh: no card at $a" >&2; exit 2; fi
        if [ ! -s "$b" ]; then echo "gemm_ladder.sh: no card at $b" >&2; exit 2; fi
        echo "== gemm.fp32.v1 fold ladder: diff =="
        echo "   A: $a"
        echo "   B: $b"
        rc=0
        localize "$a" "$b" A B || rc=$?
        echo
        echo "  the differ's own view of the first divergence:"
        /usr/bin/python3 "$DIFFER" "$a" "$b" --labels A,B 2>&1 \
            | sed 's/^/    /' | head -40 || true
        exit "$rc"
        ;;

    sabotage)
        NAME="${2:-}"
        if [ -z "$NAME" ]; then usage >&2; exit 2; fi
        if [ -n "${3:-}" ]; then OUT="$3"; fi
        mkdir -p "$OUT"
        echo "== gemm.fp32.v1 fold ladder: sabotage $NAME, $WANT_MODE, $SHA =="
        echo "   out: $OUT"
        echo
        echo "  clean build:"
        emit_card "$OUT/clean.card" "$OUT/clean.log" "none" \
            || { echo "gemm ladder sabotage: RED (clean leg)"; exit 1; }
        echo "  sabotaged build (-D MOJOLEARN_GEMM_SABOTAGE_$NAME=1):"
        # The sabotaged driver is EXPECTED to fail its own proofs. emit_card
        # reports the exit status and judges only the witnesses, which is why
        # the card survives to be localized below.
        emit_card "$OUT/sab.card" "$OUT/sab.log" "$NAME" \
            -D "MOJOLEARN_GEMM_SABOTAGE_$NAME=1" \
            || { echo "gemm ladder sabotage: RED (sabotage leg)"; exit 1; }
        echo
        rc=0
        localize "$OUT/clean.card" "$OUT/sab.card" CLEAN "SAB_$NAME" || rc=$?
        echo
        # The differ's own view, on the same two cards. `localize` above
        # answers "which level", which is this file's question; the differ
        # answers "which cells, how many ULPs apart, and is either side a
        # denormal", which is the next question and is not worth
        # re-implementing in awk.
        echo "  the differ's own view of the first divergence:"
        /usr/bin/python3 "$DIFFER" "$OUT/clean.card" "$OUT/sab.card" \
            --labels CLEAN,"SAB_$NAME" 2>&1 | sed 's/^/    /' | head -40 || true
        echo
        if [ "$rc" -eq 0 ]; then
            echo "gemm ladder sabotage $NAME: THE CARD DID NOT MOVE."
            echo
            echo "The sabotage WAS built -- the witness above says so -- and"
            echo "every stage of the ladder hashed the same on both legs. So"
            echo "this fixture cannot separate that defect from a correct"
            echo "build, and a green ladder here is not evidence that the"
            echo "defect is absent. That is exactly the failure row 9's"
            echo "correction came from. Record it as a gap in the fixture;"
            echo "do NOT tune the shapes until it goes red."
            echo "Cards in $OUT"
            exit 3
        fi
        echo "gemm ladder sabotage $NAME: the ladder LOCALIZED it (above)."
        echo "Cards in $OUT"
        echo
        echo "Read the verdict line, not the exit code: a level-0 move and a"
        echo "level-1 move are different defects and this file exists to tell"
        echo "them apart."
        ;;

    *)
        echo "gemm_ladder.sh: unknown action '$ACTION'." >&2
        usage >&2
        exit 2
        ;;
esac
