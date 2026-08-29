#!/usr/bin/env bash
# E1 remote bootstrap: run ON the GPU box (MI300X droplet, RunPod pod, or
# the Mac reference side). Executes E1_RUNBOOK.md Phases 0-3 and leaves
# every artifact under bench/results/e1/<stamp>-<host>/.
#
# usage:  bash tools/e1_bootstrap.sh
# The repo checkout this script lives in IS the source; rsync it to the
# box first (rsync carries the exact commit; no GitHub access needed):
#   rsync -a --exclude .pixi --exclude bench/external/.gbm-bench \
#       ~/CascadeProjects/mojolearn/ <box>:~/mojolearn/
#
# THE MODE: IDENTICAL, by BUILD DEFINE (-D MOJOLEARN_NUMERIC_IDENTICAL=1 via
# tools/with_identical_mode.sh; bindings into python/mojolearn/identical/;
# drivers select it with MOJOLEARN_NUMERIC_MODE=identical). No file in the
# tree is edited, nothing is reverted, the shipped FAST binaries are never
# touched.
#
# THIS PARAGRAPH USED TO END "on AMD a FAST build is a compile error at the
# 64-wide wavefront asserts, by design, so IDENTICAL is the only mode that
# runs there." LEG 13 (2026-08-25, MI325X) FALSIFIED IT. Every FAST lane in
# phase 8 BUILT and RAN on the AMD box: gemm's FAST device check reported 128
# OK lines, and cd, kde, linkage, svm and metrics all produced FAST cards that
# diverge from Apple's. The sentence is deleted rather than softened, because
# it is the sentence that would have made a missing AMD FAST column look
# intentional.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
STAMP="$(date +%Y-%m-%d_%H%M%S)-$(hostname -s)"
OUT="$REPO/bench/results/e1/$STAMP"
mkdir -p "$OUT"
LOG="$OUT/bootstrap.log"
exec > >(tee "$LOG") 2>&1

step() { echo; echo "=== $* === $(date +%T)"; }

step "provenance"
# THE COMMIT MUST NOT BE ABLE TO END A RUN. DEVIATION 1935, 2026-08-28.
#
# This was `git rev-parse HEAD | tee "$OUT/commit.txt"`, and on a box that
# received the tree as a `git archive` there is no `.git` at all -- which is
# exactly what tools/gemm_remote_leg.sh ships, because the repo is private
# and the box has no credentials. The bootstrap survived that (the pipe just
# wrote an empty file) but tools/e2_matrix_fit.py, tools/e1_traced_fit.py and
# tools/e2u_matrix_fit.py did not: each opened with a bare `git rev-parse
# HEAD` for a provenance STRING and died on CalledProcessError 128.
#
# Measured cost, the NVIDIA column of the 2026-08-28 round at cc499f7: ten of
# ten identical bindings built, and then phase 3's traced fits, phase 4's E2
# tree matrix and phase 7's E2U matrix were all lost to that line. The round
# had an Apple column, an AMD column, and an NVIDIA column with no matrix in
# it. The DigitalOcean leg never showed this because it ships a `git bundle`
# and clones it, so its boxes do have a `.git`.
#
# MOJOLEARN_COMMIT is exported so every child driver inherits the answer
# rather than each one re-deriving it from a repository that is not there.
MOJOLEARN_COMMIT="$(git rev-parse HEAD 2>/dev/null || echo "${MOJOLEARN_COMMIT:-unknown}")"
export MOJOLEARN_COMMIT
printf '%s\n' "$MOJOLEARN_COMMIT" | tee "$OUT/commit.txt"
uname -a | tee "$OUT/uname.txt"
bash bench/external/record_environment.sh > "$OUT/environment.txt" 2>&1 || true

step "pixi"
if ! command -v pixi >/dev/null; then
  curl -fsSL https://pixi.sh/install.sh | bash
  export PATH="$HOME/.pixi/bin:$PATH"
fi
pixi install

step "mode: IDENTICAL by build define (no source flip since 2026-08-23)"
# Every gate below runs through tools/with_identical_mode.sh, which injects
# -D MOJOLEARN_NUMERIC_IDENTICAL=1 into the mojo command (or exports
# MOJOLEARN_MOJO_DEFINES for scripts that call mojo themselves); the
# bindings build into python/mojolearn/identical/ and the Python drivers
# select that set with MOJOLEARN_NUMERIC_MODE=identical. Nothing in the
# tree is edited, so there is nothing to revert and no shipped binary to
# restore afterwards. (The claim that FAST cannot build on AMD stood here too
# and was falsified by leg 13; see the header.)
IDENT="$REPO/tools/with_identical_mode.sh"
export MOJOLEARN_NUMERIC_MODE=identical

# ---------------------------------------------------------------------------
# MOJOLEARN_E1_PHASES: WHICH PHASES THIS RUN EXECUTES (DEVIATION 972)
# ---------------------------------------------------------------------------
# Unset means ALL of them. That is what every existing caller gets and it is
# what a real round must use. A space or comma separated list runs only those.
#
# WHY THIS EXISTS. Leg 12 (2026-08-24) rented an RTX 4090, spent fifty minutes
# in phases 0 through 7, hit the payload's 3000s bound at bootstrap_exit=124
# and came home with identical_cards=0. tools/gemm_remote_leg.sh caps a lease
# at 60 minutes BY NAME and refuses more, so on that box the full bootstrap
# does not fit in one lease at all. That is structural rather than a tuning
# problem: phases 0-4 alone took fifteen minutes and 5-7 took the rest.
#
# Phase 8 is self-contained. Its lane arms are `pixi run mojo run -I . <driver>`
# which compile from source, so it needs nothing phase 3 builds.
#
#   MOJOLEARN_E1_PHASES=8 bash tools/e1_bootstrap.sh
#
# READ THIS BEFORE USING IT ON A ROUND. A PHASE-SUBSET COLUMN IS NOT A ROUND.
# tools/e3_round_judge.sh sections 1 to 6 read the tree matrix, E2U, the E1U
# cards, the gate lines and cross-infer; a column missing those FAILS them,
# correctly, because the evidence is not there. Only section 7, the lane
# cards, is answerable from a phase-8-only column. Use this to answer ONE lane
# question cheaply. Never use it to record a round.
run_phase() {
  [ -z "${MOJOLEARN_E1_PHASES:-}" ] && return 0
  case " $(printf '%s' "$MOJOLEARN_E1_PHASES" | tr ',' ' ') " in
    *" $1 "*) return 0 ;;
    *) echo; echo "=== phase $1 SKIPPED (MOJOLEARN_E1_PHASES=$MOJOLEARN_E1_PHASES) ==="; return 1 ;;
  esac
}

# THE TEN BINDINGS, hoisted above the phases 2026-08-29. It lived inside
# phase 3's block, so `MOJOLEARN_E1_PHASES=9` -- phase 9 builds the
# DETERMINISTIC set from the same list -- died on an unbound variable
# before it launched a single kernel. A phase subset is a supported way to
# run this file, so no phase may depend on another having run.
E1_IDENT_BINDINGS="build.sh build_estimators.sh build_gbdt.sh build_rf.sh build_trees.sh build_svm.sh build_solver.sh build_metrics.sh build_tsa.sh build_linalg.sh"

if run_phase 0; then
step "phase 0: smoke (hardware matrix, column detection)"
"$IDENT" pixi run check-hardware-matrix || echo "PHASE0-FINDING: hardware matrix (see log)"

fi

if run_phase 1; then
step "phase 1: vendor characterization (row 10 precondition)"
"$IDENT" pixi run check-ieee-arith || echo "PHASE1-FINDING: ieee-arith (see log)"
# row 12's certificate line: the printed device hash must be the SAME
# NUMBER on every vendor column (Apple measured 8705486125800438413)
"$IDENT" pixi run check-portable-translog || echo "PHASE1-FINDING: portable-translog (see log)"
"$IDENT" pixi run check-portable-sqrtcos || echo "PHASE1-FINDING: portable-sqrtcos (see log)"

fi

if run_phase 2; then
step "phase 2: gates under IDENTICAL"
for gate in check-depthwise check-lossguide-policy check-random-strength; do
  echo "--- $gate"
  "$IDENT" pixi run "$gate" || echo "PHASE2-FINDING: $gate FAILED"
done
echo "--- extratrees suite"
"$IDENT" bash extratrees/tools/check.sh || echo "PHASE2-FINDING: extratrees suite (see log)"

fi

if run_phase 3; then
step "phase 3: build IDENTICAL .so + traced fits"
# ALL TEN bindings, not five. DEVIATION 1930, 2026-08-28.
#
# This loop built five, and `python/mojolearn/_backend.py`'s `_MODULES`
# lists ten. The other five therefore came out of `select()` as
# `_MissingIdentical` stubs -- which is the DESIGNED behaviour for a binding
# that did not build (import succeeds, touching it raises by name), and it
# worked until `_solver_impl.py` added an IMPORT-TIME guard whose basename
# check a stub cannot pass. From then on the traced driver and the whole E2
# matrix died at `from . import _mojolearn_solver` with a message blaming
# `_MODULES` for not listing a module it does list. Measured today: the M4
# and the MI325X both reported it, and both reported it as PHASE3-FINDING
# plus PHASE4-FINDING rather than as a build failure, so it read as two
# unrelated driver problems.
#
# Building all ten costs minutes on a rented box and the fifty-minute budget
# has to absorb them; a stub in the identical set costs the matrix.
#
# the python package's __init__ imports cluster ->
# _mojolearn.so and friends, so an rsync'd foreign-platform .so anywhere
# in the package breaks every import ("invalid ELF header", run 2's
# finding). Remove the foreign binaries LOUDLY first, and skip each
# build's own smoke gate -- during a from-scratch five-binding build
# every gate imports siblings that do not exist yet (run 3's finding).
# The traced driver below is the real gate: it launches kernels through
# every lib. Run inside the gbmbench env so python has numpy.
# the identical set lands in python/mojolearn/identical/ (MOJOLEARN_NUMERIC_MODE
# is exported above and the build scripts read it); a foreign-platform .so
# left there from an rsync would break every import, so clear it first
rm -f python/mojolearn/identical/_mojolearn*.so
for b in $E1_IDENT_BINDINGS; do
  echo "--- bindings/$b (identical)"
  pixi run -e gbmbench bash "bindings/$b" \
    || echo "PHASE3-FINDING: bindings/$b failed (see log)"
done
# WHAT THE IDENTICAL SET ACTUALLY CONTAINS, COUNTED, because "ten builds ran"
# and "ten binaries exist" are different sentences and only the second one is
# what `select()` reads.
echo "identical set: $(ls python/mojolearn/identical/_mojolearn*.so 2>/dev/null | wc -l | tr -d ' ') of 10 binaries"
ls python/mojolearn/identical/_mojolearn*.so 2>/dev/null | sed 's|.*/|  |'
# the FAST set must exist too: the package imports every binding and the
# selector only swaps the five it loads; on a fresh box build FAST as well
# (it cannot build on AMD -- the wavefront asserts -- so a missing FAST set
# is tolerated: the selector never touches it under identical)
if [ ! -f python/mojolearn/_mojolearn.so ]; then
  for b in build.sh build_estimators.sh build_gbdt.sh build_rf.sh build_trees.sh; do
    # keep the output: "did not build" without the first error line cost a
    # leg (2026-08-23). Known on AMD: build_gbdt.sh -- the FAST histogram
    # accumulators are CatBoost's 32-lane slice layout and refuse a 64-wide
    # wavefront at compile time (hist_one_byte.mojo:202, hist_2_one_byte_base
    # .mojo:240, point_hist_half_byte_template.mojo:163); build_estimators.sh
    # built once K_LIB_JACOBI_EIGH went numeric (b943103).
    if ! MOJOLEARN_NUMERIC_MODE=fast MOJOLEARN_SKIP_BUILD_GATE=1 pixi run -e gbmbench bash "bindings/$b" > "$OUT/fast_build_${b%.sh}.log" 2>&1; then
      echo "note: FAST bindings/$b did not build here; first error:"
      grep -m2 -E 'error:|constraint failed' "$OUT/fast_build_${b%.sh}.log" | cut -c1-200 | sed 's/^/      /'
    fi
  done
fi
PYTHONPATH="$REPO/python" pixi run -e gbmbench python3 tools/e1_traced_fit.py "$OUT" \
  || PYTHONPATH="$REPO/python" python3 tools/e1_traced_fit.py "$OUT" \
  || echo "PHASE3-FINDING: traced driver failed (see log)"

fi

if run_phase 4; then
step "phase 4: E2 sub-feature matrix (every loss/bootstrap/score/estimator/searcher/bins/cat/NaN/criterion)"
# one subprocess per cell, so a device fault in one configuration leaves
# the other cards intact; e2_cells.json is rewritten after every cell.
# `tools/e2_matrix_diff.py <mac>:APPLE <gpu>:VENDOR` is the verdict table.
PYTHONPATH="$REPO/python" pixi run -e gbmbench python3 tools/e2_matrix_fit.py "$OUT" \
  || PYTHONPATH="$REPO/python" python3 tools/e2_matrix_fit.py "$OUT" \
  || echo "PHASE4-FINDING: E2 matrix driver failed (see log)"
# the Python-unreachable training paths (depthwise/lossguide growth,
# MultiClassOneVsAll, ...) get their cards from Mojo probes, one fit per
# file, when that script is present
if [ -x tools/e2_mojo_cards.sh ]; then
  bash tools/e2_mojo_cards.sh "$OUT" || echo "PHASE4-FINDING: e2_mojo_cards failed (see log)"
fi

fi

if run_phase 5; then
step "phase 5: the unsupervised cards (k-means, k-NN, DBSCAN) -- IDENTICAL"
# tools/e1_unsupervised.sh is the unsupervised lane's leg (rows 19-26); it
# re-enters itself through the injector and writes bench/results/e1u/<stamp>;
# a copy of that directory lands beside this run's artifacts
sh tools/e1_unsupervised.sh "$OUT/e1u" || echo "PHASE5-FINDING: unsupervised leg (see log)"

fi

if run_phase 6; then
step "phase 6: the linear-algebra identity gates (GEMM, column stats, Jacobi/PCA, OLS) -- both modes"
# rows 27-32; verdicts, not numbers (both modes, ~22 device runs). The
# decomposition bindings did not BUILD on AMD before 4ecf43c; the gate
# running at all here is part of the result.
pixi run check-linalg-identity || echo "PHASE6-FINDING: linalg identity (see log)"
pixi run check-unsupervised-identity || echo "PHASE6-FINDING: unsupervised identity (see log)"

fi

if run_phase 7; then
step "phase 7: E2U -- the unsupervised sub-feature matrix through the Python surface, IDENTICAL"
# tools/e2u_matrix_fit.py (67 cells: KMeans/NearestNeighbors/DBSCAN/PCA/tSVD/OLS);
# judged by tools/e2_matrix_diff.py against the Mac's e2u directory
PYTHONPATH="$REPO/python" pixi run -e gbmbench python3 tools/e2u_matrix_fit.py "$OUT/e2u" \
  || PYTHONPATH="$REPO/python" python3 tools/e2u_matrix_fit.py "$OUT/e2u" \
  || echo "PHASE7-FINDING: E2U matrix driver failed (see log)"

fi

if run_phase 8; then
step "phase 8: the classical lanes (gemm, cd, kde, linkage, svm, metrics) -- both modes, cards + checks"
# Andrew's order 2026-08-23 (via the orchestrator): everything that exists
# today must be bit-identical across all three GPUs. One card per lane per
# mode (the drivers read MOJOLEARN_IDENTITY_TRACE from the environment and
# print their compiled mode on the banner), plus each lane's gate in both
# modes. Judged by tools/e3_round_judge.sh section 7: the IDENTICAL cards
# Apple vs NVIDIA vs AMD through identity_trace_diff; FAST cards recorded.
# A lane that fails here is a FINDING, never an abort: the others still run.
mkdir -p "$OUT/lanes"
# MOJOLEARN_E1_LANES: WHICH PHASE-8 LANES RUN, AND IN THIS ORDER (DEVIATION 973)
#
# Unset means all of them, so every existing caller is unchanged. Gated inside
# run_lane_arm and run_lane_check rather than around each of the fourteen call
# sites, because one gate cannot be forgotten and fourteen can.
#
# WHY. Leg 12's phase-8-only run reached the work bound with identical_cards=2:
# gemm and cd landed, the other five did not. Phase 8's order is
# `gemm cd kde linkage svm metrics mamba` and MAMBA IS LAST, while gemm's
# device check is the largest compile in the set. On the M4 all seven take 74
# seconds against a warm mojo cache; on a cold rented box the first two ate a
# fifty-minute budget. A lane that is last is a lane that does not run.
#
#   MOJOLEARN_E1_LANES=mamba bash tools/e1_bootstrap.sh
#
# This selects, it does not reorder: the calls below still run in their written
# order and the filter only skips. To put a lane FIRST, ask for that lane ALONE.
# A skipped lane is announced, never silent, so a column cannot be mistaken for
# a fuller one -- and the judge already treats a missing IDENTICAL card as a
# hard failure, which is the behavior that keeps this honest.
lane_enabled() {
  [ -z "${MOJOLEARN_E1_LANES:-}" ] && return 0
  case " $(printf '%s' "$MOJOLEARN_E1_LANES" | tr ',' ' ') " in
    *" $1 "*) return 0 ;;
    *) return 1 ;;
  esac
}
run_lane_arm() {  # <lane> <mode fast|identical> <cmd...>
  local lane="$1" mode="$2"; shift 2
  lane_enabled "$lane" || { echo "  $lane [$mode]: SKIPPED (MOJOLEARN_E1_LANES=$MOJOLEARN_E1_LANES)"; return 0; }
  local card="$OUT/lanes/$lane.$mode.card" log="$OUT/lanes/$lane.$mode.log"
  rm -f "$card"
  if [ "$mode" = identical ]; then
    MOJOLEARN_IDENTITY_TRACE="$card" "$IDENT" "$@" > "$log" 2>&1 \
      || echo "PHASE8-FINDING: $lane [$mode] failed (see $log)"
  else
    MOJOLEARN_IDENTITY_TRACE="$card" MOJOLEARN_MOJO_DEFINES= MOJOLEARN_NUMERIC_MODE=fast "$@" > "$log" 2>&1 \
      || echo "PHASE8-FINDING: $lane [$mode] failed (see $log)"
  fi
  if [ -f "$card" ]; then
    # read the mode BACK from the card's own header or the driver's banner,
    # never from the flag that was passed (the shared-checkout rule)
    echo "  $lane [$mode]: card $(grep -vc '^#\|^$' "$card") records -- $( (grep -m1 -oE '\[(FAST|IDENTICAL)\]|mode (FAST|IDENTICAL)' "$card" "$log" 2>/dev/null | head -1 | sed 's/^[^:]*://') )"
  else
    echo "  $lane [$mode]: NO CARD written ($(grep -m1 -E 'rror|FAIL' "$log" | cut -c1-120))"
  fi
}
# WHAT ENDED THE RUN, NOT THE FIRST LINE THAT LOOKS ALARMING. DEVIATION 1933.
#
# This used to be `grep -m1 -E 'Unhandled|rror:|FAIL'`, and the AMD leg of
# 2026-08-28 shows what that costs. Its gemm lane came home as
#
#   PHASE8-FINDING: gemm [fast] check FAILED:  FAIL pca.transform.wide...
#       16 of 16 cells differ from gemm_oracle
#
# and that line is NOT a failure at all. `gemm_device_check` prints it from
# `check_device_matches_oracle`, which under FAST is a REPORT by design --
# the file says so at the seam: "under FAST both sides are the unpinned
# spelling ... this is a measurement and not an assertion." The run actually
# ended on a DIFFERENT gate, `check_device_is_batch_invariant`, which IS
# asserted in both modes because batch invariance is a property of the
# kernel's shape rather than of the arithmetic pins.
#
# So the finding named an expected measurement as the cause and hid a real
# defect behind it. Read the exception first, then a hard error, and only
# fall back to a FAIL line when neither exists.
_lane_cause() {
  local _log="$1"
  grep -m1 -E 'Unhandled exception' "$_log" 2>/dev/null | cut -c1-200 && return 0
  grep -m1 -E 'error:' "$_log" 2>/dev/null | cut -c1-200 && return 0
  grep -m1 -E 'FAIL' "$_log" 2>/dev/null | cut -c1-200
}

run_lane_check() {  # <lane> <mode> <cmd...>
  local lane="$1" mode="$2"; shift 2
  lane_enabled "$lane" || return 0
  local log="$OUT/lanes/$lane.$mode.check.log"
  if [ "$mode" = identical ]; then
    "$IDENT" "$@" > "$log" 2>&1 && echo "  $lane [$mode] check OK ($(grep -c ' OK' "$log") OK lines)" \
      || echo "PHASE8-FINDING: $lane [$mode] check FAILED: $(_lane_cause "$log")"
  else
    MOJOLEARN_MOJO_DEFINES= MOJOLEARN_NUMERIC_MODE=fast "$@" > "$log" 2>&1 && echo "  $lane [$mode] check OK ($(grep -c ' OK' "$log") OK lines)" \
      || echo "PHASE8-FINDING: $lane [$mode] check FAILED: $(_lane_cause "$log")"
  fi
}
# THE IDENTICAL PASS RUNS FIRST, AND THE ORDER IS LOAD-BEARING ON A RENTAL.
# DEVIATION 868, 2026-08-24. This loop read `for mode in fast identical` until
# now, which meant the ENTIRE FAST pass ran before the first IDENTICAL card was
# written. On a rented box under a work bound, a slow FAST pass therefore spent
# the whole lease and the leg came home with an empty `lanes/` -- zero identical
# cards, which is the only arm `tools/e3_round_judge.sh` section 7 ASSERTS. The
# FAST cards are recorded, never judged, so they are the half that can be lost.
#
# The two passes are independent: `run_lane_arm` rebuilds per mode and each
# writes its own card, so swapping the order cannot move a bit in either one.
# The leg-11 artifacts are unaffected and stay comparable.
for mode in identical fast; do
  echo "-- lanes, $mode --"
  MOJOLEARN_GEMM_CARD_ARM=device MOJOLEARN_GEMM_CARD_HOST_CAP=1 run_lane_arm gemm "$mode" pixi run mojo run -I . bench/gemm_card_main.mojo
  run_lane_check gemm "$mode" pixi run mojo run -I . gemm/mojo_only/gemm_device_check.mojo
  run_lane_arm cd "$mode" pixi run mojo run -I . solver/cd_main.mojo
  run_lane_check cd "$mode" pixi run check-cd
  run_lane_arm kde "$mode" pixi run mojo run -I . kde/kde_main.mojo
  run_lane_check kde "$mode" pixi run check-kde
  run_lane_arm linkage "$mode" pixi run mojo run -I . hierarchy/linkage_main.mojo
  run_lane_check linkage "$mode" pixi run check-linkage
  run_lane_arm svm "$mode" pixi run mojo run -I . svm/svc_main.mojo -- card
  run_lane_check svm "$mode" pixi run check-svm
  # mamba, added 2026-08-23 once contract section 8 clauses (a)-(f) were all
  # gated on Apple. The gate is its own card driver, so the arm and the check
  # are the same binary:
  #
  # THE PARENTHESIS THAT STOOD HERE SAID "(it honors MOJOLEARN_IDENTITY_TRACE)"
  # AND IT WAS FALSE UNTIL 2026-08-24. mamba_check.mojo read a hardcoded
  # TRACE_PATH, so the card landed in /tmp and this loop reported "NO CARD
  # written" for a lane whose gate had just passed. Fixed in that file under
  # DEVIATION 970; the sentence is corrected here rather than left, because it
  # is the sentence that made the defect invisible.
  # the arm captures the card, the check asserts. FAST MODE HAS NEVER BEEN
  # BUILT FOR THIS LANE, so a PHASE8-FINDING on the fast arm is EXPECTED here
  # and is information, not an alarm. The IDENTICAL arm is the one the leg is
  # for. Shape defaults to the tiny one; do not widen it on a rented box.
  run_lane_arm mamba "$mode" pixi run mojo run -I . mamba/mojo_only/mamba_check.mojo
  run_lane_check mamba "$mode" pixi run check-mamba-block
  # iforest and transformer, added 2026-08-28 (DEVIATION 1932). BOTH LANES
  # ALREADY HAD EVERYTHING EXCEPT A LINE HERE, and that is the finding:
  #
  #   * isolation_forest/mojo_only/if_check.mojo BUILT a 123-stage card and
  #     wrote it to a scratch path nobody collects, so iforest had ZERO cells
  #     in every cross-vendor round while its own gate went green. It now
  #     honours MOJOLEARN_IDENTITY_TRACE like mamba does (mamba needed the
  #     same repair under DEVIATION 970).
  #   * transformer/mojo_only/transformer_check.mojo has honoured
  #     MOJOLEARN_IDENTITY_TRACE since DEVIATION 1101 and its own docstring
  #     says "tools/e1_bootstrap.sh phase 8 sets it" -- a lane built to be
  #     here that was never listed. Its SCOPE line said "nothing cross-vendor
  #     until a leg runs" and no leg could run, because the loop did not
  #     name it.
  #
  # CLAUSE (d) IS TURNED ON FOR THE TRANSFORMER ARM, AND IT IS THE ONE THAT
  # CHANGES THE LEDGER. It is opt-in in that file and defaults off, and the
  # gate says why that matters: two of its thirteen sabotage arms
  # (S07_ROPE_RELATIVE_POSITION and S19_VALUE_SUM_VIA_GEMM) are falsifiable
  # ONLY under clause (d), so a run without it has ELEVEN arms and not
  # thirteen. Measured on the M4 2026-08-28: clause (d) PASSES, 4 decode
  # steps bit-identical to the prefill on all 11,632 compared cells, with its
  # own control showing 57 misaligned stage comparisons that DO differ.
  #
  # CLAUSE (e) IS LEFT OFF AND THIS IS THE REASON, RECORDED RATHER THAN
  # QUIETLY OMITTED. It is the section 8 planted audit: a NaN or infinity in
  # any of thirteen named inputs must be REFUSED BY NAME before any stage is
  # recorded. Run today it aborts the whole driver on its FIRST plant --
  # "llama: NaN in input_layernorm.weight at flat index 16 REFUSED" escapes
  # as an unhandled exception. The refusal itself is CORRECT and arguably
  # early in the good direction: `LlamaDeviceWeights.__init__` refuses at
  # UPLOAD, and the clause's `try` wraps only the later
  # `llama_decoder_layer_forward_planted` call, so the expected raise happens
  # outside the block that was written to catch it. That is a defect in the
  # clause, not in the block, and turning it on would abort every lane after
  # transformer in this loop. It is not on the judge's allowlist and it must
  # not be added to one: fix the try, then turn it on.
  #
  # CLAUSE (d) RUNS ON THE IDENTICAL ARM ONLY, AND THE REASON IS A
  # MEASUREMENT. Run under FAST on the M4 on 2026-08-28 it FAILS: 91
  # stage-tokens differ, first at token 0 `q_proj.out` on 26 of 32 cells.
  # That is not a defect and it is not a gate problem -- contract section 7.2
  # makes decode == prefill true BY CONSTRUCTION for the IDENTICAL profile
  # (one spelling serves both paths, RoPE reads the absolute position, S11
  # contracts over a head_dim of equal length in both, S17 and S19 are serial
  # ascending chains seeded +0.0 over a tail that is exactly +0.0). FAST
  # promises none of that. Asking the FAST arm a question only the identical
  # profile answers would put a red line in every round for a property FAST
  # was never claimed to have, and the allowlist is not the place to hide it:
  # the fix is to not ask.
  #
  # KEEP THE NUMBER THOUGH. "decode == prefill holds under IDENTICAL and
  # breaks at q_proj.out under FAST, same box, same commit, same fixture" is
  # one of the cleanest statements of what the flag buys anywhere in this
  # tree, and it belongs in the paper beside the GEMM oracle counts.
  _tfx_clause_d=""
  [ "$mode" = identical ] && _tfx_clause_d=1
  run_lane_arm iforest "$mode" pixi run mojo run -I . isolation_forest/mojo_only/if_check.mojo
  MOJOLEARN_TRANSFORMER_CHECK_CLAUSE_D="$_tfx_clause_d" \
    run_lane_arm transformer "$mode" pixi run mojo run -I . transformer/mojo_only/transformer_check.mojo
  # THE THIRTEEN LANES THAT WERE BUILT FOR THIS AND NEVER LISTED.
  # DEVIATION 1937, 2026-08-28. Andrew asked the obvious question -- is
  # EVERYTHING identical -- and the honest answer was that sixteen of the
  # tree's twenty-nine algorithm directories had ever been in a round. The
  # other thirteen were not unbuilt: every one carries `ftz`,
  # `identical_mul_add`, a `GLOBAL_NUMERIC_MODE` gate and a check driver, and
  # most already emitted a full identity card. `training/` alone has 242
  # `ftz` calls and three card-emitting drivers.
  #
  # Five (arima, embedding, hdbscan, ivf, training) already honoured
  # MOJOLEARN_IDENTITY_TRACE and needed nothing but this line. Seven wrote
  # their card to a fixed scratch path and got the same `card_path()` repair
  # isolation forest got this morning and mamba got under DEVIATION 970.
  #
  # tsa POINTS AT tsa_main.mojo, NOT AT stationarity_check.mojo, and the
  # sentence that stood here was wrong. It read "tsa IS DELIBERATELY ABSENT:
  # stationarity_check.mojo builds no identity card at all ... it needs a
  # card first; that is a lane's worth of work and not a line in a loop."
  # True of the CHECK driver, false about the LANE. `tsa/tsa_main.mojo` has
  # built a complete eleven-stage card since it was written and honours
  # MOJOLEARN_IDENTITY_TRACE through the bare `IdentityTrace()` constructor,
  # exactly as metrics, linkage, cd, spectral and arima do here. It was one
  # line away the whole time, and the comment claiming otherwise is what kept
  # it out of every round.
  #
  # ONE DRIVER PER LANE NAME, and for tsa that matters more than usual: the
  # env constructor does NOT truncate, so listing both tsa_main.mojo and a
  # carded stationarity_check.mojo under the name `tsa` would have the second
  # run APPEND into the first's card and produce duplicate tags.
  run_lane_arm arima "$mode" pixi run mojo run -I . arima/mojo_only/arima_check.mojo
  run_lane_arm cholesky "$mode" pixi run mojo run -I . cholesky/mojo_only/cholesky_check.mojo
  run_lane_arm embedding "$mode" pixi run mojo run -I . embedding/mojo_only/embedding_check.mojo
  run_lane_arm gp "$mode" pixi run mojo run -I . gaussian_process/mojo_only/gp_check.mojo
  run_lane_arm hdbscan "$mode" pixi run mojo run -I . hdbscan/mojo_only/hdbscan_check.mojo
  run_lane_arm holtwinters "$mode" pixi run mojo run -I . holtwinters/mojo_only/hw_check.mojo
  run_lane_arm ivf "$mode" pixi run mojo run -I . ivf/mojo_only/ivf_check.mojo
  run_lane_arm kernelmethods "$mode" pixi run mojo run -I . kernel_methods/mojo_only/km_check.mojo
  run_lane_arm gmm "$mode" pixi run mojo run -I . mixture/mojo_only/gmm_check.mojo
  run_lane_arm resample "$mode" pixi run mojo run -I . resample/mojo_only/resample_check.mojo
  run_lane_arm spectral "$mode" pixi run mojo run -I . spectral/mojo_only/spectral_check.mojo
  run_lane_arm training-loss "$mode" pixi run mojo run -I . training/mojo_only/loss_check.mojo
  run_lane_arm training-optimizer "$mode" pixi run mojo run -I . training/mojo_only/optimizer_check.mojo
  run_lane_arm training-step "$mode" pixi run mojo run -I . training/mojo_only/train_step_check.mojo
  run_lane_arm tsa "$mode" pixi run mojo run -I . tsa/tsa_main.mojo
  run_lane_arm metrics "$mode" pixi run mojo run -I . metrics/metrics_main.mojo
  for t in check-metrics-labels check-metrics-regression check-metrics-silhouette check-metrics-trust; do
    run_lane_check "metrics-${t#check-metrics-}" "$mode" pixi run "$t"
  done
done

fi

if run_phase 9; then
step "phase 9: RUN-TO-RUN DETERMINISM -- the middle tier, on this vendor's silicon"
# THE QUESTION NO OTHER PHASE ASKS. Phases 3-8 all ask a CROSS-VENDOR
# question: they write cards here so a second machine's cards can be diffed
# against them. That inference only exists where a card exists, it needs two
# machines, and it says nothing at all about FAST -- which is exactly the arm
# whose instability we want to price.
#
# This phase asks the one-box question directly: same fit, same input, same
# GPU, twice -- same bits? It is the whole evidence for `deterministic`, the
# middle tier, and until 2026-08-29 it had been asked on ONE column (an Apple
# M4) and written down as if it covered three. That is the standing error this
# phase exists to stop repeating.
#
# THE DETERMINISTIC SET IS BUILT HERE and nowhere else in this file: phase 3
# builds FAST and IDENTICAL, because those are what the cards need. The middle
# tier needs its own ten binaries and they land under
# python/mojolearn/deterministic/.
rm -f python/mojolearn/deterministic/_mojolearn*.so
for b in $E1_IDENT_BINDINGS; do
  if ! MOJOLEARN_NUMERIC_MODE=deterministic MOJOLEARN_SKIP_BUILD_GATE=1 \
       pixi run -e gbmbench bash "bindings/$b" > "$OUT/det_build_${b%.sh}.log" 2>&1; then
    echo "PHASE9-FINDING: deterministic bindings/$b did not build; first error:"
    grep -m2 -E 'error:|constraint failed' "$OUT/det_build_${b%.sh}.log" | cut -c1-200 | sed 's/^/      /'
  fi
done
echo "deterministic set: $(ls python/mojolearn/deterministic/_mojolearn*.so 2>/dev/null | wc -l | tr -d ' ') of 10 binaries"

mkdir -p "$OUT/stability"
# EVERY TIER, INCLUDING FAST. A tier that is not measured cannot be compared,
# and FAST is the baseline the other two are worth something against. A tier
# whose binaries did not build reports its import error into its own file
# rather than being skipped: "not measured" and "measured clean" must not look
# the same in the fetched directory.
for m in fast deterministic identical; do
  MOJOLEARN_NUMERIC_MODE=$m PYTHONPATH="$REPO/python" \
    pixi run -e gbmbench python3 tools/repeat_run_stability.py \
      --repeats 6 --json "$OUT/stability/$m.json" \
      > "$OUT/stability/$m.txt" 2>&1 \
    || echo "PHASE9-FINDING: stability arm $m returned non-zero (see stability/$m.txt)"
  echo "--- stability $m"
  tail -4 "$OUT/stability/$m.txt" 2>/dev/null | sed 's/^/    /'
done

# THE PROBE THAT ACTUALLY CATCHES IT. Sequential repeats above UNDER-REPORT:
# one process with one device context leaves the queue empty between launches
# and an arrival-order defect reproduces its own last answer. What moves it is
# contention BETWEEN CONTEXTS -- all three tiers live at once, called
# round-robin, which is what a serving process holding two models looks like.
# On the M4 the sequential arm called the k-NN lane STABLE under FAST and this
# probe found three different answers in 24 calls.
PYTHONPATH="$REPO/python" pixi run -e gbmbench python3 \
    tools/repeat_run_stability.py --concurrent \
    > "$OUT/stability/concurrent.txt" 2>&1 \
  || echo "PHASE9-FINDING: concurrent probe returned non-zero (see stability/concurrent.txt)"
echo "--- stability concurrent"
tail -6 "$OUT/stability/concurrent.txt" 2>/dev/null | sed 's/^/    /'

fi
step "done"
echo "artifacts in $OUT"
echo "next: fetch this directory beside the other machine's and run"
echo "  python3 tools/e2_matrix_diff.py <mac>:APPLE <gpu>:VENDOR --write"
echo "  python3 tools/identity_trace_diff.py <mac>/<cell>.card <gpu>/<cell>.card"
