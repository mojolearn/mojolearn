#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# afn_ab.sh <tag> <binding> <board-lane> <shape> <reps> "<defines A>" "<defines B>"
#
# The Apple FAST NEURAL A/B (docs/apple-fast/PLAN-neural.md): two builds of ONE
# neural binding, arm A with MOJOLEARN_MOJO_BUILD_FLAGS="<defines A>" and arm B
# with "<defines B>" ("" = no define), each installed into the package in turn
# and raced ALONE through tools/bench_board_neural.py (`--arms ours-fast`,
# MOJOLEARN_NUMERIC_MODE=fast; no torch opponent is ever re-run), alternated
# A B A B ... <reps> times, AFN_ROUNDS timed rounds per race. Then the judge:
# the two arms' kept outputs compared (tools/afn_ab.py compare-outputs: losses
# on a train lane, the output on a forward lane), and on the lanes
# tools/neural_fast_quality.py covers (samba, mlp, the blocks) that tool run
# for arm A and then arm B at ONE seed from the same init and batches, judged
# by its `pair` rule. Arm B's .so is left installed.
#
#   binding     linalg | transformer | mamba | training | byte_lm | embedding | x_cnn
#               (bindings/build_<binding>.sh)
#   board-lane  a GPU lane of tools/bench_board_neural.py LANES: lm-train-step,
#               lm-forward, gemm, gemm-bf16, gemm-int8, transformer-forward,
#               mamba1-forward, mamba2-forward, mamba3-forward, samba-train-step,
#               samba-forward, mlp-train-step (the *-infer and lm-host lanes run
#               our CPU and are refused by name)
#   shape       full (the board) | small (a smoke)
#   reps        alternations (A B) >= 1
#   defines     arm A's and arm B's MOJOLEARN_MOJO_BUILD_FLAGS, e.g.
#               "" and "-D MOJOLEARN_AFN_GEMM_SIMDGROUP"
#
# BASELINE MODE (the tier itself): the arm words `identical` and `fast`:
#     afn_ab.sh <tag> <binding> <lane> <shape> <reps> identical fast
# builds arm A under MOJOLEARN_NUMERIC_MODE=identical (no define, raced as
# `ours`) and arm B under fast (no define, raced as `ours-fast`).
#
# Output: ~/mq/out/race-<tag>/ (AFN_OUT): race.log carries every line below;
# <A|B>-<rep>/race.txt is each race's driver output, <A|B>-<rep>/res/ its JSON
# and kept outputs; quality/ the judge's files. Lines (grep `AFN-`):
#   AFN-DEF <tag> head= bind= lane= shape= A='' B=''      the run
#   AFN-BUILD <tag> arm= mode= rc= defines=''             each build (or reused)
#   AFN-AB-RUN <tag> arm= rep= lane= shape= mode= status= median_ms= rounds= rc= defines='' quality=
#   AFN-AB <tag> arm=A|B lane= shape= median_ms= rounds= reps= runs=[] status= defines=''
#   AFN-DEF-SUMMARY <tag> A=<ms> B=<ms> ratio=<B/A>
#   AFN-QUALITY <tag> ... status=OK|DIFF|NONE            the output judge, then the tool's pair
#   AFN-QUALITY-TOOL <tag> ... status=SKIPPED|FAILED     when the tool judge could not run
#   AFN-FAIL <tag> ...                                   a race with no median (first errors follow)
#
# Env (every path is a parameter):
#   AFN_ROUNDS=3           timed rounds per race (one warm-up round precedes them)
#   AFN_DEF_ROOT=~/afn-def builds under $AFN_DEF_ROOT/<tag>/{A,B}.so (+ build logs)
#   AFN_SKIP_BUILD=1       reuse $AFN_DEF_ROOT/<tag>/{A,B}.so when present
#   AFN_OUT=~/mq/out/race-<tag>
#   AFN_PYTHON             the interpreter with numpy (default: the newest
#                          ~/board-*/cache/venv/bin/python, else python3)
#   AFN_QUALITY_TOOL=1     0 skips tools/neural_fast_quality.py (the output judge always runs)
#   AFN_CORPUS             samba judge corpus under training/corpus/<name>/input.txt
#                          (default: the first of enwik8, tinyshakespeare, pile_github present)
#   AFN_QSTEPS=100         samba judge steps; AFN_QEPOCHS=20 mlp judge epochs
#   AFN_REL_TOL=1e-4 AFN_LOSS_TOL=1e-3   the output judge's tolerances (tools/afn_ab.py)
#   AFN_READY_SECONDS=1800 AFN_ROUND_SECONDS=1800   the driver's waits
#   AFN_FAST_DIR=python/mojolearn AFN_IDENTICAL_DIR=python/mojolearn/identical
#                          where a FAST / IDENTICAL .so is installed
#   MOJOLEARN_COMPILE_JOBS passed through to the build scripts
#
# Run inside a BUILT tree (the other bindings a FAST import needs exist), on
# the Apple box, from the branch root or anywhere (it cd's to the repo).
# Written without running it (the tier lane never measures): every argument is
# checked, every build rc is checked, a failed race prints its first errors.
set -euo pipefail

# not tested: consume the shared AFN26 experiment definitions without copying
# flags by hand. This source-only entry prints a plan and exits before any
# build/install/timing path. Legacy positional invocations remain unchanged.
# Example: afn_ab.sh --experiment-plan AFN26-E08 --variant threads64
if [ "${1:-}" = --experiment-plan ]; then
  shift
  exec "${AFN_PYTHON:-python3}" "$(dirname "$0")/apple_fast_neural_ideas.py" plan "$@"
fi
if [ "${1:-}" = --experiment-list ]; then
  shift
  exec "${AFN_PYTHON:-python3}" "$(dirname "$0")/apple_fast_neural_ideas.py" list "$@"
fi

usage() {
  echo "       afn_ab.sh --experiment-plan AFN26-ID [--variant NAME]   (metadata only, not tested)" >&2
  echo "       afn_ab.sh --experiment-list                           (metadata only)" >&2
  echo "usage: afn_ab.sh <tag> <binding> <board-lane> <shape> <reps> \"<defines A>\" \"<defines B>\"" >&2
  echo "       afn_ab.sh <tag> <binding> <board-lane> <shape> <reps> identical fast   (baseline)" >&2
  echo "  binding: linalg|transformer|mamba|training|byte_lm|embedding|x_cnn; shape: full|small" >&2
  echo "  board-lane custom (binding embedding|x_cnn): tools/afn_custom_time.py, no board lane" >&2
  echo "  board-lane: lm-train-step lm-forward gemm gemm-bf16 gemm-int8 transformer-forward" >&2
  echo "              mamba1-forward mamba2-forward mamba3-forward samba-train-step samba-forward mlp-train-step" >&2
  [ $# -gt 0 ] && echo "afn_ab.sh: $*" >&2
  exit 2
}

[ $# -eq 7 ] || usage "expected 7 arguments, got $#"
TAG=$1 BIND=$2 LANE=$3 SHAPE=$4 REPS=$5 SPEC_A=$6 SPEC_B=$7

case $TAG in ''|*[!A-Za-z0-9._-]*) usage "tag '$TAG' must be [A-Za-z0-9._-]+" ;; esac
case $BIND in linalg|transformer|mamba|training|byte_lm|embedding|x_cnn) ;;
  *) usage "unknown binding '$BIND'" ;; esac
case $LANE in
  lm-train-step|lm-forward|gemm|gemm-bf16|gemm-int8|transformer-forward|mamba1-forward|mamba2-forward|\
  mamba3-forward|samba-train-step|samba-forward|mlp-train-step) ;;
  custom) case $BIND in embedding|x_cnn) ;; *) usage "lane custom times the embedding or x_cnn binding only (tools/afn_custom_time.py), got '$BIND'" ;; esac ;;
  *-infer|lm-host-train-step) usage "lane '$LANE' runs our CPU: never raced, in no numeric mode" ;;
  *) usage "unknown board lane '$LANE'" ;;
esac
case $SHAPE in full|small) ;; *) usage "shape '$SHAPE' is not full|small" ;; esac
case $REPS in ''|*[!0-9]*|0) usage "reps '$REPS' must be an integer >= 1" ;; esac
ROUNDS=${AFN_ROUNDS:-3}
case $ROUNDS in ''|*[!0-9]*|0) usage "AFN_ROUNDS '$ROUNDS' must be an integer >= 1" ;; esac
case $SPEC_A in identical|fast) ;; -D*|'') ;; *) usage "defines A '$SPEC_A' must start with -D (or be \"\", identical, fast)" ;; esac
case $SPEC_B in identical|fast) ;; -D*|'') ;; *) usage "defines B '$SPEC_B' must start with -D (or be \"\", identical, fast)" ;; esac
if [ "$SPEC_A" = "$SPEC_B" ]; then usage "arm A and arm B are the same build ('$SPEC_A')"; fi

here=$(cd "$(dirname "$0")/.." && pwd); cd "$here"
script=bindings/build_$BIND.sh
[ -f "$script" ] || usage "$script is missing"
FAST_DIR=${AFN_FAST_DIR:-python/mojolearn}
IDENT_DIR=${AFN_IDENTICAL_DIR:-python/mojolearn/identical}
SO_NAME=_mojolearn_$BIND.so
head=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)

# the interpreter: AFN_PYTHON, else the newest board venv, else python3
VP=${AFN_PYTHON:-}
if [ -z "$VP" ]; then
  for b in $(ls -d "$HOME"/board-* 2>/dev/null | sort -r); do
    if [ -x "$b/cache/venv/bin/python" ]; then VP=$b/cache/venv/bin/python; break; fi
  done
fi
VP=${VP:-python3}
command -v "$VP" >/dev/null 2>&1 || [ -x "$VP" ] || usage "python '$VP' not found (AFN_PYTHON)"

out=${AFN_DEF_ROOT:-$HOME/afn-def}/$TAG; mkdir -p "$out"
OUT=${AFN_OUT:-$HOME/mq/out/race-$TAG}; mkdir -p "$OUT"
LOG=$OUT/race.log
say() { echo "$*" | tee -a "$LOG"; }

arm_mode() { case $1 in identical) echo identical ;; *) echo fast ;; esac; }
arm_defs() { case $1 in identical|fast) echo "" ;; *) echo "$1" ;; esac; }
race_arm() { case $1 in identical) echo ours ;; *) echo ours-fast ;; esac; }
so_path() { case $1 in identical) echo "$IDENT_DIR/$SO_NAME" ;; *) echo "$FAST_DIR/$SO_NAME" ;; esac; }

MODE_A=$(arm_mode "$SPEC_A"); DEFS_A=$(arm_defs "$SPEC_A")
MODE_B=$(arm_mode "$SPEC_B"); DEFS_B=$(arm_defs "$SPEC_B")
say "AFN-DEF $TAG head=$head bind=$BIND lane=$LANE shape=$SHAPE reps=$REPS rounds=$ROUNDS python=$VP A='$SPEC_A' B='$SPEC_B'"

build() {  # $1 arm (A|B), $2 spec
  local arm=$1 spec=$2 mode defs log rc=0 built
  mode=$(arm_mode "$spec"); defs=$(arm_defs "$spec")
  if [ "${AFN_SKIP_BUILD:-0}" = 1 ] && [ -f "$out/$arm.so" ]; then
    say "AFN-BUILD $TAG arm=$arm mode=$mode rc=reused so=$out/$arm.so defines='$defs'"
    return 0
  fi
  log=$out/build_$arm.log
  if [ "$BIND" = byte_lm ]; then
    # build_byte_lm.sh refuses an existing output: build into a fresh directory
    rm -rf "$out/build-$arm"; mkdir -p "$out/build-$arm"
    MOJOLEARN_BYTE_LM_OUTDIR=$out/build-$arm MOJOLEARN_NUMERIC_MODE=$mode \
      MOJOLEARN_MOJO_BUILD_FLAGS="$defs" MOJOLEARN_SKIP_BUILD_GATE=1 \
      bash "$script" > "$log" 2>&1 || rc=$?
    built=$out/build-$arm/$SO_NAME
  else
    MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_MOJO_BUILD_FLAGS="$defs" MOJOLEARN_SKIP_BUILD_GATE=1 \
      bash "$script" > "$log" 2>&1 || rc=$?
    built=$(so_path "$mode")
  fi
  say "AFN-BUILD $TAG arm=$arm mode=$mode rc=$rc defines='$defs'"
  if [ "$rc" != 0 ] || [ ! -f "$built" ]; then
    { echo "AFN-BUILD-FAIL $TAG arm=$arm log=$log built=$built"; grep -m 5 -B 2 -A 8 -i error "$log" | cut -c1-300; } | tee -a "$LOG"
    exit 1
  fi
  cp "$built" "$out/$arm.so"
}

install_arm() {  # $1 arm, $2 spec: that arm's .so into the package for its mode
  local dst
  dst=$(so_path "$(arm_mode "$2")")
  mkdir -p "$(dirname "$dst")"
  cp "$out/$1.so" "$dst.tmp" && mv -f "$dst.tmp" "$dst"
}

race_once() {  # $1 arm, $2 spec, $3 rep
  local arm=$1 spec=$2 rep=$3 mode defs rarm d rc=0 line m st q
  mode=$(arm_mode "$spec"); defs=$(arm_defs "$spec"); rarm=$(race_arm "$spec")
  d=$OUT/$arm-$rep; rm -rf "$d"; mkdir -p "$d"
  env MOJOLEARN_NUMERIC_MODE=$mode MOJOLEARN_BENCH_INSTALLED=0 PYTHONPATH="$here/python" \
    MOJOLEARN_REPO_COMMIT=$head \
    $( [ "$LANE" = custom ] && echo "$VP tools/afn_custom_time.py --binding $BIND --rounds $ROUNDS --out $d/res --arm $rarm" \
       || echo "$VP tools/bench_board_neural.py race --lane $LANE --shape $SHAPE --arms $rarm --rounds $ROUNDS --out $d/res --work $d/work --ours-python $VP --keep-outputs --ready-seconds ${AFN_READY_SECONDS:-1800} --warmup-seconds ${AFN_ROUND_SECONDS:-1800} --round-seconds ${AFN_ROUND_SECONDS:-1800}" ) \
      > "$d/race.txt" 2>&1 || rc=$?
  rm -rf "$d/work"
  line=$(grep -m 1 "^NEURAL lane=$LANE arm=$rarm " "$d/race.txt" || true)
  m=$(echo "$line" | grep -o 'median_ms=[^ ]*' | cut -d= -f2 || true)
  st=$(echo "$line" | grep -o 'status=[^ ]*' | cut -d= -f2 || true)
  q=$(echo "$line" | grep -o 'quality=.*' | cut -c9-300 || true)
  [ "$m" = None ] && m=
  say "AFN-AB-RUN $TAG arm=$arm rep=$rep lane=$LANE shape=$SHAPE mode=$mode status=${st:-none} median_ms=${m:-none} rounds=$ROUNDS rc=$rc defines='$defs' quality=${q:-none}"
  if [ -z "$m" ]; then
    { echo "AFN-FAIL $TAG arm=$arm rep=$rep race=$d/race.txt"
      grep -E -m 5 'REFUSED|Traceback|Error|error|refused' "$d/race.txt" | cut -c1-300
      tail -n 4 "$d/race.txt" | cut -c1-300; } | tee -a "$LOG"
  fi
}

build A "$SPEC_A"; build B "$SPEC_B"
for r in $(seq 1 "$REPS"); do
  install_arm A "$SPEC_A"; race_once A "$SPEC_A" "$r"
  install_arm B "$SPEC_B"; race_once B "$SPEC_B" "$r"
done
"$VP" tools/afn_ab.py summary --log "$LOG" --tag "$TAG" --lane "$LANE" --shape "$SHAPE" \
  --rounds "$ROUNDS" --a-defines "$SPEC_A" --b-defines "$SPEC_B" | tee -a "$LOG" || true

# --- the judge, part 1: the two arms' kept outputs (always) -------------------
"$VP" tools/afn_ab.py compare-outputs --tag "$TAG" --lane "$LANE" \
  --a "$OUT/A-$REPS/res" --b "$OUT/B-$REPS/res" \
  --rel-tol "${AFN_REL_TOL:-1e-4}" --loss-tol "${AFN_LOSS_TOL:-1e-3}" | tee -a "$LOG" || true

# --- the judge, part 2: tools/neural_fast_quality.py, arm A then arm B --------
Q=$OUT/quality; rm -rf "$Q"; mkdir -p "$Q"
nfq() {  # $1 arm, $2 spec, rest: subcommand args; runs with that arm installed
  local arm=$1 spec=$2 mode rc=0; shift 2
  mode=$(arm_mode "$spec")
  install_arm "$arm" "$spec"
  env MOJOLEARN_NUMERIC_MODE=$mode PYTHONPATH="$here/python" MOJOLEARN_NFQ_PACKAGE_DIR="$here/python" \
    "$VP" tools/neural_fast_quality.py "$@" --mode "$mode" > "$Q/$arm.txt" 2>&1 || rc=$?
  if [ "$rc" != 0 ]; then
    { echo "AFN-QUALITY-TOOL $TAG arm=$arm cmd=$1 rc=$rc status=FAILED"
      grep -E -m 5 'Traceback|Error|error|REFUSED|missing' "$Q/$arm.txt" | cut -c1-300; } | tee -a "$LOG"
  fi
  return $rc
}
tool_judge() {
  local corpus=${AFN_CORPUS:-} c
  if [ -z "$corpus" ]; then
    for c in enwik8 tinyshakespeare pile_github; do
      [ -f "training/corpus/$c/input.txt" ] && { corpus=$c; break; }
    done
  fi
  case $LANE in
    samba-train-step|samba-forward)
      [ -n "$corpus" ] && [ -f "training/corpus/$corpus/input.txt" ] || {
        say "AFN-QUALITY-TOOL $TAG lane=$LANE status=SKIPPED no corpus under training/corpus/<name>/input.txt (AFN_CORPUS)"; return 0; }
      nfq A "$SPEC_A" samba --corpus "$corpus" --seed 0 --steps "${AFN_QSTEPS:-100}" \
          --out "$Q/A.json" --init-out "$Q/init.npz" || return 0
      nfq B "$SPEC_B" samba --corpus "$corpus" --seed 0 --steps "${AFN_QSTEPS:-100}" \
          --out "$Q/B.json" --init "$Q/init.npz" || return 0 ;;
    mlp-train-step)
      "$VP" tools/neural_fast_quality.py mlp-data --out-npz "$Q/mlp.npz" > "$Q/data.txt" 2>&1 || {
        say "AFN-QUALITY-TOOL $TAG lane=$LANE status=SKIPPED mlp-data failed (scikit-learn?): $(grep -m 1 -E 'Error|error' "$Q/data.txt" | cut -c1-200)"; return 0; }
      nfq A "$SPEC_A" mlp --data "$Q/mlp.npz" --dataset wine --seed 0 --epochs "${AFN_QEPOCHS:-20}" --out "$Q/A.json" || return 0
      nfq B "$SPEC_B" mlp --data "$Q/mlp.npz" --dataset wine --seed 0 --epochs "${AFN_QEPOCHS:-20}" --out "$Q/B.json" || return 0 ;;
    transformer-forward|mamba1-forward|mamba2-forward|mamba3-forward)
      nfq A "$SPEC_A" blocks --seeds 1 --out "$Q/A.json" || return 0
      nfq B "$SPEC_B" blocks --seeds 1 --out "$Q/B.json" || return 0 ;;
    *)
      say "AFN-QUALITY-TOOL $TAG lane=$LANE status=SKIPPED no neural_fast_quality subcommand for this lane (the output judge above is the rule)"
      return 0 ;;
  esac
  "$VP" tools/neural_fast_quality.py pair --ref "$Q/A.json" --cand "$Q/B.json" --tag "$TAG" | tee -a "$LOG" || true
}
if [ "${AFN_QUALITY_TOOL:-1}" = 1 ]; then tool_judge; else say "AFN-QUALITY-TOOL $TAG status=SKIPPED AFN_QUALITY_TOOL=0"; fi

# arm B's .so stays installed
install_arm B "$SPEC_B"
say "AFN-DONE $TAG installed=B ($(so_path "$MODE_B")) log=$LOG"
