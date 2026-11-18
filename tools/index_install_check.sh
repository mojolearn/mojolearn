#!/bin/bash
# tools/index_install_check.sh -- THE USER'S INSTALL OF A SPLIT LINUX RELEASE,
# RESOLVED FROM AN INDEX, on an NVIDIA box and an AMD box at once (2026-09-26).
#
#   bash tools/index_install_check.sh <V> testpypi|pypi            DRY RUN of both legs
#   bash tools/index_install_check.sh <V> testpypi|pypi --rent     rents both boxes
#
# The release columns install LOCAL wheel files. Nothing else tests what a user
# types, `pip install mojolearn==V` resolved from TestPyPI or PyPI, which is
# where the core <-> plugin exact-pin cycle and the publish order can break.
# This runs tools/release_wheel_smoke.sh --from-index on two boxes IN PARALLEL:
#
#   NVIDIA  --vendor cuda on RunPod (the smoke's default GPU, or --gpu)
#   AMD     --vendor hip --provider do (DigitalOcean), unless --amd-provider says
#           otherwise (the Hot Aisle default and its walk to DigitalOcean were removed
#           2026-11-18, lane fx-dead-small; recoverable at 7c1aad6561)
#
# and prints one PASS/FAIL line per vendor with its evidence directory.
#
# BEFORE ANYTHING IS RENTED (and in the dry run), tools/index_release_check.py
# precheck asks the index's JSON API for mojolearn, mojolearn-nvidia and
# mojolearn-amd at V and refuses by name when one is missing, has no manylinux
# x86_64 wheel, is yanked, or (from the wheel's own METADATA) the core does not
# pin both plugins at ==V or a plugin does not pin mojolearn==V.
#
# Options:
#   --rent                     create the boxes; without it both legs are dry runs
#   --out DIR                  default ~/mojolearn-evidence/index-install/<V>/<index>/<stamp>/
#                              (nvidia/ and amd/ inside, one per leg, plus each leg's log)
#   --gpu NAME                 the NVIDIA leg's RunPod GPU type
#   --amd-provider P           do (default, DigitalOcean), auto (RunPod, then DigitalOcean), runpod
#   --expected-source-commit C the installed package must record commit C (both legs)
#   --nvidia-column SEL        also run the release column of SEL (a cuda selection) on the NVIDIA box
#   --amd-column SEL           the same on the AMD box (a hip selection)
#   --ref-column FILE          repeatable: diff each column against FILE (as the smoke's --ref-column)
#   --lease MIN, --smoke-seconds N   passed to both legs
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
SMOKE="$ROOT/tools/release_wheel_smoke.sh"
CHECK="$ROOT/tools/index_release_check.py"

say() { printf '[%s index-install] %s\n' "$(date +%T)" "$*"; }
die() { printf '\nREFUSED: %s\n' "$*" >&2; exit 1; }

VERSION=""; INDEX=""; RENT=0; OUT=""; GPU=""; AMD_PROVIDER=do; COMMIT=""
NV_COL=""; AMD_COL=""; REFS=(); COMMON=()
while [ $# -gt 0 ]; do
    case "$1" in
        --rent) RENT=1 ;;
        --out) shift; OUT="${1:-}" ;;
        --gpu) shift; GPU="${1:-}" ;;
        --amd-provider) shift; AMD_PROVIDER="${1:-}" ;;
        --expected-source-commit) shift; COMMIT="${1:-}" ;;
        --nvidia-column) shift; NV_COL="${1:-}" ;;
        --amd-column) shift; AMD_COL="${1:-}" ;;
        --ref-column) shift; REFS+=(--ref-column "${1:-}") ;;
        --lease|--smoke-seconds) COMMON+=("$1" "${2:-}"); shift ;;
        -h|--help) sed -n '2,/^set -u$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) if [ -z "$VERSION" ]; then VERSION="$1"; elif [ -z "$INDEX" ]; then INDEX="$1"; else die "unexpected argument '$1'"; fi ;;
    esac
    shift
done
[ -n "$VERSION" ] && [ -n "$INDEX" ] || die "usage: tools/index_install_check.sh <V> testpypi|pypi [--rent] (see --help)"
case "$INDEX" in testpypi|pypi) ;; *) die "the index must be testpypi or pypi, not '$INDEX'" ;; esac
case "$AMD_PROVIDER" in auto|runpod|do) ;; *) die "--amd-provider must be do, auto or runpod" ;; esac
[ -z "$COMMIT" ] || printf '%s' "$COMMIT" | grep -Eq '^[0-9a-f]{40}$' || die "--expected-source-commit must be the full 40-hex commit"
[ -z "$NV_COL" ] || [ -f "$NV_COL" ] || die "no --nvidia-column selection $NV_COL"
[ -z "$AMD_COL" ] || [ -f "$AMD_COL" ] || die "no --amd-column selection $AMD_COL"
[ "${#REFS[@]}" = 0 ] || [ -n "$NV_COL$AMD_COL" ] || die "--ref-column needs --nvidia-column or --amd-column"

STAMP=$(date -u +%Y%m%d-%H%M%S)
[ -n "$OUT" ] || OUT="${MOJOLEARN_EVIDENCE_ROOT:-$HOME/mojolearn-evidence}/index-install/$VERSION/$INDEX/$STAMP"

echo "== index_install_check: pip install mojolearn==$VERSION from $INDEX, NVIDIA and AMD in parallel, $([ "$RENT" = 1 ] && echo RENT || echo 'DRY RUN') =="
# THE PRECHECK, before anything is rented: the index must serve all three.
python3 "$CHECK" precheck --index "$INDEX" --version "$VERSION" \
    || die "the index precheck refused (above); nothing was rented. pip install mojolearn==$VERSION cannot resolve from $INDEX until all three projects serve $VERSION"

# the arguments of one leg: vendor-specific first, then what both share
leg_args() {  # vendor provider column -> the smoke's argument list, one per line
    printf '%s\n' --from-index "$INDEX" --version "$VERSION" --vendor "$1"
    [ "$1" = cuda ] || printf '%s\n' --provider "$2"
    [ "$1" != cuda ] || [ -z "$GPU" ] || printf '%s\n' --gpu "$GPU"
    [ -z "$COMMIT" ] || printf '%s\n' --expected-source-commit "$COMMIT"
    if [ -n "$3" ]; then
        printf '%s\n' --column "$3"
        [ "${#REFS[@]}" = 0 ] || printf '%s\n' "${REFS[@]}"
    fi
    [ "${#COMMON[@]}" = 0 ] || printf '%s\n' "${COMMON[@]}"
    [ "$RENT" = 0 ] || printf '%s\n' --rent
}
run_leg() {  # label vendor provider column outdir log
    _args=()
    while IFS= read -r _a; do _args+=("$_a"); done < <(leg_args "$2" "$3" "$4")
    bash "$SMOKE" "${_args[@]}" --out "$5" > "$6" 2>&1
}
amd_leg() {  # outdir-base log
    run_leg amd hip "$AMD_PROVIDER" "$AMD_COL" "$1" "$2"
}

if [ "$RENT" = 1 ]; then
    mkdir -p "$OUT" || die "cannot create $OUT"
    LOGS="$OUT"
else
    LOGS=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-index-install.XXXXXX")
fi
say "NVIDIA leg -> $OUT/nvidia (log $LOGS/nvidia.log)"
run_leg nvidia cuda runpod "$NV_COL" "$OUT/nvidia" "$LOGS/nvidia.log" &
NV_PID=$!
say "AMD leg    -> $OUT/amd (log $LOGS/amd.log; provider $AMD_PROVIDER)"
amd_leg "$OUT/amd" "$LOGS/amd.log" &
AMD_PID=$!
wait "$NV_PID"; NV_RC=$?
wait "$AMD_PID"; AMD_RC=$?

verdict() {  # rc log -> the leg's verdict line from the smoke, or its refusal
    _line=$(grep -E '^verdict=' "$2" | tail -1)
    [ -n "$_line" ] || _line=$(grep -E '^REFUSED: ' "$2" | tail -1)
    [ -n "$_line" ] || _line=$(tail -1 "$2")
    printf '%s' "$_line"
}
echo
if [ "$RENT" = 0 ]; then
    for _l in nvidia amd; do
        echo "---- $_l leg (dry run) ----"
        sed 's/^/  /' "$LOGS/$_l.log"
    done
    rm -rf "$LOGS"
    echo
    [ "$NV_RC" = 0 ] && echo "DRY RUN NVIDIA: plan composed" || echo "DRY RUN NVIDIA: REFUSED (exit $NV_RC)"
    [ "$AMD_RC" = 0 ] && echo "DRY RUN AMD: plan composed" || echo "DRY RUN AMD: REFUSED (exit $AMD_RC)"
    echo "DRY RUN: nothing was created and nothing was billed. Add --rent to create both boxes."
    [ "$NV_RC" = 0 ] && [ "$AMD_RC" = 0 ]
    exit $?
fi
{
    echo "index=$INDEX version=$VERSION stamp=$STAMP"
    if [ "$NV_RC" = 0 ]; then echo "NVIDIA PASS  $OUT/nvidia"; else echo "NVIDIA FAIL  $OUT/nvidia  ($(verdict "$NV_RC" "$LOGS/nvidia.log"))"; fi
    if [ "$AMD_RC" = 0 ]; then echo "AMD    PASS  $OUT/amd"; else echo "AMD    FAIL  $OUT/amd  ($(verdict "$AMD_RC" "$LOGS/amd.log"))"; fi
} | tee "$OUT/summary.txt"
[ "$NV_RC" = 0 ] && [ "$AMD_RC" = 0 ]
