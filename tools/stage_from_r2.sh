#!/bin/sh
# tools/stage_from_r2.sh -- DEVIATION 2704 (Andrew, 2026-09-13: "cloudflare has
# datasets already saved and when using runpod we should ALWAYS use them"; and:
# "make sure all of our shit ships corpora from R2 instead of downloading").
#
# THE ONE LINE A RUNNER CALLS AFTER THE SOURCE IS UNPACKED ON THE BOX:
#
#   sh tools/stage_from_r2.sh "<ssh flags+target>" [key...]
#
# Runs ON THIS MACHINE (the credential in ~/.mojolearn_r2 never leaves it:
# tools/dataset_store.sh presigns). Stages each key onto the box at the store's
# box path, verified against bench/results/dataset_store/manifest.tsv, then
# links a corpus key into the box's source tree where tools/fetch_corpus_*.sh
# look, so their own --check passes and nothing is downloaded. The gbm-bench
# npz keys land under /root/datasets/gbm-bench, which is GBM_BENCH_DATA's
# default on a box, so tools/speed_gbdt_arm.py returns early without a fetch.
#
#   keys                  default: the board's four
#                           gbm-bench/taxi/taxi_speed.npz
#                           gbm-bench/istella/istella_speed.npz
#                           corpus/enwik8/input.txt
#                           corpus/pile_github/input.txt
#                         The forest datasets are pinned too and land where
#                         tools/speed_gbdt_arm.py reads them (added 2026-09-17)
#                           gbm-bench/higgs/higgs_speed.npz
#                           gbm-bench/covtype/covtype_speed.npz
#                           gbm-bench/year/year_speed.npz
#                         MOJOLEARN_STAGE_KEYS="" (empty, set) stages nothing.
#   MOJOLEARN_STAGE_REMOTE_SH   the shell that runs the staging commands on the
#                         box (default `sh -s`; Hot Aisle needs `sudo -n -H sh -s`
#                         because the ssh user is not root)
#   MOJOLEARN_STAGE_BOX_REPO    the box's source tree (default /root/mojolearn)
#   MOJOLEARN_STAGE_STRICT=1    a staging failure exits non-zero (default: it is
#                         reported on the last line and the caller goes on, so
#                         the body's own fetchers still run; a raw download on a
#                         rented box is then VISIBLE in the leg's evidence)
#
# Prints one summary line last: "R2 STAGED n key(s), m linked" or "R2 STAGING
# FAILED ...". POSIX sh.
set -u
TARGET="${1:?usage: stage_from_r2.sh \"<ssh flags+target>\" [key...]}"; shift
ROOT=$(cd "$(dirname "$0")/.." && pwd)
BOX_REPO="${MOJOLEARN_STAGE_BOX_REPO:-/root/mojolearn}"
REMOTE_SH="${MOJOLEARN_STAGE_REMOTE_SH:-sh -s}"
if [ "$#" -gt 0 ]; then
    KEYS="$*"
elif [ "${MOJOLEARN_STAGE_KEYS+set}" = set ]; then
    KEYS="$MOJOLEARN_STAGE_KEYS"
else
    KEYS="gbm-bench/taxi/taxi_speed.npz gbm-bench/istella/istella_speed.npz corpus/enwik8/input.txt corpus/pile_github/input.txt"
fi
[ -n "$KEYS" ] || { echo "R2 STAGED 0 key(s) (MOJOLEARN_STAGE_KEYS is empty)"; exit 0; }
R2FILE="${MOJOLEARN_R2_FILE:-$HOME/.mojolearn_r2}"
if [ ! -f "$R2FILE" ]; then
    echo "R2 STAGING FAILED: no $R2FILE on this machine; the body's fetchers will download from the origins"
    [ "${MOJOLEARN_STAGE_STRICT:-0}" = 1 ] && exit 1
    exit 0
fi
_t0=$(date +%s)
# shellcheck disable=SC2086
if ! MOJOLEARN_STAGE_REMOTE_SH="$REMOTE_SH" sh "$ROOT/tools/dataset_store.sh" stage "$TARGET" $KEYS; then
    echo "R2 STAGING FAILED after $(( $(date +%s) - _t0 ))s (keys: $KEYS); the body's fetchers will download from the origins"
    [ "${MOJOLEARN_STAGE_STRICT:-0}" = 1 ] && exit 1
    exit 0
fi
# Link corpora into the source tree. The box path of a ROOT/ key is derived by
# dataset_store.sh from THIS machine's checkout path, which is not the box's.
_link=""
for k in $KEYS; do
    case "$k" in
        corpus/enwik8/input.txt)      _rel=training/corpus/enwik8/input.txt ;;
        corpus/pile_github/input.txt) _rel=training/corpus/pile_github/input.txt ;;
        *) continue ;;
    esac
    _bp=$(sh "$ROOT/tools/dataset_store.sh" box-cmd "$k" 2>/dev/null | grep -o '"/root[^"]*"' | head -1 | tr -d '"')
    [ -n "$_bp" ] || continue
    _link="$_link
[ -f '$_bp' ] && mkdir -p '$BOX_REPO/$(dirname "$_rel")' && ln -f '$_bp' '$BOX_REPO/$_rel' && echo 'linked $k -> $BOX_REPO/$_rel'"
done
_linked=0
if [ -n "$_link" ]; then
    # shellcheck disable=SC2086
    _out=$(printf '%s\n' "$_link" | ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR $TARGET "$REMOTE_SH" 2>&1) || true
    printf '%s\n' "$_out"
    _linked=$(printf '%s\n' "$_out" | grep -c '^linked ')
fi
echo "R2 STAGED $(echo "$KEYS" | wc -w | tr -d ' ') key(s) in $(( $(date +%s) - _t0 ))s, $_linked linked into $BOX_REPO"
