#!/bin/sh
# tools/opponent_wheels.sh -- the opponent libraries, pinned and held in R2,
# instead of resolved from PyPI on every rented box.
#
# WHY THIS EXISTS. DEVIATION 2704 put the corpora in R2 and left the opponents
# on PyPI, so a leg that measures CatBoost still depended on (a) PyPI being up
# at the minute the lease started and (b) a resolver choosing the same version
# it chose last time. Both of those are load bearing, because an opponent row
# in bench/OPPONENT_REFERENCE.md is only valid for the tuple
#   (GPU model, driver, LIBRARY VERSION, dataset or shape, parameters)
# and `pipget cuml-cu12` with no `==` is a different library version every few
# weeks. A row measured against one set of bytes and quoted against another is
# a wrong number with no symptom.
#
# So the bytes are pinned by size AND sha256 in
# bench/results/dataset_store/manifest.tsv, exactly the way a corpus is, and a
# box refuses a wheel whose bytes do not match. The R2 credential never leaves
# this machine: tools/dataset_store.sh presigns here and the box receives only
# a short-lived URL.
#
#   sh tools/opponent_wheels.sh sets
#   sh tools/opponent_wheels.sh pins <set>
#   sh tools/opponent_wheels.sh fetch <set>     # resolve + download HERE, write
#                                               # the committed member list
#   sh tools/opponent_wheels.sh box-install <dir> <pkg>...
#                                               # what a BOX runs: offline
#                                               # install out of a staged set
#
# Then, as for any other store key:
#   sh tools/dataset_store.sh manifest
#   sh tools/dataset_store.sh push   opponents/<set>
#   sh tools/stage_from_r2.sh "<ssh flags+target>" opponents/<set>
#
# THE SETS ARE SPLIT BY WHAT A LEG NEEDS, NOT BY VENDOR. A trees leg wants 325
# MB; making it also carry RAPIDS would be 1.6 GB of wheels for a family that
# never imports cuml. The split is the whole reason this is cheap.
#
# THE PYTHON TAG IS PART OF THE SET NAME AND IT IS NOT COSMETIC. catboost,
# cuml-cu12, scikit-learn and numpy all ship per-interpreter wheels; xgboost
# and lightgbm are py3-none. `cp311` is the tag of the image these legs rent
# (runpod/pytorch:2.4.0-py3.11-cuda12.4.1). A box on another interpreter finds
# no compatible wheel in the set and REFUSES BY NAME -- which is the correct
# outcome, and better than silently resolving something else from PyPI. Add a
# set for that interpreter rather than widening this one.
#
# POSIX sh. `fetch` needs python3 + pip on this Mac and downloads only; it
# compiles nothing and runs no opponent.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
WHEEL_ROOT="${MOJOLEARN_OPPONENT_WHEEL_DIR:-$HOME/opponent-wheels}"
LIST_DIR="$ROOT/bench/results/dataset_store/groups"

# set <TAB> extra index (or -) <TAB> pinned requirements
#
# EVERY VERSION HERE IS THE ONE A ROW IN bench/OPPONENT_REFERENCE.md WAS
# MEASURED AT. Changing one is not a maintenance edit: it invalidates every
# row measured at the old version, and the table says so in its header. If you
# move a pin, move the rows with it or mark them.
#
# cuvs-cu12 IS 26.8.1 AND NOT 26.8.0 ON PURPOSE. There is no 26.8.0 of
# cuvs-cu12 on pypi.nvidia.com at all (checked 2026-09-17: the index jumps
# 26.6.0 -> 26.8.1), which is why `cuvs-cu12==26.8.0` in a leg resolves to
# nothing. cuml-cu12 IS 26.8.0; the two packages are not released in lockstep.
setspec() {
    printf '%s\n' \
"trees-linux-x86_64-cp311	-	catboost==1.2.10 xgboost==3.2.0 lightgbm==4.7.0 scikit-learn==1.7.2" \
"rapids-linux-x86_64-cp311	https://pypi.nvidia.com	cuml-cu12==26.8.0 cuvs-cu12==26.8.1"
}

# The interpreter and platform tags a set resolves for, derived from its name
# so the name cannot drift from what is inside it.
tags_for() {
    case "$1" in
        *-cp311) echo "311" ;;
        *-cp310) echo "310" ;;
        *-cp312) echo "312" ;;
        *) echo "" ;;
    esac
}

spec_field() { setspec | awk -F'\t' -v s="$1" -v f="$2" '$1==s {print $f}'; }

list_file_for() { printf '%s/opponents@%s.txt\n' "$LIST_DIR" "$1"; }

cmd_sets() {
    printf '%-34s %-26s %s\n' SET INDEX PINS
    setspec | while IFS='	' read -r s idx reqs; do
        printf '%-34s %-26s %s\n' "$s" "$idx" "$reqs"
    done
}

cmd_pins() {
    s="${1:?usage: pins <set>}"
    _r=$(spec_field "$s" 3)
    [ -n "$_r" ] || { echo "unknown set: $s (see 'sets')" >&2; return 1; }
    for _p in $_r; do echo "$_p"; done
}

cmd_fetch() {
    s="${1:?usage: fetch <set>}"
    _reqs=$(spec_field "$s" 3)
    [ -n "$_reqs" ] || { echo "unknown set: $s (see 'sets')" >&2; return 1; }
    _idx=$(spec_field "$s" 2)
    _pyver=$(tags_for "$s")
    [ -n "$_pyver" ] || { echo "set name does not end in a -cpNNN tag: $s" >&2; return 1; }
    _dir="$WHEEL_ROOT/$s"
    mkdir -p "$_dir" "$LIST_DIR"
    _extra=""
    [ "$_idx" = "-" ] || _extra="--extra-index-url $_idx"
    echo "resolving $s into $_dir ..."
    # --only-binary=:all: because a set that needs a compiler on the box is not
    # a mirror, it is the source build this exists to avoid. The five platform
    # tags are the manylinux flavours these projects actually publish; pip
    # picks the newest each package offers.
    # shellcheck disable=SC2086
    nice -n 19 python3 -m pip download --no-input --disable-pip-version-check \
        -d "$_dir" $_extra --only-binary=:all: \
        --implementation cp --python-version "$_pyver" \
        --platform manylinux_2_28_x86_64 --platform manylinux_2_27_x86_64 \
        --platform manylinux_2_24_x86_64 --platform manylinux2014_x86_64 \
        --platform manylinux_2_17_x86_64 \
        $_reqs || { echo "resolve FAILED for $s" >&2; return 1; }
    # THE MEMBER LIST IS WHAT THE RESOLVER PRODUCED, WRITTEN DOWN AND
    # COMMITTED. It is the set's inventory: dataset_store.sh stages exactly
    # these names and manifest.tsv carries a size and sha256 for each, so a
    # wheel that appears in the directory later and is not in this list is not
    # staged and cannot reach a box.
    _lf=$(list_file_for "$s")
    {
        echo "# opponents/$s -- written by tools/opponent_wheels.sh fetch"
        echo "# pins: $_reqs"
        [ "$_idx" = "-" ] || echo "# extra index: $_idx"
        ls -1 "$_dir" | grep '\.whl$' | LC_ALL=C sort
    } > "$_lf"
    _n=$(grep -c '\.whl$' "$_lf")
    _b=$(cat "$_dir"/*.whl 2>/dev/null | wc -c | tr -d ' ')
    echo "wrote $_lf ($_n wheels, $_b bytes)"
    echo "next: sh tools/dataset_store.sh manifest && sh tools/dataset_store.sh push opponents/$s"
}

# WHAT A RENTED BOX RUNS. --no-index is the point: the box installs out of the
# staged, sha256-verified directory and never reaches PyPI, so the install is
# the same bytes every lease and works on a box with no route to pypi.org.
#
# It does NOT pass --force-reinstall or --upgrade: a dependency the image
# already satisfies stays as the image has it, so staging a wheel set cannot
# quietly lift an image's numpy out from under its torch.
cmd_box_install() {
    _dir="${1:?usage: box-install <staged dir> <pkg>...}"; shift
    [ "$#" -gt 0 ] || { echo "no packages given" >&2; return 1; }
    [ -d "$_dir" ] || { echo "no staged wheel set at $_dir" >&2; return 2; }
    _n=$(ls -1 "$_dir" 2>/dev/null | grep -c '\.whl$') || _n=0
    [ "$_n" -gt 0 ] || { echo "staged wheel set $_dir holds no wheels" >&2; return 2; }
    python3 -m pip install --no-input --disable-pip-version-check \
        --no-index --find-links "$_dir" "$@"
}

case "${1:-}" in
    sets)        shift; cmd_sets "$@" ;;
    pins)        shift; cmd_pins "$@" ;;
    fetch)       shift; cmd_fetch "$@" ;;
    box-install) shift; cmd_box_install "$@" ;;
    *) sed -n '2,40p' "$0"; exit 2 ;;
esac
