#!/bin/bash
# tools/test_bisect_hang_group.sh -- checks tools/bisect_hang_group.sh without compiling anything:
#   bash tools/test_bisect_hang_group.sh
# 1. bash -n (and shellcheck when installed) on the script and this test.
# 2. --dry-run prints the plan: control-head, control-all and one candidate per suspect, each with the
#    development_build.py command (one compiler job, serial, the stall bound, its own cache and evidence).
# 3. The committed base archive covers every default suspect (content or the absent list) and names its base.
# 4. --export-base round trip when this checkout has the base commit.
# Needs only bash, tar and git; touches only a mktemp dir.
# shellcheck disable=SC2015  # `check && ok || bad`: ok only echoes, it cannot fail
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
S="$R/tools/bisect_hang_group.sh"
A="$R/tools/bisect_hang_base/e49ac7df70.tgz"
L="$R/tools/bisect_hang_base/classical-host-suspects.txt"
T=$(mktemp -d "${TMPDIR:-/tmp}/bisect-test.XXXXXX")
trap 'rm -rf "$T"' EXIT
FAILS=0
ok() { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; FAILS=$((FAILS + 1)); }

for f in "$S" "$0"; do
    if bash -n "$f"; then ok "bash -n $(basename "$f")"; else bad "bash -n $(basename "$f")"; fi
    if command -v shellcheck > /dev/null; then
        if shellcheck "$f"; then ok "shellcheck $(basename "$f")"; else bad "shellcheck $(basename "$f")"; fi
    fi
done

N=$(grep -c -v -E '^[[:space:]]*(#|$)' "$L")
bash "$S" --group _mojolearn_dev_runtime_classical_host --set host --base e49ac7df70 --suspects "$L" \
    --out "$T/out" --work "$T/work" --parallel 4 --dry-run > "$T/plan.txt" 2>&1
rc=$?
[ "$rc" = 0 ] && ok "dry run exits 0" || bad "dry run rc=$rc: $(head -n 3 "$T/plan.txt")"
grep -q '^DRY RUN' "$T/plan.txt" && ok "dry run says so" || bad "no DRY RUN banner"
got=$(grep -c '^PLAN ' "$T/plan.txt")
[ "$got" = $((N + 2)) ] && ok "plan has $got candidates (2 controls + $N suspects)" || bad "plan has $got candidates, want $((N + 2))"
grep -q $'^PLAN control-head\trevert=-\t' "$T/plan.txt" && ok "control-head reverts nothing" || bad "control-head line"
grep -q $'^PLAN control-all\trevert=ALL\t' "$T/plan.txt" && ok "control-all reverts all" || bad "control-all line"
grep -q $'^PLAN c00\trevert=bindings/admitted_call.mojo\t' "$T/plan.txt" && ok "c00 is the first suspect" || bad "c00 line"
[ "$(grep -c 'MOJOLEARN_COMPILE_JOBS=1 .*tools/development_build.py build --sets host --group _mojolearn_dev_runtime_classical_host --parallel 1 --stall-seconds 420 --group-seconds 3600 ' "$T/plan.txt")" = $((N + 2)) ] \
    && ok "every candidate runs one serial group build with the stall bound" || bad "candidate command shape"
grep -q -- "--cache $T/work/cache/c03 --evidence $T/out/cand/c03 --json $T/out/cand/c03.json" "$T/plan.txt" \
    && ok "own cache, evidence and json per candidate" || bad "per-candidate paths"
[ -f "$T/out/plan.tsv" ] && [ "$(wc -l < "$T/out/plan.tsv" | tr -d ' ')" = $((N + 2)) ] && ok "plan.tsv written" || bad "plan.tsv"
[ ! -e "$T/work/wt" ] && ok "dry run copied nothing" || bad "dry run made trees"

bash "$S" --group g --set host --base e49ac7df70 --out "$T/o2" --work "$T/o2/work" --dry-run > /dev/null 2>&1
[ $? = 2 ] && ok "--work inside --out is refused" || bad "--work inside --out accepted"
bash "$S" --group g --set gpu --base e49ac7df70 --out "$T/o3" --dry-run > /dev/null 2>&1
[ $? = 2 ] && ok "--set gpu is refused" || bad "--set gpu accepted"

# the committed archive covers every suspect
if [ -f "$A" ]; then
    mkdir -p "$T/a" && tar -C "$T/a" -xzf "$A"
    miss=0
    while IFS= read -r f; do
        [ -f "$T/a/$f" ] || grep -qxF "$f" "$T/a/.bisect_absent" || { echo "  not covered: $f"; miss=1; }
    done < <(grep -v -E '^[[:space:]]*(#|$)' "$L")
    [ "$miss" = 0 ] && ok "archive covers all $N suspects" || bad "archive misses suspects"
    case "$(cat "$T/a/.bisect_base" 2> /dev/null)" in e49ac7df70*) ok "archive names its base" ;; *) bad "archive base" ;; esac
else
    bad "no committed archive $A"
fi

# --export-base round trip (only where git has the base)
if git -C "$R" rev-parse --verify --quiet 'e49ac7df70^{commit}' > /dev/null 2>&1; then
    bash "$S" --export-base "$T/x.tgz" --base e49ac7df70 --suspects "$L" > /dev/null 2>&1 && mkdir -p "$T/x" && tar -C "$T/x" -xzf "$T/x.tgz"
    if [ -d "$T/x" ] && diff -r "$T/a" "$T/x" > /dev/null 2>&1; then ok "export-base reproduces the archive"; else bad "export-base differs from the archive"; fi
else
    echo "SKIP: export-base round trip (no git history for e49ac7df70 here)"
fi

# history mode: the committed export and its dry-run plan
HD="$R/tools/bisect_hang_history"
if [ -f "$HD/history.tsv" ]; then
    NH=$(( $(wc -l < "$HD/history.tsv") - 1 ))
    bash "$S" --group _mojolearn_dev_runtime_classical_host --set host --history "$HD" \
        --out "$T/hout" --work "$T/hwork" --dry-run > "$T/hplan.txt" 2>&1
    rc=$?
    [ "$rc" = 0 ] && ok "history dry run exits 0" || bad "history dry run rc=$rc: $(head -n 3 "$T/hplan.txt")"
    [ "$(grep -c '^PLAN h[0-9]' "$T/hplan.txt")" = "$NH" ] && ok "history plan has $NH commits" || bad "history plan size"
    grep -q $'^PLAN h00\trevert=H:h00\t0 e49ac7df70 ' "$T/hplan.txt" && ok "h00 is the base" || bad "h00 line"
    case "$(awk -F'\t' '$1 == "head" { print $2 }' "$HD/meta.tsv")" in f9e06ecacf*) ok "export head is f9e06ecacf" ;; *) bad "export head" ;; esac
    mkdir -p "$T/hb" && tar -C "$T/hb" -xzf "$HD/blobs.tgz"
    lost=$(cut -f2 "$HD"/manifest/*.tsv | awk '$0 != "-"' | sort -u | while IFS= read -r b; do [ -f "$T/hb/$b" ] || echo "$b"; done | wc -l | tr -d ' ')
    [ "$lost" = 0 ] && ok "every manifest blob is in blobs.tgz" || bad "$lost manifest blobs missing from blobs.tgz"
    np=$(awk -F'\t' '$1 == "paths" { print $2 }' "$HD/meta.tsv")
    short=0; for m in "$HD"/manifest/*.tsv; do [ "$(wc -l < "$m" | tr -d ' ')" = "$np" ] || short=1; done
    [ "$short" = 0 ] && ok "every manifest lists all $np paths" || bad "a manifest is short"
    # where git has the commits: h00 is the base plus the overlay, the last is the head
    if git -C "$R" rev-parse --verify --quiet 'f9e06ecacf^{commit}' > /dev/null 2>&1; then
        git -C "$R" ls-tree -r f9e06ecacf | awk -F'\t' '{ split($1, a, " "); print $2 "\t" a[3] }' > "$T/head.tree"
        last=$(printf 'h%02d' $((NH - 1)))
        n=$(awk -F'\t' 'NR == FNR { h[$1] = $2; next } { x = ($1 in h) ? h[$1] : "-"; if (x != $2) n++ } END { print n + 0 }' "$T/head.tree" "$HD/manifest/$last.tsv")
        [ "$n" = 0 ] && ok "$last is the head content" || bad "$last differs from the head in $n paths"
    fi
else
    bad "no committed history export $HD"
fi

echo "FAILS=$FAILS"
[ "$FAILS" = 0 ]
