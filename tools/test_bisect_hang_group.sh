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

echo "FAILS=$FAILS"
[ "$FAILS" = 0 ]
