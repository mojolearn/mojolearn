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

# root mode: dry-run plan, then the rewriter itself (python3 only; no compiler)
RT=bindings/_mojolearn_runtime_classical_host.mojo
if [ -f "$R/$RT" ]; then
    NI=$(grep -c -E '^from[[:space:]]+(bindings|providers)\.[A-Za-z0-9_.]+[[:space:]]+import[[:space:]]+' "$R/$RT")
    bash "$S" --group _mojolearn_dev_runtime_classical_host --set host --root-ddmin "$RT" \
        --out "$T/qout" --work "$T/qwork" --dry-run > "$T/qplan.txt" 2>&1
    rc=$?
    [ "$rc" = 0 ] && ok "root dry run exits 0" || bad "root dry run rc=$rc: $(head -n 3 "$T/qplan.txt")"
    [ "$(grep -c '^PLAN ' "$T/qplan.txt")" = $((2 * NI + 2)) ] && ok "root plan: keep, none, $NI loo, $NI loi" || bad "root plan size"
    grep -q $'^PLAN loo-00\trevert=R:loo:0\tloo 00 from ' "$T/qplan.txt" && grep -q $'^PLAN loi-00\trevert=R:loi:0\t' "$T/qplan.txt" \
        && ok "loo-00 and loi-00 planned" || bad "loo/loi lines"
    grep -q -- '--stall-seconds 300 ' "$T/qplan.txt" && ok "root mode stalls at 300 s by default" || bad "root stall default"
    W="$R/tools/bisect_root_rewrite.py"
    if command -v python3 > /dev/null; then
        python3 "$W" list "$R/$RT" > "$T/imports.tsv"
        [ "$(wc -l < "$T/imports.tsv" | tr -d ' ')" = "$NI" ] && ok "rewriter and plan agree on $NI imports" || bad "rewriter import count"
        python3 "$W" rewrite "$R/$RT" --mode keep --out "$T/keep.mojo" > /dev/null && cmp -s "$R/$RT" "$T/keep.mojo" \
            && ok "keep re-emits the root unchanged" || bad "keep changed the root"
        i=0
        while IFS=$'\t' read -r _ names line; do  # every import: loo drops its line and every mention; loi keeps it
            python3 "$W" rewrite "$R/$RT" --mode loo --index "$i" --out "$T/loo.mojo" > /dev/null || bad "loo $i failed"
            grep -qxF "$line" "$T/loo.mojo" && bad "loo $i kept its import"
            for n in ${names//,/ }; do grep -q -E "(^|[^A-Za-z0-9_.])$n([^A-Za-z0-9_]|\$)" "$T/loo.mojo" && bad "loo $i still mentions $n"; done
            python3 "$W" rewrite "$R/$RT" --mode loi --index "$i" --out "$T/loi.mojo" > /dev/null || bad "loi $i failed"
            grep -qxF "$line" "$T/loi.mojo" || bad "loi $i lost its import"
            grep -q 'return module.finalize()' "$T/loi.mojo" || bad "loi $i lost the builder"
            i=$((i + 1))
        done < "$T/imports.tsv"
        ok "loo/loi checked for $i imports (failures listed above)"
        python3 "$W" rewrite "$R/$RT" --mode none --out "$T/none.mojo" > /dev/null
        [ "$(grep -c -E '^from[[:space:]]+(bindings|providers)\.' "$T/none.mojo")" = 0 ] && grep -q 'return module.finalize()' "$T/none.mojo" \
            && ok "none drops every candidate import and keeps the builder" || bad "none rewrite"
    fi
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
    bash "$S" --group _mojolearn_dev_runtime_classical_host --set host --history "$HD" --history-range 12:45 \
        --out "$T/rout" --work "$T/rwork" --dry-run > "$T/rplan.txt" 2>&1
    [ "$(grep -c '^PLAN h[0-9]' "$T/rplan.txt")" = 34 ] && ok "--history-range 12:45 selects 34 commits" || bad "range plan size"
    grep -q $'^PLAN h12\t' "$T/rplan.txt" && grep -q $'^PLAN h45\t' "$T/rplan.txt" && ! grep -q $'^PLAN h46\t' "$T/rplan.txt" \
        && ok "range is inclusive h12..h45" || bad "range ends"
    if [ "$(awk -F'\t' '$1 == "overlay" { print $2 }' "$HD/meta.tsv")" = patches ]; then
        grep -q 'patches .*(22)' "$T/rplan.txt" && ok "dry run names the 22 overlay patches" || bad "dry run patch count"
        miss=0; while IFS=$'\t' read -r s _ _ pf; do [ "$s" = seq ] || [ -s "$HD/patches/$pf" ] || miss=1; done < "$HD/patches/order.tsv"
        [ "$miss" = 0 ] && ok "every listed patch exists" || bad "a listed patch is missing"
    fi
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
