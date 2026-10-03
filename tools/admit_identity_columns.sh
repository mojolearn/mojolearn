#!/bin/bash
# admit_identity_columns.sh [--build-host] [--check-only] <records dir...>
# admit_identity_columns.sh [--build-host] --models <models dir>
#
# THE LOCAL HALF of a reference-table regeneration (docs/VERIFY.md
# "Regenerating the reference table"). Runs on the Mac in a checkout of the
# branch that will carry the new table; fits nothing but the self-test's one
# `ols/base` lane on the HOST column (MOJOLEARN_VENDOR=cpu), so no Metal.
#
# Column mode (records from tools/record_identity_column.sh on each box):
#   1. every <records dir>/*.identical.json must pass
#      tools/identity_columns.py check (the same _verify_reference.admit() the
#      table builder applies, plus this harness's fixture bytes); all must name
#      ONE commit, and that commit must be in this repository (its commit time
#      decides "newest wins")
#   2. copies them, with their provenance, to
#      bench/results/identity_break/<UTC date>-0836/<vendor label>/
#   3. regenerates python/mojolearn/verify_reference/table.json from every
#      committed column (`verify --all --emit-reference`; newest commit wins per
#      cell part and device class, older classes kept as superseded)
#   4. tools/identity_columns.py report: routine-profile reference parts taken
#      from the new commit vs. still resting on an older one
#   5. `verify --self-test --cpu-threads 3` and `verify --coverage` against the
#      new table, from source, host column
#   --check-only does 1, then 3-5 on a scratch table; nothing in the tree changes.
#
# Models mode (the <outdir>/models of `record_identity_column.sh <gpu> <outdir>
# --models`, run on a GPU box AFTER the new table is committed and pushed):
#   replaces python/mojolearn/verify_reference/models/ with it, refusing a
#   manifest that drops a model the shipped bundle carries
#   (ALLOW_MODEL_DROP=1 overrides) or names a hash the table does not carry,
#   then reruns the self-test and coverage.
#
# Needs the host bindings `estimators` and `x_decomp` (ols and its TSQR) and
# the portable math library in this checkout; --build-host builds exactly those (through
# ~/mojolearn-evidence/compile_slot.sh when present, -j 1).
#
# Environment: PYTHON (default `pixi run -e default python`, else python3),
#   ADMIT_DATE (default today UTC), EVIDENCE (default
#   ~/mojolearn-evidence/ref-regen/<date>) for the long logs.
# Commits nothing; prints the git commands to run.
set -uo pipefail

die() { echo "admit_identity_columns: $*" >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || die "cannot enter $ROOT"

BUILD_HOST=0; CHECK_ONLY=0; MODELS=""
dirs=()
while [ $# -gt 0 ]; do
    case $1 in
        --build-host) BUILD_HOST=1 ;;
        --check-only) CHECK_ONLY=1 ;;
        --models) shift; MODELS=${1:-}; [ -n "$MODELS" ] || die "--models needs a directory" ;;
        -*) die "unknown option $1" ;;
        *) d=$(cd "$1" 2>/dev/null && pwd) || die "no such directory $1"; dirs+=("$d") ;;
    esac
    shift
done
[ -n "$MODELS" ] || [ ${#dirs[@]} -gt 0 ] || die "usage: admit_identity_columns.sh [--build-host] [--check-only] <records dir...> | --models <dir>"

if [ -n "${PYTHON:-}" ]; then
    read -r -a PY <<< "$PYTHON"
elif command -v pixi > /dev/null 2>&1 && [ -f pixi.toml ]; then
    PY=(pixi run -e default python)
else
    PY=(python3)
fi
DATE=${ADMIT_DATE:-$(date -u +%F)}
EVID=${EVIDENCE:-$HOME/mojolearn-evidence/ref-regen/$DATE}
mkdir -p "$EVID"
TABLE=python/mojolearn/verify_reference/table.json
MODELDIR=python/mojolearn/verify_reference/models

# --- the host bindings the CLI import and the self-test need ---------------
if [ "$(uname -s)" = Darwin ]; then MATH=python/mojolearn/.dylibs/libMojolearnMath.dylib
else MATH=python/mojolearn/.libs/libMojolearnMath.so; fi
# ols on the host column: LinearRegression (estimators) + its TSQR (x_decomp)
HOST_FAMILIES=(estimators x_decomp)
missing=()
for fam in "${HOST_FAMILIES[@]}"; do
    [ -f "python/mojolearn/host/_mojolearn_${fam}_host.so" ] || missing+=("$fam")
done
if [ ${#missing[@]} -gt 0 ] || [ ! -f "$MATH" ]; then
    if [ $BUILD_HOST = 1 ]; then
        SLOT=(); [ -x "$HOME/mojolearn-evidence/compile_slot.sh" ] && SLOT=(bash "$HOME/mojolearn-evidence/compile_slot.sh")
        if [ ! -f "$MATH" ]; then
            PYTHONPATH="$ROOT/packaging/portable_math" "${PY[@]}" -c \
                "import pathlib, stage; stage.build(pathlib.Path('$ROOT/$MATH'))" > "$EVID/build-math.log" 2>&1 \
                || die "portable math build failed; see $EVID/build-math.log"
        fi
        for fam in "${missing[@]}"; do
            "${SLOT[@]}" env MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_COMPILE_JOBS=1 \
                sh "bindings/build_${fam}_host.sh" > "$EVID/build-${fam}-host.log" 2>&1
            rc=$?; [ $rc = 0 ] || { grep -m 5 -B 2 -A 8 error "$EVID/build-${fam}-host.log"; die "$fam host build rc=$rc"; }
        done
    else
        die "missing host bindings (${missing[*]}) or $MATH. Build them (or pass --build-host):
    PYTHONPATH=packaging/portable_math python -c \"import pathlib, stage; stage.build(pathlib.Path('$MATH'))\"
    for f in ${HOST_FAMILIES[*]}; do bash ~/mojolearn-evidence/compile_slot.sh env MOJOLEARN_TARGET_COLUMN=cpu MOJOLEARN_COMPILE_JOBS=1 sh bindings/build_\${f}_host.sh; done"
    fi
fi

cli() {  # the verifier from source, host column, identical tier
    env MOJOLEARN_VENDOR=cpu MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$ROOT/python" \
        "${PY[@]}" -m mojolearn verify "$@"
}

checks() {  # $1 = table; self-test then coverage; sets ST_RC, COV_RC
    local t=$1
    cli --self-test --cpu-threads 3 --reference-table "$t" > "$EVID/self-test.log" 2>&1
    ST_RC=$?
    grep -E '^  (untouched|perturbed)|^RESULT' "$EVID/self-test.log" | head -5
    echo "SELFTEST rc=$ST_RC table=$t (log $EVID/self-test.log)"
    cli --coverage --reference-table "$t" --json-out "$EVID/coverage.json" > "$EVID/coverage.log" 2>&1
    COV_RC=$?
    grep -m 3 -iE 'admission|legacy|reference' "$EVID/coverage.log" | cut -c1-200
    echo "COVERAGE rc=$COV_RC (report $EVID/coverage.json)"
}

# =========================================================== models mode
if [ -n "$MODELS" ]; then
    [ -f "$MODELS/models.json" ] || die "$MODELS/models.json missing"
    "${PY[@]}" - "$MODELS" "$MODELDIR" "$TABLE" "${ALLOW_MODEL_DROP:-0}" <<'EOF' || die "models refused"
import json, sys, os, hashlib
new_dir, old_dir, table_path, allow_drop = sys.argv[1:5]
new = json.load(open(os.path.join(new_dir, "models.json")))
old = json.load(open(os.path.join(old_dir, "models.json")))
table = json.load(open(table_path))
pairs = lambda m: {(x["lane"], x["fixture"]) for x in m["models"]}
dropped = sorted(pairs(old) - pairs(new))
if dropped and allow_drop != "1":
    sys.exit(f"REFUSED: the new manifest drops {len(dropped)} shipped models, e.g. {dropped[:4]}")
bad = []
for m in new["models"]:
    if not os.path.isfile(os.path.join(new_dir, m["file"])):
        bad.append(f"{m['file']}: missing")
        continue
    ent = (table["cells"].get(f"{m['lane']}/{m['fixture']}") or {}).get("model") or {}
    if ent.get("ref") != m["model_hash"]:
        bad.append(f"{m['file']}: hash {m['model_hash']} vs table {ent.get('ref')}")
if bad:
    sys.exit("REFUSED: " + "; ".join(bad[:6]))
print(f"MODELS ok: {len(new['models'])} models, dropped {len(dropped)}")
EOF
    git rm -q -r "$MODELDIR" > /dev/null 2>&1 || rm -rf "$MODELDIR"
    mkdir -p "$MODELDIR"
    "${PY[@]}" -c "
import json, shutil, sys, os
src, dst = sys.argv[1:3]
m = json.load(open(os.path.join(src, 'models.json')))
for x in m['models']:
    shutil.copyfile(os.path.join(src, x['file']), os.path.join(dst, x['file']))
shutil.copyfile(os.path.join(src, 'models.json'), os.path.join(dst, 'models.json'))
" "$MODELS" "$MODELDIR" || die "copy failed"
    git add -A "$MODELDIR"
    checks "$TABLE"
    echo "NEXT: git commit -m 'verify_reference/models: regenerated against the 0.8.36 table' -- $MODELDIR && git push"
    [ $ST_RC = 0 ] && [ $COV_RC = 0 ]
    exit $?
fi

# =========================================================== column mode
files=()
for d in "${dirs[@]}"; do
    for f in "$d"/*.identical.json; do [ -f "$f" ] && files+=("$f"); done
done
[ ${#files[@]} -gt 0 ] || die "no *.identical.json under ${dirs[*]}"
"${PY[@]}" tools/identity_columns.py check "${files[@]}" > "$EVID/check.txt" 2>&1
crc=$?
cut -c1-300 "$EVID/check.txt"
[ $crc = 0 ] || die "a record is not admissible (see above); nothing changed"
commits=$(sed -n 's/.* commit=\([0-9a-f]*\) .*/\1/p' "$EVID/check.txt" | sort -u)
[ "$(wc -l <<< "$commits" | tr -d ' ')" = 1 ] || die "records name more than one commit: $(echo $commits)"
C=$("${PY[@]}" -c "import json,sys; print(json.load(open(sys.argv[1]))['commit'])" "${files[0]}")
git cat-file -e "$C^{commit}" 2>/dev/null || die "commit $C is not in this repository (git fetch origin first)"

if [ $CHECK_ONLY = 1 ]; then
    tmp=$(mktemp -d "$EVID/check-only.XXXXXX")
    mkdir -p "$tmp/records"; cp "${files[@]}" "$tmp/records/"
    cli --all --emit-reference "$tmp/table.json" \
        --records "$ROOT/bench/results/identity_break" --records "$tmp/records" > "$tmp/emit.log" 2>&1 \
        || die "emit-reference failed; see $tmp/emit.log"
    tail -n 1 "$tmp/emit.log" | cut -c1-400
    "${PY[@]}" tools/identity_columns.py report --table "$tmp/table.json" --commit "$C" --out "$tmp/report.json"
    checks "$tmp/table.json"
    echo "CHECK-ONLY done; scratch table $tmp/table.json (tree unchanged)"
    [ $ST_RC = 0 ] && [ $COV_RC = 0 ]
    exit $?
fi

DEST=bench/results/identity_break/$DATE-0836
mkdir -p "$DEST"
{
    echo "# Reference columns at ${C:0:12} ($DATE)"
    echo
    echo "Written by \`tools/record_identity_column.sh\` (identity_break.py --json, identical tier,"
    echo "default fixture, one device, every part, every fixture, --repeats 2) and admitted by"
    echo "\`tools/admit_identity_columns.sh\`, which regenerated"
    echo "\`python/mojolearn/verify_reference/table.json\` from every committed column."
    echo "Source commit: \`$C\`."
    echo
    echo "| record | class / vendor / cells | sha256 |"
    echo "|---|---|---|"
} > "$DEST/README.md"
for f in "${files[@]}"; do
    label=$("${PY[@]}" -c "import json,sys; print(json.load(open(sys.argv[1]))['vendor'])" "$f")
    sub="$DEST/$label"; mkdir -p "$sub"
    base=$(basename "$f")
    if [ -f "$sub/$base" ] && ! cmp -s "$f" "$sub/$base"; then die "$sub/$base exists with other bytes"; fi
    cp "$f" "$sub/$base"
    prov="${f%.identical.json}.provenance.txt"; [ -f "$prov" ] && cp "$prov" "$sub/"
    line=$(grep -F "$f " "$EVID/check.txt" | sed 's/^IDCOLUMN [^ ]* admissible=yes //' | cut -d' ' -f1-2,7-)
    echo "| \`$label/$base\` | $line | \`$(shasum -a 256 "$f" | cut -c1-16)\` |" >> "$DEST/README.md"
done

cli --all --emit-reference "$TABLE" > "$EVID/emit.log" 2>&1 || die "emit-reference failed; see $EVID/emit.log"
tail -n 1 "$EVID/emit.log" > "$DEST/admission-summary.json"
cut -c1-400 "$DEST/admission-summary.json"
grep -F "$DEST" "$EVID/emit.log" | grep -c '^use ' | sed 's/^/new records used by the builder: /'
"${PY[@]}" tools/identity_columns.py report --table "$TABLE" --commit "$C" --out "$DEST/reference-report.json"
{
    echo
    echo "Builder summary: \`admission-summary.json\`. Routine-profile reference parts from this"
    echo "commit vs. still older: \`reference-report.json\` (tools/identity_columns.py report)."
} >> "$DEST/README.md"
checks "$TABLE"
echo "NEXT (after SELFTEST rc=0):"
echo "  git add $DEST $TABLE && git commit -m 'verify_reference: table regenerated from ${C:0:9} columns' && git push"
echo "  then on one GPU box: tools/record_identity_column.sh <gpu-label> <outdir> --models  (this pushed head)"
echo "  then here: tools/admit_identity_columns.sh --models <outdir>/models"
[ $ST_RC = 0 ] && [ $COV_RC = 0 ]
