#!/bin/bash
# admit_identity_columns.sh [--build-host] [--check-only] [--lanes a,b] <records dir...>
# admit_identity_columns.sh [--build-host] --models <models dir>
#
# THE LOCAL HALF of a reference-table regeneration (docs/VERIFY.md
# "Regenerating the reference table"). Runs on the Mac in a checkout of the
# branch that will carry the new table; fits nothing but the self-test's one
# `ols/base` lane on the HOST column (MOJOLEARN_VENDOR=cpu), so no Metal.
#
# Column mode (records from tools/record_identity_column.sh on each box):
#   Incoming columns are retained under bench/results/identity_break before
#   generation. Commit the records and updated table together after checks.
#   1. every <records dir>/*.identical.json must pass
#      tools/identity_columns.py check (the same _verify_reference.admit() the
#      table builder applies, plus this harness's fixture bytes); all must name
#      ONE commit, commits with equal code, or (REUSE GOOD OLD RUNS, Andrew
#      2026-10-31) older commits whose every lane is superseded by a newer
#      record of its class or has an unchanged closure (identity_columns.py
#      reuse-check); every commit must be in this repository (its commit time
#      decides "newest wins")
#   2. retains byte-identical copies under bench/results/identity_break/,
#      reusing existing copies without requiring a separate prerequisite commit
#   3. regenerates python/mojolearn/verify_reference/table.json from every
#      repository column for the selected lanes (`verify --all --emit-reference`
#      with `--reference-table` and `--lanes`; newest commit wins per cell part
#      and device class, older classes kept as superseded)
#      INCREMENTAL RE-RECORD (Andrew 2026-10-10): re-record only what changed;
#      every other row is kept. Records of only the changed lanes
#      (record_identity_column.sh ONLY_CHANGED=table) are admitted here like a
#      full column: their cells win for those lanes (newest commit), and every
#      other lane keeps the rows its older committed columns already give it.
#      Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never
#      a reason to rerun. One sample per part is admissible (min_repeats=1);
#      incoming files may be partial vendor updates; selected numerical parts
#      must retain agreeing Apple, NVIDIA, AMD and cpu witnesses in the final
#      table. THE CPU COLUMN IS REQUIRED (Andrew 2026-10-24): a cpu value that
#      differs from the GPU reference, at any commit, or a missing cpu column
#      refuses the admission as a bug to fix; cpu N/A is accepted only where
#      the harness declares it (identity_break GPU_ONLY_LANES and
#      RECORD_EXCLUDED_PREFIXES, and the model part that loads the GPU's file).
#      Admission reuses identity evidence; it does not re-run the GPU checks.
#   4. tools/identity_columns.py report: routine-profile reference parts taken
#      from the new commit vs. still resting on an older one
#   5. `verify --self-test --cpu-threads 3` and `verify --coverage` against the
#      new table, from source, host column
#   6. THE PTX COLUMN (`nvidia-ptx-<gpu>-sm80` records, Andrew 2026-10-10): the
#      table's `ptx_divergent` must be empty, i.e. every PTX digest equals the
#      NVIDIA == AMD reference; a differing part fails the admission (PTX rc=1)
#   --check-only does 1, then 3-5 on a scratch table; nothing in the tree changes.
#   --lanes selects complete lanes; default is every lane in incoming records.
#   Unselected table rows and original records stay intact. Selected numerical
#   parts must have agreeing Apple, NVIDIA, AMD and cpu columns (retained
#   agreeing columns count; cpu N/A only where declared). Missing, conflicted
#   or cpu-disagreeing parts are refused before table mutation.
#   The final table is installed only after candidate checks pass.
#
# Models mode (the <outdir>/models of `record_identity_column.sh <gpu> <outdir>
# --models`, run on a GPU box AFTER the new table is committed and pushed):
#   replaces python/mojolearn/verify_reference/models/ with it, refusing a
#   manifest that drops a model the shipped bundle carries
#   (ALLOW_MODEL_DROP=1 overrides) or names a hash the table does not carry,
#   then reruns the self-test and coverage.
#
# Needs the host bindings `estimators` and `x_decomp` (ols and its TSQR), served
# by either their own files or the merged index, and the portable math library
# in this checkout; --build-host builds only missing bindings (through
# ~/mojolearn-evidence/compile_slot.sh when present, -j 1).
#
# Environment: PYTHON (default `pixi run -e default python`, else python3),
#   ADMIT_DATE (default today UTC), REF_TAG (directory suffix, default 0836), EVIDENCE (default
#   ~/mojolearn-evidence/ref-regen/<date>) for the long logs.
# Commits nothing; prints the git commands to run.
set -uo pipefail

die() { echo "admit_identity_columns: $*" >&2; exit 2; }
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT" || die "cannot enter $ROOT"

BUILD_HOST=0; CHECK_ONLY=0; MODELS=""; LANES=""
dirs=()
while [ $# -gt 0 ]; do
    case $1 in
        --build-host) BUILD_HOST=1 ;;
        --check-only) CHECK_ONLY=1 ;;
        --lanes) shift; LANES=${1:-}; [ -n "$LANES" ] || die "--lanes needs a comma-separated list" ;;
        --models) shift; MODELS=${1:-}; [ -n "$MODELS" ] || die "--models needs a directory" ;;
        -*) die "unknown option $1" ;;
        *) d=$(cd "$1" 2>/dev/null && pwd) || die "no such directory $1"; dirs+=("$d") ;;
    esac
    shift
done
[ -n "$MODELS" ] || [ ${#dirs[@]} -gt 0 ] || die "usage: admit_identity_columns.sh [--build-host] [--check-only] [--lanes a,b] <records dir...> | --models <dir>"

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
# Use the runtime's path-selection rules without bootstrapping the package.
# This accepts compact release and nested development files only when the
# actual indexed artifact exists, and preserves per-binding precedence.
"${PY[@]}" tools/identity_host_preflight.py --package python/mojolearn \
    --report "$EVID/host-bindings.json" "${HOST_FAMILIES[@]}" \
    > "$EVID/host-missing.txt" 2> "$EVID/host-preflight.log" \
    || die "host binding preflight failed; see $EVID/host-preflight.log"
while IFS= read -r fam; do
    [ -z "$fam" ] || missing+=("$fam")
done < "$EVID/host-missing.txt"
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
    # THE PTX COLUMN (Andrew 2026-10-10: PTX is a normal target; no flag). A
    # `nvidia-ptx-*` record is admitted like every column; IDENTICAL on PTX holds
    # when its digests equal the NVIDIA == AMD reference. Each differing part is
    # a PTX codegen bug to fix (docs/NVIDIA_PTX_IDENTITY.md), never a rerun and
    # never a separate qualification step: this admission refuses it.
    PTX_DIV=$("${PY[@]}" -c "import json,sys; print(len(json.load(open(sys.argv[1])).get('ptx_divergent') or []))" "$t")
    PTX_COLS=$("${PY[@]}" -c "import json,sys; print(sum(r.get('class') == 'ptx' for r in json.load(open(sys.argv[1]))['records']))" "$t")
    PTX_RC=0; [ "$PTX_DIV" = 0 ] || PTX_RC=1
    "${PY[@]}" -c "import json,sys; [print('  PTX-DIVERGENT', r['cell'], r['part'], 'ptx', r['ptx'], 'reference', r['reference']) for r in (json.load(open(sys.argv[1])).get('ptx_divergent') or [])[:10]]" "$t"
    echo "PTX rc=$PTX_RC ptx_records=$PTX_COLS divergent_parts=$PTX_DIV"
}

# =========================================================== models mode
if [ -n "$MODELS" ]; then
    [ -z "$LANES" ] || die "--lanes is only supported for column admission"
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
preview_args=()
if [ $CHECK_ONLY = 1 ]; then
    mkdir -p "$ROOT/bench/results/identity_break"
    PREVIEW_INPUTS=$(mktemp -d "$ROOT/bench/results/identity_break/.admission-preview-XXXXXX") \
        || die "cannot prepare temporary record inputs"
    trap 'rm -rf -- "$PREVIEW_INPUTS"' EXIT
    preview_args=(--preview-dir "$PREVIEW_INPUTS")
fi
"${PY[@]}" tools/identity_admission_guard.py inputs --source-root "$ROOT" \
    ${preview_args[@]+"${preview_args[@]}"} "${files[@]}" \
    > "$EVID/repository-records.txt" || die "could not retain incoming records in MojoLearn; table unchanged"
files=()
while IFS= read -r -d '' f; do files+=("$f"); done < "$EVID/repository-records.txt"
"${PY[@]}" tools/identity_columns.py check "${files[@]}" > "$EVID/check.txt" 2>&1
crc=$?
cut -c1-300 "$EVID/check.txt"
[ $crc = 0 ] || die "a record is not admissible (see above); nothing changed"
commits=$(sed -n 's/.* commit=\([0-9a-f]*\) .*/\1/p' "$EVID/check.txt" | sort -u)
# One commit, or several commits whose trees differ only outside the code (bench/results, docs, CHANGELOG):
# then the kernels the records exercised are the same bits and the columns are one witness pair
# (2026-10-12: the AMD column was recorded at a5c56140e and the NVIDIA column at 5b0dfc1ef, the board-ingest
# and CHANGELOG commits between them; re-recording a vendor for a docs commit is exactly the double work
# Andrew forbade on 2026-10-10). The admitted commit is the newest of them.
TMP_CODEDIFF=$(mktemp)
ncommits=$(wc -l <<< "$commits" | tr -d ' ')
for c in $commits; do git cat-file -e "$c^{commit}" 2>/dev/null || die "commit $c is not in this repository (git fetch origin first)"; done
C=$(for c in $commits; do echo "$(git log -1 --format=%ct $c) $c"; done | sort -n | tail -n 1 | cut -d' ' -f2)
if [ "$ncommits" != 1 ]; then
    first=$(echo $commits | cut -d' ' -f1)
    code_equal=1
    for c in $commits; do
        # tools/identity_code_equal.py: docs, boards, tools, the docs site, package metadata and docstring-only
        # Python changes cannot move a record's bits (2026-10-12: 80 docstring rewrites between the two columns).
        python3 tools/identity_code_equal.py "$first" "$c" > "$TMP_CODEDIFF" || { code_equal=0; break; }
    done
    if [ $code_equal = 1 ]; then
        echo "records name $ncommits commits with identical code ($(echo $commits)); admitting under the newest"
    else
        # REUSE GOOD OLD RUNS (Andrew 2026-10-31): reused older records next to
        # incremental records of the changed lanes. Every lane an older record
        # carries is superseded by a newer record of its class, or its closure
        # (tools/lane_select.py) is unchanged since that record's commit.
        "${PY[@]}" tools/identity_columns.py reuse-check --commit "$C" "${files[@]}" > "$EVID/reuse-check.txt" 2>&1
        rrc=$?
        grep -m 20 -E 'REUSE-' "$EVID/reuse-check.txt" | cut -c1-300
        [ $rrc = 0 ] || die "records name more than one commit and an older record's lane closure changed since its commit ($(head -n 3 "$TMP_CODEDIFF" | tr '\n' ' ')...); see $EVID/reuse-check.txt"
        echo "records name $ncommits commits; every older row is superseded or its lane closure is unchanged; admitting under the newest"
    fi
fi
git cat-file -e "$C^{commit}" 2>/dev/null || die "commit $C is not in this repository (git fetch origin first)"

# Freeze the base before building a candidate. Neither reporting nor coverage
# exits nonzero for conflicting references; validate the table explicitly.
# Scope to the incoming lanes by default, or an explicit complete-lane subset.
# Original records are always preserved whole, including held/conflicting lanes.
LANES=$("${PY[@]}" tools/identity_admission_guard.py lanes --lanes "$LANES" "${files[@]}") \
    || die "invalid admission scope"
tmp=$(mktemp -d "$EVID/admission.XXXXXX")
cp "$TABLE" "$tmp/base.json" || die "cannot retain base table"
record_args=(--records "$ROOT/bench/results/identity_break")
if [ $CHECK_ONLY = 1 ]; then
    for f in "${files[@]}"; do
        case "$f" in "$PREVIEW_INPUTS/"*) record_args+=(--records "$f") ;; esac
    done
fi
candidate() {
    cli --all --emit-reference "$tmp/table.json" --reference-table "$tmp/base.json" \
        --lanes "$LANES" "$@" > "$tmp/emit.log" 2>&1 \
        || die "emit-reference refused; table unchanged; see $tmp/emit.log"
    "${PY[@]}" tools/identity_admission_guard.py check --base "$tmp/base.json" \
        --candidate "$tmp/table.json" --lanes "$LANES" > "$tmp/agreement.json" \
        || die "GPU and cpu reference agreement refused; table unchanged; see $tmp/agreement.json"
}
candidate "${record_args[@]}"
if [ $CHECK_ONLY = 1 ]; then
    "${PY[@]}" tools/identity_columns.py report --table "$tmp/table.json" --commit "$C" --out "$tmp/report.json"
    checks "$tmp/table.json"
    echo "CHECK-ONLY done; scratch table $tmp/table.json (tree unchanged)"
    [ $ST_RC = 0 ] && [ $COV_RC = 0 ] && [ $PTX_RC = 0 ]
    exit $?
fi

# Admission receipts are not raw columns; keep them outside the input tree.
DEST=bench/results/reference_admissions/$DATE-${REF_TAG:-0836}
mkdir -p "$DEST"
{
    echo "# Reference columns at ${C:0:12} ($DATE)"
    echo
    echo "Columns recorded by \`tools/record_identity_column.sh\` (identity_break.py --json, identical tier,"
    echo "default fixture, one device, every part, every fixture, --repeats 1) and admitted by"
    echo "\`tools/admit_identity_columns.sh\`, which regenerated"
    echo "\`python/mojolearn/verify_reference/table.json\` for these lanes, preserving every other row:"
    echo "\`$LANES\`. Original source records are retained whole."
    echo "Source commit: \`$C\`."
    echo
    echo "| record | class / vendor / cells | sha256 |"
    echo "|---|---|---|"
} > "$DEST/README.md"
for f in "${files[@]}"; do
    line=$(grep -F "$f " "$EVID/check.txt" | sed 's/^IDCOLUMN [^ ]* admissible=yes //' | cut -d' ' -f1-2,7-)
    echo "| \`${f#"$ROOT/"}\` | $line | \`$(shasum -a 256 "$f" | cut -c1-16)\` |" >> "$DEST/README.md"
done

# The candidate already cites its final repository inputs; retain those exact
# bytes and commit them together with the checked table.
tail -n 1 "$tmp/emit.log" > "$DEST/admission-summary.json"
cp "$tmp/agreement.json" "$DEST/agreement-summary.json"
cut -c1-400 "$DEST/admission-summary.json"
"${PY[@]}" tools/identity_columns.py report --table "$tmp/table.json" --commit "$C" --out "$DEST/reference-report.json"
{
    echo
    echo "Builder summary: \`admission-summary.json\`; selected GPU and cpu agreement: \`agreement-summary.json\`."
    echo "Routine-profile reference parts from this"
    echo "commit vs. still older: \`reference-report.json\` (tools/identity_columns.py report)."
} >> "$DEST/README.md"
checks "$tmp/table.json"
[ $ST_RC = 0 ] && [ $COV_RC = 0 ] && [ $PTX_RC = 0 ] \
    || die "candidate checks failed; reference table unchanged"
"${PY[@]}" tools/identity_admission_guard.py publish --base "$tmp/base.json" \
    --candidate "$tmp/table.json" --lanes "$LANES" --destination "$TABLE" \
    > "$tmp/publication.json" || die "table publication refused; see $tmp/publication.json"
echo "NEXT (after SELFTEST rc=0):"
printf '  git add -- %q %q ' "$DEST" "$TABLE"
for f in "${files[@]}"; do
    printf '%q ' "${f#"$ROOT/"}"
    for prov in "${f%/*}"/*.provenance.txt "${f%/*}"/*.source-report.json.gz "${f%/*}"/*.verifier-run.json; do
        [ ! -f "$prov" ] || printf '%q ' "${prov#"$ROOT/"}"
    done
done
echo "&& git commit -m 'verify_reference: table regenerated from ${C:0:9} columns' && git push"
echo "  Re-record portable model files only if their admitted model/batch hashes changed."
echo "  Preserve every unaffected model in the replacement manifest; --models replaces the whole bundle."
[ $ST_RC = 0 ] && [ $COV_RC = 0 ] && [ $PTX_RC = 0 ]
