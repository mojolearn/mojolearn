#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
#
# DO PARALLEL AMD COMPILES CHANGE RESULTS, OR ONLY BYTES? (lane/amd-compile-speed, 2026-10-20)
#
# packaging/linux/build_sets.sh compiled every gfx* GPU set with one compiler
# worker because cold gfx942 builds at -j 2 gave different .so bytes run to
# run. The release contract is the output digest, not the bytes. This script,
# run ON the AMD box from a checkout at the branch head, answers:
#   1. results: does every build give the same identity_break digests?
#   2. bytes:   how many distinct binaries at -j 1 and at -j N, and of which
#               canonical class (tools/compare_gpu_builds.py: IDENTICAL,
#               REORDERED, SAME-KERNELS = layout only; CODE-DIFFERS = compiler);
#   3. speed:   wall time per build, and the merged identical tier at -j 1 vs -j N.
#
#   bash tools/amd_compile_variance.sh
#
# Steps, every build with a COLD Mojo cache (a fresh private MODULAR_CACHE_DIR
# and the default $MODULAR_HOME/cache/.mojo_cache the build scripts name,
# both wiped before each build):
#   a. bindings/build_mixture.sh (identical, gfx942, column amd) at -j 1 twice
#      and -j N three times, interleaved j1 jN j1 jN jN;
#   b. per build: sha256 of the .so, every embedded code object's sha256,
#      sorted kernel names, the canonical digests (one JSON line, manifests.jsonl);
#   c. per build: tools/identity_break.py --lanes gmm,gmm-random-init on the
#      GPU (--require-backend hip), a result digest over every cell's hashes,
#      and identity_break --diff against the first build;
#   d. per build: wall seconds;
#   e. bindings/merged_build.sh gpu (identical, MOJOLEARN_GPU_ARCHS=gfx942),
#      once at -j 1 and once at -j N, cold, into private output directories;
#   f. SUMMARY.txt: one line per build and one VERDICT line.
#
# Knobs (all optional): AMD_VARIANCE_OUT (/root/mb-out/amd-variance),
# AMD_VARIANCE_JOBS (8), AMD_VARIANCE_SCRIPT (build_mixture.sh),
# AMD_VARIANCE_SO (_mojolearn_mixture), AMD_VARIANCE_LANES (gmm,gmm-random-init),
# AMD_VARIANCE_SKIP_MERGED=1, AMD_VARIANCE_MERGED_TIMEOUT (7200 s per merged build),
# MOJOLEARN_BUILD_PIXI_ENV (default, as the release legs).
# The small text results are mirrored to /root/results-amd-variance so the
# box archiver (tools/box_archive.sh) brings them home.
#
# It changes nothing outside its output directory except the tree's own
# python/mojolearn/identical/<so>.so, which it restores at the end.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO" || exit 9
export PATH="$HOME/.pixi/bin:$PATH"
OUT="${AMD_VARIANCE_OUT:-/root/mb-out/amd-variance}"
NJ="${AMD_VARIANCE_JOBS:-8}"
SCRIPT="${AMD_VARIANCE_SCRIPT:-build_mixture.sh}"
SO="${AMD_VARIANCE_SO:-_mojolearn_mixture}"
LANES="${AMD_VARIANCE_LANES:-gmm,gmm-random-init}"
MERGED_TIMEOUT="${AMD_VARIANCE_MERGED_TIMEOUT:-7200}"
PIXI_ENV="${MOJOLEARN_BUILD_PIXI_ENV:-default}"
MIRROR=/root/results-amd-variance
[[ "$NJ" =~ ^[1-9][0-9]?$ ]] || { echo "AMD_VARIANCE_JOBS must be 1..99" >&2; exit 2; }

rm -rf "$OUT"
mkdir -p "$OUT/so" "$OUT/logs" "$OUT/verify" "$OUT/merged" "$OUT/diffs" || exit 9
SUMMARY="$OUT/SUMMARY.txt"
BUILDS="$OUT/builds.tsv"   # tag jobs rc seconds sha256 verify_rc result_digest diff_rc
: > "$BUILDS"
say() { echo "[$(date -u +%T) amd_variance] $*" | tee -a "$OUT/run.log"; }

# ------------------------------------------------------------------ facts
{
    echo "commit $(git rev-parse HEAD 2>/dev/null)"
    echo "dirty $(git status --porcelain --untracked-files=no 2>/dev/null | wc -l)"
    echo "host $(hostname) nproc $(nproc 2>/dev/null)"
    echo "gpu $(rocminfo 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')"
    echo "mojo $(pixi run -e "$PIXI_ENV" mojo --version 2>/dev/null | tail -1)"
    echo "pixi_env $PIXI_ENV jobs_n $NJ script $SCRIPT so $SO lanes $LANES"
} > "$OUT/facts.txt"
MODULAR_HOME_DEFAULT=$(pixi run -e "$PIXI_ENV" printenv MODULAR_HOME 2>/dev/null | tail -1)
echo "modular_home ${MODULAR_HOME_DEFAULT:-<unset>}" >> "$OUT/facts.txt"
OBJDUMP=""
for c in "${MOJOLEARN_LLVM_OBJDUMP:-}" /opt/rocm/llvm/bin/llvm-objdump "$(command -v llvm-objdump 2>/dev/null)"; do
    [[ -n "$c" && -x "$c" ]] && { OBJDUMP="$c"; break; }
done
echo "llvm_objdump ${OBJDUMP:-<none>}" >> "$OUT/facts.txt"
[[ -n "$OBJDUMP" ]] && export MOJOLEARN_LLVM_OBJDUMP="$OBJDUMP"
say "facts: $(tr '\n' ';' < "$OUT/facts.txt" | cut -c1-300)"

WORK="$OUT/work"
mkdir -p "$WORK"
# THE TREE'S OWN COPY of the binding is kept aside and put back at the end.
TREE_SO="python/mojolearn/identical/$SO.so"
[[ -e "$TREE_SO" ]] && cp -p "$TREE_SO" "$WORK/tree_so.orig"
restore() {
    if [[ -e "$WORK/tree_so.orig" ]]; then cp -p "$WORK/tree_so.orig" "$TREE_SO"; else rm -f "$TREE_SO"; fi
}
trap restore EXIT

cold_cache() {   # $1 = tag; sets MODULAR_CACHE_DIR to a fresh empty directory
    rm -rf "$WORK/cache-$1"
    mkdir -p "$WORK/cache-$1"
    export MODULAR_CACHE_DIR="$WORK/cache-$1"
    if [[ -n "$MODULAR_HOME_DEFAULT" && -d "$MODULAR_HOME_DEFAULT/cache/.mojo_cache" ]]; then
        rm -rf "$MODULAR_HOME_DEFAULT/cache/.mojo_cache"
    fi
}
cache_files() {  # $1 = tag: files the build left in its private cache (proves the variable was read)
    find "$WORK/cache-$1" -type f 2>/dev/null | wc -l | tr -d ' '
}
now_ms() { date +%s%3N; }

COMMON_ENV=(MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_GPU_ARCHS=gfx942 MOJOLEARN_TARGET_COLUMN=amd
            MOJOLEARN_SKIP_BUILD_GATE=1 MOJOLEARN_BINCACHE=0 MOJOLEARN_PACKAGE_BYTE_LM=1)

# ------------------------------------------------- a-d: the small binding
REF_JSON=""
n=0
for j in 1 "$NJ" 1 "$NJ" "$NJ"; do
    n=$((n + 1))
    tag="b${n}_j${j}"
    cold_cache "$tag"
    rm -f "$TREE_SO"
    t0=$(now_ms)
    env "${COMMON_ENV[@]}" MOJOLEARN_COMPILE_JOBS="$j" \
        pixi run -e "$PIXI_ENV" bash "bindings/$SCRIPT" > "$OUT/logs/$tag.build.log" 2>&1
    rc=$?
    secs=$(( ($(now_ms) - t0) ))
    secs=$(printf '%d.%03d' $((secs / 1000)) $((secs % 1000)))
    sha="-"; vrc="-"; rdig="-"; drc="-"
    if [[ "$rc" = 0 && -f "$TREE_SO" ]]; then
        cp -p "$TREE_SO" "$OUT/so/$tag.so"
        sha=$(sha256sum "$OUT/so/$tag.so" | cut -c1-64)
        python3 tools/compare_gpu_builds.py --manifest "$OUT/so/$tag.so" \
            | python3 -c 'import json,sys; m=json.loads(sys.stdin.read()); m["tag"]=sys.argv[1]; print(json.dumps(m, sort_keys=True))' "$tag" \
            >> "$OUT/manifests.jsonl"
        # c. the digest test on the GPU, with exactly this build in the tree
        env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$REPO/python" \
            timeout 1800 pixi run -e "$PIXI_ENV" python tools/identity_break.py --lanes "$LANES" \
            --require-backend hip --json "$OUT/verify/$tag.json" > "$OUT/logs/$tag.verify.log" 2>&1
        vrc=$?
        if [[ -s "$OUT/verify/$tag.json" ]]; then
            rdig=$(python3 - "$OUT/verify/$tag.json" <<'PY'
import hashlib, json, sys
cells = json.load(open(sys.argv[1])).get("cells", {})
keep = ("verdict", "hashes", "infer", "model", "batch", "rlpair")
rows = [f"{k} " + json.dumps({p: c.get(p) for p in keep}, sort_keys=True) for k, c in sorted(cells.items())]
print(f"{hashlib.sha256(chr(10).join(rows).encode()).hexdigest()[:16]}:{len(rows)}cells")
PY
)
            if [[ -z "$REF_JSON" ]]; then
                REF_JSON="$OUT/verify/$tag.json"; drc=ref
            else
                env MOJOLEARN_NUMERIC_MODE=identical PYTHONPATH="$REPO/python" \
                    pixi run -e "$PIXI_ENV" python tools/identity_break.py --diff "$REF_JSON" "$OUT/verify/$tag.json" \
                    > "$OUT/logs/$tag.diff.log" 2>&1
                drc=$?
            fi
        fi
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$tag" "$j" "$rc" "$secs" "$sha" "$vrc" "$rdig" "$drc" \
        "$(cache_files "$tag")" >> "$BUILDS"
    say "$tag -j $j build_rc=$rc ${secs}s sha=${sha:0:16} verify_rc=$vrc result=$rdig diff_rc=$drc cache_files=$(cache_files "$tag")"
    rm -rf "$WORK/cache-$tag"
done

# Every distinct pair of binaries, compared down to the kernel (disassembled when
# llvm-objdump is present). The first build of each distinct sha is its witness.
python3 - "$OUT" <<'PY' > "$OUT/pairs.tsv"
import itertools, json, sys
sys.path.insert(0, "tools")
import compare_gpu_builds as cgb
out = sys.argv[1]
ms = [json.loads(l) for l in open(f"{out}/manifests.jsonl")]
first = {}
for m in ms:
    first.setdefault(m["sha256"], m)
for a, b in itertools.combinations(first.values(), 2):
    print(f"{a['tag']}\t{b['tag']}\t{cgb.classify(a, b)}")
PY
while IFS=$'\t' read -r a b cls; do
    python3 tools/compare_gpu_builds.py --show 40 "$OUT/so/$a.so" "$OUT/so/$b.so" > "$OUT/diffs/${a}_vs_${b}.txt" 2>&1
done < "$OUT/pairs.tsv"

# ------------------------------------------------- e: the merged identical tier
MERGED="$OUT/merged.tsv"   # jobs rc seconds files cache_files
: > "$MERGED"
if [[ "${AMD_VARIANCE_SKIP_MERGED:-0}" != 1 ]]; then
    for j in 1 "$NJ"; do
        tag="merged_j$j"
        cold_cache "$tag"
        rm -rf "$OUT/merged/$tag"
        mkdir -p "$OUT/merged/$tag"
        t0=$(now_ms)
        env "${COMMON_ENV[@]}" MOJOLEARN_COMPILE_JOBS="$j" \
            timeout "$MERGED_TIMEOUT" pixi run -e "$PIXI_ENV" sh bindings/merged_build.sh gpu --outdir "$OUT/merged/$tag" \
            > "$OUT/logs/$tag.build.log" 2>&1
        rc=$?
        secs=$(( ($(now_ms) - t0) ))
        secs=$(printf '%d.%03d' $((secs / 1000)) $((secs % 1000)))
        files=$(find "$OUT/merged/$tag" -name '*.so' | wc -l | tr -d ' ')
        (cd "$OUT/merged/$tag" && sha256sum ./*.so) > "$OUT/merged/$tag.sha256" 2>/dev/null
        printf '%s\t%s\t%s\t%s\t%s\n' "$j" "$rc" "$secs" "$files" "$(cache_files "$tag")" >> "$MERGED"
        say "$tag build_rc=$rc ${secs}s files=$files cache_files=$(cache_files "$tag")"
        rm -rf "$WORK/cache-$tag"
    done
    if [[ -d "$OUT/merged/merged_j1" && -d "$OUT/merged/merged_j$NJ" ]]; then
        python3 tools/compare_gpu_builds.py --canonical --show 40 "$OUT/merged/merged_j1" "$OUT/merged/merged_j$NJ" \
            > "$OUT/diffs/merged_j1_vs_j$NJ.txt" 2>&1
        echo "$?" > "$OUT/diffs/merged_canonical_rc"
    fi
fi

# ------------------------------------------------- f: SUMMARY and VERDICT
python3 - "$OUT" "$NJ" <<'PY' > "$SUMMARY"
import json, os, sys
sys.path.insert(0, "tools")
import compare_gpu_builds as cgb
out, nj = sys.argv[1], sys.argv[2]
facts = open(f"{out}/facts.txt").read().split("\n")
print("# amd_compile_variance " + " | ".join(f for f in facts if f))
rows = [l.rstrip("\n").split("\t") for l in open(f"{out}/builds.tsv") if l.strip()]
ms = {}
if os.path.exists(f"{out}/manifests.jsonl"):
    ms = {m["tag"]: m for m in map(json.loads, open(f"{out}/manifests.jsonl"))}
ref = next((ms[r[0]] for r in rows if r[0] in ms), None)
for tag, j, rc, secs, sha, vrc, rdig, drc, cache in rows:
    m = ms.get(tag)
    cls = cgb.classify(ref, m) if (ref and m) else "-"
    extra = (f"code_objects={m['code_objects']} kernels={len(m['kernel_names'])} kernel_digest={m['kernel_digest'][:16]} "
             f"co_set={m['code_object_set'][:16]}" if m else "")
    print(f"BUILD {tag} jobs={j} build_rc={rc} seconds={secs} sha256={sha[:16]} class_vs_first={cls} "
          f"verify_rc={vrc} result={rdig} diff_vs_first_rc={drc} cold_cache_files={cache} {extra}")
merged = [l.rstrip("\n").split("\t") for l in open(f"{out}/merged.tsv") if l.strip()] if os.path.exists(f"{out}/merged.tsv") else []
for j, rc, secs, files, cache in merged:
    print(f"MERGED identical gfx942 jobs={j} build_rc={rc} seconds={secs} files={files} cold_cache_files={cache}")
mrc = open(f"{out}/diffs/merged_canonical_rc").read().strip() if os.path.exists(f"{out}/diffs/merged_canonical_rc") else "-"
mline = ""
if os.path.exists(f"{out}/diffs/merged_j1_vs_j{nj}.txt"):
    tail = [l for l in open(f"{out}/diffs/merged_j1_vs_j{nj}.txt") if l.startswith("#")]
    mline = tail[-1].strip("# \n") if tail else ""
pairs = [l.rstrip("\n").split("\t") for l in open(f"{out}/pairs.tsv") if l.strip()] if os.path.exists(f"{out}/pairs.tsv") else []
for a, b, cls in pairs:
    print(f"PAIR {a} {b} class={cls} detail=diffs/{a}_vs_{b}.txt")

built = [r for r in rows if r[2] == "0" and r[4] != "-"]
verified = [r for r in built if r[5] == "0" and r[6] != "-"]
complete = len(rows) == 5 and len(built) == 5 and len(verified) == 5
results = ("RESULTS-EQUAL" if complete and len({r[6] for r in verified}) == 1
           and all(r[7] in ("ref", "0") for r in verified)
           else "RESULTS-DIFFER" if len({r[6] for r in verified}) > 1 or any(r[7] not in ("ref", "0", "-") for r in verified)
           else "RESULTS-INCOMPLETE")
by_j = {}
for r in built:
    by_j.setdefault(r[1], []).append(r)
def distinct(j):
    rs = by_j.get(j, [])
    return f"{len({r[4] for r in rs})}/{len(rs)}"
def mean(j):
    rs = by_j.get(j, [])
    return f"{sum(float(r[3]) for r in rs) / len(rs):.1f}" if rs else "-"
classes = sorted({c for _, _, c in pairs}) or ["none"]
source = ("compiler" if "CODE-DIFFERS" in classes else
          "packaging" if set(classes) & {"REORDERED", "SAME-KERNELS"} else "none")
cold = all(r[8] not in ("0", "-") for r in rows)
msecs = {j: s for j, rc, s, *_ in merged if rc == "0"}
faster = mean(nj) != "-" and mean("1") != "-" and float(mean(nj)) < float(mean("1"))
rec = (f"AMD_COMPILE_JOBS_DEFAULT={nj}" if results == "RESULTS-EQUAL" and faster and cold
       and (not merged or mrc in ("0", "1")) else "keep AMD_COMPILE_JOBS_DEFAULT=1")
print(f"VERDICT results={results} bytes_j1={distinct('1')} bytes_j{nj}={distinct(nj)} pair_classes={','.join(classes)} "
      f"variance_source={source} mean_s_j1={mean('1')} mean_s_j{nj}={mean(nj)} "
      f"merged_s_j1={msecs.get('1', '-')} merged_s_j{nj}={msecs.get(nj, '-')} merged_canonical_rc={mrc} "
      f"merged=\"{mline}\" cold_cache_proven={'yes' if cold else 'NO'} recommend=\"{rec}\""
      + (" modular_ask=owed(docs/MODULAR_ASKS.md: gfx942 codegen differs run to run)" if source == "compiler" else ""))
PY
say "summary: $(tail -1 "$SUMMARY" | cut -c1-400)"

# Mirror the small text results where the box archiver looks.
rm -rf "$MIRROR"
mkdir -p "$MIRROR"
cp -p "$SUMMARY" "$OUT"/*.tsv "$OUT"/*.jsonl "$OUT"/facts.txt "$OUT"/run.log "$MIRROR"/ 2>/dev/null
cp -rp "$OUT/diffs" "$OUT/verify" "$MIRROR"/ 2>/dev/null
grep -q '^VERDICT ' "$SUMMARY" || exit 1
exit 0
