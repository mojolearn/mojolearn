#!/bin/bash
# consolidate-neural pod leg: compile the roots of one target set from roots.tsv (define off and on),
# and on the cpu set also every host family through the release binding cache with DECLARED outputs
# (tools/bincache.py:1180-1186), then `import mojolearn` + the tier_dir probe. Compile only; no runtime.
# Usage on the pod: SET=cpu|sm_89|gfx942 bash experiments/consolidate_neural_20261028/pod_roots.sh
set -u
OUT=${LEG_OUT:?}; SET=${SET:?}; mkdir -p "$OUT/logs" "$OUT/bin"
rm -f python/mojolearn/host/*.so
CORES=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))')
MOJO_VER=$(pixi run -e default mojo --version 2>&1 | tail -n 1)
git rev-parse HEAD > "$OUT/commit.txt" 2>/dev/null || true
{ echo "set=$SET"; echo "cores_affinity=$CORES"; echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"; echo "nproc=$(nproc)"; echo "mem=$(free -g | awk '/Mem/{print $2}')G"; echo "mojo_version=$MOJO_VER"; } > "$OUT/resources.txt"
sh tokenizer/tools/gen_unicode_table.sh > "$OUT/logs/gen_unicode.log" 2>&1; echo "gen_unicode_rc=$?" >> "$OUT/resources.txt"
PAR=4; JOBS=$(( CORES / PAR )); [ $JOBS -lt 1 ] && JOBS=1
echo "parallel=$PAR jobs=$JOBS" >> "$OUT/resources.txt"
export OUT JOBS SET
if [ "$SET" = cpu ]; then
  export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=$JOBS MOJOLEARN_COMPILE_JOBS=$JOBS MOJOLEARN_LINUX_CPU=x86-64-v3 MOJOLEARN_BINCACHE_OUT="$OUT/bincache"
  fam() { local f=$1 t0=$(date +%s); MOJOLEARN_BINCACHE_OUTPUTS="python/mojolearn/host/_mojolearn_${f}_host.so" pixi run -e default python tools/bincache.py build bindings/build_${f}_host.sh > "$OUT/logs/host_$f.log" 2>&1; printf 'FAMILY\t%s\trc=%s\t%ss\n' "$f" "$?" "$(( $(date +%s)-t0 ))" >> "$OUT/families.tsv"; }
  export -f fam
  # the two heaviest families first and alone-ish, so they never OOM next to each other
  { echo "fam runtime_classical"; echo "fam tokenizer"; for f in $(ls bindings/build_*_host.sh | sed 's/.*build_//; s/_host.sh//' | grep -vxE 'runtime_classical|tokenizer'); do echo "fam $f"; done; } > "$OUT/plan.txt"
  xargs -P $PAR -L 1 bash -c '"$@"' _ < "$OUT/plan.txt"
  sha256sum python/mojolearn/host/*.so > "$OUT/so_sha256.txt" 2>/dev/null
  total=$(wc -l < "$OUT/plan.txt"); ok=$(grep -c 'rc=0' "$OUT/families.tsv"); nso=$(ls python/mojolearn/host/*.so 2>/dev/null | wc -l)
  bad=$(grep -liE 'refus|destination exists|output already exists' "$OUT"/logs/host_*.log 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' ')
  echo "HOST_FAMILIES total=$total ok=$ok so_files=$nso refused_or_exists=[${bad}]" >> "$OUT/summary.txt"
  grep -h BINCACHE "$OUT"/logs/host_*.log 2>/dev/null | awk '{print $2}' | sort | uniq -c > "$OUT/bincache_rows.txt"
  cat > "$OUT/smoke.py" <<'PY'
import sys, importlib
import mojolearn
mods = ["mojolearn.ensemble", "mojolearn.neighbors", "mojolearn.decomposition", "mojolearn.svm", "mojolearn.cluster", "mojolearn.linear_model", "mojolearn.preprocessing", "mojolearn.metrics", "mojolearn.models"]
failed = []
for m in mods:
    try: importlib.import_module(m)
    except ModuleNotFoundError as e:
        if e.name == m: print("absent", m); continue
        failed.append((m, repr(e)))
    except Exception as e: failed.append((m, repr(e)))
missing = [n for n, v in vars(mojolearn).items() if type(v).__name__ == "_MissingUpperTier"]
print("missing_bindings", len(missing), missing[:40])
for f in failed: print("FAIL", f)
sys.exit(1 if failed else 0)
PY
  pixi run -e default python "$OUT/smoke.py" > "$OUT/smoke_import.log" 2>&1; echo "smoke_import rc=$?" >> "$OUT/summary.txt"
  pixi run -e default python -c "import mojolearn._backend as b; print('tier_dir', b.tier_dir('identical'))" > "$OUT/probe_tier_dir.log" 2>&1; echo "probe_tier_dir rc=$?" >> "$OUT/summary.txt"
fi
root() { # tag kind entry defines...
  local tag=$1 kind=$2 e=$3; shift 3; local t0=$(date +%s) tf ol="--optimization-level 0" emit="--emit shared-lib" o="$OUT/bin/$tag.so"
  case $SET in cpu) tf="-D MOJOLEARN_COLUMN_CPU";; sm_89) tf="--target-accelerator sm_89 -D MOJOLEARN_COLUMN_NVIDIA";; gfx942) tf="--target-accelerator gfx942 -D MOJOLEARN_COLUMN_AMD"; ol="";; esac
  [ "$kind" = exe ] && { emit=""; o="$OUT/bin/$tag.bin"; }
  pixi run -e default mojo build $ol -j $JOBS $emit --target-cpu x86-64-v3 $tf -D MOJOLEARN_NUMERIC_IDENTICAL=1 "$@" -I . -I bindings "$e" -o "$o" > "$OUT/logs/root_$tag.log" 2>&1
  local rc=$? sha=-; [ -f "$o" ] && sha=$(sha256sum "$o" | cut -d' ' -f1); rm -f "$o"
  printf 'ROOT\t%s\t%s\t%s\trc=%s\t%ss\tjobs=%s\topt=%s\tdefines=%s\tsha256=%s\n' "$SET" "$tag" "$e" "$rc" "$(( $(date +%s)-t0 ))" "$JOBS" "${ol:-default}" "$*" "$sha" >> "$OUT/roots.tsv"
}
export -f root
awk -F'\t' -v s="$SET" '$1==s {d=$5; printf "root %s %s %s %s\n", $2, $3, $4, d}' experiments/consolidate_neural_20261028/roots.tsv > "$OUT/root_plan.txt"
xargs -P $PAR -L 1 bash -c '"$@"' _ < "$OUT/root_plan.txt"
nr=$(wc -l < "$OUT/root_plan.txt"); okr=$(grep -c 'rc=0' "$OUT/roots.tsv" 2>/dev/null)
echo "ROOTS set=$SET total=$nr ok=$okr" >> "$OUT/summary.txt"
cat "$OUT/summary.txt"
