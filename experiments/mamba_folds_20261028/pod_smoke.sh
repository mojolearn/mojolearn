#!/bin/bash
# Copy of ~/mojolearn-evidence/integration-provider-queue-20261028/scripts/pod_smoke_5b.sh for lane mamba-folds: host .so
# not retained, no fit (owner 2026-10-28: the gate is compiles + host families through the cache + import + tier_dir).
set -u
OUT=${LEG_OUT:?}; mkdir -p "$OUT/logs" "$OUT/bin"
CORES=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))'); PAR=4; JOBS=$(( CORES / PAR ))
MOJO_VER=$(pixi run -e default mojo --version 2>&1 | tail -n 1)
{ echo "cores_affinity=$CORES"; echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"; echo "parallel=$PAR"; echo "jobs=$JOBS"; echo "mojo_version=$MOJO_VER"; } > "$OUT/resources.txt"
sh tokenizer/tools/gen_unicode_table.sh > "$OUT/logs/gen_unicode.log" 2>&1; echo "gen_unicode_rc=$?" >> "$OUT/resources.txt"
find . -name '*.mojo' -not -path './.pixi/*' -print0 | sort -z | xargs -0 sha256sum > "$OUT/source-manifest.sha256"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=$JOBS MOJOLEARN_COMPILE_JOBS=$JOBS MOJOLEARN_LINUX_CPU=x86-64-v3 OUT JOBS MOJOLEARN_BINCACHE_OUT="$OUT/bincache"
fam() { local f=$1 t0=$(date +%s); pixi run -e default python tools/bincache.py build bindings/build_${f}_host.sh > "$OUT/logs/host_$f.log" 2>&1; printf 'host\t%s\trc=%s\t%ss\n' "$f" "$?" "$(( $(date +%s)-t0 ))" >> "$OUT/results.tsv"; }
root() { local n=$1 e=$2 t0=$(date +%s); pixi run -e default mojo build --optimization-level 0 -j $JOBS --emit shared-lib --target-cpu x86-64-v3 -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_CPU -I . -I bindings "$e" -o "$OUT/bin/$n.so" > "$OUT/logs/root_$n.log" 2>&1; printf 'root\t%s\trc=%s\t%ss\n' "$n" "$?" "$(( $(date +%s)-t0 ))" >> "$OUT/results.tsv"; }
dev() { local n=$1 e=$2 a=$3 t0=$(date +%s) ol="--optimization-level 0" col=NVIDIA; case $a in gfx*) ol=""; col=AMD;; esac; pixi run -e default mojo build $ol -j $JOBS --emit shared-lib --target-cpu x86-64-v3 --target-accelerator $a -D MOJOLEARN_NUMERIC_IDENTICAL=1 -D MOJOLEARN_COLUMN_$col -I . -I bindings "$e" -o "$OUT/bin/$n.so" > "$OUT/logs/dev_$n.log" 2>&1; printf 'dev\t%s\trc=%s\t%ss\n' "$n" "$?" "$(( $(date +%s)-t0 ))" >> "$OUT/results.tsv"; }
export -f fam root dev
{ for f in runtime_classical tokenizer x_sequence trees gbdt forest $(ls bindings/build_*_host.sh | sed 's/.*build_//; s/_host.sh//' | grep -vxE 'runtime_classical|tokenizer|x_sequence|trees|gbdt|forest'); do echo "fam $f"; done; } > "$OUT/plan.txt"
xargs -P $PAR -L 1 bash -c '"$@"' _ < "$OUT/plan.txt"
ls -l python/mojolearn/host/ > "$OUT/host_dir.txt" 2>&1; sha256sum python/mojolearn/host/*.so > "$OUT/so_sha256.txt" 2>/dev/null
cat > "$OUT/smoke.py" <<'EOF'
import sys, importlib
import mojolearn
mods = ["mojolearn.ensemble", "mojolearn.neighbors", "mojolearn.decomposition", "mojolearn.svm", "mojolearn.cluster", "mojolearn.linear_model", "mojolearn.preprocessing", "mojolearn.metrics"]
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
EOF
smoke() { pixi run -e default python "$OUT/smoke.py" > "$OUT/smoke_$1.log" 2>&1; echo "smoke_$1 rc=$?" >> "$OUT/summary.txt"; }
smoke after
pixi run -e default python -c "import mojolearn._backend as b; print('tier_dir', b.tier_dir('identical'))" > "$OUT/probe_tier_dir.log" 2>&1; echo "probe_tier_dir rc=$?" >> "$OUT/summary.txt"
: # host .so not retained (size); hashes in so_sha256.txt
total=$(wc -l < "$OUT/plan.txt"); ok=$(grep -c 'rc=0' "$OUT/results.tsv"); echo "INTEG_BUILD total=$total ok=$ok" >> "$OUT/summary.txt"
grep -q 'smoke_after rc=0' "$OUT/summary.txt" && grep -q 'probe_tier_dir rc=0' "$OUT/summary.txt" && [ "$ok" = "$total" ]
