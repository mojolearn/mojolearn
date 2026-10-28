# Box command (attempt 3) for tools/runpod_cpu_leg.sh with the binding cache ON.
# Host bindings follow the integrator's recipe
# (~/mojolearn-evidence/integration-provider-queue-20261028/scripts/pod_smoke_5b.sh):
# bindings/build_<family>_host.sh through tools/bincache.py, 6 in parallel, all
# allocated cores split among them; a family already restored is kept. Then the
# import check and the probe. Everything lands in $LEG_OUT.
set -u
T0=$(date +%s)
O=${LEG_OUT:?}; mkdir -p "$O/logs"
CORES=$(python3 -c 'import os;print(len(os.sched_getaffinity(0)))'); PAR=6; JOBS=$(( CORES / PAR ))
{ echo "cores_affinity=$CORES"; echo "cgroup_cpu_max=$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)"; echo "parallel=$PAR"; echo "jobs=$JOBS";
  echo "mojo_version=$(pixi run -e default mojo --version 2>&1 | tail -n 1)"; } > "$O/resources.txt"
sh tokenizer/tools/gen_unicode_table.sh > "$O/logs/gen_unicode.log" 2>&1; echo "gen_unicode_rc=$?" >> "$O/resources.txt"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=$JOBS MOJOLEARN_COMPILE_JOBS=$JOBS MOJOLEARN_LINUX_CPU=x86-64-v3 O MOJOLEARN_BINCACHE_OUT="$O/bincache"
fam() { local f=$1 t0=$(date +%s)
  if [ -e "python/mojolearn/host/_mojolearn_${f}_host.so" ]; then printf 'host\t%s\tpresent\t0s\n' "$f" >> "$O/results.tsv"; return; fi
  pixi run -e default python tools/bincache.py build bindings/build_${f}_host.sh > "$O/logs/host_$f.log" 2>&1
  printf 'host\t%s\trc=%s\t%ss\n' "$f" "$?" "$(( $(date +%s)-t0 ))" >> "$O/results.tsv"; }
export -f fam
ls bindings/build_*_host.sh | sed 's/.*build_//; s/_host.sh//' | sort > "$O/families.txt"
# Largest families first (from the integrator's timings).
{ for f in runtime_classical tokenizer x_sequence trees gbdt forest; do echo "fam $f"; done
  grep -vxE 'runtime_classical|tokenizer|x_sequence|trees|gbdt|forest' "$O/families.txt" | sed 's/^/fam /'; } > "$O/plan.txt"
timeout -k 30 1800 xargs -P $PAR -L 1 bash -c '"$@"' _ < "$O/plan.txt"
echo "build_rc=$? seconds=$(( $(date +%s) - T0 ))" >> "$O/resources.txt"
sha256sum python/mojolearn/host/*.so > "$O/binary_manifest.sha256" 2>/dev/null
for f in $(cat "$O/families.txt"); do [ -e "python/mojolearn/host/_mojolearn_${f}_host.so" ] || echo "$f"; done > "$O/families_missing.txt"
echo "families=$(wc -l < "$O/families.txt") missing=$(wc -l < "$O/families_missing.txt")" >> "$O/resources.txt"
MOJOLEARN_VENDOR=cpu pixi run -e default python -c "import mojolearn,mojolearn._backend as b;print('vendor',mojolearn.vendor());print('missing',len(b._MISSING),sorted(b._MISSING)[:60])" > "$O/import_check.log" 2>&1
echo "import_rc=$?" >> "$O/resources.txt"
LEFT=$(( 52 * 60 - ( $(date +%s) - T0 ) )); DEADLINE=$(( LEFT - 330 )); [ "$DEADLINE" -ge 60 ] || DEADLINE=60
echo "probe_deadline=$DEADLINE" >> "$O/resources.txt"
MOJOLEARN_VENDOR=cpu pixi run -e default python tools/python_runtime_probe.py --out "$O/probe" --workers "$CORES" \
    --group-timeout 300 --call-timeout 60 --deadline-seconds "$DEADLINE" > "$O/probe.log" 2>&1
echo "probe_rc=$? total_seconds=$(( $(date +%s) - T0 ))" >> "$O/resources.txt"
tar -C "$O" -czf "$O/probe.tgz" probe 2>/dev/null
exit 0
