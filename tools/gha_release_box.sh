#!/bin/bash
# tools/gha_release_box.sh -- ONE JOB OF THE GITHUB LINUX RELEASE BUILD, run
# INSIDE the pinned image (the CPU build box's rocm/dev-ubuntu-22.04 digest)
# on a GitHub-hosted runner by .github/workflows/release-linux-build.yml.
# Never on the Mac, never on a rented box.
#
#   bash /tooling/tools/gha_release_box.sh shard    ARCH   # this runner's slice of one set
#   bash /tooling/tools/gha_release_box.sh assemble ARCH   # the whole set, from the slices
#
# WHAT IT IS. The box half of tools/runpod_cpu_leg.sh (pixi, the locked
# default env, the toolchain check, the portable math library) followed by
# exactly the command tools/release_linux_build.sh runs on its CPU pod: the
# route overlay recorded before and after, then
#   tools/release_linux_cpu_box.sh "$LEG_OUT/release" ARCH
# with MOJOLEARN_RELEASE_NO_DEVICE=1 under it, through the binding cache.
# So the tree, the provenance and every cache key are the CPU build box's.
#
# THE SPLIT. A set is ~76 builds (44 GPU bindings over three tiers, 32 host
# bindings). A `shard` job runs the same route with
# MOJOLEARN_BINCACHE_SHARD_ONLY naming its builds (tools/bincache.py returns at
# once for every other one), so each compiled binding lands, packed and keyed,
# in the box-local hot directory; the route then fails its completeness check,
# as it must for a partial tree, and this script judges the shard by its
# builds alone (shard.tsv). The `assemble` job gets every shard's hot
# directory and runs the unchanged route with no filter: each build is a
# verified hot-hit (or an R2 hit), so the tree it writes is the one a single
# box would have written, and a build no shard delivered compiles here.
#
# Mounts (the workflow's docker run): /root/mojolearn the source tree,
# /tooling the tooling checkout (read-only), /root/.mojolearn_bincache the URL
# map and hot directory, /root/leg_out the results.
# Environment: MOJOLEARN_COMMIT, GHA_BUILD_JOBS, GHA_SHARD_ONLY (shard).
set -u
MODE=${1:?shard or assemble}
ARCH=${2:?arch}
R=/root/mojolearn
T=/tooling
OUT=/root/leg_out
PIXIVER=0.77.0
case "$MODE" in shard|assemble) ;; *) echo "mode must be shard or assemble" >&2; exit 2 ;; esac
case "$ARCH" in sm_90a|sm_89) VENDOR=cuda ;; gfx942) VENDOR=hip ;; *) echo "unknown arch $ARCH" >&2; exit 2 ;; esac
[[ "${MOJOLEARN_COMMIT:-}" =~ ^[0-9a-f]{40}$ ]] || { echo "MOJOLEARN_COMMIT must be a full SHA" >&2; exit 2; }
JOBS=${GHA_BUILD_JOBS:-2}
[[ "$JOBS" =~ ^[1-9][0-9]?$ && "$JOBS" -le 16 ]] || { echo "GHA_BUILD_JOBS must be 1..16" >&2; exit 2; }
if [[ "$MODE" = shard && -z "${GHA_SHARD_ONLY:-}" ]]; then echo "a shard needs GHA_SHARD_ONLY" >&2; exit 2; fi
mkdir -p "$OUT/GHA"
G="$OUT/GHA"
ph() { printf '%s\t%s\n' "$1" "$(date +%s)" >> "$G/phases.tsv"; }
say() { echo "[$(date -u +%T) gha-box $MODE $ARCH] $*"; }
ph start

# ---------------------------------------------------------------- the box
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; head -4 /etc/os-release; free -g | head -2; df -h /root | tail -1
  echo "runner=${GHA_RUNNER:-} run=${GHA_RUN:-} job=${GHA_JOB:-}"; } > "$G/box.txt" 2>&1
# The RunPod CPU pod's start script installs curl (and sshd) when the image
# lacks it; xz extracts the pinned CUDA tools archive (build_sets.sh). Neither
# is part of the asserted toolchain (GCC 11.4, ld 2.38, glibc 2.35).
need=""
command -v curl > /dev/null || need="$need curl ca-certificates"
command -v xz > /dev/null || need="$need xz-utils"
if [[ -n "$need" ]]; then
    say "apt-get install$need"
    ( export DEBIAN_FRONTEND=noninteractive
      apt-get -o Acquire::Retries=3 update -qq && apt-get -o Acquire::Retries=3 install -y -qq --no-install-recommends $need ) \
        > "$G/apt.log" 2>&1 || { say "apt-get failed"; tail -5 "$G/apt.log"; exit 3; }
fi

# ------------------------------------------------ memory, sampled all along
# The cgroup of this container is the job's memory: memory.peak when the
# kernel keeps it, and the sampled maximum of memory.current and of host
# MemTotal - MemAvailable either way.
(
    max_cg=0; max_host=0
    while :; do
        cg=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
        used=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{print (t-a)*1024}' /proc/meminfo)
        (( cg > max_cg )) && max_cg=$cg
        (( used > max_host )) && max_host=$used
        printf 'cgroup_sampled_max_bytes=%s\nhost_used_sampled_max_bytes=%s\n' "$max_cg" "$max_host" > "$G/memory.sampled"
        sleep 2
    done
) &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

# ---------------------------------------------------------------- pixi
ph env_start
export PIXI_HOME=/root/.pixi PIXI_NO_PATH_UPDATE=1
PIXI=/root/.pixi/bin/pixi
for _try in 1 2 3; do
    [[ -x "$PIXI" ]] && break
    curl -fsSL --max-time 300 https://pixi.sh/install.sh | PIXI_VERSION="$PIXIVER" bash >> "$G/pixi_install.log" 2>&1
    [[ -x "$PIXI" ]] || sleep 10
done
[[ -x "$PIXI" ]] || { say "pixi did not install"; tail -5 "$G/pixi_install.log"; exit 3; }
export PATH="$R/.pixi/envs/default/bin:/root/.pixi/bin:$PATH"
cd "$R" || exit 3
ok=0
for _try in 1 2 3; do
    if "$PIXI" install --locked -e default > "$G/pixi_env_default.log" 2>&1; then ok=1; break; fi
    sleep 15
done
[[ "$ok" = 1 ]] || { say "pixi install --locked -e default failed"; tail -8 "$G/pixi_env_default.log"; exit 3; }
"$R/.pixi/envs/default/bin/mojo" --version >> "$G/box.txt" 2>&1 || { say "mojo does not run"; exit 3; }
ph env_end

# The host math library, as tools/runpod_cpu_leg.sh builds it on every CPU pod
# before its command (packaging/portable_math/stage.py's own recipe).
env PYTHONPATH="$R/packaging/portable_math" "$R/.pixi/envs/default/bin/python3" -c \
    "import pathlib, stage; stage.build(pathlib.Path('$R/python/mojolearn/.libs/libMojolearnMath.so'))" \
    > "$G/portable_math.log" 2>&1 || { say "portable math build failed"; tail -5 "$G/portable_math.log"; exit 3; }

# ---------------------------------------------------------------- the route overlay
# tools/release_linux_build.sh's list and record, byte for byte in format.
OVERLAY="tools/release061_remote_build.sh tools/cpu_build_guard.py tools/release_linux_cpu_box.sh tools/bincache.py"
mkdir -p "$OUT/release"
{ for f in $OVERLAY; do printf 'before %s %s\n' "$f" "$( [ -f "$f" ] && sha256sum "$f" | cut -c1-64 || echo absent)"; done; } > "$OUT/release/overlay.txt"
for f in $OVERLAY; do cp "$T/$f" "$R/$f" || exit 3; done
for f in $OVERLAY; do printf 'after %s %s\n' "$f" "$(sha256sum "$f" | cut -c1-64)"; done >> "$OUT/release/overlay.txt"

# ---------------------------------------------------------------- the command
ph run_start
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=$JOBS MOJOLEARN_LINUX_CPU=x86-64-v3
export MOJOLEARN_COMMIT PYTHONPATH="$R/python" LEG_OUT="$OUT"
export MOJOLEARN_BINCACHE_OUT="$OUT/bincache" MOJOLEARN_BINCACHE=1
unset MOJOLEARN_BINCACHE_SHARD_ONLY
[[ "$MODE" = shard ]] && export MOJOLEARN_BINCACHE_SHARD_ONLY="$GHA_SHARD_ONLY"
[[ -f /root/.mojolearn_bincache/urls.tsv ]] || { say "no URL map at /root/.mojolearn_bincache/urls.tsv"; exit 3; }
say "release_linux_cpu_box.sh, $JOBS jobs, map $(grep -c '^get' /root/.mojolearn_bincache/urls.tsv) gets / $(grep -c '^put' /root/.mojolearn_bincache/urls.tsv) slots, hot $(ls /root/.mojolearn_bincache/hot 2>/dev/null | wc -l)"
rc=0
bash tools/release_linux_cpu_box.sh "$OUT/release" "$ARCH" > "$G/route.log" 2>&1 || rc=$?
ph run_end
tail -40 "$G/route.log"
kill $SAMPLER 2>/dev/null
cat "$G/memory.sampled" > "$G/memory.txt" 2>/dev/null
echo "cgroup_peak_bytes=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)" >> "$G/memory.txt"
echo "mem_total_bytes=$(awk '/^MemTotal:/{print $2*1024}' /proc/meminfo)" >> "$G/memory.txt"
echo "nproc=$(nproc)" >> "$G/memory.txt"
df -B1 /root | tail -1 | awk '{print "disk_used_bytes=" $3 "\ndisk_free_bytes=" $4}' >> "$G/memory.txt"
RB="$OUT/release/$VENDOR-$ARCH/release-build"
grep -h '"guard"' "$RB/full46-build.log" 2>/dev/null > "$G/guard.jsonl"
cat "$G/memory.txt" "$G/guard.jsonl"

PROV="$RB/build/bincache/provenance.tsv"
# Every archive in the hot directory by its sha256: the Mac promotes an R2
# inbox upload only when the object there is exactly the archive this job
# built (tools/release_github_build.py), so a leaked slot URL cannot plant
# other bytes under a key.
for a in /root/.mojolearn_bincache/hot/*.tar.gz; do
    [[ -f "$a" ]] && printf '%s\t%s\n' "$(basename "$a" .tar.gz)" "$(sha256sum "$a" | cut -c1-64)"
done > "$G/hot_sha256.tsv"
if [[ "$MODE" = shard ]]; then
    # THE SHARD IS JUDGED BY ITS BUILDS: every listed build has a row whose
    # outcome placed or built its binary (the route's own exit is the partial
    # tree's refusal, expected). A build that died (a TIMEOUT, an OOM kill)
    # leaves no row and is named.
    "$R/.pixi/envs/default/bin/python3" - "$PROV" "$RB/build/bincache/keys" "$GHA_SHARD_ONLY" "$G/shard.tsv" <<'PY'
import json, pathlib, sys
prov, keys, only, out = sys.argv[1:]
want = only.split()
seen = {}
p = pathlib.Path(prov)
for line in (p.read_text().splitlines() if p.is_file() else []):
    c = line.split("\t")
    if len(c) != 6 or c[3] == "-":
        continue
    try:
        mode = json.loads((pathlib.Path(keys) / (c[3] + ".json")).read_text())["numeric_mode"]
    except (OSError, ValueError, KeyError):
        continue
    seen["%s:%s" % (mode, c[1].rsplit("/", 1)[-1])] = (c[2], c[4], c[3])
bad = 0
with open(out, "w") as fh:
    for w in want:
        outcome, secs, key = seen.get(w, ("NO-ROW (build killed or never ran)", "-", "-"))
        good = outcome in ("hit", "hot-hit") or (outcome.startswith("miss") and "+built" in outcome
                                                  and "build-failed" not in outcome)
        bad += not good
        fh.write("%s\t%s\t%s\t%s\t%s\n" % (w, "ok" if good else "FAILED", outcome, secs, key))
print(open(out).read(), end="")
sys.exit(1 if bad else 0)
PY
    src=$?
    say "shard: $(grep -c $'\tok\t' "$G/shard.tsv") of $(wc -l < "$G/shard.tsv") builds ok, $(wc -l < "$G/hot_sha256.tsv") archives in the hot directory"
    exit "$src"
fi
# assemble: the route's own verdict, plus how every binary was obtained
awk -F'\t' '{print $3}' "$PROV" 2>/dev/null | sort | uniq -c > "$G/outcomes.txt"
say "assemble: route exit $rc; outcomes:"; cat "$G/outcomes.txt"
exit "$rc"
