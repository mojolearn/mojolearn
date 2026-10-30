#!/bin/bash
# tools/gha_release_box.sh -- ONE JOB OF THE GITHUB LINUX RELEASE BUILD, run
# INSIDE the pinned image (the CPU build box's rocm/dev-ubuntu-22.04 digest)
# on a GitHub-hosted runner by .github/workflows/release-linux-build.yml.
# Never on the Mac, never on a rented box.
#
#   bash /tooling/tools/gha_release_box.sh shard    ARCH   # this runner's slice of one set
#   bash /tooling/tools/gha_release_box.sh assemble ARCH   # the whole set, from the slices
#   bash /tooling/tools/gha_release_box.sh host     host   # the host producer (release_host_build.py)
#   bash tooling/tools/gha_release_box.sh swap GIB RECORD  # ON THE RUNNER, before docker: add swap
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
# THE MERGED LAYOUT (2026-10-26). A merged shard is ONE compile (one
# `mojo build` of a directory set's root), the same compile the cpu box runs:
# same image, path, flags and compiler workers, so the same bytes and keys.
# When its recorded peak exceeds the runner's 15 GiB the workflow first runs
# `swap` here ON THE RUNNER (not in the container): a swapfile on the larger
# of /mnt and /, the size the plan gave (tools/release_github_build.py
# UNIT_PEAK_GIB), capped at that disk's free space less a reserve, recorded in
# RECORD. The container has no memory limit, so the compile pages to it.
#
# THE HOST PRODUCER (`host`): /root/mojolearn is a GIT CHECKOUT of the commit
# (the producer reads HEAD, a clean tracked tree and `git archive` of the
# donor's revision), /root/donor/build the donor runtime build. It runs the
# SOURCE's own tools/release_host_build.py build (no overlay: the producer
# refuses tools from outside the frozen checkout) with the default pixi env
# and the pinned patchelf 0.17.2, as the 0.8.58 box did
# (docs/plans/RETAINED_DEVICE_HOST_COMPOSITION_20261025.md), into
# /root/leg_out/host-build.
#
# THE CACHE WITHOUT R2. GHA_PRUNE_HOT=1 (no map_url: actions/cache restored
# the hot directory): after a shard, every archive no build of this shard used
# or made is removed, so the cache saves only current archives.
#
# Mounts (the workflow's docker run): /root/mojolearn the source tree,
# /tooling the tooling checkout (read-only), /root/.mojolearn_bincache the URL
# map and hot directory, /root/leg_out the results; host: /root/donor.
# Environment: MOJOLEARN_COMMIT, GHA_BUILD_JOBS, GHA_SHARD_ONLY (shard),
# GHA_RELEASE_MODES (MOJOLEARN_RELEASE_MODES for the route), GHA_PRUNE_HOT.
set -u

# swap GIB RECORD: on the runner, with sudo. The reserve keeps the build's own
# disk (tree, image, pixi env, outputs) free when the swap must share /.
swap_up() {
    local want=$1 rec=$2 dir=/ free m cap reserve=30 made f=/mojolearn-build.swap
    mkdir -p "$(dirname "$rec")"
    [[ "$want" =~ ^[0-9]+$ ]] || { echo "swap: GiB must be a number, got '$want'" >&2; return 2; }
    if (( want == 0 )); then { echo "requested_gib=0 made_gib=0"; free -g; } > "$rec"; return 0; fi
    free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
    if [[ -d /mnt ]]; then
        m=$(df -BG --output=avail /mnt | tail -1 | tr -dc 0-9)
        if (( m - 4 > free - reserve )); then dir=/mnt; free=$m; reserve=4; f=/mnt/mojolearn-build.swap; fi
    fi
    cap=$((free - reserve)); (( cap > 0 )) || cap=0
    made=$want; (( made <= cap )) || made=$cap
    if (( made > 0 )); then
        { sudo fallocate -l "${made}G" "$f" && sudo chmod 600 "$f" && sudo mkswap "$f" > /dev/null && sudo swapon "$f"; } \
            || { echo "swap: could not add ${made}G at $f" >&2; made=0; }
    fi
    { echo "requested_gib=$want made_gib=$made file=$f disk=$dir disk_free_gib=$free reserve_gib=$reserve"
      swapon --show; free -g; } | tee "$rec"
    (( made >= want )) || echo "::warning::swap: ${made} of ${want} GiB on $dir; the compile may run out of memory"
    return 0
}
MODE=${1:?shard, assemble, host, ptx-stage or swap}
if [[ "$MODE" = swap ]]; then swap_up "${2:?GiB}" "${3:?record file}"; exit $?; fi
ARCH=${2:?arch}
R=/root/mojolearn
T=/tooling
OUT=/root/leg_out
PIXIVER=0.77.0
case "$MODE" in shard|assemble|host|ptx-stage|development) ;; *) echo "mode must be shard, assemble, host, ptx-stage, development or swap" >&2; exit 2 ;; esac
case "$MODE:$ARCH" in
    host:host|development:host) VENDOR=none ;;
    host:*) echo "host mode takes the arch 'host'" >&2; exit 2 ;;
    *:sm_90a|*:sm_89|*:sm_80) VENDOR=cuda ;;
    *:gfx942|development:gfx950) VENDOR=hip ;;
    *) echo "unknown arch $ARCH" >&2; exit 2 ;;
esac
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
[[ "$MODE" = host || "$MODE" = ptx-stage || "$MODE" = development ]] && ! command -v git > /dev/null && need="$need git"
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
    max_cg=0; max_host=0; max_swap=0
    while :; do
        cg=$(cat /sys/fs/cgroup/memory.current 2>/dev/null || echo 0)
        # the RAM cgroup stops at the runner's 16 GB; what a merged compile pages out is here
        sw=$(cat /sys/fs/cgroup/memory.swap.current 2>/dev/null || echo 0)
        (( sw > max_swap )) && max_swap=$sw
        # %.0f, never print or %d: the image's mawk prints values above 2^31
        # in exponent form (2.16e+09, a bash arithmetic error every sample)
        # and clamps %d at 2147483647 (the runs of 2026-10-20 and 10-21).
        used=$(awk '/^MemTotal:/{t=$2} /^MemAvailable:/{a=$2} END{printf "%.0f\n", (t-a)*1024}' /proc/meminfo)
        (( cg > max_cg )) && max_cg=$cg
        (( used > max_host )) && max_host=$used
        printf 'cgroup_sampled_max_bytes=%s\nhost_used_sampled_max_bytes=%s\ncgroup_swap_sampled_max_bytes=%s\n' \
            "$max_cg" "$max_host" "$max_swap" > "$G/memory.sampled"
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

if [[ "$MODE" = ptx-stage ]]; then
    [[ "$ARCH" = sm_80 ]] || { say "PTX staging requires sm_80"; exit 3; }
    git config --global --add safe.directory "$R"
    [[ "$(git -C "$R" rev-parse HEAD)" = "$MOJOLEARN_COMMIT" ]] || { say "source checkout differs"; exit 3; }
    "$R/.pixi/envs/default/bin/python3" -m venv /root/release-tools > "$G/patchelf.log" 2>&1
    /root/release-tools/bin/python -m pip install -q --disable-pip-version-check --only-binary=:all: \
        --retries 2 --timeout 30 'patchelf==0.17.2.4' >> "$G/patchelf.log" 2>&1
    [[ $(/root/release-tools/bin/patchelf --version) = "patchelf 0.17.2" ]] || exit 3
    COMPILER_SHA=$(sha256sum "$R/.pixi/envs/default/bin/mojo" | cut -c1-64)
    rc=0
    # This branch precedes portable-math and device build setup deliberately:
    # every binary is retained; the producer invokes only packaging and audits.
    "$R/.pixi/envs/default/bin/python3" tools/release_ptx_assembly.py assemble \
        --root "$R" --shards /root/compiled-shards --host-build /root/host-build \
        --out "$OUT/ptx-assembly" --pixi-env default --compiler-sha256 "$COMPILER_SHA" \
        --patchelf /root/release-tools/bin/patchelf > "$G/ptx-assembly.log" 2>&1 || rc=$?
    printf 'exit=%s\nruntime_qualified=false\n' "$rc" > "$G/ptx-assembly.txt"
    kill $SAMPLER 2>/dev/null || true
    cat "$G/memory.sampled" > "$G/memory.txt" 2>/dev/null || true
    printf 'cgroup_peak_bytes=%s\n' "$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)" >> "$G/memory.txt"
    exit "$rc"
fi

if [[ "$MODE" = host ]]; then
    # ------------------------------------------------ the host producer
    # The checkout belongs to the runner's user; git refuses it to root without this.
    git config --global --add safe.directory "$R"
    [[ "$(git -C "$R" rev-parse HEAD)" = "$MOJOLEARN_COMMIT" ]] || { say "the checkout is not $MOJOLEARN_COMMIT"; exit 3; }
    [[ -f /root/donor/build/build-provenance.json ]] || { say "no donor build at /root/donor/build"; exit 3; }
    # patchelf 0.17.2.4, the stager every release route pins (release_linux_cpu_box.sh)
    "$R/.pixi/envs/default/bin/python3" -m venv /root/release-tools > "$G/patchelf.log" 2>&1 \
        && /root/release-tools/bin/python -m pip install -q --disable-pip-version-check --only-binary=:all: \
            --retries 2 --timeout 30 'patchelf==0.17.2.4' >> "$G/patchelf.log" 2>&1 \
        || { say "patchelf did not install"; tail -5 "$G/patchelf.log"; exit 3; }
    [[ $(/root/release-tools/bin/patchelf --version) = "patchelf 0.17.2" ]] || { say "patchelf 0.17.2 required"; exit 3; }
    COMPILER_SHA=$(sha256sum "$R/.pixi/envs/default/bin/mojo" | cut -c1-64)
    { echo "source_commit=$MOJOLEARN_COMMIT"
      echo "donor_commit=$("$R/.pixi/envs/default/bin/python3" -c 'import json; print(json.load(open("/root/donor/build/build-provenance.json"))["source_commit"])')"
      echo "compiler_sha256=$COMPILER_SHA"
      echo "producer_sha256=$(sha256sum "$R/tools/release_host_build.py" | cut -c1-64)"; } > "$G/host.txt"
    ph run_start
    rc=0
    reuse_host_args=()
    [[ "${GHA_REUSE_HOST_GROUPS:-false}" = true ]] && reuse_host_args+=(--reuse-unchanged-host)
    ( cd "$R" && env -u MOJOLEARN_COMMIT HOME=/root PATH="/root/.pixi/bin:$PATH" \
        "$R/.pixi/envs/default/bin/python3" tools/release_host_build.py build --root "$R" \
        --runtime-build /root/donor/build --output "$OUT/host-build" --pixi-env default \
        --patchelf /root/release-tools/bin/patchelf --compiler-sha256 "$COMPILER_SHA" "${reuse_host_args[@]}" ) > "$G/host.log" 2>&1 || rc=$?
    ph run_end
    echo "exit=$rc" >> "$G/host.txt"
    tail -5 "$G/host.log" >> "$G/host.txt"
    kill $SAMPLER 2>/dev/null
    cat "$G/memory.sampled" > "$G/memory.txt" 2>/dev/null
    echo "cgroup_peak_bytes=$(cat /sys/fs/cgroup/memory.peak 2>/dev/null || echo unavailable)" >> "$G/memory.txt"
    echo "cgroup_swap_peak_bytes=$(cat /sys/fs/cgroup/memory.swap.peak 2>/dev/null || echo unavailable)" >> "$G/memory.txt"
    cat "$G/host.txt"
    exit "$rc"
fi

# The host math library, as tools/runpod_cpu_leg.sh builds it on every CPU pod
# before its command (packaging/portable_math/stage.py's own recipe).
env PYTHONPATH="$R/packaging/portable_math" "$R/.pixi/envs/default/bin/python3" -c \
    "import pathlib, stage; stage.build(pathlib.Path('$R/python/mojolearn/.libs/libMojolearnMath.so'))" \
    > "$G/portable_math.log" 2>&1 || { say "portable math build failed"; tail -5 "$G/portable_math.log"; exit 3; }

# Explicit selected development groups use the source's existing builder.
# They do not enter the legacy release route or alter its worker/recipe policy.
if [[ "$MODE" = development ]]; then
    git config --global --add safe.directory "$R"
    "$R/.pixi/envs/default/bin/python3" "$T/tools/gha_development_group.py" \
        --root "$R" --commit "$MOJOLEARN_COMMIT" --arch "$ARCH" \
        --group "${GHA_DEVELOPMENT_GROUP:?selected physical group required}" \
        --out "$OUT/development"
    exit $?
fi

# ---------------------------------------------------------------- the route overlay
# tools/release_linux_build.sh's list and record, byte for byte in format.
OVERLAY="tools/release061_remote_build.sh tools/cpu_build_guard.py tools/release_linux_cpu_box.sh tools/bincache.py tools/release_modes.py"
mkdir -p "$OUT/release"
{ for f in $OVERLAY; do printf 'before %s %s\n' "$f" "$( [ -f "$f" ] && sha256sum "$f" | cut -c1-64 || echo absent)"; done; } > "$OUT/release/overlay.txt"
for f in $OVERLAY; do cp "$T/$f" "$R/$f" || exit 3; done
for f in $OVERLAY; do printf 'after %s %s\n' "$f" "$(sha256sum "$f" | cut -c1-64)"; done >> "$OUT/release/overlay.txt"

# ---------------------------------------------------------------- the command
ph run_start
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=$JOBS MOJOLEARN_LINUX_CPU=x86-64-v3
# The release modes choose the tiers, as tools/release_linux_build.sh exports
# them to the cpu box (MOJOLEARN_RELEASE_* is never a cache key field).
export MOJOLEARN_RELEASE_MODES="${GHA_RELEASE_MODES:-identical}"
# One merged compile on 4 vCPU, possibly paging: the merged bound from
# tools/release_limits.sh (not MOJOLEARN_*, so never a key field either).
. "$T/tools/release_limits.sh"
export RELEASE_MERGED_TIMEOUT_SECONDS="$GHA_MERGED_TIMEOUT_SECONDS"
export MOJOLEARN_COMMIT PYTHONPATH="$R/python" LEG_OUT="$OUT"
export MOJOLEARN_BINCACHE_OUT="$OUT/bincache" MOJOLEARN_BINCACHE=1
unset MOJOLEARN_BINCACHE_SHARD_ONLY
if [[ -n "${GHA_REUSE_SHARDS_RUN:-}" ]]; then
    [[ "$MODE" = assemble ]] || { say "retained shards are only for assembly"; exit 3; }
    export MOJOLEARN_BINCACHE_REQUIRE_HIT=1
    echo "reuse_shards_run=$GHA_REUSE_SHARDS_RUN" > "$G/reuse-shards.txt"
fi
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
    echo "cgroup_swap_peak_bytes=$(cat /sys/fs/cgroup/memory.swap.peak 2>/dev/null || echo unavailable)" >> "$G/memory.txt"
echo "mem_total_bytes=$(awk '/^MemTotal:/{printf "%.0f\n", $2*1024}' /proc/meminfo)" >> "$G/memory.txt"
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
        doc = json.loads((pathlib.Path(keys) / (c[3] + ".json")).read_text())
        mode = doc["numeric_mode"]
    except (OSError, ValueError, KeyError):
        continue
    name = c[1].rsplit("/", 1)[-1]
    args = list(doc.get("args") or [])
    entry = "%s:%s" % (mode, name)
    # tools/bincache.py shard_entry: a merged build is one group,
    # "<mode>:merged_build.sh:<stem>"; the merged host groups are "host:..."
    if name == "merged_build.sh":
        if args[:1] == ["host"]:
            mode = "host"
        entry = "%s:%s" % (mode, name)
        if "--group" in args[:-1]:
            entry += ":" + args[args.index("--group") + 1]
    seen[entry] = (c[2], c[4], c[3])
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
    if [[ "${GHA_PRUNE_HOT:-0}" = 1 ]]; then
        # keep only the archives this shard's builds used or made (shard.tsv's key column)
        keep=$(awk -F'\t' '$5 != "-" {print $5}' "$G/shard.tsv")
        for a in /root/.mojolearn_bincache/hot/*.tar.gz; do
            [[ -f "$a" ]] || continue
            k=$(basename "$a" .tar.gz)
            grep -qxF "$k" <<< "$keep" || { rm -f "$a"; echo "pruned $k" >> "$G/hot_pruned.txt"; }
        done
        for a in /root/.mojolearn_bincache/hot/*.tar.gz; do
            [[ -f "$a" ]] && printf '%s\t%s\n' "$(basename "$a" .tar.gz)" "$(sha256sum "$a" | cut -c1-64)"
        done > "$G/hot_sha256.tsv"
    fi
    say "shard: $(grep -c $'\tok\t' "$G/shard.tsv") of $(wc -l < "$G/shard.tsv") builds ok, $(wc -l < "$G/hot_sha256.tsv") archives in the hot directory"
    exit "$src"
fi
# assemble: the route's own verdict, plus how every binary was obtained
awk -F'\t' '{print $3}' "$PROV" 2>/dev/null | sort | uniq -c > "$G/outcomes.txt"
say "assemble: route exit $rc; outcomes:"; cat "$G/outcomes.txt"
exit "$rc"
