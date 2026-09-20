#!/bin/bash
# tools/runpod_cpu_leg.sh -- ONE RunPod CPU pod (no GPU) per lane, for heavy
# CPU work that should not run on the shared Mac: host binding builds, the
# identity_break CPU column, sabotage builds, pytest modules. 2026-09-15.
#
#   bash tools/runpod_cpu_leg.sh --lane NAME --cmd 'COMMAND' [options]   # DRY RUN
#   bash tools/runpod_cpu_leg.sh --lane NAME --cmd 'COMMAND' ... --rent  # creates
#   bash tools/runpod_cpu_leg.sh list          mojolearn-cpu-* pods on the account
#   bash tools/runpod_cpu_leg.sh reap POD_ID   DELETE one mojolearn-cpu pod, verify
#
# Options:
#   --lane NAME          lane tag, [a-z0-9-]; the pod is mojolearn-cpu-NAME-<stamp>.
#                        REFUSES if a pod with this tag is already live.
#   --cmd 'COMMAND'      run with bash inside the unpacked repo on the pod;
#   --cmd-file FILE      or the command read from a file. $LEG_OUT is the
#                        results directory that comes back to the Mac.
#   --worktree DIR       whose TRACKED files ship (default: this checkout).
#                        Uncommitted edits to tracked files ship too; the
#                        commit and the dirty count are recorded.
#   --include PATH       also ship this tracked path (repeatable). bench/results,
#                        mamba/corpus and bench/oracle_* are left out by default.
#   --build a,b          host families built BEFORE the command, through the
#                        R2 binding cache: bindings/build_<family>_host.sh
#   --sabotage-build a,b the same families built with --sabotage-defines into
#                        python/mojolearn/host-sabotage (the negative control,
#                        cached under its own key namespace)
#   --sabotage-defines S default "-D MOJOLEARN_HOST_SABOTAGE=1"
#   --envs a,b           pixi environments installed (default: default)
#   --vcpu N             vCPUs (default 8)       --flavors cpu3c,cpu5c
#   --lease MIN          on-pod self-delete after MIN minutes (default 60)
#   --jobs N             MOJOLEARN_BUILD_JOBS for --build (default 8; keyed)
#   --image IMG          default runpod/base:1.3.1-ubuntu2204 (keyed)
#   --disk GB            container disk (default 40)
#   --out DIR            results (default <worktree>/bench/results/runpod_cpu/<stamp>-<lane>)
#   --no-bincache        build from source
#   --envcache           restore and upload .pixi/envs from R2 (default OFF: measured
#                        slower than a locked install on RunPod, 37 s against 12 s)
#   --max-pods N         refuse when N mojolearn-cpu pods are live (default 8, MOJOLEARN_RUNPOD_CPU_MAX_PODS)
#   --stage 'KEYS'       R2 dataset keys staged onto the box after the source
#                        (tools/stage_from_r2.sh, MOJOLEARN_STAGE_STRICT=1: a
#                        staging failure stops the leg before anything runs), e.g.
#                        'corpus/enwik8/input.txt corpus/pile_github/input.txt',
#                        which land at training/corpus/<name>/input.txt.
#                        Log: $OUT/stage_r2.log. Default: nothing staged.
#   --rent               actually create the pod. Without it: a dry run that
#                        creates nothing and costs nothing.
#
# WHAT A RENTED RUN DOES, IN ORDER. A Mac dead-man is armed BEFORE the create
# (it deletes the pod by id, or by name, after lease + ready timeout + 10 min).
# Create, printing the cost per hour. Wait for ssh. Arm the ON-POD watchdog
# (tools/runpod_guard.sh arm: the pod DELETEs itself through the API at the
# lease) and read it back: its pid is alive and its token answers a GET with
# 200. Ship the tracked source as a tarball (sha256 checked on both ends).
# Hand the box presigned R2 URLs over ssh stdin (never argv, never the
# credential). On the box: restore or install the pixi binary and envs, build
# the host bindings through tools/bincache.py, run the command, then upload
# whatever was a cache miss. Fetch $LEG_OUT, promote binding uploads, DELETE
# the pod and ask the API until it is gone. The dead-man is cancelled only
# after that verified delete. timings.tsv splits create, arm, stage, env,
# build, run, fetch and teardown.
#
# CACHES (docs/RUNPOD_CPU_LEG.md): the pixi binary and each pixi env
# (tools/runpod_cpu_cache.py), and the host bindings (tools/bincache.py,
# partition none/<image>; production under bincache/v1, sabotage under
# bincache/sabotage-v1). The Mac never uses any of them: nothing on the Mac
# stages a URL map, so tools/bincache.py stays a pass-through there.
#
# Keys: ~/.mojolearn_runpod_key (or MOJOLEARN_RUNPOD_KEY_FILE), ~/.mojolearn_r2.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
RP=https://rest.runpod.io/v1
RP_V2=https://api.runpod.io/v2
BOX_REPO=/root/mojolearn
PIXI_VERSION=0.77.0
READY_TIMEOUT=600
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=30 -o BatchMode=yes"

LANE=""; CMD=""; CMD_FILE=""; WORKTREE=""; INCLUDES=""; BUILD=""; SAB_BUILD=""
SAB_DEFINES="-D MOJOLEARN_HOST_SABOTAGE=1"; ENVS="default"; VCPU=8; FLAVORS="cpu3c,cpu5c"
LEASE=60; JOBS=8; IMAGE="runpod/base:1.3.1-ubuntu2204"; DISK=40; OUT=""; BINCACHE=1; ENVCACHE=0
STAGE_KEYS=""
MAX_PODS=${MOJOLEARN_RUNPOD_CPU_MAX_PODS:-8}; RENT=0

say() { printf '[%s cpu-leg] %s\n' "$(date +%T)" "$*"; }
die() { printf '\nREFUSED: %s\n' "$*" >&2; exit 1; }
now() { date +%s; }

TMPD=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-cpu-leg.XXXXXX")
CURLRC="$TMPD/rp.curlrc"
POD_ID=""; POD_TERMINATED=0; DEADMAN_PID=""; DEADMAN_DIR=""; SSH_TARGET=""; COST_HR=""
T_POST=""; T_SSH=""; T_ARMED=""; T_STAGED=""; T_DONE=""; T_FETCHED=""; T_DEL0=""; T_DEL1=""

# ---------------------------------------------------------------------------
# the API, with the key kept out of every argv (tools/gemm_remote_leg.sh's rule)
# ---------------------------------------------------------------------------
load_key() {
    _kf="${MOJOLEARN_RUNPOD_KEY_FILE:-$HOME/.mojolearn_runpod_key}"
    if [ -z "${RUNPOD_API_KEY:-}" ] && [ -f "$_kf" ]; then
        _perm=$(stat -f '%OLp' "$_kf" 2>/dev/null || stat -c '%a' "$_kf" 2>/dev/null)
        [ "$_perm" = 600 ] || die "key file $_kf is mode $_perm, must be 600"
        RUNPOD_API_KEY=$(cat "$_kf")
    fi
    [ -n "${RUNPOD_API_KEY:-}" ] || return 1
    export RUNPOD_API_KEY
    ( umask 077; printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$RUNPOD_API_KEY" > "$CURLRC" )
    return 0
}

rp_call() {  # METHOD URL [json file]; sets RP_CODE, body in $TMPD/rp.body
    : > "$TMPD/rp.body"
    if [ -n "${3:-}" ]; then
        RP_CODE=$(curl -K "$CURLRC" -o "$TMPD/rp.body" -w '%{http_code}' -X "$1" \
            -H 'Content-Type: application/json' --data-binary "@$3" "$2" 2>>"$TMPD/curl.err") || RP_CODE=000
    else
        RP_CODE=$(curl -K "$CURLRC" -o "$TMPD/rp.body" -w '%{http_code}' -X "$1" "$2" 2>>"$TMPD/curl.err") || RP_CODE=000
    fi
}

rp_py() {  # python over the parsed body `d` (a list becomes {"items": [...]})
    python3 - "$TMPD/rp.body" "$@" <<'PYEOF' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
if isinstance(d, list):
    d = {"items": d}
what = sys.argv[2]
pods = d.get("items") or d.get("pods") or []
if what == "id":
    print(d.get("id") or "")
elif what == "cost":
    print(d.get("costPerHr") or d.get("adjustedCostPerHr") or "")
elif what == "ssh":
    ip = d.get("publicIp") or ""
    pm = d.get("portMappings") or {}
    port = pm.get("22") if isinstance(pm, dict) else ""
    if ip and port:
        print("-p %s root@%s" % (port, ip))
elif what == "status":
    print(d.get("desiredStatus") or "")
elif what == "names":
    for p in pods:
        print("%s\t%s\t%s\t%s" % (p.get("id"), p.get("name"), p.get("desiredStatus"), p.get("costPerHr")))
elif what == "byname":
    print(" ".join(str(p.get("id")) for p in pods if p.get("name") == sys.argv[3]))
elif what == "hasid":
    print("yes" if any(p.get("id") == sys.argv[3] for p in pods) else "no")
PYEOF
}

bssh() {  # shellcheck disable=SC2086
    ssh $SSH_OPTS $SSH_TARGET "$@"
}

r2py() {  # run a tools/*.py with ~/.mojolearn_r2 as an env prefix, never exported here
    _creds="${MOJOLEARN_R2_CREDS:-$HOME/.mojolearn_r2}"
    [ -f "$_creds" ] || { echo "no $_creds" >&2; return 1; }
    (
        # shellcheck disable=SC1090
        . "$_creds"
        R2_ACCOUNT_ID="${R2_ACCOUNT_ID:-}" R2_ACCESS_KEY_ID="${R2_ACCESS_KEY_ID:-}" \
        R2_SECRET_ACCESS_KEY="${R2_SECRET_ACCESS_KEY:-}" R2_BUCKET="${R2_BUCKET:-}" \
            python3 "$@"
    )
}

# ---------------------------------------------------------------------------
# list / reap
# ---------------------------------------------------------------------------
verify_gone() {  # pod id; 0 when the API says it is gone
    _i=1
    while [ "$_i" -le 8 ]; do
        rp_call GET "$RP/pods/$1"
        _code=$RP_CODE
        _st=$(rp_py status)
        rp_call GET "$RP/pods"
        _listed=$(rp_py hasid "$1")
        if { [ "$_code" = 404 ] || [ "$_st" = TERMINATED ]; } && [ "$_listed" = no ]; then
            echo "  VERIFIED: $1 is gone (GET pod HTTP $_code${_st:+ status $_st}; not in the pod listing)"
            return 0
        fi
        echo "  $1 not yet gone (GET HTTP $_code status '${_st:-?}', listed=$_listed), attempt $_i/8"
        sleep 10
        _i=$((_i + 1))
    done
    return 1
}

delete_pod() {
    for _u in "$RP/pods/$1" "$RP_V2/pods/$1"; do
        rp_call DELETE "$_u"
        echo "  DELETE $_u -> HTTP $RP_CODE"
        case "$RP_CODE" in 2*|404) break ;; esac
    done
}

case "${1:-}" in
    list)
        load_key || die "no RunPod key"
        rp_call GET "$RP/pods"
        [ "${RP_CODE#2}" != "$RP_CODE" ] || die "pod listing HTTP $RP_CODE"
        rp_py names | awk -F'\t' '$2 ~ /^mojolearn-cpu-/' ; exit 0 ;;
    reap)
        [ -n "${2:-}" ] || die "usage: reap POD_ID"
        load_key || die "no RunPod key"
        rp_call GET "$RP/pods/$2"
        _n=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("name",""))' "$TMPD/rp.body" 2>/dev/null)
        case "$_n" in mojolearn-cpu-*) ;; *) die "$2 is named '$_n', not mojolearn-cpu-*; not this runner's pod" ;; esac
        delete_pod "$2"; verify_gone "$2"; exit $? ;;
esac

while [ $# -gt 0 ]; do
    case "$1" in
        --lane) shift; LANE="${1:-}" ;;
        --cmd) shift; CMD="${1:-}" ;;
        --cmd-file) shift; CMD_FILE="${1:-}" ;;
        --worktree) shift; WORKTREE="${1:-}" ;;
        --include) shift; INCLUDES="$INCLUDES
${1:-}" ;;
        --build) shift; BUILD="${1:-}" ;;
        --sabotage-build) shift; SAB_BUILD="${1:-}" ;;
        --sabotage-defines) shift; SAB_DEFINES="${1:-}" ;;
        --envs) shift; ENVS="${1:-}" ;;
        --vcpu) shift; VCPU="${1:-}" ;;
        --flavors) shift; FLAVORS="${1:-}" ;;
        --lease) shift; LEASE="${1:-}" ;;
        --jobs) shift; JOBS="${1:-}" ;;
        --image) shift; IMAGE="${1:-}" ;;
        --disk) shift; DISK="${1:-}" ;;
        --out) shift; OUT="${1:-}" ;;
        --no-bincache) BINCACHE=0 ;;
        --envcache) ENVCACHE=1 ;;
        --no-envcache) ENVCACHE=0 ;;
        --max-pods) shift; MAX_PODS="${1:-}" ;;
        --stage) shift; STAGE_KEYS="${1:-}" ;;
        --rent) RENT=1 ;;
        -h|--help) sed -n '2,68p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown argument '$1' (see --help)" ;;
    esac
    shift
done

# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------
printf '%s' "$LANE" | grep -Eq '^[a-z0-9][a-z0-9-]{0,30}$' || die "--lane must match [a-z0-9][a-z0-9-]{0,30}"
if [ -n "$CMD_FILE" ]; then
    [ -f "$CMD_FILE" ] || die "--cmd-file $CMD_FILE does not exist"
    CMD=$(cat "$CMD_FILE")
fi
[ -n "$CMD" ] || die "--cmd or --cmd-file is required"
for _n in "$VCPU" "$LEASE" "$JOBS" "$DISK" "$MAX_PODS"; do
    printf '%s' "$_n" | grep -Eq '^[0-9]+$' || die "numeric option '$_n' is not a number"
done
[ "$VCPU" -ge 2 ] && [ "$VCPU" -le 32 ] || die "--vcpu must be 2..32"
[ "$LEASE" -ge 10 ] && [ "$LEASE" -le 180 ] || die "--lease must be 10..180 minutes"
[ "$MAX_PODS" -ge 1 ] && [ "$MAX_PODS" -le 12 ] || die "--max-pods must be 1..12"
printf '%s' "$FLAVORS" | grep -Eq '^cpu[35][cgm](,cpu[35][cgm])*$' || die "--flavors must be cpu3c/cpu3g/cpu3m/cpu5c/cpu5g/cpu5m, comma separated"
printf '%s' "$ENVS" | grep -Eq '^[a-z0-9_-]+(,[a-z0-9_-]+)*$' || die "--envs is a comma separated list of pixi env names"
case ",$ENVS," in *,default,*) ;; *) die "--envs must include default (the build and the command use its python and mojo)" ;; esac
[ -n "$WORKTREE" ] || WORKTREE=$(git rev-parse --show-toplevel 2>/dev/null) || die "not in a git checkout; pass --worktree"
WORKTREE=$(cd "$WORKTREE" && pwd) || die "--worktree $WORKTREE is not a directory"
for _f in $(printf '%s' "$BUILD,$SAB_BUILD" | tr , ' '); do
    printf '%s' "$_f" | grep -Eq '^[a-z_]+$' || die "family '$_f' is not [a-z_]+"
    [ -f "$WORKTREE/bindings/build_${_f}_host.sh" ] || die "no bindings/build_${_f}_host.sh in $WORKTREE"
done
COMMIT=$(git -C "$WORKTREE" rev-parse HEAD) || die "cannot read HEAD of $WORKTREE"
DIRTY=$(git -C "$WORKTREE" status --porcelain --untracked-files=no | wc -l | tr -d ' ')
STAMP=$(date -u +%Y%m%d-%H%M%S)
POD_NAME="mojolearn-cpu-$LANE-$STAMP"
[ -n "$OUT" ] || OUT="$WORKTREE/bench/results/runpod_cpu/$(date -u +%Y-%m-%d_%H%M%S)-$LANE"

# ---------------------------------------------------------------------------
# composed artifacts (dry run and rent alike)
# ---------------------------------------------------------------------------
make_source() {
    ( cd "$WORKTREE" && git ls-files -z -- . ':!bench/results' ':!mamba/corpus' ':!bench/oracle_*' ':!bench/minentropy_oracle.txt' ) > "$TMPD/files0"
    if [ -n "$INCLUDES" ]; then
        printf '%s\n' "$INCLUDES" | while IFS= read -r _inc; do
            [ -n "$_inc" ] || continue
            ( cd "$WORKTREE" && git ls-files -z -- "$_inc" ) >> "$TMPD/files0"
        done
    fi
    python3 - "$WORKTREE" "$TMPD/files0" "$TMPD/src.tgz" <<'PYEOF'
import gzip, hashlib, os, sys, tarfile
root, lst, dest = sys.argv[1:]
names = sorted(set(n for n in open(lst, "rb").read().decode().split("\0") if n))
n = 0
with gzip.GzipFile(dest, "wb", compresslevel=1, mtime=0) as gz, tarfile.open(fileobj=gz, mode="w", format=tarfile.PAX_FORMAT) as tf:
    for name in names:
        p = os.path.join(root, name)
        if not os.path.lexists(p):
            continue                     # tracked but deleted in the working tree
        tf.add(p, arcname=name, recursive=False)
        n += 1
h = hashlib.sha256(open(dest, "rb").read()).hexdigest()
print("%d\t%d\t%s" % (n, os.path.getsize(dest), h))
PYEOF
}

write_create_request() {
    python3 - "$1" "$POD_NAME" "$IMAGE" "$FLAVORS" "$VCPU" "$DISK" <<'PYEOF'
import json, sys
out, name, image, flavors, vcpu, disk = sys.argv[1:]
req = {"name": name, "imageName": image, "computeType": "CPU",
       "cpuFlavorIds": flavors.split(","), "vcpuCount": int(vcpu),
       "containerDiskInGb": int(disk), "volumeInGb": 0, "ports": ["22/tcp"],
       "supportPublicIp": True, "cloudType": "SECURE"}
open(out, "w").write(json.dumps(req, indent=2) + "\n")
PYEOF
}

write_box_script() {
    cat > "$TMPD/box.sh.in" <<'BOX_EOF'
#!/bin/bash
# Written by tools/runpod_cpu_leg.sh; runs DETACHED on the pod.
set -u
R=@BOXREPO@
OUT=/root/leg_out
MAP=/root/.mojolearn_cache/urls.tsv
C=/root/.mojolearn_cache/work
PIXIVER=@PIXIVER@
mkdir -p "$OUT" "$C"
ph() { printf '%s\t%s\n' "$1" "$(date +%s)" >> "$OUT/phases.tsv"; }
note() { printf '%s\n' "$*" >> "$OUT/cache.tsv"; }
url() { [ -f "$MAP" ] || return 0; awk -F'\t' -v k="$1" -v n="$2" -v v="$3" '$1==k && $2==n && $3==v {print $4; exit}' "$MAP"; }
sha() { sha256sum "$1" | cut -d' ' -f1; }
fetch_verified() {   # kind name dest: GET the sidecar and the object; the bytes must hash to the sidecar
    _g=$(url "$1" "$2.tar" get); _s=$(url "$1" "$2.sha" get)
    [ -n "$_g" ] && [ -n "$_s" ] || return 1
    curl -fsS --retry 3 -o "$C/$2.sha" "$_s" || return 1
    curl -fsS --retry 3 -o "$3" "$_g" || return 1
    if [ "$(sha "$3")" != "$(cut -c1-64 "$C/$2.sha")" ]; then
        note "$1:$2	REJECTED	sha256 mismatch"; rm -f "$3"; return 1
    fi
}
upload_verified() {  # kind name file: the object first, the sidecar LAST
    _p=$(url "$1" "$2.tar" put); _ps=$(url "$1" "$2.sha" put)
    [ -n "$_p" ] && [ -n "$_ps" ] || return 1
    sha "$3" > "$C/$2.sha.up"
    curl -fsS --retry 3 -T "$3" -o /dev/null "$_p" || return 1
    curl -fsS --retry 3 -T "$C/$2.sha.up" -o /dev/null "$_ps" || return 1
}
cd "$R" || { echo 9 > "$OUT/cmd.exit"; touch "$OUT/DONE"; exit 9; }
{ uname -a; nproc; grep -m1 'model name' /proc/cpuinfo; grep -o -m1 -w avx2 /proc/cpuinfo; head -4 /etc/os-release; free -g | head -2; df -h /root | tail -1; } > "$OUT/box.txt" 2>&1

ph env_start
export PIXI_HOME=/root/.pixi PIXI_NO_PATH_UPDATE=1
mkdir -p /root/.pixi/bin
PIXI=/root/.pixi/bin/pixi
pixi_state=miss
if fetch_verified pixi pixi "$C/pixi.bin"; then
    install -m 755 "$C/pixi.bin" "$PIXI" && pixi_state=restored
fi
if [ ! -x "$PIXI" ]; then
    for _try in 1 2 3; do
        curl -fsSL --max-time 300 https://pixi.sh/install.sh | PIXI_VERSION="$PIXIVER" bash >> "$OUT/pixi_install.log" 2>&1
        [ -x "$PIXI" ] && break
        sleep 10
    done
fi
export PATH="$R/.pixi/envs/default/bin:/root/.pixi/bin:$PATH"
pv=$("$PIXI" --version 2>/dev/null | awk '{print $2}')
note "pixi	$pixi_state	version=$pv"
: > "$C/envs_to_upload"
for E in $(echo "@ENVS@" | tr , ' '); do
    t0=$(date +%s); st=miss
    if [ "@ENVCACHE@" = 1 ] && fetch_verified env "$E" "$C/$E.tar.gz"; then
        mkdir -p "$R/.pixi/envs" && rm -rf "$R/.pixi/envs/$E" && tar -C "$R/.pixi/envs" -xzf "$C/$E.tar.gz" && st=restored
        rm -f "$C/$E.tar.gz"
    fi
    t1=$(date +%s)
    # ALWAYS run the locked install: on a restored env it must be a no-op, and
    # its seconds say whether pixi accepted the restored prefix.
    "$PIXI" install --locked -e "$E" > "$OUT/pixi_env_$E.log" 2>&1
    rc=$?
    t2=$(date +%s)
    note "env:$E	$st	install_rc=$rc	restore_s=$((t1 - t0))	pixi_install_s=$((t2 - t1))"
    if [ "@ENVCACHE@" = 1 ] && [ "$st" = miss ] && [ "$rc" = 0 ] && [ "$pv" = "$PIXIVER" ]; then echo "$E" >> "$C/envs_to_upload"; fi
done
"$R/.pixi/envs/default/bin/python3" -c 'import numpy, sys; print(sys.version.split()[0], numpy.__version__)' >> "$OUT/box.txt" 2>&1
# AN ENV THAT INSTALLS IS NOT AN ENV THAT RUNS. On 2026-09-15 `pixi install
# --locked` succeeded on a glibc 2.31 image whose mojo then could not load
# (GLIBC_2.35 not found), and both envs were uploaded. Nothing uploads unless
# the toolchain answers here.
if "$R/.pixi/envs/default/bin/mojo" --version >> "$OUT/box.txt" 2>&1; then
    note "toolchain	ok	$("$R/.pixi/envs/default/bin/mojo" --version 2>&1 | grep -m1 Mojo)"
else
    note "toolchain	BROKEN	mojo --version failed; no env is uploaded"
    : > "$C/envs_to_upload"
fi
ph env_end

ph build_start
PY="$R/.pixi/envs/default/bin/python3"
export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_BUILD_JOBS=@JOBS@ MOJOLEARN_LINUX_CPU=x86-64-v3
export MOJOLEARN_BINCACHE_OUT="$OUT/bincache"
[ "@BINCACHE@" = 1 ] || export MOJOLEARN_BINCACHE=0
for f in $(echo "@BUILD@" | tr , ' '); do
    t=$(date +%s)
    "$PY" tools/bincache.py build "bindings/build_${f}_host.sh" > "$OUT/build_$f.log" 2>&1
    printf 'build\t%s\t%s\t%s\n' "$f" "$?" "$(( $(date +%s) - t ))" >> "$OUT/status.tsv"
done
for f in $(echo "@SABBUILD@" | tr , ' '); do
    t=$(date +%s)
    env MOJOLEARN_BUILD_EXTRA_DEFINES="@SABDEFINES@" MOJOLEARN_HOST_OUTDIR=python/mojolearn/host-sabotage \
        MOJOLEARN_FOREST_HOST_OUTDIR=python/mojolearn/host-sabotage MOJOLEARN_BYTE_LM_HOST_OUTDIR=python/mojolearn/host-sabotage \
        MOJOLEARN_BINCACHE_NEGATIVE=1 "$PY" tools/bincache.py build "bindings/build_${f}_host.sh" > "$OUT/build_sabotage_$f.log" 2>&1
    printf 'sabotage-build\t%s\t%s\t%s\n' "$f" "$?" "$(( $(date +%s) - t ))" >> "$OUT/status.tsv"
done
sha256sum python/mojolearn/host/*.so python/mojolearn/host-sabotage/*.so > "$OUT/so_sha256.txt" 2>/dev/null
ph build_end

# THE HOST MATH LIBRARY, OR NOTHING IMPORTS (2026-09-20).
# `python/mojolearn/_portable_math.py` dlopens .libs/libMojolearnMath.so, and
# `_training_impl.py`'s `def kaiming_uniform(self, shape, fan_in,
# a=math.sqrt(5.0))` evaluates it as a DEFAULT ARGUMENT at class definition
# time, so `import mojolearn` needs it unconditionally. It is not lazy and no
# command can avoid it. Nothing under bindings/ builds it, `.libs/` is
# gitignored so the shipped tarball carries nothing, and the only thing in the
# tree that compiles it is packaging/macos/build_release_wheel.sh, which does
# not run on Linux. A developer Mac has it sitting in the checkout from some
# past wheel build and never notices; a freshly rented box cannot import the
# package at all.
#
# MEASURED ON THREE PODS TODAY -- zaho1l0oqoy2hh, g60mkatgi6epm4 and
# epn4g2y79weyad -- every host binding built, both arms differing on all 32,
# `missing_bindings.txt` EMPTY, and all three commands dead one second in at
# `OSError: .../libMojolearnMath.so: cannot open shared object file`, having
# recorded not one cell. $0.34 for three builds and no column.
#
# tools/gap_column_leg.sh learned this on 2026-09-19 and builds it; this
# runner did not, so the lesson sat in one leg body while every CPU leg that
# imports the package kept walking into it. It lands here instead, once, for
# every command this runner will ever run.
#
# This calls the tree's OWN recipe, packaging/portable_math/stage.py's
# build(), rather than retyping its compiler flags: -ffp-contract=off,
# -fno-fast-math, -march=x86-64-v3 and -nostdlib ARE the arithmetic contract,
# and a second copy of them here would be a second answer to it.
ph portable_math_start
t=$(date +%s)
env PYTHONPATH="$R/packaging/portable_math" "$PY" -c \
    "import pathlib, stage; stage.build(pathlib.Path('$R/python/mojolearn/.libs/libMojolearnMath.so'))" \
    > "$OUT/portable_math.log" 2>&1
printf 'portable-math\t%s\t%s\n' "$?" "$(( $(date +%s) - t ))" >> "$OUT/status.tsv"
ls -l "$R/python/mojolearn/.libs/" >> "$OUT/portable_math.log" 2>&1
ph portable_math_end

ph run_start
export MOJOLEARN_COMMIT=@COMMIT@ PYTHONPATH="$R/python" LEG_OUT="$OUT"
unset MOJOLEARN_BINCACHE_OUT
bash /root/leg_user_cmd.sh > "$OUT/cmd.log" 2>&1
echo $? > "$OUT/cmd.exit"
ph run_end

ph upload_start
while IFS= read -r E; do
    [ -n "$E" ] || continue
    t=$(date +%s)
    if command -v pigz > /dev/null 2>&1; then Z="pigz -1"; else Z="gzip -1"; fi
    tar -C "$R/.pixi/envs" -cf - "$E" | $Z > "$C/$E.tar.gz"
    sz=$(stat -c %s "$C/$E.tar.gz")
    if [ "$sz" -ge 5000000000 ]; then
        note "env:$E	not-uploaded	bytes=$sz exceeds one R2 PUT"
    elif upload_verified env "$E" "$C/$E.tar.gz"; then
        note "env:$E	uploaded	bytes=$sz	seconds=$(( $(date +%s) - t ))"
    else
        note "env:$E	upload-failed	bytes=$sz"
    fi
    rm -f "$C/$E.tar.gz"
done < "$C/envs_to_upload"
if [ "$pixi_state" = miss ] && [ "$pv" = "$PIXIVER" ]; then
    if upload_verified pixi pixi "$PIXI"; then note "pixi	uploaded	bytes=$(stat -c %s "$PIXI")"; else note "pixi	upload-failed"; fi
fi
ph upload_end
touch "$OUT/DONE"
BOX_EOF
    # A '|' or '&' in a substituted value would break sed; refuse rather than mangle.
    for _v in "$SAB_DEFINES" "$ENVS" "$BUILD" "$SAB_BUILD"; do
        case "$_v" in *'|'*|*'&'*|*'\'*) die "a value contains | & or backslash: $_v" ;; esac
    done
    sed -e "s|@BOXREPO@|$BOX_REPO|g" -e "s|@PIXIVER@|$PIXI_VERSION|g" -e "s|@ENVCACHE@|$ENVCACHE|g" \
        -e "s|@BINCACHE@|$BINCACHE|g" -e "s|@ENVS@|$ENVS|g" -e "s|@JOBS@|$JOBS|g" -e "s|@BUILD@|$BUILD|g" \
        -e "s|@SABBUILD@|$SAB_BUILD|g" -e "s|@SABDEFINES@|$SAB_DEFINES|g" -e "s|@COMMIT@|$COMMIT|g" \
        "$TMPD/box.sh.in" > "$TMPD/box.sh"
    if grep -n '@[A-Z]*@' "$TMPD/box.sh"; then die "unsubstituted placeholder in the box script"; fi
    [ "$(grep -c "$COMMIT" "$TMPD/box.sh")" -ge 1 ] || die "the commit did not land in the box script"
    bash -n "$TMPD/box.sh" || die "the box script is not valid bash"
    printf '%s\n' "$CMD" > "$TMPD/user_cmd.sh"
    bash -n "$TMPD/user_cmd.sh" || die "--cmd is not valid bash"
}

write_deadman() {  # dir seconds; composes and checks, never arms
    ( umask 077; mkdir -p "$1"; if [ -f "$CURLRC" ]; then cp "$CURLRC" "$1/curlrc"; else : > "$1/curlrc"; fi )
    cat > "$1/deadman.sh" <<'DM_EOF'
#!/bin/sh
# tools/runpod_cpu_leg.sh's Mac dead-man: ends the pod if the runner is gone.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
sleep @SECS@
echo "$(date -u +%FT%TZ) dead-man firing for @NAME@" >> "$D/deadman.log"
ids=""
[ -s "$D/pod_id.txt" ] && ids="$(cat "$D/pod_id.txt")"
if [ -z "$ids" ]; then
    curl -K "$D/curlrc" -o "$D/pods.json" "@RP@/pods" >> "$D/deadman.log" 2>&1
    ids="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(" ".join(p["id"] for p in d if p.get("name")==sys.argv[2]))' "$D/pods.json" "@NAME@" 2>/dev/null)"
fi
for id in $ids; do
    for u in "@RP@/pods/$id" "@RPV2@/pods/$id"; do
        c="$(curl -K "$D/curlrc" -o /dev/null -w '%{http_code}' -X DELETE "$u" 2>>"$D/deadman.log")"
        echo "$(date -u +%FT%TZ) DELETE $u -> $c" >> "$D/deadman.log"
        case "$c" in 2*|404) break ;; esac
    done
done
rm -f "$D/curlrc"
DM_EOF
    sed -i.bak -e "s|@SECS@|$2|g" -e "s|@NAME@|$POD_NAME|g" -e "s|@RP@|$RP|g" -e "s|@RPV2@|$RP_V2|g" "$1/deadman.sh"
    rm -f "$1/deadman.sh.bak"
    if grep -q '@[A-Z0-9]*@' "$1/deadman.sh"; then return 1; fi
    sh -n "$1/deadman.sh"
}

# ---------------------------------------------------------------------------
# teardown: runs on EVERY exit once a pod id exists
# ---------------------------------------------------------------------------
write_timings() {
    [ -n "$T_POST" ] || return 0
    python3 - "$OUT" "$T_POST" "${T_SSH:-}" "${T_ARMED:-}" "${T_STAGED:-}" "${T_DONE:-}" "${T_FETCHED:-}" \
        "${T_DEL0:-}" "${T_DEL1:-}" "${COST_HR:-}" "${BOX_SKEW:-0}" <<'PYEOF'
import os, sys
out, post, ssh, armed, staged, done, fetched, del0, del1, cost, skew = sys.argv[1:]
f = lambda s: float(s) if s else None
post, ssh, armed, staged, done, fetched, del0, del1 = map(f, (post, ssh, armed, staged, done, fetched, del0, del1))
skew = float(skew or 0)
ph = {}
p = os.path.join(out, "remote", "leg_out", "phases.tsv")
if os.path.exists(p):
    for line in open(p):
        k, v = line.split("\t")
        ph[k] = float(v) - skew          # box clock to Mac clock
rows = []
def span(name, a, b):
    rows.append((name, "%.0f" % (b - a) if a is not None and b is not None else "n/a"))
span("create (POST to ssh up)", post, ssh)
span("arm (on-pod watchdog)", ssh, armed)
span("stage (source + R2 URL maps)", armed, staged)
span("env (pixi binary + envs)", ph.get("env_start"), ph.get("env_end"))
span("build (host bindings)", ph.get("build_start"), ph.get("build_end"))
span("run (command)", ph.get("run_start"), ph.get("run_end"))
span("create to first result (POST to command end)", post, ph.get("run_end"))
span("cache upload (misses only)", ph.get("upload_start"), ph.get("upload_end"))
span("fetch", done, fetched)
span("teardown (DELETE to verified gone)", del0, del1)
span("billed (POST to verified gone)", post, del1)
with open(os.path.join(out, "timings.tsv"), "w") as fh:
    for k, v in rows:
        fh.write("%s\t%s\n" % (k, v))
        print("  %-48s %s s" % (k, v))
if cost and post and del1:
    usd = float(cost) * (del1 - post) / 3600.0
    line = "spend\t$%.4f at $%s/hr" % (usd, cost)
    open(os.path.join(out, "timings.tsv"), "a").write(line + "\n")
    print("  " + line.replace("\t", " "))
PYEOF
}

teardown() {
    _rc=$?
    trap - EXIT INT TERM
    if [ -n "$POD_ID" ]; then
        echo
        echo "== teardown (exit $_rc) =="
        bssh 'rm -f /root/.mojolearn_cache/urls.tsv /root/.mojolearn_bincache/urls.tsv' > /dev/null 2>&1 || true
        T_DEL0=$(now)
        delete_pod "$POD_ID"
        if verify_gone "$POD_ID"; then POD_TERMINATED=1; fi
        T_DEL1=$(now)
        { echo "pod=$POD_ID"; echo "name=$POD_NAME"; echo "terminated_verified=$POD_TERMINATED"; echo "exit=$_rc"; echo "at=$(date -u +%FT%TZ)"; } >> "$OUT/teardown.txt"
        write_timings
    fi
    if [ -n "$DEADMAN_PID" ]; then
        if [ -z "$POD_ID" ] || [ "$POD_TERMINATED" = 1 ]; then
            pkill -P "$DEADMAN_PID" 2>/dev/null || true
            kill "$DEADMAN_PID" 2>/dev/null && echo "  dead-man cancelled (pid $DEADMAN_PID)"
            rm -rf "$DEADMAN_DIR"
        else
            echo "  ##########################################################"
            echo "  # $POD_ID WAS NOT CONFIRMED GONE. The dead-man stays armed"
            echo "  # (pid $DEADMAN_PID) and the on-pod watchdog fires at the lease."
            echo "  # End it by hand:  bash tools/runpod_cpu_leg.sh reap $POD_ID"
            echo "  ##########################################################"
        fi
    fi
    rm -rf "$TMPD"
    exit "$_rc"
}

# ---------------------------------------------------------------------------
# plan (both modes)
# ---------------------------------------------------------------------------
echo "== runpod_cpu_leg: lane $LANE, $([ "$RENT" = 1 ] && echo RENT || echo 'DRY RUN') =="
echo "  worktree $WORKTREE  commit $(printf '%s' "$COMMIT" | cut -c1-12)  dirty tracked files: $DIRTY"
echo "  pod $POD_NAME  image $IMAGE  flavors $FLAVORS  vCPU $VCPU  disk ${DISK}GB  lease ${LEASE}m"
echo "  envs $ENVS  build [$BUILD]  sabotage-build [$SAB_BUILD]  jobs $JOBS  bincache=$BINCACHE envcache=$ENVCACHE"
SRC=$(make_source) || die "the source tarball did not build"
echo "  source: $(echo "$SRC" | cut -f1) files, $(echo "$SRC" | cut -f2) bytes gz, sha256 $(echo "$SRC" | cut -f3 | cut -c1-16)"
[ "$(echo "$SRC" | cut -f2)" -le 40000000 ] || die "the source tarball is over 40 MB; the uplink is the rental's bottleneck (leg-archive-uplink-limit)"
write_box_script
write_create_request "$TMPD/create.json"
echo "  box script OK (bash -n, no placeholder, commit baked in); create request: $(tr -d ' \n' < "$TMPD/create.json")"
python3 "$ROOT/tools/runpod_cpu_cache.py" keys --repo "$WORKTREE" --envs "$ENVS" --prefix "$BOX_REPO" \
    --image "$IMAGE" --pixi-version "$PIXI_VERSION" | grep -v '^#' | sed 's/^/  cache key: /'
write_deadman "$TMPD/dm-check" 60 || die "the dead-man script did not compose"
rm -rf "$TMPD/dm-check"
echo "  dead-man composes (sh -n)"

if [ "$RENT" != 1 ]; then
    if load_key; then
        rp_call GET "$RP/pods"
        echo "  pod listing HTTP $RP_CODE; mojolearn-cpu pods live:"
        rp_py names | awk -F'\t' '$2 ~ /^mojolearn-cpu-/ {print "    " $0}'
        _lane_live=$(rp_py names | awk -F'\t' -v p="mojolearn-cpu-$LANE-" 'index($2, p) == 1' | wc -l | tr -d ' ')
        [ "$_lane_live" = 0 ] || echo "  A RENT NOW WOULD REFUSE: $_lane_live pod(s) with lane tag $LANE are live"
    else
        echo "  no RunPod key here; a rent would refuse"
    fi
    if r2py "$ROOT/tools/runpod_cpu_cache.py" status --repo "$WORKTREE" --envs "$ENVS" --prefix "$BOX_REPO" \
            --image "$IMAGE" --pixi-version "$PIXI_VERSION" > "$TMPD/status" 2>&1; then
        sed 's/^/  R2 /' "$TMPD/status"
    else
        echo "  R2 status unavailable: $(tail -1 "$TMPD/status")"
    fi
    echo
    echo "DRY RUN: nothing was created and nothing was billed. Add --rent to create the pod."
    rm -rf "$TMPD"
    exit 0
fi

# ---------------------------------------------------------------------------
# RENT
# ---------------------------------------------------------------------------
load_key || die "no RunPod key (MOJOLEARN_RUNPOD_KEY_FILE or ~/.mojolearn_runpod_key)"
[ -f "${MOJOLEARN_R2_CREDS:-$HOME/.mojolearn_r2}" ] || die "no R2 credentials; the caches are the point of this runner"
mkdir -p "$OUT" || die "cannot create $OUT"
trap teardown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cp "$TMPD/create.json" "$OUT/create_request.json"
cp "$TMPD/box.sh" "$OUT/box_script.sh"
cp "$TMPD/user_cmd.sh" "$OUT/user_cmd.sh"
{ echo "lane=$LANE"; echo "commit=$COMMIT"; echo "dirty_tracked=$DIRTY"; echo "worktree=$WORKTREE"; echo "source=$SRC"; } > "$OUT/leg.txt"

say "pre-flight: pods on this account"
rp_call GET "$RP/pods"
case "$RP_CODE" in 2*) ;; *) die "pod listing HTTP $RP_CODE; a runner that cannot list cannot verify a delete" ;; esac
rp_py names | awk -F'\t' '$2 ~ /^mojolearn-cpu-/ {print "    " $0}'
_lane_live=$(rp_py names | awk -F'\t' -v p="mojolearn-cpu-$LANE-" 'index($2, p) == 1' | wc -l | tr -d ' ')
[ "$_lane_live" = 0 ] || die "a pod with lane tag $LANE is already live; one pod per lane (bash tools/runpod_cpu_leg.sh list)"
_all_live=$(rp_py names | awk -F'\t' '$2 ~ /^mojolearn-cpu-/' | wc -l | tr -d ' ')
[ "$_all_live" -lt "$MAX_PODS" ] || die "$_all_live mojolearn-cpu pods are live, the cap is $MAX_PODS"

DEADMAN_DIR="${TMPDIR:-/tmp}/mojolearn-cpu-deadman-$$"
_dm_secs=$(( READY_TIMEOUT + LEASE * 60 + 600 ))
write_deadman "$DEADMAN_DIR" "$_dm_secs" || die "the dead-man did not compose; nothing created"
nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$DEADMAN_DIR/deadman.sh" > /dev/null 2>&1 < /dev/null &
DEADMAN_PID=$!
sleep 1
kill -0 "$DEADMAN_PID" 2>/dev/null || { DEADMAN_PID=""; die "the dead-man did not start; nothing created"; }
say "dead-man ARMED before the create: pid $DEADMAN_PID, fires in ${_dm_secs}s, keyed by $POD_NAME"

say "creating $POD_NAME. THE BILL STARTS HERE."
T_POST=$(now)
rp_call POST "$RP/pods" "$TMPD/create.json"
cp "$TMPD/rp.body" "$OUT/create_response.json"
POD_ID=$(rp_py id)
if [ -z "$POD_ID" ]; then
    rp_call GET "$RP/pods"
    POD_ID=$(rp_py byname "$POD_NAME" | awk '{print $1}')
    [ -n "$POD_ID" ] && { printf '%s\n' "$POD_ID" > "$DEADMAN_DIR/pod_id.txt"; die "create response unparsed but $POD_ID exists by name; tearing it down"; }
    die "create FAILED (HTTP $RP_CODE): $(head -c 300 "$OUT/create_response.json")"
fi
printf '%s\n' "$POD_ID" > "$DEADMAN_DIR/pod_id.txt"
printf '%s\n' "$POD_ID" > "$OUT/pod_id.txt"
cp "$OUT/create_response.json" "$TMPD/rp.body"
COST_HR=$(rp_py cost)
say "pod $POD_ID created. COST \$${COST_HR:-?}/hr"

_deadline=$(( $(now) + READY_TIMEOUT ))
while [ "$(now)" -lt "$_deadline" ]; do
    rp_call GET "$RP/pods/$POD_ID"
    [ -n "$COST_HR" ] || COST_HR=$(rp_py cost)
    SSH_TARGET=$(rp_py ssh)
    if [ -n "$SSH_TARGET" ] && bssh 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK; then break; fi
    SSH_TARGET=""
    sleep 10
done
[ -n "$SSH_TARGET" ] || die "READY TIMEOUT: no ssh after ${READY_TIMEOUT}s"
T_SSH=$(now)
cp "$TMPD/rp.body" "$OUT/pod_ready.json"
say "ssh up ($SSH_TARGET) after $((T_SSH - T_POST))s; cost \$${COST_HR:-?}/hr"

say "arming the ON-POD watchdog ($LEASE minutes)"
_hp=$(printf '%s' "$SSH_TARGET" | awk '{print "[" substr($3, index($3, "@") + 1) "]:" $2}')
ssh-keygen -R "$_hp" > /dev/null 2>&1 || true     # runpod_guard.sh uses accept-new; a reused ip:port must not refuse it
if ! MOJOLEARN_LEASE_DIR="$OUT/lease" sh "$ROOT/tools/runpod_guard.sh" arm "$POD_ID" "$SSH_TARGET" "$LEASE" > "$OUT/arm.log" 2>&1; then
    sed 's/^/    /' "$OUT/arm.log"; die "ARM REFUSED; the pod is not used"
fi
bssh "p=\$(cat /tmp/mojolearn-lease.pid 2>/dev/null); kill -0 \"\$p\" 2>/dev/null && echo WATCHDOG_ALIVE pid=\$p; grep -o 'sleep [0-9]*' /tmp/mojolearn-lease.sh; grep -c $POD_ID /tmp/mojolearn-lease.sh; curl -s -o /dev/null -w 'TOKEN_GET_%{http_code}\n' -K /tmp/mojolearn-lease.curlrc $RP/pods/$POD_ID" > "$OUT/watchdog_check.txt" 2>&1
sed 's/^/    /' "$OUT/watchdog_check.txt"
grep -q WATCHDOG_ALIVE "$OUT/watchdog_check.txt" && grep -q TOKEN_GET_200 "$OUT/watchdog_check.txt" \
    || die "the on-pod watchdog is not alive or its token does not answer 200"
T_ARMED=$(now)

say "staging the source ($(echo "$SRC" | cut -f2) bytes)"
bssh "rm -rf $BOX_REPO /root/leg_out /root/.mojolearn_cache && mkdir -p $BOX_REPO && cat > /root/src.tgz" < "$TMPD/src.tgz" || die "source upload failed"
_rsha=$(bssh "sha256sum /root/src.tgz | cut -c1-64; tar -xzf /root/src.tgz -C $BOX_REPO && rm -f /root/src.tgz && echo UNPACKED" 2>&1)
[ "$(echo "$_rsha" | head -1)" = "$(echo "$SRC" | cut -f3)" ] && echo "$_rsha" | grep -q UNPACKED \
    || die "source sha256 differs on the box or did not unpack: $_rsha"
BOX_SKEW=$(( $(bssh 'date +%s') - $(now) ))
echo "box_clock_minus_mac=$BOX_SKEW" >> "$OUT/leg.txt"
_exp=$(( LEASE * 60 + 1800 ))
if ! r2py "$ROOT/tools/runpod_cpu_cache.py" plan --repo "$WORKTREE" --envs "$ENVS" --prefix "$BOX_REPO" \
        --image "$IMAGE" --pixi-version "$PIXI_VERSION" --expires "$_exp" > "$TMPD/cachemap" 2> "$OUT/cache_plan.log"; then
    echo "  cache plan failed: $(tail -1 "$OUT/cache_plan.log"); the box installs from scratch"
    : > "$TMPD/cachemap"
fi
chmod 600 "$TMPD/cachemap"
grep '^#key' "$TMPD/cachemap" > "$OUT/cache_keys.tsv" || true
say "$(tail -1 "$OUT/cache_plan.log")"
{ printf 'umask 077\nmkdir -p /root/.mojolearn_cache\ncat > /root/.mojolearn_cache/urls.tsv <<'"'"'CACHE_MAP_EOF'"'"'\n'
  cat "$TMPD/cachemap"
  printf 'CACHE_MAP_EOF\necho CACHE_MAP_OK\n'
} | bssh 'sh -s' | grep -q CACHE_MAP_OK || die "the cache URL map did not land"
rm -f "$TMPD/cachemap"
if [ "$BINCACHE" = 1 ] && [ -n "$BUILD$SAB_BUILD" ]; then
    _neg=0; [ -n "$SAB_BUILD" ] && _neg=1
    MOJOLEARN_BINCACHE_NEGATIVE=$_neg sh "$ROOT/tools/bincache_leg.sh" stage "$SSH_TARGET" "runpod-cpu:$IMAGE" > "$OUT/bincache_stage.log" 2>&1 || true
    say "$(tail -1 "$OUT/bincache_stage.log")"
fi
if [ -n "$STAGE_KEYS" ]; then
    say "staging R2 keys: $STAGE_KEYS"
    # shellcheck disable=SC2086
    MOJOLEARN_STAGE_STRICT=1 MOJOLEARN_STAGE_BOX_REPO="$BOX_REPO" sh "$ROOT/tools/stage_from_r2.sh" "$SSH_TARGET" $STAGE_KEYS \
        > "$OUT/stage_r2.log" 2>&1 || die "R2 STAGING FAILED (strict): $(tail -1 "$OUT/stage_r2.log")"
    say "$(tail -1 "$OUT/stage_r2.log")"
fi
bssh 'umask 022; cat > /root/leg_box.sh' < "$TMPD/box.sh" || die "box script upload failed"
bssh 'umask 022; cat > /root/leg_user_cmd.sh' < "$TMPD/user_cmd.sh" || die "command upload failed"
T_STAGED=$(now)

say "starting the box pipeline (env, build, command, cache upload)"
bssh 'nohup bash /root/leg_box.sh > /root/leg_box.log 2>&1 < /dev/null & echo STARTED' | grep -q STARTED || die "the box pipeline did not start"
_poll_end=$(( T_ARMED + LEASE * 60 - 300 ))
_seen=0; _fails=0
while :; do
    sleep 20
    # ONLY AN SSH FAILURE COUNTS. The remote command ends in `true`: on
    # 2026-09-15 it ended in `[ -f DONE ] && echo`, which exits 1 on every
    # poll before DONE, so 15 ordinary polls read as 15 failures and the
    # runner tore the pod down in the middle of the cache upload.
    _o=$(bssh 'cat /root/leg_out/phases.tsv 2>/dev/null; if [ -f /root/leg_out/DONE ]; then echo __DONE__; fi; true' 2>/dev/null)
    if [ $? = 0 ]; then _fails=0; else _fails=$((_fails + 1)); fi
    _n=$(printf '%s\n' "$_o" | grep -c "	")
    if [ "$_n" -gt "$_seen" ]; then
        printf '%s\n' "$_o" | grep "	" | tail -n $((_n - _seen)) | while IFS="	" read -r _k _t; do say "  box: $_k"; done
        _seen=$_n
    fi
    printf '%s\n' "$_o" | grep -q __DONE__ && break
    [ "$_fails" -lt 15 ] || { say "15 polls failed in a row; fetching what exists"; break; }
    [ "$(now)" -lt "$_poll_end" ] || { say "the lease is nearly spent; fetching what exists"; break; }
done
T_DONE=$(now)

say "fetching \$LEG_OUT"
mkdir -p "$OUT/remote"
bssh 'cp /root/leg_box.log /root/leg_out/ 2>/dev/null; tar czf - -C /root leg_out' | ( cd "$OUT/remote" && tar xzf - ) \
    || echo "  FETCH FAILED"
T_FETCHED=$(now)
if [ -f "$OUT/remote/leg_out/bincache/uploads.tsv" ]; then
    sh "$ROOT/tools/bincache_leg.sh" promote "$OUT/remote/leg_out/bincache" > "$OUT/bincache_promote.log" 2>&1 || true
    say "$(tail -1 "$OUT/bincache_promote.log")"
fi
[ -f "$OUT/remote/leg_out/cache.tsv" ] && sed 's/^/    cache: /' "$OUT/remote/leg_out/cache.tsv"
[ -f "$OUT/remote/leg_out/bincache/provenance.tsv" ] && cut -f2,3,5 "$OUT/remote/leg_out/bincache/provenance.tsv" | sed 's/^/    bincache: /'
_cmd_exit=$(cat "$OUT/remote/leg_out/cmd.exit" 2>/dev/null || echo missing)
say "command exit: $_cmd_exit   results: $OUT"
[ "$_cmd_exit" = 0 ]
