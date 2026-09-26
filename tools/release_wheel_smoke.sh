#!/bin/bash
# tools/release_wheel_smoke.sh -- the Linux light-route smoke of ONE exact final
# wheel on ONE rented GPU box, and nothing else (2026-09-22).
#
#   bash tools/release_wheel_smoke.sh <wheel> --expected-source-commit <40-hex> [options]          DRY RUN
#   bash tools/release_wheel_smoke.sh <wheel> --expected-source-commit <40-hex> [options] --rent   rents
#   bash tools/release_wheel_smoke.sh <wheel> --expected-source-commit <40-hex> --ssh '<target>'   an existing box
#
# It runs tools/qualify_verifier_wheel.py --scope expanded (the checklist's
# step 5 smoke) on the box and brings results.json home for
# `tools/release_linux_publish.sh <wheel> <tag> pypi <work> --light-smoke <out>/results.json`.
#
# WHY A DEDICATED RUNNER. Until 0.8.14 this smoke rode tools/gemm_remote_leg.sh
# nvidia with a MOJOLEARN_GEMM_LEG_EXTRA body: that leg installs pixi and the
# repo's environments, runs its card and driver gates, refuses --source-ref
# outside the mamba payload, demands a local Metal card, and left the Mac to
# poll for /root/wheel_smoke_ready and scp the wheel by hand. Its source
# archive upload stalled for 14 minutes with no timeout. The smoke needs none
# of that: it installs the wheel into a fresh venv from PyPI's numpy. So this
# ships exactly two files, the wheel and qualify_verifier_wheel.py (stdlib
# only), and every transfer has a deadline.
#
# Options:
#   --out DIR            results (default ~/mojolearn-evidence/release-smoke/<version>/<stamp>-linux)
#   --gpu NAME           RunPod GPU type (default "NVIDIA GeForce RTX 4090", sm_89,
#                        which the wheel carries; the runner default of gemm_remote_leg.sh)
#   --image IMG          default runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04
#   --cuda VERSIONS      allowedCudaVersions, default "13.0" (the MAX runtime needs driver >= 580)
#   --lease MIN          on-box self-delete after MIN minutes (default 45, 10..90)
#   --smoke-seconds N    bound on the smoke itself (default 1800; 0.8.13 took 50 s on an H100)
#   --ssh TARGET         run on an existing box (e.g. "-p 22 root@1.2.3.4"): nothing
#                        is rented or deleted. The final wheel is packed on the Mac
#                        from all three legs, so no build box holds it; use this for
#                        a box you already have up.
#   --rent               create the box. Without it (and without --ssh): a dry run.
#   --vendor cuda|hip    the GPU family (default cuda). hip rents a RunPod AMD
#                        Instinct MI300X (gfx942, the wheel's AMD set) on
#                        rocm/dev-ubuntu-22.04:6.4.1-complete with the repo's ssh
#                        bootstrap, or a DigitalOcean MI325X (gfx942 too, see
#                        --provider); the expanded smoke runs on cuda only.
#   --provider P         where the box is rented: runpod | hotaisle | do | auto
#                        (default auto). runpod rents from RunPod only; hotaisle rents
#                        a Hot Aisle MI300X VM only; do rents a DigitalOcean GPU
#                        droplet only; auto walks runpod, hotaisle, do: RunPod ONCE,
#                        and when RunPod answers "There are no instances currently
#                        available" (0.8.16, 2026-09-23: no MI300X stock) Hot Aisle,
#                        and when Hot Aisle refuses BEFORE creating anything (no key,
#                        no stock, no slot, the balance, the cap) DigitalOcean.
#                        --vendor cuda rents from RunPod only (auto means runpod there).
#   --hotaisle-spec S    1gpu | 2gpu | auto (default auto, MOJOLEARN_HOTAISLE_RELEASE_SPEC):
#                        the 1x MI300X VM, the 2x MI300X VM (60-minute minimum,
#                        GPU 0 only: ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0),
#                        or 1gpu with 2gpu only when no 1x VM is in stock
#   --hotaisle-cap USD   refuse a Hot Aisle VM whose whole horizon (lease + ready
#                        timeout + 10 min, at the live price, never less than the
#                        offering's minimum reservation) costs more (default 10)
#   --do-size SLUG       the DigitalOcean size (default gpu-mi325x1-256gb: one MI325X,
#                        gfx942, the wheel's AMD set; $3.80/h on 2026-09-11)
#   --do-regions LIST    DigitalOcean regions tried in order (default tor1,nyc2; tor1 is
#                        where every AMD leg of this repository has rented)
#   --do-image ID        the DigitalOcean image (default 188571990, the ROCm image
#                        tools/do_release061_leg.sh builds the AMD set on)
#   --column SELECTION   also run the release pass's cells for the lanes in
#                        SELECTION (a verify_lanes --write-selection file for this
#                        vendor) from the INSTALLED wheel: fixtures base,denormal,odd,
#                        one fit, --fail-on-refused, --require-backend <vendor>.
#                        column.json comes home.
#   --ref-column FILE    with --column, repeatable: diff the GPU column against
#                        these columns of the same commit in ONE
#                        tools/identity_break.py --diff (diff-ref-<vendor>.txt);
#                        any DIVERGENT cell fails the run, named lane/fixture/part.
#                        tools/release.py passes the Apple (Metal) column of the
#                        release, and the CPU column too when it ran one.
#   --cpu-column FILE    the same as --ref-column FILE (kept for old command lines).
#   --plugin WHEEL       THE SPLIT LINUX PACKAGES (python/mojolearn/gpu_plugins.py):
#                        <wheel> is then the core `mojolearn` and WHEEL a plugin of
#                        the same version (mojolearn_nvidia-* or mojolearn_amd-*).
#                        REPEATABLE, one per plugin, and one of them must be the
#                        plugin of this --vendor. The core REQUIRES both plugins at
#                        its version (`pip install mojolearn` installs all three),
#                        so tools/release.py passes both: the box installs exactly
#                        what a user gets. All are shipped, sha256-checked on the
#                        box and installed together, by the smoke and by the
#                        column; the receipt names every plugin and its sha256.
#                        With a plugin the expanded smoke runs on hip too, so each
#                        plugin has a receipt of its own vendor
#                        (tools/check_light_release.py).
#
# A RENTED RUNPOD RUN, IN ORDER: Mac dead-man armed BEFORE the create (lease +
# ready timeout + 10 min, by id or by name); create; wait for ssh (600 s); arm
# the ON-POD watchdog (tools/runpod_guard.sh arm) and read it back; upload the
# two files (sha256 checked on the box); run the smoke detached and poll; fetch
# the output directory; DELETE and verify gone. Teardown runs on EVERY exit.
# The receipt is then checked here: status PASSED, the wheel's sha256 and the
# expected source commit.
#
# A RENTED DIGITALOCEAN RUN (--vendor hip with --provider do, or auto after
# RunPod had no stock): the token is ~/.mojolearn_do_token (0600, outside the
# repository, read and written by shell builtins into a 0600 curl config, never
# in an argv, delivered to the droplet on ssh STDIN); the shared GPU lock
# /tmp/mojolearn-do-gpu.lock is taken (ONE GPU droplet at a time across every
# session and lane on this Mac; held = refused, as tools/do_extra_leg.sh); the
# Mac dead-man is armed BEFORE the create (keyed by tag smoke + name, and the
# id once known); the droplet is created in the first region that accepts it;
# wait for active + a public IPv4 + three consecutive ssh successes; arm the
# ON-DROPLET self-destruct (sleeps the lease, then DELETEs its own droplet
# through the API) and verify it: process alive, id baked in, token GET 200.
# Then the SAME box-side flow as RunPod. DELETE, then GET until 404 (a 204 only
# acknowledges), and only after the 404 are the lock released and the dead-man
# cancelled. DigitalOcean bills until DESTROYED, never on power-off.
#
# A RENTED HOT AISLE RUN (--vendor hip with --provider hotaisle, or auto after
# RunPod had no stock), 2026-09-25: tools/hotaisle_vm_lib.sh's guards (its
# header): key ~/.mojolearn_hotaisle_key (0600, in no argv), a slot shared with
# tools/hotaisle_leg.sh, the whole horizon priced live and refused above
# --hotaisle-cap or when the balance cannot hold it plus $5, a Mac dead-man
# BEFORE the create, the description PATCHed, ssh as hotaisle with sudo, an
# ON-BOX watchdog verified from two sessions, gfx942 read from rocminfo. Then
# the SAME box-side flow as the others, every command run as root through
# `sudo -n bash -c`, natively on the VM's Ubuntu 24.04 host as on the
# DigitalOcean droplet. DELETE ?force=true, then GET 404 or absent from the
# listing; only then are the dead-man cancelled and the slot released.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
READY_TIMEOUT=600
# The box-side directory. Overridable only so tools/test_release_wheel_smoke.py
# can drive the whole --ssh path against a local shim; never a production knob.
RDIR=${MOJOLEARN_SMOKE_REMOTE_DIR:-/root/wheel-smoke}
printf '%s' "$RDIR" | grep -Eq '^/[A-Za-z0-9/_.-]*/wheel-smoke$' || { echo "bad MOJOLEARN_SMOKE_REMOTE_DIR" >&2; exit 2; }
# The on-droplet self-destruct lives beside it, outside the directory the
# upload wipes.
GUARD="${RDIR}-guard"
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes"

WHEEL=""; COMMIT=""; OUT=""; GPU="${MOJOLEARN_SMOKE_GPU:-NVIDIA GeForce RTX 4090}"
IMAGE="${MOJOLEARN_SMOKE_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
CUDA="13.0"; LEASE=45; SMOKE_SECONDS=1800; SSH_GIVEN=""; RENT=0
VENDOR=cuda; SELECTION=""; REFS=(); GPU_SET=0; IMAGE_SET=0; CUDA_SET=0
PROVIDER=auto
PLUGINS=(); PLUGIN=""; PLUGIN_SHA=""
DO_SIZE="${MOJOLEARN_SMOKE_DO_SIZE:-gpu-mi325x1-256gb}"
DO_REGIONS="${MOJOLEARN_SMOKE_DO_REGIONS:-tor1,nyc2}"
DO_IMAGE="${MOJOLEARN_SMOKE_DO_IMAGE:-188571990}"
# The API root is overridable only so the test can stand a local shim in for
# DigitalOcean (the RunPod one, RP, is the same in tools/runpod_pod_lib.sh).
DO_API=${MOJOLEARN_SMOKE_DO_API:-https://api.digitalocean.com/v2}
DO_SSH_KEY_FP="df:f7:6b:0c:56:da:48:a5:6f:6d:ae:44:af:de:f3:0b"   # "andrew macbook m4 air", ~/.ssh/id_ed25519
DO_TAG=smoke
DO_TOKFILE="${MOJOLEARN_DO_TOKEN_FILE:-$HOME/.mojolearn_do_token}"
DO_GPU_LOCK="${MOJOLEARN_DO_GPU_LOCK:-/tmp/mojolearn-do-gpu.lock}"
DO_LOCK_STALE_SECONDS=6000
HA_SPEC_ARG="${MOJOLEARN_HOTAISLE_RELEASE_SPEC:-auto}"
HA_CAP_USD=10
BOX_SUDO=0          # 1 on Hot Aisle: every box command runs as root through sudo -n bash -c
BOX_ENV=""          # the column's GPU pin on a 2x MI300X VM

say() { printf '[%s wheel-smoke] %s\n' "$(date +%T)" "$*"; }
die() { printf '\nREFUSED: %s\n' "$*" >&2; exit 1; }
now() { date +%s; }

TMPD=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-wheel-smoke.XXXXXX")
CURLRC="$TMPD/rp.curlrc"
DO_CURLRC="$TMPD/do.curlrc"
POD_NAME=""
# shellcheck source=tools/runpod_pod_lib.sh
. "$ROOT/tools/runpod_pod_lib.sh"
# shellcheck source=tools/hotaisle_vm_lib.sh
. "$ROOT/tools/hotaisle_vm_lib.sh"

while [ $# -gt 0 ]; do
    case "$1" in
        --expected-source-commit) shift; COMMIT="${1:-}" ;;
        --out) shift; OUT="${1:-}" ;;
        --gpu) shift; GPU="${1:-}"; GPU_SET=1 ;;
        --image) shift; IMAGE="${1:-}"; IMAGE_SET=1 ;;
        --cuda) shift; CUDA="${1:-}"; CUDA_SET=1 ;;
        --vendor) shift; VENDOR="${1:-}" ;;
        --provider) shift; PROVIDER="${1:-}" ;;
        --do-size) shift; DO_SIZE="${1:-}" ;;
        --do-regions) shift; DO_REGIONS="${1:-}" ;;
        --do-image) shift; DO_IMAGE="${1:-}" ;;
        --hotaisle-spec) shift; HA_SPEC_ARG="${1:-}" ;;
        --hotaisle-cap) shift; HA_CAP_USD="${1:-}" ;;
        --column) shift; SELECTION="${1:-}" ;;
        --plugin) shift; PLUGINS+=("${1:-}") ;;
        --ref-column|--cpu-column) shift; REFS+=("${1:-}") ;;
        --lease) shift; LEASE="${1:-}" ;;
        --smoke-seconds) shift; SMOKE_SECONDS="${1:-}" ;;
        --ssh) shift; SSH_GIVEN="${1:-}" ;;
        --rent) RENT=1 ;;
        -h|--help) sed -n '2,/^set -u$/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'; exit 0 ;;
        -*) die "unknown option '$1' (see --help)" ;;
        *) [ -z "$WHEEL" ] || die "one wheel only"; WHEEL="$1" ;;
    esac
    shift
done

# ---------------------------------------------------------------- validation
case "$VENDOR" in
    cuda) ;;
    hip) [ "$GPU_SET" = 1 ] || GPU="AMD Instinct MI300X OAM"
         [ "$IMAGE_SET" = 1 ] || IMAGE="rocm/dev-ubuntu-22.04:6.4.1-complete"
         [ "$CUDA_SET" = 1 ] || CUDA="" ;;
    *) die "--vendor must be cuda or hip" ;;
esac
case "$PROVIDER" in runpod|hotaisle|do|auto) ;; *) die "--provider must be runpod, hotaisle, do or auto" ;; esac
if [ "$VENDOR" = cuda ]; then
    case "$PROVIDER" in do|hotaisle) die "--provider $PROVIDER is for --vendor hip; the cuda smoke rents from RunPod" ;; esac
    PROVIDER=runpod
fi
case "$HA_SPEC_ARG" in 1gpu|2gpu|auto) HA_SPEC_WANT=$HA_SPEC_ARG ;; *) die "--hotaisle-spec must be 1gpu, 2gpu or auto" ;; esac
printf '%s' "$HA_CAP_USD" | grep -Eq '^[0-9]+(\.[0-9]{1,2})?$' || die "--hotaisle-cap must be a dollar figure like 10 or 7.50"
HA_CAP_CENTS=$(awk -v d="$HA_CAP_USD" 'BEGIN { printf "%d", d * 100 + 0.5 }')
printf '%s' "$DO_SIZE" | grep -Eq '^gpu-[a-z0-9-]+$' || die "--do-size '$DO_SIZE' is not a DigitalOcean GPU size slug"
printf '%s' "$DO_REGIONS" | grep -Eq '^[a-z0-9]+(,[a-z0-9]+)*$' || die "--do-regions must be region slugs separated by commas"
printf '%s' "$DO_IMAGE" | grep -Eq '^[0-9]+$' || die "--do-image must be a numeric DigitalOcean image id"
# The expanded smoke runs on cuda, and on hip when a split plugin is given
# (the amd plugin's light-route receipt); a combined wheel's hip leg is its
# column alone.
RUN_SMOKE=0
{ [ "$VENDOR" = cuda ] || [ "${#PLUGINS[@]}" -gt 0 ]; } && RUN_SMOKE=1
[ "$RUN_SMOKE" = 1 ] || [ -n "$SELECTION" ] || die "--vendor hip runs no smoke; give --column (or --plugin)"
LANES=""
if [ -n "$SELECTION" ]; then
    [ -f "$SELECTION" ] || die "no selection file $SELECTION"
    LANES=$(python3 - "$SELECTION" "$VENDOR" <<'PY'
import json, re, sys
d = json.load(open(sys.argv[1]))
if d.get("backend") != sys.argv[2]:
    sys.exit("the selection is for backend %r, not %r" % (d.get("backend"), sys.argv[2]))
lanes = d.get("lanes") or []
if not lanes or not all(re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", n) for n in lanes):
    sys.exit("the selection names no lanes, or a malformed lane")
print(",".join(lanes))
PY
) || die "bad --column selection: $LANES"
fi
for _ref in ${REFS[@]+"${REFS[@]}"}; do
    [ -n "$SELECTION" ] || die "--ref-column needs --column"
    [ -f "$_ref" ] || die "no reference column $_ref"
done
[ -n "$WHEEL" ] && [ -f "$WHEEL" ] || die "no wheel file given ($WHEEL)"
WHEEL=$(cd "$(dirname "$WHEEL")" && pwd)/$(basename "$WHEEL")
case "$(basename "$WHEEL")" in mojolearn-*-manylinux*_x86_64.whl) ;; *) die "$(basename "$WHEEL") is not a final manylinux x86_64 mojolearn wheel (smoke the repaired, stripped one)" ;; esac
printf '%s' "$COMMIT" | grep -Eq '^[0-9a-f]{40}$' || die "--expected-source-commit must be the full 40-hex commit"
for _n in "$LEASE" "$SMOKE_SECONDS"; do printf '%s' "$_n" | grep -Eq '^[0-9]+$' || die "'$_n' is not a number"; done
[ "$LEASE" -ge 10 ] && [ "$LEASE" -le 90 ] || die "--lease must be 10..90 minutes"
[ "$SMOKE_SECONDS" -ge 60 ] && [ "$SMOKE_SECONDS" -lt $(( LEASE * 60 - 300 )) ] || die "--smoke-seconds must be at least 60 and leave 5 minutes of the lease"
[ "$RENT" = 0 ] || [ -z "$SSH_GIVEN" ] || die "--rent and --ssh are exclusive"
QUALIFY="$ROOT/tools/qualify_verifier_wheel.py"
[ -f "$QUALIFY" ] || die "no $QUALIFY"
WHEEL_INFO=$(python3 - "$WHEEL" "$COMMIT" 2>&1 <<'PY'
import hashlib, re, sys, zipfile
wheel, want = sys.argv[1:]
with zipfile.ZipFile(wheel) as z:
    commit = z.read('mojolearn/identity_columns/COMMIT').decode().strip()
    z.getinfo('mojolearn/verify_reference/models/models.json')
if commit != want:
    sys.exit('the wheel records source commit %s, not %s' % (commit, want))
version = wheel.rsplit('/', 1)[-1].split('-')[1]
print(hashlib.sha256(open(wheel, 'rb').read()).hexdigest(), version)
PY
) || die "the wheel does not match: $WHEEL_INFO"
WHEEL_SHA=${WHEEL_INFO%% *}; VERSION=${WHEEL_INFO##* }
# Every --plugin: a mojolearn_nvidia/mojolearn_amd wheel of this version, one
# per distribution, and one of them this --vendor's (PLUGIN, the one the
# receipt must show loaded). PLUGIN_BASES / PLUGIN_PATHS / PLUGIN_PAIRS name
# them all for the box and the receipt check.
PLUGIN_PATHS=""; PLUGIN_BASES=""; PLUGIN_PAIRS=""
case "$VENDOR" in cuda) _pfx=mojolearn_nvidia ;; *) _pfx=mojolearn_amd ;; esac
for _p in ${PLUGINS[@]+"${PLUGINS[@]}"}; do
    [ -f "$_p" ] || die "no plugin wheel $_p"
    _p=$(cd "$(dirname "$_p")" && pwd)/$(basename "$_p")
    _b=$(basename "$_p")
    case "$_b" in
        "mojolearn_nvidia-$VERSION-"*-manylinux*_x86_64.whl|"mojolearn_amd-$VERSION-"*-manylinux*_x86_64.whl) ;;
        *) die "$_b is not a mojolearn_nvidia/mojolearn_amd $VERSION manylinux x86_64 plugin (the $_pfx plugin of --vendor $VENDOR is required)" ;;
    esac
    case " $PLUGIN_BASES " in *" ${_b%%-*}-"*) die "two --plugin wheels of ${_b%%-*}" ;; esac
    _s=$(shasum -a 256 "$_p" | cut -d' ' -f1)
    PLUGIN_PATHS="$PLUGIN_PATHS $_p"; PLUGIN_BASES="$PLUGIN_BASES $_b"; PLUGIN_PAIRS="$PLUGIN_PAIRS,$_b=$_s"
    case "$_b" in "$_pfx-"*) PLUGIN=$_p; PLUGIN_SHA=$_s ;; esac
done
PLUGIN_BASES=${PLUGIN_BASES# }; PLUGIN_PATHS=${PLUGIN_PATHS# }; PLUGIN_PAIRS=${PLUGIN_PAIRS#,}
[ "${#PLUGINS[@]}" = 0 ] || [ -n "$PLUGIN" ] \
    || die "no --plugin is the $_pfx $VERSION manylinux x86_64 plugin of --vendor $VENDOR"
PLUGIN_BASE=""; [ -z "$PLUGIN" ] || PLUGIN_BASE=$(basename "$PLUGIN")
# the box-side arguments: every plugin, from the run directory
PLUGIN_BOX_ARGS=""; PLUGIN_BOX_PATHS=""
for _b in $PLUGIN_BASES; do
    PLUGIN_BOX_ARGS="$PLUGIN_BOX_ARGS --plugin $RDIR/$_b"; PLUGIN_BOX_PATHS="$PLUGIN_BOX_PATHS $RDIR/$_b"
done
QUALIFY_SHA=$(shasum -a 256 "$QUALIFY" | cut -d' ' -f1)
STAMP=$(date -u +%Y%m%d-%H%M%S)
POD_NAME="mojolearn-smoke-$(printf '%s' "$VERSION" | tr -c 'a-z0-9\n' '-')-$STAMP"
[ -n "$OUT" ] || OUT="${MOJOLEARN_EVIDENCE_ROOT:-$HOME/mojolearn-evidence}/release-smoke/$VERSION/$STAMP-linux"
[ ! -e "$OUT/results.json" ] || die "$OUT/results.json exists; use a fresh --out"
CREATE="$TMPD/create.json"
python3 - "$CREATE" "$POD_NAME" "$IMAGE" "$GPU" "$CUDA" "$VENDOR" "$ROOT/tools/runpod_ssh_bootstrap.sh" <<'PY' || die "the create request did not compose"
import json, sys
from pathlib import Path
out, name, image, gpu, cuda, vendor, bootstrap = sys.argv[1:]
req = {"name": name, "imageName": image, "gpuTypeIds": [gpu], "gpuCount": 1,
       "cloudType": "SECURE", "containerDiskInGb": 30, "volumeInGb": 0,
       "ports": ["22/tcp"], "supportPublicIp": True, "interruptible": False}
if cuda:
    req["allowedCudaVersions"] = [v.strip() for v in cuda.split(",") if v.strip()]
if vendor == "hip" and image.startswith("rocm/"):
    # plain ROCm images have no ssh; the repo's bootstrap (as tools/gemm_remote_leg.sh)
    req["dockerEntrypoint"] = ["/bin/bash", "-lc"]
    req["dockerStartCmd"] = [Path(bootstrap).read_text()]
open(out, "w").write(json.dumps(req, indent=2) + "\n")
PY
do_create_json() {  # region -> the DigitalOcean create request on stdout
    printf '{"name":"%s","region":"%s","size":"%s","image":%s,"ssh_keys":["%s"],"tags":["%s"]}\n' \
        "$POD_NAME" "$1" "$DO_SIZE" "$DO_IMAGE" "$DO_SSH_KEY_FP" "$DO_TAG"
}
DO_FIRST_REGION=${DO_REGIONS%%,*}

# The box-side command. It picks the newest python3.1x the image has, runs the
# smoke bounded, and writes the exit code last so the poll can see it.
BOX="$TMPD/box.sh"
cat > "$BOX" <<BOX_EOF
#!/bin/bash
set -u
cd $RDIR || exit 9
{ nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || rocm-smi --showproductname 2>/dev/null; uname -a; } > box.txt 2>&1
PY=\$(command -v python3.13 || command -v python3.12 || command -v python3.11 || command -v python3.10 || command -v python3)
echo "python=\$PY \$(\$PY --version 2>&1)" >> box.txt
sha256sum "$(basename "$WHEEL")" $PLUGIN_BASES qualify_verifier_wheel.py >> box.txt
if [ "$RUN_SMOKE" = 1 ]; then
timeout -k 20 $SMOKE_SECONDS "\$PY" qualify_verifier_wheel.py "$RDIR/$(basename "$WHEEL")"$PLUGIN_BOX_ARGS \\
    --scope expanded --python "\$PY" --expected-source-commit $COMMIT --output $RDIR/out > smoke.log 2>&1
echo \$? > smoke.exit
fi
if [ -n "$LANES" ]; then
  # The release column from the INSTALLED wheel, run outside any checkout.
  # A fresh DigitalOcean image may still hold the apt lock from cloud-init.
  "\$PY" -m venv $RDIR/rv > column_venv.log 2>&1 || { (apt-get -o DPkg::Lock::Timeout=120 update -qq && apt-get -o DPkg::Lock::Timeout=120 install -y -qq python3-venv) >> column_venv.log 2>&1 && "\$PY" -m venv $RDIR/rv >> column_venv.log 2>&1; }
  $RDIR/rv/bin/pip install --disable-pip-version-check "$RDIR/$(basename "$WHEEL")"$PLUGIN_BOX_PATHS numpy >> column_venv.log 2>&1
  echo "install_exit=\$?" > column.txt
  mkdir -p $RDIR/run && cd $RDIR/run
  export MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMMIT=$COMMIT
  $RDIR/rv/bin/python -c 'import mojolearn as m; print("version", m.__version__, "vendor", m.vendor())' >> $RDIR/column.txt 2>&1
  $RDIR/rv/bin/python -m mojolearn verify --self-test > $RDIR/selftest.log 2>&1; echo "selftest_exit=\$?" >> $RDIR/column.txt
  timeout -k 20 $SMOKE_SECONDS $RDIR/rv/bin/python -m mojolearn._identity_break --lanes "$LANES" --json $RDIR/column.json \\
      --repeats 1 --fixtures base,denormal,odd --fail-on-refused --require-backend $VENDOR --no-batch --no-rlpair > $RDIR/column.log 2>&1
  echo \$? > $RDIR/column.exit
  cd $RDIR
fi
echo done > $RDIR/box.done
BOX_EOF
bash -n "$BOX" || die "the box command is not valid bash"

# ---------------------------------------------------------------- DigitalOcean primitives
do_token_hygiene() {  # a reason on stdout and 1 when the file must not be used (as tools/do_extra_leg.sh)
    [ -f "$DO_TOKFILE" ] || { echo "token file $DO_TOKFILE does not exist"; return 1; }
    _p=$(stat -f '%OLp' "$DO_TOKFILE" 2>/dev/null || stat -c '%a' "$DO_TOKFILE" 2>/dev/null || echo '?')
    [ "$_p" = 600 ] || { echo "token file $DO_TOKFILE is mode $_p, must be 600"; return 1; }
    case "$(cd "$(dirname "$DO_TOKFILE")" && pwd)/" in "$ROOT"/*) echo "token file $DO_TOKFILE is INSIDE the repository"; return 1 ;; esac
    if git -C "$ROOT" ls-files --error-unmatch "$DO_TOKFILE" > /dev/null 2>&1; then
        echo "token file $DO_TOKFILE is TRACKED BY GIT"; return 1
    fi
    return 0
}
do_load_token() {  # read by a builtin, written by a builtin: no process sees the token
    _t=""
    IFS= read -r _t < "$DO_TOKFILE" || [ -n "$_t" ] || return 1
    _t="${_t//[$'\t\r\n ']/}"
    [ -n "$_t" ] || return 1
    ( umask 077; printf 'header = "Authorization: Bearer %s"\nsilent\nshow-error\n' "$_t" > "$DO_CURLRC" )
    unset _t
    return 0
}
do_call() {  # METHOD URL [json file]; sets DO_CODE, body in $TMPD/do.body
    : > "$TMPD/do.body"
    if [ -n "${3:-}" ]; then
        DO_CODE=$(curl -K "$DO_CURLRC" --max-time 60 -o "$TMPD/do.body" -w '%{http_code}' -X "$1" \
            -H 'Content-Type: application/json' --data-binary "@$3" "$2" 2>>"$TMPD/curl.err") || DO_CODE=000
    else
        DO_CODE=$(curl -K "$DO_CURLRC" --max-time 60 -o "$TMPD/do.body" -w '%{http_code}' -X "$1" "$2" 2>>"$TMPD/curl.err") || DO_CODE=000
    fi
}
do_py() {  # python over the parsed DigitalOcean body
    python3 - "$TMPD/do.body" "$@" <<'PYEOF' 2>/dev/null || true
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    sys.exit(0)
what = sys.argv[2]
droplets = d.get("droplets") or []
if what == "id":
    print((d.get("droplet") or {}).get("id") or "")
elif what == "state":  # status and the public IPv4 of one droplet
    x = d.get("droplet") or {}
    ips = [n.get("ip_address") for n in (x.get("networks") or {}).get("v4", []) if n.get("type") == "public"]
    print(x.get("status") or "unknown", ips[0] if ips else "")
elif what == "byname":
    print(" ".join(str(x.get("id")) for x in droplets if x.get("name") == sys.argv[3]))
elif what == "names":
    for x in droplets:
        print("%s\t%s\t%s\t%s\t%s" % (x.get("id"), x.get("name"), x.get("status"),
                                      (x.get("size") or {}).get("slug"), (x.get("region") or {}).get("slug")))
elif what == "gpu_live":
    print("yes" if any(str((x.get("size") or {}).get("slug") or "").startswith("gpu-") for x in droplets) else "no")
elif what == "size":  # price_hourly, regions and availability of one size
    for s in d.get("sizes") or []:
        if s.get("slug") == sys.argv[3]:
            print(s.get("price_hourly") or "", "available" if s.get("available") else "unavailable",
                  " ".join(s.get("regions") or []))
            break
PYEOF
}
do_write_deadman() {  # dir seconds; composes and checks, never arms
    ( umask 077; mkdir -p "$1"; if [ -f "$DO_CURLRC" ]; then cp "$DO_CURLRC" "$1/curlrc"; else : > "$1/curlrc"; fi )
    cat > "$1/deadman.sh" <<'DM_EOF'
#!/bin/sh
# tools/release_wheel_smoke.sh's Mac dead-man for a DigitalOcean droplet: ends
# the droplet if the runner is gone. Keyed by tag AND name, plus the id once the
# runner learned it (the worst case is a create that succeeded and an id nobody
# parsed). The token is in the 0600 curl config beside this file, in no argv.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
sleep @SECS@
L="$D/deadman.log"
echo "$(date -u +%FT%TZ) dead-man firing for tag @TAG@ name @NAME@" >> "$L"
ids=""
[ -s "$D/droplet_id.txt" ] && ids="$(cat "$D/droplet_id.txt")"
curl -K "$D/curlrc" --max-time 30 -o "$D/droplets.json" "@API@/droplets?tag_name=@TAG@&per_page=200" >> "$L" 2>&1
ids="$ids $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(" ".join(str(x["id"]) for x in d.get("droplets", []) if x.get("name")==sys.argv[2]))' "$D/droplets.json" "@NAME@" 2>/dev/null)"
for id in $ids; do
    c="$(curl -K "$D/curlrc" --max-time 30 -o /dev/null -w '%{http_code}' -X DELETE "@API@/droplets/$id" 2>>"$L")"
    echo "$(date -u +%FT%TZ) DELETE $id -> $c" >> "$L"
done
rm -f "$D/curlrc"
DM_EOF
    sed -i.bak -e "s|@SECS@|$2|g" -e "s|@NAME@|$POD_NAME|g" -e "s|@TAG@|$DO_TAG|g" -e "s|@API@|$DO_API|g" "$1/deadman.sh"
    rm -f "$1/deadman.sh.bak"
    if grep -q '@[A-Z0-9]*@' "$1/deadman.sh"; then return 1; fi
    sh -n "$1/deadman.sh"
}
do_write_selfkill() {  # file id seconds; the ON-DROPLET self-destruct, composed and checked
    cat > "$1" <<'SK_EOF'
#!/bin/sh
# tools/release_wheel_smoke.sh's ON-DROPLET dead-man. RUNS ON THE DROPLET,
# DETACHED: the Mac dead-man dies with the Mac; this one does not. The token is
# in the 0600 curl config beside this file (delivered on ssh stdin), in no argv.
set -u
D="$(cd "$(dirname "$0")" && pwd)"
sleep @SECS@
for attempt in 1 2 3; do
    code=$(curl -K "$D/curlrc" --max-time 30 -o "$D/selfkill.body" -w '%{http_code}' -X DELETE '@API@/droplets/@ID@')
    echo "$(date -u +%FT%TZ) DELETE @ID@ attempt $attempt -> $code" >> "$D/selfkill.out"
    case "$code" in 2*|404) break ;; esac
    sleep 10
done
SK_EOF
    sed -i.bak -e "s|@SECS@|$3|g" -e "s|@ID@|$2|g" -e "s|@API@|$DO_API|g" "$1"
    rm -f "$1.bak"
    if grep -q '@[A-Z0-9]*@' "$1"; then return 1; fi
    sh -n "$1"
}
do_lock_age() {  # seconds since the lock directory was made; 1 when it is absent
    _m=$(stat -f %m "$DO_GPU_LOCK" 2>/dev/null || stat -c %Y "$DO_GPU_LOCK" 2>/dev/null) || return 1
    echo $(( $(now) - _m ))
}
do_take_lock() {  # mkdir is the atomic test-and-set; the owner file names the run
    mkdir "$DO_GPU_LOCK" 2>/dev/null || return 1
    DO_LOCK_HELD=1
    { echo "lane=release-smoke:$VERSION"; echo "script=tools/release_wheel_smoke.sh"; echo "pid=$$"
      echo "nonce=$DO_LOCK_NONCE"; echo "droplet_name=$POD_NAME"; echo "utc=$(date -u +%FT%TZ)"; echo "out=$OUT"
    } > "$DO_GPU_LOCK/owner"
}
do_release_lock() {  # only after the destroy is confirmed, and only when the owner file still carries our nonce
    [ "$DO_LOCK_HELD" = 1 ] || return 0
    if grep -qx "nonce=$DO_LOCK_NONCE" "$DO_GPU_LOCK/owner" 2>/dev/null; then
        rm -rf "$DO_GPU_LOCK" && echo "  released the shared GPU lock $DO_GPU_LOCK"
    else
        echo "  !! $DO_GPU_LOCK no longer carries this run's nonce; left in place"
    fi
    DO_LOCK_HELD=0
}
do_destroy() {  # DROPLET_ID, or a sweep by tag and name; DO_GONE=1 only on a GET 404
    _ids="$DROPLET_ID"
    if [ -z "$_ids" ]; then
        do_call GET "$DO_API/droplets?tag_name=$DO_TAG&per_page=200"
        [ "$DO_CODE" = 200 ] || { echo "  could not list droplets (HTTP $DO_CODE) to find $POD_NAME; destruction UNCONFIRMED"; return 1; }
        _ids=$(do_py byname "$POD_NAME")
        [ -n "$_ids" ] || { echo "  no droplet tagged $DO_TAG named $POD_NAME (listing HTTP 200)"; DO_GONE=1; return 0; }
    fi
    _all=1
    for _id in $_ids; do
        for _i in 1 2 3 4 5 6; do
            do_call DELETE "$DO_API/droplets/$_id"
            echo "  DELETE droplet $_id -> HTTP $DO_CODE"
            case "$DO_CODE" in 204|404) break ;; esac
            sleep 10
        done
        _g=0
        for _i in 1 2 3 4 5 6 7 8; do  # DELETE 204 acknowledges; only GET 404 proves absence
            do_call GET "$DO_API/droplets/$_id"
            echo "  GET droplet $_id -> HTTP $DO_CODE (attempt $_i/8)"
            [ "$DO_CODE" = 404 ] && { _g=1; break; }
            sleep 5
        done
        [ "$_g" = 1 ] || _all=0
    done
    if [ "$_all" = 1 ]; then DO_GONE=1; echo "  VERIFIED: droplet $_ids is gone (GET 404)"; fi
    return 0
}

echo "== release_wheel_smoke: $([ -n "$SSH_GIVEN" ] && echo "EXISTING BOX $SSH_GIVEN" || { [ "$RENT" = 1 ] && echo RENT || echo 'DRY RUN'; }) =="
echo "  wheel    $(basename "$WHEEL")  sha256 $WHEEL_SHA  commit $COMMIT"
[ -z "$PLUGIN" ] || echo "  plugins  $PLUGIN_BASES (installed with the core; this vendor's: $PLUGIN_BASE sha256 $PLUGIN_SHA)"
echo "  ships    the wheel + tools/qualify_verifier_wheel.py (sha256 $(printf %s "$QUALIFY_SHA" | cut -c1-16)...) and nothing else"
echo "  smoke    qualify_verifier_wheel.py --scope expanded, bounded ${SMOKE_SECONDS}s"
echo "  out      $OUT"
if [ -z "$SSH_GIVEN" ]; then
    case "$PROVIDER" in
        auto) echo "  provider auto: RunPod once, Hot Aisle when RunPod has no '$GPU' to give, DigitalOcean when Hot Aisle refuses before a create" ;;
        runpod) echo "  provider runpod" ;;
        hotaisle) echo "  provider hotaisle (Hot Aisle only)" ;;
        do) echo "  provider do (DigitalOcean only)" ;;
    esac
    if [ "$PROVIDER" = runpod ] || [ "$PROVIDER" = auto ]; then
        echo "  pod      $POD_NAME  gpu '$GPU'  image $IMAGE  cuda [$CUDA]  lease ${LEASE}m"
        echo "  create   $(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))' "$CREATE")"
        write_deadman "$TMPD/dm-check" 60 || die "the dead-man script did not compose"
        rm -rf "$TMPD/dm-check"
        echo "  dead-man composes (sh -n)"
    fi
    if [ "$PROVIDER" = hotaisle ] || [ "$PROVIDER" = auto ]; then
        echo "  hotaisle MI300X VM spec $HA_SPEC_WANT, team $HA_TEAM, lease ${LEASE}m, horizon $(( LEASE + HA_READY_SECONDS / 60 + 10 ))m priced live, cap \$$HA_CAP_USD, box commands as root via sudo -n bash -c"
        ha_write_deadman "$TMPD/ha-dm-check" "$(( $(now) + 60 ))" /dev/null || die "the Hot Aisle dead-man did not compose"
        ha_write_watchdog "$TMPD/ha-dm-check/watchdog.sh" "${RDIR}-guard" 60 DRYRUN_REF || die "the Hot Aisle on-box watchdog did not compose"
        rm -rf "$TMPD/ha-dm-check"
        echo "  Hot Aisle dead-men compose (Mac dead-man sh -n; on-box watchdog sh -n)"
    fi
    if [ "$PROVIDER" = do ] || [ "$PROVIDER" = auto ]; then
        echo "  droplet  $POD_NAME  size $DO_SIZE  regions $DO_REGIONS (in order)  image $DO_IMAGE  tag $DO_TAG  ssh key $DO_SSH_KEY_FP  lease ${LEASE}m"
        echo "  create   $(do_create_json "$DO_FIRST_REGION")"
        do_write_deadman "$TMPD/do-dm-check" 60 || die "the DigitalOcean dead-man did not compose"
        do_write_selfkill "$TMPD/do-dm-check/selfkill.sh" 0 60 || die "the on-droplet self-destruct did not compose"
        rm -rf "$TMPD/do-dm-check"
        echo "  dead-men compose (Mac dead-man sh -n; on-droplet self-destruct sh -n)"
        if _age=$(do_lock_age); then
            echo "  lock     $DO_GPU_LOCK is HELD (${_age}s old) by: $(tr '\n' ' ' < "$DO_GPU_LOCK/owner" 2>/dev/null || echo 'no owner file'); a rent refuses unless it is over ${DO_LOCK_STALE_SECONDS}s old with no GPU droplet live"
        else
            echo "  lock     $DO_GPU_LOCK is free (a rent takes it; a dry run never does)"
        fi
    fi
fi
if [ "$RENT" = 0 ] && [ -z "$SSH_GIVEN" ]; then
    if [ "$PROVIDER" = runpod ] || [ "$PROVIDER" = auto ]; then
        if load_key; then
            rp_call GET "$RP/pods"
            echo "  pod listing HTTP $RP_CODE; mojolearn-smoke pods live:"
            rp_py names | awk -F'\t' '$2 ~ /^mojolearn-smoke-/ {print "    " $0}'
        else
            echo "  no RunPod key here; a rent would $([ "$PROVIDER" = auto ] && echo 'go straight to Hot Aisle' || echo refuse)"
        fi
    fi
    if [ "$PROVIDER" = hotaisle ] || [ "$PROVIDER" = auto ]; then
        if _why=$(ha_key_hygiene) && ha_load_key; then
            echo "  key      $HA_KEYFILE present, 0600, outside the repository"
            ha_call GET "teams/$HA_TEAM/balance/"
            echo "  balance  HTTP $HA_CODE; $(ha_dollars "$(ha_py balance)")"
            ha_call GET "teams/$HA_TEAM/virtual_machines/available/"
            echo "  stock    HTTP $HA_CODE:"; ha_py offers | sed 's/^/    /'
            read -r _f _spec _q _p _m _c <<< "$(ha_py pick "$HA_SPEC_WANT" "$TMPD/ha_pick.json")"
            echo "  pick     spec $HA_SPEC_WANT -> $_f $_spec quantity $_q ${_p} cents/h min_reservation $_m min"
            ha_call GET "teams/$HA_TEAM/virtual_machines/"
            echo "  VMs      HTTP $HA_CODE; $(ha_py count) live on the team"
        else
            echo "  no Hot Aisle key here (${_why:-empty key file}); a rent would $([ "$PROVIDER" = auto ] && echo 'go on to DigitalOcean' || echo refuse)"
        fi
    fi
    if [ "$PROVIDER" = do ] || [ "$PROVIDER" = auto ]; then
        if _why=$(do_token_hygiene) && do_load_token; then
            echo "  token    $DO_TOKFILE present, 0600, outside the repository"
            do_call GET "$DO_API/sizes?per_page=200"
            read -r _price _avail _regions <<< "$(do_py size "$DO_SIZE")"
            echo "  sizes    HTTP $DO_CODE; $DO_SIZE \$${_price:-?}/h ${_avail:-unlisted}, offered in: ${_regions:-none}"
            do_call GET "$DO_API/droplets?tag_name=$DO_TAG&per_page=200"
            echo "  droplet listing HTTP $DO_CODE; droplets tagged $DO_TAG live:"
            do_py names | sed 's/^/    /'
        else
            echo "  no DigitalOcean token here (${_why:-empty token file}); a rent would $([ "$PROVIDER" = auto ] && echo 'have no fallback' || echo refuse)"
        fi
    fi
    echo
    echo "DRY RUN: nothing was created and nothing was billed. Add --rent to create the box."
    rm -rf "$TMPD"
    exit 0
fi

# ---------------------------------------------------------------- the run
POD_ID=""; POD_TERMINATED=0; DEADMAN_PID=""; DEADMAN_DIR=""; SSH_TARGET="$SSH_GIVEN"; COST_HR=""; T_POST=""
DROPLET_ID=""; DO_CREATE_ATTEMPTED=0; DO_GONE=0; DO_DEADMAN_PID=""; DO_DEADMAN_DIR=""; DO_LOCK_HELD=0
DO_LOCK_NONCE="$$-$STAMP"; DO_COST_HR=""; DO_T_POST=""; DO_REGION=""; PROVIDER_USED=""
bssh() {  # shellcheck disable=SC2086
    ssh $SSH_OPTS $SSH_TARGET "$@"
}
teardown() {
    _rc=$?
    trap - EXIT INT TERM
    if [ -n "$POD_ID" ]; then
        echo; echo "== teardown (exit $_rc) =="
        delete_pod "$POD_ID"
        if verify_gone "$POD_ID"; then POD_TERMINATED=1; fi
        _t=$(now)
        { echo "pod=$POD_ID"; echo "name=$POD_NAME"; echo "terminated_verified=$POD_TERMINATED"; echo "exit=$_rc"
          echo "at=$(date -u +%FT%TZ)"
          [ -n "$COST_HR" ] && [ -n "$T_POST" ] && python3 -c "import sys; print('spend=\$%.4f at \$%s/hr, %d s' % (float(sys.argv[1]) * (int(sys.argv[2]) - int(sys.argv[3])) / 3600, sys.argv[1], int(sys.argv[2]) - int(sys.argv[3])))" "$COST_HR" "$_t" "$T_POST"
        } >> "$OUT/teardown.txt"
        sed 's/^/  /' "$OUT/teardown.txt"
    fi
    if [ "$DO_CREATE_ATTEMPTED" = 1 ]; then
        echo; echo "== teardown (exit $_rc): DigitalOcean =="
        do_destroy
        _t=$(now)
        { echo "droplet=${DROPLET_ID:-unknown}"; echo "name=$POD_NAME"; echo "region=$DO_REGION"; echo "destroy_confirmed=$DO_GONE"
          echo "exit=$_rc"; echo "at=$(date -u +%FT%TZ)"
          [ -n "$DO_COST_HR" ] && [ -n "$DO_T_POST" ] && python3 -c "import sys; print('spend=\$%.4f at \$%s/hr, %d s' % (float(sys.argv[1]) * (int(sys.argv[2]) - int(sys.argv[3])) / 3600, sys.argv[1], int(sys.argv[2]) - int(sys.argv[3])))" "$DO_COST_HR" "$_t" "$DO_T_POST"
        } >> "$OUT/teardown.txt"
        sed 's/^/  /' "$OUT/teardown.txt"
    fi
    if [ "$HA_CREATE_ATTEMPTED" = 1 ] || [ -n "$HA_DEADMAN_PID" ] || [ -n "$HA_SLOT" ]; then
        echo; echo "== teardown (exit $_rc): Hot Aisle =="
        HA_RECORD="$OUT/teardown.txt"
        ha_teardown || { [ "$_rc" = 0 ] && _rc=1; }
        _hb=""
        if [ -n "$HA_CURLRC" ]; then ha_call GET "teams/$HA_TEAM/balance/"; _hb=$(ha_py balance); fi
        { echo "provider=hotaisle"; echo "vm=${HA_VMREF:-none}"; echo "name=${HA_VMNAME:-none}"; echo "spec=${HA_SPEC_USED:-none}"
          echo "destroy_confirmed=$HA_GONE"; echo "exit=$_rc"; echo "at=$(date -u +%FT%TZ)"
          echo "balance_before_cents=${HA_BAL_BEFORE:-unknown} balance_after_cents=${_hb:-unknown}"; ha_spend
        } >> "$OUT/teardown.txt"
        sed 's/^/  /' "$OUT/teardown.txt"
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
            echo "  # End it by hand:  sh tools/runpod_guard.sh reap --force $POD_ID"
            echo "  ##########################################################"
        fi
    fi
    if [ -n "$DO_DEADMAN_PID" ] || [ "$DO_LOCK_HELD" = 1 ]; then
        if [ "$DO_CREATE_ATTEMPTED" = 0 ] || [ "$DO_GONE" = 1 ]; then
            if [ -n "$DO_DEADMAN_PID" ]; then
                pkill -P "$DO_DEADMAN_PID" 2>/dev/null || true
                kill "$DO_DEADMAN_PID" 2>/dev/null && echo "  DigitalOcean dead-man cancelled (pid $DO_DEADMAN_PID)"
                rm -rf "$DO_DEADMAN_DIR"
            fi
            do_release_lock
        else
            echo "  ##########################################################"
            echo "  # DROPLET ${DROPLET_ID:-<unknown id> named $POD_NAME} MAY STILL BE BILLING."
            echo "  # The API did not confirm it is gone. BOTH dead-men stay armed"
            echo "  # (Mac pid ${DO_DEADMAN_PID:-none}, dir ${DO_DEADMAN_DIR:-none}; the on-droplet"
            echo "  # one fires by id at the lease). Destroy it by hand now:"
            echo "  #   https://cloud.digitalocean.com/droplets"
            echo "  # and only then: kill ${DO_DEADMAN_PID:-<pid>}; rm -rf ${DO_DEADMAN_DIR:-<dir>}"
            [ "$DO_LOCK_HELD" = 1 ] && echo "  # The shared GPU lock stays HELD until then: rm -rf $DO_GPU_LOCK"
            echo "  ##########################################################"
            [ "$_rc" = 0 ] && _rc=1
        fi
    fi
    rm -rf "$TMPD"
    exit "$_rc"
}

mkdir -p "$OUT" || die "cannot create $OUT"
[ -n "$SSH_GIVEN" ] && trap 'rm -rf "$TMPD"' EXIT
{ echo "wheel=$WHEEL"; echo "wheel_sha256=$WHEEL_SHA"; echo "commit=$COMMIT"; echo "qualify_sha256=$QUALIFY_SHA"
  [ -z "$PLUGIN" ] || { echo "plugin=$PLUGIN"; echo "plugin_sha256=$PLUGIN_SHA"; echo "plugins=$PLUGIN_PAIRS"; }
  echo "target=${SSH_GIVEN:-rented, provider $PROVIDER}"; echo "started=$(date -u +%FT%TZ)"; } > "$OUT/smoke.txt"
cp "$BOX" "$OUT/box.sh"

rent_runpod() {  # sets POD_ID and SSH_TARGET; 1 (nothing created) when RunPod had no stock and --provider auto
    cp "$CREATE" "$OUT/create_request.json"
    rp_call GET "$RP/pods"
    case "$RP_CODE" in 2*) ;; *) die "pod listing HTTP $RP_CODE; a runner that cannot list cannot verify a delete" ;; esac
    DEADMAN_DIR="${TMPDIR:-/tmp}/mojolearn-smoke-deadman-$$"
    _dm_secs=$(( READY_TIMEOUT + LEASE * 60 + 600 ))
    write_deadman "$DEADMAN_DIR" "$_dm_secs" || die "the dead-man did not compose; nothing created"
    nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$DEADMAN_DIR/deadman.sh" > /dev/null 2>&1 < /dev/null &
    DEADMAN_PID=$!
    sleep 1
    kill -0 "$DEADMAN_PID" 2>/dev/null || { DEADMAN_PID=""; die "the dead-man did not start; nothing created"; }
    say "dead-man ARMED before the create: pid $DEADMAN_PID, fires in ${_dm_secs}s, keyed by $POD_NAME"

    say "creating $POD_NAME ($GPU) on RunPod. THE BILL STARTS HERE."
    T_POST=$(now)
    rp_call POST "$RP/pods" "$CREATE"
    _ccode=$RP_CODE     # the listing below overwrites RP_CODE
    cp "$TMPD/rp.body" "$OUT/create_response.json"
    POD_ID=$(rp_py id)
    if [ -z "$POD_ID" ]; then
        _body=$(head -c 300 "$OUT/create_response.json")
        rp_call GET "$RP/pods"
        POD_ID=$(rp_py byname "$POD_NAME" | awk '{print $1}')
        [ -n "$POD_ID" ] && { printf '%s\n' "$POD_ID" > "$DEADMAN_DIR/pod_id.txt"; die "create response unparsed but $POD_ID exists by name; tearing it down"; }
        if [ "$PROVIDER" = auto ] && printf '%s' "$_body" | grep -qi 'no instances currently available'; then
            # THE FALLBACK DECISION (0.8.16, 2026-09-23). RunPod answered HTTP 200 with
            # {"error":"create pod: There are no instances currently available"}: no
            # MI300X to give, nothing created (the listing has no pod by this name).
            say "RunPod has no '$GPU' to give (HTTP $_ccode: $_body); nothing was created"
            { echo "runpod=no_stock http=$_ccode at=$(date -u +%FT%TZ)"; echo "runpod_body=$_body"; } >> "$OUT/provider.txt"
            pkill -P "$DEADMAN_PID" 2>/dev/null || true
            kill "$DEADMAN_PID" 2>/dev/null && say "RunPod dead-man cancelled (pid $DEADMAN_PID); nothing to guard"
            rm -rf "$DEADMAN_DIR"; DEADMAN_PID=""; DEADMAN_DIR=""; T_POST=""
            say "RunPod created nothing; FALLING BACK to the next provider (Hot Aisle, then DigitalOcean; gfx942 either way)"
            return 1
        fi
        die "create FAILED (HTTP $_ccode): $_body"
    fi
    printf '%s\n' "$POD_ID" > "$DEADMAN_DIR/pod_id.txt"
    printf '%s\n' "$POD_ID" > "$OUT/pod_id.txt"
    cp "$OUT/create_response.json" "$TMPD/rp.body"
    COST_HR=$(rp_py cost)
    say "pod $POD_ID created. COST \$${COST_HR:-?}/hr"
    _deadline=$(( $(now) + READY_TIMEOUT ))
    SSH_TARGET=""
    while [ "$(now)" -lt "$_deadline" ]; do
        rp_call GET "$RP/pods/$POD_ID"
        [ -n "$COST_HR" ] || COST_HR=$(rp_py cost)
        SSH_TARGET=$(rp_py ssh)
        if [ -n "$SSH_TARGET" ] && with_timeout 40 ssh $SSH_OPTS $SSH_TARGET 'echo SSH-OK' 2>/dev/null | grep -q SSH-OK; then break; fi
        SSH_TARGET=""
        sleep 10
    done
    [ -n "$SSH_TARGET" ] || die "READY TIMEOUT: no ssh after ${READY_TIMEOUT}s"
    say "ssh up ($SSH_TARGET) after $(( $(now) - T_POST ))s"
    say "arming the ON-POD watchdog ($LEASE minutes)"
    _hp=$(printf '%s' "$SSH_TARGET" | awk '{print "[" substr($3, index($3, "@") + 1) "]:" $2}')
    ssh-keygen -R "$_hp" > /dev/null 2>&1 || true
    if ! MOJOLEARN_LEASE_DIR="$OUT/lease" with_timeout 180 sh "$ROOT/tools/runpod_guard.sh" arm "$POD_ID" "$SSH_TARGET" "$LEASE" > "$OUT/arm.log" 2>&1; then
        sed 's/^/    /' "$OUT/arm.log"; die "ARM REFUSED; the pod is not used"
    fi
    with_timeout 60 ssh $SSH_OPTS $SSH_TARGET "p=\$(cat /tmp/mojolearn-lease.pid 2>/dev/null); kill -0 \"\$p\" 2>/dev/null && echo WATCHDOG_ALIVE pid=\$p; curl -s --max-time 20 -o /dev/null -w 'TOKEN_GET_%{http_code}\n' -K /tmp/mojolearn-lease.curlrc $RP/pods/$POD_ID" > "$OUT/watchdog_check.txt" 2>&1
    sed 's/^/    /' "$OUT/watchdog_check.txt"
    grep -q WATCHDOG_ALIVE "$OUT/watchdog_check.txt" && grep -q TOKEN_GET_200 "$OUT/watchdog_check.txt" \
        || die "the on-pod watchdog is not alive or its token does not answer 200"
    PROVIDER_USED=runpod
    echo "provider=runpod pod=$POD_ID gpu=$GPU" >> "$OUT/provider.txt"
}

rent_do() {  # sets DROPLET_ID and SSH_TARGET, or dies with nothing left billing
    say "renting from DigitalOcean: $DO_SIZE in $DO_REGIONS (image $DO_IMAGE)"
    _why=$(do_token_hygiene) || die "no usable DigitalOcean token: $_why"
    do_load_token || die "the DigitalOcean token file $DO_TOKFILE is empty"
    # The account must answer before anything is created, and the lock must be free.
    do_call GET "$DO_API/droplets?per_page=200"
    [ "$DO_CODE" = 200 ] || die "DigitalOcean droplet listing HTTP $DO_CODE; a runner that cannot list cannot verify a destroy"
    _live=$(do_py gpu_live)
    if _age=$(do_lock_age); then
        _owner=$(tr '\n' ' ' < "$DO_GPU_LOCK/owner" 2>/dev/null)
        if [ "$_age" -gt "$DO_LOCK_STALE_SECONDS" ] && [ "$_live" = no ]; then
            say "breaking the STALE shared GPU lock (${_age}s old, no GPU droplet live): ${_owner:-no owner file}"
            echo "gpu_lock_broken_stale=age ${_age}s owner ${_owner:-none}" >> "$OUT/provider.txt"
            rm -rf "$DO_GPU_LOCK"
        else
            die "REFUSING to create $POD_NAME: the shared GPU lock $DO_GPU_LOCK is held (${_age}s old) by: ${_owner:-no owner file}. One GPU droplet at a time across every session (CONTRIBUTING.md). Nothing was created."
        fi
    fi
    do_take_lock || die "REFUSING to rent: another run took $DO_GPU_LOCK first. Nothing was created."
    say "shared GPU lock taken ($DO_GPU_LOCK)"
    echo "gpu_lock=taken $(date -u +%FT%TZ) $DO_GPU_LOCK" >> "$OUT/provider.txt"
    do_call GET "$DO_API/sizes?per_page=200"
    read -r DO_COST_HR _ <<< "$(do_py size "$DO_SIZE")"

    DO_DEADMAN_DIR="${TMPDIR:-/tmp}/mojolearn-smoke-do-deadman-$$"
    _dm_secs=$(( READY_TIMEOUT + LEASE * 60 + 600 ))
    do_write_deadman "$DO_DEADMAN_DIR" "$_dm_secs" || die "the DigitalOcean dead-man did not compose; nothing created"
    nohup sh -c 'trap "" HUP INT; exec sh "$0"' "$DO_DEADMAN_DIR/deadman.sh" > /dev/null 2>&1 < /dev/null &
    DO_DEADMAN_PID=$!
    sleep 1
    kill -0 "$DO_DEADMAN_PID" 2>/dev/null || { DO_DEADMAN_PID=""; die "the DigitalOcean dead-man did not start; nothing created"; }
    say "dead-man ARMED before the create: pid $DO_DEADMAN_PID, fires in ${_dm_secs}s, keyed by tag $DO_TAG + $POD_NAME"

    DO_CREATE_ATTEMPTED=1
    for _r in $(printf '%s' "$DO_REGIONS" | tr ',' ' '); do
        DO_REGION=$_r
        do_create_json "$_r" > "$TMPD/do-create.json"
        cp "$TMPD/do-create.json" "$OUT/create_request-do-$_r.json"
        say "creating $POD_NAME ($DO_SIZE, $_r, image $DO_IMAGE) on DigitalOcean. THE BILL STARTS HERE."
        DO_T_POST=$(now)
        do_call POST "$DO_API/droplets" "$TMPD/do-create.json"
        _ccode=$DO_CODE     # the listing below overwrites DO_CODE
        cp "$TMPD/do.body" "$OUT/create_response-do-$_r.json"
        DROPLET_ID=$(do_py id)
        [ -n "$DROPLET_ID" ] && break
        _body=$(head -c 300 "$OUT/create_response-do-$_r.json")
        sleep 5
        do_call GET "$DO_API/droplets?tag_name=$DO_TAG&per_page=200"
        [ "$DO_CODE" = 200 ] || die "create in $_r answered no id ($_body) and the listing failed (HTTP $DO_CODE); the teardown sweeps by name"
        DROPLET_ID=$(do_py byname "$POD_NAME" | awk '{print $1}')
        if [ -n "$DROPLET_ID" ]; then
            say "ADOPTED droplet $DROPLET_ID found by name after an unreadable create response"
            echo "adopted_by_name=1" >> "$OUT/provider.txt"
            break
        fi
        DO_T_POST=""
        case "$_ccode" in
            422) say "$_r refused $DO_SIZE (HTTP 422: $_body); nothing was created; trying the next region" ;;
            *) die "create FAILED in $_r (HTTP $_ccode): $_body" ;;
        esac
    done
    [ -n "$DROPLET_ID" ] || die "no region in $DO_REGIONS would create $DO_SIZE; nothing was created"
    case "$DROPLET_ID" in *[!0-9]*) die "the droplet id '$DROPLET_ID' is not numeric" ;; esac
    printf '%s\n' "$DROPLET_ID" > "$DO_DEADMAN_DIR/droplet_id.txt"
    printf '%s\n' "$DROPLET_ID" > "$OUT/droplet_id.txt"
    say "droplet $DROPLET_ID created in $DO_REGION. COST \$${DO_COST_HR:-?}/hr"
    _deadline=$(( $(now) + READY_TIMEOUT ))
    IP=""
    while [ "$(now)" -lt "$_deadline" ]; do
        do_call GET "$DO_API/droplets/$DROPLET_ID"
        read -r _st IP <<< "$(do_py state)"
        [ "$_st" = active ] && [ -n "$IP" ] && break
        IP=""
        sleep 10
    done
    [ -n "$IP" ] || die "READY TIMEOUT: the droplet was not active with a public IPv4 after ${READY_TIMEOUT}s"
    say "active at $IP after $(( $(now) - DO_T_POST ))s"
    SSH_TARGET="root@$IP"
    # Three CONSECUTIVE successes: a droplet answers once and resets while sshd settles.
    _ok=0
    while [ "$(now)" -lt "$_deadline" ]; do
        if with_timeout 40 ssh $SSH_OPTS $SSH_TARGET true 2>/dev/null; then
            _ok=$((_ok + 1)); [ "$_ok" -ge 3 ] && break
        else
            _ok=0
        fi
        sleep 5
    done
    [ "$_ok" -ge 3 ] || die "READY TIMEOUT: ssh never settled on $IP"
    say "ssh settled ($SSH_TARGET) after $(( $(now) - DO_T_POST ))s"

    say "arming the ON-DROPLET self-destruct ($LEASE minutes)"
    do_write_selfkill "$TMPD/selfkill.sh" "$DROPLET_ID" $(( LEASE * 60 )) || die "the on-droplet self-destruct did not compose; destroying unused"
    cp "$TMPD/selfkill.sh" "$OUT/droplet_deadman.sh"
    with_timeout 60 ssh $SSH_OPTS $SSH_TARGET "umask 077; mkdir -p $GUARD && cat > $GUARD/curlrc && chmod 600 $GUARD/curlrc" < "$DO_CURLRC" \
        || die "could not deliver the token for the self-destruct; destroying unused"
    with_timeout 60 ssh $SSH_OPTS $SSH_TARGET "umask 077; cat > $GUARD/selfkill.sh && chmod 700 $GUARD/selfkill.sh" < "$TMPD/selfkill.sh" \
        || die "could not deliver the self-destruct; destroying unused"
    with_timeout 60 ssh $SSH_OPTS $SSH_TARGET "nohup sh $GUARD/selfkill.sh > $GUARD/selfkill.log 2>&1 < /dev/null &
echo \$! > $GUARD/selfkill.pid; sleep 1
p=\$(cat $GUARD/selfkill.pid); kill -0 \"\$p\" 2>/dev/null && echo WATCHDOG_ALIVE pid=\$p
echo ID_BAKED_IN=\$(grep -c 'droplets/$DROPLET_ID' $GUARD/selfkill.sh)
curl -s --max-time 20 -o /dev/null -w 'TOKEN_GET_%{http_code}\n' -K $GUARD/curlrc $DO_API/droplets/$DROPLET_ID" > "$OUT/watchdog_check.txt" 2>&1
    sed 's/^/    /' "$OUT/watchdog_check.txt"
    grep -q WATCHDOG_ALIVE "$OUT/watchdog_check.txt" && grep -q '^ID_BAKED_IN=[1-9]' "$OUT/watchdog_check.txt" && grep -q TOKEN_GET_200 "$OUT/watchdog_check.txt" \
        || die "the on-droplet self-destruct is not alive, has no id baked in, or its token does not answer 200; destroying unused"
    { echo "provider=do droplet=$DROPLET_ID size=$DO_SIZE region=$DO_REGION image=$DO_IMAGE ip=$IP"
      echo "on_droplet_deadman_seconds=$(( LEASE * 60 ))"; echo "local_deadman_pid=$DO_DEADMAN_PID seconds=$_dm_secs"; } >> "$OUT/provider.txt"
    PROVIDER_USED=do
}

rent_hotaisle() {  # sets SSH_TARGET and BOX_SUDO; 1 when NOTHING was created (HA_REFUSED says why)
    say "renting from Hot Aisle: MI300X VM spec $HA_SPEC_WANT, lease ${LEASE}m, cap \$$HA_CAP_USD"
    ha_rent "release-smoke-$(printf '%s' "$VERSION" | tr -c 'A-Za-z0-9.\n' '-')" "$LEASE" "$HA_CAP_CENTS" \
        "$OUT/provider.txt" "${RDIR}-guard" "$OUT" || return 1
    SSH_TARGET="$HA_TARGET"; BOX_SUDO=1
    # The release column is a one-GPU column: on the 2x VM it runs on GPU 0 only.
    [ "$HA_SPEC_USED" = 2gpu ] && BOX_ENV="ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0"
    echo "provider=hotaisle vm=$HA_VMREF name=$HA_VMNAME spec=$HA_SPEC_USED ip=$HA_SSH_IP port=$HA_SSH_PORT price_cents_per_hour=$HA_PRICE gfx=$HA_GFX${BOX_ENV:+ pin=$BOX_ENV}" >> "$OUT/provider.txt"
    PROVIDER_USED=hotaisle
}

if [ -z "$SSH_GIVEN" ]; then
    trap teardown EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    if [ "$PROVIDER" = runpod ] || [ "$PROVIDER" = auto ]; then
        if load_key; then
            rent_runpod || true
        elif [ "$PROVIDER" = runpod ]; then
            die "no RunPod key (MOJOLEARN_RUNPOD_KEY_FILE or ~/.mojolearn_runpod_key)"
        else
            say "no RunPod key here (MOJOLEARN_RUNPOD_KEY_FILE or ~/.mojolearn_runpod_key); auto goes on to Hot Aisle"
            echo "runpod=no_key" >> "$OUT/provider.txt"
        fi
    fi
    if [ -z "$SSH_TARGET" ] && { [ "$PROVIDER" = hotaisle ] || [ "$PROVIDER" = auto ]; }; then
        if ! rent_hotaisle; then
            [ "$PROVIDER" = auto ] || die "Hot Aisle: $HA_REFUSED"
            say "Hot Aisle created nothing ($HA_REFUSED); FALLING BACK to DigitalOcean ($DO_SIZE, gfx942)"
        fi
    fi
    [ -n "$SSH_TARGET" ] || rent_do
    echo "rented=$PROVIDER_USED target=$SSH_TARGET" >> "$OUT/smoke.txt"
fi

# THE BOX FLOW, the same on every provider. bx runs one command on the box as
# root: directly on RunPod, DigitalOcean and --ssh, through `sudo -n bash -c`
# on Hot Aisle (whose login is the hotaisle user). stdin passes through.
bx() {  # <seconds> <command>
    _bs=$1; shift
    if [ "$BOX_SUDO" = 1 ]; then
        # shellcheck disable=SC2086
        with_timeout "$_bs" ssh $SSH_OPTS $SSH_TARGET "$(ha_root_cmd "$1")"
    else
        # shellcheck disable=SC2086
        with_timeout "$_bs" ssh $SSH_OPTS $SSH_TARGET "$1"
    fi
}
BOX_START="nohup bash $RDIR/box.sh > $RDIR/box.log 2>&1 < /dev/null & echo STARTED"
if [ "$BOX_SUDO" = 1 ]; then
    # detached from the ssh session and sudo, as tools/hotaisle_leg.sh starts its body
    BOX_START="cd / && if command -v setsid > /dev/null 2>&1; then S=setsid; else S=; fi; ${BOX_ENV:+env $BOX_ENV }\$S nohup bash $RDIR/box.sh > $RDIR/box.log 2>&1 < /dev/null & echo STARTED"
    echo "box_sudo=1${BOX_ENV:+ box_env=$BOX_ENV}" >> "$OUT/smoke.txt"
fi

# Uploads: two files, each bounded, each hashed on the box.
_wb=$(wc -c < "$WHEEL" | tr -d ' ')
_up_secs=$(( 120 + _wb / 200000 ))     # 200 kB/s floor on the Mac's uplink
say "uploading the wheel ($_wb bytes, bound ${_up_secs}s) and the smoke driver"
bx 60 "rm -rf $RDIR && mkdir -p $RDIR" < /dev/null || die "could not prepare $RDIR"
bx "$_up_secs" "cat > $RDIR/$(basename "$WHEEL")" < "$WHEEL" || die "wheel upload failed or exceeded ${_up_secs}s"
for _p in $PLUGIN_PATHS; do
    _pb=$(wc -c < "$_p" | tr -d ' ')
    say "uploading the plugin $(basename "$_p") ($_pb bytes)"
    bx $(( 120 + _pb / 200000 )) "cat > $RDIR/$(basename "$_p")" < "$_p" || die "plugin upload failed or exceeded its bound"
done
bx 60 "cat > $RDIR/qualify_verifier_wheel.py" < "$QUALIFY" || die "driver upload failed"
bx 60 "cat > $RDIR/box.sh" < "$BOX" || die "box command upload failed"
_remote=$(bx 120 "cd $RDIR && sha256sum $(basename "$WHEEL") $PLUGIN_BASES qualify_verifier_wheel.py" < /dev/null 2>&1)
printf '%s\n' "$_remote" | grep -q "^$WHEEL_SHA " || die "wheel sha256 differs on the box: $_remote"
for _pair in $(printf '%s' "$PLUGIN_PAIRS" | tr ',' ' '); do
    printf '%s\n' "$_remote" | grep -q "^${_pair##*=} " || die "plugin ${_pair%%=*} sha256 differs on the box: $_remote"
done
printf '%s\n' "$_remote" | grep -q "^$QUALIFY_SHA " || die "driver sha256 differs on the box: $_remote"
say "both files landed, sha256 verified on the box"

say "starting the smoke (detached; bound ${SMOKE_SECONDS}s)"
bx 60 "$BOX_START" < /dev/null | grep -q STARTED \
    || die "the smoke did not start"
_end=$(( $(now) + SMOKE_SECONDS + 120 )); _fails=0; SMOKE_EXIT=""
while [ "$(now)" -lt "$_end" ]; do
    sleep 15
    _o=$(bx 45 "cat $RDIR/box.done 2>/dev/null; true" < /dev/null 2>/dev/null)
    if [ $? = 0 ]; then _fails=0; else _fails=$((_fails + 1)); fi
    [ -n "$(printf '%s' "$_o" | tr -d '[:space:]')" ] && break
    [ "$_fails" -lt 12 ] || { say "12 polls failed in a row; fetching what exists"; break; }
done
SMOKE_EXIT=$(bx 45 "cat $RDIR/smoke.exit 2>/dev/null; true" < /dev/null 2>/dev/null | tr -d '[:space:]')
say "smoke exit: ${SMOKE_EXIT:-none}"

say "fetching the results"
mkdir -p "$OUT/remote"
bx 300 "cd $RDIR && tar czf - box.txt box.log smoke.log smoke.exit out column.txt column_venv.log selftest.log column.log column.exit column.json column.json.errors.txt 2>/dev/null" < /dev/null \
    | ( cd "$OUT/remote" && tar xzf - ) || echo "  FETCH INCOMPLETE"
[ -f "$OUT/remote/box.txt" ] && sed 's/^/  box: /' "$OUT/remote/box.txt"
if [ -f "$OUT/remote/out/results.json" ]; then
    cp "$OUT/remote/out/results.json" "$OUT/results.json"
fi
echo "finished=$(date -u +%FT%TZ) smoke_exit=${SMOKE_EXIT:-none}" >> "$OUT/smoke.txt"

# The receipt, judged here: about THIS wheel, THIS commit, and PASSED.
_v=0
if [ "$RUN_SMOKE" = 1 ]; then
python3 - "$OUT/results.json" "$WHEEL_SHA" "$COMMIT" "$VENDOR" "$PLUGIN_PAIRS" <<'PY' | tee -a "$OUT/smoke.txt"
import json, sys
path, sha, commit, want_vendor, pairs = sys.argv[1:]
try:
    d = json.load(open(path))
except Exception as exc:
    print('verdict=FAILED no results.json (%s)' % exc); sys.exit(1)
problems = []
if d.get('status') != 'PASSED': problems.append('status %s reason %s' % (d.get('status'), d.get('reason')))
if d.get('wheel_sha256') != sha: problems.append('receipt is about wheel %s' % d.get('wheel_sha256'))
if d.get('source_commit') != commit: problems.append('receipt names commit %s' % d.get('source_commit'))
if d.get('scope') != 'expanded': problems.append('scope %s' % d.get('scope'))
vendor = (d.get('installed') or {}).get('vendor')
if vendor != want_vendor: problems.append('installed vendor %s, not %s' % (vendor, want_vendor))
got = sorted((p.get('wheel', '').rsplit('/', 1)[-1], p.get('wheel_sha256')) for p in d.get('plugins') or [])
want = sorted(tuple(pair.split('=', 1)) for pair in pairs.split(',') if pair)
if got != want: problems.append('receipt plugins %s, want %s' % (got, want))
jobs = d.get('jobs') or []
print('verdict=%s jobs=%d vendor=%s%s' % ('PASSED' if not problems else 'FAILED', len(jobs), vendor,
      '' if not problems else ' ' + '; '.join(problems)))
sys.exit(1 if problems else 0)
PY
_v=${PIPESTATUS[0]}
fi
# The column, judged here: it ran to the end on this vendor, and (with
# --ref-column) no cell differs from any reference column of the same commit.
if [ -n "$LANES" ]; then
    [ -f "$OUT/remote/column.txt" ] && sed 's/^/  column: /' "$OUT/remote/column.txt"
    _cx=$(tr -d '[:space:]' < "$OUT/remote/column.exit" 2>/dev/null)
    [ -f "$OUT/remote/column.json" ] && cp "$OUT/remote/column.json" "$OUT/column-$VENDOR.json"
    if [ "$_cx" != 0 ] || [ ! -f "$OUT/remote/column.json" ]; then
        say "COLUMN FAILED (exit ${_cx:-none}); refused cells and their errors:"
        grep -E '^# CELL .* REFUSED' "$OUT/remote/column.log" 2>/dev/null | head -20
        [ -f "$OUT/remote/column.json.errors.txt" ] && head -40 "$OUT/remote/column.json.errors.txt"
        _v=1
    else
        cp "$OUT/remote/column.json" "$OUT/column-$VENDOR.json"
        say "column complete: $OUT/column-$VENDOR.json"
        if [ "${#REFS[@]}" -gt 0 ]; then
            python3 "$ROOT/tools/identity_break.py" --diff "${REFS[@]}" "$OUT/column-$VENDOR.json" > "$OUT/diff-ref-$VENDOR.txt" 2>&1
            _dv=$(grep -c 'DIVERGENT' "$OUT/diff-ref-$VENDOR.txt" || true)
            grep -E '^summary' "$OUT/diff-ref-$VENDOR.txt" | sed 's/^/  /'
            if ! grep -q '^summary' "$OUT/diff-ref-$VENDOR.txt"; then
                say "the reference diff printed no summary:"; tail -20 "$OUT/diff-ref-$VENDOR.txt"
                _v=1
            elif [ "${_dv:-0}" != 0 ]; then
                say "DIVERGENT against the reference column(s) ${REFS[*]}:"; grep 'DIVERGENT' "$OUT/diff-ref-$VENDOR.txt" | head -40
                _v=1
            else
                say "no DIVERGENT cell against the reference column(s) ($OUT/diff-ref-$VENDOR.txt)"
            fi
        fi
    fi
fi
[ "$_v" = 0 ] && say "PASSED. Evidence: $OUT" || say "FAILED. Logs: $OUT/remote"
exit "$_v"
