#!/bin/bash
# tools/release_wheel_smoke.sh -- the Linux light-route smoke of ONE exact final
# wheel on ONE rented CUDA pod, and nothing else (2026-09-22).
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
#   --lease MIN          on-pod self-delete after MIN minutes (default 45, 10..90)
#   --smoke-seconds N    bound on the smoke itself (default 1800; 0.8.13 took 50 s on an H100)
#   --ssh TARGET         run on an existing box (e.g. "-p 22 root@1.2.3.4"): nothing
#                        is rented or deleted. The final wheel is packed on the Mac
#                        from all three legs, so no build box holds it; use this for
#                        a box you already have up.
#   --rent               create the pod. Without it (and without --ssh): a dry run.
#   --vendor cuda|hip    the GPU family (default cuda). hip rents a RunPod AMD
#                        Instinct MI300X (gfx942, the wheel's AMD set) on
#                        rocm/dev-ubuntu-22.04:6.4.1-complete with the repo's ssh
#                        bootstrap; the expanded smoke runs on cuda only.
#   --column SELECTION   also run the release pass's cells for the lanes in
#                        SELECTION (a verify_lanes --write-selection file for this
#                        vendor) from the INSTALLED wheel: fixtures base,denormal,odd,
#                        one fit, --fail-on-refused, --require-backend <vendor>.
#                        column.json comes home.
#   --cpu-column FILE    with --column: diff the GPU column against this CPU
#                        column of the same commit (tools/identity_break.py --diff);
#                        any DIVERGENT cell fails the run, named lane/fixture/part.
#
# A RENTED RUN, IN ORDER: Mac dead-man armed BEFORE the create (lease + ready
# timeout + 10 min, by id or by name); create; wait for ssh (600 s); arm the
# ON-POD watchdog (tools/runpod_guard.sh arm) and read it back; upload the two
# files (sha256 checked on the box); run the smoke detached and poll; fetch
# the output directory; DELETE and verify gone. Teardown runs on EVERY exit.
# The receipt is then checked here: status PASSED, the wheel's sha256 and the
# expected source commit.
set -u

ROOT=$(cd "$(dirname "$0")/.." && pwd)
READY_TIMEOUT=600
# The box-side directory. Overridable only so tools/test_release_wheel_smoke.py
# can drive the whole --ssh path against a local shim; never a production knob.
RDIR=${MOJOLEARN_SMOKE_REMOTE_DIR:-/root/wheel-smoke}
printf '%s' "$RDIR" | grep -Eq '^/[A-Za-z0-9/_.-]*/wheel-smoke$' || { echo "bad MOJOLEARN_SMOKE_REMOTE_DIR" >&2; exit 2; }
SSH_OPTS="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=4 -o BatchMode=yes"

WHEEL=""; COMMIT=""; OUT=""; GPU="${MOJOLEARN_SMOKE_GPU:-NVIDIA GeForce RTX 4090}"
IMAGE="${MOJOLEARN_SMOKE_IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
CUDA="13.0"; LEASE=45; SMOKE_SECONDS=1800; SSH_GIVEN=""; RENT=0
VENDOR=cuda; SELECTION=""; CPU_COLUMN=""; GPU_SET=0; IMAGE_SET=0; CUDA_SET=0

say() { printf '[%s wheel-smoke] %s\n' "$(date +%T)" "$*"; }
die() { printf '\nREFUSED: %s\n' "$*" >&2; exit 1; }
now() { date +%s; }

TMPD=$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-wheel-smoke.XXXXXX")
CURLRC="$TMPD/rp.curlrc"
POD_NAME=""
# shellcheck source=tools/runpod_pod_lib.sh
. "$ROOT/tools/runpod_pod_lib.sh"

while [ $# -gt 0 ]; do
    case "$1" in
        --expected-source-commit) shift; COMMIT="${1:-}" ;;
        --out) shift; OUT="${1:-}" ;;
        --gpu) shift; GPU="${1:-}"; GPU_SET=1 ;;
        --image) shift; IMAGE="${1:-}"; IMAGE_SET=1 ;;
        --cuda) shift; CUDA="${1:-}"; CUDA_SET=1 ;;
        --vendor) shift; VENDOR="${1:-}" ;;
        --column) shift; SELECTION="${1:-}" ;;
        --cpu-column) shift; CPU_COLUMN="${1:-}" ;;
        --lease) shift; LEASE="${1:-}" ;;
        --smoke-seconds) shift; SMOKE_SECONDS="${1:-}" ;;
        --ssh) shift; SSH_GIVEN="${1:-}" ;;
        --rent) RENT=1 ;;
        -h|--help) sed -n '2,45p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
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
[ "$VENDOR" = cuda ] || [ -n "$SELECTION" ] || die "--vendor hip runs no smoke; give --column"
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
[ -z "$CPU_COLUMN" ] || { [ -n "$SELECTION" ] || die "--cpu-column needs --column"; [ -f "$CPU_COLUMN" ] || die "no CPU column $CPU_COLUMN"; }
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
sha256sum "$(basename "$WHEEL")" qualify_verifier_wheel.py >> box.txt
if [ "$VENDOR" = cuda ]; then
timeout -k 20 $SMOKE_SECONDS "\$PY" qualify_verifier_wheel.py "$RDIR/$(basename "$WHEEL")" \\
    --scope expanded --python "\$PY" --expected-source-commit $COMMIT --output $RDIR/out > smoke.log 2>&1
echo \$? > smoke.exit
fi
if [ -n "$LANES" ]; then
  # The release column from the INSTALLED wheel, run outside any checkout.
  "\$PY" -m venv $RDIR/rv > column_venv.log 2>&1 || { (apt-get update -qq && apt-get install -y -qq python3-venv) >> column_venv.log 2>&1 && "\$PY" -m venv $RDIR/rv >> column_venv.log 2>&1; }
  $RDIR/rv/bin/pip install --disable-pip-version-check "$RDIR/$(basename "$WHEEL")" numpy >> column_venv.log 2>&1
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

echo "== release_wheel_smoke: $([ -n "$SSH_GIVEN" ] && echo "EXISTING BOX $SSH_GIVEN" || { [ "$RENT" = 1 ] && echo RENT || echo 'DRY RUN'; }) =="
echo "  wheel    $(basename "$WHEEL")  sha256 $WHEEL_SHA  commit $COMMIT"
echo "  ships    the wheel + tools/qualify_verifier_wheel.py (sha256 $(printf %s "$QUALIFY_SHA" | cut -c1-16)...) and nothing else"
echo "  smoke    qualify_verifier_wheel.py --scope expanded, bounded ${SMOKE_SECONDS}s"
echo "  out      $OUT"
if [ -z "$SSH_GIVEN" ]; then
    echo "  pod      $POD_NAME  gpu '$GPU'  image $IMAGE  cuda [$CUDA]  lease ${LEASE}m"
    echo "  create   $(python3 -c 'import json,sys; print(json.dumps(json.load(open(sys.argv[1]))))' "$CREATE")"
    write_deadman "$TMPD/dm-check" 60 || die "the dead-man script did not compose"
    rm -rf "$TMPD/dm-check"
    echo "  dead-man composes (sh -n)"
fi
if [ "$RENT" = 0 ] && [ -z "$SSH_GIVEN" ]; then
    if load_key; then
        rp_call GET "$RP/pods"
        echo "  pod listing HTTP $RP_CODE; mojolearn-smoke pods live:"
        rp_py names | awk -F'\t' '$2 ~ /^mojolearn-smoke-/ {print "    " $0}'
    else
        echo "  no RunPod key here; a rent would refuse"
    fi
    echo
    echo "DRY RUN: nothing was created and nothing was billed. Add --rent to create the pod."
    rm -rf "$TMPD"
    exit 0
fi

# ---------------------------------------------------------------- the run
POD_ID=""; POD_TERMINATED=0; DEADMAN_PID=""; DEADMAN_DIR=""; SSH_TARGET="$SSH_GIVEN"; COST_HR=""; T_POST=""
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
    rm -rf "$TMPD"
    exit "$_rc"
}

mkdir -p "$OUT" || die "cannot create $OUT"
[ -n "$SSH_GIVEN" ] && trap 'rm -rf "$TMPD"' EXIT
{ echo "wheel=$WHEEL"; echo "wheel_sha256=$WHEEL_SHA"; echo "commit=$COMMIT"; echo "qualify_sha256=$QUALIFY_SHA"
  echo "target=${SSH_GIVEN:-rented $GPU}"; echo "started=$(date -u +%FT%TZ)"; } > "$OUT/smoke.txt"
cp "$BOX" "$OUT/box.sh"

if [ -z "$SSH_GIVEN" ]; then
    load_key || die "no RunPod key (MOJOLEARN_RUNPOD_KEY_FILE or ~/.mojolearn_runpod_key)"
    trap teardown EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
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

    say "creating $POD_NAME ($GPU). THE BILL STARTS HERE."
    T_POST=$(now)
    rp_call POST "$RP/pods" "$CREATE"
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
fi

# Uploads: two files, each bounded, each hashed on the box.
_wb=$(wc -c < "$WHEEL" | tr -d ' ')
_up_secs=$(( 120 + _wb / 200000 ))     # 200 kB/s floor on the Mac's uplink
say "uploading the wheel ($_wb bytes, bound ${_up_secs}s) and the smoke driver"
with_timeout 60 ssh $SSH_OPTS $SSH_TARGET "rm -rf $RDIR && mkdir -p $RDIR" || die "could not prepare $RDIR"
with_timeout "$_up_secs" ssh $SSH_OPTS $SSH_TARGET "cat > $RDIR/$(basename "$WHEEL")" < "$WHEEL" || die "wheel upload failed or exceeded ${_up_secs}s"
with_timeout 60 ssh $SSH_OPTS $SSH_TARGET "cat > $RDIR/qualify_verifier_wheel.py" < "$QUALIFY" || die "driver upload failed"
with_timeout 60 ssh $SSH_OPTS $SSH_TARGET "cat > $RDIR/box.sh" < "$BOX" || die "box command upload failed"
_remote=$(with_timeout 120 ssh $SSH_OPTS $SSH_TARGET "cd $RDIR && sha256sum $(basename "$WHEEL") qualify_verifier_wheel.py" 2>&1)
printf '%s\n' "$_remote" | grep -q "^$WHEEL_SHA " || die "wheel sha256 differs on the box: $_remote"
printf '%s\n' "$_remote" | grep -q "^$QUALIFY_SHA " || die "driver sha256 differs on the box: $_remote"
say "both files landed, sha256 verified on the box"

say "starting the smoke (detached; bound ${SMOKE_SECONDS}s)"
with_timeout 60 ssh $SSH_OPTS $SSH_TARGET "nohup bash $RDIR/box.sh > $RDIR/box.log 2>&1 < /dev/null & echo STARTED" | grep -q STARTED \
    || die "the smoke did not start"
_end=$(( $(now) + SMOKE_SECONDS + 120 )); _fails=0; SMOKE_EXIT=""
while [ "$(now)" -lt "$_end" ]; do
    sleep 15
    _o=$(with_timeout 45 ssh $SSH_OPTS $SSH_TARGET "cat $RDIR/box.done 2>/dev/null; true" 2>/dev/null)
    if [ $? = 0 ]; then _fails=0; else _fails=$((_fails + 1)); fi
    [ -n "$(printf '%s' "$_o" | tr -d '[:space:]')" ] && break
    [ "$_fails" -lt 12 ] || { say "12 polls failed in a row; fetching what exists"; break; }
done
SMOKE_EXIT=$(with_timeout 45 ssh $SSH_OPTS $SSH_TARGET "cat $RDIR/smoke.exit 2>/dev/null; true" 2>/dev/null | tr -d '[:space:]')
say "smoke exit: ${SMOKE_EXIT:-none}"

say "fetching the results"
mkdir -p "$OUT/remote"
with_timeout 300 ssh $SSH_OPTS $SSH_TARGET "cd $RDIR && tar czf - box.txt box.log smoke.log smoke.exit out column.txt column_venv.log selftest.log column.log column.exit column.json column.json.errors.txt 2>/dev/null" \
    | ( cd "$OUT/remote" && tar xzf - ) || echo "  FETCH INCOMPLETE"
[ -f "$OUT/remote/box.txt" ] && sed 's/^/  box: /' "$OUT/remote/box.txt"
if [ -f "$OUT/remote/out/results.json" ]; then
    cp "$OUT/remote/out/results.json" "$OUT/results.json"
fi
echo "finished=$(date -u +%FT%TZ) smoke_exit=${SMOKE_EXIT:-none}" >> "$OUT/smoke.txt"

# The receipt, judged here: about THIS wheel, THIS commit, and PASSED.
_v=0
if [ "$VENDOR" = cuda ]; then
python3 - "$OUT/results.json" "$WHEEL_SHA" "$COMMIT" <<'PY' | tee -a "$OUT/smoke.txt"
import json, sys
path, sha, commit = sys.argv[1:]
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
if vendor != 'cuda': problems.append('installed vendor %s, not cuda' % vendor)
jobs = d.get('jobs') or []
print('verdict=%s jobs=%d vendor=%s%s' % ('PASSED' if not problems else 'FAILED', len(jobs), vendor,
      '' if not problems else ' ' + '; '.join(problems)))
sys.exit(1 if problems else 0)
PY
_v=${PIPESTATUS[0]}
fi
# The column, judged here: it ran to the end on this vendor, and (with
# --cpu-column) no cell differs from the CPU column of the same commit.
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
        if [ -n "$CPU_COLUMN" ]; then
            python3 "$ROOT/tools/identity_break.py" --diff "$CPU_COLUMN" "$OUT/column-$VENDOR.json" > "$OUT/diff-cpu-$VENDOR.txt" 2>&1
            _dv=$(grep -c 'DIVERGENT' "$OUT/diff-cpu-$VENDOR.txt" || true)
            grep -E '^summary' "$OUT/diff-cpu-$VENDOR.txt" | sed 's/^/  /'
            if [ "${_dv:-0}" != 0 ]; then
                say "DIVERGENT against the CPU column:"; grep 'DIVERGENT' "$OUT/diff-cpu-$VENDOR.txt" | head -40
                _v=1
            else
                say "no DIVERGENT cell against the CPU column ($OUT/diff-cpu-$VENDOR.txt)"
            fi
        fi
    fi
fi
[ "$_v" = 0 ] && say "PASSED. Evidence: $OUT" || say "FAILED. Logs: $OUT/remote"
exit "$_v"
