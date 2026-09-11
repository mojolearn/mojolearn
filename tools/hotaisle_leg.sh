#!/usr/bin/env bash
# tools/hotaisle_leg.sh. THE ONLY WAY ANY LANE RENTS AN AMD GPU ON HOT AISLE.
# One guarded AMD MI300X VM: slot, create, tag, watchdog, ship the commit,
# pixi and the gates, the body (two bodies, one per GPU, on the 2gpu spec),
# fetch, DELETE, verify gone.
#
#   MOJOLEARN_GEMM_LEG_EXTRA=tools/attention_step_leg.sh \
#   MOJOLEARN_GEMM_LEG_OUT=bench/results/e1g/$(date -u +%Y-%m-%d_%H%M%S)-amd-mi300x-hotaisle-<lane> \
#   MOJOLEARN_HOTAISLE_EXTRA_ENV='MOJOLEARN_X=1 MODULAR_Y=2' \
#   bash tools/hotaisle_leg.sh amd [--rent | --probe] [--minutes N] [--skip-gates]
#
#   MOJOLEARN_HOTAISLE_SPEC=2gpu MOJOLEARN_HOTAISLE_GPU_ONLY=1 \
#   MOJOLEARN_GEMM_LEG_EXTRA=<GPU 0 body> MOJOLEARN_GEMM_LEG_EXTRA_B=<GPU 1 body> \
#   [MOJOLEARN_GEMM_LEG_OUT=<dir> MOJOLEARN_GEMM_LEG_OUT_B=<dir>] \
#   bash tools/hotaisle_leg.sh amd [--rent] [--skip-gates]     two bodies on one 2x MI300X VM
#   MOJOLEARN_HOTAISLE_SPEC=2gpu MOJOLEARN_HOTAISLE_GPU_ONLY=1 \
#   bash tools/hotaisle_leg.sh amd --rent --test-2gpu          the pinning test, before any real 2gpu use
#
#   bash tools/hotaisle_leg.sh status        this team's VMs, descriptions, states, slots (both legs of a 2gpu VM), balance
#   bash tools/hotaisle_leg.sh reap <vm>     DELETE ?force=true a mojolearn:* VM, verify gone
#
#   (no mode)  DRY RUN: no key read, no API call, nothing created
#   --probe    free GETs only: roles, VM limit, balance, stock and price, VMs, ssh key
#   --rent     BILLS: creates one VM
#   --spec S   the same as MOJOLEARN_HOTAISLE_SPEC=S
#
# ENVIRONMENT
#   MOJOLEARN_GEMM_LEG_EXTRA      the body (POSIX sh), run from /root/mojolearn on the VM (GPU 0 on 2gpu)
#   MOJOLEARN_GEMM_LEG_OUT        evidence dir; the VM's /root/gemm_leg_out lands in <out>/remote/
#   MOJOLEARN_HOTAISLE_EXTRA_ENV  NAME=value words for the body (MOJOLEARN_* or MODULAR_*, values [A-Za-z0-9_.,:/=-])
#   MOJOLEARN_GEMM_LEG_EXTRA_B, MOJOLEARN_GEMM_LEG_OUT_B, MOJOLEARN_HOTAISLE_EXTRA_ENV_B, MOJOLEARN_HOTAISLE_LANE_B
#                                 the GPU 1 body, its evidence dir, env and lane: 2gpu only, validated the
#                                 same way, and REFUSED on 13core or 8core rather than silently dropped
#   MOJOLEARN_HOTAISLE_GPU_ONLY   must be 1 on 2gpu: the caller's word that neither body is a CPU opponent row
#   MOJOLEARN_GPU_ARCHS           one arch; unset reads it from rocminfo on the VM (the MI300X is gfx942)
#   MOJOLEARN_HOTAISLE_KEY_FILE   default ~/.mojolearn_hotaisle_key (0600, one line)
#   MOJOLEARN_HOTAISLE_SPEC       13core (default, comparable CPU opponent rows), 8core, or 2gpu (the
#                                 2x MI300X VM, 26 cores: GPU rows only, labeled mi300x-2gpu-vm)
#   MOJOLEARN_HOTAISLE_LANE       tag in the VM description; default the body's basename
#   MOJOLEARN_HOTAISLE_RUNTIME    auto (default: docker, else podman, else native), docker, podman, native
#   MOJOLEARN_HOTAISLE_IMAGE      default rocm/dev-ubuntu-22.04:6.4.1-complete (the RunPod AMD image)
#   MOJOLEARN_HOTAISLE_TEAM       default andrews-team
# The body sees MOJOLEARN_TARGET_COLUMN=amd and MOJOLEARN_GPU_ARCHS exported,
# /root/mojolearn (git archive of HEAD, no .git), /root/gemm_leg_out, pixi on
# PATH with `pixi install` done, exactly as tools/do_extra_leg.sh provides.
# In a container runtime the body runs inside the image with /dev/kfd and
# /dev/dri passed through and host /root mounted at /root, so the paths are
# the same; the API key lives outside /root on the host and is not mounted.
#
# SAFETY, BAKED IN (none of it is optional)
#   a. SLOTS. At most min(3, the team's maximum_virtual_machines) VMs across
#      every session on this Mac: mkdir /tmp/mojolearn-hotaisle-slot.N, first
#      free, owner file (lane, pid, nonce, utc, VM ref). All busy: poll every
#      60 s. Released only after the delete is verified. A slot older than 100
#      minutes whose owner's VM is absent is logged as possibly broken, and
#      removed only when its owner pid is also dead.
#   b. Balance at or above 500 cents or the leg refuses by name. The chosen
#      spec must show Quantity > 0; otherwise wait and retry up to 30 minutes.
#      13core and 8core match exactly one GPU; only 2gpu picks the 2x MI300X
#      offering (exactly two MI300X GPUs), and it also needs a balance at or
#      above that offering's minimum reservation price.
#   c. A detached Mac-side dead-man armed BEFORE the create, keyed by the VM's
#      deployment_id as soon as the create returns it. At the deadline it
#      DELETEs with force and verifies, even if this script is gone.
#   d. An on-box watchdog armed before any work (setsid, root, key in a 0600
#      file under /var/lib/mojolearn-hotaisle). It sleeps to the deadline and
#      DELETEs its own VM with force. Verified by pid alive from a SECOND ssh
#      session, the ref baked in, and a GET of its own VM with the key that
#      returns 200 and this leg's description. Unverifiable means delete unused.
#   e. EXIT/INT/TERM trap: DELETE ?force=true, then poll until GET 404 or the
#      VM is absent from a 200 listing (not merely stopped). The verification
#      line is logged. Unverified: banner, dead-man left armed, slot kept.
#   f. --minutes defaults to 60 and 60 is the maximum. More is refused. On
#      2gpu 60 is also the minimum (the offering's minimum reservation).
#   g. The description is PATCHed to mojolearn:<lane>:<utc> right after the
#      create. reap, status and every delete refuse a VM whose description is
#      not mojolearn:* (the leg's own VM may also be empty if the PATCH never
#      landed, since its id came from this leg's own create response).
#   Creates are serialized by /tmp/mojolearn-hotaisle-create.lock so an
#   unreadable create response is adopted by listing diff without ambiguity.
#   The key is read by the builtin `read`, written by the builtin `printf` into
#   a 0600 curl config read with `curl -K`, reaches the VM on ssh stdin, and is
#   in no argv on either machine. Both process lists are searched for it.
#
# TEST-ONLY FLAGS (they exist to prove the guards; a real leg never uses them)
#   --test-watchdog  ships nothing. The Mac issues no delete before deadline +
#                    10 min: the Mac dead-man moves there as a backstop and the
#                    trap leaves a verified watchdog alone. PASS = the VM is
#                    verified gone within 8 minutes after the deadline.
#   --bare           ships no source, no pixi, no gates, no container; the body
#                    runs natively from an empty /root/mojolearn. For the trap test.
#   --test-2gpu      2gpu only, see below.
#
# THE 2GPU SPEC (MOJOLEARN_HOTAISLE_SPEC=2gpu). The team limit is 2 VMs, so the
# only fan-out is more GPUs per VM. The offering (probe 2026-09-11 16:05Z):
# 2x MI300X, 26 cores, 448 GB, 598 cents/h, minimum reservation 60 min. One
# such VM takes ONE slot and runs TWO bodies, with ONE Mac dead-man, ONE on-box
# watchdog and ONE verified delete keyed on deployment_id for the whole VM.
#   1. REFUSED before anything: a lease under 60 minutes; --bare,
#      --test-watchdog or runtime native; MOJOLEARN_HOTAISLE_GPU_ONLY unset;
#      a body that marks a CPU opponent row (below); both bodies on one OUT.
#      WARNING printed: both legs should need at least 40 minutes each.
#   2. The single-GPU steps, once for the VM: slot, stock (exactly two MI300X
#      GPUs in one offering), balance, dead-man, create, tag, running, ssh,
#      watchdog, key-in-ps, device probe, runtime (docker or podman), arch.
#   3. GPU MAP, read on the host into gpu_map.txt and leg.txt: KFD topology GPU
#      nodes in node order (drm_render_minor, location_id and domain give the
#      PCI address), /sys/class/drm/renderD* (driver, PCI address) and
#      rocm-smi --showbus. GPU g is the g-th KFD GPU node. When both GPUs have a
#      render node whose address agrees between KFD and DRM, pin_mode=render:
#      each container gets /dev/kfd and ONLY its renderD node. Otherwise
#      pin_mode=dri: /dev/kfd and the whole /dev/dri to both containers, the
#      pinning rests on the visible-devices variables, and leg.txt says so.
#   4. ONE bundle upload, unpacked twice into /root/leg-a/mojolearn and
#      /root/leg-b/mojolearn (each body its own writable source copy). ONE
#      image pull; a failed pull deletes the VM (no native fallback).
#   5. PIN CALIBRATION, after the pull. For GPU g, candidate containers
#      (its devices, ROCR_VISIBLE_DEVICES=r, HIP_VISIBLE_DEVICES=h) run a probe
#      that asks HIP itself (python3 ctypes into libamdhip64: hipGetDeviceCount,
#      hipGetDevicePciBusId) and rocminfo (GPU agents, BDFID). Candidates in
#      order: (g,g) the literal pin; in render mode (0,0), the view the renderD
#      restriction may leave; then (g,0), ROCR in host order with HIP relative
#      to it. The pin is the first candidate where HIP sees EXACTLY ONE device
#      at GPU g's PCI address (rocminfo decides only when HIP cannot load).
#      Both GPUs must pin, to different addresses, or the VM is deleted unused.
#   6. Each body runs in its own container: --device /dev/kfd plus its
#      devices, -e ROCR_VISIBLE_DEVICES and -e HIP_VISIBLE_DEVICES as pinned,
#      host /root/leg-a (GPU 0) or /root/leg-b (GPU 1) mounted at /root. The
#      body sees exactly the single-GPU world (/root/mojolearn,
#      /root/gemm_leg_out, /root/gemm_leg_extra.sh, pixi in /root/.pixi) under
#      its own timeout(1), sentinel and body_exit. Its leg.txt carries
#      size=mi300x-2gpu-vm, gpu_index and the pin lines.
#   7. Poll both every 30 s. A body that finishes is fetched to its own OUT at
#      once while the other keeps running; one body failing never ends the
#      other. The EXIT trap deletes the VM only when both finished or the poll
#      deadline (lease minus fetch reserve, 360 s on 2gpu) passed.
#   8. teardown.txt and deadman.txt are copied into the GPU 1 OUT. Every
#      leg.txt of a 2gpu leg (both OUTs, Mac side and box side) says
#      size=mi300x-2gpu-vm: GPU rows from a shared VM are their own tuple and
#      are never mixed with 1x MI300X VM rows without that label.
# CPU OPPONENT ROWS NEVER RUN ON 2GPU: 26 cores is not the 13-core CPU tuple.
# A body is refused when its path, or any tools/ or bench/ shell script its
# non-comment lines reach, is or names an entry of CPU_OPPONENT_DENY (the trees
# and classical opponent drivers and their Python arms), or when its env words
# carry a CPU_OPPONENT_ENV_DENY prefix. MOJOLEARN_HOTAISLE_GPU_ONLY=1 is still
# required: the scan cannot see a body that builds a script name at run time.
#
# --test-2gpu (with --rent and SPEC=2gpu; no body variables): ships no source,
# rents one 2x VM, maps and calibrates the pins, then runs two runner-written
# tiny bodies, one per pinned container, that print what they see (the pin
# probe: HIP and rocminfo; plus rocm-smi and raw rocminfo), hold about 2
# minutes, and print it again. PASS = each body saw exactly one GPU at both
# prints, at its own pinned address; the two addresses differ; the bodies
# overlapped in time and both exited 0; and the VM is verified gone.
# test_2gpu=PASS or FAIL lands in both leg.txt. It runs before any real use.
# Inside a container rocm-smi reads sysfs and still lists both GPUs; that is
# recorded, never used as the verdict.
#
# RUN OWED, 2gpu (nothing of it has run on a real VM; the orchestrator runs it):
#   1. The dry run with two bodies, GREEN, both composed commands and pins shown:
#        MOJOLEARN_HOTAISLE_SPEC=2gpu MOJOLEARN_HOTAISLE_GPU_ONLY=1 MOJOLEARN_GPU_ARCHS=gfx942 \
#        MOJOLEARN_GEMM_LEG_EXTRA=tools/gemm_longk_leg.sh \
#        MOJOLEARN_GEMM_LEG_EXTRA_B=tools/attention_round3_leg.sh \
#        bash tools/hotaisle_leg.sh amd --skip-gates
#   2. MOJOLEARN_HOTAISLE_SPEC=2gpu MOJOLEARN_HOTAISLE_GPU_ONLY=1 \
#        bash tools/hotaisle_leg.sh amd --rent --test-2gpu            -> test_2gpu=PASS
#   Only then a real 2gpu leg (step 1 with --rent).
#
# VERIFIED ON REAL VMs, 2026-09-11 (see RUNNER RESULTS below)
#
# THE HOT AISLE API (swagger https://admin.hotaisle.app/api/docs/swagger.json,
# read 2026-09-11, and live GETs)
#   Header `Authorization: Token <key>`. GET /teams/ gives effective_roles and
#   maximum_virtual_machines (2 on andrews-team). GET /teams/{t}/balance/:
#   available_balance in cents. GET .../virtual_machines/available/: Quantity,
#   OnDemandPrice (cents/hour), MinimumReservationMinutes, Specs. POST
#   .../virtual_machines/ with the Specs object: 200 VirtualMachineDetails,
#   401 team VM limit, 402 balance, 404 no stock, 428 no ssh key; provisioning
#   continues if the request is cancelled. GET/PATCH/DELETE .../{vm}/ where vm
#   is the name or deployment_id; PATCH {description} -> 204 (create has no
#   name field); DELETE ?force=true -> 204, needs operator, blocks until the
#   reset is complete. STOP KEEPS BILLING; only DELETE ends it. SSH user
#   `hotaisle`, port from ssh_access.port.
#
# RUNNER RESULTS, 2026-09-11, team andrews-team (VM limit 2, so 2 slots)
#   1. WATCHDOG (--test-watchdog, 8core, --minutes 4): VM d74538ca-d6f8-4b86-
#      a435-0e22805d863e. Watchdog fired 15:27:27Z; the Mac sent no DELETE and
#      verified GONE 15:27:36Z (GET 404, absent from the 200 listing), 26 s
#      after the deadline; Mac dead-man cancelled unfired. $0.10. PASS.
#   2. TRAP (--bare, 8core, --minutes 10): VM 88b94dcb-1bc1-4400-95c3-
#      a030ed98a679. TERM to the runner 31 s into the body (15:37:57Z); trap
#      DELETE 204 at 15:38:05Z, verified GONE 15:38:07Z. $0.10. PASS. (A first
#      attempt, VM c1fdd781, killed too late and ended by the normal teardown.)
#   3. BODY SMOKE (13core, --minutes 20, --skip-gates, tools/hotaisle_smoke_body.sh):
#      VM 86fe2678-3a9e-4e8b-8078-652e07dd5910, docker runtime, commit
#      1346e447. pixi_install_exit=0, extra_exit=0, body_exit=0; normal
#      teardown DELETE 204, verified GONE 1 s later. $0.20. PASS. (A first
#      attempt, VM 5e7aedaf, installed no pixi: a `< /dev/null` on the
#      installer's `sh` fed it nothing. Fixed; that VM was verified gone.)
#   TOTAL SPEND, all five VMs: 80 to 90 cents (2500 -> 2490 -> 4955 -> 4945
#   -> 4940 -> 4920 at the last teardown, 4910 minutes later with no VM live:
#   charges keep posting briefly after a delete; the 4955 includes a $25.00
#   top-up made outside this lane).
#   THE BOX. Ubuntu 24.04.4, kernel 6.8, amdgpu, host ROCm 7.2.4 with rocm-smi
#   and rocminfo ("AMD Instinct MI300X VF", gfx942; a bare gfx grep also hits
#   a stray "gfx9", so only the rocminfo Name: field is read). /dev/kfd and
#   /dev/dri present. Passwordless sudo for hotaisle. Docker 29.5.3 and podman
#   4.9.3 answer, so runtime auto is docker. Host python3 3.12.3; in the
#   rocm/dev-ubuntu-22.04:6.4.1-complete container 3.10.12, curl present.
#   Disk 12T (51G used). 13core = Xeon Platinum 8470, 13 cores, 224 GB;
#   8core = Xeon Platinum 8462Y+, 8 cores. CREATE returns 200 in about 5 s,
#   state running 11 to 15 s after, ssh settled 35 to 80 s after. Names are
#   RECYCLED (enc1-gpuvm015 came back for three deployments), so every
#   delete is keyed by deployment_id, never the name. DELETE ?force=true
#   answers 204 in about 7 s and GET is 404 within 3 s.
#   TIMINGS. Image pull 97 to 112 s (in the background during upload).
#   Bundle 9.65 MB over ssh stdin in 3 to 4 s. pixi install (cold, fresh VM,
#   inside the container, pixi installer included) 4 s; Mojo 1.0.0
#   (ed45d567), pixi python 3.14.7. Base binding, IDENTICAL, gfx942
#   (tools/with_identical_mode.sh sh bindings/build.sh, default
#   MOJOLEARN_COMPILE_JOBS) 56 s, exit 0, wrote
#   python/mojolearn/identical/_mojolearn.so ("gate SKIPPED by
#   MOJOLEARN_SKIP_BUILD_GATE" in its log). Create to body finished: 5.5 min.
#   DATA FROM THE VM. NYC TLC CDN yellow_tripdata_2024-01.parquet HEAD: HTTP/2
#   200, 49,961,641 bytes (reachable, unlike the DigitalOcean droplets).
#   Istella-S (tools/speed_gbdt_arm.py ISTELLA_URL) HEAD: HTTP 200,
#   472,129,615 bytes.
#   KNOWN LIMITS. Slots and the create lock live in /tmp on THIS Mac: a leg
#   launched from another machine is not counted (the team VM limit of 2
#   still refuses a third create; the spec says HTTP 401, not observed). Opponent rows from this box are a
#   new tuple (Hot Aisle MI300X VF); never mix them with MI325X or H100 rows.
#   The body runs in a ROCm 6.4.1 userland on a ROCm 7.2.4 host driver.
set -uo pipefail

# RUN FROM AN IMMUTABLE SNAPSHOT (tools/do_extra_leg.sh does the same): bash
# reads a script lazily by offset, and a leg lasts long enough for an edit.
if [ "${MOJOLEARN_HOTAISLE_FROZEN:-0}" != 1 ]; then
  _repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
  _snapdir="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-hotaisle-snapshot.XXXXXX")" || exit 2
  cp "${BASH_SOURCE[0]}" "$_snapdir/hotaisle_leg.sh" || { rm -rf "$_snapdir"; exit 2; }
  chmod 555 "$_snapdir/hotaisle_leg.sh"
  trap 'rm -rf "$_snapdir"' EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM
  MOJOLEARN_HOTAISLE_FROZEN=1 MOJOLEARN_HOTAISLE_REPO="$_repo" \
    bash "$_snapdir/hotaisle_leg.sh" "$@" &
  _child=$!
  # Forward a TERM or INT to the leg AS TERM so its own trap runs the teardown
  # (a background child of a non-interactive shell ignores INT and cannot trap it).
  trap 'kill -TERM "$_child" 2>/dev/null; wait "$_child"; exit $?' TERM
  trap 'kill -TERM "$_child" 2>/dev/null; wait "$_child"; exit $?' INT
  wait "$_child"
  exit $?
fi

REPO="${MOJOLEARN_HOTAISLE_REPO:?}"
cd "$REPO" || exit 2
API=https://admin.hotaisle.app/api
TEAM="${MOJOLEARN_HOTAISLE_TEAM:-andrews-team}"
KEYFILE="${MOJOLEARN_HOTAISLE_KEY_FILE:-$HOME/.mojolearn_hotaisle_key}"
SPEC="${MOJOLEARN_HOTAISLE_SPEC:-13core}"
RUNTIME_WANT="${MOJOLEARN_HOTAISLE_RUNTIME:-auto}"
IMAGE="${MOJOLEARN_HOTAISLE_IMAGE:-rocm/dev-ubuntu-22.04:6.4.1-complete}"
SSH_KEY="$HOME/.ssh/id_ed25519"
SSH_KEY_FP="SHA256:pqDQ15Jijc636E5M3/2IvTGR7YKL11+JzQwGYtBvtQU"
SLOT_PREFIX=/tmp/mojolearn-hotaisle-slot
CREATE_LOCK=/tmp/mojolearn-hotaisle-create.lock
MAX_SLOTS=3
MIN_BALANCE_CENTS=500
SLOT_STALE_SECONDS=6000
SLOT_WAIT_MINUTES="${MOJOLEARN_HOTAISLE_SLOT_WAIT_MINUTES:-240}"
STOCK_WAIT_MINUTES=30
FETCH_RESERVE="${FETCH_RESERVE:-240}"
MAX_BUNDLE_BYTES="${MOJOLEARN_HOTAISLE_MAX_BYTES:-15000000}"
BOX_DIR=/var/lib/mojolearn-hotaisle
BOX_RC="$BOX_DIR/curlrc"
LEG2_LABEL=mi300x-2gpu-vm
TEST2_HOLD_TICKS=12   # --test-2gpu: each tiny body holds 12 x 10 s
# THE 2GPU CPU OPPONENT DENY LIST: the trees and classical opponent drivers
# (CatBoost, scikit-learn, LightGBM, XGBoost, cuML on the CPU or beside our
# arm) and the Python arms they run. Their CPU tuple is the 13-core VM.
CPU_OPPONENT_DENY="trees_leg.sh trees_amd_leg.sh trees_amd_remote.sh trees_identical_remote.sh trees_identical_ab.sh vendor_trees_leg.sh vendor_preflight.sh local_speed_run.sh do_speed_leg.sh gbdt_accuracy_ab.sh grow_policy_ab.sh nvidia_bench.sh nvidia_forest_bench.sh knn_reference_leg.sh speed_gbdt_arm.py speed_cuml_arm.py vendor_preflight.py catboost_arm.py catboost_end2end_arm.py catboost_logloss_arm.py catboost_multiclass_arm.py catboost_reference.py knn_cuml_reference.py forest_speed_arm.py classical_ladder_arm.py nvidia_identical_trees.py rf_higgs_columns_ab.py forest_inference_ab.py"
CPU_OPPONENT_ENV_DENY="MOJOLEARN_SPEED_ MOJOLEARN_VT_ MOJOLEARN_TREES_ MOJOLEARN_LADDER_ MOJOLEARN_KNN_REF_"

usage() { sed -n '2,48p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }
log() { printf '[%s hotaisle] %s\n' "$(date +%T)" "$*"; }
die() { printf '\n%s\n' "$1" >&2; exit "${2:-1}"; }
utc() { date -u +%Y-%m-%dT%H:%M:%SZ; }
utc_of() { date -u -r "$1" +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "@$1" +%Y-%m-%dT%H:%M:%SZ; }
nap() { sleep "$1" & wait $! 2>/dev/null; }  # a sleep a trapped signal interrupts
sha256_of() { if command -v shasum > /dev/null 2>&1; then shasum -a 256 "$1"; else sha256sum "$1"; fi | awk '{print $1}'; }

# ------------------------------------------------------------------ arguments
CMD=leg; MODE=dry; MINUTES=60; GATES=1; TEST_WATCHDOG=0; BARE=0; REAP_REF=""
TEST_2GPU=0
while [ $# -gt 0 ]; do
  case "$1" in
    amd) ;;
    nv|nvidia) echo "Hot Aisle is AMD only; NVIDIA legs use RunPod" >&2; exit 2 ;;
    status) CMD=status ;;
    reap) CMD=reap; shift; REAP_REF="${1:-}" ;;
    --probe) MODE=probe ;;
    --rent) MODE=rent ;;
    --dry-run) MODE=dry ;;
    --minutes) shift; MINUTES="${1:-}" ;;
    --minutes=*) MINUTES="${1#--minutes=}" ;;
    --skip-gates) GATES=0 ;;
    --test-watchdog) TEST_WATCHDOG=1 ;;
    --bare) BARE=1 ;;
    --spec) shift; SPEC="${1:-}" ;;
    --spec=*) SPEC="${1#--spec=}" ;;
    --test-2gpu) TEST_2GPU=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument '$1'" >&2; usage >&2; exit 2 ;;
  esac
  shift
done
case "$MINUTES" in ''|*[!0-9]*) echo "--minutes must be a whole number" >&2; exit 2 ;; esac
if [ "$MINUTES" -gt 60 ]; then
  echo "--minutes $MINUTES REFUSED: 60 is the maximum lease (a second leg, never an extension)" >&2; exit 2
fi
_min=10; [ "$BARE" = 1 ] && _min=5; [ "$TEST_WATCHDOG" = 1 ] && _min=3
[ "$MINUTES" -ge "$_min" ] || { echo "--minutes must be at least $_min for this mode" >&2; exit 2; }
[ "$TEST_WATCHDOG" = 1 ] && [ "$BARE" = 1 ] && { echo "--test-watchdog and --bare are separate tests" >&2; exit 2; }
SPEC_DESC="1x MI300X"
case "$SPEC" in
  13core) SPEC_CORES=13 ;;
  8core)  SPEC_CORES=8 ;;
  2gpu)   SPEC_CORES=26; SPEC_DESC="2x MI300X" ;;
  *) echo "MOJOLEARN_HOTAISLE_SPEC='$SPEC': 13core, 8core or 2gpu (2gpu is the 2x MI300X VM: GPU rows only)" >&2; exit 2 ;;
esac
if [ "$SPEC" = 2gpu ]; then
  if [ "$MINUTES" -lt 60 ]; then
    echo "--minutes $MINUTES REFUSED for --spec 2gpu: the 2x MI300X offering bills a 60-minute minimum reservation, so a 2gpu lease is 60 minutes" >&2; exit 2
  fi
  if [ "$TEST_WATCHDOG" = 1 ] || [ "$BARE" = 1 ]; then
    echo "--test-watchdog and --bare are single-GPU tests; the 2gpu spec has --test-2gpu" >&2; exit 2
  fi
  if [ "$RUNTIME_WANT" = native ]; then
    echo "MOJOLEARN_HOTAISLE_RUNTIME=native REFUSED for --spec 2gpu: each body needs its own pinned container" >&2; exit 2
  fi
  [ "$FETCH_RESERVE" -ge 360 ] 2>/dev/null || FETCH_RESERVE=360   # two fetches
elif [ "$TEST_2GPU" = 1 ]; then
  echo "--test-2gpu needs MOJOLEARN_HOTAISLE_SPEC=2gpu (or --spec 2gpu)" >&2; exit 2
fi
case "$RUNTIME_WANT" in auto|docker|podman|native) ;; *) echo "MOJOLEARN_HOTAISLE_RUNTIME: auto, docker, podman or native" >&2; exit 2 ;; esac
case "$IMAGE" in *[!A-Za-z0-9_.:/@-]*) echo "MOJOLEARN_HOTAISLE_IMAGE: letters, digits and _.:/@- only" >&2; exit 2 ;; esac
case "$TEAM" in ''|*[!A-Za-z0-9_-]*) echo "MOJOLEARN_HOTAISLE_TEAM: letters, digits, _ and - only" >&2; exit 2 ;; esac
GPU_ARCHS="${MOJOLEARN_GPU_ARCHS:-}"
case "$GPU_ARCHS" in *[!A-Za-z0-9_]*)
  echo "MOJOLEARN_GPU_ARCHS='$GPU_ARCHS': exactly one architecture name (one mojo build is one GPU arch)" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------ state
TMPD="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-hotaisle.XXXXXX")" || exit 2
chmod 700 "$TMPD"
CURLRC=""; TOKPAT=""; OUT="$TMPD/out"
VMREF=""; VMNAME=""; DEPLOY_ID=""; SSH_IP=""; SSH_PORT=22; DESC=""
CREATE_ATTEMPTED=0; CREATE_LOCK_HELD=0; DESTROY_CONFIRMED=0
DEADMAN_PID=""; DEADMAN_DIR=""; DEADLINE_EPOCH=0; LEG_START=0
SLOT=""; NONCE="$$-$(date -u +%Y%m%dT%H%M%SZ)"; WATCHDOG_OK=0; WDT_DONE=0
KEY_RED=0; FETCH_RED=0; BODY_STATE=not_started; SSH=(ssh); SSHN=(ssh -n)
BAL_BEFORE=""
# 2gpu state: index 0 is GPU 0 (host /root/leg-a), index 1 is GPU 1 (/root/leg-b).
LETTERS=(a b); OUT_B=""; REAL_OUT_B=""; OUTS=("" ""); LANE_A=""; LANE_B=""
LEG_EXTRA_B=""; EXTRA_ENV_B=""; EXTRA_SHA_B=none
B_STATE=(pending pending); B_EXIT=(- -); B_SINCE=(0 0); B_DONE=(0 0); FETCHED=(0 0); RPIDS=("" "")
PIN_MODE=""; GPU_MAP_SOURCE=none; GPU_COUNT_SEEN=""; G_RENDER=("" ""); G_BDF=("" "")
PIN_DEVICES=("" ""); PIN_ROCR=("" ""); PIN_HIP=("" ""); PIN_BUS=("" ""); PIN_VIA=("" ""); PIN_CAND=("" "")
TEST2_BODIES=NOT_RUN

# ---- JSON helper, one file, also copied beside the dead-man ----
cat > "$TMPD/j.py" <<'PY'
import json, sys
def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None
cmd, d = sys.argv[1], load(sys.argv[2])
a = sys.argv[3:]
if cmd == "teams":
    for t in d or []:
        if t.get("handle") == a[0]:
            print("yes" if "operator" in (t.get("effective_roles") or []) else "no",
                  t.get("maximum_virtual_machines", 0))
            break
    else:
        print("absent 0")
elif cmd == "balance":
    print((d or {}).get("available_balance", -1) if isinstance(d, dict) else -1)
elif cmd == "offers":
    for e in d or []:
        s = e.get("Specs") or {}
        g = s.get("gpus") or []
        print("  %sx %s  cpu_cores=%s ram=%dGB  quantity=%s  %s cents/h  min_reservation=%s min" % (
            sum(x.get("count", 0) for x in g), ",".join(x.get("model", "?") for x in g),
            s.get("cpu_cores"), (s.get("ram_capacity") or 0) // 2**30, e.get("Quantity"),
            e.get("OnDemandPrice"), e.get("MinimumReservationMinutes")))
elif cmd == "pick":
    cores, found = int(a[0]), []
    for e in d or []:
        s = e.get("Specs") or {}
        g = s.get("gpus") or []
        if (len(g) == 1 and g[0].get("count") == 1 and g[0].get("model") == "MI300X"
                and s.get("cpu_cores") == cores):
            found.append(e)
    if not found:
        print("none 0 0 0")
    else:
        found.sort(key=lambda e: -(e.get("Quantity") or 0))
        best = found[0]
        with open(a[1], "w") as f:
            json.dump(best["Specs"], f)
        print("found", best.get("Quantity", 0), best.get("OnDemandPrice", 0),
              best.get("MinimumReservationMinutes", 0))
elif cmd == "pick2":
    found = []
    for e in d or []:
        s = e.get("Specs") or {}
        g = s.get("gpus") or []
        if (g and all(x.get("model") == "MI300X" for x in g)
                and sum((x.get("count") or 0) for x in g) == 2):
            found.append(e)
    if not found:
        print("none 0 0 0 0")
    else:
        found.sort(key=lambda e: -(e.get("Quantity") or 0))
        best = found[0]
        with open(a[0], "w") as f:
            json.dump(best["Specs"], f)
        print("found", best.get("Quantity") or 0, best.get("OnDemandPrice") or 0,
              best.get("MinimumReservationMinutes") or 0, (best.get("Specs") or {}).get("cpu_cores") or 0)
elif cmd == "vm":
    v = d if isinstance(d, dict) else {}
    sa = v.get("ssh_access") or {}
    print("\t".join(str(x) for x in [v.get("name", ""), v.get("deployment_id", ""),
          sa.get("ip_address") or v.get("ip_address", ""), sa.get("port") or 22,
          v.get("description", "") or ""]))
elif cmd == "desc":
    print((d or {}).get("description", "") or "" if isinstance(d, dict) else "")
elif cmd == "state":
    print((d or {}).get("state", "unknown") if isinstance(d, dict) else "unknown")
elif cmd == "list":
    for v in d or []:
        print("\t".join(str(x) for x in [v.get("name", ""), v.get("deployment_id", ""),
              v.get("ip_address", ""), v.get("description", "") or ""]))
elif cmd == "count":
    print(len(d) if isinstance(d, list) else -1)
elif cmd == "inlist":
    if not isinstance(d, list):
        print("unknown")
    else:
        refs = set(x for x in a if x)
        print("yes" if any(v.get("name") in refs or v.get("deployment_id") in refs for v in d) else "no")
elif cmd == "ids":
    for v in d or []:
        print(v.get("deployment_id", ""), v.get("name", ""))
elif cmd == "sshkey":
    print("yes" if any(k.get("fingerprint") == a[0] for k in d or []) else "no")
PY
J() { python3 "$TMPD/j.py" "$@"; }

# ------------------------------------------------------------------ the key
key_hygiene() {
  local perm
  [ -f "$KEYFILE" ] || { echo "key file $KEYFILE does not exist"; return 1; }
  perm=$(stat -f '%OLp' "$KEYFILE" 2>/dev/null || stat -c '%a' "$KEYFILE" 2>/dev/null || echo '?')
  [ "$perm" = 600 ] || { echo "key file $KEYFILE is mode $perm, must be 600"; return 1; }
  case "$(cd "$(dirname "$KEYFILE")" && pwd)/" in "$REPO"/*) echo "key file $KEYFILE is INSIDE the repository"; return 1 ;; esac
  return 0
}
load_key() {
  local K="" why
  why=$(key_hygiene) || die "REFUSING: $why" 2
  IFS= read -r K < "$KEYFILE" || [ -n "$K" ] || die "REFUSING: the key file is empty." 2
  K="${K//[$'\t\r\n ']/}"
  case "$K" in ''|*[!A-Za-z0-9._-]*) die "REFUSING: the key file holds characters outside [A-Za-z0-9._-]." 2 ;; esac
  CURLRC="$TMPD/curlrc"; TOKPAT="$TMPD/key.pattern"
  ( umask 077
    printf 'header = "Authorization: Token %s"\nsilent\nshow-error\n' "$K" > "$CURLRC"
    printf '%s\n' "$K" > "$TOKPAT" )
  K=""
}
api() {  # <method> <path under /api/> <body out> [max-time] [json body file]; prints the HTTP code, 000 on transport failure
  local c mt="${4:-60}"
  if [ -n "${5:-}" ]; then
    c=$(curl -K "$CURLRC" --max-time "$mt" -o "$3" -w '%{http_code}' -X "$1" \
          -H 'Content-Type: application/json' --data-binary "@$5" "$API/$2" 2>>"$TMPD/curl.err") || c=000
  else
    c=$(curl -K "$CURLRC" --max-time "$mt" -o "$3" -w '%{http_code}' -X "$1" "$API/$2" 2>>"$TMPD/curl.err") || c=000
  fi
  printf '%s' "${c:-000}"
}
redact() {  # <file>: replace any occurrence of the key
  [ -f "$1" ] && [ -n "$TOKPAT" ] || return 0
  python3 - "$TOKPAT" "$1" <<'PY'
import sys
tok = open(sys.argv[1]).read().strip()
p = sys.argv[2]
data = open(p, encoding="utf-8", errors="replace").read()
if tok and tok in data:
    open(p, "w").write(data.replace(tok, "<redacted>"))
PY
}
balance_cents() {  # prints cents, or -1
  local c
  c=$(api GET "teams/$TEAM/balance/" "$TMPD/balance.json")
  [ "$c" = 200 ] || { echo -1; return; }
  J balance "$TMPD/balance.json"
}
dollars() { [ "$1" -ge 0 ] 2>/dev/null && printf '$%d.%02d' $(( $1 / 100 )) $(( $1 % 100 )) || printf 'unknown'; }
vm_desc_ok() { case "$1" in mojolearn:*) return 0 ;; esac; return 1; }

# gone = GET 404, or absent from a 200 team listing. Prints "yes get=.. listed=.. state=..".
gone_check() {  # <ref> [<other ref>]
  local g l inl st="n/a"
  g=$(api GET "teams/$TEAM/virtual_machines/$1/" "$TMPD/gone_vm.json")
  if [ "$g" = 200 ]; then
    [ "$(api GET "teams/$TEAM/virtual_machines/$1/state/" "$TMPD/gone_state.json")" = 200 ] \
      && st=$(J state "$TMPD/gone_state.json")
  fi
  l=$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/gone_list.json")
  inl=unknown; [ "$l" = 200 ] && inl=$(J inlist "$TMPD/gone_list.json" "$1" "${2:-}")
  if [ "$g" = 404 ] || { [ "$l" = 200 ] && [ "$inl" = no ]; }; then
    printf 'yes get=%s list=%s listed=%s state=%s' "$g" "$l" "$inl" "$st"; return 0
  fi
  printf 'no get=%s list=%s listed=%s state=%s' "$g" "$l" "$inl" "$st"; return 1
}

# DELETE ?force=true, then verify gone for up to <verify seconds>. Sets GONE_LINE.
delete_and_verify() {  # <ref> <other ref or ""> <expected description or ""> <verify seconds> <record file>
  local ref=$1 other=$2 want=$3 secs=$4 rec=$5 c d i t0 line
  GONE_LINE=""
  if line=$(gone_check "$ref" "$other"); then
    GONE_LINE="verified_gone ref=$ref $line utc=$(utc) (already gone, no DELETE sent)"
    echo "$GONE_LINE" >> "$rec"; return 0
  fi
  c=$(api GET "teams/$TEAM/virtual_machines/$ref/" "$TMPD/del_vm.json")
  if [ "$c" = 200 ]; then
    d=$(J desc "$TMPD/del_vm.json")
    if [ -n "$d" ] && { ! vm_desc_ok "$d" || { [ -n "$want" ] && [ "$d" != "$want" ]; }; }; then
      echo "delete REFUSED ref=$ref: description '$d' is not this leg's ('${want:-mojolearn:*}')" >> "$rec"
      log "!! delete REFUSED: $ref has description '$d', not '${want:-mojolearn:*}'"
      return 2
    fi
  fi
  for i in 1 2 3 4; do
    c=$(api DELETE "teams/$TEAM/virtual_machines/$ref/?force=true" "$TMPD/del_body.txt" 900)
    log "DELETE $ref ?force=true -> HTTP $c"
    echo "delete ref=$ref attempt $i -> HTTP $c utc=$(utc) body=$(head -c 200 "$TMPD/del_body.txt" 2>/dev/null | tr '\n' ' ')" >> "$rec"
    case "$c" in 2*|404) break ;; esac
    nap 15
  done
  t0=$(date +%s)
  while :; do
    if line=$(gone_check "$ref" "$other"); then
      GONE_LINE="verified_gone ref=$ref $line utc=$(utc) after=$(( $(date +%s) - t0 ))s"
      echo "$GONE_LINE" >> "$rec"; log "$GONE_LINE"; return 0
    fi
    echo "verify ref=$ref $line utc=$(utc)" >> "$rec"
    [ $(( $(date +%s) - t0 )) -lt "$secs" ] || break
    nap 15
  done
  GONE_LINE="NOT_VERIFIED ref=$ref $line utc=$(utc)"
  echo "$GONE_LINE" >> "$rec"; log "!! $GONE_LINE"
  return 1
}

# ------------------------------------------------------------------ slots
slot_age() { local m; m=$(stat -f %m "$1" 2>/dev/null || stat -c %Y "$1" 2>/dev/null) || return 1; echo $(( $(date +%s) - m )); }
slot_field() { sed -n "s/^$2=//p" "$1/owner" 2>/dev/null | tail -1; }
slot_cap() {  # min(MAX_SLOTS, the team's maximum_virtual_machines)
  local cap=$MAX_SLOTS
  [ "${TEAM_MAX_VMS:-0}" -gt 0 ] 2>/dev/null && [ "$TEAM_MAX_VMS" -lt "$cap" ] && cap=$TEAM_MAX_VMS
  echo "$cap"
}
print_legs() {  # <slot dir> <indent>: the two legs of a 2gpu VM, when the slot holds one
  [ -f "$1/legs" ] || return 0
  local now g lane body out since state ex
  now=$(date +%s)
  while IFS=$'\t' read -r g lane body out since state ex; do
    case "$since" in ''|*[!0-9]*) since=$now ;; esac
    printf '%sleg %s  lane=%s  body=%s  out=%s  elapsed=%ss  state=%s  body_exit=%s\n' \
      "$2" "$g" "$lane" "$body" "$out" "$(( now - since ))" "$state" "$ex"
  done < "$1/legs"
}
list_slots() {
  local n s
  for n in 1 2 3; do
    s="$SLOT_PREFIX.$n"
    if [ -d "$s" ]; then
      printf '  slot %s  HELD %ss  %s\n' "$n" "$(slot_age "$s")" "$(tr '\n' ' ' < "$s/owner" 2>/dev/null || echo 'no owner file')"
      print_legs "$s" '          '
    else
      printf '  slot %s  free\n' "$n"
    fi
  done
}
try_take_slot() {  # <cap>
  local n
  for n in $(seq 1 "$1"); do
    if mkdir "$SLOT_PREFIX.$n" 2>/dev/null; then
      SLOT="$SLOT_PREFIX.$n"
      { echo "lane=$LANE"; echo "pid=$$"; echo "nonce=$NONCE"; echo "utc=$(utc)"; echo "out=$REAL_OUT"; } > "$SLOT/owner"
      return 0
    fi
  done
  return 1
}
check_stale_slots() {
  local n s age pid ref
  for n in 1 2 3; do
    s="$SLOT_PREFIX.$n"
    [ -d "$s" ] || continue
    age=$(slot_age "$s") || continue
    [ "$age" -gt "$SLOT_STALE_SECONDS" ] || continue
    pid=$(slot_field "$s" pid); ref=$(slot_field "$s" vm_ref)
    [ "$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/stale_list.json")" = 200 ] || continue
    if [ -z "$ref" ] || [ "$(J inlist "$TMPD/stale_list.json" "$ref")" = no ]; then
      if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        log "slot $s is ${age}s old, its VM ${ref:-<none recorded>} is absent, owner pid $pid is ALIVE: may be broken, left alone"
      else
        log "slot $s is ${age}s old, its VM ${ref:-<none recorded>} is absent, owner pid ${pid:-?} is dead: may be broken, REMOVED"
        echo "slot_broken_stale=$s age=${age}s owner=$(tr '\n' ' ' < "$s/owner" 2>/dev/null)" >> "$OUT/leg.txt"
        rm -rf "$s"
      fi
    fi
  done
}
release_slot() {
  [ -n "$SLOT" ] || return 0
  if grep -qx "nonce=$NONCE" "$SLOT/owner" 2>/dev/null; then
    rm -rf "$SLOT" && log "released $SLOT"
    echo "slot=released $SLOT $(utc)" >> "$OUT/leg.txt"
  else
    log "!! $SLOT no longer carries this leg's nonce; left in place"
  fi
  SLOT=""
}

# ------------------------------------------------------------------ status / reap
if [ "$CMD" = status ] || [ "$CMD" = reap ]; then
  OUT="$TMPD"; : > "$OUT/leg.txt"
  trap 'rm -rf "$TMPD"' EXIT
  load_key
  _b=$(balance_cents)
  echo "team $TEAM balance $(dollars "$_b") ($_b cents)"
  if [ "$CMD" = status ]; then
    c=$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/list.json")
    [ "$c" = 200 ] || die "GET virtual_machines -> HTTP $c" 2
    echo "VMs ($(J count "$TMPD/list.json")):"
    J list "$TMPD/list.json" | while IFS=$'\t' read -r n id ip d; do
      st=unknown
      [ "$(api GET "teams/$TEAM/virtual_machines/$id/state/" "$TMPD/st.json")" = 200 ] && st=$(J state "$TMPD/st.json")
      own="NOT OURS, never touched"; vm_desc_ok "$d" && own=ours
      printf '  %s  deployment_id=%s  ip=%s  state=%s  description=%s  (%s)\n' "$n" "$id" "$ip" "$st" "${d:-<empty>}" "$own"
      for _s in "$SLOT_PREFIX".1 "$SLOT_PREFIX".2 "$SLOT_PREFIX".3; do
        [ -f "$_s/legs" ] || continue
        _r=$(slot_field "$_s" vm_ref)
        if [ -n "$_r" ] && { [ "$_r" = "$id" ] || [ "$_r" = "$n" ]; }; then print_legs "$_s" '      '; fi
      done
    done
    echo "slots (/tmp, this Mac):"; list_slots
    exit 0
  fi
  case "$REAP_REF" in ''|*[!A-Za-z0-9_.-]*) die "reap <vm>: a VM name or deployment_id" 2 ;; esac
  c=$(api GET "teams/$TEAM/virtual_machines/$REAP_REF/" "$TMPD/reap.json")
  if [ "$c" = 200 ]; then
    d=$(J desc "$TMPD/reap.json")
    vm_desc_ok "$d" || die "REFUSING to reap $REAP_REF: its description '${d:-<empty>}' is not mojolearn:*" 3
    IFS=$'\t' read -r _n _id _ip _port _d < <(J vm "$TMPD/reap.json")
    delete_and_verify "$REAP_REF" "$_id" "" 600 "$TMPD/reap.txt"; rc=$?
  else
    delete_and_verify "$REAP_REF" "" "" 60 "$TMPD/reap.txt"; rc=$?
  fi
  cat "$TMPD/reap.txt"
  if [ "$rc" = 0 ]; then
    for _s in "$SLOT_PREFIX".1 "$SLOT_PREFIX".2 "$SLOT_PREFIX".3; do
      [ -d "$_s" ] || continue
      _r=$(slot_field "$_s" vm_ref); _p=$(slot_field "$_s" pid)
      if [ -n "$_r" ] && { [ "$_r" = "$REAP_REF" ] || [ "$_r" = "${_id:-}" ]; }; then
        if [ -n "$_p" ] && kill -0 "$_p" 2>/dev/null; then
          echo "slot $_s names this VM and its owner pid $_p is alive; left for its owner to release"
        else
          rm -rf "$_s" && echo "released $_s (owner pid ${_p:-?} dead, VM verified gone)"
        fi
      fi
    done
  fi
  echo "team $TEAM balance after $(dollars "$(balance_cents)")"
  exit "$rc"
fi

# ------------------------------------------------------------------ the leg
LEG_EXTRA="${MOJOLEARN_GEMM_LEG_EXTRA:-}"
if [ "$MODE" != probe ] && [ "$TEST_2GPU" = 0 ]; then
  [ -n "$LEG_EXTRA" ] || die "MOJOLEARN_GEMM_LEG_EXTRA=<body.sh> is required: running a body is this runner's whole job" 2
  [ -f "$LEG_EXTRA" ] || die "MOJOLEARN_GEMM_LEG_EXTRA=$LEG_EXTRA does not exist" 2
fi
LEG_EXTRA_B="${MOJOLEARN_GEMM_LEG_EXTRA_B:-}"
EXTRA_ENV_B="${MOJOLEARN_HOTAISLE_EXTRA_ENV_B:-}"
if [ "$MODE" != probe ]; then
  if [ "$SPEC" != 2gpu ]; then
    if [ -n "$LEG_EXTRA_B${MOJOLEARN_GEMM_LEG_OUT_B:-}$EXTRA_ENV_B${MOJOLEARN_HOTAISLE_LANE_B:-}" ]; then
      die "REFUSED: MOJOLEARN_GEMM_LEG_EXTRA_B, MOJOLEARN_GEMM_LEG_OUT_B, MOJOLEARN_HOTAISLE_EXTRA_ENV_B and MOJOLEARN_HOTAISLE_LANE_B belong to the GPU 1 body of --spec 2gpu; the $SPEC spec runs one body and will not silently drop the other" 2
    fi
  else
    [ "${MOJOLEARN_HOTAISLE_GPU_ONLY:-0}" = 1 ] \
      || die "REFUSED: --spec 2gpu needs MOJOLEARN_HOTAISLE_GPU_ONLY=1, the caller's word that neither body is a CPU opponent row (26 cores is not the 13-core CPU tuple)" 2
    if [ "$TEST_2GPU" = 1 ]; then
      [ -z "$LEG_EXTRA$LEG_EXTRA_B$EXTRA_ENV_B${MOJOLEARN_HOTAISLE_EXTRA_ENV:-}" ] \
        || die "--test-2gpu runs its own two tiny bodies: unset MOJOLEARN_GEMM_LEG_EXTRA, MOJOLEARN_GEMM_LEG_EXTRA_B and both extra envs" 2
    else
      [ -n "$LEG_EXTRA_B" ] || die "MOJOLEARN_GEMM_LEG_EXTRA_B=<body.sh> is required on --spec 2gpu: it is the GPU 1 body" 2
      [ -f "$LEG_EXTRA_B" ] || die "MOJOLEARN_GEMM_LEG_EXTRA_B=$LEG_EXTRA_B does not exist" 2
    fi
    printf '\nWARNING: the 2x MI300X VM bills its 60-minute minimum reservation whatever the bodies take.\n         Pair two legs that each need at least 40 minutes; a short leg leaves its GPU idle on the bill.\n\n' >&2
  fi
fi
LANE="${MOJOLEARN_HOTAISLE_LANE:-$(basename "${LEG_EXTRA:-probe}" .sh)}"
if [ "$TEST_2GPU" = 1 ]; then LANE="${MOJOLEARN_HOTAISLE_LANE:-test2gpu}"; fi
case "$LANE" in ''|*[!A-Za-z0-9_.-]*) die "MOJOLEARN_HOTAISLE_LANE='$LANE': letters, digits and _.- only (it goes in the VM description)" 2 ;; esac
LANE_A="$LANE"
if [ "$SPEC" = 2gpu ]; then
  _lane_b=$(basename "${LEG_EXTRA_B:-probe}" .sh)
  if [ "$TEST_2GPU" = 1 ]; then _lane_b=test2gpu; fi
  LANE_B="${MOJOLEARN_HOTAISLE_LANE_B:-$_lane_b}"
  case "$LANE_B" in ''|*[!A-Za-z0-9_.-]*) die "MOJOLEARN_HOTAISLE_LANE_B='$LANE_B': letters, digits and _.- only (it goes in the VM description)" 2 ;; esac
  LANE="2gpu.$LANE_A.$LANE_B"   # the VM's lane: its description, slot owner and create lock
fi
STAMP="$(date -u +%Y-%m-%d_%H%M%S)"
CARD_FULL="${MOJOLEARN_GEMM_CARD_FULL:-}"
LEG_DUMP="${MOJOLEARN_IDENTITY_TRACE_DUMP:-}"
for _v in "$CARD_FULL" "$LEG_DUMP"; do
  case "$_v" in *[!A-Za-z0-9_.,:-]*) die "MOJOLEARN_GEMM_CARD_FULL / MOJOLEARN_IDENTITY_TRACE_DUMP: letters, digits and _.,:- only" 2 ;; esac
done
EXTRA_ENV="${MOJOLEARN_HOTAISLE_EXTRA_ENV:-}"

# THE 2GPU CPU OPPONENT SCAN. One line per mark, nothing when clean. The body's
# own path, then every tools/ or bench/ shell script named on a non-comment
# line, recursively (six levels): a basename on CPU_OPPONENT_DENY, or a deny
# name anywhere outside a comment. Then the env words' name prefixes.
cpu_opponent_marks() {  # <body path> <env words>
  local queue="$1" next f b n w seen=" " depth=0
  while [ -n "${queue// /}" ] && [ "$depth" -lt 6 ]; do
    next=""
    for f in $queue; do
      case "$seen" in *" $f "*) continue ;; esac
      seen="$seen$f "
      b=$(basename "$f")
      for n in $CPU_OPPONENT_DENY; do
        [ "$b" = "$n" ] && echo "$f is on the CPU opponent deny list"
      done
      case "$f" in *.sh) [ -f "$f" ] || continue ;; *) continue ;; esac
      grep -v '^[[:space:]]*#' "$f" > "$TMPD/deny_scan.txt" 2>/dev/null
      for n in $CPU_OPPONENT_DENY; do
        grep -qF "$n" "$TMPD/deny_scan.txt" && echo "$f names $n (a CPU opponent driver) outside a comment"
      done
      next="$next $(grep -oE '(tools|bench)/[A-Za-z0-9_./-]+\.sh' "$TMPD/deny_scan.txt" | sort -u | tr '\n' ' ')"
    done
    queue=$next
    depth=$((depth + 1))
  done
  for w in $2; do
    for n in $CPU_OPPONENT_ENV_DENY; do
      case "$w" in "$n"*) echo "env ${w%%=*} marks a CPU opponent row (prefix $n)" ;; esac
    done
  done
  return 0
}
if [ "$SPEC" = 2gpu ] && [ "$MODE" != probe ] && [ "$TEST_2GPU" = 0 ]; then
  _marks_a=$(cpu_opponent_marks "$LEG_EXTRA" "$EXTRA_ENV")
  _marks_b=$(cpu_opponent_marks "$LEG_EXTRA_B" "$EXTRA_ENV_B")
  if [ -n "$_marks_a$_marks_b" ]; then
    die "REFUSED on --spec 2gpu: a body marks a CPU opponent row, whose tuple is the 13-core VM (this VM has 26 cores):
$( { printf '%s\n' "$_marks_a" | grep . | sed 's/^/  GPU 0 body: /'; printf '%s\n' "$_marks_b" | grep . | sed 's/^/  GPU 1 body: /'; } )" 2
  fi
fi

if [ "$SPEC" = 2gpu ]; then
  OUT="${MOJOLEARN_GEMM_LEG_OUT:-bench/results/e1g/${STAMP}-amd-${LEG2_LABEL}-hotaisle-${LANE_A}-gpu0}"
else
  OUT="${MOJOLEARN_GEMM_LEG_OUT:-bench/results/e1g/${STAMP}-amd-mi300x-hotaisle-${LANE}}"
fi
case "$OUT" in /*) ;; *) OUT="$REPO/$OUT" ;; esac
REAL_OUT="$OUT"
if [ "$SPEC" = 2gpu ] && [ "$MODE" != probe ]; then
  OUT_B="${MOJOLEARN_GEMM_LEG_OUT_B:-bench/results/e1g/${STAMP}-amd-${LEG2_LABEL}-hotaisle-${LANE_B}-gpu1}"
  case "$OUT_B" in /*) ;; *) OUT_B="$REPO/$OUT_B" ;; esac
  REAL_OUT_B="$OUT_B"
  [ "${REAL_OUT_B%/}" != "${REAL_OUT%/}" ] || die "MOJOLEARN_GEMM_LEG_OUT and MOJOLEARN_GEMM_LEG_OUT_B name one directory; each body needs its own" 2
fi
if [ "$MODE" != rent ]; then
  OUT="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-hotaisle-$MODE.XXXXXX")" || exit 2
  if [ -n "$OUT_B" ]; then OUT_B="$(mktemp -d "${TMPDIR:-/tmp}/mojolearn-hotaisle-$MODE-gpu1.XXXXXX")" || exit 2; fi
fi
mkdir -p "$OUT" || die "cannot create $OUT" 2
if [ -n "$OUT_B" ]; then mkdir -p "$OUT_B" || die "cannot create $OUT_B" 2; fi
OUTS=("$OUT" "$OUT_B")
SIZE_LABEL="mi300x-${SPEC}"
if [ "$SPEC" = 2gpu ]; then SIZE_LABEL="$LEG2_LABEL"; fi

# shellcheck disable=SC2317
cancel_deadman() {
  [ -n "$DEADMAN_PID" ] || return 0
  pkill -KILL -P "$DEADMAN_PID" 2>/dev/null
  kill -KILL "$DEADMAN_PID" 2>/dev/null && log "Mac dead-man cancelled (pid $DEADMAN_PID)"
  [ -n "$DEADMAN_DIR" ] && rm -rf "$DEADMAN_DIR"
  echo "mac_deadman=cancelled $(utc)" >> "$OUT/deadman.txt"
  DEADMAN_PID=""; DEADMAN_DIR=""
}
# shellcheck disable=SC2317
release_create_lock() {
  [ "$CREATE_LOCK_HELD" = 1 ] || return 0
  grep -qx "nonce=$NONCE" "$CREATE_LOCK/owner" 2>/dev/null && rm -rf "$CREATE_LOCK"
  CREATE_LOCK_HELD=0
}
# shellcheck disable=SC2317
adopt_new_vm() {  # sets VMREF from a VM absent from the pre-create snapshot; returns 1 when none or ambiguous
  local c new n
  c=$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/adopt.json")
  [ "$c" = 200 ] || return 1
  new=$(J ids "$TMPD/adopt.json" | while read -r id nm; do
          grep -qx "$id" "$TMPD/pre_ids.txt" 2>/dev/null || echo "$id $nm"
        done)
  n=$(printf '%s' "$new" | grep -c . )
  [ "$n" = 1 ] || { [ "$n" -gt 1 ] && log "!! $n new VMs appear; adopting none: $(echo "$new" | tr '\n' ';')"; return 1; }
  DEPLOY_ID=${new%% *}; VMNAME=${new#* }; VMREF=$DEPLOY_ID
  return 0
}

# shellcheck disable=SC2317
teardown() {
  local rc=$? line
  trap '' INT TERM
  trap - EXIT
  if [ "$CREATE_ATTEMPTED" = 1 ]; then
    log "teardown (exit $rc)"
    echo "== teardown $(utc) exit=$rc vm=${VMREF:-unknown} ==" >> "$OUT/teardown.txt"
    if [ -z "$VMREF" ]; then
      for _i in 1 2 3 4 5 6 7 8 9 10 11 12; do adopt_new_vm && break; sleep 10; done
      if [ -n "$VMREF" ]; then
        log "ADOPTED $VMREF at teardown after an unreadable create"
        printf '%s\n' "$VMREF" > "$DEADMAN_DIR/vm_ref.txt" 2>/dev/null
      fi
    fi
    if [ -z "$VMREF" ]; then
      if [ "$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/final.json")" = 200 ] \
         && [ -z "$(J ids "$TMPD/final.json" | while read -r id nm; do grep -qx "$id" "$TMPD/pre_ids.txt" || echo "$id"; done)" ]; then
        echo "verified_gone ref=none: the 200 listing shows no VM absent from the pre-create snapshot utc=$(utc)" >> "$OUT/teardown.txt"
        DESTROY_CONFIRMED=1
      fi
    elif [ "$TEST_WATCHDOG" = 1 ] && [ "$WATCHDOG_OK" = 1 ] && [ "$WDT_DONE" = 0 ]; then
      log "watchdog test interrupted before its verdict: the verified on-box watchdog and the late Mac backstop are LEFT to end $VMREF"
      echo "watchdog_test=INTERRUPTED; delete left to the on-box watchdog and the backstop" >> "$OUT/teardown.txt"
    elif delete_and_verify "$VMREF" "$VMNAME" "$DESC" 600 "$OUT/teardown.txt"; then
      DESTROY_CONFIRMED=1
    fi
    release_create_lock
    echo "destroy_confirmed=$DESTROY_CONFIRMED" >> "$OUT/teardown.txt"
  fi
  release_create_lock
  if [ "$CREATE_ATTEMPTED" = 0 ] || [ "$DESTROY_CONFIRMED" = 1 ]; then
    cancel_deadman
    release_slot
  else
    {
      echo
      echo "  ############################################################"
      echo "  # HOT AISLE VM ${VMREF:-<unknown>} ${VMNAME:+($VMNAME) }MAY STILL BE BILLING."
      echo "  # The API did not confirm it is gone. The Mac dead-man (pid ${DEADMAN_PID:-none})"
      echo "  # and the on-box watchdog are LEFT ARMED. End it by hand now:"
      echo "  #   bash tools/hotaisle_leg.sh reap ${VMREF:-<vm>}"
      echo "  # Slot ${SLOT:-none} stays HELD until then."
      echo "  ############################################################"
    } | tee -a "$OUT/teardown.txt" >&2
    echo "mac_deadman=LEFT_ARMED pid=$DEADMAN_PID" >> "$OUT/deadman.txt"
    [ "$rc" = 0 ] && rc=1
  fi
  if [ "$TEST_2GPU" = 1 ] && [ "$MODE" = rent ] && [ "$CREATE_ATTEMPTED" = 1 ]; then
    if [ "$TEST2_BODIES" = PASS ] && [ "$DESTROY_CONFIRMED" = 1 ]; then
      line="test_2gpu=PASS (each pinned body saw exactly one GPU at its pinned address at both prints, the two differ, the bodies overlapped and exited 0, the VM is verified gone)"
    else
      line="test_2gpu=FAIL (bodies=$TEST2_BODIES destroy_confirmed=$DESTROY_CONFIRMED)"
      [ "$rc" = 0 ] && rc=1
    fi
    log "$line"
    echo "$line" >> "$OUT/leg.txt"
    echo "$line" >> "$OUT/teardown.txt"
    [ -n "$OUT_B" ] && echo "$line" >> "$OUT_B/leg.txt"
  fi
  if [ "$MODE" = rent ]; then
    line=$(balance_cents)
    log "team balance at the end $(dollars "$line") (at the start $(dollars "${BAL_BEFORE:--1}"))"
    echo "balance_after_cents=$line" >> "$OUT/leg.txt"
    echo "exit=$rc" >> "$OUT/teardown.txt"
  fi
  if [ "$SPEC" = 2gpu ] && [ "$MODE" = rent ] && [ -n "$OUT_B" ] && [ -d "$OUT_B" ]; then
    cp "$OUT/teardown.txt" "$OUT/deadman.txt" "$OUT_B/" 2>/dev/null
    { echo "balance_after_cents=$line"; echo "vm_records_copied=teardown.txt deadman.txt from $OUT"; } >> "$OUT_B/leg.txt"
  fi
  [ -n "$TMPD" ] && rm -rf "$TMPD"
  exit "$rc"
}
trap teardown EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ------------------------------------------------------------------ probe
if [ "$MODE" = probe ]; then
  load_key
  c=$(api GET "teams/" "$TMPD/teams.json"); echo "GET teams/ -> $c"
  read -r _op TEAM_MAX_VMS < <(J teams "$TMPD/teams.json" "$TEAM")
  echo "  team $TEAM operator=$_op maximum_virtual_machines=$TEAM_MAX_VMS (slots used: $(slot_cap))"
  _b=$(balance_cents); echo "  balance $(dollars "$_b") ($_b cents; floor $MIN_BALANCE_CENTS)"
  c=$(api GET "teams/$TEAM/virtual_machines/available/" "$TMPD/avail.json"); echo "GET available -> $c"
  J offers "$TMPD/avail.json"
  if [ "$SPEC" = 2gpu ]; then
    read -r _f _q _p _m _c < <(J pick2 "$TMPD/avail.json" "$TMPD/spec.json")
    echo "  MOJOLEARN_HOTAISLE_SPEC=2gpu -> $_f quantity=$_q price=$_p cents/h min_reservation=$_m cpu_cores=$_c"
  else
    read -r _f _q _p _m < <(J pick "$TMPD/avail.json" "$SPEC_CORES" "$TMPD/spec.json")
    echo "  MOJOLEARN_HOTAISLE_SPEC=$SPEC -> $_f quantity=$_q price=$_p cents/h min_reservation=$_m"
  fi
  c=$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/list.json"); echo "GET virtual_machines -> $c ($(J count "$TMPD/list.json") VMs)"
  J list "$TMPD/list.json" | sed 's/^/  /'
  c=$(api GET "user/ssh_keys/" "$TMPD/keys.json"); echo "GET user/ssh_keys -> $c; this Mac's key registered: $(J sshkey "$TMPD/keys.json" "$SSH_KEY_FP")"
  echo "slots:"; list_slots
  exit 0
fi

# ------------------------------------------------------------- local checks
RED=0; BLOCK=0
rok()    { printf '  ok     %s\n' "$1"; }
rbad()   { RED=1;   printf '  FAIL   %s\n' "$1"; }
rblock() { BLOCK=1; printf '  BLOCK  %s\n' "$1"; }
SHIPS_SOURCE=1; [ "$BARE" = 1 ] || [ "$TEST_WATCHDOG" = 1 ] && SHIPS_SOURCE=0
if [ "$TEST_2GPU" = 1 ]; then SHIPS_SOURCE=0; fi

COMMIT="$(git -C "$REPO" rev-parse HEAD)" || die "not a git checkout: $REPO" 2
COMMIT_LINE="$(git -C "$REPO" log -1 --format='%h parent %p' "$COMMIT")"
if [ "$SPEC" = 2gpu ]; then
  echo "== hotaisle_leg: one Hot Aisle 2x MI300X VM running two extra bodies, one pinned container per GPU =="
  echo "   mode      $MODE$( [ "$TEST_2GPU" = 1 ] && echo ' TEST-2GPU')"
  echo "   commit    $COMMIT_LINE"
  echo "   spec      2gpu (2x MI300X, $SPEC_CORES cores; ONE VM, ONE slot, ONE watchdog, ONE delete), team $TEAM"
  echo "   lease     $MINUTES minutes (the offering's minimum reservation; Mac dead-man and on-box watchdog at that deadline)"
  echo "   label     $LEG2_LABEL (in every leg.txt of this leg; never mixed with 1x MI300X VM rows)"
  echo "   vm lane   $LANE"
  echo "   GPU 0     ${LEG_EXTRA:-<the runner tiny test body>}   lane $LANE_A"
  echo "             out $REAL_OUT"
  echo "   GPU 1     ${LEG_EXTRA_B:-<the runner tiny test body>}   lane $LANE_B"
  echo "             out $REAL_OUT_B"
  echo "   gates     $( [ "$GATES" = 1 ] && echo 'device check + card, per body on its own GPU' || echo 'SKIPPED (--skip-gates)')"
  echo "   archs     ${GPU_ARCHS:-<unset: read from rocminfo on the VM>}   column amd"
  echo "   runtime   $RUNTIME_WANT (image $IMAGE; native is refused)"
  echo "   WARNING   the 60-minute minimum is billed whatever the bodies take: pair two legs that each need at least 40 minutes"
else
  echo "== hotaisle_leg: one Hot Aisle 1x MI300X leg running an extra body =="
  echo "   mode      $MODE$( [ "$TEST_WATCHDOG" = 1 ] && echo ' TEST-WATCHDOG')$( [ "$BARE" = 1 ] && echo ' BARE')"
  echo "   commit    $COMMIT_LINE"
  echo "   spec      $SPEC (1x MI300X, $SPEC_CORES cores), team $TEAM"
  echo "   lease     $MINUTES minutes (Mac dead-man and on-box watchdog at that deadline)"
  echo "   body      $LEG_EXTRA   lane $LANE"
  echo "   gates     $( [ "$GATES" = 1 ] && echo 'device check + card' || echo 'SKIPPED (--skip-gates)')"
  echo "   archs     ${GPU_ARCHS:-<unset: read from rocminfo on the VM>}   column amd"
  echo "   runtime   $RUNTIME_WANT (image $IMAGE)"
  echo "   out       $REAL_OUT"
fi
echo
echo "== local checks =="
DIRTY="$(git -C "$REPO" status --porcelain -- . ':!bench/results' 2>/dev/null)"
if [ -n "$DIRTY" ] && [ "$SHIPS_SOURCE" = 1 ]; then
  rblock "the tree is DIRTY (minus bench/results); a real leg refuses. Launch from git worktree add --detach:"
  printf '%s\n' "$DIRTY" | head -20 | sed 's/^/           /'
elif [ -n "$DIRTY" ]; then
  printf '  info   the tree is dirty; this test mode ships no source\n'
else
  rok "the tree is clean (minus bench/results)"
fi
gen_extra_env() {  # <env words> <the variable they came from>: the sourced file; refusals as comments
  local _w _n _v
  echo "# Generated by tools/hotaisle_leg.sh from $2; sourced before the extra body."
  for _w in $1; do
    case "$_w" in
      MOJOLEARN_HOTAISLE_*=*|MOJOLEARN_DO_*=*|MOJOLEARN_RUNPOD_*=*|MOJOLEARN_GEMM_LEG_*=*|MOJOLEARN_GPU_ARCHS=*|MOJOLEARN_TARGET_COLUMN=*)
        printf '# REFUSED (runner-owned name): %s\n' "${_w%%=*}" ;;
      MOJOLEARN_[A-Z0-9_]*=*|MODULAR_[A-Z0-9_]*=*)
        _n=${_w%%=*}; _v=${_w#*=}
        case "$_n" in *[!A-Z0-9_]*) printf '# REFUSED (name characters): %s\n' "$_n"; continue ;; esac
        case "$_v" in
          *[!A-Za-z0-9_.,:/=-]*) printf '# REFUSED (value characters): %s\n' "$_n" ;;
          *) printf "export %s='%s'\n" "$_n" "$_v" ;;
        esac ;;
      *) printf '# REFUSED (not NAME=value with a MOJOLEARN_ or MODULAR_ name): %s\n' "${_w%%=*}" ;;
    esac
  done
}
if [ "$TEST_2GPU" = 1 ]; then
  printf '  info   --test-2gpu: no extra bodies; the runner writes two tiny pinned test bodies\n'
  EXTRA_SHA=none
else
  if sh -n "$LEG_EXTRA" 2> "$TMPD/extra_syntax.err"; then
    rok "the extra body is valid sh: $LEG_EXTRA"
  else
    rbad "the extra body is NOT valid sh: $(head -3 "$TMPD/extra_syntax.err")"
  fi
  cp "$LEG_EXTRA" "$OUT/extra_body.sh"
  EXTRA_SHA="$(sha256_of "$OUT/extra_body.sh")"

  gen_extra_env "$EXTRA_ENV" MOJOLEARN_HOTAISLE_EXTRA_ENV > "$OUT/extra_env.sh"
  if ! grep -q '^# REFUSED' "$OUT/extra_env.sh" && sh -n "$OUT/extra_env.sh" 2>/dev/null; then
    rok "the extra body environment: $(grep -c '^export ' "$OUT/extra_env.sh" | tr -d ' ') export(s)$( [ -n "$EXTRA_ENV" ] && echo ": $EXTRA_ENV")"
  else
    rbad "MOJOLEARN_HOTAISLE_EXTRA_ENV is refused: $(grep '^# REFUSED' "$OUT/extra_env.sh" | tr '\n' ' ')"
  fi
  if [ "$SPEC" = 2gpu ]; then
    if sh -n "$LEG_EXTRA_B" 2> "$TMPD/extra_syntax_b.err"; then
      rok "the GPU 1 extra body is valid sh: $LEG_EXTRA_B"
    else
      rbad "the GPU 1 extra body is NOT valid sh: $(head -3 "$TMPD/extra_syntax_b.err")"
    fi
    cp "$LEG_EXTRA_B" "$OUT_B/extra_body.sh"
    EXTRA_SHA_B="$(sha256_of "$OUT_B/extra_body.sh")"
    gen_extra_env "$EXTRA_ENV_B" MOJOLEARN_HOTAISLE_EXTRA_ENV_B > "$OUT_B/extra_env.sh"
    if ! grep -q '^# REFUSED' "$OUT_B/extra_env.sh" && sh -n "$OUT_B/extra_env.sh" 2>/dev/null; then
      rok "the GPU 1 extra body environment: $(grep -c '^export ' "$OUT_B/extra_env.sh" | tr -d ' ') export(s)$( [ -n "$EXTRA_ENV_B" ] && echo ": $EXTRA_ENV_B")"
    else
      rbad "MOJOLEARN_HOTAISLE_EXTRA_ENV_B is refused: $(grep '^# REFUSED' "$OUT_B/extra_env.sh" | tr '\n' ' ')"
    fi
    rok "no CPU opponent mark on either body (deny list, the scripts they reach, env prefixes; MOJOLEARN_HOTAISLE_GPU_ONLY=1 given)"
  fi
fi

if _why=$(key_hygiene); then
  rok "key file present, 0600, outside the repository$( [ "$MODE" = dry ] && echo ' (not read by a dry run)')"
elif [ "$MODE" = dry ]; then
  printf '  info   %s (a dry run does not need it)\n' "$_why"
else
  die "REFUSING to rent: $_why" 2
fi
if ssh-keygen -lf "$SSH_KEY.pub" 2>/dev/null | grep -qF "$SSH_KEY_FP"; then
  rok "ssh key $SSH_KEY is the registered one ($SSH_KEY_FP)"
else
  rbad "ssh key $SSH_KEY.pub is missing or is not $SSH_KEY_FP"
fi
echo "  info   slots:"; list_slots | sed 's/^/       /'

# ---- the bundle ----
BUNDLE_BYTES=0; BUNDLE_SHA=none
if [ "$SHIPS_SOURCE" = 1 ]; then
  ARCHIVE_EXCLUDES=(':!bench/results' ':!mamba/corpus' ':!bench/oracle*' ':!bench/minentropy_oracle.txt' ':!*.bin')
  if git -C "$REPO" archive --format=tar -o "$TMPD/src.tar" "$COMMIT" -- . "${ARCHIVE_EXCLUDES[@]}"; then
    gzip -9 -c "$TMPD/src.tar" > "$TMPD/src.tgz"
    tar tf "$TMPD/src.tar" | grep -v '/$' > "$OUT/bundle_files.txt"
    BUNDLE_BYTES=$(wc -c < "$TMPD/src.tgz" | tr -d ' ')
    BUNDLE_SHA=$(sha256_of "$TMPD/src.tgz")
    mkdir "$TMPD/archive" && tar xf "$TMPD/src.tar" -C "$TMPD/archive"
    rm -f "$TMPD/src.tar"
    if grep -Eq '(^bench/results/|^mamba/corpus/|\.bin$)' "$OUT/bundle_files.txt"; then
      rbad "the bundle carries an excluded path"
    fi
    if [ "$BUNDLE_BYTES" -gt "$MAX_BUNDLE_BYTES" ]; then
      rblock "the bundle is $BUNDLE_BYTES bytes gzipped, over the $MAX_BUNDLE_BYTES cap"
    else
      rok "bundle $BUNDLE_BYTES bytes gzipped, $(wc -l < "$OUT/bundle_files.txt" | tr -d ' ') files, sha256 ${BUNDLE_SHA:0:16}"
    fi
    REQUIRED="pixi.toml pixi.lock"
    [ "$GATES" = 1 ] && REQUIRED="$REQUIRED tools/with_identical_mode.sh tools/gemm_card.sh gemm/checks/gemm_device_check.mojo"
    for _need in $REQUIRED; do
      [ -f "$TMPD/archive/$_need" ] || rbad "the bundle does not contain $_need"
    done
    ( cd "$TMPD/archive" && find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
        | LC_ALL=C sort | xargs shasum -a 256 | shasum -a 256 | awk '{print $1}' ) > "$OUT/source_sha256_local.txt"
  else
    rbad "git archive of $COMMIT failed"
  fi
fi

# ---- the scripts that run on the VM ----
cat > "$TMPD/remote_body.sh.template" <<'REMOTE_BODY'
#!/bin/sh
# Generated by tools/hotaisle_leg.sh. RUNS ON THE HOT AISLE VM (in the
# container when runtime is docker or podman). The same steps, in the same
# order, as tools/do_extra_leg.sh's remote body, so a MOJOLEARN_GEMM_LEG_EXTRA
# body sees the same world. `set -u`, not `set -e`: a red gate is a result.
set -u
ROOT=/root/mojolearn
OUT=/root/gemm_leg_out
mkdir -p "$OUT" "$ROOT"
cd "$ROOT" || exit 9
HOME=/root
export HOME

{
  echo "vendor=amd"
  echo "commit=@COMMIT@"
  echo "card_full=@CARDFULL@"
  echo "trace_dump=@DUMP@"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "provider=hotaisle"
  echo "size=@SIZE@"
  echo "runtime=@RUNTIME@"
  echo "image=@IMAGE@"
  echo "gates=@GATES@"
  echo "bare=@BARE@"
  echo "gpu_archs=@GPUARCHS@"
  echo "target_column=amd"
} > "$OUT/leg.txt"

MOJOLEARN_GPU_ARCHS="@GPUARCHS@"
export MOJOLEARN_GPU_ARCHS
MOJOLEARN_TARGET_COLUMN=amd
export MOJOLEARN_TARGET_COLUMN

uname -a > "$OUT/uname.txt" 2>&1
rocm-smi --showproductname > "$OUT/gpu.txt" 2>&1 || echo "no vendor smi tool answered" >> "$OUT/gpu.txt"

if [ "@BARE@" != "1" ]; then
    { find . -name '*.mojo' -not -path './.pixi/*' -not -path './bench/results/*' \
        | LC_ALL=C sort | xargs sha256sum ; } \
      | sha256sum | awk '{print $1}' > "$OUT/source_sha256.txt"

    if ! command -v curl > /dev/null 2>&1 && command -v apt-get > /dev/null 2>&1; then
        { apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y curl ca-certificates ; } \
            > "$OUT/apt_curl.log" 2>&1 < /dev/null
    fi
    if [ ! -x "$HOME/.pixi/bin/pixi" ] && ! command -v pixi > /dev/null 2>&1; then
        # No `< /dev/null` on this sh: its stdin IS the installer (a redirect
        # there made the first smoke install nothing, 2026-09-11).
        curl -fsSL https://pixi.sh/install.sh | sh > "$OUT/pixi_install.log" 2>&1
    fi
    PATH="$HOME/.pixi/bin:$PATH"
    export PATH
    command -v pixi > "$OUT/pixi_which.txt" 2>&1 || echo "NO PIXI" >> "$OUT/pixi_which.txt"

    t0=$(date +%s)
    pixi install > "$OUT/pixi_env.log" 2>&1 < /dev/null
    echo "pixi_install_exit=$?" >> "$OUT/leg.txt"
    echo "pixi_install_seconds=$(( $(date +%s) - t0 ))" >> "$OUT/leg.txt"
    pixi run mojo --version > "$OUT/mojo_version.txt" 2>&1 < /dev/null || true

    if [ "@GATES@" = "1" ]; then
        tools/with_identical_mode.sh pixi run mojo run -I . \
            gemm/checks/gemm_device_check.mojo > "$OUT/device_check.log" 2>&1 < /dev/null
        echo "device_check_exit=$?" >> "$OUT/leg.txt"
        MOJOLEARN_GEMM_CARD_FULL="@CARDFULL@" MOJOLEARN_IDENTITY_TRACE_DUMP="@DUMP@" \
            sh tools/gemm_card.sh device "$OUT/amd.card" > "$OUT/card_driver.log" 2>&1 < /dev/null
        echo "card_exit=$?" >> "$OUT/leg.txt"
    else
        echo "device_check_exit=SKIPPED" >> "$OUT/leg.txt"
        echo "card_exit=SKIPPED" >> "$OUT/leg.txt"
    fi
fi

if [ -f /root/gemm_leg_extra.sh ]; then
    (
        if [ -f /root/gemm_leg_extra_env.sh ]; then
            . /root/gemm_leg_extra_env.sh
        fi
        sh /root/gemm_leg_extra.sh
    ) > "$OUT/extra.log" 2>&1 < /dev/null
    echo "extra_exit=$?" >> "$OUT/leg.txt"
    cp /root/gemm_leg_extra_env.sh "$OUT/extra_env.sh" 2>/dev/null || true
fi

echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/leg.txt"
: > /root/gemm_leg.done
echo REMOTE_BODY_DONE
REMOTE_BODY

cat > "$TMPD/watchdog.sh.template" <<'WATCHDOG'
#!/bin/sh
# Written by tools/hotaisle_leg.sh. RUNS ON THE VM AS ROOT, DETACHED (setsid).
# Sleeps to the lease deadline, then DELETEs THIS VM through the API with
# force. The key is in @BOXRC@ (0600), never in an argv.
set -u
trap '' HUP INT
echo $$ > @BOXDIR@/watchdog.pid
T=$(( $(date +%s) + @SECS@ ))
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) armed ref=@VMREF@ fires_in=@SECS@s" >> @BOXDIR@/watchdog.out
while [ "$(date +%s)" -lt "$T" ]; do sleep 10; done
n=1
while [ "$n" -le 20 ]; do
    code=$(curl -K @BOXRC@ --max-time 900 -o @BOXDIR@/watchdog.body -w '%{http_code}' \
        -X DELETE '@API@/teams/@TEAM@/virtual_machines/@VMREF@/?force=true')
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) DELETE @VMREF@ attempt $n -> $code" >> @BOXDIR@/watchdog.out
    case "$code" in 2*|404) exit 0 ;; esac
    sleep 15
    n=$((n + 1))
done
WATCHDOG

cat > "$TMPD/watchdog_arm.sh.template" <<'ARM'
set -u
if [ ! -s @BOXRC@ ]; then echo NO_CREDENTIAL_ON_BOX; exit 3; fi
chmod 600 @BOXRC@
chmod 700 @BOXDIR@ @BOXDIR@/watchdog.sh
rm -f @BOXDIR@/watchdog.pid
if command -v setsid > /dev/null 2>&1; then
    setsid nohup sh @BOXDIR@/watchdog.sh > @BOXDIR@/watchdog.log 2>&1 < /dev/null &
else
    nohup sh @BOXDIR@/watchdog.sh > @BOXDIR@/watchdog.log 2>&1 < /dev/null &
fi
i=0
while [ "$i" -lt 15 ] && [ ! -s @BOXDIR@/watchdog.pid ]; do sleep 1; i=$((i + 1)); done
pid=$(cat @BOXDIR@/watchdog.pid 2>/dev/null)
if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then echo "WATCHDOG_ALIVE pid=$pid"; else echo WATCHDOG_DEAD; fi
echo "REF_BAKED_IN=$(grep -c 'virtual_machines/@VMREF@/?force=true' @BOXDIR@/watchdog.sh)"
echo "TOKEN_GET_HTTP=$(curl -K @BOXRC@ --max-time 30 -o @BOXDIR@/self.json -w '%{http_code}' '@API@/teams/@TEAM@/virtual_machines/@VMREF@/')"
if grep -q '"description": *"@DESC@"' @BOXDIR@/self.json; then echo DESC_MATCH; else echo DESC_MISMATCH; fi
ARM

cat > "$TMPD/device_probe.sh" <<'PROBE'
set -u
echo "== os";        . /etc/os-release 2>/dev/null && echo "OS=$PRETTY_NAME"; uname -r
echo "== cpu";       echo "NPROC=$(nproc)"; lscpu 2>/dev/null | grep -E '^(Model name|Socket|Core|Thread|CPU\(s\))'
echo "== memory";    free -g 2>/dev/null | head -2
echo "== disk";      df -h / /root 2>/dev/null
echo "== python";    python3 --version 2>&1
echo "== devices";   ls -l /dev/kfd /dev/dri 2>&1 | head -12
[ -e /dev/kfd ] && echo KFD_PRESENT || echo KFD_ABSENT
lsmod 2>/dev/null | grep -E '^amdgpu' | head -1
echo "== rocm";      ls -d /opt/rocm* 2>&1 | head -3
if command -v rocm-smi > /dev/null 2>&1; then echo ROCM_SMI_PRESENT; rocm-smi --showproductname 2>&1 | head -20; else echo ROCM_SMI_ABSENT; fi
if command -v rocminfo > /dev/null 2>&1; then
    echo ROCMINFO_PRESENT
    # The agent Name: field only. A bare `grep -Eo 'gfx[0-9a-f]+'` also matched
    # a stray "gfx9" on the MI300X (2026-09-11) and counted two archs.
    rocminfo 2>/dev/null | awk '$1 == "Name:" && $2 ~ /^gfx[0-9a-f]+$/ {print "GFX=" $2}' | sort -u
else
    echo ROCMINFO_ABSENT
fi
echo "== runtimes"
if command -v docker > /dev/null 2>&1 && docker info > /dev/null 2>&1; then echo DOCKER_OK; docker --version; fi
if command -v podman > /dev/null 2>&1 && podman info > /dev/null 2>&1; then echo PODMAN_OK; podman --version; fi
for t in setsid timeout curl tar gzip sha256sum; do echo "TOOL_$t=$(command -v "$t" || echo ABSENT)"; done
PROBE

cat > "$TMPD/remote_unpack.sh.template" <<'REMOTE_UNPACK'
set -eu
cd /root
got=$(sha256sum extra_src.tgz | awk '{print $1}')
if [ "$got" != "@SHA@" ]; then echo "ARCHIVE SHA MISMATCH: sent @SHA@ got $got"; exit 9; fi
echo ARCHIVE-SHA-OK
rm -rf /root/mojolearn /root/gemm_leg_out /root/gemm_leg.done
mkdir -p /root/mojolearn
tar -xzf extra_src.tgz -C /root/mojolearn
rm -f extra_src.tgz
echo "UNPACKED $(find /root/mojolearn -type f | wc -l) files"
REMOTE_UNPACK

cat > "$TMPD/remote_start.sh.template" <<'REMOTE_START'
set -u
rm -f /root/gemm_leg.done /root/gemm_leg_console.log
mkdir -p /root/gemm_leg_out /root/mojolearn
RT="@RUNTIME@"
if [ "$RT" = docker ] || [ "$RT" = podman ]; then
    "$RT" rm -f mojolearn-leg > /dev/null 2>&1
    setsid nohup sh -c '"$0" run --rm --name mojolearn-leg --device /dev/kfd --device /dev/dri \
        --security-opt seccomp=unconfined --ipc=host --network host \
        -e HOME=/root -v /root:/root -w /root/mojolearn @IMAGE@ \
        timeout -k 30 @WORK@ sh /root/gemm_leg.sh; rc=$?; "$0" rm -f mojolearn-leg > /dev/null 2>&1; \
        mkdir -p /root/gemm_leg_out; echo "body_exit=$rc" >> /root/gemm_leg_out/leg.txt' "$RT" \
        > /root/gemm_leg_console.log 2>&1 < /dev/null &
else
    setsid nohup sh -c 'timeout -k 30 @WORK@ sh /root/gemm_leg.sh; rc=$?; mkdir -p /root/gemm_leg_out; echo "body_exit=$rc" >> /root/gemm_leg_out/leg.txt' \
        > /root/gemm_leg_console.log 2>&1 < /dev/null &
fi
echo "REMOTE_PID=$!"
REMOTE_START

cat > "$TMPD/pull_start.sh.template" <<'PULL'
set -u
rm -f /root/mojolearn-pull.done
setsid nohup sh -c 't0=$(date +%s); "$0" pull @IMAGE@ > /root/mojolearn-pull.log 2>&1; rc=$?; echo "pull_exit=$rc pull_seconds=$(( $(date +%s) - t0 ))" > /root/mojolearn-pull.done' "@RUNTIME@" \
    > /dev/null 2>&1 < /dev/null &
echo PULL_STARTED
PULL

cat > "$TMPD/deadman.sh.template" <<'DEADMAN'
#!/bin/sh
# Written by tools/hotaisle_leg.sh. RUNS ON THIS MAC, DETACHED. Ends the VM
# that leg created if the leg is no longer here to do it. Keyed by the VM ref
# the leg writes beside this file as soon as the create returns it. The key
# is in the 0600 curl config beside this file and in no argv.
set -u
trap '' HUP INT TERM
D='@DMDIR@'
L="$D/deadman.log"
now() { date -u +%Y-%m-%dT%H:%M:%SZ; }
while [ "$(date +%s)" -lt @DEADLINE@ ]; do sleep 20; done
echo "$(now) dead-man firing" >> "$L"
ref=""
[ -s "$D/vm_ref.txt" ] && ref="$(cat "$D/vm_ref.txt")"
if [ -z "$ref" ]; then
    echo "mac_deadman_fired $(now) no VM ref was recorded; nothing to delete" >> '@RECORD@'
    rm -f "$D/curlrc"; exit 0
fi
c=$(curl -K "$D/curlrc" --max-time 60 -o "$D/vm.json" -w '%{http_code}' "@API@/teams/@TEAM@/virtual_machines/$ref/" 2>> "$L")
if [ "$c" = 200 ]; then
    d=$(python3 "$D/j.py" desc "$D/vm.json")
    if [ -n "$d" ] && [ "$d" != '@DESC@' ]; then
        echo "mac_deadman_fired $(now) REFUSED ref=$ref: description '$d' is not '@DESC@'" >> '@RECORD@'
        rm -f "$D/curlrc"; exit 0
    fi
    n=1
    while [ "$n" -le 6 ]; do
        c=$(curl -K "$D/curlrc" --max-time 900 -o /dev/null -w '%{http_code}' -X DELETE "@API@/teams/@TEAM@/virtual_machines/$ref/?force=true" 2>> "$L")
        echo "$(now) DELETE $ref attempt $n -> $c" >> "$L"
        case "$c" in 2*|404) break ;; esac
        sleep 15; n=$((n + 1))
    done
fi
gone=0; n=1; g=""; l=""; inl=""
while [ "$n" -le 40 ]; do
    g=$(curl -K "$D/curlrc" --max-time 60 -o /dev/null -w '%{http_code}' "@API@/teams/@TEAM@/virtual_machines/$ref/" 2>> "$L")
    l=$(curl -K "$D/curlrc" --max-time 60 -o "$D/list.json" -w '%{http_code}' "@API@/teams/@TEAM@/virtual_machines/" 2>> "$L")
    inl=$(python3 "$D/j.py" inlist "$D/list.json" "$ref")
    if [ "$g" = 404 ] || { [ "$l" = 200 ] && [ "$inl" = no ]; }; then gone=1; break; fi
    sleep 15; n=$((n + 1))
done
echo "mac_deadman_fired $(now) ref=$ref verified_gone=$gone get=$g list=$l listed=$inl" >> '@RECORD@'
if [ "$gone" = 1 ] && grep -qx 'nonce=@NONCE@' '@SLOT@/owner' 2>/dev/null && ! kill -0 @LEGPID@ 2>/dev/null; then
    rm -rf '@SLOT@' && echo "mac_deadman released @SLOT@" >> '@RECORD@'
fi
rm -f "$D/curlrc"
DEADMAN

# ---- the 2gpu scripts (the rest of the templates serve both specs) ----
cat > "$TMPD/gpu_map.sh" <<'GPUMAP'
# Written by tools/hotaisle_leg.sh. RUNS ON THE 2x MI300X VM AS ROOT: which GPU
# is which. KFD topology GPU nodes in node order (the order ROCr enumerates),
# the DRM render nodes, rocm-smi --showbus. The runner parses the KFD, DRM and
# SMI lines; everything else is the record.
set -u
echo "== kfd topology: KFD <node> <drm_render_minor> <pci address>"
if [ -d /sys/class/kfd/kfd/topology/nodes ]; then
    for n in $(ls /sys/class/kfd/kfd/topology/nodes | sort -n); do
        d=/sys/class/kfd/kfd/topology/nodes/$n
        gid=$(cat "$d/gpu_id" 2>/dev/null)
        if [ -z "$gid" ] || [ "$gid" = 0 ]; then continue; fi
        minor=$(awk '$1 == "drm_render_minor" {print $2}' "$d/properties" 2>/dev/null)
        loc=$(awk '$1 == "location_id" {print $2}' "$d/properties" 2>/dev/null)
        dom=$(awk '$1 == "domain" {print $2}' "$d/properties" 2>/dev/null)
        case "$dom" in ''|*[!0-9]*) dom=0 ;; esac
        case "$loc" in
            ''|*[!0-9]*) bdf=unknown ;;
            *) bdf=$(printf '%04x:%02x:%02x.%x' "$dom" $((loc >> 8 & 255)) $((loc >> 3 & 31)) $((loc & 7))) ;;
        esac
        echo "KFD $n ${minor:-unknown} $bdf"
    done
else
    echo "KFD_ABSENT"
fi
echo "== drm render nodes: DRM <minor> <node> <driver> <pci address>"
for r in /sys/class/drm/renderD*; do
    [ -e "$r/device" ] || continue
    name=$(basename "$r")
    drv=$(basename "$(readlink -f "$r/device/driver" 2>/dev/null)" 2>/dev/null)
    bdf=$(basename "$(readlink -f "$r/device" 2>/dev/null)" 2>/dev/null)
    echo "DRM ${name#renderD} $name ${drv:-unknown} ${bdf:-unknown}"
done
echo "== rocm-smi --showbus: SMI <index> <pci address>"
if command -v rocm-smi > /dev/null 2>&1; then
    rocm-smi --showbus > /tmp/mojolearn-showbus.txt 2>&1
    sed 's/^/RAW /' /tmp/mojolearn-showbus.txt
    awk '/^GPU\[/ && /PCI Bus/ { i = $1; gsub(/[^0-9]/, "", i); print "SMI", i, tolower($NF) }' /tmp/mojolearn-showbus.txt
    rm -f /tmp/mojolearn-showbus.txt
else
    echo "SMI_ABSENT"
fi
echo "== devices"
ls -l /dev/kfd /dev/dri 2>&1
GPUMAP

cat > "$TMPD/pin_probe.sh" <<'PINPROBE'
# Written by tools/hotaisle_leg.sh. RUNS INSIDE ONE CONTAINER on the 2x MI300X
# VM: what HIP and rocminfo see through this container's devices and its
# ROCR_VISIBLE_DEVICES / HIP_VISIBLE_DEVICES. HIP is asked directly (ctypes
# into libamdhip64), because HIP is what a Mojo GPU program opens.
set -u
echo "PROBE_ENV ROCR_VISIBLE_DEVICES=${ROCR_VISIBLE_DEVICES:-<unset>} HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-<unset>}"
echo "PROBE_DRI $(ls /dev/dri 2>&1 | tr '\n' ' ')"
if command -v rocminfo > /dev/null 2>&1; then
    rocminfo > /tmp/mojolearn-rocminfo.txt 2>&1
    echo "ROCMINFO_EXIT $?"
    awk '
        $1 == "Agent" { a++ }
        $1 == "Name:" && !(a in nm) { nm[a] = $2 }
        $1 == "Device" && $2 == "Type:" { ty[a] = $3 }
        $1 == "BDFID:" { bd[a] = $2 }
        END {
            for (i = 1; i <= a; i++) {
                if (ty[i] != "GPU") continue
                if (bd[i] == "") { printf "ROCMINFO_GPU bdf=unknown name=%s\n", nm[i]; continue }
                b = bd[i] + 0
                printf "ROCMINFO_GPU bdf=%02x:%02x.%x name=%s bdfid=%s\n", int(b / 256) % 256, int(b / 8) % 32, b % 8, nm[i], bd[i]
            }
        }' /tmp/mojolearn-rocminfo.txt
else
    echo "ROCMINFO_ABSENT"
fi
PY=""
for c in python3 python; do
    if command -v "$c" > /dev/null 2>&1; then PY=$c; break; fi
done
if [ -z "$PY" ]; then
    echo "HIP_UNAVAILABLE no python in the image"
else
    "$PY" - <<'PY'
import ctypes
lib = None
for p in ("libamdhip64.so", "/opt/rocm/lib/libamdhip64.so", "libamdhip64.so.6", "/opt/rocm/lib/libamdhip64.so.6"):
    try:
        lib = ctypes.CDLL(p)
        break
    except OSError:
        pass
if lib is None:
    print("HIP_UNAVAILABLE libamdhip64 does not load")
else:
    n = ctypes.c_int(0)
    rc = lib.hipGetDeviceCount(ctypes.byref(n))
    count = n.value if rc == 0 else 0
    print("HIP_COUNT rc=%d count=%d" % (rc, count))
    for i in range(count):
        buf = ctypes.create_string_buffer(64)
        r = lib.hipGetDevicePciBusId(buf, 64, i)
        print("HIP_DEVICE index=%d rc=%d bus=%s" % (i, r, buf.value.decode("ascii", "replace").lower()))
PY
fi
PINPROBE

cat > "$TMPD/remote_unpack2.sh.template" <<'REMOTE_UNPACK2'
set -eu
cd /root
got=$(sha256sum extra_src.tgz | awk '{print $1}')
if [ "$got" != "@SHA@" ]; then echo "ARCHIVE SHA MISMATCH: sent @SHA@ got $got"; exit 9; fi
echo ARCHIVE-SHA-OK
rm -rf /root/leg-a /root/leg-b /root/mojolearn-pin /root/mojolearn /root/gemm_leg_out /root/gemm_leg.done
for x in a b; do
    mkdir -p /root/leg-$x/mojolearn
    tar -xzf extra_src.tgz -C /root/leg-$x/mojolearn
    echo "UNPACKED_$x $(find /root/leg-$x/mojolearn -type f | wc -l) files"
done
rm -f extra_src.tgz
REMOTE_UNPACK2

cat > "$TMPD/remote_start2.sh.template" <<'REMOTE_START2'
# Generated by tools/hotaisle_leg.sh. RUNS ON THE 2x MI300X VM AS ROOT: two
# bodies, one container per GPU, each with its own /root (host /root/leg-a or
# /root/leg-b), devices, visible-devices variables, timeout, sentinel and
# body_exit.
set -u
RT="@RUNTIME@"
for x in a b; do
    rm -f /root/leg-$x/gemm_leg.done /root/leg-$x/gemm_leg_console.log
    mkdir -p /root/leg-$x/gemm_leg_out /root/leg-$x/mojolearn
    "$RT" rm -f mojolearn-leg-$x > /dev/null 2>&1
done
setsid nohup sh -c '"$0" run --rm --name mojolearn-leg-a --device /dev/kfd @DEVICES_A@ \
    -e ROCR_VISIBLE_DEVICES=@ROCR_A@ -e HIP_VISIBLE_DEVICES=@HIP_A@ \
    --security-opt seccomp=unconfined --ipc=host --network host \
    -e HOME=/root -v /root/leg-a:/root -w /root/mojolearn @IMAGE@ \
    timeout -k 30 @WORK@ sh /root/gemm_leg.sh; rc=$?; "$0" rm -f mojolearn-leg-a > /dev/null 2>&1; \
    mkdir -p /root/leg-a/gemm_leg_out; echo "body_exit=$rc" >> /root/leg-a/gemm_leg_out/leg.txt' "$RT" \
    > /root/leg-a/gemm_leg_console.log 2>&1 < /dev/null &
echo "REMOTE_PID_A=$!"
setsid nohup sh -c '"$0" run --rm --name mojolearn-leg-b --device /dev/kfd @DEVICES_B@ \
    -e ROCR_VISIBLE_DEVICES=@ROCR_B@ -e HIP_VISIBLE_DEVICES=@HIP_B@ \
    --security-opt seccomp=unconfined --ipc=host --network host \
    -e HOME=/root -v /root/leg-b:/root -w /root/mojolearn @IMAGE@ \
    timeout -k 30 @WORK@ sh /root/gemm_leg.sh; rc=$?; "$0" rm -f mojolearn-leg-b > /dev/null 2>&1; \
    mkdir -p /root/leg-b/gemm_leg_out; echo "body_exit=$rc" >> /root/leg-b/gemm_leg_out/leg.txt' "$RT" \
    > /root/leg-b/gemm_leg_console.log 2>&1 < /dev/null &
echo "REMOTE_PID_B=$!"
REMOTE_START2

cat > "$TMPD/test_body.sh.template" <<'TEST_BODY'
#!/bin/sh
# Generated by tools/hotaisle_leg.sh --test-2gpu. RUNS IN ONE PINNED CONTAINER
# on the 2x MI300X VM (GPU @G@): what this container sees, a hold of about two
# minutes while the other container does the same, then what it sees again.
set -u
OUT=/root/gemm_leg_out
mkdir -p "$OUT"
{
  echo "vendor=amd"
  echo "provider=hotaisle"
  echo "size=@SIZE@"
  echo "runtime=@RUNTIME@"
  echo "image=@IMAGE@"
  echo "test_2gpu_body_gpu=@G@"
  echo "started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "started_epoch=$(date +%s)"
  echo "target_column=amd"
} > "$OUT/leg.txt"
sh /root/pin_probe.sh > "$OUT/visible_start.txt" 2>&1
{ rocm-smi --showproductname --showbus 2>&1 || echo "rocm-smi did not answer"; } > "$OUT/rocm_smi.txt"
{ rocminfo 2>&1 || echo "rocminfo did not answer"; } > "$OUT/rocminfo.txt"
cat "$OUT/visible_start.txt"
i=0
while [ "$i" -lt @HOLDTICKS@ ]; do
    sleep 10
    i=$((i + 1))
    echo "$(date -u +%H:%M:%S) holding" >> "$OUT/hold.txt"
done
sh /root/pin_probe.sh > "$OUT/visible_end.txt" 2>&1
cat "$OUT/visible_end.txt"
echo "finished=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$OUT/leg.txt"
echo "finished_epoch=$(date +%s)" >> "$OUT/leg.txt"
: > /root/gemm_leg.done
echo TEST_BODY_DONE
TEST_BODY

subst() {  # <template> <out>: replace every placeholder, then prove none survived
  sed -e "s|@COMMIT@|$COMMIT|g" -e "s|@CARDFULL@|$CARD_FULL|g" -e "s|@DUMP@|$LEG_DUMP|g" \
      -e "s|@SIZE@|$SIZE_LABEL|g" -e "s|@RUNTIME@|${RUNTIME:-native}|g" -e "s|@IMAGE@|$IMAGE|g" \
      -e "s|@GATES@|$GATES|g" -e "s|@BARE@|$BARE|g" -e "s|@GPUARCHS@|${S_ARCHS:-$GPU_ARCHS}|g" \
      -e "s|@WORK@|${WORK_SECONDS:-0}|g" -e "s|@SHA@|$BUNDLE_SHA|g" -e "s|@SECS@|${S_SECS:-60}|g" \
      -e "s|@VMREF@|${VMREF:-DRYRUN_REF}|g" -e "s|@API@|$API|g" -e "s|@TEAM@|$TEAM|g" \
      -e "s|@DESC@|${DESC:-mojolearn:dry:run}|g" -e "s|@BOXRC@|$BOX_RC|g" -e "s|@BOXDIR@|$BOX_DIR|g" \
      -e "s|@DMDIR@|${DEADMAN_DIR:-/tmp/dry}|g" -e "s|@DEADLINE@|${S_DEADLINE:-0}|g" \
      -e "s|@RECORD@|$OUT/deadman.txt|g" -e "s|@NONCE@|$NONCE|g" -e "s|@SLOT@|${SLOT:-/tmp/dry-slot}|g" \
      -e "s|@LEGPID@|$$|g" \
      "$1" > "$2"
  if grep -q '@[A-Z][A-Z_]*@' "$2"; then grep -n '@[A-Z][A-Z_]*@' "$2" | sed 's/^/           /'; return 1; fi
  return 0
}
check_posix() {  # <file> <label>
  sh -n "$1" 2> "$TMPD/sh_n.err" || { rbad "$2 is not valid sh: $(head -2 "$TMPD/sh_n.err")"; return 1; }
  if command -v dash > /dev/null 2>&1; then
    dash -n "$1" 2> "$TMPD/sh_n.err" || { rbad "$2 is not valid dash: $(head -2 "$TMPD/sh_n.err")"; return 1; }
  fi
  awk '!/^[[:space:]]*#/ && (/exec -a/ || /\[\[/ || /(^|[;{ \t])local / || /<\(/ || /(^|[;{ \t])function / || /(^|[;{ \t])source / || /echo -e/) { print FNR ": " $0 }' "$1" > "$TMPD/bashisms"
  [ -s "$TMPD/bashisms" ] && { rbad "$2 has a BASHISM (the VM runs dash):"; sed 's/^/           /' "$TMPD/bashisms"; return 1; }
  return 0
}
# ---- the 2gpu helpers ----
subst2() {  # <template> <out> [<gpu index>]: the 2gpu placeholders, then subst
  sed -e "s|@DEVICES_A@|${PIN_DEVICES[0]}|g" -e "s|@ROCR_A@|${PIN_ROCR[0]}|g" -e "s|@HIP_A@|${PIN_HIP[0]}|g" \
      -e "s|@DEVICES_B@|${PIN_DEVICES[1]}|g" -e "s|@ROCR_B@|${PIN_ROCR[1]}|g" -e "s|@HIP_B@|${PIN_HIP[1]}|g" \
      -e "s|@G@|${3:-0}|g" -e "s|@HOLDTICKS@|$TEST2_HOLD_TICKS|g" "$1" > "$2.pre" || return 1
  subst "$2.pre" "$2"
  local r=$?
  rm -f "$2.pre"
  return "$r"
}
insert_pin_lines() {  # <body script> <key=value file> <out>: echo lines right after its one target_column=amd line
  sed 's/.*/  echo "&"/' "$2" > "$2.echo"
  awk -v f="$2.echo" '{ print } $0 == "  echo \"target_column=amd\"" { while ((getline l < f) > 0) print l; close(f); n++ } END { exit (n == 1 ? 0 : 1) }' "$1" > "$3"
}
set_example_pins() {  # the dry run and the local checks: a plausible map and each GPU's first candidate, shown as EXAMPLE
  PIN_MODE=render; GPU_MAP_SOURCE=example; G_RENDER=(renderD128 renderD129); G_BDF=(0000:c1:00.0 0000:c2:00.0)
  PIN_DEVICES=("--device /dev/dri/renderD128" "--device /dev/dri/renderD129"); PIN_ROCR=(0 1); PIN_HIP=(0 1)
  PIN_BUS=(0000:c1:00.0 0000:c2:00.0); PIN_VIA=(example example); PIN_CAND=(1 1)
  RUNTIME="${RUNTIME_WANT/auto/docker}"
}
gen_calibrate() {  # <script out>: one probe container per GPU and candidate, in order; the list goes to $TMPD/pin_cands.txt
  local g k c r h dev cands seen rt="${RUNTIME:-docker}"
  : > "$TMPD/pin_cands.txt"
  {
    echo "# Generated by tools/hotaisle_leg.sh. RUNS ON THE 2x MI300X VM AS ROOT after the image pull:"
    echo "# one probe container per GPU and candidate pin, each under timeout(1)."
    echo "set -u"
    for g in 0 1; do
      if [ "$PIN_MODE" = render ]; then dev="/dev/dri/${G_RENDER[$g]}"; cands="$g,$g 0,0 $g,0"; else dev=/dev/dri; cands="$g,$g $g,0"; fi
      k=0; seen=" "
      for c in $cands; do
        case "$seen" in *" $c "*) continue ;; esac
        seen="$seen$c "; k=$((k + 1)); r=${c%,*}; h=${c#*,}
        echo "$g $k $r $h $dev" >> "$TMPD/pin_cands.txt"
        echo "echo 'BEGIN $g $k'"
        echo "timeout -k 10 120 $rt run --rm --name mojolearn-pin-$g-$k --device /dev/kfd --device $dev -e ROCR_VISIBLE_DEVICES=$r -e HIP_VISIBLE_DEVICES=$h --security-opt seccomp=unconfined --ipc=host -v /root/mojolearn-pin:/pin:ro $IMAGE sh /pin/probe.sh 2>&1"
        echo "echo \"END $g $k rc=\$?\""
        echo "$rt rm -f mojolearn-pin-$g-$k > /dev/null 2>&1"
      done
    done
  } > "$1"
}
pin_kv() {  # <gpu index> [body]: the pin as key=value lines (with body: also gpu_index and vm_share, for the box's leg.txt)
  local g=$1
  if [ "${2:-}" = body ]; then
    echo "gpu_index=$g"
    echo "vm_share=one 2x MI300X VM, two bodies, one pinned container per GPU; never mixed with 1x MI300X VM rows"
  fi
  echo "pin_mode=$PIN_MODE"
  echo "pin_devices=/dev/kfd ${PIN_DEVICES[$g]#--device }"
  echo "rocr_visible_devices=${PIN_ROCR[$g]}"
  echo "hip_visible_devices=${PIN_HIP[$g]}"
  echo "pin_candidate=${PIN_CAND[$g]}"
  echo "pin_verified_by=${PIN_VIA[$g]}"
  echo "pin_bus=${PIN_BUS[$g]}"
  echo "gpu_map_source=$GPU_MAP_SOURCE"
  echo "pin_host_render=${G_RENDER[$g]:-unknown}"
  echo "pin_host_bdf=${G_BDF[$g]:-unknown}"
  if [ "$PIN_MODE" != render ]; then
    echo "gpu_pinning_note=the renderD mapping could not be read on the box: /dev/dri passed whole to both containers, pinning rests on ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES as verified by the pin probe"
  fi
}
bdf_tail() { printf '%s\n' "$1" | tr 'A-F' 'a-f' | sed -n 's/^.*\([0-9a-f][0-9a-f]:[0-9a-f][0-9a-f]\.[0-9a-f]\)$/\1/p'; }
probe_read() {  # <one pin probe's output>: sets PR_COUNT, PR_BUS, PR_VIA
  PR_COUNT=-1; PR_BUS=""; PR_VIA=none
  [ -f "$1" ] || return 0
  if grep -q '^HIP_COUNT rc=0 ' "$1"; then
    PR_VIA=hip
    PR_COUNT=$(sed -n 's/^HIP_COUNT rc=0 count=\([0-9]*\).*/\1/p' "$1" | head -1)
    PR_BUS=$(sed -n 's/^HIP_DEVICE index=0 rc=0 bus=\([0-9a-f:.]*\).*/\1/p' "$1" | head -1)
  elif grep -q '^HIP_COUNT ' "$1"; then
    PR_VIA=hip; PR_COUNT=0   # HIP answered with an error: no device in this view
  elif grep -q '^HIP_UNAVAILABLE' "$1"; then
    PR_VIA=rocminfo
    PR_COUNT=$(grep -c '^ROCMINFO_GPU ' "$1")
    PR_BUS=$(sed -n 's/^ROCMINFO_GPU bdf=\([0-9a-f:.]*\) .*/\1/p' "$1" | head -1)
  fi
  PR_COUNT=${PR_COUNT:-0}
  return 0
}
map_gpus() {  # <gpu_map.txt>: sets PIN_MODE, GPU_MAP_SOURCE, GPU_COUNT_SEEN, G_RENDER, G_BDF
  local kn dn sn g m b r
  G_RENDER=("" ""); G_BDF=("" ""); PIN_MODE=dri; GPU_MAP_SOURCE=none
  kn=$(grep -c '^KFD [0-9]' "$1")
  dn=$(awk '$1 == "DRM" && $4 == "amdgpu"' "$1" | grep -c .)
  sn=$(grep -c '^SMI [0-9]' "$1")
  GPU_COUNT_SEEN="kfd=$kn drm_amdgpu=$dn rocm_smi=$sn"
  if [ "$kn" = 2 ]; then
    GPU_MAP_SOURCE=kfd; g=0
    while read -r _ _ m b; do
      G_BDF[g]=$b
      if [ -n "$(awk -v m="$m" -v b="$b" '$1 == "DRM" && $2 == m && $4 == "amdgpu" && tolower($5) == tolower(b)' "$1")" ]; then
        G_RENDER[g]="renderD$m"; GPU_MAP_SOURCE=kfd+drm
      fi
      g=$((g + 1))
    done < <(grep '^KFD [0-9]' "$1")
  elif [ "$dn" = 2 ]; then
    GPU_MAP_SOURCE=drm; g=0
    while read -r _ m r _ b; do G_RENDER[g]=$r; G_BDF[g]=$b; g=$((g + 1)); done < <(awk '$1 == "DRM" && $4 == "amdgpu"' "$1" | sort -k2,2n)
  elif [ "$sn" = 2 ]; then
    GPU_MAP_SOURCE=rocm-smi; g=0
    while read -r _ _ b; do G_BDF[g]=$b; g=$((g + 1)); done < <(grep '^SMI [0-9]' "$1")
  fi
  for g in 0 1; do
    case "${G_RENDER[$g]}" in renderD[0-9]*) case "${G_RENDER[$g]#renderD}" in *[!0-9]*) G_RENDER[g]="" ;; esac ;; *) G_RENDER[g]="" ;; esac
    case "${G_BDF[$g]}" in ''|*[!0-9A-Fa-f:.]*) G_BDF[g]="" ;; esac
  done
  if [ -n "${G_RENDER[0]}" ] && [ -n "${G_RENDER[1]}" ] && [ "${G_RENDER[0]}" != "${G_RENDER[1]}" ]; then PIN_MODE=render; fi
  return 0
}
calibrate_pins() {  # <calibration output>: sets PIN_* per GPU; 0 only when both GPUs pinned, to different addresses
  local g k r h dev want _g
  for g in 0 1; do
    PIN_ROCR[g]=""; PIN_HIP[g]=""; PIN_DEVICES[g]=""; PIN_BUS[g]=""; PIN_VIA[g]=""; PIN_CAND[g]=""
    want=$(bdf_tail "${G_BDF[$g]}")
    while read -r _g k r h dev; do
      [ "$_g" = "$g" ] || continue
      awk -v g="$g" -v k="$k" '$1 == "BEGIN" && $2 == g && $3 == k { on = 1; next } $1 == "END" && $2 == g && $3 == k { on = 0 } on' "$1" > "$TMPD/pin_cand_out.txt"
      probe_read "$TMPD/pin_cand_out.txt"
      echo "gpu$g candidate=$k devices=/dev/kfd,$dev rocr=$r hip=$h via=$PR_VIA count=$PR_COUNT bus=${PR_BUS:-none} host_bdf=${G_BDF[$g]:-unknown}" >> "$OUT/pin_decisions.txt"
      if [ "$PR_COUNT" != 1 ] || [ -z "$(bdf_tail "$PR_BUS")" ]; then continue; fi
      if [ -n "$want" ] && [ "$(bdf_tail "$PR_BUS")" != "$want" ]; then continue; fi
      PIN_ROCR[g]=$r; PIN_HIP[g]=$h; PIN_DEVICES[g]="--device $dev"; PIN_BUS[g]=$PR_BUS; PIN_VIA[g]=$PR_VIA; PIN_CAND[g]=$k
      echo "gpu$g PINNED to candidate $k" >> "$OUT/pin_decisions.txt"
      break
    done < "$TMPD/pin_cands.txt"
  done
  [ -n "${PIN_BUS[0]}" ] && [ -n "${PIN_BUS[1]}" ] && [ "$(bdf_tail "${PIN_BUS[0]}")" != "$(bdf_tail "${PIN_BUS[1]}")" ]
}
legs_write() {  # the slot's legs file, which `status` prints
  { [ "$SPEC" = 2gpu ] && [ -n "$SLOT" ] && [ -d "$SLOT" ]; } || return 0
  grep -qx "nonce=$NONCE" "$SLOT/owner" 2>/dev/null || return 0
  {
    printf 'gpu0\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LANE_A" "${LEG_EXTRA:-runner-test-body}" "$REAL_OUT" "${B_SINCE[0]}" "${B_STATE[0]}" "${B_EXIT[0]}"
    printf 'gpu1\t%s\t%s\t%s\t%s\t%s\t%s\n' "$LANE_B" "${LEG_EXTRA_B:-runner-test-body}" "$REAL_OUT_B" "${B_SINCE[1]}" "${B_STATE[1]}" "${B_EXIT[1]}"
  } > "$SLOT/legs.tmp" && mv -f "$SLOT/legs.tmp" "$SLOT/legs"
}
fetch_body2() {  # <gpu index>: /root/leg-<x>/gemm_leg_out -> <its out>/remote/, its exits into its leg.txt; once
  local g=$1 x o secs left v k lsha rsha
  x=${LETTERS[$g]}; o=${OUTS[$g]}
  [ "${FETCHED[$g]}" = 1 ] && return 0
  FETCHED[g]=1
  echo "body=${B_STATE[$g]}" >> "$o/leg.txt"
  left=$((DEADLINE_EPOCH - $(date +%s) - 90)); secs=600
  [ "$left" -lt "$secs" ] && secs=$left
  [ "$secs" -ge 60 ] || secs=60
  mkdir -p "$o/remote"
  if with_deadline "$secs" /dev/null "${SSHN[@]}" "sudo -n sh -c 'cd /root/leg-$x/gemm_leg_out && tar czf - --exclude=./tools-venv .'" > "$TMPD/remote_$x.tgz" \
     && tar xzf "$TMPD/remote_$x.tgz" -C "$o/remote"; then
    log "GPU $g: fetched /root/leg-$x/gemm_leg_out -> $o/remote/ ($(wc -c < "$TMPD/remote_$x.tgz" | tr -d ' ') bytes)"
  else
    FETCH_RED=1; log "GPU $g: FETCH FAILED or hit its ${secs}s bound"
    echo "fetch=FAILED" >> "$o/leg.txt"
  fi
  rm -rf "$o/remote/tools-venv"
  with_deadline 60 /dev/null "${SSHN[@]}" "sudo -n cat /root/leg-$x/gemm_leg_console.log /root/mojolearn-pull.done" > "$o/remote_console.log" 2>/dev/null || true
  if [ "$SHIPS_SOURCE" = 1 ]; then
    lsha=$(cat "$OUT/source_sha256_local.txt" 2>/dev/null); rsha=$(cat "$o/remote/source_sha256.txt" 2>/dev/null)
    if [ -n "$lsha" ] && [ "$lsha" = "$rsha" ]; then
      echo "source_sha256_match=yes" >> "$o/leg.txt"
    else
      echo "source_sha256_match=NO local=$lsha remote=$rsha" >> "$o/leg.txt"; FETCH_RED=1
    fi
  fi
  for k in pixi_install_exit pixi_install_seconds device_check_exit card_exit extra_exit body_exit; do
    v=$(sed -n "s/^$k=//p" "$o/remote/leg.txt" 2>/dev/null | tail -1)
    echo "remote_$k=${v:-<absent>}" >> "$o/leg.txt"
    log "  GPU $g $k=${v:-<absent>}"
  done
  v=$(sed -n 's/^body_exit=//p' "$o/remote/leg.txt" 2>/dev/null | tail -1)
  B_EXIT[g]=${v:--}
  echo "finished=$(utc)" >> "$o/leg.txt"
  legs_write
}
test2_verdict() {  # --test-2gpu: sets TEST2_BODIES to PASS or FAIL and records every reading in both leg.txt
  local g o f v line ok=1 s0 e0 s1 e1
  for g in 0 1; do
    o=${OUTS[$g]}
    for f in visible_start visible_end; do
      probe_read "$o/remote/$f.txt"
      line="test_2gpu_gpu${g}_$f=count:$PR_COUNT bus:${PR_BUS:-none} via:$PR_VIA pinned_bus:${PIN_BUS[$g]}"
      echo "$line" >> "$o/leg.txt"; log "$line"
      if [ "$PR_COUNT" != 1 ] || [ -z "$(bdf_tail "$PR_BUS")" ] || [ "$(bdf_tail "$PR_BUS")" != "$(bdf_tail "${PIN_BUS[$g]}")" ]; then ok=0; fi
    done
    v=$(sed -n 's/^body_exit=//p' "$o/remote/leg.txt" 2>/dev/null | tail -1)
    if [ "$v" != 0 ]; then ok=0; log "GPU $g test body_exit=${v:-<absent>}"; fi
  done
  s0=$(sed -n 's/^started_epoch=//p' "${OUTS[0]}/remote/leg.txt" 2>/dev/null | tail -1)
  e0=$(sed -n 's/^finished_epoch=//p' "${OUTS[0]}/remote/leg.txt" 2>/dev/null | tail -1)
  s1=$(sed -n 's/^started_epoch=//p' "${OUTS[1]}/remote/leg.txt" 2>/dev/null | tail -1)
  e1=$(sed -n 's/^finished_epoch=//p' "${OUTS[1]}/remote/leg.txt" 2>/dev/null | tail -1)
  case "$s0:$e0:$s1:$e1" in
    *[!0-9:]*|:*|*::*|*:) line="test_2gpu_overlap=UNREADABLE gpu0=${s0:-?}..${e0:-?} gpu1=${s1:-?}..${e1:-?}"; ok=0 ;;
    *) if [ "$s0" -lt "$e1" ] && [ "$s1" -lt "$e0" ]; then line="test_2gpu_overlap=yes gpu0=$s0..$e0 gpu1=$s1..$e1"
       else line="test_2gpu_overlap=NO gpu0=$s0..$e0 gpu1=$s1..$e1"; ok=0; fi ;;
  esac
  if [ "$(bdf_tail "${PIN_BUS[0]}")" = "$(bdf_tail "${PIN_BUS[1]}")" ]; then ok=0; line="$line pinned_buses=SAME"; fi
  TEST2_BODIES=FAIL; [ "$ok" = 1 ] && TEST2_BODIES=PASS
  for o in "$OUT" "$OUT_B"; do printf '%s\ntest_2gpu_bodies=%s\n' "$line" "$TEST2_BODIES" >> "$o/leg.txt"; done
  log "$line; bodies $TEST2_BODIES"
}

_ok=1
for _t in remote_body watchdog watchdog_arm remote_unpack remote_start pull_start deadman; do
  if subst "$TMPD/$_t.sh.template" "$TMPD/check_$_t.sh"; then
    check_posix "$TMPD/check_$_t.sh" "$_t.sh" || _ok=0
  else
    rbad "UNSUBSTITUTED PLACEHOLDER in $_t.sh"; _ok=0
  fi
done
check_posix "$TMPD/device_probe.sh" device_probe.sh || _ok=0
[ "$_ok" = 1 ] && rok "the remote body, watchdog, arm, unpack, start, pull and Mac dead-man scripts substitute cleanly and pass sh -n, dash -n and the bashism scan"
if [ "$SPEC" = 2gpu ]; then
  _ok2=1
  set_example_pins
  WORK_SECONDS=$((MINUTES * 60 - FETCH_RESERVE - 600))   # EXAMPLE only; the real bound is computed when the bodies start
  for _t in remote_unpack2 remote_start2 test_body; do
    if subst2 "$TMPD/$_t.sh.template" "$TMPD/check_$_t.sh" 0; then
      check_posix "$TMPD/check_$_t.sh" "$_t.sh" || _ok2=0
    else
      rbad "UNSUBSTITUTED PLACEHOLDER in $_t.sh"; _ok2=0
    fi
  done
  check_posix "$TMPD/gpu_map.sh" gpu_map.sh || _ok2=0
  check_posix "$TMPD/pin_probe.sh" pin_probe.sh || _ok2=0
  gen_calibrate "$TMPD/check_pin_calibrate.sh"
  check_posix "$TMPD/check_pin_calibrate.sh" pin_calibrate.sh || _ok2=0
  pin_kv 0 body > "$TMPD/check_pin_0.txt"
  if insert_pin_lines "$TMPD/check_remote_body.sh" "$TMPD/check_pin_0.txt" "$TMPD/check_remote_body_pinned.sh" \
     && insert_pin_lines "$TMPD/check_test_body.sh" "$TMPD/check_pin_0.txt" "$TMPD/check_test_body_pinned.sh"; then
    check_posix "$TMPD/check_remote_body_pinned.sh" "remote_body.sh with pin lines" || _ok2=0
    check_posix "$TMPD/check_test_body_pinned.sh" "test_body.sh with pin lines" || _ok2=0
  else
    rbad "the pin lines found no single 'target_column=amd' line to follow in a body script"; _ok2=0
  fi
  [ "$_ok2" = 1 ] && rok "the 2gpu GPU map, pin probe, pin calibration, two-copy unpack, two-body start and test body scripts, and both body scripts with pin lines, pass sh -n, dash -n and the bashism scan"
fi
cp "$TMPD/check_remote_body.sh" "$OUT/remote_body.sh"

{
  echo "commit=$COMMIT_LINE"
  echo "commit_sha=$COMMIT"
  echo "provider=hotaisle"
  echo "vendor=amd"
  echo "team=$TEAM"
  echo "spec=$SPEC"
  echo "lane=$LANE"
  echo "minutes=$MINUTES"
  echo "gates=$GATES"
  echo "bare=$BARE"
  echo "test_watchdog=$TEST_WATCHDOG"
  echo "gpu_archs_requested=${GPU_ARCHS:-<unset>}"
  echo "target_column=amd"
  echo "runtime_requested=$RUNTIME_WANT"
  echo "image=$IMAGE"
  echo "extra=$LEG_EXTRA"
  echo "extra_sha256=$EXTRA_SHA"
  echo "bundle_bytes=$BUNDLE_BYTES"
  echo "bundle_sha256=$BUNDLE_SHA"
  echo "source_sha256_local=$(cat "$OUT/source_sha256_local.txt" 2>/dev/null)"
  echo "mode=$MODE"
  echo "started=$(utc)"
} > "$OUT/leg.txt"
if [ "$SPEC" = 2gpu ]; then
  {
    echo "size=$LEG2_LABEL"
    echo "vm_share=one 2x MI300X VM, two bodies, one pinned container per GPU; this dir is GPU 0 and holds the VM records"
    echo "gpu_index=0"
    echo "lane_gpu0=$LANE_A"
    echo "lane_gpu1=$LANE_B"
    echo "extra_gpu1=${LEG_EXTRA_B:-<test body>}"
    echo "extra_gpu1_sha256=$EXTRA_SHA_B"
    echo "out_gpu1=$REAL_OUT_B"
    echo "gpu_only_ack=${MOJOLEARN_HOTAISLE_GPU_ONLY:-0}"
    echo "test_2gpu=$TEST_2GPU"
  } >> "$OUT/leg.txt"
  {
    echo "commit=$COMMIT_LINE"
    echo "commit_sha=$COMMIT"
    echo "provider=hotaisle"
    echo "vendor=amd"
    echo "team=$TEAM"
    echo "spec=$SPEC"
    echo "size=$LEG2_LABEL"
    echo "vm_share=one 2x MI300X VM, two bodies, one pinned container per GPU; this dir is GPU 1"
    echo "gpu_index=1"
    echo "lane=$LANE"
    echo "lane_gpu1=$LANE_B"
    echo "minutes=$MINUTES"
    echo "gates=$GATES"
    echo "test_2gpu=$TEST_2GPU"
    echo "gpu_archs_requested=${GPU_ARCHS:-<unset>}"
    echo "target_column=amd"
    echo "runtime_requested=$RUNTIME_WANT"
    echo "image=$IMAGE"
    echo "extra=${LEG_EXTRA_B:-<test body>}"
    echo "extra_sha256=$EXTRA_SHA_B"
    echo "bundle_bytes=$BUNDLE_BYTES"
    echo "bundle_sha256=$BUNDLE_SHA"
    echo "source_sha256_local=$(cat "$OUT/source_sha256_local.txt" 2>/dev/null)"
    echo "mode=$MODE"
    echo "started=$(utc)"
    echo "vm_records=$REAL_OUT (slot, create, dead-man, watchdog, GPU map, pin calibration; teardown.txt and deadman.txt are copied here at the end)"
    echo "out_gpu0=$REAL_OUT"
  } > "$OUT_B/leg.txt"
fi

if [ "$MODE" = dry ] && [ "$SPEC" = 2gpu ]; then
  set_example_pins
  WORK_SECONDS=$((MINUTES * 60 - FETCH_RESERVE - 600))
  gen_calibrate "$OUT/pin_calibrate.example.sh"
  subst2 "$TMPD/remote_start2.sh.template" "$OUT/remote_start2.example.sh"
  for _g in 0 1; do
    pin_kv "$_g" body > "$TMPD/dry_pin_$_g.txt"
    if [ "$TEST_2GPU" = 1 ]; then
      subst2 "$TMPD/test_body.sh.template" "$TMPD/dry_body_$_g.sh" "$_g"
    else
      subst "$TMPD/remote_body.sh.template" "$TMPD/dry_body_$_g.sh"
    fi
    insert_pin_lines "$TMPD/dry_body_$_g.sh" "$TMPD/dry_pin_$_g.txt" "${OUTS[$_g]}/remote_body.sh"
  done
  echo
  echo "== EXAMPLE VALUES BELOW: render nodes, PCI addresses and pins are read and verified on the box, never assumed =="
  for _g in 0 1; do
    _x=${LETTERS[$_g]}
    if [ "$_g" = 0 ]; then _body=$LEG_EXTRA; _env=$EXTRA_ENV; _o=$REAL_OUT; _sha=$EXTRA_SHA
    else _body=$LEG_EXTRA_B; _env=$EXTRA_ENV_B; _o=$REAL_OUT_B; _sha=$EXTRA_SHA_B; fi
    echo
    echo "== GPU $_g body, composed =="
    echo "   host dir   /root/leg-$_x, mounted at /root in its own container (/root/mojolearn is its own source copy)"
    echo "   container  $RUNTIME run --rm --name mojolearn-leg-$_x --device /dev/kfd ${PIN_DEVICES[$_g]} -e ROCR_VISIBLE_DEVICES=${PIN_ROCR[$_g]} -e HIP_VISIBLE_DEVICES=${PIN_HIP[$_g]} --security-opt seccomp=unconfined --ipc=host --network host -e HOME=/root -v /root/leg-$_x:/root -w /root/mojolearn $IMAGE timeout -k 30 <work seconds> sh /root/gemm_leg.sh"
    if [ "$TEST_2GPU" = 1 ]; then
      echo "   body       the runner's tiny test body: pin probe, rocm-smi, rocminfo, hold $((TEST2_HOLD_TICKS * 10)) s, pin probe again"
    else
      echo "   body       /root/gemm_leg.sh: pixi install, gates, then (. /root/gemm_leg_extra_env.sh; sh /root/gemm_leg_extra.sh) > /root/gemm_leg_out/extra.log"
      echo "   extra      $_body (sha256 ${_sha:0:16})"
      echo "   env        $(grep -c '^export ' "${OUTS[$_g]}/extra_env.sh" | tr -d ' ') export(s)${_env:+: $_env}"
    fi
    echo "   pin        EXAMPLE mode $PIN_MODE: devices /dev/kfd ${PIN_DEVICES[$_g]#--device }, ROCR_VISIBLE_DEVICES=${PIN_ROCR[$_g]}, HIP_VISIBLE_DEVICES=${PIN_HIP[$_g]} (the first candidate; calibration decides)"
    echo "   evidence   /root/leg-$_x/gemm_leg_out -> $_o/remote/"
    echo
    echo "== the GPU $_g remote body (/root/leg-$_x/gemm_leg.sh; its pin lines are EXAMPLES) =="
    cat "${OUTS[$_g]}/remote_body.sh"
  done
  echo; echo "== the GPU map, read on the host =="; cat "$TMPD/gpu_map.sh"
  echo; echo "== the pin probe, run inside every candidate container and twice in each test body =="; cat "$TMPD/pin_probe.sh"
  echo; echo "== the pin calibration (EXAMPLE map: GPU 0 renderD128 at 0000:c1:00.0, GPU 1 renderD129 at 0000:c2:00.0) =="; cat "$OUT/pin_calibrate.example.sh"
  echo; echo "== the two-body start wrapper (EXAMPLE pins, EXAMPLE work bound ${WORK_SECONDS}s) =="; cat "$OUT/remote_start2.example.sh"
  echo; echo "== the on-box watchdog ($BOX_DIR/watchdog.sh: ONE for the whole VM; ref and seconds filled at arm time) =="; cat "$TMPD/check_watchdog.sh"
  echo
  echo "== what --rent does on 2gpu, in order =="
  echo "   1. refuse: lease under 60 min, no MOJOLEARN_HOTAISLE_GPU_ONLY=1, a CPU opponent body or env, one OUT for both,"
  echo "      a dirty tree (when source ships), a bad key file, a broken script, an oversized bundle"
  echo "   2. GET teams (operator role, VM limit), take ONE slot, balance >= $MIN_BALANCE_CENTS cents"
  echo "   3. wait for Quantity > 0 on the 2x MI300X offering; its minimum reservation <= the lease; balance >= its price"
  echo "   4. ARM THE MAC DEAD-MAN, then under the create lock: snapshot, POST   [THE BILL STARTS HERE: 60 minutes minimum]"
  echo "   5. PATCH description mojolearn:$LANE:<utc>, verify it; wait for running; ssh settle as hotaisle; sudo -n"
  echo "   6. key to $BOX_RC on stdin; arm ONE watchdog for the VM; verify pid (two sessions), ref, GET 200 + description"
  echo "   7. key-in-ps both ends; device probe; runtime docker or podman (native deletes); GPU arch from rocminfo"
  echo "   8. GPU map (KFD, DRM, rocm-smi) -> pin_mode render or dri, into both leg.txt; not 2 KFD GPUs deletes"
  echo "   9. image pull in the background; $( [ "$TEST_2GPU" = 1 ] && echo 'no source (test)' || echo 'ONE bundle upload, sha256 check, unpacked to /root/leg-a and /root/leg-b')"
  echo "  10. pull done (a failed pull deletes); pin calibration: HIP must see exactly one device at the GPU's address; both"
  echo "      GPUs pinned to different addresses, or delete unused; pin lines into both leg.txt and both remote bodies"
  echo "  11. both bodies start, one pinned container each, own timeout(1); poll 30 s; a finished body is fetched to its OUT at once"
  echo "  12. both finished or the poll deadline: DELETE ?force=true; verify gone; cancel the dead-man; release the slot;"
  echo "      teardown.txt and deadman.txt copied to the GPU 1 OUT$( [ "$TEST_2GPU" = 1 ] && echo '; test_2gpu=PASS or FAIL in both leg.txt')"
  echo "   dry-run artifacts kept in $OUT and $OUT_B"
  [ "$RED" = 1 ] && { echo "DRY RUN: RED. This script is broken (a FAIL above). Nothing rented."; exit 1; }
  [ "$BLOCK" = 1 ] && { echo "DRY RUN: plumbing GREEN, and a real leg is BLOCKED (see BLOCK above). Nothing rented."; exit 3; }
  echo "DRY RUN: GREEN. Nothing rented."
  exit 0
fi

if [ "$MODE" = dry ]; then
  echo
  echo "== the remote body (/root/gemm_leg.sh) =="; cat "$OUT/remote_body.sh"
  echo; echo "== the on-box watchdog ($BOX_DIR/watchdog.sh; ref and seconds filled at arm time) =="; cat "$TMPD/check_watchdog.sh"
  echo; echo "== the start wrapper =="; cat "$TMPD/check_remote_start.sh"
  echo
  echo "== what --rent does, in order =="
  echo "   1. refuse a dirty tree (when source ships), a bad key file, a broken script, an oversized bundle"
  echo "   2. GET teams (operator role, VM limit), take a slot (/tmp/mojolearn-hotaisle-slot.N), balance >= $MIN_BALANCE_CENTS cents"
  echo "   3. wait for Quantity > 0 on the $SPEC spec (up to $STOCK_WAIT_MINUTES min); print price and balance"
  echo "   4. ARM THE MAC DEAD-MAN, then under the create lock: snapshot, POST   [THE BILL STARTS HERE]"
  echo "   5. PATCH description mojolearn:$LANE:<utc>, verify it; wait for running; ssh settle as hotaisle; sudo -n"
  echo "   6. key to $BOX_RC on stdin; arm the watchdog; verify pid (two sessions), ref, GET 200 + description"
  echo "   7. key-in-ps both ends; device probe; runtime (docker/podman/native); GPU arch from rocminfo"
  echo "   8. image pull in the background; stream the bundle over ssh stdin, sha256 check, unpack"
  echo "   9. body under timeout(1) at the work bound; poll 30 s; fetch /root/gemm_leg_out -> <out>/remote/"
  echo "  10. DELETE ?force=true; verify GET 404 or absent from the listing; cancel the dead-man; release the slot"
  echo "   dry-run artifacts kept in $OUT"
  [ "$RED" = 1 ] && { echo "DRY RUN: RED. This script is broken (a FAIL above). Nothing rented."; exit 1; }
  [ "$BLOCK" = 1 ] && { echo "DRY RUN: plumbing GREEN, and a real leg is BLOCKED (see BLOCK above). Nothing rented."; exit 3; }
  echo "DRY RUN: GREEN. Nothing rented."
  exit 0
fi

# ------------------------------------------------------ from here it can bill
[ "$RED" = 1 ] && die "REFUSING to rent: a local check FAILED above." 1
[ "$BLOCK" = 1 ] && die "REFUSING to rent: a local check BLOCKED above." 3
load_key

echo
echo "== pre-flight =="
c=$(api GET "teams/" "$TMPD/teams.json")
[ "$c" = 200 ] || die "REFUSING to rent: GET teams/ -> HTTP $c. Nothing was created." 2
read -r _op TEAM_MAX_VMS < <(J teams "$TMPD/teams.json" "$TEAM")
[ "$_op" = yes ] || die "REFUSING to rent: this key lacks the operator role on $TEAM (create and DELETE need it). Nothing was created." 2
BAL_BEFORE=$(balance_cents)
log "team $TEAM balance at the start $(dollars "$BAL_BEFORE") ($BAL_BEFORE cents); VM limit $TEAM_MAX_VMS, slots $(slot_cap)"
echo "balance_before_cents=$BAL_BEFORE" >> "$OUT/leg.txt"
[ "$BAL_BEFORE" -ge "$MIN_BALANCE_CENTS" ] 2>/dev/null \
  || die "REFUSED: balance $(dollars "$BAL_BEFORE") is below the \$5.00 floor ($MIN_BALANCE_CENTS cents). Nothing was created." 3
c=$(api GET "user/ssh_keys/" "$TMPD/keys.json")
[ "$c" = 200 ] && [ "$(J sshkey "$TMPD/keys.json" "$SSH_KEY_FP")" = yes ] \
  || die "REFUSING to rent: $SSH_KEY_FP is not registered on the account (HTTP $c); ssh would fail. Nothing was created." 2

# ---- a. the slot ----
_t0=$(date +%s); _said=0
while :; do
  if try_take_slot "$(slot_cap)"; then
    # Capacity is the team's, not only this Mac's: a VM outside the slots counts.
    if [ "$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/cap.json")" = 200 ] \
       && [ "$(J count "$TMPD/cap.json")" -lt "$TEAM_MAX_VMS" ]; then
      break
    fi
    log "slot taken but the team already runs $(J count "$TMPD/cap.json") of $TEAM_MAX_VMS VMs; releasing and waiting"
    release_slot
  fi
  [ "$_said" = 0 ] && { log "all slots busy (or the team is at its VM limit); polling every 60 s"; list_slots; _said=1; }
  check_stale_slots
  [ $(( $(date +%s) - _t0 )) -lt $(( SLOT_WAIT_MINUTES * 60 )) ] \
    || die "REFUSED: no slot freed in $SLOT_WAIT_MINUTES minutes. Nothing was created." 3
  nap 60
done
echo "slot=$SLOT taken $(utc)" >> "$OUT/leg.txt"
log "slot $SLOT taken (lane $LANE)"
if [ "$SPEC" = 2gpu ]; then
  { echo "spec=2gpu"; echo "out_b=$REAL_OUT_B"; } >> "$SLOT/owner"
  _now=$(date +%s); B_SINCE=("$_now" "$_now")
  legs_write
  echo "slot=$SLOT taken $(utc) (the VM's one slot)" >> "$OUT_B/leg.txt"
fi

# ---- b. stock ----
_t0=$(date +%s)
while :; do
  c=$(api GET "teams/$TEAM/virtual_machines/available/" "$TMPD/avail.json")
  if [ "$SPEC" = 2gpu ]; then
    read -r _f _qty _price _minres _cores < <(J pick2 "$TMPD/avail.json" "$TMPD/create_request.json")
  else
    read -r _f _qty _price _minres < <(J pick "$TMPD/avail.json" "$SPEC_CORES" "$TMPD/create_request.json")
  fi
  if [ "$c" = 200 ] && [ "$_f" = found ] && [ "$_qty" -gt 0 ]; then break; fi
  [ $(( $(date +%s) - _t0 )) -lt $(( STOCK_WAIT_MINUTES * 60 )) ] \
    || die "REFUSED: the $SPEC $SPEC_DESC spec showed no stock for $STOCK_WAIT_MINUTES minutes (last HTTP $c, $_f, quantity ${_qty:-0}). Nothing was created." 3
  log "no stock on $SPEC (HTTP $c, $_f, quantity ${_qty:-0}); retrying in 60 s"
  nap 60
done
if [ "$SPEC" = 2gpu ]; then
  [ "$_minres" -le "$MINUTES" ] 2>/dev/null \
    || die "REFUSED: the 2x MI300X offering has MinimumReservationMinutes $_minres, above the $MINUTES-minute lease. Nothing was created." 3
  _floor=$(( _price * _minres / 60 + 1 ))
  [ "$_floor" -ge "$MIN_BALANCE_CENTS" ] || _floor=$MIN_BALANCE_CENTS
  [ "$BAL_BEFORE" -ge "$_floor" ] 2>/dev/null \
    || die "REFUSED: balance $(dollars "$BAL_BEFORE") is below $(dollars "$_floor"), the 2x MI300X minimum reservation ($_minres min at $_price cents/h). Nothing was created." 3
  echo "cpu_cores=$_cores balance_floor_cents=$_floor" >> "$OUT/leg.txt"
else
  [ "$_minres" -le 10 ] || die "REFUSED: the $SPEC spec has MinimumReservationMinutes $_minres. Nothing was created." 3
fi
cp "$TMPD/create_request.json" "$OUT/create_request.json"
cp "$TMPD/avail.json" "$OUT/offering.json"
log "spec $SPEC: quantity $_qty, $_price cents/hour, minimum reservation $_minres min; $MINUTES min costs at most $(dollars $(( _price * MINUTES / 60 + 1 )))"
echo "price_cents_per_hour=$_price min_reservation_minutes=$_minres" >> "$OUT/leg.txt"

# ---- c. the Mac dead-man, BEFORE the create ----
LEG_START=$(date +%s)
DEADLINE_EPOCH=$((LEG_START + MINUTES * 60))
S_DEADLINE=$DEADLINE_EPOCH
[ "$TEST_WATCHDOG" = 1 ] && S_DEADLINE=$((DEADLINE_EPOCH + 600))
DEADMAN_DIR="${TMPDIR:-/tmp}/mojolearn-hotaisle-deadman-$$"
DEADMAN_DIR="${DEADMAN_DIR//\/\//\/}"
DESC="mojolearn:$LANE:$(date -u +%Y%m%dT%H%M%SZ)"
( umask 077; mkdir -p "$DEADMAN_DIR" )
cp "$CURLRC" "$DEADMAN_DIR/curlrc"; chmod 600 "$DEADMAN_DIR/curlrc"
cp "$TMPD/j.py" "$DEADMAN_DIR/j.py"
subst "$TMPD/deadman.sh.template" "$DEADMAN_DIR/deadman.sh" || die "THE MAC DEAD-MAN DID NOT BUILD. Nothing was created." 1
nohup sh -c 'trap "" HUP INT TERM; exec sh "$0"' "$DEADMAN_DIR/deadman.sh" > /dev/null 2>&1 < /dev/null &
DEADMAN_PID=$!
disown "$DEADMAN_PID" 2>/dev/null  # no "Killed: 9" job line when a clean teardown cancels it
sleep 1
kill -0 "$DEADMAN_PID" 2>/dev/null || { DEADMAN_PID=""; die "THE MAC DEAD-MAN DID NOT START. Nothing was created." 1; }
{
  echo "mac_deadman_pid=$DEADMAN_PID"
  echo "mac_deadman_dir=$DEADMAN_DIR"
  echo "mac_deadman_fires_at=$(utc_of "$S_DEADLINE")$( [ "$TEST_WATCHDOG" = 1 ] && echo ' (TEST-WATCHDOG: late backstop, deadline + 600 s)')"
  echo "mac_deadman_keyed_by=vm_ref.txt (deployment_id from the create response) and description $DESC"
} > "$OUT/deadman.txt"
log "Mac dead-man ARMED before the create: pid $DEADMAN_PID, fires at $(utc_of "$S_DEADLINE")"
if ps -axo command= 2>/dev/null | grep -q -F -f "$TOKPAT"; then
  KEY_RED=1; log "!! THE KEY IS VISIBLE IN THIS MAC'S PROCESS LIST"; echo "local_key_in_ps=VISIBLE" >> "$OUT/leg.txt"
else
  echo "local_key_in_ps=not_visible" >> "$OUT/leg.txt"
fi

# ---- the create, serialized ----
_t0=$(date +%s)
until mkdir "$CREATE_LOCK" 2>/dev/null; do
  _age=$(slot_age "$CREATE_LOCK") || _age=0
  if [ "$_age" -gt 900 ]; then
    log "breaking a stale create lock (${_age}s): $(tr '\n' ' ' < "$CREATE_LOCK/owner" 2>/dev/null)"
    rm -rf "$CREATE_LOCK"; continue
  fi
  [ $(( $(date +%s) - _t0 )) -lt 900 ] || die "REFUSED: the create lock stayed held for 15 minutes. Nothing was created." 3
  nap 5
done
CREATE_LOCK_HELD=1
{ echo "nonce=$NONCE"; echo "pid=$$"; echo "lane=$LANE"; echo "utc=$(utc)"; } > "$CREATE_LOCK/owner"
c=$(api GET "teams/$TEAM/virtual_machines/" "$TMPD/pre.json")
[ "$c" = 200 ] || die "REFUSING to rent: the pre-create listing returned HTTP $c. Nothing was created." 2
J ids "$TMPD/pre.json" | awk '{print $1}' > "$TMPD/pre_ids.txt"

echo
echo "== the VM =="
log "creating $SPEC_DESC $SPEC"
CREATE_ATTEMPTED=1
c=$(api POST "teams/$TEAM/virtual_machines/" "$OUT/create_response.json" 300 "$TMPD/create_request.json")
redact "$OUT/create_response.json"
echo "create_http=$c create_utc=$(utc)" >> "$OUT/leg.txt"
case "$c" in
  401|402|403|404|428)
    log "create refused: HTTP $c: $(head -c 300 "$OUT/create_response.json")"
    nap 10
    adopt_new_vm || die "create REFUSED by the API (HTTP $c) and no new VM appears. Nothing is billing." 4 ;;
esac
if [ -z "$VMREF" ]; then
  IFS=$'\t' read -r VMNAME DEPLOY_ID _ip _port _d < <(J vm "$OUT/create_response.json")
  VMREF="$DEPLOY_ID"
fi
if [ -z "$VMREF" ]; then
  log "create returned no deployment_id (HTTP $c): $(head -c 300 "$OUT/create_response.json")"
  for _i in $(seq 1 18); do adopt_new_vm && break; nap 10; done
  [ -n "$VMREF" ] || die "create FAILED (no deployment_id and no new VM after 3 minutes). The teardown looks once more." 4
  log "ADOPTED $VMREF ($VMNAME) by listing diff after an unreadable create"
  echo "adopted_by_listing_diff=1" >> "$OUT/leg.txt"
fi
case "$VMREF" in *[!A-Za-z0-9_.-]*) die "the VM ref '$VMREF' has unexpected characters" 4 ;; esac
printf '%s\n' "$VMREF" > "$DEADMAN_DIR/vm_ref.txt"
printf 'vm_ref=%s\nvm_name=%s\n' "$VMREF" "$VMNAME" >> "$SLOT/owner"
printf 'vm_ref=%s\nvm_name=%s\n' "$VMREF" "$VMNAME" >> "$OUT/leg.txt"
log "VM $VMNAME deployment_id $VMREF (Mac dead-man now keyed by it)"

# {vm} is "name or deployment ID": prove the deployment_id form answers, else use the name.
c=$(api GET "teams/$TEAM/virtual_machines/$VMREF/" "$TMPD/vm.json")
if [ "$c" != 200 ] && [ -n "$VMNAME" ]; then
  c2=$(api GET "teams/$TEAM/virtual_machines/$VMNAME/" "$TMPD/vm.json")
  if [ "$c2" = 200 ]; then
    log "GET by deployment_id -> $c, by name -> 200: using the name as the ref"
    VMREF="$VMNAME"; printf '%s\n' "$VMREF" > "$DEADMAN_DIR/vm_ref.txt"; echo "vm_ref_is_name=1" >> "$OUT/leg.txt"
  fi
fi

# ---- g. the description ----
printf '{"description":"%s"}\n' "$DESC" > "$TMPD/patch.json"
_tagged=0
for _i in 1 2 3 4 5; do
  c=$(api PATCH "teams/$TEAM/virtual_machines/$VMREF/" "$TMPD/patch.out" 60 "$TMPD/patch.json")
  if [ "$(api GET "teams/$TEAM/virtual_machines/$VMREF/" "$TMPD/vm.json")" = 200 ] && [ "$(J desc "$TMPD/vm.json")" = "$DESC" ]; then
    _tagged=1; break
  fi
  log "PATCH description -> HTTP $c, not yet visible; retrying"
  nap 5
done
release_create_lock
[ "$_tagged" = 1 ] || die "the description PATCH never landed; deleting the VM unused" 5
echo "description=$DESC" >> "$OUT/leg.txt"
log "description $DESC"

_t0=$(date +%s); _state=unknown
while [ $(( $(date +%s) - _t0 )) -lt 600 ]; do
  [ "$(api GET "teams/$TEAM/virtual_machines/$VMREF/state/" "$TMPD/state.json")" = 200 ] && _state=$(J state "$TMPD/state.json")
  echo "$(utc) state=$_state" >> "$OUT/vm_states.txt"
  [ "$_state" = running ] && break
  nap 10
done
[ "$_state" = running ] || die "the VM never reached running (last state $_state)" 5
api GET "teams/$TEAM/virtual_machines/$VMREF/" "$TMPD/vm.json" > /dev/null
IFS=$'\t' read -r _n _id SSH_IP SSH_PORT _d < <(J vm "$TMPD/vm.json")
cp "$TMPD/vm.json" "$OUT/vm_details.json"; redact "$OUT/vm_details.json"
[ -n "$SSH_IP" ] || die "the VM has no ssh address" 5
case "$SSH_PORT" in ''|*[!0-9]*) SSH_PORT=22 ;; esac
log "running after $(( $(date +%s) - LEG_START ))s; ssh hotaisle@$SSH_IP -p $SSH_PORT"
echo "ssh=hotaisle@$SSH_IP:$SSH_PORT running_after_seconds=$(( $(date +%s) - LEG_START ))" >> "$OUT/leg.txt"

SSH_OPTS=(-p "$SSH_PORT" -i "$SSH_KEY" -o IdentitiesOnly=yes -o BatchMode=yes -o StrictHostKeyChecking=accept-new
          -o "UserKnownHostsFile=$TMPD/known_hosts" -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4)
SSH=(ssh "${SSH_OPTS[@]}" "hotaisle@$SSH_IP")
SSHN=(ssh -n "${SSH_OPTS[@]}" "hotaisle@$SSH_IP")
_ok=0
for _i in $(seq 1 90); do
  if "${SSHN[@]}" true 2>/dev/null; then _ok=$((_ok + 1)); [ "$_ok" -ge 3 ] && break; else _ok=0; fi
  nap 5
done
[ "$_ok" -ge 3 ] || die "ssh never settled on $SSH_IP:$SSH_PORT" 5
log "ssh settled after $(( $(date +%s) - LEG_START ))s"
"${SSHN[@]}" 'sudo -n true && echo SUDO_OK' > "$TMPD/sudo.out" 2>&1
grep -q SUDO_OK "$TMPD/sudo.out" || die "passwordless sudo is not available for hotaisle ($(head -c 200 "$TMPD/sudo.out")); deleting unused" 6

# Remote helpers. rexec: the script is saved as the hotaisle user's mktemp file
# and run as root with stdin from /dev/null. rput: a file to a root path.
rexec() { "${SSH[@]}" 'f=$(mktemp) && cat > "$f" && sudo -n -H sh "$f" < /dev/null; rc=$?; rm -f "$f"; exit $rc' < "$1"; }
rput() { "${SSH[@]}" "sudo -n sh -c 'umask 077; mkdir -p \$(dirname $2); cat > $2 && chmod $3 $2'" < "$1"; }

# ---- d. the on-box watchdog, before any work ----
S_SECS=$((DEADLINE_EPOCH - $(date +%s)))
[ "$S_SECS" -ge 60 ] || S_SECS=60
subst "$TMPD/watchdog.sh.template" "$OUT/watchdog.sh" || die "the watchdog did not substitute" 1
subst "$TMPD/watchdog_arm.sh.template" "$TMPD/watchdog_arm.sh" || die "the watchdog arm did not substitute" 1
rput "$CURLRC" "$BOX_RC" 600 || die "could not deliver the key for the watchdog; deleting unused" 6
rput "$OUT/watchdog.sh" "$BOX_DIR/watchdog.sh" 700 || die "could not deliver the watchdog; deleting unused" 6
rexec "$TMPD/watchdog_arm.sh" > "$TMPD/arm.out" 2>&1
sed 's/^/    /' "$TMPD/arm.out"
_wpid=$(sed -n 's/^WATCHDOG_ALIVE pid=//p' "$TMPD/arm.out" | tr -d '\r')
nap 3
"${SSHN[@]}" "sudo -n sh -c 'kill -0 $_wpid 2>/dev/null && echo WATCHDOG_STILL_ALIVE_SECOND_SESSION'" > "$TMPD/arm2.out" 2>&1
sed 's/^/    /' "$TMPD/arm2.out"
{
  echo "watchdog_seconds=$S_SECS"
  echo "watchdog_fires_at=$(utc_of $(( $(date +%s) + S_SECS )))"
  grep -E '^(WATCHDOG_|REF_BAKED_IN=|TOKEN_GET_HTTP=|DESC_)' "$TMPD/arm.out" "$TMPD/arm2.out" | sed 's/^[^:]*://; s/^/watchdog_/'
} >> "$OUT/deadman.txt"
if ! grep -q '^WATCHDOG_ALIVE' "$TMPD/arm.out" || ! grep -q '^REF_BAKED_IN=[1-9]' "$TMPD/arm.out" \
   || ! grep -q '^TOKEN_GET_HTTP=200' "$TMPD/arm.out" || ! grep -q '^DESC_MATCH' "$TMPD/arm.out" \
   || ! grep -q WATCHDOG_STILL_ALIVE_SECOND_SESSION "$TMPD/arm2.out"; then
  die "THE ON-BOX WATCHDOG COULD NOT BE VERIFIED (pid, second session, ref, GET 200 or description). Deleting the VM unused." 6
fi
WATCHDOG_OK=1
log "on-box watchdog ARMED and verified (pid $_wpid alive in two sessions, ref $VMREF, GET 200, description matches, ${S_SECS}s)"

rput "$TOKPAT" "$BOX_DIR/key.pattern" 600
printf '%s\n' "ps -eo args= > $BOX_DIR/ps.txt 2>/dev/null || ps ax > $BOX_DIR/ps.txt" \
  "if grep -q -F -f $BOX_DIR/key.pattern $BOX_DIR/ps.txt; then echo KEY_VISIBLE_IN_PS; else echo KEY_NOT_IN_PS; fi" \
  "rm -f $BOX_DIR/ps.txt $BOX_DIR/key.pattern" > "$TMPD/ps.sh"
rexec "$TMPD/ps.sh" > "$TMPD/ps.out" 2>&1
if grep -q KEY_NOT_IN_PS "$TMPD/ps.out"; then
  echo "box_key_in_ps=not_visible" >> "$OUT/leg.txt"
else
  KEY_RED=1; echo "box_key_in_ps=$(tr '\n' ' ' < "$TMPD/ps.out")" >> "$OUT/leg.txt"
  log "!! THE KEY IS VISIBLE (or unverifiable) IN THE VM'S PROCESS LIST"
fi

# ---- device probe, runtime, arch ----
rexec "$TMPD/device_probe.sh" > "$OUT/device_probe.txt" 2>&1
sed 's/^/    [box] /' "$OUT/device_probe.txt"
RUNTIME=native
case "$RUNTIME_WANT" in
  auto) if grep -q '^DOCKER_OK' "$OUT/device_probe.txt"; then RUNTIME=docker
        elif grep -q '^PODMAN_OK' "$OUT/device_probe.txt"; then RUNTIME=podman; fi ;;
  docker|podman) grep -q "^$(echo "$RUNTIME_WANT" | tr a-z A-Z)_OK" "$OUT/device_probe.txt" \
                   || die "MOJOLEARN_HOTAISLE_RUNTIME=$RUNTIME_WANT but $RUNTIME_WANT does not answer on the VM" 6
                 RUNTIME=$RUNTIME_WANT ;;
esac
[ "$BARE" = 1 ] && RUNTIME=native
grep -q '^KFD_PRESENT' "$OUT/device_probe.txt" || die "/dev/kfd is absent on the VM: no AMD compute device. Deleting." 6
echo "runtime=$RUNTIME" >> "$OUT/leg.txt"
log "runtime $RUNTIME"
if [ "$SPEC" = 2gpu ]; then
  echo "runtime=$RUNTIME" >> "$OUT_B/leg.txt"
  [ "$RUNTIME" != native ] || die "the 2gpu spec needs docker or podman on the VM (one pinned container per GPU) and the runtime is native. Deleting." 6
fi
if [ "$TEST_WATCHDOG" = 0 ] && [ "$RUNTIME" != native ]; then
  subst "$TMPD/pull_start.sh.template" "$TMPD/pull_start.sh" && rexec "$TMPD/pull_start.sh" > "$TMPD/pull.out" 2>&1
  log "image pull started in the background: $(tr '\n' ' ' < "$TMPD/pull.out")"
fi
box_archs() { sed -n 's/^GFX=//p' "$1" | sort -u; }
BOX_ARCHS="$(box_archs "$OUT/device_probe.txt")"
echo "box_gfx=$(echo $BOX_ARCHS)" >> "$OUT/leg.txt"

# ---- --test-watchdog: nothing ships; the watchdog must end the VM ----
if [ "$TEST_WATCHDOG" = 1 ]; then
  echo
  echo "== TEST-WATCHDOG: the Mac sends no delete; waiting for the on-box watchdog =="
  while [ "$(date +%s)" -lt "$DEADLINE_EPOCH" ]; do nap 15; done
  log "deadline reached; polling for the VM to be gone (8 minutes grace)"
  _t0=$(date +%s)
  while :; do
    if _line=$(gone_check "$VMREF" "$VMNAME"); then
      echo "watchdog_test=PASS verified_gone ref=$VMREF $_line utc=$(utc) after_deadline=$(( $(date +%s) - DEADLINE_EPOCH ))s" | tee -a "$OUT/teardown.txt" "$OUT/leg.txt"
      DESTROY_CONFIRMED=1; WDT_DONE=1
      exit 0
    fi
    echo "$(utc) $_line" >> "$OUT/vm_states.txt"
    if [ $(( $(date +%s) - _t0 )) -ge 480 ]; then
      echo "watchdog_test=FAIL still present $_line utc=$(utc); the trap deletes it now" | tee -a "$OUT/teardown.txt" "$OUT/leg.txt"
      WDT_DONE=1
      exit 1
    fi
    nap 15
  done
fi

# ---- the arch ----
if [ -z "$GPU_ARCHS" ] && [ -z "$BOX_ARCHS" ] && [ "$RUNTIME" != native ]; then
  log "no host rocminfo; waiting for the image to read the arch from inside it"
  while [ "$(date +%s)" -lt $((DEADLINE_EPOCH - FETCH_RESERVE - 300)) ]; do
    "${SSHN[@]}" 'sudo -n test -f /root/mojolearn-pull.done && echo PULL_DONE' 2>/dev/null | grep -q PULL_DONE && break
    nap 15
  done
  printf '%s\n' "$RUNTIME run --rm --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined $IMAGE sh -c 'rocminfo 2>/dev/null | grep -Eo \"gfx[0-9a-f]+\" | sort -u | sed \"s/^/GFX=/\"'" > "$TMPD/arch.sh"
  rexec "$TMPD/arch.sh" > "$OUT/arch_from_image.txt" 2>&1
  BOX_ARCHS="$(box_archs "$OUT/arch_from_image.txt")"
fi
_n_archs=$(printf '%s\n' "$BOX_ARCHS" | grep -c .)
if [ -z "$GPU_ARCHS" ]; then
  [ "$_n_archs" = 1 ] || die "MOJOLEARN_GPU_ARCHS is unset and rocminfo on the VM gave $_n_archs gfx names ('$(echo $BOX_ARCHS)'); exactly one is required. Deleting." 6
  S_ARCHS="$BOX_ARCHS"
  log "MOJOLEARN_GPU_ARCHS=$S_ARCHS read from rocminfo"
else
  S_ARCHS="$GPU_ARCHS"
  if [ "$_n_archs" -ge 1 ] && ! printf '%s\n' "$BOX_ARCHS" | grep -qx "$GPU_ARCHS"; then
    die "MOJOLEARN_GPU_ARCHS=$GPU_ARCHS but rocminfo on the VM reports '$(echo $BOX_ARCHS)'. One mojo build is one GPU arch. Deleting." 6
  fi
fi
echo "gpu_archs=$S_ARCHS" >> "$OUT/leg.txt"

# ---- the source ----
with_deadline() {  # <seconds> <stdin file> <cmd...>: the command's status, or 124 at the deadline
  # stdin is an explicit redirect on the async command: without one, bash
  # gives a background job /dev/null, whatever the caller redirected.
  local secs=$1 infile=$2 pid waited=0
  shift 2
  "$@" < "$infile" &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if [ "$waited" -ge "$secs" ]; then pkill -P "$pid" 2>/dev/null; kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; return 124; fi
    sleep 1; waited=$((waited + 1))
  done
  wait "$pid"
}

# ================================================================ the 2gpu VM
# The 2gpu flow from the GPU map to its exit (header: THE 2GPU SPEC, steps 3 to
# 8). The EXIT trap is the same one delete for the whole VM. The single-GPU
# flow below this block never runs on 2gpu and is unchanged.
if [ "$SPEC" = 2gpu ]; then
  echo
  echo "== the GPU map (2gpu) =="
  rexec "$TMPD/gpu_map.sh" > "$OUT/gpu_map.txt" 2>&1
  sed 's/^/    [box] /' "$OUT/gpu_map.txt"
  map_gpus "$OUT/gpu_map.txt"
  _kn=$(grep -c '^KFD [0-9]' "$OUT/gpu_map.txt")
  if [ "$_kn" -gt 0 ] && [ "$_kn" != 2 ]; then
    die "the 2x MI300X VM shows $_kn KFD GPU nodes, not 2 ($GPU_COUNT_SEEN). Deleting." 6
  fi
  for _o in "$OUT" "$OUT_B"; do
    {
      echo "gpu_map_source=$GPU_MAP_SOURCE seen=$GPU_COUNT_SEEN"
      echo "pin_mode=$PIN_MODE"
      echo "gpu0_host_render=${G_RENDER[0]:-unknown} gpu0_host_bdf=${G_BDF[0]:-unknown}"
      echo "gpu1_host_render=${G_RENDER[1]:-unknown} gpu1_host_bdf=${G_BDF[1]:-unknown}"
      if [ "$PIN_MODE" != render ]; then
        echo "gpu_pinning_note=the renderD mapping could not be read on the box: /dev/dri goes whole to both containers and pinning rests on ROCR_VISIBLE_DEVICES and HIP_VISIBLE_DEVICES"
      fi
    } >> "$_o/leg.txt"
  done
  log "GPU map from $GPU_MAP_SOURCE ($GPU_COUNT_SEEN): pin_mode $PIN_MODE; GPU 0 ${G_RENDER[0]:-?} ${G_BDF[0]:-?}; GPU 1 ${G_RENDER[1]:-?} ${G_BDF[1]:-?}"

  echo
  echo "== the source (2gpu: one upload, two copies) =="
  if [ "$SHIPS_SOURCE" = 1 ]; then
    _up0=$(date +%s); _up_ok=0
    for _try in 1 2 3; do
      _left=$((DEADLINE_EPOCH - FETCH_RESERVE - $(date +%s)))
      [ "$_left" -gt 120 ] || break
      if with_deadline "$_left" "$TMPD/src.tgz" "${SSH[@]}" "sudo -n sh -c 'cat > /root/extra_src.tgz'"; then _up_ok=1; break; fi
      log "upload attempt $_try failed; retrying in 15 s"
      nap 15
    done
    [ "$_up_ok" = 1 ] || die "the bundle upload failed" 7
    _up_s=$(( $(date +%s) - _up0 ))
    log "uploaded $BUNDLE_BYTES bytes in ${_up_s}s over ssh stdin"
    echo "upload_seconds=$_up_s" >> "$OUT/leg.txt"
    subst "$TMPD/remote_unpack2.sh.template" "$TMPD/remote_unpack2.sh" || die "the two-copy unpack script did not substitute" 1
    rexec "$TMPD/remote_unpack2.sh" > "$TMPD/unpack.out" 2>&1
    sed 's/^/    /' "$TMPD/unpack.out"
    { grep -q '^ARCHIVE-SHA-OK' "$TMPD/unpack.out" && [ "$(grep -c '^UNPACKED_[ab] ' "$TMPD/unpack.out")" = 2 ]; } \
      || die "the VM refused or failed to unpack the bundle into both copies" 7
  else
    printf 'rm -rf /root/leg-a /root/leg-b /root/mojolearn-pin\nmkdir -p /root/leg-a/mojolearn /root/leg-b/mojolearn\n' > "$TMPD/prep2.sh"
    rexec "$TMPD/prep2.sh" > /dev/null 2>&1
  fi

  log "waiting for the image pull (one pull for both bodies)"
  while [ "$(date +%s)" -lt $((DEADLINE_EPOCH - FETCH_RESERVE - 180)) ]; do
    "${SSHN[@]}" 'sudo -n cat /root/mojolearn-pull.done 2>/dev/null' > "$TMPD/pull.done" 2>/dev/null
    grep -q pull_exit= "$TMPD/pull.done" && break
    nap 15
  done
  _pull="$(tr -d '\r' < "$TMPD/pull.done" 2>/dev/null)"
  echo "image_pull=${_pull:-NOT_DONE}" >> "$OUT/leg.txt"
  log "image pull: ${_pull:-NOT DONE}"
  case "$_pull" in
    *pull_exit=0*) ;;
    *) die "the image pull did not finish cleanly; a 2gpu VM has no native fallback (one pinned container per GPU). Deleting." 7 ;;
  esac

  echo
  echo "== pin calibration (2gpu) =="
  rput "$TMPD/pin_probe.sh" /root/mojolearn-pin/probe.sh 644 || die "could not ship the pin probe" 7
  gen_calibrate "$OUT/pin_calibrate.sh"
  cp "$TMPD/pin_cands.txt" "$OUT/pin_candidates.txt"
  rexec "$OUT/pin_calibrate.sh" > "$OUT/pin_calibration.txt" 2>&1
  sed 's/^/    [box] /' "$OUT/pin_calibration.txt"
  : > "$OUT/pin_decisions.txt"
  if ! calibrate_pins "$OUT/pin_calibration.txt"; then
    sed 's/^/    /' "$OUT/pin_decisions.txt"
    die "PINNING NOT VERIFIED: each GPU needs a candidate container where HIP sees exactly one device at that GPU's address, and the two addresses must differ (pin_decisions.txt). Deleting the VM unused." 6
  fi
  sed 's/^/    /' "$OUT/pin_decisions.txt"
  cp "$OUT/pin_decisions.txt" "$OUT_B/pin_decisions.txt"
  for _g in 0 1; do
    pin_kv "$_g" >> "${OUTS[$_g]}/leg.txt"
    log "GPU $_g pinned: /dev/kfd ${PIN_DEVICES[$_g]#--device } ROCR_VISIBLE_DEVICES=${PIN_ROCR[$_g]} HIP_VISIBLE_DEVICES=${PIN_HIP[$_g]} -> ${PIN_BUS[$_g]} (via ${PIN_VIA[$_g]}, candidate ${PIN_CAND[$_g]})"
  done

  for _g in 0 1; do
    _x=${LETTERS[$_g]}; _o=${OUTS[$_g]}
    pin_kv "$_g" body > "$TMPD/pin_body_$_g.txt"
    if [ "$TEST_2GPU" = 1 ]; then
      subst2 "$TMPD/test_body.sh.template" "$TMPD/body_pre_$_g.sh" "$_g" || die "the GPU $_g test body did not substitute" 1
    else
      subst "$TMPD/remote_body.sh.template" "$TMPD/body_pre_$_g.sh" || die "the GPU $_g remote body did not substitute" 1
    fi
    insert_pin_lines "$TMPD/body_pre_$_g.sh" "$TMPD/pin_body_$_g.txt" "$_o/remote_body.sh" || die "the GPU $_g pin lines did not go into its body" 1
    sh -n "$_o/remote_body.sh" || die "the GPU $_g body with its pin lines is not valid sh" 1
    if [ "$TEST_2GPU" = 1 ]; then
      rput "$TMPD/pin_probe.sh" "/root/leg-$_x/pin_probe.sh" 644 || die "could not ship the GPU $_g test probe" 7
    else
      rput "$_o/extra_body.sh" "/root/leg-$_x/gemm_leg_extra.sh" 644 || die "could not ship the GPU $_g extra body" 7
      rput "$_o/extra_env.sh" "/root/leg-$_x/gemm_leg_extra_env.sh" 644 || die "could not ship the GPU $_g extra body environment" 7
    fi
    rput "$_o/remote_body.sh" "/root/leg-$_x/gemm_leg.sh" 644 || die "could not ship the GPU $_g remote body" 7
    if [ "$TEST_2GPU" = 1 ]; then _what="the tiny test body"; elif [ "$_g" = 0 ]; then _what=$LEG_EXTRA; else _what=$LEG_EXTRA_B; fi
    log "shipped the GPU $_g body to /root/leg-$_x ($_what)"
  done

  echo
  echo "== the work (2gpu: two bodies) =="
  WORK_SECONDS=$((DEADLINE_EPOCH - FETCH_RESERVE - $(date +%s)))
  [ "$WORK_SECONDS" -ge 120 ] || die "only ${WORK_SECONDS}s of lease left for the work; starting neither body" 8
  subst2 "$TMPD/remote_start2.sh.template" "$OUT/remote_start2.sh" || die "the two-body start wrapper did not substitute" 1
  for _o in "$OUT" "$OUT_B"; do echo "work_seconds=$WORK_SECONDS" >> "$_o/leg.txt"; done
  rexec "$OUT/remote_start2.sh" > "$OUT/remote_start2.log" 2>&1
  RPIDS[0]=$(sed -n 's/^REMOTE_PID_A=//p' "$OUT/remote_start2.log" | tr -d '\r' | tail -1)
  RPIDS[1]=$(sed -n 's/^REMOTE_PID_B=//p' "$OUT/remote_start2.log" | tr -d '\r' | tail -1)
  for _g in 0 1; do
    case "${RPIDS[$_g]}" in ''|*[!0-9]*) die "THE GPU $_g BODY DID NOT START (no pid). Read $OUT/remote_start2.log." 8 ;; esac
  done
  _now=$(date +%s)
  B_STATE=(running running); B_SINCE=("$_now" "$_now")
  legs_write
  BODY_STATE=running
  log "GPU 0 pid ${RPIDS[0]}, GPU 1 pid ${RPIDS[1]} ($RUNTIME), each bound ${WORK_SECONDS}s; polling every 30 s"
  # shellcheck disable=SC2016  # the poll script's own variables, expanded on the VM
  {
    echo "for p in a:${RPIDS[0]} b:${RPIDS[1]}; do"
    echo '    x=${p%%:*}; pid=${p#*:}'
    echo '    if [ -f /root/leg-$x/gemm_leg.done ]; then echo "$x=LEG_DONE"; elif kill -0 "$pid" 2>/dev/null; then echo "$x=LEG_RUNNING"; else echo "$x=LEG_GONE"; fi'
    echo 'done'
  } > "$TMPD/poll2.sh"
  POLL_DEADLINE=$((DEADLINE_EPOCH - FETCH_RESERVE + 60))
  _unreach=0
  while :; do
    if [ "$(date +%s)" -ge "$POLL_DEADLINE" ]; then
      for _g in 0 1; do
        if [ "${B_DONE[$_g]}" = 0 ]; then
          B_STATE[_g]=partial_deadline; B_DONE[_g]=1; FETCH_RED=1
          log "GPU $_g body: OUTER POLL DEADLINE reached. Fetching what exists."
        fi
      done
      break
    fi
    _st=$(rexec "$TMPD/poll2.sh" 2>/dev/null) || _st=""
    case "$_st" in
      *a=LEG_*) _unreach=0 ;;
      *) _unreach=$((_unreach + 1)); [ "$_unreach" = 1 ] && log "poll: the VM did not answer (the bodies are detached; retrying)" ;;
    esac
    for _g in 0 1; do
      [ "${B_DONE[$_g]}" = 0 ] || continue
      _x=${LETTERS[$_g]}
      case "$_st" in
        *"$_x=LEG_DONE"*)
          _rp=${RPIDS[$_g]}
          "${SSHN[@]}" "sudo -n sh -c 'for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $_rp 2>/dev/null || break; sleep 1; done'" 2>/dev/null
          B_STATE[_g]="done"; B_DONE[_g]=1
          log "GPU $_g body finished (sentinel on the VM); fetching it now"
          fetch_body2 "$_g" ;;
        *"$_x=LEG_GONE"*)
          B_STATE[_g]=partial_died; B_DONE[_g]=1; FETCH_RED=1
          log "THE GPU $_g BODY PROCESS IS GONE AND WROTE NO SENTINEL. Fetching a partial run; the other body keeps running."
          fetch_body2 "$_g" ;;
      esac
    done
    legs_write
    if [ "${B_DONE[0]}" = 1 ] && [ "${B_DONE[1]}" = 1 ]; then break; fi
    nap 30
  done
  BODY_STATE="gpu0:${B_STATE[0]} gpu1:${B_STATE[1]}"

  echo
  echo "== fetch (2gpu) =="
  for _g in 0 1; do fetch_body2 "$_g"; done
  for _o in "$OUT" "$OUT_B"; do echo "lease_used_seconds=$(( $(date +%s) - LEG_START ))" >> "$_o/leg.txt"; done
  if [ "$TEST_2GPU" = 1 ]; then test2_verdict; fi
  log "both legs done ($BODY_STATE); deleting the VM (EXIT trap)"
  [ "$FETCH_RED" = 1 ] && exit 1
  [ "$KEY_RED" = 1 ] && exit 1
  if [ "$TEST_2GPU" = 1 ] && [ "$TEST2_BODIES" != PASS ]; then exit 1; fi
  exit 0
fi

if [ "$SHIPS_SOURCE" = 1 ]; then
  echo
  echo "== the source =="
  _up0=$(date +%s); _up_ok=0
  for _try in 1 2 3; do
    _left=$((DEADLINE_EPOCH - FETCH_RESERVE - $(date +%s)))
    [ "$_left" -gt 120 ] || break
    if with_deadline "$_left" "$TMPD/src.tgz" "${SSH[@]}" "sudo -n sh -c 'cat > /root/extra_src.tgz'"; then _up_ok=1; break; fi
    log "upload attempt $_try failed; retrying in 15 s"
    nap 15
  done
  [ "$_up_ok" = 1 ] || die "the bundle upload failed" 7
  _up_s=$(( $(date +%s) - _up0 ))
  log "uploaded $BUNDLE_BYTES bytes in ${_up_s}s over ssh stdin"
  echo "upload_seconds=$_up_s" >> "$OUT/leg.txt"
  subst "$TMPD/remote_unpack.sh.template" "$TMPD/remote_unpack.sh" || die "the unpack script did not substitute" 1
  rexec "$TMPD/remote_unpack.sh" > "$TMPD/unpack.out" 2>&1
  sed 's/^/    /' "$TMPD/unpack.out"
  grep -q '^ARCHIVE-SHA-OK' "$TMPD/unpack.out" && grep -q '^UNPACKED ' "$TMPD/unpack.out" || die "the VM refused or failed to unpack the bundle" 7
else
  printf 'rm -rf /root/mojolearn /root/gemm_leg_out /root/gemm_leg.done\nmkdir -p /root/mojolearn\n' > "$TMPD/bare_unpack.sh"
  rexec "$TMPD/bare_unpack.sh" > /dev/null 2>&1
fi
subst "$TMPD/remote_body.sh.template" "$OUT/remote_body.sh" || die "the remote body did not substitute" 1
rput "$OUT/extra_body.sh" /root/gemm_leg_extra.sh 644 || die "could not ship the extra body" 7
rput "$OUT/extra_env.sh" /root/gemm_leg_extra_env.sh 644 || die "could not ship the extra body environment" 7
rput "$OUT/remote_body.sh" /root/gemm_leg.sh 644 || die "could not ship the remote body" 7
log "shipped the extra body ($LEG_EXTRA, sha256 ${EXTRA_SHA:0:16}) and the remote body"

if [ "$RUNTIME" != native ]; then
  log "waiting for the image pull"
  while [ "$(date +%s)" -lt $((DEADLINE_EPOCH - FETCH_RESERVE - 180)) ]; do
    "${SSHN[@]}" 'sudo -n cat /root/mojolearn-pull.done 2>/dev/null' > "$TMPD/pull.done" 2>/dev/null
    grep -q pull_exit= "$TMPD/pull.done" && break
    nap 15
  done
  _pull="$(tr -d '\r' < "$TMPD/pull.done" 2>/dev/null)"
  echo "image_pull=${_pull:-NOT_DONE}" >> "$OUT/leg.txt"
  log "image pull: ${_pull:-NOT DONE}"
  case "$_pull" in
    *pull_exit=0*) ;;
    *) log "!! the image pull did not finish cleanly; falling back to native"; RUNTIME=native; echo "runtime=native (image pull failed)" >> "$OUT/leg.txt"
       subst "$TMPD/remote_body.sh.template" "$OUT/remote_body.sh" && rput "$OUT/remote_body.sh" /root/gemm_leg.sh 644 ;;
  esac
fi

# ---- the work, detached and polled ----
echo
echo "== the work =="
WORK_SECONDS=$((DEADLINE_EPOCH - FETCH_RESERVE - $(date +%s)))
[ "$WORK_SECONDS" -ge 120 ] || die "only ${WORK_SECONDS}s of lease left for the work; not starting it" 8
subst "$TMPD/remote_start.sh.template" "$OUT/remote_start.sh" || die "the start wrapper did not substitute" 1
echo "work_seconds=$WORK_SECONDS" >> "$OUT/leg.txt"
rexec "$OUT/remote_start.sh" > "$OUT/remote_start.log" 2>&1
RPID=$(sed -n 's/^REMOTE_PID=//p' "$OUT/remote_start.log" | tr -d '\r' | tail -1)
case "$RPID" in ''|*[!0-9]*) die "THE PAYLOAD DID NOT START (no pid). Read $OUT/remote_start.log." 8 ;; esac
BODY_STATE=running
log "remote pid $RPID ($RUNTIME), bound ${WORK_SECONDS}s; polling every 30 s"
POLL_DEADLINE=$((DEADLINE_EPOCH - FETCH_RESERVE + 60))
_unreach=0
while :; do
  if [ "$(date +%s)" -ge "$POLL_DEADLINE" ]; then
    log "OUTER POLL DEADLINE reached. Fetching what exists."; BODY_STATE=partial_deadline; FETCH_RED=1; break
  fi
  _st=$("${SSHN[@]}" "sudo -n sh -c 'if [ -f /root/gemm_leg.done ]; then echo LEG_DONE; elif kill -0 $RPID 2>/dev/null; then echo LEG_RUNNING; else echo LEG_GONE; fi'" 2>/dev/null) || _st=""
  case "$_st" in
    *LEG_DONE*)
      "${SSHN[@]}" "sudo -n sh -c 'for i in 1 2 3 4 5 6 7 8 9 10; do kill -0 $RPID 2>/dev/null || break; sleep 1; done'" 2>/dev/null
      log "the body finished (sentinel on the VM)"; BODY_STATE="done"; break ;;
    *LEG_RUNNING*) _unreach=0 ;;
    *LEG_GONE*)
      log "THE BODY PROCESS IS GONE AND WROTE NO SENTINEL. Fetching a partial run."; BODY_STATE=partial_died; FETCH_RED=1; break ;;
    *)
      _unreach=$((_unreach + 1))
      [ "$_unreach" = 1 ] && log "poll: the VM did not answer (the body is detached; retrying)" ;;
  esac
  nap 30
done
echo "body=$BODY_STATE" >> "$OUT/leg.txt"

echo
echo "== fetch =="
_left=$((DEADLINE_EPOCH - $(date +%s) - 90))
FETCH_SECONDS=600
[ "$_left" -lt "$FETCH_SECONDS" ] && FETCH_SECONDS=$_left
[ "$FETCH_SECONDS" -ge 60 ] || FETCH_SECONDS=60
mkdir -p "$OUT/remote"
if with_deadline "$FETCH_SECONDS" /dev/null "${SSHN[@]}" "sudo -n sh -c 'cd /root/gemm_leg_out && tar czf - --exclude=./tools-venv .'" > "$TMPD/remote.tgz" \
   && tar xzf "$TMPD/remote.tgz" -C "$OUT/remote"; then
  log "fetched /root/gemm_leg_out -> $OUT/remote/ ($(wc -c < "$TMPD/remote.tgz" | tr -d ' ') bytes)"
else
  FETCH_RED=1; log "FETCH FAILED or hit its ${FETCH_SECONDS}s bound; deleting regardless"
fi
rm -rf "$OUT/remote/tools-venv"
with_deadline 60 /dev/null "${SSHN[@]}" 'sudo -n cat /root/gemm_leg_console.log /root/mojolearn-pull.done' > "$OUT/remote_console.log" 2>/dev/null || true
if [ "$SHIPS_SOURCE" = 1 ]; then
  _local_sha=$(cat "$OUT/source_sha256_local.txt" 2>/dev/null)
  _remote_sha=$(cat "$OUT/remote/source_sha256.txt" 2>/dev/null)
  if [ -n "$_local_sha" ] && [ "$_local_sha" = "$_remote_sha" ]; then
    echo "source_sha256_match=yes" >> "$OUT/leg.txt"
  else
    echo "source_sha256_match=NO local=$_local_sha remote=$_remote_sha" >> "$OUT/leg.txt"; FETCH_RED=1
  fi
fi
for _k in pixi_install_exit pixi_install_seconds device_check_exit card_exit extra_exit body_exit; do
  _v=$(sed -n "s/^$_k=//p" "$OUT/remote/leg.txt" 2>/dev/null | tail -1)
  echo "remote_$_k=${_v:-<absent>}" >> "$OUT/leg.txt"
  log "  $_k=${_v:-<absent>}"
done
echo "finished=$(utc)" >> "$OUT/leg.txt"
echo "lease_used_seconds=$(( $(date +%s) - LEG_START ))" >> "$OUT/leg.txt"
log "leg done; deleting (EXIT trap)"
[ "$FETCH_RED" = 1 ] && exit 1
[ "$KEY_RED" = 1 ] && exit 1
exit 0
