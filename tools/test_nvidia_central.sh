#!/bin/bash
# tools/test_nvidia_central.sh -- tests the shared NVIDIA pod tooling with NO
# rental: two ubuntu:22.04 containers stand in for two RunPod pods (an `ssh`
# shim on PATH turns `ssh ... root@<container> <cmd>` into `docker exec`), with
# a fake nvidia-smi, a stand-in runpod_guard watchdog and a fake key file.
#   1. tools/test_gpu_queue_box.sh in a container (the queue, plain mode)
#   2. tools/nvidia_central.sh: up (balance refusal, stock fallback 4 -> 2,
#      idempotent at N), install (queue + lease, the fixed watchdog retired,
#      the Mac watcher), sync, sh, submit/queue/status/log/cancel, run, fetch,
#      down refusal while busy, the watcher's delete when a pod stops answering
#   3. tools/dev_pod.sh refusals: NVIDIA up without the central tool, and
#      extend/down/run/sync of a shared pod key
# Needs docker and network (apt git; the pod's git tree fetches the merge base
# from GitHub). Run it under one Mac slot:
#   bash tools/mac_slot.sh run -- bash tools/test_nvidia_central.sh
set -u   # no pipefail: the checks pipe into grep -q, which closes the pipe early
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
W=$(mktemp -d "${TMPDIR:-/tmp}/nvct.XXXXXX")
C1=nvct-a-$$; C2=nvct-b-$$
FAILS=0
ok() { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; FAILS=$((FAILS + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
waitfor() { local i=0; while [ $i -lt $1 ]; do eval "$2" && return 0; sleep 1; i=$((i + 1)); done; return 1; }
cleanup() {
    for p in $(cat "$W"/devpods/nvc*/deadman.pid 2>/dev/null); do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done
    docker rm -f $C1 $C2 > /dev/null 2>&1; rm -rf "$W"
}
[ "${KEEP:-0}" = 1 ] || trap cleanup EXIT; echo "work dir $W containers $C1 $C2"

echo "== 1. the box queue"
docker run --rm --cpus 1 -v "$ROOT/tools:/src:ro" ubuntu:22.04 bash /src/test_gpu_queue_box.sh /src/gpu_queue_box.sh > "$W/box.log" 2>&1
check "tools/test_gpu_queue_box.sh passes in a container" "grep -q 'RESULT: 0 failure' $W/box.log"
grep '^FAIL' "$W/box.log"

echo "== 2. fake pods"
for c in $C1 $C2; do
    docker run -d --name $c --cpus 1 ubuntu:22.04 sleep 7200 > /dev/null
    docker exec $c bash -c 'export DEBIAN_FRONTEND=noninteractive; apt-get update -qq > /dev/null && apt-get install -y -qq git ca-certificates procps > /dev/null' \
        || { echo "apt failed in $c"; exit 1; }
done
# two GPUs on $C1 (the 2x pod), four would be the same code; a stand-in lease watchdog and key file on each
for c in $C1 $C2; do
    docker exec $c bash -c 'cat > /usr/local/bin/nvidia-smi <<EOF
#!/bin/bash
case "\$1" in -L) echo "GPU 0: NVIDIA GeForce RTX 4090 (UUID: GPU-0)"; echo "GPU 1: NVIDIA GeForce RTX 4090 (UUID: GPU-1)" ;; *) echo "0, fake, 0 %, 0 MiB" ;; esac
EOF
chmod 755 /usr/local/bin/nvidia-smi; umask 077; echo "header = fake" > /tmp/mojolearn-lease.curlrc
nohup sleep 7000 > /dev/null 2>&1 < /dev/null & echo $! > /tmp/mojolearn-lease.pid'
done
mkdir -p "$W/bin" "$W/devpods" "$W/nvc"
cat > "$W/bin/ssh" <<'EOF'
#!/bin/bash
# test shim: ssh [-o X]... [-p N] [-tt] user@container <command> -> docker exec
while [ $# -gt 0 ]; do case "$1" in -o|-p) shift 2 ;; -t|-tt|-T) shift ;; *) break ;; esac; done
host=${1#*@}; shift
exec docker exec -i "$host" bash -c "$*"
EOF
chmod 755 "$W/bin/ssh"
# a fake dev_pod: `up` is out of stock at 4 GPUs and makes a 2-GPU "pod" (the next free container); `down` retires state
cat > "$W/fake_devpod.sh" <<EOF
#!/bin/bash
set -u
cmd=\$1; k=\$2; D=$W/devpods/\$k
echo "fake dev_pod \$* GPU_COUNT=\${MOJOLEARN_DEVPOD_GPU_COUNT:-} VIA=\${MOJOLEARN_DEVPOD_VIA_CENTRAL:-} DOWN=\${MOJOLEARN_NVC_DOWN:-}" >> $W/devpod_calls
case \$cmd in
up)
    [ "\${MOJOLEARN_DEVPOD_GPU_COUNT}" = 2 ] || { echo "dev_pod: RunPod out of stock for 5 min (1 attempts); nothing was created" >&2; exit 1; }
    c=\$(cat $W/next_container); mkdir -p \$D
    printf 'PROVIDER=runpod\nVENDOR=nvidia\nPOD_ID=fake-%s\nPOD_NAME=x\nSSH_TARGET=%q\nCOST_HR=0.69\nGPU=%q\nBOX_SUDO=0\nBOX_ENV=\nBOX_DIR=/root/mojolearn\nGPU_COUNT=2\n' \$k "-p 22 root@\$c" "NVIDIA GeForce RTX 4090" > \$D/state.env
    nohup sleep 7100 > /dev/null 2>&1 < /dev/null & echo \$! > \$D/deadman.pid
    echo "dev_pod: ready" ;;
down) p=\$(cat \$D/deadman.pid 2>/dev/null); [ -z "\$p" ] || kill \$p 2>/dev/null; mv \$D \$D.down-test; echo "dev_pod: \$k box down" ;;
esac
EOF
chmod 755 "$W/fake_devpod.sh"
export MOJOLEARN_ORCHESTRATOR=1 PATH="$W/bin:$PATH" MOJOLEARN_DEVPOD_STATE="$W/devpods" MOJOLEARN_NVC_HOME="$W/nvc" MOJOLEARN_NVC_CONF="$W/none.env" \
    MOJOLEARN_NVC_NO_API=1 MOJOLEARN_NVC_DEVPOD="$W/fake_devpod.sh" MOJOLEARN_NVC_WATCH_FAKE_DELETE=1 NVC_WATCH_SECONDS=3 \
    NVC_STOCK_WAIT=0 NVC_STOCK_WAIT_1=0
N="$ROOT/tools/nvidia_central.sh"

echo "== up"
check "up refuses without the orchestrator flag" "MOJOLEARN_ORCHESTRATOR=0 $ROOT/tools/nvidia_central.sh up 1 2>&1 | grep -q 'only the orchestrator provisions machines'"
check "down refuses without the orchestrator flag" "! MOJOLEARN_ORCHESTRATOR=0 $ROOT/tools/nvidia_central.sh down --all 2>/dev/null"
echo $C1 > "$W/next_container"
MOJOLEARN_NVC_TEST_BALANCE=-4.41 $N up 1 > "$W/up0.log" 2>&1
check "up refuses on a negative balance, nothing rented" "grep -q 'balance is \\\$-4.41' $W/up0.log && [ ! -e $W/devpod_calls ]"
MOJOLEARN_NVC_TEST_BALANCE=50 $N up 1 > "$W/up1.log" 2>&1
check "up tried 4 GPUs (out of stock), then made a 2-GPU nvc1 via the central gate" \
    "grep -q 'up nvc1 60 GPU_COUNT=4 VIA=1' $W/devpod_calls && grep -q 'up nvc1 60 GPU_COUNT=2 VIA=1' $W/devpod_calls && [ -f $W/devpods/nvc1/state.env ]"
check "install: queue up, 2 slots, lease OK" "grep -q 'LEASE OK' $W/up1.log && grep -q 'nvc1: queue up, 2 GPU slots' $W/up1.log"
check "install retired the fixed runpod_guard watchdog on the pod" "docker exec $C1 bash -c '! pgrep -f \"[s]leep 7000\" > /dev/null && [ ! -e /tmp/mojolearn-lease.pid ]'"
wp=$(cat "$W/devpods/nvc1/deadman.pid")
check "the Mac watcher replaced the dead-man in deadman.pid" "kill -0 $wp && ps -p $wp -o command= | grep -q watch.sh && ! pgrep -f 'sleep 7100' > /dev/null"
MOJOLEARN_NVC_TEST_BALANCE=50 $N up 1 > "$W/up2.log" 2>&1
check "up at N is idempotent: nothing rented" "grep -q 'already up' $W/up2.log && [ \$(grep -c '^fake dev_pod up' $W/devpod_calls) = 2 ]"
echo $C2 > "$W/next_container"
MOJOLEARN_NVC_TEST_BALANCE=50 $N up 2 > "$W/up3.log" 2>&1
check "up 2 adds exactly nvc2" "[ -f $W/devpods/nvc2/state.env ] && [ ! -e $W/devpods/nvc3 ]"
check "up refuses N over the 3-pod cap" "! $N up 4 2>/dev/null"

echo "== sync, sh"
$N sync lane-a "$ROOT" > "$W/sync-a.log" 2>&1
$N sync lane-b "$ROOT" > "$W/sync-b.log" 2>&1
pa=$(cat "$W/nvc/lanes/lane-a/pod"); pb=$(cat "$W/nvc/lanes/lane-b/pod")
check "lane-a synced (patch sync to /root/mojolearn-lane-a)" "grep -q 'patch-synced' $W/sync-a.log && grep -q 'lane lane-a is on nvc' $W/sync-a.log"
check "two lanes spread over two pods" "[ -n \"$pa\" ] && [ -n \"$pb\" ] && [ $pa != $pb ]"
check "sh sees no GPU and runs in the lane tree" "[ \"\$($N sh lane-a 'echo \$CUDA_VISIBLE_DEVICES:\$PWD')\" = '-1:/root/mojolearn-lane-a' ]"
check "the lane tree's HEAD is the worktree's merge base" "[ \"\$($N sh lane-a 'git rev-parse HEAD')\" = \"\$(git -C $ROOT merge-base HEAD origin/main)\" ]"

echo "== submit, queue, status, log, cancel, run, fetch"
$N sh lane-a "mkdir -p /root/ev-lane-a && printf '#!/bin/bash\necho gpus=\$CUDA_VISIBLE_DEVICES job=\$NVQ_JOB_ID\nsleep 4\n' > /root/ev-lane-a/gate.sh"
id1=$($N submit lane-a --gpus 2 --note two /root/ev-lane-a/gate.sh)
id2=$($N submit lane-a /root/ev-lane-a/gate.sh)
check "submit prints pod-qualified ids" "[ $id1 = $pa-0001 ] && [ $id2 = $pa-0002 ]"
check "submit refuses a cap over 240" "! $N submit lane-a --cap 241 /root/ev-lane-a/gate.sh 2>/dev/null"
check "submit refuses more GPUs than the pod has" "! $N submit lane-a --gpus 3 /root/ev-lane-a/gate.sh 2>/dev/null"
$N queue > "$W/queue.log" 2>&1
check "queue lists both pods and the jobs" "grep -q '== nvc1' $W/queue.log && grep -q '== nvc2' $W/queue.log && grep -q 'lane-a' $W/queue.log"
check "cancel by another lane is refused" "! $N cancel lane-b $id2 2>/dev/null"
$N cancel lane-a $id2 > /dev/null
waitfor 30 "$N status $id1 2>/dev/null | grep -q '^job 0001: done'"
check "status: the 2-GPU job done, saw CUDA 0,1" "$N status $id1 | grep -q '^job 0001: done' && $N log $id1 | grep -q 'gpus=0,1 job=0001'"
check "the cancelled job is cancelled" "$N status $id2 | grep -q 'cancelled'"
check "run takes a free slot and sees one GPU" "[ \"\$($N run lane-b 'echo \$CUDA_VISIBLE_DEVICES' 2>/dev/null)\" = 0 ]"
$N sh lane-a 'echo result > /root/ev-lane-a/out.txt'
$N fetch lane-a /root/ev-lane-a "$W/fetched" > /dev/null
check "fetch brings a box dir back" "[ \"\$(cat $W/fetched/ev-lane-a/out.txt)\" = result ]"
check "status without an id shows the lease and lanes" "$N status 2>/dev/null | grep -q 'LEASE OK' && $N status 2>/dev/null | grep -q 'lanes: lane-a'"

echo "== down refuses while busy"
$N sh lane-a "printf '#!/bin/bash\nsleep 30\n' > /root/ev-lane-a/long.sh"
idl=$($N submit lane-a /root/ev-lane-a/long.sh); sleep 3
check "down refuses while a job runs" "! $N down $pa 2>/dev/null && [ -f $W/devpods/$pa/state.env ]"
$N cancel lane-a $idl > /dev/null

echo "== the Mac watcher deletes a pod that stops answering"
wb=$(cat "$W/devpods/$pb/deadman.pid"); cb=$( [ $pb = nvc1 ] && echo $C1 || echo $C2 )
docker exec $cb bash -c 'pkill -f "[n]vq backstop"'
waitfor 30 "grep -q 'lease OK after ensure' $W/devpods/$pb/watch-*/watch.log"
check "backstop killed: the watcher ran ensure and the lease is OK again" "grep -q 'lease OK after ensure' $W/devpods/$pb/watch-*/watch.log"
docker stop -t 1 $cb > /dev/null
i=0; while [ $i -lt 40 ] && ! grep -q 'FAKE DELETE' $W/devpods/$pb/watch-*/watch.log 2>/dev/null; do sleep 1; i=$((i + 1)); done
check "three unanswered checks: the watcher deletes the pod" "grep -q 'did not answer 3 checks' $W/devpods/$pb/watch-*/watch.log && grep -q 'FAKE DELETE' $W/devpods/$pb/watch-*/watch.log"

echo "== dev_pod.sh refusals"
DP="$ROOT/tools/dev_pod.sh"
out=$(MOJOLEARN_DEVPOD_STATE="$W/dp2" "$DP" up somelane 60 2>&1)
check "dev_pod up (NVIDIA) refuses a lane" "echo \"\$out\" | grep -q 'lanes do not rent NVIDIA pods'"
for c in extend down run sync; do
    out=$("$DP" $c $pa x 2>&1)
    check "dev_pod $c $pa refuses" "echo \"\$out\" | grep -q 'shared NVIDIA pod'"
done

echo "RESULT: $FAILS failure(s)"
[ $FAILS = 0 ]
