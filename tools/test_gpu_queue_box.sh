#!/bin/bash
# tools/test_gpu_queue_box.sh -- tests tools/gpu_queue_box.sh in PLAIN mode (the
# NVIDIA pods: no systemd) on fake GPU slots, with the lease's pod DELETE faked.
# Runs as root on Linux in a throwaway box: tools/test_nvidia_central.sh runs it
# in an ubuntu:22.04 container. Touches only $T (a sandbox queue dir, lock dir
# and lane trees); needs no GPU, no network, no key.
set -u   # no pipefail: the checks pipe into grep -q, which closes the pipe early
SRC=${1:?usage: test_gpu_queue_box.sh <path to gpu_queue_box.sh>}
T=${T:-/tmp/gq-test}
rm -rf $T; mkdir -p $T/q/bin $T/locks
cp "$SRC" $T/q/bin/nvq; chmod 755 $T/q/bin/nvq
export GQ_DIR=$T/q GQ_LOCKD=$T/locks GQ_UNIT=nvqtest
export GQ_DAEMON_POLL=1 GQ_RUNNER_POLL=1 GQ_KILL_WAIT=1 GQ_BACKSTOP_POLL=1 GQ_MIN_SECONDS=1
Q=$T/q/bin/nvq
cd $T/q && echo 4 > slots && echo nvidia > vendor && echo plain > mode && echo - > service && echo 30 > default_cap && echo 30 > max_cap
printf 'POD_ID=fakepod\nIDLE_MIN=30\nGRACE_MIN=15\nIDLE_SEC=12\nGRACE_SEC=6\nFAKE_DELETE=1\n' > lease.conf
date +%s > lease.last_busy
for l in lane-a lane-b; do mkdir -p /root/mojolearn-$l; done
EV=$T/ev; mkdir -p $EV
FAILS=0
ok() { echo "PASS: $*"; }
bad() { echo "FAIL: $*"; FAILS=$((FAILS + 1)); }
check() { if eval "$2"; then ok "$1"; else bad "$1"; fi; }
stof() { cat $T/q/$1/status 2>/dev/null; }
waitst() {  # id state seconds
    local i=0; while [ $i -lt $3 ]; do [ "$(stof $1)" = "$2" ] && return 0; sleep 1; i=$((i + 1)); done; return 1
}
job() {  # name body -> script path
    printf '#!/bin/bash\necho "$(date +%%s.%%N) start %s cuda=$CUDA_VISIBLE_DEVICES id=$NVQ_JOB_ID lane=$NVQ_LANE pwd=$PWD" >> %s/trace\n%s\necho "$(date +%%s.%%N) end %s" >> %s/trace\n' \
        "$1" "$EV" "$2" "$1" "$EV" > $EV/$1.sh
    echo $EV/$1.sh
}

echo "== ensure starts the dispatcher and the lease backstop"
$Q ensure
check "dispatcher holds its lock" "! flock -n $T/q/daemon.lock true"
check "backstop holds its lock" "! flock -n $T/q/backstop.lock true"
check "a second daemon refuses" "! $Q daemon 2>/dev/null"
check "lease-check OK" "$Q lease-check | grep -q '^LEASE OK'"

echo "== FIFO on 4 slots: a 2-GPU job at the head blocks 1-GPU jobs behind it (no backfill)"
j1=$($Q submit lane-a 3 20 "$(job j1 'sleep 4')")
j2=$($Q submit lane-b 2 20 "$(job j2 'sleep 2')")
j3=$($Q submit lane-a 1 20 "$(job j3 'sleep 1')")
check "ids are in submission order" "[ $j1 = 0001 ] && [ $j2 = 0002 ] && [ $j3 = 0003 ]"
waitst $j1 running 10
sleep 2
check "j2 (2 GPUs) waits while j1 holds 3 of 4" "[ \"\$(stof $j2)\" = queued ]"
check "j3 (1 GPU) does not jump the 2-GPU head" "[ \"\$(stof $j3)\" = queued ]"
check "pending counts both" "[ \"\$(cat $T/q/pending)\" = 2 ]"
waitst $j3 done 25 || true
check "j1 done exit 0" "[ \"\$(stof $j1)\" = done ] && [ \"\$(cat $T/q/$j1/exit)\" = 0 ]"
check "j2 then j3 ran, in order" "grep -n 'start j' $EV/trace | awk -F'start ' '{print \$2}' | cut -c1-2 | tr '\n' ' ' | grep -q '^j1 j2 j3'"
check "j1 saw 3 GPUs, as CUDA 0,1,2" "grep -q 'start j1 cuda=0,1,2 id=0001 lane=lane-a pwd=/root/mojolearn-lane-a' $EV/trace"
check "j2 saw 2 GPUs" "grep -q 'start j2 cuda=[0-3],[0-3] id=0002 lane=lane-b' $EV/trace"
check "j2 started only after j1 ended" "awk '/end j1/{e=\$1} /start j2/{s=\$1} END{exit !(s >= e)}' $EV/trace"

echo "== background work keeps the slot; a failing job is failed"
j4=$($Q submit lane-a 4 20 "$(job j4 '( sleep 4; echo "$(date +%s.%N) bg-end" >> '$EV'/trace ) & exit 3')")
waitst $j4 running 10
sleep 2
check "j4 still running while its background child lives" "[ \"\$(stof $j4)\" = running ]"
waitst $j4 failed 15
check "j4 failed with exit 3 after the child ended" "[ \"\$(stof $j4)\" = failed ] && [ \"\$(cat $T/q/$j4/exit)\" = 3 ] && grep -q bg-end $EV/trace"

echo "== the cap kills a setsid child (cap in test minutes = seconds)"
j5=$($Q submit lane-b 1 3 "$(job j5 'setsid sleep 300 & sleep 300')")
waitst $j5 timeout 30
check "j5 timed out" "[ \"\$(stof $j5)\" = timeout ]"
check "no sleep 300 survives" "! pgrep -f 'sleep 300' > /dev/null"
check "a cap over max_cap is refused" "! $Q submit lane-a 1 31 $EV/j3.sh 2>/dev/null"

echo "== cancel: queued, running, and by another lane (refused)"
j6=$($Q submit lane-a 4 20 "$(job j6 'sleep 200')")
j7=$($Q submit lane-a 1 20 "$(job j7 'true')")
waitst $j6 running 10
check "lane-b cannot cancel lane-a's job" "! $Q cancel lane-b $j6 2>/dev/null"
$Q cancel lane-a $j7 > /dev/null
check "queued j7 cancelled" "[ \"\$(stof $j7)\" = cancelled ]"
$Q cancel lane-a $j6 > /dev/null
waitst $j6 cancelled 15
check "running j6 cancelled, its sleep killed" "[ \"\$(stof $j6)\" = cancelled ] && ! pgrep -f 'sleep 200' > /dev/null"

echo "== the dispatcher restarting leaves a running job alone"
j8=$($Q submit lane-b 1 20 "$(job j8 'sleep 6')")
waitst $j8 running 10
pkill -f "nvq daemon"; sleep 1
check "dispatcher is down" "flock -n $T/q/daemon.lock true"
$Q ensure > /dev/null
sleep 2
check "j8 still running under the new dispatcher" "[ \"\$(stof $j8)\" = running ] && ! grep -q requeued $T/q/$j8/log"
waitst $j8 done 15
check "j8 done" "[ \"\$(stof $j8)\" = done ]"

echo "== a job cut short (runner and dispatcher killed: a container restart) is requeued once, then interrupted"
j9=$($Q submit lane-a 1 20 "$(job j9 'sleep 100')")
waitst $j9 running 10
pkill -f "nvq daemon"; pkill -9 -f "nvq runner $j9"; pkill -f 'sleep 100'; sleep 1
$Q ensure > /dev/null
waitst $j9 running 10
check "j9 requeued at the head and running again (RESTARTS=1)" "grep -q '^RESTARTS=1' $T/q/$j9/job.env && grep -q requeued $T/q/$j9/log"
pkill -f "nvq daemon"; pkill -9 -f "nvq runner $j9"; pkill -f 'sleep 100'; sleep 1
$Q ensure > /dev/null; sleep 2
check "j9 cut short twice: interrupted" "[ \"\$(stof $j9)\" = interrupted ]"

echo "== a FRONT job (GQ_FRONT=1, lq add --front) starts before plain queued jobs, never before the running one"
ja=$($Q submit lane-a 4 20 "$(job fa 'sleep 3')")
jb=$($Q submit lane-b 1 20 "$(job fb 'sleep 1')")
jc=$(GQ_FRONT=1 $Q submit lane-a 1 20 "$(job fc 'sleep 1')")
jd=$(GQ_FRONT=1 $Q submit lane-b 1 20 "$(job fd 'sleep 1')")
check "the front jobs carry the mark, the plain one does not" "[ -e $T/q/$jc/front ] && [ -e $T/q/$jd/front ] && [ ! -e $T/q/$jb/front ]"
waitst $jb done 40 || true
check "running job first, then the front jobs in submission order, then the plain job" "grep 'start f' $EV/trace | awk -F'start ' '{print \$2}' | cut -c1-2 | tr '\n' ' ' | grep -q '^fa fc fd fb'"

echo "== the lease: busy pushes the deadline; hold counts; idle closes and deletes"
sleep 3
dl1=$(cat $T/q/lease.deadline)
( $Q hold 'sleep 5' ) &
sleep 3
dl2=$(cat $T/q/lease.deadline)
check "a hold pushes the deadline forward" "[ $dl2 -gt $dl1 ]"
wait
exec 5>$T/locks/gpu2.lock; flock 5          # a `run` holding slot 2
sleep 3; dl3=$(cat $T/q/lease.deadline)
check "a held slot pushes the deadline forward" "[ $dl3 -gt $dl2 ]"
exec 5>&-
check "not closing while recently busy" "[ ! -e $T/q/closing ]"
i=0; while [ $i -lt 30 ] && [ ! -e $T/q/lease.deleted_ok ]; do sleep 1; i=$((i + 1)); done
check "idle past the deadline: closing and the pod DELETE issued by the dispatcher" "[ -e $T/q/closing ] && grep -q 'FAKE DELETE fakepod by dispatcher' $T/q/lease.deleted"
check "submit refuses once closing" "! $Q submit lane-a 1 20 $EV/j3.sh 2>/dev/null"
check "ensure refuses once closing" "! $Q ensure 2>/dev/null"

echo "== the backstop deletes the pod when the dispatcher is gone"
pkill -f "nvq daemon"; pkill -f "nvq backstop"; sleep 1
rm -f $T/q/closing $T/q/lease.deleted_ok $T/q/lease.deleted
date +%s > $T/q/lease.last_busy; echo $(( $(date +%s) + 12 )) > $T/q/lease.deadline
setsid nohup $Q backstop >> $T/q/lease.log 2>&1 < /dev/null &
sleep 2
check "lease-check BAD with no dispatcher? no: OK while the backstop lives and the deadline holds" "$Q lease-check | grep -q '^LEASE OK .*dispatcher=dead'"
i=0; while [ $i -lt 30 ] && ! grep -q 'by backstop' $T/q/lease.deleted 2>/dev/null; do sleep 1; i=$((i + 1)); done
check "backstop deleted the pod grace seconds past the deadline" "grep -q 'FAKE DELETE fakepod by backstop' $T/q/lease.deleted"
pkill -f "nvq backstop"; sleep 1
check "lease-check BAD once the backstop is dead" "$Q lease-check | grep -q '^LEASE BAD'"

echo "== queue and status print"
$Q queue | tail -8
$Q status $j1 | head -3
pkill -f "nvq " 2>/dev/null
echo "RESULT: $FAILS failure(s)"
[ $FAILS = 0 ]
