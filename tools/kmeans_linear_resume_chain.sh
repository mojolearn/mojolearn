#!/bin/sh
# lane/kmeans-linear-speed, resumed 2026-09-17 evening: the whole finishing
# sequence on ONE RunPod RTX 4090, detached (`nohup setsid sh
# tools/kmeans_linear_resume_chain.sh > /root/kls_out/chain.log 2>&1 < /dev/null &`),
# one phase marker per phase under /root/kls_out so the Mac side can pull
# after each. Expects /root/mojolearn = the branch tip (shipped by
# tools/trees_leg.sh) and /root/mainsrc = origin/main's source with its own
# SHIPPED_COMMIT.txt (unpacked by the Mac side before launch). Every stage is
# tools/kmeans_linear_body.sh's. POSIX sh.
set -u
R=/root/mojolearn
B="sh $R/tools/kmeans_linear_body.sh"
O=/root/kls_out
mkdir -p "$O"
mark() { echo "$1 $(date -u +%H:%M:%S)" >> "$O/chain.progress.txt"; : > "$O/phase.$1.done"; }
D80="-D MOJOLEARN_EXPERIMENTAL_KMEANS_BLOCK_ACC=1"
D81="-D MOJOLEARN_EXPERIMENTAL_KMEANS_DEVICE_SCALE=1"
S80="-D MOJOLEARN_KMEANS_BLOCK_ACC_SABOTAGE=1"
S81="-D MOJOLEARN_KMEANS_DEVICE_SCALE_SABOTAGE=1"

cd "$R" || exit 9
echo "chain start $(date -u) tip=$(cat $R/SHIPPED_COMMIT.txt) main=$(cat /root/mainsrc/SHIPPED_COMMIT.txt)" > "$O/chain.progress.txt"

# Phase 1: environment and the two benchmark blocks.
$B setup
[ -e "$O/setup/setup.done" ] || { echo "setup failed" >> "$O/chain.progress.txt"; exit 1; }
mark setup

# Phase 2: the six arms (core + estimators), base from main's source.
KLS_SRC=/root/mainsrc $B arm base ""
$B arm off ""
$B arm blk "$D80"
$B arm both "$D80 $D81"
$B arm sabo80 "$D80 $D81 $S80"
$B arm sabo81 "$D80 $D81 $S81"
for a in base off blk both sabo80 sabo81; do
    [ -e "$O/arm-$a/arm.done" ] || { echo "arm $a failed" >> "$O/chain.progress.txt"; exit 1; }
done
mark arms

# Phase 3: the interleaved A/B, the lane's numbers, before anything long.
$B ab both base 7 kmeans,ols,pca taxi,istella
$B ab blk base 7 kmeans taxi,istella
$B ab both blk 7 kmeans taxi,istella
mark ab

# Phase 4: identity, cuda column, the 16 lanes, every arm; then the diffs.
for a in base off blk both sabo80 sabo81; do $B identity $a; done
$B diff cuda.base-off-blk-both x "$O/identity-base/identity.json" "$O/identity-off/identity.json" "$O/identity-blk/identity.json" "$O/identity-both/identity.json"
$B diff sabotage80 x "$O/identity-both/identity.json" "$O/identity-sabo80/identity.json"
$B diff sabotage81 x "$O/identity-both/identity.json" "$O/identity-sabo81/identity.json"
mark identity

# Phase 5: the checks, one per process (the second check in one process hangs on this box).
( cd "$R" && export PATH="$HOME/.pixi/bin:$PATH" && pixi run mojo build -I . -D MOJOLEARN_NUMERIC_IDENTICAL=1 cluster/tools/kmeans_linear_checks_main.mojo -o /root/kls_checks ) > "$O/checks_build.log" 2>&1
mkdir -p "$O/checks"
for c in blocked scale privatized; do
    timeout -k 30 900 /root/kls_checks "$c" >> "$O/checks/checks.txt" 2>&1; echo "$c rc=$?" >> "$O/checks/checks.txt"
done
mark checks

# Phase 6: the other lanes that reach the Lloyd loop, cuda column, every arm.
for a in base off blk both sabo80 sabo81; do
    if [ "$a" = base ]; then KLS_SRC=/root/mainsrc $B arm_extra base ""; else
        case $a in off) d="";; blk) d="$D80";; both) d="$D80 $D81";; sabo80) d="$D80 $D81 $S80";; sabo81) d="$D80 $D81 $S81";; esac
        $B arm_extra "$a" "$d"
    fi
done
for a in base off blk both sabo80 sabo81; do $B identity2 $a; done
$B diff cuda2.base-off-blk-both x "$O/identity2-base/identity.json" "$O/identity2-off/identity.json" "$O/identity2-blk/identity.json" "$O/identity2-both/identity.json"
$B diff sabotage80.2 x "$O/identity2-both/identity.json" "$O/identity2-sabo80/identity.json"
$B diff sabotage81.2 x "$O/identity2-both/identity.json" "$O/identity2-sabo81/identity.json"
mark identity2

# Phase 7: the cpu column, before and after, both lane sets.
$B host base
$B host both
$B identity_cpu base
$B identity_cpu both
$B identity2_cpu base
$B identity2_cpu both
$B diff cpu.base-both x "$O/identity_cpu-base/identity.json" "$O/identity_cpu-both/identity.json"
$B diff cpu2.base-both x "$O/identity2_cpu-base/identity.json" "$O/identity2_cpu-both/identity.json"
$B diff before.cuda-vs-cpu x "$O/identity-base/identity.json" "$O/identity_cpu-base/identity.json"
$B diff after.cuda-vs-cpu x "$O/identity-both/identity.json" "$O/identity_cpu-both/identity.json"
$B diff before2.cuda-vs-cpu x "$O/identity2-base/identity.json" "$O/identity2_cpu-base/identity.json"
$B diff after2.cuda-vs-cpu x "$O/identity2-both/identity.json" "$O/identity2_cpu-both/identity.json"
mark cpu

echo "chain done $(date -u)" >> "$O/chain.progress.txt"
: > "$O/chain.done"
