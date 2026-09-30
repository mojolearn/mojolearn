#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# tools/cloudmac_segment_leg.sh -- run ONE rendered segment body (tools/lm_segment_leg.py
# render --arm apple) on a cloud Mac: the Apple provider of tools/lm_run_driver.py.
#
#   bash tools/cloudmac_segment_leg.sh --host m3ultra --body BODY.sh --out RESULTS_DIR [--minutes 600] [--poll 60]
#
# The Mac is an AWS EC2 Mac already held (dedicated host, nothing to rent or
# delete), named in ~/mojolearn-evidence/cloudmacs.tsv (name instance host ip;
# MOJOLEARN_CLOUDMAC_REG overrides) and reached as ec2-user with
# the key in MOJOLEARN_CLOUDMAC_KEY (~/.config/mojolearn/cloudmac.env). It has no GitHub access: this
# checkout's HEAD is pushed to the Mac's bare repo (~/mojolearn.git) as
# refs/steward/<sha> and checked out detached in ~/lmbox/mojolearn, where the
# body runs (the body's BOX is ~/lmbox on the apple arm). No credential goes to
# the Mac: the body carries presigned R2 URLs only, as on every rented box.
#
# ONE METAL JOB: a body still running on the Mac refuses a second (exit 3).
# The body runs detached (HUP ignored, under caffeinate so the Mac never sleeps) and
# survives an ssh drop; this script polls it, and when it exits fetches
# ~/lmbox/gemm_leg_out (checkpoints excluded: they are in R2 and pinned in the
# manifest) into RESULTS_DIR. A previous attempt's gemm_leg_out is moved to
# ~/lmbox/attempts/<utc> first, so a results directory holds one attempt.
# The token stream stays in ~/lmbox/tokens_stream across attempts.
#
# --minutes is the wait before the script gives up on a body that has stopped
# moving: past it, the script keeps waiting while status.txt or the segment
# log changed within the last hour (the Mac costs nothing by the minute), and
# never stops the body. Exit: the body's exit code, 3 busy, 4 gave up waiting.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
[ -f "$HOME/.config/mojolearn/cloudmac.env" ] && . "$HOME/.config/mojolearn/cloudmac.env"
REG="${MOJOLEARN_CLOUDMAC_REG:-$HOME/mojolearn-evidence/cloudmacs.tsv}"
KEY="${MOJOLEARN_CLOUDMAC_KEY:?set MOJOLEARN_CLOUDMAC_KEY (see ~/.config/mojolearn/cloudmac.env)}"
HOST=""; BODY=""; OUT=""; MINUTES=600; POLL=60
while [ $# -gt 0 ]; do
    case "$1" in
        --host) HOST="$2"; shift 2 ;;
        --body) BODY="$2"; shift 2 ;;
        --out) OUT="$2"; shift 2 ;;
        --minutes) MINUTES="$2"; shift 2 ;;
        --poll) POLL="$2"; shift 2 ;;
        *) sed -n 4,6p "$0"; exit 2 ;;
    esac
done
[ -n "$HOST" ] && [ -f "$BODY" ] && [ -n "$OUT" ] || { sed -n 4,6p "$0"; exit 2; }
say() { echo "$(date -u +%H:%M:%S) cloudmac $HOST: $*"; }
IP=$(awk -v n="$HOST" '$1 == n {print $4}' "$REG")
[ -n "$IP" ] || { say "no cloud Mac named $HOST in $REG"; exit 2; }
SSH_OPTS=(-i "$KEY" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR
          -o ConnectTimeout=20 -o ServerAliveInterval=30 -o ServerAliveCountMax=6 -o BatchMode=yes)
cm() { ssh "${SSH_OPTS[@]}" "ec2-user@$IP" "$@"; }
mkdir -p "$OUT"
# a previous attempt's results stay, moved aside (lm_run_driver reads no previous-* directory)
[ -d "$OUT/gemm_leg_out" ] && mv "$OUT/gemm_leg_out" "$OUT/previous-attempt-$(date -u +%Y%m%dT%H%M%SZ)"
cp "$BODY" "$OUT/body.sh"

# ---- one Metal job: refuse when a body is still running ----
if cm '[ -f ~/lmbox/body.pid ] && kill -0 "$(cat ~/lmbox/body.pid)" 2>/dev/null'; then
    say "busy: a body is still running (pid $(cm 'cat ~/lmbox/body.pid')); refusing a second Metal job"
    exit 3
fi

# ---- the code: this checkout's HEAD, pushed and checked out detached ----
SHA=$(git -C "$ROOT" rev-parse --verify HEAD) || exit 2
GIT_SSH_COMMAND="ssh ${SSH_OPTS[*]}" git -C "$ROOT" push -q -f "ec2-user@$IP:mojolearn.git" "$SHA:refs/steward/$SHA" || { say "push failed"; exit 2; }
say "pushed $SHA"
cm "set -e; mkdir -p ~/lmbox ~/lmbox/attempts
    [ -d ~/lmbox/mojolearn/.git ] || git clone -q --no-checkout ~/mojolearn.git ~/lmbox/mojolearn
    cd ~/lmbox/mojolearn && git fetch -q origin refs/steward/$SHA && git checkout -q -f --detach $SHA && git log -1 --format='checkout %h %s' | cut -c1-120
    # the body needs a python >= 3.10 for its venv (macOS ships 3.9)
    [ -x ~/.pixi/bin/python3.12 ] || ~/.pixi/bin/pixi global install python=3.12 > ~/lmbox/pixi_python.log 2>&1
    ~/.pixi/bin/python3.12 -V
    if [ -d ~/lmbox/gemm_leg_out ]; then mv ~/lmbox/gemm_leg_out ~/lmbox/attempts/\$(date -u +%Y%m%dT%H%M%SZ); fi
    rm -f ~/lmbox/body.rc ~/lmbox/body.log ~/lmbox/lm_segment_ready ~/lmbox/lm_segment_done" || { say "preparing the Mac failed"; exit 2; }
scp -q "${SSH_OPTS[@]}" "$BODY" "ec2-user@$IP:lmbox/body.sh" || { say "copying the body failed"; exit 2; }
# macOS nohup refuses without a terminal ("can't detach from console"): ignore HUP by
# hand; the pid is caffeinate's, which lives exactly as long as the body
cm 'cd ~/lmbox && sh -c "trap \"\" HUP; exec caffeinate -dims sh -c \"sh body.sh > body.log 2>&1; echo \\\$? > body.rc\"" > /dev/null 2>&1 < /dev/null & echo $! > ~/lmbox/body.pid; sleep 2; kill -0 "$(cat ~/lmbox/body.pid)" && echo started pid $(cat ~/lmbox/body.pid)' \
    || { say "the body did not start"; exit 2; }
say "body started ($(basename "$BODY")); polling every ${POLL}s"

# ---- poll: survives ssh drops; prints each new status line ----
start=$(date +%s); seen=0; rc=""
while :; do
    sleep "$POLL"
    snap=$(cm 'cat ~/lmbox/body.rc 2>/dev/null | sed "s/^/RC /"; s=$(ls ~/lmbox/gemm_leg_out/*/status.txt 2>/dev/null | head -1);
               [ -n "$s" ] && { echo "LINES $(wc -l < "$s")"; cat "$s" | sed "s/^/ST /"; };
               m=$(ls -t ~/lmbox/gemm_leg_out/*/status.txt ~/lmbox/gemm_leg_out/*/*.log 2>/dev/null | head -1);
               [ -n "$m" ] && echo "AGE $(( $(date +%s) - $(stat -f %m "$m") ))"' 2>/dev/null) || { say "poll: ssh failed; retrying"; continue; }
    lines=$(printf '%s\n' "$snap" | sed -n 's/^ST //p')
    n=$(printf '%s\n' "$lines" | grep -c . || true)
    if [ "$n" -gt "$seen" ]; then printf '%s\n' "$lines" | tail -n +"$((seen + 1))" | sed 's/^/  | /'; seen=$n; fi
    rc=$(printf '%s\n' "$snap" | sed -n 's/^RC //p' | head -1)
    [ -n "$rc" ] && break
    age=$(printf '%s\n' "$snap" | sed -n 's/^AGE //p' | head -1)
    if [ $(( $(date +%s) - start )) -gt $(( MINUTES * 60 )) ] && [ "${age:-99999}" -gt 3600 ]; then
        say "past --minutes $MINUTES and nothing moved for ${age:-?}s; giving up waiting (the body is NOT stopped)"
        rc=4; break
    fi
done
say "body exit=$rc after $(( ($(date +%s) - start) / 60 )) min; fetching results"

# ---- fetch: everything but checkpoints ----
for try in 1 2 3; do
    cm 'cd ~/lmbox && tar -cf - --exclude "*.blm" gemm_leg_out body.log body.rc 2>/dev/null' | tar -xf - -C "$OUT" && break
    say "fetch try $try failed"; sleep 20
done
ls "$OUT"/gemm_leg_out/*/segment/segment.json > /dev/null 2>&1 && say "segment.json came home" || say "no segment.json came home"
exit "$rc"
