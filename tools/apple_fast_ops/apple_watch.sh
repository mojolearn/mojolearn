#!/bin/bash
# Laptop-side watcher for M2 (builds) and M3 (queue). Grep-sized output only.
K=~/.ssh/mambik-l8.pem; S="$HOME/mojolearn-evidence/apple_watch.state"; touch "$S"
m3=$(ssh -o ConnectTimeout=20 -i $K ec2-user@54.157.1.251 'echo "pos=$(cat ~/mq/pos) total=$(wc -l < ~/mq/queue.txt) runners=$(pgrep -f "^bash /Users/ec2-user/mq.sh" | wc -l | tr -d " ") free=$(df -g ~ | tail -1 | awk "{print \$4}")G results=$(wc -l < ~/mq/results.txt)"; tail -n 1 ~/mq/watchdog.log | grep -o "\"alerts\": \[[^]]*\]"; tail -n 3 ~/mq/results.txt | cut -c1-160' 2>&1)
m2=$(ssh -o ConnectTimeout=20 -i $K ec2-user@54.237.205.45 'echo "mojo_builds=$(pgrep -f "mojo build" | wc -l | tr -d " ") free=$(df -g ~ | tail -1 | awk "{print \$4}")G"; ls -t ~/m2-arms/build_*.log 2>/dev/null | head -1 | xargs tail -n 4 2>/dev/null | grep -vE "Updating files" | cut -c1-160' 2>&1)
prev=$(cat "$S"); cur="$(echo "$m3" | head -1 | grep -o 'pos=[0-9]*') $(echo "$m3" | head -1 | grep -o 'results=[0-9]*')"
echo "M3: $m3" | head -6; echo "M2: $m2" | head -6
echo "$m3" | grep -q 'runners=0' && echo "ALERT M3 runner missing"
echo "$m3" | grep -qE 'free=[0-4]G' && echo "ALERT M3 disk low"
echo "$m2" | grep -qE 'free=[0-3]G' && echo "ALERT M2 disk low"
echo "$m2" | grep -q FAILED && echo "ALERT M2 build failed"
[ "$prev" = "$cur" ] && echo "NOTE M3 no new position/results since last check ($cur)"
echo "$cur" > "$S"
# keep M2/M3 bare mains in sync with GitHub (fast-forward only; new M2 commits get published first)
cd ~/CascadeProjects/mojolearn && export GIT_SSH_COMMAND="ssh -i $K -o ConnectTimeout=20"
git fetch -q origin main && git fetch -q m2 main && git fetch -q m3 main
if git merge-base --is-ancestor origin/main m2/main && [ "$(git rev-parse m2/main)" != "$(git rev-parse origin/main)" ]; then git push -q origin m2/main:refs/heads/main && echo "SYNC published M2 main to GitHub"; git fetch -q origin main; fi
for r in m2 m3; do [ "$(git rev-parse $r/main)" = "$(git rev-parse origin/main)" ] || { git push -q $r origin/main:refs/heads/main 2>/dev/null && echo "SYNC $r main -> GitHub $(git rev-parse --short origin/main)" || echo "ALERT $r main diverged from GitHub"; }; done
# keep opponent-only jobs at the end of the M3 queue (Andrew, Oct 4)
ssh -o ConnectTimeout=20 -i $K ec2-user@54.157.1.251 'python3 ~/mq/opp_to_end.py' 2>&1 | grep -E 'OPP_TO_END|Error|retry'
