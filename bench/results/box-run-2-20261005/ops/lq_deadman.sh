#!/bin/bash
# amd_deadman3.sh (ON the droplet, setsid nohup): delete THIS droplet when idle IDLE_MIN (no lq job, build or pixi install)
# or at the hard cap CAP_MIN. Same pattern as amd_deadman2.sh, with the lq job names in the busy test and a start line.
# Disarm: rm /root/deadman.armed. Extend the cap: echo <minutes-from-start> > /root/deadman.cap (read every pass).
ID=$1; IDLE_MIN=${IDLE_MIN:-60}; CAP_MIN=${CAP_MIN:-360}; LOG=/root/deadman.log; start=$(date +%s); last=$start
echo "$(date -u +%FT%TZ) armed droplet=$ID idle=${IDLE_MIN}m cap=${CAP_MIN}m cap_at=$(date -u -d @$((start+CAP_MIN*60)) +%FT%TZ)" >> $LOG
while true; do
  [ -f /root/deadman.armed ] || { echo "$(date -u +%FT%TZ) disarmed" >> $LOG; exit 0; }
  [ -s /root/deadman.cap ] && CAP_MIN=$(tr -dc 0-9 < /root/deadman.cap)
  pgrep -f "lq/box_job.sh|overlay_race_job|bench_board_algos|record_identity_column|bindings/build|identity_break.py|pixi install|lq_setup_busy|identical_wave|cf_driver|smoke_fit|compile_slot|mojo build" > /dev/null && last=$(date +%s)
  # box-run-2: the Mac harvester renews /root/hold-until (epoch); a live hold counts as busy.
  [ -s /root/hold-until ] && [ "$(date +%s)" -lt "$(tr -dc 0-9 < /root/hold-until)" ] && last=$(date +%s)
  now=$(date +%s); idle=$(( (now-last)/60 )); age=$(( (now-start)/60 ))
  echo "$now idle=$idle age=$age cap=$CAP_MIN" > /root/deadman.beat
  if [ $idle -ge $IDLE_MIN ] || [ $age -ge $CAP_MIN ]; then
    echo "$(date -u +%FT%TZ) idle=${idle}m age=${age}m: deleting droplet $ID" >> $LOG
    curl -s -o /dev/null -w "%{http_code}\n" -X DELETE -H "Authorization: Bearer $(cat /root/.do_token)" https://api.digitalocean.com/v2/droplets/$ID >> $LOG
    sleep 600
  fi
  sleep 120
done
