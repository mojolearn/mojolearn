# Sourced by packaging/linux/build_sets.sh (and its CPU test,
# tools/test_binding_timeout.py). One binding build, bounded in time.
#
# WHY (2026-09-25). In 0.8.19 every Linux GPU leg built 75 of 76 bindings in
# 11 to 13 minutes and then spent the rest of its bound on ONE binding, the
# FAST svm, stuck in codegen: nothing named it until the whole leg was killed
# at 2368 s. With a per-binding bound the stuck build is stopped, logged as
# TIMEOUT and named as a FINDING while the other builds carry on, so the leg
# fails in minutes with the culprit's name instead of at its deadline.
#
# THE BOUND. RELEASE_BINDING_TIMEOUT_SECONDS, default 1200. The slowest
# binding measured over 25 legs of 0.8.16 to 0.8.19 took 339 s (identical rf;
# median of the ten slowest 60 to 138 s), so 1200 s is 3.5 x that worst case.
# The variable is deliberately NOT MOJOLEARN_*: every MOJOLEARN_* variable
# reaches the binding cache key (tools/bincache.py), and a scheduling bound
# must not change which cached binary a build may reuse.
#
# NOT `timeout(1)`: GNU timeout moves its child into a new process group, and
# the NVIDIA and AMD job guards measure RSS over the build's process group,
# so a compile under timeout(1) would vanish from the guard's memory cap.
# The watchdog below keeps every process in the caller's group and stops the
# build's whole process tree itself.

binding_timeout_seconds() {
  local t="${RELEASE_BINDING_TIMEOUT_SECONDS:-1200}"
  [[ "$t" =~ ^[1-9][0-9]*$ ]] || { echo 'RELEASE_BINDING_TIMEOUT_SECONDS must be a positive integer' >&2; return 2; }
  printf '%s' "$t"
}

# ONE MERGED BUILD, BOUNDED (the merged bindings, 2026-10-20). In the merged
# layout (MOJOLEARN_MERGED_BINDINGS=1, bindings/merged_build.sh) one compile
# builds every member of a merged group at once, so one binding's bound is far
# too short for it. RELEASE_MERGED_TIMEOUT_SECONDS, default 3 x the binding
# bound (3600 s at the default): the per-binding legs built 75 bindings in 11
# to 13 minutes over four jobs (about 45 to 50 minutes of serial compile), and
# a merged group compiles its members' shared code once; the bound is
# NOT yet measured on a merged leg, so read the first leg's build_logs/*_merged_*.log
# durations and set it from them. It stays inside the 2-hour rental dead-man.
# Not MOJOLEARN_* for the same reason as the binding bound (bincache key).
merged_timeout_seconds() {
  local b t
  b=$(binding_timeout_seconds) || return 2
  t="${RELEASE_MERGED_TIMEOUT_SECONDS:-$((3 * b))}"
  [[ "$t" =~ ^[1-9][0-9]*$ ]] || { echo 'RELEASE_MERGED_TIMEOUT_SECONDS must be a positive integer' >&2; return 2; }
  printf '%s' "$t"
}

# STALLED, NOT ONLY TIMED OUT (lane xcc-retire-1, 2026-11-27). The release's
# build legs used to learn of a hung compile only at its bound (the merged
# bound is 95 min on GitHub); the per-binding cross-compile check, which the
# release no longer runs beside the GitHub legs, called it STALLED after 240 s.
# The same rule now runs here: tools/compile_watch.py (shared with
# tools/cross_compile_check.py, one rule) watches the build's process TREE and
# its log, and after RELEASE_BINDING_STALL_SECONDS (default 240; 0 turns it
# off) of CPU gaining under 0.5 s and no new log output the build is stopped,
# logged as STALLED and returns 125, so the leg fails by name in minutes. Not
# MOJOLEARN_* for the same reason as the bounds (bincache key).
_BINDING_TIMEOUT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
binding_stall_seconds() {
  local t="${RELEASE_BINDING_STALL_SECONDS:-240}"
  [[ "$t" =~ ^[0-9]+$ ]] || { echo 'RELEASE_BINDING_STALL_SECONDS must be a nonnegative integer' >&2; return 2; }
  printf '%s' "$t"
}

# _child_pids PID: PID's direct children, one per line. pgrep when present;
# else /proc (minimal Linux images and containers ship without procps, and a
# silent empty pgrep left the whole build tree running past its bound).
_child_pids() {
  local parent="$1" f line rest
  local -a fields
  if command -v pgrep >/dev/null 2>&1; then pgrep -P "$parent" 2>/dev/null; return 0; fi
  [ -d /proc ] || return 0
  for f in /proc/[0-9]*/stat; do
    { read -r line < "$f"; } 2>/dev/null || continue
    rest=${line##*) }                       # after "pid (comm) ": state ppid ...
    read -r -a fields <<< "$rest"
    if [ "${fields[1]:-}" = "$parent" ]; then f=${f#/proc/}; printf '%s\n' "${f%/stat}"; fi
  done
  return 0
}

# kill_tree SIG PID: SIG to PID's descendants (deepest first), then PID.
kill_tree() {
  local sig="$1" pid="$2" child
  for child in $(_child_pids "$pid"); do kill_tree "$sig" "$child"; done
  kill "-$sig" "$pid" 2>/dev/null || true
}

# with_binding_timeout SECONDS LOG CMD...: run CMD (stdout and stderr appended
# to LOG); return its status, or 124 after stopping its whole process tree
# once SECONDS pass (TERM, then KILL 10 s later), with a TIMEOUT line in LOG;
# or 125 once the build STALLS (above), with a STALLED line in LOG.
_stop_build_tree() {
  local pid="$1" grace=0
  kill_tree TERM "$pid"
  while kill -0 "$pid" 2>/dev/null && (( grace < 10 )); do sleep 1; grace=$((grace + 1)); done
  kill_tree KILL "$pid"
  wait "$pid" 2>/dev/null
}
with_binding_timeout() {
  local limit="$1" log="$2" pid rc=0 started=$SECONDS stall watcher="" marker
  shift 2
  stall=$(binding_stall_seconds) || return 2
  marker="$log.stalled"
  rm -f "$marker"
  "$@" >> "$log" 2>&1 &
  pid=$!
  local tool="${BINDING_STALL_TOOL:-$_BINDING_TIMEOUT_DIR/../../tools/compile_watch.py}"
  if (( stall > 0 )) && [[ -f "$tool" ]] && command -v python3 >/dev/null 2>&1; then
    python3 "$tool" watch --pid "$pid" --log "$log" --stall "$stall" --marker "$marker" \
      --poll "${RELEASE_BINDING_STALL_POLL:-5}" > /dev/null 2>&1 &
    watcher=$!
  fi
  while kill -0 "$pid" 2>/dev/null; do
    if (( SECONDS - started >= limit )); then
      echo "TIMEOUT after ${limit}s (RELEASE_BINDING_TIMEOUT_SECONDS); stopping the build's process tree" >> "$log"
      [[ -n "$watcher" ]] && { kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null; }
      _stop_build_tree "$pid"
      return 124
    fi
    if [[ -s "$marker" ]]; then
      echo "STALLED: $(cat "$marker") (RELEASE_BINDING_STALL_SECONDS); stopping the build's process tree" >> "$log"
      rm -f "$marker"
      [[ -n "$watcher" ]] && wait "$watcher" 2>/dev/null
      _stop_build_tree "$pid"
      return 125
    fi
    sleep 1
  done
  [[ -n "$watcher" ]] && { kill "$watcher" 2>/dev/null; wait "$watcher" 2>/dev/null; }
  rm -f "$marker"
  wait "$pid" || rc=$?
  return "$rc"
}
