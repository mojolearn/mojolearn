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

# kill_tree SIG PID: SIG to PID's descendants (deepest first), then PID.
kill_tree() {
  local sig="$1" pid="$2" child
  for child in $(pgrep -P "$pid" 2>/dev/null); do kill_tree "$sig" "$child"; done
  kill "-$sig" "$pid" 2>/dev/null || true
}

# with_binding_timeout SECONDS LOG CMD...: run CMD (stdout and stderr appended
# to LOG); return its status, or 124 after stopping its whole process tree
# once SECONDS pass (TERM, then KILL 10 s later), with a TIMEOUT line in LOG.
with_binding_timeout() {
  local limit="$1" log="$2" pid rc=0 started=$SECONDS
  shift 2
  "$@" >> "$log" 2>&1 &
  pid=$!
  while kill -0 "$pid" 2>/dev/null; do
    if (( SECONDS - started >= limit )); then
      echo "TIMEOUT after ${limit}s (RELEASE_BINDING_TIMEOUT_SECONDS); stopping the build's process tree" >> "$log"
      kill_tree TERM "$pid"
      local grace=0
      while kill -0 "$pid" 2>/dev/null && (( grace < 10 )); do sleep 1; grace=$((grace + 1)); done
      kill_tree KILL "$pid"
      wait "$pid" 2>/dev/null
      return 124
    fi
    sleep 1
  done
  wait "$pid" || rc=$?
  return "$rc"
}
