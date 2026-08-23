#!/bin/sh
# Serialize heavy builds/tests across parallel sessions.
# macOS has no flock(1), so this blocks on a Python fcntl lock without polling.
# Usage: tools/with_build_lock.sh <command> [args...]
#
# RE-ENTRANT (2026-08-23). A process that already holds the lock runs the
# command directly. An fcntl flock is per open-file-description, so a child
# of the holder that opens the file again BLOCKS ON ITS OWN ANCESTOR:
# `with_identical_mode.sh pixi run check-unsupervised-identity` took the
# lock, the gate script's FAST arm called this again, and the pair sat for
# 25 minutes with every other session's build queued behind them. The
# environment variable is the holder's mark; it is inherited by every
# descendant and by nothing else.
if [ "${MOJOLEARN_BUILD_LOCK_HELD:-}" = "1" ]; then
    exec "$@"
fi
MOJOLEARN_BUILD_LOCK_HELD=1
export MOJOLEARN_BUILD_LOCK_HELD
exec /usr/bin/python3 -c '
import fcntl, subprocess, sys
f = open("/tmp/cbsym-build.lock", "w")
fcntl.flock(f, fcntl.LOCK_EX)
sys.exit(subprocess.call(sys.argv[1:]))
' "$@"
