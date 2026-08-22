"""The launch log: kernel names in enqueue order, for trace attribution.

WHY THIS EXISTS. Apple's stock 'Metal System Trace' template records the
Shader Timeline instrument Disabled, `xctrace` exposes no way to enable it
from the command line (the `--instrument 'GPU'` spelling records nothing
into `gpu-shader-profiler-interval` either -- measured 2026-08-22), and
MAX exposes no Metal debug-label API. So a trace knows every dispatch's
DURATION but not its NAME: `metal-gpu-intervals` labels rows by encoder,
and this port issues one unnamed encoder per launch.

What the trace cannot name, the host can: Metal's compute channel on this
device is a single in-order queue, so DEVICE EXECUTION ORDER IS HOST
ENQUEUE ORDER. Every `enqueue_function` site in the forest path calls
`log_launch("<site name>")` first; with `RF_LAUNCH_LOG=<path>` set, the
names land in that file one per line, and joining line `i` to the trace's
i-th Compute interval (sorted by start) attributes every nanosecond to a
kernel site. `ensemble/bench/profile_metal.py` performs the join and
REFUSES it unless the two counts match exactly -- a misaligned join
mislabels every row after the first slip, so an inexact one is worthless.

DISABLED IS THE SHIPPING STATE. One `getenv` per launch (~9.4k per
benchmark fit) when unset; the fits that produced the certified
2026-08-21 numbers carry the same call, so the price is inside every
measured number, not beside it. Enabled runs are TRACE SUBJECTS, not
measurements: an append-per-launch `open()` is a syscall on the enqueue
path and quiet_window would rightly refuse to certify one.
"""

from std.os import getenv


def log_launch(name: StringSlice) raises:
    """Append `name` to `$RF_LAUNCH_LOG` if set; free-ish when unset."""
    var path = String(getenv("RF_LAUNCH_LOG"))
    if path == "":
        return
    with open(path, "a") as fh:
        fh.write(String(name) + "\n")
