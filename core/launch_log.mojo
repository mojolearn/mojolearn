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

DISABLED IS THE SHIPPING STATE. DEVIATION 1895: the env read happens
ONCE per process, not once per launch. This used to call
`String(getenv("RF_LAUNCH_LOG"))` on EVERY launch (~9.4k per benchmark
fit) -- a libc environ walk plus a `String` construction on the enqueue
path -- so the disabled path was priced into every measured number. The
path is now read on first use into a process global (the stdlib's
`_Global`, the same mechanism `std.python` keeps its interpreter handle
in) and every later call is one pointer load and one Bool test. The
variable is therefore LATCHED at first launch: flipping RF_LAUNCH_LOG
mid-process no longer changes behavior, which no harness does -- the
profile scripts export it before the run. Enabled runs are TRACE
SUBJECTS, not measurements: an append-per-launch `open()` is a syscall
on the enqueue path and quiet_window would rightly refuse to certify
one.
"""

from std.ffi import _Global
from std.os import getenv


struct _LaunchLogState(Defaultable, Movable):
    """The once-read `RF_LAUNCH_LOG` value (DEVIATION 1895)."""

    var path: String
    var enabled: Bool

    def __init__(out self):
        self.path = String(getenv("RF_LAUNCH_LOG"))
        self.enabled = self.path != ""


comptime _LAUNCH_LOG = _Global[
    StorageType=_LaunchLogState,
    name="RFLaunchLog",
    init_fn=_LaunchLogState.__init__,
]


def log_launch(name: StringSlice) raises:
    """Append `name` to `$RF_LAUNCH_LOG` if it was set at first use;
    one pointer load and one Bool test when it was not."""
    var st = _LAUNCH_LOG.get_or_create_ptr()
    if not st[].enabled:
        return
    with open(st[].path, "a") as fh:
        fh.write(String(name) + "\n")
