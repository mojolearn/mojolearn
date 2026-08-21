#!/usr/bin/env python3
"""Refuse a wheel whose extension contains no compiled GPU kernels.

THIS CHECK EXISTS BECAUSE A GREEN BUILD SHIPPED A BROKEN WHEEL.
mojolearn 0.1.0a2 was built on a GitHub-hosted macOS runner, passed every gate
in the release workflow -- portable-baseline build, ISA disassembly, minos/tag
agreement, twine metadata, install and import on python 3.10 through 3.14 --
was published to TestPyPI, installed cleanly from the index on a real M4, and
then failed on the first fit with

    Failed to create Metal function: core_row_norms_row_norm_kernel...

The function was not in the binary. Compare the same commit built two ways:

                                 local (M4)     hosted runner
    __TEXT/__const               692,543 B          49,039 B
    AIR / metallib markers              98                 0
    kernel-like symbols                 21                 7

The hosted runner compiles the HOST half and silently emits no Metal shader
code, because it has no GPU it recognizes. `mojo build` exits 0. Nothing
downstream noticed, and nothing could: every other gate examines the host
binary or the package metadata, and the wheel imports perfectly because
importing does not touch the device.

WHY THIS AND NOT "RUN A FIT". Running a fit is the better check and is not
available where it is needed: the machine that can be tricked into producing
this wheel is precisely the machine that cannot run one. This check is static,
so it works on the runner that has the problem.

WHAT IT DOES NOT PROVE. That the kernels are correct, or complete, or that any
particular one is present. It proves only that GPU code was emitted at all,
which is the difference between the two builds above. Correctness is
`packaging/macos/verify_wheel.sh` with no flag, on real Apple silicon.
"""

import subprocess
import sys

# Below this, the extension is a host-only build. The observed values are 98
# (working) and 0 (broken), so the threshold is not finely tuned and does not
# need to be; it separates "some GPU code" from "none".
MIN_MARKERS = 10


def markers(path):
    out = subprocess.run(
        ["strings", "-a", path], capture_output=True, text=True
    )
    if out.returncode != 0:
        raise SystemExit(f"strings failed on {path}")
    n = 0
    for line in out.stdout.splitlines():
        if "air.main" in line or "metallib" in line or "AIR" in line:
            n += 1
    return n


def check(path):
    n = markers(path)
    if n < MIN_MARKERS:
        print(
            f"FAIL {path}: {n} AIR/metallib markers (need >= {MIN_MARKERS}).\n"
            "     This extension contains NO compiled Metal kernels. It will\n"
            "     import fine and fail on the first fit with 'Failed to create\n"
            "     Metal function'. Build on a machine with a real Apple GPU.",
            file=sys.stderr,
        )
        return 1
    print(f"  {path}: GPU kernels embedded, {n} AIR/metallib markers")
    return 0


def main(paths):
    # EVERY EXTENSION IN THE WHEEL, not the first one. The failure this file
    # exists to catch is a whole build emitting no Metal at all, which would
    # hit every extension at once -- but a wheel now carries more than one,
    # and checking a subset is how a gate stops covering what it names.
    return max(check(p) for p in paths)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage: check_gpu_embedded.py <extension.so> [<extension.so> ...]")
        raise SystemExit(2)
    raise SystemExit(main(sys.argv[1:]))
