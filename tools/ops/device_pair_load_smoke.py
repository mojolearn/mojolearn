#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Does every CPU training image load beside its GPU sibling in one process?

`device="cpu"` on a GPU installation runs the CPU identical image of a family
in the process that already holds the family's GPU image. This smoke loads
each pair of `mojolearn.host_surface.cpu_training_routes()` (GPU binding
name -> host binding basename) in its OWN child process, GPU image first,
then the host image, and prints one line per pair: OK, the Python error, or
the signal that killed the child (an abort at import is a Mojo type or a
named table published by both images).

    python tools/ops/device_pair_load_smoke.py            # every pair
    python tools/ops/device_pair_load_smoke.py _mojolearn_transformer _mojolearn_mamba

Run it on an INSTALLED GPU build (the box's start-up smoke), never on the
laptop. The pairs are independent units: one child per pair, as many at once
as the allocation has cores (each child imports the package and two images;
measured memory per child is what the box report states). Exit status 1 when
any pair fails. A testing tool: nothing here is product execution.
"""
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

CHILD = r"""
import sys
import mojolearn
from mojolearn import _backend
gpu, host = sys.argv[1], sys.argv[2]
if _backend._CPU_ONLY is not None:
    print("SKIP cpu-only installation"); raise SystemExit(0)
_backend.binding(gpu, "identical")
print("gpu image loaded", flush=True)
module = _backend.load_host_module(host)
print("OK", module.__name__)
"""


def cores():
    try:
        return max(1, len(os.sched_getaffinity(0)))  # the allocation, not the host's visible CPUs
    except AttributeError:
        return max(1, os.cpu_count() or 1)


def one(pair):
    gpu, host = pair
    done = subprocess.run([sys.executable, "-c", CHILD, gpu, host], capture_output=True, text=True)
    last = (done.stdout.strip().splitlines() or [""])[-1]
    if done.returncode == 0 and last.startswith(("OK", "SKIP")):
        return gpu, host, True, last
    reason = (done.stderr.strip().splitlines() or [last or "no output"])[-1]
    if done.returncode < 0:
        reason = "killed by signal %d after: %s | %s" % (-done.returncode, last or "nothing", reason)
    return gpu, host, False, reason[:400]


def main(argv):
    from mojolearn import host_surface
    routes = host_surface.cpu_training_routes()
    pairs = sorted((gpu, host) for gpu, host in routes.items() if not argv or gpu in argv)
    workers = min(len(pairs), cores()) or 1
    print("pairs: %d, children at once: %d (cores of the allocation: %d)" % (len(pairs), workers, cores()))
    failed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:  # one independent child per pair, across every core
        for gpu, host, ok, text in pool.map(one, pairs):
            failed += not ok
            print("%s\t%s\t%s\t%s" % ("PASS" if ok else "FAIL", gpu, host, text))
    print("device pair load smoke: %d of %d pairs load together" % (len(pairs) - failed, len(pairs)))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
