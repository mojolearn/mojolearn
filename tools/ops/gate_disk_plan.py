#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE RELEASE GATE JOB'S DISK PLAN: what one NVIDIA/AMD gate job (tools/ops/release_gate_job.sh) writes on its box,
the free space the job refuses to start without, and the container disk a release pod is rented with.

0.8.66 run t (1x RTX 4090 RunPod pod, 80 GB overlay disk, 2026-11-08) filled the disk: the measurement smoke's work
dir reached 35 GB beside the runtime qualifier's venv (6.3 GB), the smoke's fixture venv (4.7 GB) and measured venv
(1 GB), the results dir (14 GB) and the out dir (14 GB); the smoke died, its final upload and the pod's lease writes
failed. The figures below are those measurements; the smoke's share is now its cap (tools/release_measurement_smoke.py
--max-work-gb), never what it happens to write.

    python3 tools/ops/gate_disk_plan.py job-gb      # free GB a gate job needs at its start (GATE_DISK_NEED_GB)
    python3 tools/ops/gate_disk_plan.py pod-gb      # container disk GB to rent for a release pod (dev_pod.sh DISK_GB)
    python3 tools/ops/gate_disk_plan.py show        # the table
"""
import math
import sys

#: GB one gate job writes, by item (0.8.66 run t, measured on the pod unless the note says otherwise)
JOB_WRITES_GB = {
    "runtime qualifier venv (out/runtime-<vendor>/qualify/run/venv)": 6.3,
    "measurement smoke fixture venv (CPU torch)": 4.7,
    "measurement smoke measured venv": 1.0,
    # not measured on its own: run t's listed items came to 75 GB of the 80 GB disk, so the identity venv, the
    # tree and the pixi environment together held at most 5 GB; the identity venv is counted at that bound
    "identity venv (bounded by run t's full disk)": 5.0,
    # the reports, measured with the qualifier venv still inside them (copy_reports excludes venvs since run t, so
    # this overstates; kept until a job's own DISK lines measure the new figure)
    "reports, work out dir": 14.0,
    "reports, results copy (persisted to R2)": 14.0,
}
#: the measurement smoke's footprint cap (tools/release_measurement_smoke.py MAX_WORK_GB; the job passes
#: GATE_MEASURE_SMOKE_MAX_GB when set)
SMOKE_CAP_GB = 8.0
#: what is on a pod before the job: the image's writable layer, the tree and its pixi environment (bounded with the
#: identity venv above by run t's full disk)
POD_BASE_GB = 5.0
#: headroom on the rented disk (Andrew's brief, 2026-11-08: the gate's known footprint + 20 GB)
HEADROOM_GB = 20.0
#: never rent less than the disk the pods had (tools/dev_pod.sh's former fixed DISK_GB)
POD_MIN_GB = 80


def job_need_gb(smoke_cap_gb=SMOKE_CAP_GB):
    """Whole GB a gate job writes: the measured items plus the smoke's cap."""
    return int(math.ceil(sum(JOB_WRITES_GB.values()) + smoke_cap_gb))


def pod_disk_gb(smoke_cap_gb=SMOKE_CAP_GB):
    """Container disk GB for a release pod: the job's need, the pod's base and the headroom, rounded up to 10."""
    need = job_need_gb(smoke_cap_gb) + POD_BASE_GB + HEADROOM_GB
    return max(POD_MIN_GB, int(math.ceil(need / 10.0)) * 10)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    cmd = argv[0] if argv else "show"
    if cmd == "job-gb":
        print(job_need_gb())
    elif cmd == "pod-gb":
        print(pod_disk_gb())
    elif cmd == "show":
        for k, v in JOB_WRITES_GB.items():
            print("%6.1f GB  %s" % (v, k))
        print("%6.1f GB  measurement smoke cap" % SMOKE_CAP_GB)
        print("%6d GB  job need (free at start)" % job_need_gb())
        print("%6d GB  pod disk (need + base %.0f + headroom %.0f, at least %d)" % (
            pod_disk_gb(), POD_BASE_GB, HEADROOM_GB, POD_MIN_GB))
    else:
        print("usage: gate_disk_plan.py job-gb|pod-gb|show", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
