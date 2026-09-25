#!/usr/bin/env python3
"""How many release bindings a build box compiles at once, and the RSS cap.

THE RULE (lane/build-parallelism, 2026-09-25). packaging/linux/build_sets.sh
runs MOJOLEARN_BUILD_JOBS build scripts at a time, each with two compiler
workers (MOJOLEARN_COMPILE_JOBS=2, one for gfx942). Until now every GPU leg
ran four at a time under a fixed 12 GiB process-group cap, whatever the box.
Now the box decides:

    jobs = min(16, usable_cores // 2, floor(0.6 * available_GiB / PER_JOB_GIB))
    cap  = ceil(PER_JOB_GIB * jobs) + MARGIN_GIB

usable_cores is the process's CPU affinity, lowered to a cgroup cpu.max quota
when one is set (a RunPod container sees the host's cores in its affinity).
available_GiB is MemAvailable, lowered to the cgroup memory.max headroom when
one is set (a container sees the host's /proc/meminfo). A box with fewer than
two usable cores, or too little memory for one job, is REFUSED by name: the
leg fails before it compiles instead of meeting the RSS guard or the OOM
killer half way through.

PER_JOB_GIB is measured, not guessed. The AMD guard records the build's
process-group peak RSS (telemetry.peak_rss_bytes in full46-build.log):
0.8.16, 0.8.18 and 0.8.19 gfx942 legs at four jobs peaked at 5.75, 5.92 and
6.65 GiB (1.44 to 1.66 GiB per job); the one-at-a-time 0.7.0 to 0.8.4 legs
peaked at 1.87 to 1.97 GiB for a single job. The NVIDIA guard now records the
same telemetry: a RunPod L40S (sm_89, 13 usable cores, 173 GiB cgroup
headroom) sized itself to 6 jobs and peaked at 8.20 GiB (1.37 GiB per job)
under a 20 GiB cap; a 27-core L40S sized itself to 13 jobs and peaked at
16.0 GiB (1.23 GiB per job) under 41. 3 GiB per job is 1.5 x the largest single-job peak.

MOJOLEARN_BUILD_JOBS=N (1..16) stays an explicit override: jobs is N and the
cap follows the same formula; it is not refused for the box's size, because
an override is the operator saying they know.

    python3 tools/build_sizing.py [--jobs auto|N] [--per-job-gib G] --shell
prints BUILD_JOBS=, BUILD_RSS_GIB=, BOX_CORES=, BOX_MEM_GIB=, BUILD_PER_JOB_GIB=,
BUILD_SIZING= lines for `eval`; --json prints the same record as JSON.
Exit 3 is a refusal (message on stderr), exit 2 a usage error.
"""
import argparse
import json
import math
import os
from pathlib import Path
import sys

PER_JOB_GIB = 3.0
MARGIN_GIB = 2
MAX_JOBS = 16
MEMORY_FRACTION = 0.6
GIB = 2 ** 30


class TooSmall(RuntimeError):
    """The box cannot run even one build job within the rule."""


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def cgroup_cpu_limit(root='/sys/fs/cgroup'):
    """CPUs allowed by a cgroup v2 cpu.max or v1 cfs quota, or None."""
    text = _read(Path(root) / 'cpu.max')
    if text:
        quota, _, period = text.partition(' ')
        if quota != 'max' and period:
            return max(1, int(int(quota) // int(period)))
        return None
    quota = _read(Path(root) / 'cpu' / 'cpu.cfs_quota_us') or _read(Path(root) / 'cpu.cfs_quota_us')
    period = _read(Path(root) / 'cpu' / 'cpu.cfs_period_us') or _read(Path(root) / 'cpu.cfs_period_us')
    if quota and period and int(quota) > 0:
        return max(1, int(quota) // int(period))
    return None


def cgroup_memory_headroom(root='/sys/fs/cgroup'):
    """Bytes a cgroup memory limit still allows (limit minus usage), or None."""
    for limit_name, usage_name in (('memory.max', 'memory.current'),
                                   ('memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes'),
                                   ('memory.limit_in_bytes', 'memory.usage_in_bytes')):
        limit = _read(Path(root) / limit_name)
        if limit is None:
            continue
        if limit == 'max' or int(limit) >= 2 ** 60:
            return None
        usage = int(_read(Path(root) / usage_name) or 0)
        return max(0, int(limit) - usage)
    return None


def mem_available(meminfo='/proc/meminfo'):
    for line in Path(meminfo).read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) * 1024
    raise RuntimeError('MemAvailable missing from ' + meminfo)


def box_resources(meminfo='/proc/meminfo', cgroup_root='/sys/fs/cgroup'):
    """(usable_cores, available_bytes) for this process on this box."""
    cores = len(os.sched_getaffinity(0))
    quota = cgroup_cpu_limit(cgroup_root)
    if quota is not None:
        cores = min(cores, quota)
    available = mem_available(meminfo)
    headroom = cgroup_memory_headroom(cgroup_root)
    if headroom is not None:
        available = min(available, headroom)
    return cores, available


def size_build(cores, available_bytes, per_job_gib=PER_JOB_GIB, override=None):
    """The sizing record. Raises TooSmall (auto only) or ValueError (bad override)."""
    if per_job_gib <= 0:
        raise ValueError('per-job GiB must be positive')
    mem_gib = available_bytes / GIB
    by_cores = cores // 2
    by_memory = math.floor(MEMORY_FRACTION * mem_gib / per_job_gib)
    if override is None:
        jobs = min(MAX_JOBS, by_cores, by_memory)
        if jobs < 1:
            raise TooSmall(
                'build box too small for one release build job: %d usable core(s) (need 2) and '
                '%.1f GiB available (need %.1f GiB = %.1f GiB per job / %.1f); rent a larger box '
                'or set MOJOLEARN_BUILD_JOBS explicitly'
                % (cores, mem_gib, per_job_gib / MEMORY_FRACTION, per_job_gib, MEMORY_FRACTION))
        source = 'auto'
    else:
        if not (isinstance(override, int) and 1 <= override <= MAX_JOBS):
            raise ValueError('MOJOLEARN_BUILD_JOBS must be 1..%d' % MAX_JOBS)
        jobs, source = override, 'override'
    cap = math.ceil(per_job_gib * jobs) + MARGIN_GIB
    return dict(jobs=jobs, rss_cap_gib=cap, cores=cores, mem_available_gib=round(mem_gib, 1),
                per_job_gib=per_job_gib, sizing=source, jobs_by_cores=by_cores,
                jobs_by_memory=by_memory)


def parse_jobs(text):
    if text in (None, '', 'auto'):
        return None
    if not text.isdigit():
        raise ValueError('MOJOLEARN_BUILD_JOBS must be auto or 1..%d' % MAX_JOBS)
    return int(text)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--jobs', default=os.environ.get('MOJOLEARN_BUILD_JOBS', 'auto'))
    parser.add_argument('--per-job-gib', type=float, default=PER_JOB_GIB)
    parser.add_argument('--meminfo', default='/proc/meminfo')
    parser.add_argument('--cgroup-root', default='/sys/fs/cgroup')
    how = parser.add_mutually_exclusive_group(required=True)
    how.add_argument('--shell', action='store_true')
    how.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        override = parse_jobs(args.jobs)
        cores, available = box_resources(args.meminfo, args.cgroup_root)
        record = size_build(cores, available, args.per_job_gib, override)
    except TooSmall as exc:
        print('REFUSED: ' + str(exc), file=sys.stderr)
        return 3
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(record, sort_keys=True))
    else:
        print('BUILD_JOBS=%d' % record['jobs'])
        print('BUILD_RSS_GIB=%d' % record['rss_cap_gib'])
        print('BOX_CORES=%d' % record['cores'])
        print('BOX_MEM_GIB=%s' % record['mem_available_gib'])
        print('BUILD_PER_JOB_GIB=%s' % record['per_job_gib'])
        print('BUILD_SIZING=%s' % record['sizing'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
