#!/usr/bin/env python3
"""How many release bindings a build box compiles at once, and the RSS cap.

THE RULE (rented hosts are paid for whole, Andrew 2026-11-05; first
written by lane/build-parallelism, 2026-09-25). packaging/linux/build_sets.sh
runs MOJOLEARN_BUILD_JOBS build scripts at a time, each with WORKERS_PER_JOB
(two) compiler workers (MOJOLEARN_COMPILE_JOBS=2). On a Linux build box every
core works and only the memory the jobs measurably need lowers the count:

    jobs = min(ceil(usable_cores / WORKERS_PER_JOB),
               floor((available_GiB - RESERVE_GIB) / PER_JOB_GIB))
    cap  = ceil(PER_JOB_GIB * jobs) + MARGIN_GIB

The division by WORKERS_PER_JOB is not a halving: each job is two compiler
processes, so jobs x 2 workers = every usable core (an odd core count rounds
up, so no core idles). RESERVE_GIB is MARGIN_GIB plus the 2 GiB host floor
below which the NVIDIA and AMD guards stop a build. The record names the limit
that bound the count (BUILD_BOUND=cores|memory|ceiling|override).

History of the old rule, min(16, usable_cores // 2, floor(0.6 x GiB / 3))
(40d81348bb): the // 2 was the same two-workers-per-job conversion; the 16
had no measurement behind it, it mirrored the 1..16 MOJOLEARN_BUILD_JOBS
range build_sets.sh already accepted; the 0.6 memory fraction was a second
safety factor on top of PER_JOB_GIB's own 1.5 x headroom. A box with more
than 32 cores therefore idled. MAX_JOBS (128) is now only a sanity ceiling on
the value the shell scripts and the guards accept (256 cores), never reached
by a box the release rents. The laptop (darwin) keeps the old rule unchanged;
its compiles go through tools/compile_slot.sh at -j 1 anyway.

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

MOJOLEARN_BUILD_JOBS=N (1..MAX_JOBS) stays an explicit override: jobs is N and the
cap follows the same formula; it is not refused for the box's size, because
an override is the operator saying they know.

    python3 tools/build_sizing.py [--jobs auto|N] [--per-job-gib G] --shell
prints BUILD_JOBS=, BUILD_RSS_GIB=, BOX_CORES=, BOX_MEM_GIB=, BUILD_PER_JOB_GIB=,
BUILD_SIZING=, BUILD_BOUND= lines for `eval`; --json prints the same record as
JSON. Either way one `build sizing: ...` line on stderr says the chosen job
count and which limit bound it.
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
# The guards stop a build when host MemAvailable drops below 2 GiB
# (nvidia_serial_guard.py / amd_serial_guard.py); keep that plus the margin free.
GUARD_FLOOR_GIB = 2
RESERVE_GIB = MARGIN_GIB + GUARD_FLOOR_GIB
# One build job = two compiler workers (MOJOLEARN_COMPILE_JOBS=2 in
# release061_remote_build.sh and packaging/linux/build_sets.sh).
WORKERS_PER_JOB = 2
# Sanity ceiling only (the shell scripts and the guards accept up to it).
MAX_JOBS = 128
# The laptop's rule, unchanged (darwin): min(16, cores // 2, 0.6 x GiB / per job).
LAPTOP_MAX_JOBS = 16
LAPTOP_MEMORY_FRACTION = 0.6
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


def _inactive_file(stat_text, key):
    for line in (stat_text or '').splitlines():
        name, _, value = line.partition(' ')
        if name == key and value.strip().isdigit():
            return int(value)
    return 0


def cgroup_memory_headroom(root='/sys/fs/cgroup'):
    """Bytes a cgroup memory limit still allows (limit minus working set), or None.

    memory.current / usage_in_bytes count the page cache. Its inactive file
    pages are reclaimed before the cgroup OOMs, so they are subtracted, the
    same working-set rule docker stats and the kubelet use. Without this, an
    8-CPU/128 GiB build container that has read the source tree and the Pixi
    env looks short of memory, and the guard's available-memory floor can stop
    a healthy compile while its own file cache grows.
    """
    for limit_name, usage_name, stat_name, stat_key in (
            ('memory.max', 'memory.current', 'memory.stat', 'inactive_file'),
            ('memory/memory.limit_in_bytes', 'memory/memory.usage_in_bytes',
             'memory/memory.stat', 'total_inactive_file'),
            ('memory.limit_in_bytes', 'memory.usage_in_bytes', 'memory.stat', 'total_inactive_file')):
        limit = _read(Path(root) / limit_name)
        if limit is None:
            continue
        if limit == 'max' or int(limit) >= 2 ** 60:
            return None
        usage = int(_read(Path(root) / usage_name) or 0)
        reclaimable = min(usage, _inactive_file(_read(Path(root) / stat_name), stat_key))
        return max(0, int(limit) - (usage - reclaimable))
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


def _limits(cores, mem_gib, per_job_gib, platform):
    """(jobs_by_cores, jobs_by_memory, ceiling) under this platform's rule."""
    if platform == 'darwin':
        return (cores // 2, math.floor(LAPTOP_MEMORY_FRACTION * mem_gib / per_job_gib),
                LAPTOP_MAX_JOBS)
    by_cores = -(-cores // WORKERS_PER_JOB) if cores >= WORKERS_PER_JOB else 0
    by_memory = math.floor(max(0.0, mem_gib - RESERVE_GIB) / per_job_gib)
    return by_cores, by_memory, MAX_JOBS


def size_build(cores, available_bytes, per_job_gib=PER_JOB_GIB, override=None, platform=None):
    """The sizing record. Raises TooSmall (auto only) or ValueError (bad override)."""
    if per_job_gib <= 0:
        raise ValueError('per-job GiB must be positive')
    platform = sys.platform if platform is None else platform
    mem_gib = available_bytes / GIB
    by_cores, by_memory, ceiling = _limits(cores, mem_gib, per_job_gib, platform)
    if override is None:
        jobs = min(ceiling, by_cores, by_memory)
        if jobs < 1:
            if platform == 'darwin':
                need = per_job_gib / LAPTOP_MEMORY_FRACTION
            else:
                need = per_job_gib + RESERVE_GIB
            raise TooSmall(
                'build box too small for one release build job: %d usable core(s) (need 2) and '
                '%.1f GiB available (need %.1f GiB); rent a larger box '
                'or set MOJOLEARN_BUILD_JOBS explicitly' % (cores, mem_gib, need))
        source = 'auto'
        # Name the limit that bound the count; memory wins a tie with cores
        # only when it is strictly lower, so a cores-bound box says so.
        if jobs == by_memory and by_memory < by_cores:
            bound = 'memory'
        elif jobs == by_cores:
            bound = 'cores'
        else:
            bound = 'ceiling'
    else:
        if not (isinstance(override, int) and 1 <= override <= ceiling):
            raise ValueError('MOJOLEARN_BUILD_JOBS must be 1..%d' % ceiling)
        jobs, source, bound = override, 'override', 'override'
    cap = math.ceil(per_job_gib * jobs) + MARGIN_GIB
    return dict(jobs=jobs, rss_cap_gib=cap, cores=cores, mem_available_gib=round(mem_gib, 1),
                per_job_gib=per_job_gib, sizing=source, jobs_by_cores=by_cores,
                jobs_by_memory=by_memory, bound=bound, platform=platform)


def describe(record):
    """One line: the chosen job count and the limit that bound it."""
    return ('build sizing: jobs=%d bound=%s (cores=%d -> %d jobs at %d workers each; '
            'memory=%.1f GiB -> %d jobs at %.1f GiB each; platform=%s; rss_cap=%d GiB)'
            % (record['jobs'], record['bound'], record['cores'], record['jobs_by_cores'],
               WORKERS_PER_JOB, record['mem_available_gib'], record['jobs_by_memory'],
               record['per_job_gib'], record['platform'], record['rss_cap_gib']))


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
    print(describe(record), file=sys.stderr)
    if args.json:
        print(json.dumps(record, sort_keys=True))
    else:
        print('BUILD_JOBS=%d' % record['jobs'])
        print('BUILD_RSS_GIB=%d' % record['rss_cap_gib'])
        print('BOX_CORES=%d' % record['cores'])
        print('BOX_MEM_GIB=%s' % record['mem_available_gib'])
        print('BUILD_PER_JOB_GIB=%s' % record['per_job_gib'])
        print('BUILD_SIZING=%s' % record['sizing'])
        print('BUILD_BOUND=%s' % record['bound'])
    return 0


if __name__ == '__main__':
    sys.exit(main())
