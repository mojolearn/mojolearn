#!/usr/bin/env python3
"""Measured CUDA runtime configuration witness; never grants qualification.

Runs on the pod, in the batch venv, while a full collector is live. It reads
the installed core source file, the CUDA driver API version, the one visible
device's driver record and the CUDA libraries mapped by each live collector.
tools/nvidia_baseline_gpu_batch.py stages this file and its body runs it
during the native and the PTX full collections; tools/admit_nvidia_ptx.py binds
the record to its receipt by this file's SHA256. Linux only.
"""
import argparse
import ctypes
import datetime
import hashlib
import json
import os
import pathlib
import subprocess
import sys
import sysconfig
import time

SCHEMA = 'mojolearn.cuda-runtime-config-witness.v1'
LIBRARIES = ('libcuda', 'libnvidia-ptxjitcompiler', 'libnvidia-nvvm', 'libnvJitLink')


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def libraries(pid, proc=pathlib.Path('/proc')):
    rows = []
    for line in (proc / str(pid) / 'maps').read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) != 6:
            continue
        path = fields[-1]
        base = pathlib.Path(path).name
        if not path.startswith('/') or not base.startswith(LIBRARIES):
            continue
        if path not in [r['path'] for r in rows]:
            rows.append(dict(path=path, sha256=sha(path)))
    return rows


def live_collectors(proc=pathlib.Path('/proc')):
    collectors = []
    for entry in proc.iterdir():
        if not entry.name.isdigit():
            continue
        try:
            argv = (entry / 'cmdline').read_bytes().decode().split('\0')
            if not any(x.endswith('/nvidia_baseline_qualification.py') for x in argv) or 'collect' not in argv:
                continue
            collectors.append(dict(pid=int(entry.name), argv=argv, libraries=libraries(entry.name, proc)))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
    return collectors


def ready(collectors, required):
    """True when a collector naming `required` has the CUDA driver mapped."""
    return any(required in ' '.join(row['argv'])
               and any(pathlib.Path(lib['path']).name.startswith('libcuda') for lib in row['libraries'])
               for row in collectors)


def wait_for(required, seconds, poll=live_collectors, sleep=time.sleep, clock=time.monotonic):
    end = clock() + seconds
    while True:
        collectors = poll()
        if required is None or ready(collectors, required):
            return collectors
        if clock() >= end:
            raise SystemExit('No live collector with a mapped CUDA driver for: ' + required)
        sleep(5)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--require-collector', help='text the live collector argv must contain, e.g. its --out path')
    p.add_argument('--wait', type=int, default=0, help='seconds to wait for that collector to map the CUDA driver')
    a = p.parse_args()
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
    out = pathlib.Path(a.out)
    if out.exists():
        raise SystemExit('Refuse existing witness')
    core = pathlib.Path(sysconfig.get_paths()['purelib']) / 'mojolearn/identity_columns/COMMIT'
    assert core.read_text().strip() == a.source
    lib = ctypes.CDLL('libcuda.so.1')
    version = ctypes.c_int()
    rc = lib.cuDriverGetVersion(ctypes.byref(version))
    assert rc == 0
    smi = subprocess.run(['nvidia-smi', '--query-gpu=uuid,name,compute_cap,driver_version', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True, timeout=15).stdout.strip()
    uuid, name, cap, driver = [x.strip() for x in smi.split(',')]
    collectors = wait_for(a.require_collector, a.wait)
    report = dict(schema=SCHEMA, source_commit=a.source,
                  source_commit_file_sha256=sha(core), script_sha256=sha(__file__),
                  timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  python=sys.executable,
                  device=dict(uuid=uuid, name=name, compute_capability=cap, driver_version=driver),
                  cuDriverGetVersion=version.value, cuDriverGetVersion_return=rc,
                  metadata_process_libraries=libraries('self'),
                  live_collectors=collectors, qualification=False)
    if out.exists():
        raise SystemExit('Refuse existing witness')
    out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(out=str(out), device=report['device'], cuDriverGetVersion=version.value,
                          live_collectors=len(collectors))))


if __name__ == '__main__':
    main()
