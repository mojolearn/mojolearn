#!/usr/bin/env python3
"""Compile frozen candidate matrices without executing their GPU harnesses.

Process and artifact orchestration only. Identical source/define/target jobs
are compiled once and their provenance is shared explicitly across idea arms.
Cross-compilation is neither device qualification nor a Linux executable build.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--ids', nargs='+', required=True)
    parser.add_argument('--vendors', nargs='+', choices=['nvidia', 'amd', 'apple'], default=['nvidia', 'amd'])
    parser.add_argument('--arms', nargs='+', help='Restrict to named matrix arms; omitted builds every arm')
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--plan-only', action='store_true', help='Write complete jobs/arm provenance without compiling')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[2]
    sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    if subprocess.check_output(['git', 'diff', '--name-only', 'HEAD'], cwd=repo, text=True).strip():
        parser.error('commit source before compiling a frozen matrix')
    args.evidence.mkdir(parents=True, exist_ok=True)
    jobs = {}
    arms = []
    unsupported = []
    for idea in args.ids:
        matrix = json.loads((repo/'experiments/performance_ideas'/idea/'compile_matrix.json').read_text())
        for vendor in args.vendors:
            if vendor not in matrix['supported_vendors']:
                unsupported.append(dict(id=idea, vendor=vendor, reason=matrix['unsupported_reason']))
                continue
            for arm in matrix['arms']:
                if args.arms and arm['name'] not in args.arms:
                    continue
                key = json.dumps([vendor, matrix['mode'], arm['source'], sorted(arm['defines'])])
                token = hashlib.sha256(key.encode()).hexdigest()[:16]
                arms.append(dict(id=idea, arm=arm['name'], vendor=vendor, job=token))
                jobs.setdefault(token, dict(job=token, vendor=vendor, mode=matrix['mode'], **arm))
    receipt = dict(schema=1, source_sha=sha, qualification='pending_device_validation',
                   executed_gpu=False, host_platform=sys.platform, expected_arms=len(arms),
                   expected_unique_jobs=len(jobs), arms=arms, unsupported=unsupported, builds=[])
    receipt['planned_jobs'] = list(jobs.values())
    path = args.evidence/'receipt.json'

    def save():
        temporary = path.with_suffix('.json.new')
        temporary.write_text(json.dumps(receipt, indent=2)+'\n')
        temporary.replace(path)

    save()
    if not jobs:
        parser.error('no admitted source arms match the selected IDs, targets and arm filter')
    if args.plan_only:
        print(f"planned_arms={len(arms)} unique_jobs={len(jobs)} unsupported={len(unsupported)} evidence={path}")
        return 0

    def build(job):
        output = args.evidence/job['job']
        command = ['bash', str(Path.home()/'mojolearn-evidence/compile_slot.sh'),
                   sys.executable, str(repo/'gemm/experiments/native_build.py'),
                   str(repo/job['source']), '--vendor', job['vendor'], '--mode', job['mode'],
                   '--source-sha', sha, '--output', str(output)]
        for define in job['defines']:
            command += ['--define', define]
        log_path = output.with_suffix('.build.log')
        start = time.monotonic()
        with log_path.open('w') as log:
            rc = subprocess.run(command, cwd=repo, env=os.environ | {'MOJOLEARN_COMPILE_JOBS': '1'},
                                stdout=log, stderr=subprocess.STDOUT).returncode
        return dict(**job, source_sha=sha, build_argv=command, build_rc=rc,
                    elapsed_seconds=round(time.monotonic()-start, 3), log=str(log_path),
                    binary=str(output), qualification='build_failed' if rc else 'compile_passed_device_owed')

    with ThreadPoolExecutor(max_workers=max(1, min(4, args.jobs))) as pool:
        futures = [pool.submit(build, job) for job in jobs.values()]
        for future in as_completed(futures):
            result = future.result()
            receipt['builds'].append(result)
            save()
            print(f"build={result['job']} vendor={result['vendor']} rc={result['build_rc']} "
                  f"completed={len(receipt['builds'])}/{len(jobs)}", flush=True)
    failed = sum(build['build_rc'] != 0 for build in receipt['builds'])
    receipt['compile_status'] = 'failed' if failed else 'passed'
    save()
    print(f"arms={len(arms)} unique_jobs={len(jobs)} failed={failed} unsupported={len(unsupported)} evidence={path}")
    return int(failed != 0)


if __name__ == '__main__':
    sys.exit(main())
