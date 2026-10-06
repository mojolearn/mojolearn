#!/usr/bin/env python3
"""Build frozen Mojo profiles and execute their fixture/caller qualification.

Python performs process/file orchestration only. Matrix arithmetic, identity
and all device operations live in the committed Mojo harnesses. This driver
never claims resource or full-operation qualification from a microbenchmark.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--evidence', required=True, type=Path)
    parser.add_argument('--mojo', default='mojo')
    parser.add_argument('--vendor', choices=['nvidia','amd','apple'], required=True)
    parser.add_argument('--accelerator')
    parser.add_argument('--mojo-include', action='append', default=[])
    parser.add_argument('--build-only', action='store_true')
    parser.add_argument('--run-only', action='store_true')
    args = parser.parse_args()
    if args.build_only and args.run_only:
        parser.error('build-only and run-only are mutually exclusive')
    repo = Path(__file__).resolve().parents[2]
    cfg = json.loads(args.config.read_text())
    sha = subprocess.check_output(['git','rev-parse','HEAD'], cwd=repo, text=True).strip()
    if subprocess.check_output(['git','diff','--name-only','HEAD'],cwd=repo,text=True).strip():
        parser.error('freeze and commit source before qualification')
    args.evidence.mkdir(parents=True, exist_ok=True)
    receipts = []
    status = 0
    for arm in cfg['arms']:
        binary = args.evidence / (cfg['id']+'-'+arm['name'])
        argv = [args.mojo,'build','-j1','-D','MOJOLEARN_NUMERIC_IDENTICAL=1','-I',str(repo)]
        argv += ['-D','MOJOLEARN_COLUMN_'+args.vendor.upper()+'=1']
        accelerator = args.accelerator or {'nvidia':'sm_89','amd':'gfx942','apple':'apple-m2'}.get(args.vendor)
        if accelerator:
            argv += ['--target-accelerator',accelerator]
        for include in args.mojo_include:
            argv += ['-I',include]
        for define in arm.get('defines',[]):
            argv += ['-D',define]
        argv += [str(repo/cfg.get('source','gemm/experiments/profile_check.mojo')),'-o',str(binary)]
        slot = Path.home()/'mojolearn-evidence/compile_slot.sh'
        command = ['bash',str(slot),*argv]
        buildlog = binary.with_suffix('.build.log')
        env = os.environ | {'MOJOLEARN_COMPILE_JOBS':'1'}
        if args.run_only:
            previous = json.loads((args.evidence/'receipt.json').read_text())
            old_arm = next((entry for entry in previous['arms'] if entry['arm']==arm['name']),None)
            if previous['source_sha'] != sha or old_arm is None or old_arm['build_rc'] or not binary.is_file():
                parser.error('run-only requires a green binary receipt at this frozen source')
            rc = 0
        else:
            with buildlog.open('w') as log:
                rc = subprocess.run(command,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT).returncode
        receipt = dict(id=cfg['id'],arm=arm['name'],source_sha=sha,build_rc=rc,
                       build_argv=command,build_log=str(buildlog),runs=[],qualification='build_failed' if rc else 'build_passed_run_owed')
        receipts.append(receipt)
        if rc:
            status = 1
            continue
        if args.build_only:
            continue
        for index, fixture in enumerate(cfg.get('fixtures',[{}])):
            runenv = env | {key:str(value) for key,value in arm.get('environment',{}).items()} | {key:str(value) for key,value in fixture.items()}
            runlog = binary.parent/(binary.name+f'.fixture-{index}.log')
            with runlog.open('w') as log:
                result = subprocess.run([str(binary)],cwd=repo,env=runenv,stdout=log,stderr=subprocess.STDOUT)
            receipt['runs'].append(dict(fixture=fixture,exit_code=result.returncode,log=str(runlog)))
            if result.returncode:
                status = 1
        receipt['qualification'] = 'native_fixture_passed_full_caller_and_resources_owed' if not any(r['exit_code'] for r in receipt['runs']) else 'native_fixture_failed'
    (args.evidence/'receipt.json').write_text(json.dumps(dict(schema=1,id=cfg['id'],source_sha=sha,
       status='failed' if status else 'complete',full_operation_qualification=False,arms=receipts),indent=2)+'\n')
    print(f"{cfg['id']} arms={len(receipts)} build_failed={sum(r['build_rc'] != 0 for r in receipts)} exit_code={status} evidence={args.evidence}")
    return status

if __name__ == '__main__':
    sys.exit(main())
