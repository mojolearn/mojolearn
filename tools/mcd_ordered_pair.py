#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Manager-owned M3 quality only: SOURCE TAG CASE [DIRECT_PASS_TAG].
Cases direct,mcd-istella,ee-istella,mcd-taxi,ee-taxi,mcd-synthetic.
Non-direct cases require a hash/source-matched direct PASS first. No timing,
fit clock, builds, SSH, queue mutation, or opponent evaluation. B repetition
is an explicitly unscored reproducibility test, not another scored arm.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

FIXTURE='mcd-ordered-cov-v1'
CASES=('direct','mcd-istella','ee-istella','mcd-taxi','ee-taxi','mcd-synthetic')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def logged(command,path,env):
    with path.open('x') as stream:
        r=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
    if r.returncode:
        print('\n'.join(path.read_text(errors='replace').splitlines()[-12:]))
        raise RuntimeError('quality command failed or HOLD: '+str(path))


def main():
    source,tag,case,*rest=sys.argv[1:]
    assert re.fullmatch('[0-9a-f]{40}',source) and re.fullmatch('[A-Za-z0-9_.-]+',tag)
    assert case in CASES and len(rest)==(0 if case=='direct' else 1)
    root=Path(__file__).resolve().parents[1]
    os.chdir(root)
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source
    subprocess.run(['git','diff','--quiet','HEAD','--','x_decomp/','bindings/','python/',
                    'checks/','gemm/','tools/mcd_ordered_pair.py','tools/mcd_ordered_quality.py',
                    'tools/mcd_ordered_oracle.py','tools/fast_quality_rule.py'],check=True)
    arms=Path.home()/'mq/verified-arms'/source/'x_decomp'
    manifest=json.loads((arms/'manifest.json').read_text())
    assert manifest['source_sha']==source and manifest['binding']=='x_decomp'
    assert manifest['numeric_mode']=='fast' and manifest['defines_A']==''
    assert manifest['defines_B'].split()==['-D','MOJOLEARN_MCD_ORDERED_COV']
    hashes={a:digest(arms/(a+'.so')) for a in ('A','B')}
    assert hashes==manifest['hashes']
    if rest:
        assert re.fullmatch('[A-Za-z0-9_.-]+',rest[0])
        prior=json.loads((Path.home()/'mq/out'/(rest[0]+'-quality')/'PASS.json').read_text())
        assert prior['source_sha']==source and prior['hashes']==hashes
        assert prior['case']=='direct' and prior['fixture']==FIXTURE and prior['status']=='PASS'
    out=Path.home()/'mq/out'/(tag+'-quality')
    out.mkdir(parents=True,exist_ok=False)
    so=root/'python/mojolearn/_mojolearn_x_decomp.so'
    original=out/'original.so'
    had_original=so.exists()
    if had_original:
        shutil.copy2(so,original)
    env=dict(os.environ,MOJOLEARN_NUMERIC_MODE='fast',MOJOLEARN_VENDOR='apple',
             MOJOLEARN_BENCH_INSTALLED='0',PYTHONPATH=str(root/'python'),
             OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    try:
        for arm in ('A','B'):
            temp=so.with_suffix('.so.next')
            shutil.copy2(arms/(arm+'.so'),temp)
            os.replace(temp,so)
            logged([sys.executable,'tools/mcd_ordered_quality.py','dump',source,arm,case,str(out)],out/(arm+'.log'),env)
            metadata=json.loads((out/(arm+'.json')).read_text())
            assert metadata['binding_sha256']==hashes[arm] and metadata['source']==source
            assert metadata['arm']==arm and metadata['case']==case and metadata['fixture']==FIXTURE
            print('MCD-ORDERED-ARM '+json.dumps(metadata,sort_keys=True))
        logged([sys.executable,'tools/mcd_ordered_quality.py','compare',case,str(out)],out/'compare.log',env)
        report=json.loads((out/'report.json').read_text())
        assert report['status']=='PASS'  # repeat_identical is info only (FAST rule)
        receipt=dict(source_sha=source,hashes=hashes,case=case,fixture=FIXTURE,status='PASS',
                     report_sha256=digest(out/'report.json'),timing=False,
                     execution_policy='unscored quality; B repeatability test only')
        with (out/'PASS.json').open('x') as f:
            json.dump(receipt,f,sort_keys=True)
        print('MCD-ORDERED-PAIR '+json.dumps(receipt,sort_keys=True))
    finally:
        if had_original:
            temp=so.with_suffix('.so.restore')
            shutil.copy2(original,temp)
            os.replace(temp,so)
        else:
            so.unlink(missing_ok=True)


if __name__=='__main__':
    main()
