#!/usr/bin/env python3
"""Recover one known failed JSON report from frozen saved packets, never refit.

SOURCE UNIQUE_TAG [--spec]. --spec prints read-only pinned preflight JSON.
Execution is M3 serial CPU-only saved-output analysis; no native/GPU imports.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

from scoped_pca_fit import compare, record, sha, ROOT, CONTRACT, A_FLAGS, B_FLAGS

COMPILED='201fe7367461af236f4adf0331f367d8a02cd29d'
CAPTURE='c982d77989a42e89034082dcdc86cae6aaf6bc7d'
REPAIR='9f039067fb32428d0479f1b8c5c0a2af179b1d04'
CAPTURE_HELPER='feb09e9b615ed2b12c8aa4e96e4a15172a83503a25f20332a5489b89496a99b0'
INPUT='scoped-pca-fit-istella-q-v1-quality'
FILES={
 'A.npz':'f84e1464c82fb35a3f18e945f0c8abf491c507df53e90284e0a6a0bc68173dc6',
 'B.npz':'03f7e134c847e8b7a16f195d5bd3b6002f48833cd0ff3a6cbbb2a70933cfee43',
 'A.npz.json':'f7ab9d8a904ac0a5c0a560f472dd1e7ab262b81b1eb0dc08c272041143db68f4',
 'B.npz.json':'32fd9d08ab38b3ff66c8a3c25c6d3ed9e08979b7af5226cda9ddcd68afba4318',
 'oracle.npz':'ea366d5c091d5998a7f648c00222180739b7e81246780ec04d37682f66c2c06a',
 'report.json':'a16202281b80b3c55117e76ccd30b3e6c9f449e8a66c47d1db65c85cf5cc8458',
}
# Binary provenance is recorded by capture metadata; recovery never loads them.
BINARIES={'A':'1226b5c1fe2e61e9ecd03a9f7120889071aa035201f2faaecf6ea864514b55c4',
          'B':'37cba76b51ef8e8cee5b76f47a2a1d5393c539cf1efba7916b542ddeb3199d94'}
DATA='725e525180fdbb3f3714f40765668bdc11504214e59064202184f98b62eae6c2'


def partial_identity(text):
    # Exact failed serialization location is hash-bound above. Retain only the
    # fully serialized identity prefix; do not invent/parse incomplete metrics.
    marker='  "metrics": {'
    assert text.count(marker)==1
    prefix=text.split(marker,1)[0].rstrip()
    assert prefix.endswith(',')
    identity=json.loads(prefix[:-1]+'\n}')
    assert identity['source_sha']==COMPILED and identity['harness_source']==CAPTURE
    assert identity['helper_sha']==CAPTURE_HELPER and identity['contract']==CONTRACT
    assert identity['bound']==5e-6 and identity['error_regression_allowance']==0
    assert identity['data_sha']==DATA and identity['status']=='HOLD'
    m=identity['manifest']
    assert m['source_sha']==COMPILED and m['binding']=='estimators' and m['numeric_mode']=='fast'
    assert m['defines_A']==A_FLAGS and m['defines_B']==B_FLAGS and m['hashes']==BINARIES
    return identity


def spec(source,tag):
    required=[]
    prereqs=[]
    for name,digest in FILES.items():
        key=name.replace('.','-')
        required.append(key)
        prereqs.append(dict(id=key,kind='file',path='~/mq/out/'+INPUT+'/'+name,sha256=digest))
    return dict(version=1,tag=tag,harness_source=source,script='tools/scoped_pca_recover.py',
                args=[source,tag],artifacts=[],prerequisites=prereqs,
                cases=[dict(name='recover-saved-pca-HOLD-no-refit',requires=required,
                            source_files=['tools/scoped_pca_fit.py'])])


def recover(source,tag):
    assert subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip()=='Apple M3 Ultra'
    os.chdir(ROOT)
    assert subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()==source
    subprocess.run(['git','diff','--quiet','HEAD','--'],check=True)
    subprocess.run(['git','merge-base','--is-ancestor',REPAIR,source],check=True)
    # Preserve the repaired comparison implementation exactly; no new fit path.
    repaired=subprocess.check_output(['git','show',REPAIR+':tools/scoped_pca_fit.py'])
    assert sha(ROOT/'tools/scoped_pca_fit.py')==hashlib.sha256(repaired).hexdigest()
    original=subprocess.check_output(['git','show',CAPTURE+':tools/scoped_pca_fit.py'])
    assert hashlib.sha256(original).hexdigest()==CAPTURE_HELPER
    directory=Path.home()/'mq/out'/INPUT
    for name,digest in FILES.items():assert sha(directory/name)==digest,(name,'input changed')
    identity=partial_identity((directory/'report.json').read_text())
    metadata={arm:json.loads((directory/(arm+'.npz.json')).read_text()) for arm in 'AB'}
    for arm,mask,index in [('A',0,6),('B',52,7)]:
        m=metadata[arm]
        counts=[0]*9;counts[index]=1
        assert m['binary_sha']==BINARIES[arm] and m['mask']==mask and m['counts']==counts
        assert m['shape']==[2043304,220] and m['scored'] is False and m['elapsed_ms'] is None
        assert m['input_preserved'] is True
        assert m['metadata'][:13]==[2,int(arm=='B'),220,220,2043304,40,51104,1,1,220,220,1,1]
    out=Path.home()/'mq/out'/(tag+'-recovery')
    out.mkdir(parents=True,exist_ok=False)
    # Uses saved small fitted outputs + 220x220 FP64 oracle only. It neither
    # opens the full data block nor runs models, native modules or any GPU job.
    ok,rows=compare(directory)
    assert not ok,'unexpected PASS on frozen known-HOLD evidence; manual review required'
    for name,digest in FILES.items():assert sha(directory/name)==digest
    report=dict(identity,metrics=rows,packets={k:v for k,v in FILES.items() if k!='report.json'},
                A=metadata['A'],B=metadata['B'],scored=False,board_admitted=False,
                timing_admitted=False,recovered=True,recovery_source_sha=source,
                recovery_helper_sha=sha(__file__),recovery_inputs=FILES,
                recovery_input_directory=str(directory),partial_report_preserved=True,
                numerical_capture_replayed=False,full_data_oracle_rebuilt=False)
    record(out/'report.json',report)
    failures=[k for k,v in rows.items() if not v['pass_no_worse']]
    bounds=[k for k,v in rows.items() if k.endswith('_relative') and k!='reconstruction_relative'
            and (v['A']>5e-6 or v['B']>5e-6)]
    print(json.dumps(dict(status='HOLD',report=str(out/'report.json'),regressions=failures,
                          bound_failures=bounds,replayed=False)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('source');p.add_argument('tag');p.add_argument('--spec',action='store_true')
    a=p.parse_args()
    assert re.fullmatch('[0-9a-f]{40}',a.source) and re.fullmatch('[A-Za-z0-9_.-]+',a.tag)
    if a.spec:print(json.dumps(spec(a.source,a.tag),indent=2))
    else:recover(a.source,a.tag)


if __name__=='__main__':main()
