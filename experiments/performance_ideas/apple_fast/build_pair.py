#!/usr/bin/env python3
"""Frozen M2-only compilation of complete attested FAST caller prerequisites.

Run under lq M2. Never launches kernels/imports a binding. Each compile uses the
repo build script's binary validation with its kernel smoke disabled and the
machine compile-slot semaphore. Manifest attests exact source/mode/defines.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from support import ROOT


def flags(defines):
    return ' '.join('-D ' + token for token in defines)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--idea',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',default='default');a=p.parse_args()
    chip=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True)
    if 'Apple M2 Pro' not in chip:raise RuntimeError('build on existing cheap M2 queue only')
    subprocess.run(['git','diff','--quiet','HEAD','--'],cwd=ROOT,check=True)
    source=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    card=json.loads((ROOT/'experiments/performance_ideas'/a.idea/'manifest.json').read_text())
    if card['status'].startswith('blocked_'):raise RuntimeError(card['blocker'])
    a.output.mkdir(parents=True,exist_ok=False)
    binding=card['binding'];candidate=card.get('variants',{}).get(a.variant,card['candidate_defines'])
    env=dict(os.environ,MOJOLEARN_NUMERIC_MODE='fast',MOJOLEARN_COMPILE_JOBS='1',MOJOLEARN_SKIP_BUILD_GATE='1')
    slot=Path.home()/'mojolearn-evidence/compile_slot.sh'
    if not slot.is_file():raise RuntimeError('existing M2 compile-slot script missing: '+str(slot))
    dependencies={}
    def build(name,defines,destination):
        script='bindings/build.sh' if name=='core' else 'bindings/build_'+name+'.sh'
        local=dict(env,MOJOLEARN_MOJO_BUILD_FLAGS=flags(defines),MOJOLEARN_BUILD_EXTRA_DEFINES=flags(defines))
        if name=='byte_lm':local['MOJOLEARN_BYTE_LM_OUTDIR']=str(a.output/(destination.stem+'-native'))
        binary=ROOT/'python/mojolearn'/('_mojolearn.so' if name=='core' else '_mojolearn_'+name+'.so')
        if name=='byte_lm':binary=Path(local['MOJOLEARN_BYTE_LM_OUTDIR'])/'_mojolearn_byte_lm.so'
        with destination.with_suffix('.build.log').open('x') as stream:
            rc=subprocess.run(['bash',str(slot),'bash',script],cwd=ROOT,env=local,stdout=stream,stderr=subprocess.STDOUT).returncode
        if rc or not binary.is_file():raise RuntimeError('build '+name+' failed rc='+str(rc)+' log='+str(destination.with_suffix('.build.log')))
        shutil.copy2(binary,destination)
        return hashlib.sha256(destination.read_bytes()).hexdigest()
    (a.output/'dependencies').mkdir()
    for prerequisite in card.get('prerequisite_bindings',['core']):
        name='_mojolearn.so' if prerequisite=='core' else '_mojolearn_'+prerequisite+'.so'
        destination=a.output/'dependencies'/name
        digest=build(prerequisite,[],destination)
        dependencies[name]=dict(source_sha=source,numeric_mode='fast',vendor='apple',defines=[],sha256=digest)
    hashes={}
    for arm,defines in (('A',card['baseline_defines']),('B',candidate)):
        hashes[arm]=build(binding,defines,a.output/(arm+'.so'))
    manifest=dict(source_sha=source,binding=binding,numeric_mode='fast',vendor='apple',
        defines_A=flags(card['baseline_defines']),defines_B=flags(candidate),hashes=hashes,
        dependencies=dependencies,builder='existing M2 Pro',status='OK')
    (a.output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('APPLE_FAST_BUILD status=OK source='+source+' binding='+binding+' artifacts='+str(a.output))
if __name__=='__main__':main()
