#!/usr/bin/env python3
"""Separate L40S diagnostic calls, preserving raw logs and build receipts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path.cwd()
out = Path('/root/neural-requested-profile')
out.mkdir(exist_ok=True)
env = dict(os.environ, PATH='/root/.pixi/bin:' + os.environ['PATH'],
           MOJOLEARN_NUMERIC_MODE='identical', MOJOLEARN_TARGET_COLUMN='nvidia',
           MOJOLEARN_GPU_ARCHS='sm_89', MOJOLEARN_COMPILE_JOBS='2',
           MOJOLEARN_BENCH_INSTALLED='1', MOJOLEARN_TRANSFORMER_TIMING='1',
           PYTHONUNBUFFERED='1', OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2')
env.pop('PYTHONPATH', None)
phase = 'setup'
def stamp(state, **extra):
    (out/'status.json').write_text(json.dumps(dict(state=state, phase=phase,
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra), indent=2)+'\n')
def run(cmd, log, timeout=1800):
    stamp('running')
    with (out/log).open('w') as f:
        f.write('COMMAND '+json.dumps([str(x) for x in cmd])+'\n'); f.flush()
        subprocess.run([str(x) for x in cmd], cwd=root, env=env,
                       stdout=f, stderr=subprocess.STDOUT, check=True, timeout=timeout)
def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
try:
    if (out/'complete.json').exists():
        raise RuntimeError('Refusing to repeat completed diagnostics')
    (out/'source.patch').write_bytes(subprocess.check_output(['git','diff','--binary','HEAD']))
    run(['nvidia-smi','-q'], 'device.log')
    run(['pixi','install'], 'pixi.log')
    base = subprocess.check_output(['pixi','run','python3','-c','import sys;print(sys.executable)'], env=env, text=True).strip().splitlines()[-1]
    py = out/'venv/bin/python'
    run([base,'-m','venv',out/'venv'], 'venv.log')
    run([py,'-m','pip','install','mojolearn==0.8.31','numpy==2.5.2','scipy==1.18.0'], 'packages-install.log')
    run([py,'-m','pip','freeze'], 'packages.txt')
    site = Path(subprocess.check_output([py,'-c','import sysconfig;print(sysconfig.get_paths()["purelib"])'], text=True).strip())/'mojolearn'
    for src in (root/'python/mojolearn').rglob('*.py'):
        dst=site/src.relative_to(root/'python/mojolearn'); dst.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(src,dst)
    for cache in site.rglob('__pycache__'): shutil.rmtree(cache)
    manifest=json.loads((out/'saved/manifest.json').read_text())
    receipts={'candidate':manifest['candidate'],'baseline_artifacts':{},'diagnostic_artifacts':{},
        'calls_per_lane':10,'shape':'full','note':'Diagnostic timers synchronize and perturb timings; not performance race numbers.',
        'source_patch_sha256':digest(out/'source.patch')}
    target=site/'cuda/sm_89/identical'
    for module in ['transformer','mamba','byte_lm']:
        name='_mojolearn_'+module+'.so'; src=out/'saved/native/candidate'/name
        assert digest(src)==manifest['artifacts']['candidate'][name], name
        shutil.copy2(src,target/name); receipts['baseline_artifacts'][name]=digest(src)
    def profile(label, lane):
        global phase
        phase=label+'-'+lane
        run([py,root/'tools/neural_stage_timing.py','--lane',lane,'--calls','10','--json',out/(phase+'.json')],phase+'.log',900)
    for lane in ['transformer-forward','samba-train-step','lm-train-step']:
        profile('saved-binding',lane)
    phase='build-byte-lm-phase-timers'
    env['MOJOLEARN_BUILD_EXTRA_DEFINES']='-D MOJOLEARN_STEP_PHASE_TIMERS=1 -D MOJOLEARN_ATTN_PHASE_TIMERS=1'
    env['MOJOLEARN_BYTE_LM_OUTDIR']=str(out/'instrumented-native')
    run(['bash','bindings/build_byte_lm.sh'],phase+'.log',3600)
    byte=out/'instrumented-native/_mojolearn_byte_lm.so'; shutil.copy2(byte,target/byte.name)
    receipts['diagnostic_artifacts'][byte.name]={'sha256':digest(byte),'defines':env['MOJOLEARN_BUILD_EXTRA_DEFINES']}
    profile('phase-timers','lm-train-step')
    # Per-kernel attention ticks are compiled out in the saved transformer.
    # Enable their existing diagnostic define for the two block-based lanes.
    phase='build-transformer-phase-timers'
    env['MOJOLEARN_MOJO_BUILD_FLAGS']=env.pop('MOJOLEARN_BUILD_EXTRA_DEFINES')
    run(['bash','bindings/build_transformer.sh'],phase+'.log',3600)
    transformer=root/'python/mojolearn/identical/_mojolearn_transformer.so'
    shutil.copy2(transformer,target/transformer.name)
    shutil.copy2(transformer,out/'instrumented-native'/transformer.name)
    receipts['diagnostic_artifacts'][transformer.name]={'sha256':digest(transformer),'defines':env['MOJOLEARN_MOJO_BUILD_FLAGS']}
    for lane in ['transformer-forward','samba-train-step']:
        profile('phase-timers',lane)
    (out/'complete.json').write_text(json.dumps(receipts,indent=2)+'\n')
    phase='complete'; stamp('finished')
except Exception as exc:
    stamp('failed',error=str(exc)); raise
