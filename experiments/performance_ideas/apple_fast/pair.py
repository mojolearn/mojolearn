#!/usr/bin/env python3
"""Run a real-caller fixture in isolated packages from frozen verified A/B arms.

No builds, installs in the source tree, opponent reraces, or implicit GPU route.
Existing M2 build manifests must attest both FAST arms and exact defines. Every
capture includes the loaded binding hash; a copied receipt cannot admit a run.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from support import ROOT, apple_fast
sys.path.insert(0, str(ROOT / 'tools'))
from fast_quality_rule import RULE, judge


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--idea', required=True)
    p.add_argument('--arms', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variant', default='default')
    args = p.parse_args()
    apple_fast(load_product=False)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    subprocess.run(['git', 'diff', '--quiet', 'HEAD', '--'], cwd=ROOT, check=True)
    card = json.loads((ROOT / 'experiments/performance_ideas' / args.idea / 'manifest.json').read_text())
    if card['status'].startswith('blocked_'):
        raise RuntimeError(card['blocker'])
    manifest = json.loads((args.arms / 'manifest.json').read_text())
    assert manifest['source_sha'] == source and manifest['numeric_mode'] == 'fast'
    defines = card.get('variants', {}).get(args.variant)
    if defines is None:
        assert args.variant == 'default'
        defines = card['candidate_defines']
    assert [token for token in manifest['defines_A'].split() if token != '-D'] == card.get('variant_baseline_defines', {}).get(args.variant, card['baseline_defines'])
    assert [token for token in manifest['defines_B'].split() if token != '-D'] == defines
    hashes = {arm: sha(args.arms / (arm + '.so')) for arm in ('A', 'B')}
    assert hashes == manifest['hashes']
    binding = manifest['binding']
    assert binding == card.get('variant_bindings', {}).get(args.variant, card['binding'])
    args.output.mkdir(parents=True, exist_ok=False)
    # Independent structural/gradient gates run before measured public calls.
    required=card.get('native_checks',[])
    assert set(manifest.get('native_checks',{}))=={check['name'] for check in required}
    for check in required:
        receipt=manifest['native_checks'][check['name']]
        binary=args.arms/check['name']
        assert receipt['source_sha']==source and receipt['numeric_mode']=='fast' and receipt['vendor']=='apple'
        assert receipt['defines']==defines and receipt['sha256']==sha(binary)
        with (args.output/(check['name']+'.log')).open('x') as stream:
            outcome=subprocess.run([str(binary.resolve())],cwd=ROOT,env=dict(os.environ,MOJOLEARN_NUMERIC_MODE='fast',MOJOLEARN_VENDOR='apple'),stdout=stream,stderr=subprocess.STDOUT)
        if outcome.returncode:raise RuntimeError('independent prerequisite failed rc='+str(outcome.returncode)+'; '+str(args.output/(check['name']+'.log')))
    records = {}
    for arm in ('A', 'B'):
        package = args.output / arm / 'package' / 'mojolearn'
        shutil.copytree(ROOT / 'python/mojolearn', package, ignore=shutil.ignore_patterns('*.so', '__pycache__'))
        # Only enumerated FAST prerequisites and the sole IDENTICAL core
        # input-transport helper may enter this isolated package.
        prerequisites = manifest['dependencies']
        expected=card.get('variant_prerequisite_bindings',{}).get(args.variant,card.get('prerequisite_bindings',['core']))
        names={'_mojolearn.so' if item=='core' else '_mojolearn_'+item+'.so' for item in expected}
        assert set(prerequisites)==names|{'identical/_mojolearn.so'}
        for name, receipt in prerequisites.items():
            path = args.arms / 'dependencies' / name
            assert receipt['source_sha'] == source and receipt['defines']==[]
            assert receipt['vendor'] == 'apple' and receipt['target_column']=='apple' and receipt['sha256'] == sha(path)
            if name=='identical/_mojolearn.so':
                assert receipt['numeric_mode']=='identical' and receipt['role']=='input_transport_helpers'
            else:
                assert receipt['numeric_mode']=='fast' and name in names
            destination=package/name
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
        shutil.copy2(args.arms / (arm + '.so'), package / ('_mojolearn.so' if binding == 'core' else '_mojolearn_' + binding + '.so'))
        env = dict(os.environ, PYTHONPATH=str(package.parent), MOJOLEARN_NUMERIC_MODE='fast',
                   MOJOLEARN_VENDOR='apple', OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
        output = args.output / (arm + '.json')
        command = [sys.executable, str(ROOT / 'experiments/performance_ideas' / args.idea / card.get('variant_callers', {}).get(args.variant, 'caller.py')),
                   '--arm', arm, '--variant', args.variant, '--output', str(output)]
        with (args.output / (arm + '.log')).open('x') as stream:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError('capture failed rc=' + str(result.returncode) + '; retained ' + str(args.output / (arm + '.log')))
        packet = json.loads(output.read_text())
        assert packet['source_sha'] == source and packet['arm'] == arm
        assert packet['binding']['sha256'] == hashes[arm]
        packet['prerequisite_hashes'] = prerequisites
        records[arm] = packet
    A, B = records['A'], records['B']
    assert set(A['cases']) == set(B['cases']) and A['cases']
    metrics = {}
    for case in A['cases']:
        left, right = A['cases'][case], B['cases'][case]
        assert left['contract'] == right['contract']
        for field in card.get('variant_semantic_equal_fields',{}).get(args.variant,card.get('semantic_equal_fields',[])):
            assert field in left and field in right, 'missing semantic field '+field
            assert left[field] == right[field], 'semantic contract changed: '+case+'/'+field
        for name, spec in left['metrics'].items():
            other = right['metrics'][name]
            assert spec['rtol'] == other['rtol'] and spec['atol'] == other['atol']
            metrics[case + '/' + name] = judge(spec['value'], other['value'], spec['rtol'], spec['atol'], spec.get('opponent'))
    ok = all(item['ok'] for item in metrics.values())
    result = dict(schema=1, id=args.idea, status='PASS' if ok else 'HOLD_quality', source_sha=source,
                  hashes=hashes, variants=args.variant, records=records, metrics=metrics, rule=RULE,
                  promotion_authorized=False, qualification='actual caller A/B; opponent admission and repeated-call speed review owed')
    (args.output / 'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False))
    print('APPLE_FAST_PAIR status=' + result['status'] + ' metrics=' + str(len(metrics)) + ' receipt=' + str(args.output / 'receipt.json'))
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
