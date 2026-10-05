#!/usr/bin/env python3
"""Adapt one medium-wave arm's retained DART outputs into a quality-noise receipt.

Retained files only: no model imports, fits, GPU work, opponents or timers.
The wave's build-products.json is a flat relative-path -> SHA256 inventory, but
tools/identical_quality_noise.py needs PASS native-build receipts that account
for every loaded binary. This adapter derives that accounting from records that
already exist and refuses on any gap:

  * prepare identity: wave.json == prepare.json identity (PASS), quality.json
    carries the same identity, and the plan/harness files match their hashes;
  * frozen source: HEAD == identity SHA, tracked tree clean, complete binary
    inventory == build-products.json;
  * imported modules: every native-receipt-import is re-validated with the
    harness's own identical_wave_reuse.validate_receipt (donor receipt,
    producer attestation, clean donor source, toolchain, dependency proofs);
    the re-validated receipt/attestation/artifact hashes must equal the ones
    recorded at prepare time, and each copied artifact is remapped into this
    wave by its exact relative path and hash;
  * fresh modules: each remaining plan builder has a prepare step rc 0, an
    rc.json returncode 0 for exactly 'bindings/<builder>', a retained log, and
    its artifact hash in build-products.json.

The derived receipt is labelled as an adapter record, not a fresh-build
receipt, and lists the evidence it was derived from. Nothing is fabricated:
any missing record is a refusal (exit 2), never a filled-in default.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError('dart wave receipt refused: ' + message)


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def array_sha(a):
    # Same definition as tools/classical_two_datasets.sha256_array.
    import numpy as np
    a = np.ascontiguousarray(a)
    h = hashlib.sha256()
    h.update(str(a.dtype).encode())
    h.update(str(a.shape).encode())
    h.update(a.data)
    return h.hexdigest()


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def module_relative(builder):
    match = re.fullmatch(r'build(?:_([a-z0-9_]+))?\.sh', builder)
    require(match is not None, 'invalid builder name ' + builder)
    binding = {'build.sh': 'base', 'build_core_host.sh': 'base_host'}.get(builder, match.group(1))
    module = {'base': '_mojolearn', 'base_host': '_mojolearn_core_host'}.get(binding, '_mojolearn_' + str(binding))
    folder = 'host' if binding.endswith('_host') else 'identical'
    return 'python/mojolearn/' + folder + '/' + module + '.so'


def validate_prepare(wave, harness):
    identity = json.loads((wave / 'wave.json').read_text())
    prepare = json.loads((wave / 'prepare.json').read_text())
    require(prepare.get('status') == 'PASS' and prepare.get('phase') == 'prepare', 'prepare receipt is not PASS')
    require(prepare.get('identity') == identity, 'prepare identity differs from wave.json')
    quality = json.loads((wave / 'quality.json').read_text())
    require(quality.get('identity') == identity, 'quality receipt belongs to a different identity')
    sha = identity.get('sha', '')
    require(re.fullmatch('[0-9a-f]{40}', sha) is not None, 'full source SHA required')
    pinned = json.loads((harness / 'harness-sha256.json').read_text())
    for name, wanted in pinned.items():
        require(digest(harness / name) == wanted, 'harness file changed: ' + name)
    require(digest(harness / 'identical_wave_plan.json') == identity['plan_sha256'], 'plan hash differs from wave identity')
    plan = json.loads((harness / 'identical_wave_plan.json').read_text())
    return identity, prepare, plan, pinned


def validate_source(source, sha, products):
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    require(head == sha, 'source HEAD differs from identity SHA')
    require(subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--']).returncode == 0,
            'tracked source changed after freeze')
    inventory = {str(p.relative_to(source)): digest(p) for p in (source / 'python/mojolearn').rglob('*.so')}
    require(products and inventory == products, 'complete binary inventory differs from build-products.json')


def account_modules(args, wave, identity, prepare, plan, source, products):
    sys.path.insert(0, str(args.harness))
    from identical_wave_reuse import validate_receipt
    arm, sha = args.arm, identity['sha']
    steps = prepare['arms'][arm]
    by_id = {}
    imports = []
    for step in steps:
        if step['id'] == 'native-receipt-import':
            imports.append(step)
        else:
            require(step['id'] not in by_id, 'duplicate prepare step ' + step['id'])
            by_id[step['id']] = step
    require(by_id.get('worktree', {}).get('rc') == 0, 'worktree step missing or failed')
    require(by_id.get('portable-math', {}).get('rc') == 0, 'portable-math step missing or failed')
    modules, donors, imported = {}, [], {}
    for step in imports:
        require(step.get('rc') == 0, 'native import step failed')
        recorded = step['provenance']
        att = json.loads(Path(recorded['attestation']).read_text())
        again = validate_receipt(recorded['receipt'], sha=sha, vendor=identity['vendor'], arch=identity['gpu_arch'],
                                 arm=arm, builders=sorted(recorded['builders']), environment=args.environment,
                                 python=att['python']['path'], pixi=att['pixi']['path'])
        for key in ('receipt', 'receipt_sha256', 'attestation', 'attestation_sha256', 'artifact_inventory_sha256'):
            require(again[key] == recorded[key], 'imported receipt re-validation differs: ' + key)
        require(again['helper'] == recorded['helper'], 'imported helper differs')
        require(set(again['builders']) == set(recorded['builders']), 'imported builder set differs')
        for builder, row in recorded['builders'].items():
            require(again['builders'][builder]['sha256'] == row['sha256'], 'imported artifact hash differs: ' + builder)
            require(row['relative'] == module_relative(builder), 'imported relative path differs: ' + builder)
            require(builder not in imported, 'builder imported twice: ' + builder)
            imported[builder] = row
        donors.append({'receipt': recorded['receipt'], 'receipt_sha256': again['receipt_sha256'],
                       'attestation': recorded['attestation'], 'attestation_sha256': again['attestation_sha256'],
                       'revalidated_with': 'identical_wave_reuse.validate_receipt'})
        helper = recorded['helper']
        require(products.get(helper['relative']) == helper['sha256'], 'helper not remapped by exact path/hash')
    if args.expect_donor:
        expected = dict(item.split('=', 1) for item in args.expect_donor)
        seen = {d['receipt']: d for d in donors}
        for receipt, wanted in expected.items():
            att_sha = None
            if ':' in wanted:
                wanted, att_sha = wanted.split(':', 1)
            require(receipt in seen and seen[receipt]['receipt_sha256'] == wanted == digest(receipt),
                    'expected donor receipt differs: ' + receipt)
            if att_sha:
                require(seen[receipt]['attestation_sha256'] == att_sha, 'expected donor attestation differs')
    for builder in plan['builders']:
        relative = module_relative(builder)
        step = by_id.get(builder)
        require(step is not None and step.get('rc') == 0, 'builder step missing or failed: ' + builder)
        require(relative in products, 'builder artifact missing from build-products: ' + builder)
        artifact = (source / relative).resolve()
        require(artifact.is_relative_to(source.resolve()) and digest(artifact) == products[relative],
                'artifact hash differs: ' + builder)
        if builder in imported:
            require(step.get('reused') is True and step.get('sha256') == imported[builder]['sha256']
                    == products[relative], 'imported artifact not remapped by exact path/hash: ' + builder)
            evidence = {'origin': 'imported', 'donor_artifact': imported[builder]['path']}
        else:
            require(not step.get('reused'), 'reused step without import provenance: ' + builder)
            rc_path = wave / arm / 'prepare' / (builder + '.rc.json')
            log_path = wave / arm / 'prepare' / (builder + '.log')
            rc = json.loads(rc_path.read_text())
            require(rc.get('returncode') == 0 and rc.get('argv', [])[-1:] == ['bindings/' + builder],
                    'fresh builder rc record missing or failed: ' + builder)
            require(log_path.is_file(), 'fresh builder log missing: ' + builder)
            evidence = {'origin': 'fresh_wave_build', 'rc_json': str(rc_path), 'rc_json_sha256': digest(rc_path),
                        'log': str(log_path), 'log_sha256': digest(log_path)}
        modules[builder] = dict(status='PASS', artifact=str(artifact), relative=relative,
                                sha256=products[relative], **evidence)
    extra = set(imported) - set(plan['builders'])
    require(not extra, 'imported builders outside the plan: ' + ','.join(sorted(extra)))
    return modules, donors


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--wave', type=Path, required=True)
    p.add_argument('--arm', choices=('on', 'off'), required=True)
    p.add_argument('--harness', type=Path, required=True, help='pinned wave harness dir with harness-sha256.json')
    p.add_argument('--environment', type=Path, required=True, help='toolchain checkout used by the wave prepare')
    p.add_argument('--targets', type=Path, required=True, help='data-only targets.npz from identical_dart_noise_prepare')
    p.add_argument('--expect-donor', action='append', default=[],
                   help='RECEIPT=SHA256[:ATTESTATION_SHA256]; verified, never trusted')
    p.add_argument('--out-dir', type=Path, required=True)
    a = p.parse_args(argv)
    if sys.platform != 'linux':
        p.error('authorized Linux statistics server only')
    a.wave, a.harness, a.environment = a.wave.resolve(), a.harness.resolve(), a.environment.resolve()
    a.out_dir.mkdir(parents=True, exist_ok=False)
    try:
        import numpy as np
        identity, prepare, plan, pinned = validate_prepare(a.wave, a.harness)
        sha = identity['sha']
        source = a.wave / a.arm / 'source'
        products = json.loads((a.wave / a.arm / 'build-products.json').read_text())
        validate_source(source, sha, products)
        modules, donors = account_modules(a, a.wave, identity, prepare, plan, source, products)
        report_path = a.wave / a.arm / 'quality' / 'dart_reference.json'
        report = json.loads(report_path.read_text())
        # The ON reference row FAILED the old one-row threshold; the run itself completed.
        require(report.get('status') in ('PASS', 'FAILED') and report.get('opponents_executed') == 0,
                'DART gate report incomplete or executed opponents')
        reference_path = source / 'tools/identical_wave_dart_reference.json'
        reference = {r['lane']: r for r in json.loads(reference_path.read_text())['rows']}
        checks = {c['lane']: c for c in report['checks']}
        require(set(checks) == {'dart', 'dart-reg'} == set(reference), 'DART rows incomplete')
        loaded, outputs, params = {}, {}, {}
        outputs_dir = (a.wave / a.arm / 'quality' / 'dart_reference-outputs').resolve()
        for lane, check in checks.items():
            row = reference[lane]
            require(check['dataset'] == row['dataset'], lane + ': dataset differs')
            ev = check['output_evidence']
            require(ev['input_arrays'] == row['input_arrays'], lane + ': fixture input hashes differ from reference')
            out = Path(ev['path']).resolve()
            require(out.parent == outputs_dir and digest(out) == ev['sha256'], lane + ': output artifact changed')
            with np.load(out, allow_pickle=False) as z:
                arrays = {k: np.ascontiguousarray(z[k]) for k in z.files}
            require(set(arrays) == set(ev['arrays']), lane + ': output array inventory differs')
            for name, meta in ev['arrays'].items():
                require(array_sha(arrays[name]) == meta['sha256'], lane + ': output array hash differs: ' + name)
            outputs[lane] = (out, arrays)
            prov = check['provenance']
            require(prov['vendor'] == {'amd': 'hip', 'nvidia': 'cuda'}[identity['vendor']], lane + ': vendor differs')
            require(Path(prov['package']).resolve().is_relative_to(source.resolve()), lane + ': package outside source')
            for path in prov['binding_files']:
                resolved = Path(path).resolve()
                require(resolved.is_relative_to(source.resolve()), 'binding outside source: ' + path)
                relative = str(resolved.relative_to(source.resolve()))
                require(products.get(relative) == digest(resolved), 'loaded binding not in build products: ' + path)
                require(loaded.setdefault(str(resolved), products[relative]) == products[relative], 'binding hash conflict')
            params[lane] = canonical_sha(row['params'])
        accounted = {m['artifact']: m['sha256'] for m in modules.values()}
        require(all(accounted.get(k) == v for k, v in loaded.items()), 'loaded binding not accounted by a builder')
        with np.load(a.targets, allow_pickle=False) as t:
            require(array_sha(t['classification_y']) == reference['dart']['input_arrays']['yq']['sha256'],
                    'classification targets misaligned with the DART fixture')
            require(array_sha(t['regression_y']) == reference['dart-reg']['input_arrays']['yq']['sha256'],
                    'regression targets misaligned with the DART fixture')
        clf, reg = outputs['dart'][1], outputs['dart-reg'][1]
        predictions = a.out_dir / 'predictions.npz'
        with predictions.open('xb') as f:
            np.savez(f, **{'clf-repeat0-prediction': clf['pred'], 'clf-repeat0-probability': clf['proba1'],
                           'reg-repeat0-prediction': reg['pred']})
        native = {'schema': 1, 'kind': 'identical_wave_adapter_native_receipt',
                  'not_a_fresh_build_receipt': True, 'status': 'PASS', 'sha': sha, 'arm': a.arm,
                  'vendor': identity['vendor'], 'arch': identity['gpu_arch'], 'mode': 'identical',
                  'derived_from': {name: {'path': str(path), 'sha256': digest(path)} for name, path in [
                      ('wave', a.wave / 'wave.json'), ('prepare', a.wave / 'prepare.json'),
                      ('quality', a.wave / 'quality.json'),
                      ('build_products', a.wave / a.arm / 'build-products.json'),
                      ('plan', a.harness / 'identical_wave_plan.json'),
                      ('harness_pins', a.harness / 'harness-sha256.json'),
                      ('adapter', Path(__file__).resolve())]},
                  'imported_donors': donors, 'modules': modules}
        native_path = a.out_dir / 'native-receipt-adapted.json'
        write_new(native_path, native)
        receipt = {'schema': 1, 'arm': a.arm, 'numeric_mode': 'identical', 'source_sha': sha,
                   'predictions_sha256': digest(predictions), 'targets_sha256': digest(a.targets),
                   'fixture_id': 'identical_wave_dart_reference/taxi/' + canonical_sha(
                       {lane: {'dataset': r['dataset'], 'input_arrays': r['input_arrays']} for lane, r in reference.items()}),
                   'parameter_sha256': {'classification': params['dart'], 'regression': params['dart-reg']},
                   'native_build_receipts': [{'path': str(native_path), 'sha256': digest(native_path)}],
                   'loaded_native_artifacts': loaded,
                   'numeric_mode_evidence': 'identical_wave_dart_gate.py asserts numeric_mode_used==identical before its fit; source tree clean at sha',
                   'prediction_sources': {lane: {'path': str(out), 'sha256': digest(out)} for lane, (out, _) in outputs.items()},
                   'gate_report': {'path': str(report_path), 'sha256': digest(report_path)},
                   'reference': {'path': str(reference_path), 'sha256': digest(reference_path)}}
        write_new(a.out_dir / 'receipt.json', receipt)
        status = 'PASS'
    except (ValueError, KeyError, OSError, TypeError, subprocess.SubprocessError) as exc:
        write_new(a.out_dir / 'adapter-refusal.json', {'status': 'REFUSED', 'reason': str(exc)})
        print('DART_WAVE_RECEIPT REFUSED', exc)
        return 2
    print('DART_WAVE_RECEIPT', status, a.arm, 'loaded', len(loaded), 'modules', len(modules), 'out', a.out_dir)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
