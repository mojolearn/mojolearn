#!/usr/bin/env python3
"""Compare both vendors' complete identity receipts before allowing timing."""
import argparse
import hashlib
import json
from pathlib import Path
import re

BACKENDS = {'nvidia': 'cuda', 'amd': 'hip'}
ARMS = ('on', 'off')
CORE_FIELDS = {'sha', 'plan_sha256', 'data_manifest_sha256', 'neural_fixture_sha256'}


def validate_arch(vendor, arch):
    pattern = {'nvidia': r'sm_[0-9]{2,3}[a-z]?', 'amd': r'gfx[0-9a-f]+'}.get(vendor)
    if pattern is None or not isinstance(arch, str) or not re.fullmatch(pattern, arch):
        raise ValueError('invalid GPU architecture for vendor')
    return arch


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def require_hash(value, length=64):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{'+str(length)+'}', value):
        raise ValueError('invalid digest or source SHA')
    return value


def identity_core(receipt):
    identity = receipt.get('identity', {})
    if set(identity) != CORE_FIELDS | {'vendor', 'gpu_arch'}:
        raise ValueError('identity fields missing or unexpected')
    validate_arch(identity['vendor'], identity['gpu_arch'])
    require_hash(identity['sha'], 40)
    require_hash(identity['plan_sha256'])
    require_hash(identity['data_manifest_sha256'])
    if not isinstance(identity['neural_fixture_sha256'], dict):
        raise ValueError('neural fixture inventory missing')
    for digest in identity['neural_fixture_sha256'].values():
        require_hash(digest)
    return {key: identity[key] for key in sorted(CORE_FIELDS)}


def expected_cases(plan_bytes):
    plan = json.loads(plan_bytes)
    tags = [case['lane']+'--'+case['dataset'] for case in plan['cases']]
    if not tags or len(set(tags)) != len(tags):
        raise ValueError('plan cases must be nonempty and unique')
    return set(tags)


def receipt_digests(receipt, vendor, plan_bytes):
    if receipt.get('status') != 'PASS' or receipt.get('phase') != 'identity':
        raise ValueError('identity receipt incomplete or failed')
    core = identity_core(receipt)
    if receipt['identity']['vendor'] != vendor or vendor not in BACKENDS:
        raise ValueError('identity vendor mismatch')
    if core['plan_sha256'] != sha256(plan_bytes):
        raise ValueError('identity plan hash mismatch')
    tags = expected_cases(plan_bytes)
    neural = {case['fixture'] for case in json.loads(plan_bytes)['cases'] if case.get('driver') == 'neural'}
    if set(core['neural_fixture_sha256']) != neural:
        raise ValueError('identity neural fixture inventory mismatch')
    if set(receipt.get('arms', {})) != set(ARMS):
        raise ValueError('identity must contain both arms')
    backend = BACKENDS[vendor]
    expected = {tag+'--'+suffix for tag in tags for suffix in (backend, 'cpu', 'bits')}
    result = {}
    for arm in ARMS:
        steps = receipt['arms'][arm]
        ids = [step['id'] for step in steps]
        if len(ids) != len(set(ids)) or set(ids) != expected:
            raise ValueError('identity missing, extra or duplicate cases/steps in '+arm)
        if any(step.get('status') != 'PASS' or step.get('rc', 0) != 0 for step in steps):
            raise ValueError('identity step failed in '+arm)
        result[arm] = {}
        for step in steps:
            if not step['id'].endswith('--bits'):
                continue
            digests = step.get('digests', {})
            if set(digests) != {backend, 'cpu'}:
                raise ValueError('GPU/CPU digest missing in '+arm)
            for digest in digests.values():
                require_hash(digest)
            if len(set(digests.values())) != 1:
                raise ValueError('local GPU/CPU digest mismatch in '+arm)
            result[arm][step['id'][:-6]] = digests
    return result


def compare_receipts(nvidia_bytes, amd_bytes, plan_bytes):
    raw = {'nvidia': nvidia_bytes, 'amd': amd_bytes}
    receipts = {vendor: json.loads(data) for vendor, data in raw.items()}
    cores = {vendor: identity_core(receipt) for vendor, receipt in receipts.items()}
    if cores['nvidia'] != cores['amd']:
        raise ValueError('cross-vendor source/plan/data/neural identity mismatch')
    maps = {vendor: receipt_digests(receipt, vendor, plan_bytes) for vendor, receipt in receipts.items()}
    combined = {}
    for arm in ARMS:
        combined[arm] = {}
        for tag in sorted(expected_cases(plan_bytes)):
            cells = {vendor: maps[vendor][arm][tag] for vendor in BACKENDS}
            if len({digest for ds in cells.values() for digest in ds.values()}) != 1:
                raise ValueError('cross-vendor digest mismatch: '+arm+'/'+tag)
            combined[arm][tag] = cells
    return {'schema': 1, 'status': 'PASS', 'identity_core': cores['nvidia'],
            'receipt_sha256': {v: sha256(b) for v, b in raw.items()},
            'expected_cases': sorted(expected_cases(plan_bytes)), 'arms': combined}


def classify_receipts(nvidia_bytes, amd_bytes, plan_bytes):
    """Per-case classification for a wave whose identity is not uniformly PASS.

    PROVEN: both vendors' GPU and host digests present and all four equal.
    GPU_ONLY_MATCH: both GPU digests equal, a host column missing (e.g. no host
    implementation); not PROVEN. DIFFER: any present digests disagree.
    MISSING: a GPU digest absent. Only PROVEN cases may be timed."""
    raw = {'nvidia': nvidia_bytes, 'amd': amd_bytes}
    receipts = {vendor: json.loads(data) for vendor, data in raw.items()}
    cores = {vendor: identity_core(receipt) for vendor, receipt in receipts.items()}
    if cores['nvidia'] != cores['amd']:
        raise ValueError('cross-vendor source/plan/data/neural identity mismatch')
    for vendor, receipt in receipts.items():
        if receipt.get('phase') != 'identity' or receipt['identity']['vendor'] != vendor:
            raise ValueError('identity receipt phase/vendor mismatch')
        if cores[vendor]['plan_sha256'] != sha256(plan_bytes):
            raise ValueError('identity plan hash mismatch')
    tags = sorted(expected_cases(plan_bytes))
    result = {'schema': 1, 'kind': 'partial', 'identity_core': cores['nvidia'],
              'receipt_sha256': {v: sha256(b) for v, b in raw.items()}, 'expected_cases': tags,
              'classes': {}, 'cells': {}}
    for arm in ARMS:
        cells, classes = {}, {}
        steps = {v: {s['id']: s for s in receipts[v].get('arms', {}).get(arm, [])} for v in BACKENDS}
        for tag in tags:
            digests = {}
            for vendor, backend in BACKENDS.items():
                bits = steps[vendor].get(tag + '--bits', {})
                d = {k: v for k, v in bits.get('digests', {}).items() if k in (backend, 'cpu')}
                for column in (backend, 'cpu'):
                    step = steps[vendor].get(tag + '--' + column)
                    if column in d and (not step or step.get('status') != 'PASS' or step.get('rc', 0) != 0):
                        d.pop(column)
                for value in d.values():
                    require_hash(value)
                digests[vendor] = d
            gpu = [digests[v].get(b) for v, b in BACKENDS.items()]
            values = {x for ds in digests.values() for x in ds.values()}
            if None in gpu:
                cls = 'MISSING'
            elif len(values) != 1:
                cls = 'DIFFER'
            elif all(set(digests[v]) == {b, 'cpu'} for v, b in BACKENDS.items()):
                cls = 'PROVEN'
            else:
                cls = 'GPU_ONLY_MATCH'
            cells[tag] = digests
            classes.setdefault(cls, []).append(tag)
        result['cells'][arm] = cells
        result['classes'][arm] = classes
    proven_everywhere = all(set(result['classes'][arm].get('PROVEN', [])) == set(tags) for arm in ARMS)
    result['status'] = 'PASS' if proven_everywhere else 'PARTIAL'
    return result


def validate_partial_cases(proof, local_bytes, vendor, plan_bytes, cases):
    """Timing gate for a case subset: every selected case PROVEN in both arms and
    this box's receipt digests equal to the proof's."""
    if proof.get('schema') != 1 or proof.get('kind') != 'partial' or proof.get('status') not in ('PASS', 'PARTIAL'):
        raise ValueError('partial cross-vendor proof missing')
    local = json.loads(local_bytes)
    if proof.get('identity_core') != identity_core(local) or proof['identity_core']['plan_sha256'] != sha256(plan_bytes):
        raise ValueError('partial proof identity mismatch')
    if proof.get('receipt_sha256', {}).get(vendor) != sha256(local_bytes):
        raise ValueError('partial proof local receipt hash mismatch')
    backend = BACKENDS[vendor]
    for arm in ARMS:
        proven = set(proof['classes'][arm].get('PROVEN', []))
        steps = {s['id']: s for s in local['arms'][arm]}
        for tag in cases:
            if tag not in proven:
                raise ValueError('case not PROVEN in ' + arm + ': ' + tag)
            bits = steps.get(tag + '--bits', {})
            if bits.get('status') != 'PASS' or bits.get('digests') != proof['cells'][arm][tag][vendor] or set(bits['digests']) != {backend, 'cpu'}:
                raise ValueError('local digest differs from partial proof: ' + arm + '/' + tag)


def validate_proof(proof, local_bytes, vendor, plan_bytes):
    if proof.get('schema') != 1 or proof.get('status') != 'PASS':
        raise ValueError('cross-vendor proof incomplete or failed')
    local = json.loads(local_bytes)
    local_map = receipt_digests(local, vendor, plan_bytes)
    if proof.get('identity_core') != identity_core(local):
        raise ValueError('cross-vendor proof identity mismatch')
    hashes = proof.get('receipt_sha256', {})
    if set(hashes) != set(BACKENDS):
        raise ValueError('cross-vendor receipt hashes missing')
    for digest in hashes.values():
        require_hash(digest)
    if hashes[vendor] != sha256(local_bytes):
        raise ValueError('cross-vendor proof local receipt hash mismatch')
    tags = expected_cases(plan_bytes)
    if proof.get('expected_cases') != sorted(tags) or set(proof.get('arms', {})) != set(ARMS):
        raise ValueError('cross-vendor proof coverage mismatch')
    for arm in ARMS:
        if set(proof['arms'][arm]) != tags:
            raise ValueError('cross-vendor proof cases missing')
        for tag in tags:
            cells = proof['arms'][arm][tag]
            if set(cells) != set(BACKENDS):
                raise ValueError('cross-vendor proof vendor missing')
            for v, backend in BACKENDS.items():
                if set(cells[v]) != {backend, 'cpu'}:
                    raise ValueError('cross-vendor proof GPU/CPU digest missing')
                for digest in cells[v].values():
                    require_hash(digest)
            if len({d for ds in cells.values() for d in ds.values()}) != 1:
                raise ValueError('cross-vendor proof digest mismatch')
            if cells[vendor] != local_map[arm][tag]:
                raise ValueError('cross-vendor proof local digest mismatch')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nvidia', required=True, type=Path)
    parser.add_argument('--amd', required=True, type=Path)
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--partial', action='store_true', help='Per-case classes (PROVEN/GPU_ONLY_MATCH/DIFFER/MISSING) for incomplete waves; only PROVEN cases may be timed')
    args = parser.parse_args()
    if args.partial:
        with args.out.open('x') as stream:  # never overwrite an earlier proof
            proof = classify_receipts(args.nvidia.read_bytes(), args.amd.read_bytes(), args.plan.read_bytes())
            stream.write(json.dumps(proof, indent=2) + '\n')
        print('CROSS_VENDOR_PARTIAL', proof['status'], json.dumps({arm: {k: len(v) for k, v in c.items()} for arm, c in proof['classes'].items()}, sort_keys=True),
              'DIFFER', {arm: c.get('DIFFER', []) for arm, c in proof['classes'].items()})
        return
    try:
        proof = compare_receipts(args.nvidia.read_bytes(), args.amd.read_bytes(), args.plan.read_bytes())
    except (ValueError, KeyError, TypeError, OSError) as exc:
        # A failed comparison must not leave an older PASS proof at this path.
        args.out.parent.mkdir(parents=True, exist_ok=True)
        failed = args.out.with_suffix(args.out.suffix+'.new')
        failed.write_text(json.dumps({'schema': 1, 'status': 'FAIL', 'error': str(exc)})+'\n')
        failed.replace(args.out)
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix+'.new')
    tmp.write_text(json.dumps(proof, indent=2)+'\n'); tmp.replace(args.out)
    print('CROSS_VENDOR_IDENTITY PASS cases='+str(len(proof['expected_cases']))+' arms=2 proof='+str(args.out))

if __name__ == '__main__':
    main()
