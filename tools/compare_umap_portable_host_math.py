#!/usr/bin/env python3
"""Compare complete portable UMAP captures; no numerical tolerance."""
import argparse
import hashlib
import json
from pathlib import Path
from umap_identity_compare import read_capture

HOST_COUNTS = {
    'curve': 4, 'optimizer_input': 8, 'optimizer_weights': 16,
    'optimizer_reference_case': 8, 'graph_input': 9, 'rho': 3, 'sigma': 3,
    'directed': 9, 'fuzzy': 9, 'graph_layout_2': 6, 'graph_layout_3': 9,
    'transform_memberships_2': 6, 'transform_init_2': 4, 'transform_result_2': 4,
    'transform_memberships_3': 6, 'transform_init_3': 6, 'transform_result_3': 6,
}
TRANSFORM_COUNTS = {'membership': 6, '2': 6, '3': 9}


def words(path, prefix, counts, completion):
    text = path.read_text()
    if text.splitlines().count(completion) != 1:
        raise ValueError(f'{path}: missing/repeated completion marker')
    result = {}
    expected = {(name, index) for name, count in counts.items() for index in range(count)}
    for line in text.splitlines():
        if not line.startswith(prefix + ' '):
            continue
        _, name, index, value = line.split()
        key, bits = (name, int(index)), int(value)
        if key in result or key not in expected or not 0 <= bits <= 0xFFFFFFFF or bits & 0x7F800000 == 0x7F800000:
            raise ValueError(f'{path}: invalid record {line}')
        result[key] = bits
    if result.keys() != expected:
        raise ValueError(f'{path}: missing stage cells')
    return result


def summarize(left, right):
    if left.keys() != right.keys():
        raise ValueError('different capture shapes')
    changed = [key for key in left if left[key] != right[key]]
    per_stage = {}
    for stage, _ in changed:
        per_stage[stage] = per_stage.get(stage, 0) + 1
    digest = hashlib.sha256()
    for stage, index in sorted(left):
        digest.update(f'{stage}:{index}:'.encode())
        digest.update(left[stage, index].to_bytes(4, 'little'))
    return {'cells': len(left), 'changed': len(changed), 'changed_by_stage': per_stage,
            'left_records_sha256': digest.hexdigest()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence', type=Path)
    args = parser.parse_args()
    apple, linux = args.evidence / 'apple', args.evidence / 'linux-h100'
    results = {}
    for filename in ('identity.log', 'identity-broader.log'):
        results['cross_host_' + filename] = summarize(read_capture(apple / filename), read_capture(linux / filename))
        results['apple_baseline_' + filename] = summarize(read_capture(apple / ('baseline-' + filename)), read_capture(apple / filename))
    host = lambda path: words(path, 'UMAP_HOST_CELL', HOST_COUNTS, 'UMAP portable host math numerical admission and fingerprints PASS')
    transform = lambda path: words(path, 'TRANSFORM_CELL', TRANSFORM_COUNTS, 'UMAP transform native PASS')
    results['cross_host_host_stages'] = summarize(host(apple / 'host-stages.log'), host(linux / 'host-stages.log'))
    results['cross_host_transform'] = summarize(transform(apple / 'transform.log'), transform(linux / 'transform.log'))
    for name, result in results.items():
        if name.startswith('cross_host_') and result['changed']:
            raise ValueError(f'{name}: {result["changed"]} cell differences')
    print(json.dumps(results, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
