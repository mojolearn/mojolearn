#!/usr/bin/env python3
"""Retained predictions only: baseline-only margin freeze, then paired adjudication.

No model imports, fits, opponent execution, GPU work, timing, or changes to the
existing DART reference/identity gates. Execute statistics on an authorized
server. A noise margin is a declared policy, not a proof of equivalence.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
import sys


class Insufficient(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise Insufficient(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    with Path(path).open('x') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write('\n')


def validate_protocol(p, metadata):
    require(p.get('schema') == 1 and p.get('reviewed') is True,
            'protocol must be explicitly reviewed before calibration')
    require(p.get('primary_metrics') == ['accuracy', 'r2'],
            'primary metrics must remain accuracy and R2')
    require(p.get('family_alpha') == 0.05 and p.get('per_metric_alpha') == 0.025,
            'declare the two-metric Bonferroni policy: family .05, each .025')
    require(p.get('resamples') == 2000 and isinstance(p.get('seed'), int),
            'this protocol requires 2000 fixed-seed resamples')
    require(p.get('margin_rule') == 'baseline_score_minus_lower_95_bound',
            'baseline-only margin rule must be fixed before comparison')
    require(p.get('quantile_method') == 'linear', 'quantile method must be linear')
    sampling = p.get('resampling', {})
    kind = sampling.get('kind')
    require(kind in ('iid', 'groups', 'moving_block'), 'resampling design unresolved')
    require(bool(sampling.get('justification')), 'resampling justification missing')
    require(metadata.get('sampling_design'), 'target sampling provenance missing')
    require(metadata.get('source_evidence'), 'target source evidence missing')
    for row in metadata['source_evidence']:
        require(digest(row['path']) == row['sha256'], 'target source evidence changed')
    if kind == 'iid':
        require(metadata.get('iid_justified') is True and
                metadata.get('sampling_design') == 'random_iid',
                'IID rows unsupported: temporal/stride holdout is not random IID')
    else:
        require(metadata.get('resampling_unit') == sampling.get('unit'),
                'declared groups/blocks differ from target provenance')
        require(bool(metadata.get('dependence_justification')),
                'group/block dependence justification missing')
        require(sampling.get('minimum_units', 0) >= 20,
                'declare at least20 effective resampling units')
    if kind == 'moving_block':
        require(metadata.get('order_verified') is True,
                'moving blocks require verified order, not assumed file ordering')
        require(isinstance(sampling.get('block_length'), int) and
                sampling['block_length'] >= 2, 'block length missing/invalid')


def receipt(path, prediction_path, target_path, arm):
    r = read_json(path)
    require(r.get('schema') == 1 and r.get('arm') == arm, 'wrong arm receipt')
    require(r.get('numeric_mode') == 'identical', 'IDENTICAL provenance required')
    sha = r.get('source_sha', '')
    require(len(sha) == 40 and all(c in '0123456789abcdef' for c in sha),
            'full source SHA required')
    require(r.get('predictions_sha256') == digest(prediction_path),
            'prediction artifact hash mismatch')
    require(r.get('targets_sha256') == digest(target_path), 'target artifact hash mismatch')
    require(set(r.get('parameter_sha256', {})) == {'classification', 'regression'},
            'both model parameter hashes required')
    builds = r.get('native_build_receipts', [])
    require(builds, 'our source-build receipts required; no wheel fallback')
    built_artifacts = {}
    explicit_arm = False
    for row in builds:
        require(digest(row['path']) == row['sha256'], 'native build receipt changed')
        build = read_json(row['path'])
        require(build.get('status') == 'PASS', 'native source build did not pass')
        require(build.get('sha', build.get('source_sha')) == sha,
                'native build source SHA mismatch')
        require(build.get('arm', arm) == arm, 'native build arm mismatch')
        explicit_arm |= build.get('arm') == arm
        if 'modules' in build:
            for module in build['modules'].values():
                require(module.get('status') == 'PASS', 'native module failed')
                built_artifacts[module['artifact']] = module['sha256']
        elif 'artifact' in build:
            built_artifacts[build['artifact']] = build['sha256']
    loaded = r.get('loaded_native_artifacts', {})
    require(explicit_arm, 'at least one source-build receipt must attest the comparison arm')
    require(loaded, 'loaded native artifact hashes missing')
    require(all(built_artifacts.get(path) == sha256 for path, sha256 in loaded.items()),
            'loaded artifact not accounted by matching source-build receipt')
    require(r.get('fixture_id'), 'fixture identifier missing')
    return r


def load_arrays(prediction_path, target_path, protocol):
    import numpy as np
    result = {}
    with np.load(prediction_path, allow_pickle=False) as pred, np.load(target_path, allow_pickle=False) as target:
        require(set(protocol['cases']) == {'classification', 'regression'}, 'both cases required')
        for name, spec in protocol['cases'].items():
            require(spec['prediction_key'] in pred and spec['target_key'] in target,
                    name + ': retained prediction/target missing')
            p = np.asarray(pred[spec['prediction_key']]).reshape(-1).astype(np.float64)
            y = np.asarray(target[spec['target_key']]).reshape(-1).astype(np.float64)
            require(p.shape == y.shape and len(y) >= 2, name + ': length mismatch/empty')
            require(np.isfinite(p).all() and np.isfinite(y).all(), name + ': nonfinite data')
            if name == 'classification':
                require(np.isin(y, [0, 1]).all() and np.isin(p, [0, 1]).all(),
                        'binary class labels required')
            row = {'p': p, 'y': y}
            for key in ('group_key', 'order_key'):
                if spec.get(key):
                    require(spec[key] in target, name + ': grouping/order array missing')
                    value = np.asarray(target[spec[key]])
                    require(value.ndim == 1 and len(value) == len(y), 'invalid group/order shape')
                    require(np.issubdtype(value.dtype, np.integer), 'group/order IDs must be integers')
                    row[key] = value.astype(np.int64)
            prob_key = spec.get('probability_key')
            if prob_key:
                require(prob_key in pred, 'declared probability output missing')
                prob = np.asarray(pred[prob_key], dtype=np.float64)
                if prob.shape == (len(y), 2):
                    require(np.allclose(prob.sum(axis=1), 1, atol=1e-6, rtol=0),
                            'class probabilities do not sum to1')
                    prob = prob[:, 1]
                require(prob.shape == y.shape and np.isfinite(prob).all() and
                        ((prob >= 0) & (prob <= 1)).all(), 'invalid probabilities')
                row['prob'] = prob
            result[name] = row
    return result


def score(name, y, p):
    import numpy as np
    if name == 'classification':
        return float(np.mean(y == p))
    centered = y - np.mean(y)
    denominator = float(np.dot(centered, centered))
    require(denominator > 0 and math.isfinite(denominator),
            'R2 undefined for constant/degenerate resampled target')
    residual = y - p
    value = 1 - float(np.dot(residual, residual)) / denominator
    require(math.isfinite(value), 'nonfinite metric')
    return value


def wilson_lower(correct, n, alpha):
    z = NormalDist().inv_cdf(1 - alpha)
    p = correct / n
    denominator = 1 + z * z / n
    return (p + z * z / (2 * n) - z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denominator


def resamples(row, protocol, case_index):
    import numpy as np
    n = len(row['y'])
    rng = np.random.Generator(np.random.PCG64(protocol['seed'] + case_index))
    sampling = protocol['resampling']
    kind = sampling['kind']
    members = None
    order = np.arange(n, dtype=np.int64)
    if kind == 'groups':
        require('group_key' in row, 'group IDs missing')
        groups = np.unique(row['group_key'])
        require(len(groups) >= sampling['minimum_units'], 'too few independent groups')
        members = [np.flatnonzero(row['group_key'] == g) for g in groups]
    if kind == 'moving_block':
        require('order_key' in row, 'verified temporal order array missing')
        order = row['order_key']
        require(np.array_equal(np.sort(order), np.arange(n)), 'order must be a row permutation')
        length = sampling['block_length']
        require(n // length >= sampling['minimum_units'], 'too few temporal blocks')
    for _ in range(protocol['resamples']):
        if kind == 'iid':
            indices = rng.integers(0, n, size=n, dtype=np.int64)
        elif kind == 'groups':
            chosen = rng.integers(0, len(members), size=len(members))
            indices = np.concatenate([members[int(i)] for i in chosen])
        else:
            starts = rng.integers(0, n, size=math.ceil(n / length))
            offsets = (starts[:, None] + np.arange(length)[None, :]).ravel()[:n] % n
            indices = order[offsets]
        yield indices


def sample_scores(name, baseline, protocol, case_index, candidate=None):
    import numpy as np
    values = []
    stream = hashlib.sha256()
    for indices in resamples(baseline, protocol, case_index):
        stream.update(len(indices).to_bytes(8, 'little'))
        stream.update(indices.astype('<i8', copy=False).tobytes())
        y = baseline['y'][indices]
        value = score(name, y, baseline['p'][indices])
        if candidate is not None:
            value = score(name, y, candidate['p'][indices]) - value
        values.append(value)
    return np.asarray(values), stream.hexdigest()


def diagnostics(row):
    import numpy as np
    if 'prob' not in row:
        return {}
    prob = np.clip(row['prob'], 1e-15, 1 - 1e-15)
    y = row['y']
    return {'logloss': float(-np.mean(y * np.log(prob) + (1 - y) * np.log1p(-prob))),
            'logloss_clip': 1e-15, 'role': 'diagnostic_only_not_a_promotion_metric'}


def calibrate(args):
    import numpy as np
    protocol = read_json(args.protocol)
    metadata = read_json(args.target_metadata)
    validate_protocol(protocol, metadata)
    require(metadata.get('targets_sha256') == digest(args.targets), 'target metadata hash mismatch')
    baseline_receipt = receipt(args.baseline_receipt, args.baseline, args.targets, 'off')
    arrays = load_arrays(args.baseline, args.targets, protocol)
    rows = {}
    for index, name in enumerate(('classification', 'regression')):
        row = arrays[name]
        point = score(name, row['y'], row['p'])
        if name == 'classification' and protocol['resampling']['kind'] == 'iid':
            lower = wilson_lower(int(np.count_nonzero(row['y'] == row['p'])), len(row['y']), 0.025)
            method, index_hash = 'wilson_95_lower', None
        else:
            values, index_hash = sample_scores(name, row, protocol, index)
            lower = float(np.quantile(values, 0.025, method='linear'))
            method = 'baseline_percentile_bootstrap_95_lower'
        rows[name] = {'baseline_score': point, 'lower_bound': lower,
                      'margin': max(0.0, point - lower), 'method': method,
                      'n': len(row['y']), 'resample_indices_sha256': index_hash,
                      'diagnostics': diagnostics(row)}
    files = {name: {'path': str(Path(value).resolve()), 'sha256': digest(value)}
             for name, value in [('baseline', args.baseline), ('baseline_receipt', args.baseline_receipt),
                                 ('targets', args.targets), ('target_metadata', args.target_metadata),
                                 ('protocol', args.protocol)]}
    output = {'schema': 1, 'status': 'FROZEN_BASELINE_ONLY', 'protocol': protocol,
              'protocol_files': files, 'baseline_receipt': baseline_receipt,
              'metrics': rows, 'numpy_version': np.__version__,
              'script_sha256': digest(__file__), 'comparison_inspected': False,
              'scope': 'Holdout sampling uncertainty conditional on fitted baseline; no seed-training-noise claim.'}
    write_new(args.out, output)
    return output


def compare(args):
    import numpy as np
    frozen = read_json(args.frozen)
    require(digest(args.frozen) == args.frozen_sha256,
            'freeze digest differs from the baseline-only recorded commitment')
    require(frozen.get('status') == 'FROZEN_BASELINE_ONLY' and
            frozen.get('comparison_inspected') is False, 'not a valid baseline-only freeze')
    require(frozen.get('script_sha256') == digest(__file__), 'statistics implementation changed after freeze')
    require(frozen.get('numpy_version') == np.__version__, 'NumPy changed after freeze')
    files = frozen['protocol_files']
    for row in files.values():
        require(digest(row['path']) == row['sha256'], 'frozen baseline/protocol/target changed')
    protocol = frozen['protocol']
    require(read_json(files['protocol']['path']) == protocol, 'embedded protocol changed')
    validate_protocol(protocol, read_json(files['target_metadata']['path']))
    target_path = files['targets']['path']
    baseline_path = files['baseline']['path']
    old = receipt(files['baseline_receipt']['path'], baseline_path, target_path, 'off')
    new = receipt(args.candidate_receipt, args.candidate, target_path, 'on')
    for key in ('source_sha', 'targets_sha256', 'parameter_sha256', 'fixture_id', 'numeric_mode'):
        require(old[key] == new[key], 'ON/OFF ' + key + ' mismatch')
    baseline = load_arrays(baseline_path, target_path, protocol)
    candidate = load_arrays(args.candidate, target_path, protocol)
    rows = {}
    for index, name in enumerate(('classification', 'regression')):
        require(np.array_equal(baseline[name]['y'], candidate[name]['y']), 'paired targets differ')
        values, indices_hash = sample_scores(name, baseline[name], protocol, index, candidate[name])
        previous = frozen['metrics'][name]['resample_indices_sha256']
        require(previous is None or previous == indices_hash, 'frozen resampling sequence changed')
        lo, hi = [float(x) for x in np.quantile(values, [0.025, 0.975], method='linear')]
        margin = frozen['metrics'][name]['margin']
        require(math.isfinite(margin) and margin >= 0, 'invalid frozen margin')
        candidate_score = score(name, candidate[name]['y'], candidate[name]['p'])
        base_score = score(name, baseline[name]['y'], baseline[name]['p'])
        require(base_score == frozen['metrics'][name]['baseline_score'], 'baseline score changed')
        status = 'PASS' if lo >= -margin else ('FAIL' if hi < -margin else 'INCONCLUSIVE')
        rows[name] = {'status': status, 'baseline_score': base_score, 'candidate_score': candidate_score,
                      'delta_on_minus_off': candidate_score - base_score,
                      'lower_bound': lo, 'upper_bound': hi, 'frozen_margin': margin,
                      'resample_indices_sha256': indices_hash,
                      'diagnostics': {'baseline': diagnostics(baseline[name]),
                                      'candidate': diagnostics(candidate[name])}}
    statuses = [r['status'] for r in rows.values()]
    status = 'PASS' if all(s == 'PASS' for s in statuses) else ('FAIL' if 'FAIL' in statuses else 'INCONCLUSIVE')
    result = {'schema': 1, 'status': status, 'metrics': rows,
              'frozen_sha256': digest(args.frozen), 'candidate_sha256': digest(args.candidate),
              'candidate_receipt_sha256': digest(args.candidate_receipt),
              'root_decision_required': True,
              'scope': 'Predictive-quality evidence only. Does not waive identity, numerical residual, semantics or speed gates.'}
    write_new(args.out, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    c = commands.add_parser('calibrate')
    for name in ('baseline', 'baseline-receipt', 'targets', 'target-metadata', 'protocol', 'out'):
        c.add_argument('--' + name, type=Path, required=True)
    c = commands.add_parser('compare')
    for name in ('frozen', 'candidate', 'candidate-receipt', 'out'):
        c.add_argument('--' + name, type=Path, required=True)
    c.add_argument('--frozen-sha256', required=True,
                   help='digest recorded at baseline-only freeze, before candidate inspection')
    args = p.parse_args(argv)
    if sys.platform != 'linux':
        p.error('run statistics on the authorized Linux server, not the local workstation')
    if args.out.exists():
        p.error('refuse overwrite: ' + str(args.out))
    try:
        result = calibrate(args) if args.command == 'calibrate' else compare(args)
    except (Insufficient, OSError, ValueError, KeyError, TypeError) as exc:
        result = {'status': 'INSUFFICIENT', 'reason': str(exc), 'command': args.command}
        write_new(args.out, result)
    print('QUALITY_NOISE', result['status'], 'report', args.out)
    return 0 if result['status'] in ('PASS', 'FROZEN_BASELINE_ONLY') else 2


if __name__ == '__main__':
    raise SystemExit(main())
