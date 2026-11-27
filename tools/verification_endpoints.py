#!/usr/bin/env python3
"""Generate the shipped endpoint inventory from the authoritative API census.

Source reach is NOT execution evidence. Unmapped endpoints stay visible.
--check rejects source changes and newly exposed endpoints until regenerated.
"""
import argparse
import base64
from collections import Counter
import json
from pathlib import Path
import textwrap
import zlib

from light_verifier_reference import load, ROOT

OUTPUT = ROOT/'python/mojolearn/_verification_endpoint_data.py'


# These endpoints return configuration, ownership or container metadata. Their
# public boundary fixtures still run. They have no device arithmetic of their
# own to sabotage. Keep exact names so new methods require a fresh review.
NON_NUMERICAL_ENDPOINTS = {
    'mojolearn.LanguageModelConfig.__post_init__': 'validates the model configuration',
    'mojolearn.ParallelByteLanguageModelTrainer.model_ownership': 'reports allocation ownership',
    'mojolearn.ParallelByteLanguageModelTrainer.optimizer_ownership': 'reports allocation ownership',
    'mojolearn.gpu_arch_how.call': 'reports how the installed GPU architecture was selected',
    'mojolearn.linalg.require_identical.call': 'checks the selected numerical mode',
    'mojolearn.model_selection.ParameterGrid.__len__': 'counts parameter combinations',
    'mojolearn.model_selection.ParameterGrid.__iter__': 'enumerates parameter configurations',
    'mojolearn.model_selection.ParameterSampler.__len__': 'reports the number of sampled configurations',
    'mojolearn.ExperimentalTwoLevelFeatureFreq.predict_proba': 'RMSE-only estimator explicitly refuses probabilities',
    'mojolearn.resample.BootstrapResult.__init__': 'stores supplied statistics and the supplied interval bounds',
    'mojolearn.resample.PermutationTestResult.__init__': 'stores the supplied permutation statistics',
    'mojolearn.resample.MonteCarloResult.__init__': 'stores the supplied integration statistics',
    'mojolearn.models.CausalLMState.__init__': 'stores the supplied layer state, dimensions and initial position',
    'mojolearn.models.Checkpoint.__contains__': 'tests whether a weight name is present',
    'mojolearn.models.SafetensorsFile.__contains__': 'tests whether a weight name is present',
    'mojolearn.models.HFConfig.__contains__': 'tests whether a configuration key is present',
    'mojolearn.models.HFConfig.__getitem__': 'reads a model configuration field',
    'mojolearn.models.Tokenizer.unicode_version': 'returns the Unicode table version',
    'mojolearn.models.catalog.call': 'reads the bundled catalog metadata',
    'mojolearn.numeric_profile.call': 'reads the numerical profile name',
    'mojolearn.numeric_profile_measured.call': 'reads the profile measurement metadata',
    'mojolearn.numeric_profiles.call': 'lists supported numerical profiles',
    'mojolearn.set_numeric_mode.call': 'sets the process numerical mode',
    'mojolearn.set_numeric_profile.call': 'sets the process numerical profile',
    'mojolearn.ops.Program.inspect': 'describes the recorded program',
    'mojolearn.ops.Session.execution_profile_begin': 'starts profiling diagnostics',
    'mojolearn.ops.Session.execution_profile_end': 'returns profiling diagnostics',
}


def require_complete_mapping(rows, targets, routes):
    missing={row['endpoint'] for row in targets}-routes.keys()
    missing.update(row['endpoint'] for row in rows if row['status']=='unmapped')
    if missing:
        raise ValueError('unimplemented endpoint fixtures: '+', '.join(sorted(missing)))


def native_faults_by_lane(harness):
    from coverage_sabotage import arms, arm_sides

    faults = {lane: {side: [] for side in ('cpu', 'gpu')
                     if side != (harness.one_sided(lane) or (None,))[0]}
              for lane in harness.record_lanes()}
    for name, arm in arms().items():
        for lane in arm['lanes']:
            for side in arm_sides(arm):
                if side in faults.get(lane, {}):
                    faults[lane][side].append(name)
    return faults


def reviewed_endpoint_sides(path):
    """Exact public names whose product interface has only one numerical side."""
    scope = {}
    for line in path.read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        endpoint, side, why = line.split('\t')
        if side not in ('cpu', 'gpu') or not why.strip() or endpoint in scope:
            raise ValueError('invalid endpoint device scope: ' + line)
        scope[endpoint] = (side, why)
    return scope


def endpoint_fault_scope(rows, faults, scope):
    """Credit an applicable side only if a reached record lane names an arm there."""
    missing = scope.keys() - {row['endpoint'] for row in rows}
    if missing:
        raise ValueError('device scope names a removed endpoint: ' + ', '.join(sorted(missing)))
    counts = Counter()
    for row in rows:
        name = row['endpoint']
        if row['status'] == 'not_numerical':
            counts['not_numerical'] += 1
            continue
        expected = {'cpu', 'gpu'}
        if name in scope:
            side, row['device_scope_reason'] = scope[name]
            expected = {side}
        linked = {side for lane in row['lanes'] for side, arms in faults.get(lane, {}).items()
                  if side in expected and arms}
        row['required_native_fault_sides'] = sorted(expected)
        row['linked_native_fault_sides'] = sorted(linked)
        row['owed_native_fault_sides'] = sorted(expected - linked)
        counts['owed' if linked != expected else '+'.join(sorted(linked))] += 1
    return dict(counts)


def require_endpoint_native_faults(rows):
    missing = [r['endpoint'] + ' (' + ', '.join(r['owed_native_fault_sides']) + ')'
               for r in rows if r.get('owed_native_fault_sides')]
    if missing:
        raise ValueError('missing applicable native fault links: ' + ', '.join(missing))


def build():
    import lane_select as lanes
    from verify_coverage_audit import Walker, lane_reach, endpoint_rows
    walker = Walker()
    reach = lane_reach(lanes, walker, public_only=True)
    rows = endpoint_rows(walker, reach)
    # Load test controllers without importing an unbuilt product package.
    import importlib, sys, types
    package = types.ModuleType('_endpoint_fixture_sources')
    package.__path__ = [str(ROOT/'python/mojolearn')]
    sys.modules[package.__name__] = package
    runner = importlib.import_module(package.__name__+'._verify_api_runner')
    availability = importlib.import_module(package.__name__+'._verify_api_availability')
    routes = runner.registry()
    for row in rows:
        route = routes.get(row['endpoint'])
        if route:
            row['checks'] = sorted(set(row['checks']) | {'_verify_api_runner:'+route[0]})
            if row['status'] != 'not_numerical': row['status'] = 'mapped'
            row['fixture_scope'] = ('availability contract, not numerical execution'
                if row['endpoint'] in availability.ABSENT | availability.CONDITIONAL
                else 'bounded public endpoint fixture')
        if row['endpoint'] in availability.ABSENT | availability.CONDITIONAL:
            row.update(status='not_numerical', reason='availability fixture expects absent method')
        elif row['endpoint'] in NON_NUMERICAL_ENDPOINTS:
            row.update(status='not_numerical', reason=NON_NUMERICAL_ENDPOINTS[row['endpoint']])
    require_complete_mapping(rows,runner.TARGETS,routes)
    faults = native_faults_by_lane(lanes.identity_break())
    scope = reviewed_endpoint_sides(ROOT/'tools/identity_lanes/endpoint_device_scope.tsv')
    fault_counts = endpoint_fault_scope(rows, faults, scope)
    source = load('_verification_sources')
    return dict(format='mojolearn.endpoint-coverage.v1', execution='not run',
                sources=source.public_sources(ROOT/'python/mojolearn'),
                supplemental_sources=source.supplemental_sources(ROOT/'python/mojolearn'),
                harness_sources=source.harness_sources(lanes.identity_break()),
                native_fault_arms=faults, endpoint_native_fault_counts=fault_counts,
                endpoints=rows, counts=dict(Counter(row['status'] for row in rows)),
                limitations=['Source mapping is not endpoint execution or numerical qualification.',
                             'Unmapped means the analyzer did not resolve a verifier call; manual review may be needed.',
                             'Constructor options, shapes and error paths are not exhaustively covered.'])


def write(data):
    payload = base64.b85encode(zlib.compress(json.dumps(data, sort_keys=True, separators=(',', ':')).encode(), 9)).decode()
    chunks = '\n'.join('    ' + repr(s) for s in textwrap.wrap(payload, 100, break_on_hyphens=False, drop_whitespace=False))
    OUTPUT.write_text('"""Generated by tools/verification_endpoints.py --write; source mapping only."""\n'
                      'import base64 as _base64\nimport json as _json\nimport zlib as _zlib\n'
                      '_PAYLOAD = (\n' + chunks + '\n)\n'
                      'DATA = _json.loads(_zlib.decompress(_base64.b85decode(_PAYLOAD)))\ndel _PAYLOAD\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--write', action='store_true')
    mode.add_argument('--check', action='store_true')
    parser.add_argument('--json', type=Path)
    args = parser.parse_args()
    data = build()
    if args.json:
        args.json.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')
    if args.check:
        require_endpoint_native_faults(data['endpoints'])
        if not OUTPUT.exists() or load('_verification_endpoint_data').DATA != data:
            parser.exit(1, 'Endpoint inventory is stale; regenerate and review new unmapped endpoints.\n')
    else:
        write(data)
    print(json.dumps(dict(endpoints=len(data['endpoints']), counts=data['counts'], execution=data['execution'])))


if __name__ == '__main__':
    main()
