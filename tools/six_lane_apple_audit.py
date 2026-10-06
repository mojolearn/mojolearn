#!/usr/bin/env python3
"""Read-only Apple FAST full-workload/build admission inventory.

No estimator imports, compilation, device access or workload execution. Binary
hashes and exact source closure are checked against retained compile receipts;
compiler/target selection still belongs to the frozen compile/package planner.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter, defaultdict
from functools import lru_cache
import json
from pathlib import Path
from six_lane_ab import ROOT, digest, git, sha_value, write
from six_lane_catalog import source_graph


@lru_cache(None)
def file_digest(path):
    return digest(path)


def literal_assignments(path):
    tree = ast.parse(path.read_text())
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    values[target.id] = value
    return values, tree


def workload_facts():
    classical, _ = literal_assignments(ROOT/'tools/classical_two_datasets.py')
    more, _ = literal_assignments(ROOT/'tools/bench_board_more.py')
    expanded, tree = literal_assignments(ROOT/'tools/bench_board_algos.py')
    facts = {}
    for lane, block in classical['BLOCK_OF'].items():
        facts['classical/'+lane] = dict(harness='tools/classical_two_datasets.py', lane=lane,
            datasets=['taxi', 'istella'], block=block,
            cap_audit='Saved preparation contains intrinsic caps. Existing uncapped big blocks can serve PCA/OLS/KMeans; HDBSCAN additionally caps inside runner.',
            full_inputs_available=lane in ('pca', 'ols', 'kmeans'))
    capped_more = {'knn-clf', 'knn-reg', 'spectral', 'agglomerative', 'gpr', 'gpc', 'svr', 'kernel-ridge'}
    for lane, (block, binding, datasets) in more['LANES'].items():
        facts['classical2/'+lane] = dict(harness='tools/bench_board_more.py', lane=lane,
            datasets=list(datasets), block=block, binding=binding,
            intrinsic_lane_cap=lane in capped_more,
            cap_audit='GMM uncapped only with full_dataset_coverage metadata; other lane/data preparation caps require explicit audit.',
            full_inputs_available=block=='reg' and lane not in capped_more)
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id=='_add' and node.args and isinstance(node.args[0], ast.Constant)):
            continue
        lane = node.args[0].value
        kw = {x.arg:x.value for x in node.keywords}
        def value(key, default=None):
            if key not in kw:
                return default
            try:
                return ast.literal_eval(kw[key])
            except (ValueError, TypeError):
                return {'source_expression': ast.unparse(kw[key])}
        block = value('block')
        sub = value('sub', {})
        facts['algos/'+lane] = dict(harness='tools/bench_board_algos.py', lane=lane,
            source_line=node.lineno, datasets=value('datasets', ['taxi', 'istella']),
            block=block, subset_source=sub, intrinsic_lane_cap=bool(sub),
            estimator_settings_source=ast.unparse(kw['params']) if 'params' in kw else '{}',
            cap_audit='Data block cap, column slicing, special builder behavior and task outputs still require recipe admission; rows=full alone is insufficient.',
            full_inputs_available=block=='reg' and not sub)
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    store = ROOT/'experiments/six_lane_integration'
    matrix = json.loads((store/'matrix.json').read_text())
    plan = json.loads((store/'build_plan.json').read_text())
    configs = [c for c in matrix['configurations'] if c['mode']=='fast' and 'apple' in c['vendors']]
    jobs = [j for j in plan['jobs'] if j['vendor']=='apple' and j['mode']=='fast']
    closures, _ = source_graph()
    receipts = defaultdict(list)
    parse_errors = []
    for path in sorted(args.evidence.rglob('receipt.json')):
        try:
            data = json.loads(path.read_text())
        except (OSError, ValueError) as exc:
            parse_errors.append(dict(path=str(path), error=repr(exc)))
            continue
        if data.get('vendor')=='apple' and data.get('mode')=='fast':
            receipts[data['key']].append((path, data))
    records = []
    by_config = defaultdict(lambda: {'A': [], 'B': []})
    for job in jobs:
        files = {p:file_digest(str(ROOT/p)) for p in sorted(closures.get(job['binding'], {job['binding']}))}
        expected = sha_value(files)
        admitted = []
        rejected = Counter()
        for path, old in receipts.get(job['key'], []):
            if old.get('status')!='COMPILED':
                rejected['compile_failed'] += 1
                continue
            if old.get('source_closure_sha256')!=expected:
                rejected['numerical_source_closure_changed'] += 1
                continue
            if old.get('binding')!=job['binding'] or old.get('defines')!=job['defines']:
                rejected['binding_or_defines_changed'] += 1
                continue
            artifact = Path(old.get('artifact', ''))
            if not artifact.is_file():
                relocated = path.parent/artifact.name
                if not relocated.is_file():
                    rejected['binary_missing'] += 1
                    continue
                artifact = relocated
            if file_digest(str(artifact))!=old.get('artifact_sha256'):
                rejected['binary_hash_mismatch'] += 1
                continue
            admitted.append(dict(receipt=str(path), artifact=str(artifact),
                artifact_sha256=old['artifact_sha256'], original_source_sha=old['source_sha'],
                compiler=old.get('compiler'), compiler_sha256=old.get('compiler_sha256'),
                argv=old.get('argv'), source_closure_sha256=expected))
        row = dict(job, audit_status='SOURCE_COMPATIBLE_ARTIFACT' if admitted else 'MISSING_CURRENT_ARTIFACT',
                   current_source_closure_sha256=expected, artifact=admitted[-1] if admitted else None,
                   compatible_receipt_count=len(admitted), rejected_receipts=dict(rejected))
        records.append(row)
        for ref in job['configurations']:
            by_config[ref['configuration']][ref['arm']].append(dict(key=job['key'], binding=job['binding'], status=row['audit_status']))
    facts = workload_facts()
    configuration_rows = []
    for config in configs:
        arms = by_config[config['id']]
        work = []
        for item in config.get('workloads', []):
            if isinstance(item, str):
                mapped = facts.get(item)
                work.append(dict(source_workload=item, **(mapped or {'status':'RECIPE_MAPPING_PENDING'})))
            else:
                lane = item.get('lane')
                driver = item.get('driver')
                mapped = facts.get('algos/'+str(lane)) if driver=='tools/bench_board_algos.py' else None
                work.append(dict(source_workload=item, **(mapped or {'status':'FULL_ADAPTER_AND_SAVED_RECIPE_PENDING'})))
        paired = all(arms[a] and all(j['status']=='SOURCE_COMPATIBLE_ARTIFACT' for j in arms[a]) for a in ('A','B'))
        configuration_rows.append(dict(configuration=config['id'], campaign_role=config['campaign_role'],
            problems=config.get('problems', []), source_gaps=config.get('source_gaps', []),
            arm_bindings=arms, complete_paired_compile_closures=paired, workloads=work,
            queue_status='RECIPE_ADMISSION_PENDING', executable=False))
    summary = dict(source_sha=git('rev-parse', 'HEAD'), apple_fast_configurations=len(configs),
        deduplicated_build_jobs=len(jobs), build_status=dict(Counter(r['audit_status'] for r in records)),
        complete_paired_compile_configurations=sum(c['complete_paired_compile_closures'] for c in configuration_rows),
        executable_recipe_count=0, receipt_parse_errors=len(parse_errors),
        qualification='Static metadata and artifact audit only. No build, estimator import, device verification or timing. Compiler/target compatibility, packages, full data hashes/shapes, quality gates and recipe admission remain required.')
    args.output.mkdir(parents=True, exist_ok=True)
    write(args.output/'summary.json', summary)
    write(args.output/'artifacts.json', dict(jobs=records, parse_errors=parse_errors))
    write(args.output/'configurations.json', dict(configurations=configuration_rows))
    write(args.output/'missing-build-plan.json', dict(plan, jobs=[j for j,r in zip(jobs, records) if not r['artifact']]))
    print(json.dumps(summary, sort_keys=True))


if __name__=='__main__':
    main()
