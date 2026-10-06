#!/usr/bin/env python3
"""Register four reviewed full MLP estimator recipes; no models or authorization."""
import argparse
import json
from pathlib import Path
from six_lane_ab import ROOT, catalog, matrix, git, write, queue
from six_lane_mlp_variants import VARIANT, CONTRACT, contracts, profile, digest, canonical, workload, validate_variant
from six_lane_materialize import materialize


def facts_for(cell, source, directory):
    row = profile(dict(cell, workload_id=cell['original_workload_id']))
    reg = dict(variant=VARIANT,measurement_source_sha=source,
        contract_sha256=digest(ROOT/CONTRACT),original_workload_id=cell['original_workload_id'],
        original_cell_key=cell['original_cell_key'],variant_workload_id=cell['workload_id'],
        variant_cell_key=cell['key'])
    fact = dict(source_sha=source,changes_frozen_race=True,registered_input_variant=reg,
        dataset_sha256=canonical(row['arrays']),dimensions=row['actual_shapes'],
        estimator_settings=row['estimator_settings_record'],timed_boundary=row['timed_boundary'],
        intrinsic_caps=[],full_dataset_coverage=True,workload=workload(row,directory),
        resource_policy=dict(policy='Serial full machine allocation under canonical device lock; no concurrent input transfer'),
        coverage_resolutions={reason:dict(path=str(ROOT/CONTRACT),sha256=digest(ROOT/CONTRACT),
            conclusion='Explicit saved full MLP estimator recipe with retained full inputs and unchanged settings; unrelated combined-candidate scope remains pending. No individual-candidate reach or quality/promotion claim.') for reason in cell['blockers']})
    validate_variant(fact,cell)
    return fact


def run(args):
    if args.output.exists() or args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Fresh evidence directory outside source required')
    if git('status','--porcelain','--untracked-files=all'):
        raise ValueError('Registration requires immutable clean source')
    source=git('rev-parse','HEAD');contracts()
    mat=matrix(catalog())
    selected=[c for c in mat['cells'] if c.get('input_variant')==VARIANT and c['vendor']=='nvidia']
    if len(selected)!=4 or len({c['key'] for c in selected})!=4:
        raise ValueError('Expected exactly four reviewed native MLP cells')
    facts={c['workload_id']:facts_for(c,source,args.cls_directory if ':mlp-clf@' in c['workload_id'] else args.reg_directory) for c in selected}
    args.output.mkdir(parents=True)
    write(args.output/'matrix.json',dict(mat,cells=selected));write(args.output/'facts.json',facts)
    write(args.output/'registration.json',dict(variant=VARIANT,source_sha=source,
        selected=[dict(key=c['key'],workload_id=c['workload_id'],scope_cell_key=c['original_cell_key']) for c in selected],
        original_cells_preserved=True,execution_authorized=False,
        qualification='Full saved-estimator coverage only; individual candidate reach, complete model state and quality remain separate'))
    materialize(argparse.Namespace(matrix=args.output/'matrix.json',workloads=args.output/'facts.json',
        deployments=args.deployments,vendor='nvidia',target_track='nvidia-native',select=[],output=args.output/'materialized'))
    coverage=json.loads((args.output/'materialized/coverage.json').read_text())
    if coverage['blocked'] or coverage['materialized']!=4:
        raise ValueError('Full MLP admission incomplete; retain blocked coverage')
    recipes=json.loads((args.output/'materialized/recipes.json').read_text())
    for cell in selected:
        validate_variant(recipes[cell['key']],cell)
    queue(argparse.Namespace(matrix=args.output/'matrix.json',recipes=args.output/'materialized/recipes.json',
        vendor='nvidia',select=[],output=args.output/'queue.json'))
    print(json.dumps(dict(status='REGISTERED_QUEUE_NOT_AUTHORIZED',cells=4,source_sha=source,output=str(args.output))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cls-directory',type=Path,required=True)
    p.add_argument('--reg-directory',type=Path,required=True)
    p.add_argument('--deployments',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
