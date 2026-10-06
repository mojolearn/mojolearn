#!/usr/bin/env python3
"""Register reviewed full classification recipes using retained paired artifacts.

No compilation, model imports, execution or implicit authorization. All prepared
archives are hashed by the existing materializer; run admission before timing,
after transport, under the machine's measurement lock.
"""
import argparse
import copy
import json
from pathlib import Path
from six_lane_ab import ROOT, catalog, matrix, git, write, queue
from six_lane_classification_variants import (VARIANT, CONTRACT, contracts, profile,
    digest, canonical, parameter_record, validate_variant)
from six_lane_materialize import materialize


def facts_for(cell, source, receipt_path, plan_path, data_directory):
    receipt = json.loads(receipt_path.read_text())
    original = dict(cell, workload_id=cell['original_workload_id'])
    row = profile(original)
    name = row['block'] + '-' + row['dataset']
    sidecar = data_directory / (name + '.json')
    meta = json.loads(sidecar.read_text())
    block, = [b for b in receipt['blocks'] if b['name'] == name]
    inputs = [dict(path=str(data_directory / (name + ext)), sha256=block[key])
              for ext,key in (('.npz','npz_sha256'),('.json','sidecar_sha256'))]
    params = parameter_record(row, meta)
    reg = dict(variant=VARIANT, preparation_source_sha=receipt['source_sha'],
        measurement_source_sha=source, contract_sha256=digest(ROOT / CONTRACT),
        original_workload_id=cell['original_workload_id'], original_cell_key=cell['original_cell_key'],
        variant_workload_id=cell['workload_id'], variant_cell_key=cell['key'],
        preparation_receipt=dict(path=str(receipt_path),sha256=digest(receipt_path)),
        preparation_plan=dict(path=str(plan_path),sha256=digest(plan_path)))
    shapes = copy.deepcopy(row['expected_shapes_from_population_metadata'])
    work = dict(harness=row['harness'], harness_sha256=row['harness_sha256'],
        input_variant=VARIANT, lane=row['lane'], dataset=row['dataset'],
        data_directory=str(data_directory), input_files=inputs, actual_shapes=shapes,
        estimator_settings_record=params, inference=row['inference'],
        dataset_version=VARIANT + '; original loader archive ' + meta['source_archive']['sha256'],
        split=dict(population=meta['population'],fit=meta['fit_rows'],query=meta['eval_rows'],
                   original_preparation_caps=meta['original_preparation_caps'],preparation_caps=meta['preparation_caps']),
        seed=row['seed'],output_paths=row['output_paths'],output_schema=row['output_schema'],
        repeated_operations=row['repeated_operations'],
        runtime_vendor={'apple':'metal','nvidia':'cuda','amd':'hip'}[cell['vendor']],
        quality_gate_source=row['quality_gate_source'],
        capture_limitation=row['limitations'],
        intrinsic_cap_audit=dict(reviewed=True,unresolved=[],evidence=str(receipt_path)))
    fact = dict(source_sha=source,changes_frozen_race=True,registered_input_variant=reg,
        dataset_sha256=canonical(meta['arrays']),dimensions=shapes,estimator_settings=params,
        timed_boundary=row['timed_boundary'],intrinsic_caps=[],full_dataset_coverage=True,
        workload=work,resource_policy=dict(policy='serial whole-machine allocation; Apple CPU libraries unrestricted, Linux actual cgroup; shared lock; no overlapping transfer'),
        coverage_resolutions={reason:dict(path=str(receipt_path),sha256=digest(receipt_path),
            conclusion='Reviewed distinct full input variant; exact original population, transforms, settings and consumed output scope. Original capped race preserved; complete candidate reach, quality and state identity remain separate pending qualification.') for reason in cell['blockers']})
    validate_variant(fact,cell)
    return fact


def run(args):
    if args.output.exists() or args.output.resolve().is_relative_to(ROOT):
        raise ValueError('Fresh registration evidence outside source required')
    if git('status','--porcelain','--untracked-files=all'):
        raise ValueError('Registration requires an immutable clean source freeze')
    source = git('rev-parse','HEAD'); contracts()
    mat = matrix(catalog())
    selected = [c for c in mat['cells'] if c.get('input_variant') == VARIANT and c['vendor'] == args.vendor
                and (not args.workload or c['original_workload_id'] in args.workload)]
    if not selected:
        raise ValueError('No reviewed eligible full classification cells')
    if args.workload and set(args.workload) != {c['original_workload_id'] for c in selected}:
        raise ValueError('Requested an unreviewed classification cell')
    receipt_path = args.preparation_receipt.resolve(); plan_path = args.preparation_plan.resolve()
    facts = {c['workload_id']:facts_for(c,source,receipt_path,plan_path,args.data_directory.resolve()) for c in selected}
    args.output.mkdir(parents=True)
    write(args.output/'matrix.json',dict(mat,cells=selected));write(args.output/'facts.json',facts)
    write(args.output/'registration.json',dict(variant=VARIANT,source_sha=source,
        preparation_source_sha=json.loads(receipt_path.read_text())['source_sha'],
        selected=[dict(key=c['key'],workload_id=c['workload_id'],original_cell_key=c['original_cell_key'],original_workload_id=c['original_workload_id']) for c in selected],
        original_cells_preserved=True,execution_authorized=False,
        capture_scope='Full input and declared output scope only; complete fitted state, quality and candidate reach require retained independent qualification'))
    materialize(argparse.Namespace(matrix=args.output/'matrix.json',workloads=args.output/'facts.json',
        deployments=args.deployments,vendor=args.vendor,target_track=args.target_track,select=[],output=args.output/'materialized'))
    coverage=json.loads((args.output/'materialized/coverage.json').read_text())
    if coverage['blocked'] or coverage['materialized']!=len(selected):
        raise ValueError('Classification admission incomplete; inspect retained coverage, do not erase blocked cells')
    queue(argparse.Namespace(matrix=args.output/'matrix.json',recipes=args.output/'materialized/recipes.json',
        vendor=args.vendor,select=[],output=args.output/'queue.json'))
    print(json.dumps(dict(status='REGISTERED_QUEUE_NOT_AUTHORIZED',variant=VARIANT,
        cells=len(selected),source_sha=source,output=str(args.output))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preparation-receipt',type=Path,required=True)
    p.add_argument('--preparation-plan',type=Path,required=True)
    p.add_argument('--data-directory',type=Path,required=True)
    p.add_argument('--deployments',type=Path,required=True)
    p.add_argument('--vendor',choices=['apple','nvidia','amd'],required=True)
    p.add_argument('--target-track',required=True)
    p.add_argument('--workload',action='append',help='Exact original workload ID subset; no inferred cases')
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())


if __name__=='__main__':main()
