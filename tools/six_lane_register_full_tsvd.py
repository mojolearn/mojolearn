#!/usr/bin/env python3
"""Materialize only the reviewed tsvd-full-v1 recipes after offline projection.

Uses existing accepted deployment receipts. Writes an unauthorized queue; does
not build, import a model, run a measurement, clear changes_frozen_race, or
replace original capped measurements. The owner authorizes/stages that queue.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
from six_lane_ab import ROOT, catalog, matrix, git, write, queue
from six_lane_full_variants import VARIANT, digest, validate_variant
from six_lane_materialize import materialize


def facts_from_projection(directory, cells, source, vendor):
    proposal_path = directory / 'variant-recipes.json'
    receipt_path = directory / 'projection-receipt.json'
    proposal = json.loads(proposal_path.read_text())
    if proposal.get('source_sha') != source:
        raise ValueError('Projection uses another source freeze')
    facts = {}
    for row in proposal['recipes']:
        if row['vendor'] != vendor:
            raise ValueError('Projection belongs to another vendor')
        key = row['variant_cell_key']
        if key not in cells:
            raise ValueError('Projected recipe is not a registered eligible matrix cell')
        cell = cells[key]
        reg = dict(variant=VARIANT, source_sha=source,
                   original_workload_id=row['original_workload_id'], original_cell_key=row['original_cell_key'],
                   variant_workload_id=row['variant_workload_id'], variant_cell_key=key,
                   proposal=dict(path=str(proposal_path),sha256=digest(proposal_path)),
                   projection_receipt=dict(path=str(receipt_path),sha256=digest(receipt_path)))
        work = {k: copy.deepcopy(row[k]) for k in ('harness','harness_sha256','lane','dataset','input_files',
                'data_directory','actual_shapes','estimator_settings_record','inference','dataset_version',
                'split','seed','output_paths','capture_limitation','repeated_operations')}
        work.update(runtime_vendor={'nvidia':'cuda','apple':'metal'}[vendor],
                    quality_gate_source=row['harness']+':quality/'+row['lane']+'; full-input opponent and complete state qualification pending',
                    intrinsic_cap_audit=dict(reviewed=True,unresolved=[],evidence=str(receipt_path)))
        fact = dict(source_sha=source,changes_frozen_race=True,registered_input_variant=reg,
                    dataset_sha256=row['dataset_sha256'],dimensions=row['actual_shapes'],
                    estimator_settings=row['estimator_settings_record'],timed_boundary=row['timing_policy'],
                    intrinsic_caps=[],full_dataset_coverage=True,workload=work,
                    resource_policy=dict(policy=row['resource_policy']),
                    coverage_resolutions={reason:dict(path=str(receipt_path),sha256=digest(receipt_path),
                        conclusion='Explicit source-registered full input variant, exact complete raw X-only projection, original settings and split; original capped recipe preserved. Quality, complete output/state identity and constituent reach remain unqualified.') for reason in cell['blockers']})
        validate_variant(fact,cell)
        if cell['workload_id'] in facts:raise ValueError('Duplicate projected variant')
        facts[cell['workload_id']] = fact
    return facts


def run(args):
    if args.output.exists():raise ValueError('Fresh registration output required; preserve prior attempts')
    if args.output.resolve().is_relative_to(ROOT):raise ValueError('Keep evidence outside source freeze')
    source=git('rev-parse','HEAD')
    if git('status','--porcelain','--untracked-files=all'):raise ValueError('Dirty source freeze')
    mat=matrix(catalog())
    eligible={c['key']:c for c in mat['cells'] if c.get('input_variant')==VARIANT and c['vendor']==args.vendor}
    facts={}
    for directory in args.projection:
        rows=facts_from_projection(directory.resolve(),eligible,source,args.vendor)
        if set(rows)&set(facts):raise ValueError('Duplicate dataset projection')
        facts.update(rows)
    if not facts:raise ValueError('No registered projected recipes')
    selected=[cell for cell in eligible.values() if cell['workload_id'] in facts]
    args.output.mkdir(parents=True)
    write(args.output/'matrix.json',dict(mat,cells=selected))
    write(args.output/'facts.json',facts)
    write(args.output/'registration.json',dict(variant=VARIANT,source_sha=source,original_cells_preserved=True,
          selected=[dict(key=c['key'],workload_id=c['workload_id'],original_cell_key=c['original_cell_key'],original_workload_id=c['original_workload_id']) for c in selected],
          capture_scope='Full input coverage only; output/state capture limitations remain explicit; no identity admission or promotion',execution_authorized=False))
    materialize(argparse.Namespace(matrix=args.output/'matrix.json',workloads=args.output/'facts.json',
                deployments=args.deployments,vendor=args.vendor,target_track=args.target_track,select=[],output=args.output/'materialized'))
    coverage=json.loads((args.output/'materialized/coverage.json').read_text())
    if coverage['blocked'] or coverage['materialized']!=len(selected):raise ValueError('Registered variant admission incomplete; inspect retained coverage')
    queue(argparse.Namespace(matrix=args.output/'matrix.json',recipes=args.output/'materialized/recipes.json',vendor=args.vendor,select=[],output=args.output/'queue.json'))
    print(json.dumps(dict(status='REGISTERED_QUEUE_NOT_AUTHORIZED',variant=VARIANT,cells=len(selected),source_sha=source,output=str(args.output))))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--projection',type=Path,action='append',required=True)
    p.add_argument('--deployments',type=Path,required=True)
    p.add_argument('--vendor',choices=['nvidia','apple'],required=True)
    p.add_argument('--target-track',required=True)
    p.add_argument('--output',type=Path,required=True)
    run(p.parse_args())

if __name__=='__main__':main()
