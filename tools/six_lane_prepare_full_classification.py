#!/usr/bin/env python3
"""Plan or explicitly prepare new full classification archives, never models.

The plan reads source and retained small JSON only. Preparation is a separate,
reviewed operation after both CPU queues terminate, under the canonical lock.
Historical capped archives and benchmark recipes are never replaced.
"""
# cpu-route: explicit offline dataset/file preparation before estimator runtime.
import argparse
import ast
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'classification-full-v1'
LOCK = '/Users/ec2-user/mojolearn-full-f867b50e8/gpu.lock'
BASE = '/Volumes/MojoPerfBuild20261005/six-lane-full-ab-20261006/apple'
WAIT = [BASE + '/opponent-quality/' + p + '/status.json'
        for p in ('run-01', 'run-01-selector-repair')]
TERMINAL = {'COMPLETE', 'COMPLETE_WITH_UNRESOLVED_CELLS',
            'BLOCKED_FULL_INPUT_METADATA', 'FAILED_CONTROLLER'}
SOURCES = ('tools/bench_board_algos.py', 'tools/bench_board_more.py',
           'tools/classical_two_datasets.py', 'tools/speed_gbdt_arm.py',
           'tools/bench_board_params.py')


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode()).hexdigest()


def write(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def git(*args):
    result = subprocess.check_output(['git', '-C', str(ROOT), *args], text=True)
    return result if args[0] == 'show' else result.strip()


def registration(source, line):
    """Retain the exact original registration, including enclosing _add loops."""
    node = next(n for n in ast.parse(source).body
                if n.lineno <= line <= getattr(n, 'end_lineno', n.lineno))
    return ast.get_source_segment(source, node)


def plan(args):
    sha = git('rev-parse', args.source_sha + '^{commit}')
    sources = {p: git('show', sha + ':' + p) for p in SOURCES}
    audit = json.loads(args.audit.read_text())
    if text_hash(sources['tools/bench_board_algos.py']) != audit['harness_sha256']:
        raise ValueError('Refresh eligibility audit after harness source changes')
    inventory = json.loads(args.inventory.read_text())
    owner_deferral = None
    if args.owner_deferral:
        decision = json.loads(args.owner_deferral.read_text())
        if (decision.get('status') != 'DEFERRED_BY_OWNER'
                or decision.get('owned_children_surviving') != []
                or decision.get('machine_lock_exclusive_probe') is not True
                or decision.get('budget_reset') is not False):
            raise ValueError('Owner deferral lacks retained safe-stop evidence')
        owner_deferral = dict(path=decision['evidence'] + '/deferral.json',
                              sha256=digest(args.owner_deferral), reason=decision['reason'])
    rows = [r for r in audit['rows'] if r['vendor'] == 'apple'
            and r['category'] == 'CORRECT_FULL_POPULATION_PREPROCESSING_ARCHIVE_REQUIRED']
    if not rows or any(r['declared_sub'] for r in rows):
        raise ValueError('Only reviewed lanes without intrinsic lane subsets are eligible')
    recipes = []
    for row in rows:
        if row['registration_source'] != 'tools/bench_board_algos.py':
            raise ValueError('Unreviewed harness family')
        # A stale line number must not silently describe a different estimator.
        src = sources[row['registration_source']]
        snippet = registration(src, row['registration_line'])
        if repr(row['slug']) not in snippet and json.dumps(row['slug']) not in snippet:
            raise ValueError('Registration moved or changed; refresh source audit: ' + row['slug'])
        block = {'countclf': 'cls', 'nonneg': 'cls'}.get(row['block'], row['block'])
        if block not in ('cls', 'raw', 'cat'):
            raise ValueError('Unreviewed preparation block: ' + block)
        recipes.append(dict(original_workload_id=row['workload_id'],
            variant_workload_id=row['workload_id'] + '@input=' + VARIANT,
            configuration=row['configuration'], block=block, lane=row['slug'],
            dataset=row['workload_id'].split('@dataset=')[1].split('@')[0],
            source_registration=snippet, source_registration_sha256=text_hash(snippet),
            estimator_settings_policy='Unchanged original harness constructors, defaults, derived settings and lane_arrays transforms; exact observed settings must be captured and admitted by future worker.',
            declared_sub=row['declared_sub'], changes_frozen_race=True,
            status='PLANNED_VARIANT_NOT_REGISTERED_NOT_MEASURED',
            pending=['Distinct full-input variant registration', 'Exact shape/settings and complete output scope admission',
                     'Retained applicable A/B artifacts', 'Independent quality and identity acceptance']))
    archives = [r for r in inventory['archives'] if r['path'].endswith(
        ('/taxi/taxi_speed.npz', '/istella/istella_speed.npz'))]
    if len(archives) != 2:
        raise ValueError('Need exactly one retained original archive per dataset')
    # Counts attest dataset identity; they never select an implementation or cap data.
    population = dict(taxi=dict(train=4110786, query=500000, numeric_features=11,
        rule='card-paid plausible Jan/Feb2024 trips; tip >= 0.2*fare; last500000 FILTERED trips held out'),
        istella=dict(train=2043304, query=500000, numeric_features=220,
        rule='all train.txt rows, first500000 of separate test.txt; relevance > 0'))
    doc = dict(schema='mojolearn.full-classification-plan/1', variant=VARIANT,
        status='PLANNED_NO_ARRAY_READS', source_sha=sha, helper_sha256=digest(__file__),
        source_hashes={p: text_hash(s) for p, s in sources.items()},
        input_metadata=dict(audit_path=str(args.audit), audit_sha256=digest(args.audit),
                            inventory_path=str(args.inventory), inventory_sha256=digest(args.inventory)),
        source_archives=archives, archive_sha256_status='Computed and recorded only during reviewed preparation; not inferred from filename or size',
        population=population, canonical_lock=LOCK, prerequisite_status_paths=WAIT,
        owner_deferral=owner_deferral,
        execution_authorized=False, dataset_arrays_read=0, recipes=recipes,
        preserved='Original capped1M/100k inputs, settings, recipes, results and active source freezes',
        preparation='Original loaders, sentinel cleanup, full-train standardization; raw unscaled; categorical original mapping fitted on full train. No estimator imports/fits/builds.',
        output='Fresh variant directory containing conventional block names solely for original loader compatibility; sidecars carry distinct variant and original links. Never point an old race at these files.',
        scheduling='Root review required; BOTH CPU14 and failed-only repair must terminate, then canonical GPU lock must be free. No queues or timers emitted.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, doc)
    print(json.dumps(dict(status=doc['status'], recipes=len(recipes), output=str(args.output), arrays_read=0)))


def load_tool(name):
    spec = importlib.util.spec_from_file_location('full_classification_' + name, ROOT / 'tools' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prerequisites(paths, owner_deferral=None):
    if owner_deferral:
        if digest(owner_deferral['path']) != owner_deferral['sha256']:
            raise ValueError('Owner deferral evidence changed')
        decision = json.loads(Path(owner_deferral['path']).read_text())
        if (decision.get('status') != 'DEFERRED_BY_OWNER'
                or decision.get('owned_children_surviving') != []
                or decision.get('machine_lock_exclusive_probe') is not True
                or decision.get('budget_reset') is not False):
            raise ValueError('Owner deferral is not an approved safe stop')
        previous = json.loads(Path(BASE + '/next-full-continuation/status.json').read_text())
        if previous.get('status') != 'COMPLETE' or sum(g.get('completed', 0) for g in previous.get('groups', [])) != 6:
            raise ValueError('Owner-priority preceding six A/B cells must complete first')
    result = []
    for path in paths:
        state = json.loads(Path(path).read_text())
        if state.get('status') not in TERMINAL and not (owner_deferral and state.get('status') == 'DEFERRED_BY_OWNER'):
            raise ValueError('Existing opponent queue is not terminal: ' + path)
        result.append(dict(path=path, sha256=digest(path), status=state['status']))
    return result


def prepare(args):
    doc = json.loads(args.plan.read_text())
    if digest(args.plan) != args.reviewed_plan_sha256:
        raise ValueError('Explicit reviewed plan digest differs')
    if doc['variant'] != VARIANT or doc['source_sha'] != git('rev-parse', 'HEAD'):
        raise ValueError('Prepare from the exact reviewed frozen checkout')
    if git('status', '--porcelain', '--untracked-files=no'):
        raise ValueError('Frozen checkout has tracked changes')
    if doc['helper_sha256'] != digest(__file__):
        raise ValueError('Helper changed since review')
    for path, expected in doc['source_hashes'].items():
        if text_hash(git('show', 'HEAD:' + path)) != expected:
            raise ValueError('Source contract changed: ' + path)
    if doc['canonical_lock'] != LOCK or doc['prerequisite_status_paths'] != WAIT:
        raise ValueError('Canonical queue prerequisites cannot be redirected')
    if args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Preparation outputs must be outside the frozen checkout')
    prerequisites(WAIT, doc.get('owner_deferral'))
    with open(LOCK, 'r+b') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        states = prerequisites(WAIT, doc.get('owner_deferral'))
        # All source archive reads and CPU preprocessing occur under the same lock.
        import numpy as np
        ctd = load_tool('classical_two_datasets')
        algos = load_tool('bench_board_algos')
        os.environ['GBM_BENCH_DATA'] = str(args.data_root)
        harness = ctd._module('speed_gbdt_arm')
        args.output.mkdir(parents=True, exist_ok=False)
        manifest = dict(schema='mojolearn.full-classification-preparation/1', variant=VARIANT,
            source_sha=doc['source_sha'], helper_sha256=digest(__file__), plan_sha256=digest(args.plan),
            plan=dict(path=str(args.plan.resolve()), sha256=digest(args.plan)),
            source_hashes=doc['source_hashes'], owner_deferral=doc.get('owner_deferral'),
            prerequisite_states=states, lock=LOCK, started_at=time.time(), status='PREPARING',
            model_executions=0, builds=0, source_archives=[], blocks=[], recipes=doc['recipes'])
        write(args.output / 'started.json', manifest)
        try:
            for ds in ('taxi', 'istella'):
                archive = args.data_root / ds / (ds + '_speed.npz')
                retained, = [r for r in doc['source_archives'] if r['path'].endswith('/' + ds + '/' + ds + '_speed.npz')]
                if archive.stat().st_size != retained['bytes']:
                    raise ValueError('Retained source archive size changed: ' + ds)
                # Require an intact cached NPZ; never invoke download/parquet recovery.
                with zipfile.ZipFile(archive) as z:
                    if z.testzip() is not None:
                        raise ValueError('Damaged cached archive: ' + ds)
                before = digest(archive)
                source_record = dict(path=str(archive), bytes=archive.stat().st_size, sha256=before)
                manifest['source_archives'].append(source_record)
                xtr, xte, ytr, yte = algos._tab_loader(ctd, harness, ds, 'cls')
                expected = doc['population'][ds]
                if list(xtr.shape) != [expected['train'], expected['numeric_features']] or list(xte.shape) != [expected['query'], expected['numeric_features']]:
                    raise ValueError('Original classification population differs: ' + ds)
                source_shapes = dict(X=list(xtr.shape), Xq=list(xte.shape), y=list(ytr.shape), yq=list(yte.shape))
                blocks = sorted({r['block'] for r in doc['recipes'] if r['dataset'] == ds})
                for block in blocks:
                    rec = dict(rule='tools/six_lane_prepare_full_classification.py prepare',
                        variant=VARIANT, changes_frozen_race=True, dataset=ds, block=block,
                        source_sha=doc['source_sha'], source_archive=source_record,
                        source_dimensions=source_shapes, population=expected['rule'],
                        original_preparation_caps=dict(train=1000000, query=100000),
                        preparation_caps=dict(train=None, query=None), full_dataset_coverage=True,
                        seed=algos.SEED, fit_rows='all original loader training rows',
                        eval_rows='all original loader query rows',
                        admission='INPUTS_ONLY; distinct recipe/settings/output/artifact/quality admission remains pending')
                    if block == 'cat' and ds == 'taxi':
                        # Original categorical loader densifies on all train rows, before sampling.
                        cat = harness.load_taxi('shipped', categorical=True)
                        X = np.asarray(cat.X_train)[:, algos.TAXI_CAT_COLUMNS]
                        Xq = np.asarray(cat.X_test)[:, algos.TAXI_CAT_COLUMNS]
                        if not np.array_equal(cat.y_train, ytr) or not np.array_equal(cat.y_test, yte):
                            raise ValueError('Categorical and numeric populations differ')
                        rec['columns'] = list(algos.TAXI_CAT_COLUMNS)
                        del cat
                    else:
                        X, bad = ctd.clean_sentinel(xtr)
                        Xq, badq = ctd.clean_sentinel(xte)
                        rec['sentinel_cells_replaced'] = dict(X=bad, Xq=badq)
                        if block == 'cls':
                            X, Xq = ctd.standardize(X, Xq)
                            rec['scaling'] = 'original float64 mean/std fitted on all train rows'
                        elif block == 'raw':
                            rec['scaling'] = 'none; original sentinel cleanup only'
                        else:
                            # Exact saved categorical feature recipe, not a numerical dispatch rule.
                            cols = np.argsort(-X.var(axis=0), kind='stable')[:8]
                            fit_codes = np.empty((X.shape[0], len(cols)))
                            query_codes = np.empty((Xq.shape[0], len(cols)))
                            edges_record = []
                            for j, col in enumerate(cols):
                                edges = np.quantile(X[:, col].astype(np.float64), np.linspace(0, 1, 17)[1:-1])
                                fit_codes[:, j] = np.searchsorted(edges, X[:, col], side='right')
                                query_codes[:, j] = np.searchsorted(edges, Xq[:, col], side='right')
                                edges_record.append(edges.tolist())
                            X, Xq = fit_codes, query_codes
                            rec.update(columns=cols.tolist(), quantile_edges=edges_record,
                                       scaling='original8 highest-variance fulltrain columns;16 quantile codes')
                    arrays = {k: np.ascontiguousarray(v, dtype=np.float32)
                              for k, v in dict(X=X, Xq=Xq, y=ytr, yq=yte).items()}
                    if block == 'cat':
                        # Same data-derived setting as _derived_params(categorical-nb).
                        # Retain it while the full matrices are resident, avoiding a second read.
                        minimum = np.maximum(arrays['X'].max(axis=0), arrays['Xq'].max(axis=0)).astype(np.int64) + 1
                        rec['derived_parameters'] = {'categorical-nb': {'min_categories': dict(
                            values=minimum.tolist(), dtype=str(minimum.dtype), shape=list(minimum.shape),
                            sha256=ctd.sha256_array(minimum), original_representation='list[int]')}}
                    name = block + '-' + ds
                    ctd._write_block(str(args.output), name, arrays, rec)
                    manifest['blocks'].append(dict(name=name, arrays=rec['arrays'],
                        npz_sha256=digest(args.output / (name + '.npz')),
                        sidecar_sha256=digest(args.output / (name + '.json')),
                        input_files=[dict(path=str((args.output / (name + ext)).resolve()),
                                          sha256=digest(args.output / (name + ext))) for ext in ('.npz', '.json')]))
                    del X, Xq, arrays
                if digest(archive) != before:
                    raise ValueError('Source archive changed during preparation: ' + ds)
                del xtr, xte, ytr, yte
            if doc['source_sha'] != git('rev-parse', 'HEAD') or git('status', '--porcelain', '--untracked-files=no'):
                raise ValueError('Source freeze changed during preparation; retain failed attempt')
            manifest.update(status='PREPARED_NOT_MEASURED_NOT_ADMITTED', finished_at=time.time())
            write(args.output / 'receipt.json', manifest)
        except BaseException as exc:
            manifest.update(status='FAILED_PREPARATION', error=repr(exc), finished_at=time.time())
            write(args.output / 'failure.json', manifest)
            raise
    print(json.dumps(dict(status=manifest['status'], blocks=len(manifest['blocks']), output=str(args.output))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='command', required=True)
    a = sub.add_parser('plan')
    a.add_argument('--source-sha', required=True)
    a.add_argument('--audit', type=Path, required=True)
    a.add_argument('--inventory', type=Path, required=True)
    a.add_argument('--owner-deferral', type=Path, help='Explicit retained owner reprioritization; also requires prior six A/B cells complete')
    a.add_argument('--output', type=Path, required=True)
    a = sub.add_parser('prepare')
    a.add_argument('--plan', type=Path, required=True)
    a.add_argument('--reviewed-plan-sha256', required=True)
    a.add_argument('--data-root', type=Path, required=True)
    a.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    (plan if args.command == 'plan' else prepare)(args)


if __name__ == '__main__':
    main()
