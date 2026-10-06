#!/usr/bin/env python3
"""Plan/project a separately named full-input TSVD recipe; never run a model.

`plan` reads committed source and small retained JSON metadata only. `project`
later streams one accepted raw big-X member into a fresh X-only archive under
the owning worker's existing exclusive lock. Both emit reviewable documents,
never queues, build commands, board admissions or execution authorization.
"""
# cpu-route: offline archive/header/file handling before estimator runtime.
import argparse
import ast
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
VARIANT = 'tsvd-full-v1'
CHUNK = 4 * 1024 * 1024
SOURCES = ('tools/bench_board_algos.py', 'tools/bench_board_more.py',
           'tools/bench_board_params.py', 'tools/classical_two_datasets.py',
           'python/mojolearn/_expansion_decomp.py', 'python/mojolearn/decomposition.py')
CLASSES = {'tsvd': 'TruncatedSVD', 'fastica': 'FastICA',
           'gaussian-rp': 'GaussianRandomProjection', 'sparse-rp': 'SparseRandomProjection',
           'incremental-pca': 'IncrementalPCA'}
LOCKS = {'nvidia': '/root/six-lane-full-ab-20261006/device-measurement.lock',
         'amd': '/root/six-lane-full-ab-20261006/device-measurement.lock',
         'apple': '/Users/ec2-user/mojolearn-full-f867b50e8/gpu.lock'}


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(CHUNK), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value):
    return sha_bytes(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


def write(path, value):
    # Exclusive creation also protects partial attempts and historical receipts.
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def git(*args):
    return subprocess.check_output(['git', '-C', str(ROOT), *args])


def defaults(source, name):
    cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == name)
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    args = init.args
    result = {a.arg: ast.literal_eval(v) for a, v in zip(args.args[-len(args.defaults):], args.defaults)} if args.defaults else {}
    result.update({a.arg: ast.literal_eval(v) for a, v in zip(args.kwonlyargs, args.kw_defaults)})
    return result


def assigned(source, name):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
    return node.value


def registration(source, lane):
    node = next(n.value for n in ast.parse(source).body
                if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Name) and n.value.func.id == '_add'
                and isinstance(n.value.args[0], ast.Constant) and n.value.args[0].value == lane)
    fields = {kw.arg: kw.value for kw in node.keywords}
    if ast.literal_eval(fields['block']) != 'tsvd':
        raise ValueError('Registration no longer uses TSVD inputs')
    params = fields['params']
    if not isinstance(params, ast.Call) or not isinstance(params.func, ast.Name) or params.func.id != 'dict':
        raise ValueError('Saved params are no longer literal dict kwargs')
    values = {}
    for kw in params.keywords:
        values[kw.arg] = 7 if isinstance(kw.value, ast.Name) and kw.value.id == 'SEED' else ast.literal_eval(kw.value)
    return values


def settings(sources, lane):
    ignore = assigned(sources['tools/bench_board_params.py'], 'IGNORE')
    if not isinstance(ignore, ast.Call) or ignore.func.id != 'frozenset':
        raise ValueError('Unreviewed parameter ignore schema')
    ignore = frozenset(ast.literal_eval(ignore.args[0]))
    for path in ('tools/bench_board_params.py', 'tools/bench_board_algos.py', 'tools/bench_board_more.py'):
        if ast.literal_eval(assigned(sources[path], 'SEED')) != 7:
            raise ValueError('Saved lane seed changed: ' + path)
    if lane == 'tsvd':
        more = sources['tools/bench_board_more.py']
        if ast.literal_eval(assigned(more, 'TSVD_ROWS')) != 1_000_000:
            raise ValueError('Original TSVD preparation contract changed')
        p = defaults(sources['python/mojolearn/decomposition.py'], CLASSES[lane])
        p.update(n_components=ast.literal_eval(assigned(more, 'TSVD_COMPONENTS')),
                 algorithm='covariance_eigh', n_iter=5, n_oversamples=10, tol=0.0, random_state=7)
        # Explicit source signature check: changes require review, not guessed settings.
        tree = ast.parse(more)
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                 and isinstance(n.func.value, ast.Name) and n.func.value.id == 'ml' and n.func.attr == 'TruncatedSVD']
        call, = calls
        saved = {kw.arg: kw.value for kw in call.keywords}
        for k in ('algorithm', 'n_iter', 'n_oversamples', 'tol'):
            if ast.literal_eval(saved[k]) != p[k]:
                raise ValueError('More TSVD saved constructor changed: ' + k)
        if not isinstance(saved['n_components'], ast.Name) or saved['n_components'].id != 'TSVD_COMPONENTS':
            raise ValueError('More TSVD component contract changed')
        if not isinstance(saved['random_state'], ast.Name) or saved['random_state'].id != 'SEED':
            raise ValueError('More TSVD seed contract changed')
    else:
        kwargs = registration(sources['tools/bench_board_algos.py'], lane)
        p = {} if lane == 'randomized-svd' else defaults(sources['python/mojolearn/_expansion_decomp.py'], CLASSES[lane])
        p.update(kwargs)
    return dict(__record__=True, library='mojolearn',
                source='declared' if lane == 'randomized-svd' else 'get_params',
                params={k: v for k, v in p.items() if k not in ignore})


def metadata_inputs(path):
    rows = json.loads(path.read_text())
    result = {}
    for row in rows:
        ds, meta = row['dataset'], row['metadata']
        if ds not in ('taxi', 'istella') or ds in result:
            raise ValueError('Expected one retained Taxi and Istella record')
        arr = meta['arrays']['X']
        n, features = arr['shape']
        if (meta['block'] != 'big' or meta['scaling'] != 'none' or arr['dtype'] != 'float32'
                or meta['fit_rows'] != [0, n] or meta['fit_rows_available'] != n or n <= 1 or features <= 0):
            raise ValueError('Expected complete raw regression-population big X')
        if 'regression=True' not in meta['loader']:
            raise ValueError('Missing original regression-population loader provenance')
        files = {Path(f['path']).suffix: f for f in row['input_files']}
        if len(files) != 2 or set(files) != {'.npz', '.json'}:
            raise ValueError('Exact source NPZ and JSON required')
        result[ds] = dict(dataset=ds, source_input_files=row['input_files'], source_metadata=meta,
                          X=arr, expanded_shapes={'X': [max(n - n // 10, 1), features], 'Xq': [n // 10, features]},
                          more_shapes={'X': [n, features]}, output_keys=['X'], output_npz_sha256=None,
                          output_json_sha256=None, output_array_sha256=None, status='NOT_PROJECTED')
    if set(result) != {'taxi', 'istella'}:
        raise ValueError('Both retained full input datasets must be represented')
    return result


def plan(args):
    source = git('rev-parse', args.source_sha + '^{commit}').decode().strip()
    sources = {p: git('show', source + ':' + p).decode() for p in SOURCES}
    matrix = json.loads(git('show', source + ':experiments/six_lane_integration/matrix.json'))
    configs = {c['id']: c for c in matrix['configurations']}
    inputs = metadata_inputs(args.source_facts)
    doc = dict(schema='mojolearn.full-tsvd-plan/1', variant=VARIANT, source_sha=source,
               projector_sha256=digest(Path(__file__)), source_hashes={p: sha_bytes(s.encode()) for p, s in sources.items()},
               source_facts=dict(path=str(args.source_facts), sha256=digest(args.source_facts)), inputs=inputs,
               execution_authorized=False, status='PLANNED_NO_DATA_READS', dataset_arrays_read=0,
               preserve_original='Original 1M-row recipes, estimators, opponent roster, results and active freezes remain intact.',
               timing_policy='Preparation/load, existing lane split, construction, operation, required synchronization and consumed outputs; hashes and independent quality outside timer.',
               locks=LOCKS, scheduling={'nvidia': 'After current expanded/GMM/PLS chain and a new reviewed main freeze.',
                                       'amd': 'Only after native gfx receipts are accepted and source/input preparation is serialized under its canonical lock.',
                                       'apple': 'Only after queued CPU14 completes and a new reviewed main freeze; do not delay its quality budget.'},
               retained_deployments={}, recipes=[], unresolved=['New full-input variant registration and root review; no automatic admission.',
                    'Complete fitted-state/returned-output identity scope where noted.',
                    'AMD and default/PTX admission requires accepted full bindings; this planner does not establish artifact availability.'])
    for vendor in ('nvidia', 'amd', 'apple'):
        p = getattr(args, vendor + '_deployments')
        doc['retained_deployments'][vendor] = (dict(path=str(p), sha256=digest(p), contents=json.loads(p.read_text()),
            scope='Caller-supplied retained metadata. No binding opened, rehashed, compiled or requalified here.') if p else
            dict(status='PENDING_ATTACHMENT', scope='Reuse owning peer accepted A/B package receipts; do not build or infer coverage.'))
        cfg = 'AF.X.complete-proposed' if vendor == 'apple' else 'I.X.complete-proposed'
        for ds in ('taxi', 'istella'):
            for lane in ('tsvd', 'incremental-pca', 'randomized-svd', 'gaussian-rp', 'sparse-rp', 'fastica'):
                prefix = ('more:' if lane == 'tsvd' else 'expanded:') if vendor != 'apple' else ('classical2/' if lane == 'tsvd' else 'algos/')
                original = prefix + lane + '@dataset=' + ds
                cells = [c for c in matrix['cells'] if c['configuration'] == cfg and c['vendor'] == vendor and c['workload_id'] == original]
                if not cells:  # No invented Apple/other vendor race or roster entry.
                    continue
                cell, = cells
                more = lane == 'tsvd'
                harness = 'tools/bench_board_more.py' if more else 'tools/bench_board_algos.py'
                paths = ['$.components'] if more or lane == 'randomized-svd' else ['$.pred']
                if lane == 'incremental-pca':
                    paths += ['$.components', '$.mean']
                scope = ('Fitted model-state contract pending; components alone do not establish state identity.' if more else
                         'U and singular values omitted by existing function runner; returned-output coverage incomplete; no fitted model owner.' if lane == 'randomized-svd' else
                         'Complete fitted-state contract pending; full transformed query outputs are not full model-state identity.')
                params = settings(sources, lane)
                inference = 'not_applicable' if more or lane == 'randomized-svd' else 'separate'
                variant_id = original + '@input=' + VARIANT
                doc['recipes'].append(dict(variant_workload_id=variant_id, original_workload_id=original,
                    original_cell_key=cell['key'], variant_cell_key=canonical_hash([cfg, vendor, variant_id])[:20],
                    configuration=cfg, configuration_sha256=canonical_hash(configs[cfg]),
                    implementation_ids=cell['implementation_ids'], mode=cell['mode'], vendor=vendor, dataset=ds,
                    lane=lane, harness=harness, harness_sha256=doc['source_hashes'][harness],
                    estimator_settings_record=params, actual_shapes=inputs[ds]['more_shapes' if more else 'expanded_shapes'],
                    inference=inference, output_paths=paths, output_dtype='float64',
                    required_bindings=['bindings/_mojolearn_estimators.mojo' if more else 'bindings/_mojolearn_x_decomp.mojo'],
                    split=(dict(fit=[0, inputs[ds]['X']['shape'][0]], held_out='none; original loader test split not used') if more else
                           dict(fit=[0, inputs[ds]['expanded_shapes']['X'][0]], query=[inputs[ds]['expanded_shapes']['X'][0], inputs[ds]['X']['shape'][0]],
                                rule='Original cut=max(N-N//10,1), X-only archive; no loader Xq; randomized-svd does not use query arrays')),
                    seed=7, original_preparation_cap=1_000_000, new_preparation_cap=None,
                    capture_limitation=scope, source_coverage_pending=cell.get('source_coverage_pending', []),
                    status='DRAFT_FULL_INPUT_VARIANT_NOT_ADMITTED', changes_frozen_race=True,
                    registration='Separate variant must be reviewed/registered; do not remove this guard to route it as the original 1M race.',
                    qualification='UNMEASURED_QUALITY_PENDING_IDENTITY_PENDING', input_files=[], dataset_sha256=None))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write(args.output, doc)
    print(json.dumps(dict(status=doc['status'], source_sha=source, recipes=len(doc['recipes']), output=str(args.output), array_reads=0)))


@contextmanager
def worker_lock(path):
    # Never create a second lock accidentally; use the already established one.
    with path.open('r+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def stamp(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def exact(stream, size):
    value = stream.read(size)
    if len(value) != size:
        raise ValueError('Truncated NPY member')
    return value


def npy_header(stream):
    prefix = exact(stream, 8)
    if prefix[:6] != b'\x93NUMPY' or prefix[6:] not in (b'\x01\x00', b'\x02\x00', b'\x03\x00'):
        raise ValueError('Unsupported NPY format')
    width = 2 if prefix[6] == 1 else 4
    length = exact(stream, width)
    count = struct.unpack('<H' if width == 2 else '<I', length)[0]
    if count > 65536:
        raise ValueError('Unexpected oversized NPY header')
    header = exact(stream, count)
    meta = ast.literal_eval(header.decode('utf-8' if prefix[6] == 3 else 'latin1'))
    if (set(meta) != {'descr', 'fortran_order', 'shape'} or meta['fortran_order'] is not False
            or meta['descr'] not in ('<f4', '=f4') or (meta['descr'] == '=f4' and sys.byteorder != 'little')
            or not isinstance(meta['shape'], tuple) or len(meta['shape']) != 2
            or any(type(n) is not int or n <= 0 for n in meta['shape'])):
        raise ValueError('Expected C-order little-endian float32 matrix; do not convert values')
    return prefix + length + header, meta['shape']


def project_x(source, destination, expected):
    with zipfile.ZipFile(source) as archive:
        members = [info for info in archive.infolist() if info.filename == 'X.npy']
        if len(members) != 1:
            raise ValueError('Expected one unambiguous X.npy source member')
        with archive.open(members[0]) as stream:
            header, shape = npy_header(stream)
            if list(shape) != expected['shape']:
                raise ValueError('Source member shape differs from accepted metadata')
            h = hashlib.sha256(b'float32' + str(shape).encode())
            copied = 0
            info = zipfile.ZipInfo('X.npy', date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            with zipfile.ZipFile(destination, 'x', allowZip64=True) as output:
                with output.open(info, 'w', force_zip64=True) as target:
                    target.write(header)
                    for chunk in iter(lambda: stream.read(CHUNK), b''):
                        target.write(chunk)
                        h.update(chunk)
                        copied += len(chunk)
            if copied != shape[0] * shape[1] * 4 or h.hexdigest() != expected['sha256']:
                raise ValueError('X body size/hash differs; retained partial output is not admitted')
    return dict(dtype='float32', shape=list(shape), sha256=h.hexdigest(), encoding='C-order little-endian float32',
                hash_encoding='sha256(str(dtype) + str(shape_tuple) + C-order array bytes); classical_two_datasets.sha256_array')


def project(args):
    doc = json.loads(args.plan.read_text())
    if doc.get('schema') != 'mojolearn.full-tsvd-plan/1' or doc.get('variant') != VARIANT:
        raise ValueError('Unknown reviewed plan schema')
    if doc['projector_sha256'] != digest(Path(__file__)):
        raise ValueError('Projector changed since planning; create a new plan')
    if doc['source_sha'] != git('rev-parse', 'HEAD').decode().strip():
        raise ValueError('Use the new reviewed source freeze named by the plan')
    if git('status', '--porcelain', '--untracked-files=all').strip():
        raise ValueError('Source worktree is dirty; preserve active freeze and use a reviewed clean checkout')
    if args.lock_file != Path(doc['locks'][args.vendor]):
        raise ValueError('Use the canonical existing worker lock recorded in the plan')
    if args.output.resolve().is_relative_to(ROOT.resolve()):
        raise ValueError('Projection outputs must be outside the frozen source checkout')
    # The root must also schedule Apple after CPU14 and native after its chain;
    # acquiring a momentarily free lock is not scheduling/measurement approval.
    with worker_lock(args.lock_file):
        args.output.mkdir(parents=True, exist_ok=False)
        receipt = dict(schema='mojolearn.full-tsvd-projection/1', status='STARTED',
                       source_sha=doc['source_sha'], projector_sha256=doc['projector_sha256'],
                       plan=dict(path=str(args.plan), sha256=digest(args.plan)), dataset=args.dataset,
                       vendor=args.vendor, variant=VARIANT, lock=str(args.lock_file),
                       model_executions=0, compilations=0, execution_authorized=False)
        try:
            record = doc['inputs'][args.dataset]
            files = []
            for item in record['source_input_files']:
                path = args.input_directory / Path(item['path']).name
                before = stamp(path)
                if digest(path) != item['sha256'] or stamp(path) != before:
                    raise ValueError('Source input hash/metadata changed: ' + str(path))
                files.append(dict(path=str(path), sha256=item['sha256'], stamp=before))
            npz, = [f for f in files if f['path'].endswith('.npz')]
            meta, = [f for f in files if f['path'].endswith('.json')]
            original = json.loads(Path(meta['path']).read_text())
            if original != record['source_metadata']:
                raise ValueError('Source JSON differs from retained metadata')
            archive = args.output / ('tsvd-' + args.dataset + '.npz')
            arr = project_x(Path(npz['path']), archive, record['X'])
            if any(stamp(Path(f['path'])) != tuple(f['stamp']) for f in files):
                raise ValueError('Source inputs changed during projection')
            block = dict(block='tsvd', dataset=args.dataset, seed=7, variant=VARIANT,
                         dataset_version=VARIANT + '; complete regression training population; raw X-only projection',
                         loader=original['loader'], scaling='none', smoke_max_rows=None,
                         source_input_files=[{k: f[k] for k in ('path', 'sha256')} for f in files],
                         source_sha=doc['source_sha'], source_metadata=record['source_metadata'],
                         arrays={'X': {k: arr[k] for k in ('dtype', 'shape', 'sha256')}},
                         fit_rows=[0, arr['shape'][0]], fit_rows_available=arr['shape'][0],
                         full_dataset_coverage=True, original_preparation_cap=1_000_000, intrinsic_caps=[],
                         sentinel_cells_replaced=original.get('sentinel_cells_replaced'),
                         rule='Exact X.npy byte projection; no rows/columns/value conversion; no original Xq/labels/init',
                         written=datetime.now(timezone.utc).isoformat())
            metadata = args.output / ('tsvd-' + args.dataset + '.json')
            write(metadata, block)
            output_files = [dict(path=str(p), sha256=digest(p)) for p in (archive, metadata)]
            recipes = []
            for template in doc['recipes']:
                if template['dataset'] != args.dataset or template['vendor'] != args.vendor:
                    continue
                row = copy.deepcopy(template)
                row.update(status='PROJECTED_PENDING_ROOT_REVIEW_AND_VARIANT_REGISTRATION', input_files=output_files,
                           dataset_sha256=canonical_hash(block['arrays']), dataset_version=block['dataset_version'],
                           data_directory=str(args.output), projected_array=arr,
                           retained_deployment_metadata=doc['retained_deployments'][args.vendor],
                           resource_policy='Serial arms on full allocated worker, actual pools/cgroup recorded by the future scored execution',
                           repeated_operations=1, timing_policy=doc['timing_policy'],
                           quality_policy='Existing task metrics/gates, matching full-input opponent evidence pending; no historical ratio substitution')
                recipes.append(row)
            if (doc['source_sha'] != git('rev-parse', 'HEAD').decode().strip()
                    or git('status', '--porcelain', '--untracked-files=all').strip()):
                raise ValueError('Source freeze changed during projection; preserve attempt without admission')
            write(args.output / 'variant-recipes.json', dict(schema='mojolearn.full-tsvd-recipes/1',
                  source_sha=doc['source_sha'], execution_authorized=False, recipes=recipes,
                  warning='Separate full variant; existing master frozen-race guards intentionally remain enabled. Root reviews registration/deployment before launch.'))
            receipt.update(status='PROJECTED_NOT_ADMITTED', source_input_files=files, output_files=output_files,
                           X=arr, recipes=len(recipes), active_runs_modified=False)
        except Exception as exc:
            receipt.update(status='FAILED', error=repr(exc))
            raise
        finally:
            write(args.output / 'projection-receipt.json', receipt)
        print(json.dumps({k: receipt[k] for k in ('status', 'dataset', 'vendor', 'recipes')}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('plan', help='Read source/JSON only; no NPZ access')
    p.add_argument('--source-sha', required=True)
    p.add_argument('--source-facts', type=Path, required=True)
    p.add_argument('--nvidia-deployments', type=Path)
    p.add_argument('--apple-deployments', type=Path)
    p.add_argument('--amd-deployments', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(fn=plan)
    p = sub.add_parser('project', help='Explicit later offline data projection under existing worker lock; no models/jobs')
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--dataset', choices=('taxi', 'istella'), required=True)
    p.add_argument('--vendor', choices=('nvidia', 'amd', 'apple'), required=True)
    p.add_argument('--input-directory', type=Path, required=True)
    p.add_argument('--lock-file', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.set_defaults(fn=project)
    args = parser.parse_args()
    args.fn(args)


if __name__ == '__main__':
    main()
