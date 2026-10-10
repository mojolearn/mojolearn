#!/usr/bin/env python3
"""Opt-in installed-artifact guard and actual benchmark worker receipts.

No kernel or measured call is wrapped. MemProbe registers an exit callback;
receipts are collected only after the worker has finished its timed work.
Legacy native runs stay readable; guarded runs cannot resume legacy evidence.
"""
import atexit
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

SCHEMA = 'mojolearn-board-artifacts/1'
ENV = 'MOJOLEARN_BOARD_ARTIFACT_MANIFEST'
RECEIPTS = 'MOJOLEARN_BOARD_RECEIPTS'
_registered = False


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def requested_path():
    """'ptx' when the worker selects the PTX set by name (MOJOLEARN_GPU_ARCH=sm_80;
    Andrew 2026-10-10: PTX is a normal target; no flag), else 'native'."""
    return 'ptx' if os.environ.get('MOJOLEARN_GPU_ARCH', '').strip().lower() == 'sm_80' else 'native'


def identity(path):
    if not path:
        if requested_path() != 'native':
            raise ValueError('Forced PTX boards require --artifact-manifest; legacy smoke is not path evidence')
        return None
    d = json.loads(Path(path).read_text())
    if d.get('schema') != SCHEMA or d.get('code_path') not in ('native', 'ptx'):
        raise ValueError('Invalid board artifact manifest schema/code_path')
    if d['code_path'] != requested_path():
        raise ValueError('Artifact manifest code path differs from requested loader path')
    if len(d.get('source_commit', '')) != 40 or d.get('numeric_mode') != 'identical':
        raise ValueError('Guarded board requires full source SHA and IDENTICAL numeric mode')
    files = d.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Empty artifact inventory')
    for name, digest in files.items():
        p = Path(name)
        if p.is_absolute() or '..' in p.parts or not name.endswith('.so') or len(digest) != 64:
            raise ValueError('Invalid artifact inventory member')
    out = dict(schema=SCHEMA, source_commit=d['source_commit'], code_path=d['code_path'],
               numeric_mode=d['numeric_mode'], files=files)
    if 'vendor' in d:
        if d['vendor'] not in ('cuda', 'hip') or (d['vendor'] == 'hip' and d['code_path'] != 'native'):
            raise ValueError('Invalid guarded vendor/code path')
        out['vendor'] = d['vendor']
    if d.get('installation') == 'source':
        if d['code_path'] != 'native' or not Path(d.get('source_root', '')).is_absolute():
            raise ValueError('Source layout is explicit native-only with absolute source_root')
        out.update(installation='source', source_root=d['source_root'])
    elif d.get('installation') not in (None, 'wheel'):
        raise ValueError('Unknown installation layout')
    runtime = d.get('runtime_files')
    if runtime is not None:
        if not isinstance(runtime, dict) or not runtime:
            raise ValueError('Empty common runtime inventory')
        for name, digest in runtime.items():
            if not Path(name).is_absolute() or '..' in Path(name).parts or len(digest) != 64:
                raise ValueError('Runtime inventory requires absolute paths and hashes')
        out['runtime_files'] = runtime
    if d['code_path'] == 'ptx' and not runtime:
        raise ValueError('Paired PTX board requires pinned common runtime_files')
    return out


def source_check(manifest, package):
    package = Path(package).resolve()
    if manifest.get('installation') == 'source':
        root = Path(manifest['source_root']).resolve()
        if package != root / 'python' / 'mojolearn':
            raise ValueError('Loaded package is outside the declared benchmark source checkout')
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        if head != manifest['source_commit'] or subprocess.run(
                ['git', '-C', str(root), 'diff', '--quiet', 'HEAD'], capture_output=True).returncode:
            raise ValueError('Benchmark source must be the clean frozen numerical commit')
    elif (package / 'identity_columns/COMMIT').read_text().strip() != manifest['source_commit']:
        raise ValueError('Installed numerical source differs from artifact manifest')


def installed_check(manifest, package):
    package = Path(package).resolve()
    source_check(manifest, package)
    for rel, digest in manifest['files'].items():
        path = (package / rel).resolve()
        if not path.is_relative_to(package) or sha(path) != digest:
            raise ValueError('Installed artifact bytes differ: ' + rel)
    for filename, digest in manifest.get('runtime_files', {}).items():
        if sha(filename) != digest:
            raise ValueError('Common runtime bytes differ: ' + filename)


def loaded_bindings():
    """Inventory the worker's actual extensions, including lazy/nested hosts.

    Module aliases are retained; role derives from the loaded file, not the
    alias. This observes sys.modules and never imports a benchmark binding.
    """
    rows = []
    for name, module in sorted(list(sys.modules.items())):
        if not name.startswith('mojolearn.'):
            continue
        filename = vars(module).get('__file__') if module is not None else None
        if not filename:
            continue
        path = Path(filename).resolve()
        if path.suffix != '.so' or not (path.name == '_mojolearn.so' or path.name.startswith('_mojolearn_')):
            continue
        rows.append(dict(module=name, file=str(path), sha256=sha(path),
                         role='host' if path.stem.endswith('_host') else 'gpu'))
    return rows


RUNTIME_PREFIXES = ('libAsync', 'libKGEN', 'libMSupport', 'libMojo', 'libstdc++', 'libgcc_s')


def loaded_runtime(manifest, maps_path='/proc/self/maps'):
    """Check actual worker mappings after timing; LD_LIBRARY_PATH is no proof."""
    expected = manifest.get('runtime_files')
    if not expected:
        return []                 # legacy native evidence remains readable
    pinned = {str(Path(p).resolve()): digest for p, digest in expected.items()}
    observed = set()
    for line in Path(maps_path).read_text().splitlines():
        fields = line.split(None, 5)
        if len(fields) != 6 or not fields[5].startswith('/'):
            continue
        raw = fields[5]
        for escaped, actual in ((r'\040', ' '), (r'\011', '\t'), (r'\012', '\n'), (r'\134', '\\')):
            raw = raw.replace(escaped, actual)
        filename = str(Path(raw).resolve())
        if filename in pinned or Path(raw).name.startswith(RUNTIME_PREFIXES):
            observed.add(filename)
    if not observed:
        raise ValueError('No actual common runtime mappings witnessed')
    rows = []
    for filename in sorted(observed):
        digest = sha(filename)
        if pinned.get(filename) != digest:
            raise ValueError('Loaded runtime outside pinned common environment: ' + filename)
        rows.append(dict(file=filename, sha256=digest))
    return rows


def hardware_receipt(vendor):
    if vendor == 'cuda':
        rows = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid,name,compute_cap,driver_version', '--format=csv,noheader,nounits'], text=True, timeout=20).strip().splitlines()
        if len(rows) != 1:
            raise ValueError('Guarded board requires one unambiguous physical GPU')
        return rows[0]
    if vendor == 'hip':
        data = json.loads(subprocess.check_output(['/opt/rocm/bin/rocm-smi', '--showuniqueid', '--showproductname', '--showdriverversion', '--json'], text=True, timeout=20))
        cards = [name for name in data if name.startswith('card')]
        if len(cards) != 1 or not isinstance(data[cards[0]], dict):
            raise ValueError('Guarded board requires one unambiguous ROCm GPU')
        values = data[cards[0]]
        valid_text = lambda value: str(value).strip().lower() not in ('', 'n/a', 'none', 'unknown', 'null')
        def valid_unique(value):
            if not valid_text(value):
                return False
            try:
                return int(str(value).strip(), 16) > 0
            except ValueError:
                return False
        if not any('unique' in k.lower() and valid_unique(v) for k,v in values.items()):
            raise ValueError('ROCm GPU unique identity is missing')
        if not any('series' in k.lower() and valid_text(v) for k,v in values.items()) or not valid_text(data.get('system', {}).get('Driver version')):
            raise ValueError('ROCm GPU product/driver evidence is missing')
        return json.dumps(data, sort_keys=True)
    raise ValueError('Unsupported guarded GPU vendor')


def collect(manifest):
    # This must be the worker's already imported package. Importing a fresh
    # package in the parent would say nothing about the timed child.
    ml = sys.modules.get('mojolearn')
    backend = sys.modules.get('mojolearn._backend')
    if ml is None or backend is None:
        raise ValueError('Timed worker did not import mojolearn')
    package = Path(ml.__file__).resolve().parent
    # The parent verifies the whole inventory once before any race. Hash
    # only actually loaded extensions here, after timings, so a large full
    # wheel inventory is not repeatedly scanned at every worker shutdown.
    source_check(manifest, package)
    plugin = backend.gpu_plugin() or {}
    source_native = manifest.get('installation') == 'source' and manifest['code_path'] == 'native' and not plugin
    if (not source_native and plugin.get('code_format') != manifest['code_path']) or ml.vendor() != manifest.get('vendor', 'cuda'):
        raise ValueError('Worker selected wrong code path/vendor')
    if ml.numeric_mode() != manifest['numeric_mode']:
        raise ValueError('Worker numeric mode differs')
    loaded = []
    for row in loaded_bindings():
        path = Path(row['file']).resolve()
        rel = path.relative_to(package).as_posix()
        if manifest['files'].get(rel) != row['sha256']:
            raise ValueError('Loaded extension outside pinned inventory: ' + rel)
        loaded.append(dict(module=row['module'], file=rel, sha256=row['sha256'], role=row['role']))
    gpu = [r for r in loaded if r['role'] == 'gpu']
    if not gpu:
        raise ValueError('No actual GPU binding loads witnessed')
    runtime = backend.baseline_selection_receipt()
    if manifest['code_path'] == 'ptx':
        if not runtime or runtime.get('selected') != 'ptx' or runtime.get('source_commit') != manifest['source_commit']:
            raise ValueError('Missing PTX runtime selection receipt')
        root = Path(backend._BASELINE_ROOT).resolve()
        actual = {((package / r['file']).relative_to(root).as_posix(), r['sha256']) for r in gpu}
        if {(r['file'], r['sha256']) for r in runtime.get('loaded_files', [])} != actual:
            raise ValueError('PTX runtime receipt differs from actual loaded bindings')
    elif runtime is not None:
        raise ValueError('Native run selected the PTX set')
    hardware = hardware_receipt(manifest.get('vendor', 'cuda'))
    return dict(status='verified', artifact_identity=manifest, loaded_files=loaded,
                runtime_selection=runtime, loaded_runtime_files=loaded_runtime(manifest), hardware=hardware, numeric_mode=ml.numeric_mode(),
                source_commit=manifest['source_commit'])


def register_worker(library):
    global _registered
    if _registered or library != 'mojolearn' or not os.environ.get(ENV) or not os.environ.get(RECEIPTS):
        return
    _registered = True
    def finish():
        row = dict(pid=os.getpid(), argv=sys.argv, final=True)
        try:
            row.update(collect(identity(os.environ[ENV])))
        except Exception as exc:
            row.update(status='invalid', error=str(exc))
        directory = Path(os.environ[RECEIPTS])
        directory.mkdir(parents=True, exist_ok=True)
        temp = directory / (str(os.getpid()) + '.tmp')
        temp.write_text(json.dumps(row, sort_keys=True) + '\n')
        temp.replace(directory / (str(os.getpid()) + '.json'))
    atexit.register(finish)


def read_receipts(directory, manifest, arms):
    rows = [json.loads(p.read_text()) for p in sorted(Path(directory).glob('*.json'))]
    if not rows:
        raise ValueError('No same-process final worker receipt')
    for row in rows:
        if row.get('status') != 'verified' or not row.get('final') or row.get('artifact_identity') != manifest:
            raise ValueError('Invalid final worker provenance: ' + row.get('error', str(row.get('pid'))))
    # Separate classical/neural/algorithm workers identify --arm. Trees use
    # one shared process, witnessed with --arms and the full loaded inventory.
    witnessed = set()
    for row in rows:
        argv = row.get('argv', [])
        if any(str(a).endswith("forest_speed_arm.py") for a in argv):
            witnessed.add("ours")
        for flag in ('--arm', '--arms'):
            if flag in argv and argv.index(flag) + 1 < len(argv):
                witnessed.update(argv[argv.index(flag) + 1].split(','))
    if not set(arms) <= witnessed:
        raise ValueError('Missing worker receipt for arms: ' + ','.join(sorted(set(arms) - witnessed)))
    return rows




def compare_boards(native, ptx):
    """Compare paired native and PTX timing evidence on one box.

    Missing output digests are explicitly counted. Identity is the reference
    table's PTX column (docs/NVIDIA_PTX_IDENTITY.md), not this comparison.
    """
    configs = [d.get('config', {}) for d in (native, ptx)]
    identities = [c.get('artifact_identity') for c in configs]
    if not all(identities) or [i.get('code_path') for i in identities] != ['native', 'ptx']:
        raise ValueError('Require guarded native then guarded PTX board')
    if not identities[0].get('runtime_files') or identities[0].get('runtime_files') != identities[1].get('runtime_files'):
        raise ValueError('Paired boards require the same pinned common runtime environment')
    for field in ('source_commit', 'numeric_mode'):
        if identities[0].get(field) != identities[1].get(field):
            raise ValueError('Different numerical ' + field)
    for field in ('vendor', 'modes', 'families', 'lanes', 'datasets', 'rows', 'rounds', 'seed', 'infer', 'neural_shape', 'data_sha256', 'harness_sha256'):
        if configs[0].get(field) != configs[1].get(field):
            raise ValueError('Different board settings/data: ' + field)
    if configs[0].get('rounds') != 1 or not configs[0].get('data_sha256'):
        raise ValueError('Require one scored round and recorded dataset hashes')
    hardware = native.get('box', {}).get('artifact_hardware')
    if not hardware or hardware != ptx.get('box', {}).get('artifact_hardware'):
        raise ValueError('Different physical GPU/driver')
    for field in ('packages', 'python'):
        if native.get('box', {}).get(field) != ptx.get('box', {}).get(field):
            raise ValueError('Different installed environment: ' + field)
    if native.get('plan') != ptx.get('plan') or not native.get('plan'):
        raise ValueError('Different or empty race coverage')
    compared = []
    missing_hashes = []
    for rid in native['plan']:
        records = [board.get('races', {}).get(rid) for board in (native, ptx)]
        if any(not r or r.get('status') != 'done' or r.get('params_check') != 'MATCHED' for r in records):
            raise ValueError('Incomplete or failed race: ' + rid)
        for record, ident in zip(records, identities):
            receipts = record.get('worker_provenance') or []
            if not receipts or any(r.get('status') != 'verified' or r.get('artifact_identity') != ident or r.get('hardware') != hardware for r in receipts):
                raise ValueError('Missing/mismatched actual worker evidence: ' + rid)
            pinned = {str(Path(p).resolve()): h for p, h in ident['runtime_files'].items()}
            if any(not r.get('loaded_runtime_files') or any(pinned.get(x['file']) != x['sha256'] for x in r['loaded_runtime_files']) for r in receipts):
                raise ValueError('Missing/mismatched actual runtime mappings: ' + rid)
        for phase in ('cells', 'infer_cells'):
            cells = [{c['arm']: c for c in r.get(phase, []) if c.get('library') == 'mojolearn' or c.get('arm', '').startswith('ours')} for r in records]
            if cells[0].keys() != cells[1].keys():
                raise ValueError('Different measured arm coverage: ' + rid)
            for arm in cells[0]:
                a, b = cells[0][arm], cells[1][arm]
                for field in ('settings', 'params', 'shape', 'rows', 'mode'):
                    if a.get(field) != b.get(field):
                        raise ValueError('Different cell settings: ' + rid + '/' + arm + '/' + field)
                if a.get('status') != 'ok' or b.get('status') != 'ok':
                    raise ValueError('Failed cell: ' + rid + '/' + arm)
                if a.get('hash') and b.get('hash'):
                    if a['hash'] != b['hash']:
                        raise ValueError('Output digest mismatch: ' + rid + '/' + arm + '/' + phase)
                else:
                    missing_hashes.append(rid + '/' + arm + '/' + phase)
                ms = [c.get('median_ms') for c in (a, b)]
                if not all(isinstance(v, (int, float)) and v > 0 for v in ms):
                    raise ValueError('Missing timing: ' + rid + '/' + arm)
                compared.append(dict(race=rid, arm=arm, phase=phase, native_ms=ms[0], ptx_ms=ms[1], ptx_over_native=ms[1]/ms[0]))
    return dict(schema='mojolearn-paired-board-comparison/1', source_commit=identities[0]['source_commit'],
                hardware=hardware, races=len(native['plan']), comparisons=compared,
                missing_output_digests=missing_hashes)


if __name__ == '__main__':
    import argparse
    import importlib.util
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--verify')
    group.add_argument('--compare', nargs=2, metavar=('NATIVE_BOARD', 'PTX_BOARD'))
    parser.add_argument('--out')
    args = parser.parse_args()
    if args.verify:
        manifest = identity(args.verify)
        spec = importlib.util.find_spec('mojolearn')
        if spec is None or spec.origin is None:
            raise SystemExit('Installed mojolearn package not found')
        installed_check(manifest, Path(spec.origin).parent)
    else:
        if not args.out:
            parser.error('--compare requires --out')
        result = compare_boards(*(json.loads(Path(p).read_text()) for p in args.compare))
        Path(args.out).write_text(json.dumps(result, indent=2) + '\n')
