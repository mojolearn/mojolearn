#!/usr/bin/env python3
"""Retain compact maintainer results in MojoLearn without moving their originals.

No network, Git mutations, reference promotion, or background collection. The
caller owns execution/admission; retention never turns a partial run into a pass.
"""
import argparse
import hashlib
import gzip
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
MAX_SUMMARY_BYTES = 4 * 1024 * 1024
MAX_BOARD_BYTES = 64 * 1024 * 1024
MAX_IDENTITY_BYTES = 64 * 1024 * 1024
MAX_IDENTITY_REPORT_BYTES = 128 * 1024 * 1024


def repository(repo=None):
    root = Path(repo or os.environ.get('MOJOLEARN_SUMMARY_REPO') or ROOT).resolve()
    if not (root / '.git').exists() or not (root / 'tools').is_dir():
        raise ValueError(f'{root} is not a MojoLearn checkout; retain on the collecting checkout')
    return root


def slug(value):
    value = re.sub(r'[^a-zA-Z0-9_.-]+', '-', str(value)).strip('.-')
    if not value or len(value) > 160:
        raise ValueError('invalid summary stream name')
    return value


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_bytes(path, limit=MAX_SUMMARY_BYTES):
    with Path(path).open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f'{path} exceeds the {limit}-byte retention input limit')
    return raw


def read_json(path, limit=MAX_SUMMARY_BYTES):
    return json.loads(read_bytes(path, limit))


def pick(doc, keys):
    return {key: doc[key] for key in keys.split() if key in doc}


def json_safe(value):
    # Existing board JSON can contain infinity (e.g. an unbounded parameter)
    # or NaN from a failed score. Preserve the value explicitly, never as zero
    # or null, while producing JSON readable outside Python.
    if isinstance(value, float) and not math.isfinite(value):
        return {'$float': 'NaN' if math.isnan(value) else ('Infinity' if value > 0 else '-Infinity')}
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def atomic_json(path, doc, limit=MAX_SUMMARY_BYTES):
    """Bound the output before replacing the previous, complete snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temp = Path(stream.name)
            size = 0
            for chunk in json.JSONEncoder(sort_keys=True, ensure_ascii=True,
                                          allow_nan=False, separators=(',', ':')).iterencode(json_safe(doc)):
                raw = chunk.encode()
                size += len(raw)
                if size + 1 > limit:
                    raise ValueError(f'summary exceeds {limit} bytes; original results remain in place')
                stream.write(raw)
            stream.write(b'\n')
        if path.exists() and digest(path) == digest(temp):
            return path
        os.replace(temp, path)
        return path
    finally:
        if temp is not None:
            temp.unlink(missing_ok=True)


def destination(root, relative):
    path = root / relative
    if path.resolve() != path:
        raise ValueError(f'retention destination must not contain symlinks: {path}')
    return path


def retain(kind, stream, payload, repo=None):
    if kind not in ('measurements', 'verification'):
        raise ValueError('unknown retention category')
    root = repository(repo)
    path = destination(root, Path('docs') / kind / 'latest' / 'auto' / (slug(stream) + '.json'))
    return atomic_json(path, dict(schema='mojolearn.run-summary.v1',
        kind=kind, stream=stream, results=payload,
        scope='Retained results only; no automatic admission, reference regeneration or release qualification.'))


def try_retain(function, *args, **kwargs):
    """Report retention errors separately, preserving the run's original verdict."""
    try:
        return function(*args, **kwargs)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError) as exc:
        print(f'RETENTION FAILED: {exc}', file=sys.stderr)
        return None


def intern(contexts, value):
    value = json_safe(value)
    key = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    contexts.setdefault(key, value)
    return key


def compact_board(board, contexts=None):
    own_contexts = contexts is None
    if own_contexts:
        contexts = {}
    result = pick(board, 'schema commit updated created vendor plan measurement_selection '
                        'skipped_recipes failed_preflight_recipes')
    # Config is a defined selection, never a copy of controller environments or credentials.
    result['config_ref'] = intern(contexts, pick(board.get('config') or {},
        'vendor modes rows rounds neural_shape families lanes datasets source_commit '
        'mojolearn_version artifact_identity smoke_gate smoke shard data data_sha256 harness_sha256 seed'))
    result['box_ref'] = intern(contexts, pick(board.get('box') or {}, 'gpu host mojolearn repo os python packages'))
    result['races'] = {}
    for key, race in board.get('races', {}).items():
        row = pick(race, 'status rc family lane dataset started finished measurement_completion '
                        'evidence_failure evidence_receipt audit rerace skipped reason')
        for phase in ('cells', 'infer_cells'):
            row[phase] = [pick(cell, 'arm library mode status reason error median_ms rounds '
                'times_ms warmup_ms quality hash params timing_boundary hardware '
                'comparability admitted_opponents pairs_with') for cell in race.get(phase, [])]
            for saved, cell in zip(row[phase], race.get(phase, [])):
                measurement = cell.get('measurement') or {}
                saved['provenance_ref'] = intern(contexts, pick(measurement,
                    'source artifacts dataset shape seed software timing retained_outputs'))
                saved['measurement'] = pick(measurement, 'id schema completeness missing phase')
        result['races'][key] = row
    if own_contexts:
        result['contexts'] = contexts
    return result


def retain_board(out, board, *, smoke=False, exit_code=None, repo=None):
    vendor = board.get('vendor') or board.get('config', {}).get('vendor') or 'unknown'
    stream = 'bench-board-' + str(vendor) + ('-smoke' if smoke else '')
    payload = dict(source_directory=str(Path(out).resolve()), exit_code=exit_code,
                   execution_state='in-progress' if exit_code is None else 'finished',
                   run_kind='smoke' if smoke else 'measurement', board=compact_board(board))
    smoke_path = Path(out) / 'smoke.json'
    if smoke and exit_code is not None and smoke_path.exists():
        payload['smoke'] = read_json(smoke_path)
    return retain('measurements', stream, payload, repo)


def retain_campaign(config, journal):
    out = Path(config['out'])
    paths = []
    for vendor in config['vendors']:
        name = vendor['vendor']
        if name not in journal.get('vendors', {}):
            continue
        boards, contexts = {}, {}
        for path in sorted((out / name / 'boards').glob('*.json')):
            boards[path.name] = compact_board(read_json(path, MAX_BOARD_BYTES), contexts)
            boards[path.name]['original_sha256'] = digest(path)
        payload = dict(source_directory=str(out.resolve()),
            config=pick(config, 'source_commit version data_keys'),
            controller=pick(journal, 'source_commit started finished stopped status error'),
            vendor=journal.get('vendors', {}).get(name, {'status': 'not-started'}),
            boards=boards, contexts=contexts)
        paths.append(retain('measurements', 'campaign-' + name, payload, config['repo']))
    return paths


def retain_verification(producer, reports, *, metadata=None, repo=None):
    payload = dict(metadata=metadata or {}, reports={}, originals=[])
    for path in map(Path, reports):
        if path.name in payload['reports']:
            raise ValueError(f'duplicate report name: {path.name}')
        raw = read_bytes(path)
        payload['reports'][path.name] = json.loads(raw)
        payload['originals'].append(dict(path=str(path.resolve()), sha256=hashlib.sha256(raw).hexdigest()))
    if not payload['reports']:
        raise ValueError('no verification reports to retain')
    return retain('verification', producer, payload, repo)


def immutable_bytes(path, raw, *, limit=MAX_SUMMARY_BYTES):
    """Install complete bytes without replacing a prior record or a racing writer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temp = Path(stream.name)
        stream.write(raw)
    try:
        try:
            os.link(temp, path)
        except FileExistsError:
            if read_bytes(path, limit) != raw:
                raise ValueError(f'refusing to overwrite immutable identity evidence: {path}')
    finally:
        temp.unlink()


def identity_companions(path, record):
    """Validate every referenced verifier companion before publishing a column."""
    observation = record.get('verifier_observation') or {}
    if not observation:
        return []
    files = []
    for field, hash_field in (('report', 'report_sha256'), ('launch', 'launch_sha256')):
        name = observation.get(field)
        if name is None and field == 'launch':
            continue  # Historical conversion retained no producer receipt.
        if not isinstance(name, str) or Path(name).name != name or name in ('.', '..'):
            raise ValueError('unsafe or missing verifier companion filename: ' + field)
        expected = observation.get(hash_field)
        if not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected):
            raise ValueError('missing verifier companion digest: ' + field)
        raw = read_bytes(path.with_name(name), MAX_IDENTITY_BYTES)
        checked = raw
        if field == 'report':
            if observation.get('report_encoding') != 'gzip':
                raise ValueError('unsupported verifier report encoding')
            with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
                checked = stream.read(MAX_IDENTITY_REPORT_BYTES + 1)
            if len(checked) > MAX_IDENTITY_REPORT_BYTES:
                raise ValueError('verifier source report exceeds retention limit')
        if hashlib.sha256(checked).hexdigest() != expected:
            raise ValueError('verifier companion digest differs: ' + field)
        files.append(dict(name=name, raw=raw, field=field, sha256=hashlib.sha256(raw).hexdigest()))
    if observation.get('launch') and not path.with_name(path.name.replace('.identical.json', '.provenance.txt')).is_file():
        raise ValueError('strict verifier column lacks its provenance companion')
    return files


def retain_identity(path, repo=None):
    """Keep the exact column bytes at an immutable path, before table admission."""
    root = repository(repo)
    path = Path(path)
    if not path.name.endswith('.identical.json'):
        raise ValueError('expected a *.identical.json column')
    raw = read_bytes(path, MAX_IDENTITY_BYTES)
    record = json.loads(raw)
    companions = identity_companions(path, record)
    sha = hashlib.sha256(raw).hexdigest()
    target = destination(root, Path('bench/results/identity_break/retained') / sha / path.name)
    # Publish the column only after its relative companions are durable here.
    retained_companions = []
    for companion in companions:
        sidecar = target.with_name(companion['name'])
        immutable_bytes(sidecar, companion['raw'], limit=MAX_IDENTITY_BYTES)
        retained_companions.append(dict(path=sidecar.relative_to(root).as_posix(),
            kind=companion['field'], sha256=companion['sha256']))
    provenance = path.with_name(path.name.replace('.identical.json', '.provenance.txt'))
    retained_provenance = None
    if provenance.exists():
        raw_provenance = read_bytes(provenance)
        sidecar = target.with_name(target.name.replace('.identical.json', '.provenance.txt'))
        # A repeated collection may have a fresh dated receipt for identical
        # column bytes. Preserve both receipts, with only one copy of the column.
        if sidecar.exists() and read_bytes(sidecar) != raw_provenance:
            sidecar = target.with_name(hashlib.sha256(raw_provenance).hexdigest() + '.provenance.txt')
        immutable_bytes(sidecar, raw_provenance)
        retained_provenance = sidecar.relative_to(root).as_posix()
    immutable_bytes(target, raw, limit=MAX_IDENTITY_BYTES)
    retain('verification', 'identity-' + str(record.get('vendor', 'unknown')),
           dict(record=target.relative_to(root).as_posix(), sha256=sha,
                provenance=retained_provenance, companions=retained_companions,
                original=str(path.resolve()),
                metadata=pick(record, 'commit vendor mode partial_column parts_omitted')),
           root)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path)
    sub = parser.add_subparsers(dest='kind', required=True)
    verification = sub.add_parser('verification', help='collect existing JSON results without rerunning')
    verification.add_argument('--producer', required=True)
    verification.add_argument('reports', nargs='+', type=Path)
    identity = sub.add_parser('identity', help='retain exact reference input bytes without admitting them')
    identity.add_argument('record', type=Path)
    args = parser.parse_args()
    if args.kind == 'identity':
        path = try_retain(retain_identity, args.record, repo=args.repo)
    else:
        path = try_retain(retain_verification, args.producer, args.reports, repo=args.repo)
    if path:
        print(path)
    return 0 if path else 1


if __name__ == '__main__':
    raise SystemExit(main())
