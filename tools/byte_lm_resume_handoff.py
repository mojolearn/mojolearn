#!/usr/bin/env python3
"""Root-authored compact transport witnesses; stdlib/file operations only.

Authoring requires full raw baseline admission locally. Remote hash checks
are diagnostic pre/postchecks, never a replacement for the final all-raw
byte_lm_state_compare.py admission. No model/runtime import or execution.
"""
import argparse
import os
from pathlib import Path
import stat

import byte_lm_state_compare as _state
from byte_lm_state_compare import (
    canonical, checkpoint, check_metadata, compatible,
    load_capture, parse, read, receipt, registry, require, same_steps, sha,
    signature, safe_path, initial_raw, filename,
)
from byte_lm_validation_admit import admit

SCHEMA = 'mojolearn.byte-lm.compact-handoff.v1'
MAX_HANDOFF = 2 * 1024 * 1024


def require_default_shape():
    """The compact handoff path has not been extended past the certified shape.

    DEVIATION 2682 made the comparator's profile and array counts follow whichever
    shape a run binds. This file reads both, so it must read them off the module
    rather than copies taken at import, or it would quietly check a second shape's
    capture against the first shape's counts. It does read them live now, and it
    still refuses anything but the default by name, because carrying a second
    shape through checkpoint transfer is work nobody has done or tested."""
    require(_state.SHAPE == _state.byte_lm_shape.Shape(),
            'compact handoff covers the certified b2-l32 shape only; this run is '
            + _state.PROFILE)


def bundle_bound(directory):
    root = safe_path(directory)
    require(root.is_dir(), 'handoff directory missing')
    pending, count, total = [(root, 0)], 0, 0
    while pending:
        parent, depth = pending.pop()
        with os.scandir(parent) as entries:
            for entry in entries:
                count += 1
                require(count <= 128, 'compact handoff exceeds 128 entries')
                info = entry.stat(follow_symlinks=False)
                require(not stat.S_ISLNK(info.st_mode), 'handoff symlink refused')
                if stat.S_ISDIR(info.st_mode):
                    require(depth < 6, 'handoff path depth exceeded')
                    pending.append((Path(entry.path), depth + 1))
                else:
                    require(stat.S_ISREG(info.st_mode) and info.st_size <= 16 * 1024 * 1024,
                            'handoff file type/size exceeded')
                    total += info.st_size
                    require(total <= 32 * 1024 * 1024, 'compact handoff exceeds 32 MiB')
    return root


def relative(name):
    require(isinstance(name, str) and 0 < len(name) <= 1024 and
            not Path(name).is_absolute() and len(Path(name).parts) <= 8 and
            all(part not in ('', '.', '..') for part in name.split('/')), 'unsafe handoff path')
    return Path(name)


def exclusive(path, raw):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        with os.fdopen(fd, 'wb', closefd=False) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.fchmod(fd, 0o444)
    finally:
        os.close(fd)


def digest(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def copy_receipt(source, destination, receipt_name, artifact_sha, vendor, kind):
    path = source / receipt_name
    receipt(path, artifact_sha, vendor, kind)
    raw = read(path, limit=65536)
    report = parse(raw)
    exclusive(destination / receipt_name, raw)
    # Preserve each original relative filename: the receipt itself is not
    # rewritten. These are small logs/commands/results, not raw trajectories.
    for key in ('command', 'guard_log', 'result'):
        name = relative(report[key]['file'])
        body = read(source / name, limit=8 * 1024 * 1024)
        target = destination / name
        if target.exists():
            require(read(target) == body, 'conflicting receipt artifact')
        else:
            exclusive(target, body)
    return dict(receipt_sha256=sha(raw), artifact_sha256=artifact_sha, vendor=vendor, job_kind=kind)


def author(baseline, head, output):
    baseline, output = safe_path(baseline), safe_path(output)
    if head is not None:
        head = safe_path(head)
    # This call validates complete raw steps, independent oracle, actual
    # baseline binary and every successful campaign job before handoff exists.
    admitted = admit(baseline)
    base = load_capture(baseline / 'full128', 'continuous', admitted['vendor'])
    capture = base
    if head is not None:
        require(read(head / 'exit_code').strip() == b'0', 'head campaign failed')
        capture = load_capture(head / 'head64', 'head64', admitted['vendor'])
        compatible(base, capture)
        require(same_steps(base, capture) == 64, 'head raw steps differ from admitted baseline')
    output.mkdir(parents=False, exist_ok=False)
    vendor = admitted['vendor']
    chains = []
    for name, artifact, kind in (
            ('byte-full128.receipt.json', base['summary_sha256'], 'capture'),
            ('byte-step1.receipt.json', sha(read(baseline / 'step1/summary.json')), 'capture'),
            ('byte-gradient-oracle.receipt.json', sha(read(baseline / 'gradient-oracle.json')), 'oracle')):
        result = copy_receipt(baseline, output / 'evidence/baseline', name, artifact, vendor, kind)
        result['file'] = 'evidence/baseline/' + name
        chains.append(result)
    primary = chains[0]['file']
    if head is not None:
        result = copy_receipt(head, output / 'evidence/head', 'head64.receipt.json',
                              capture['summary_sha256'], vendor, 'capture')
        result['file'] = 'evidence/head/head64.receipt.json'
        chains.append(result)
        primary = result['file']
        exclusive(output / 'head64.checkpoint.json', capture['checkpoint'])
    else:
        binary = read(baseline / 'bindings/_mojolearn_byte_lm.so')
        require(sha(binary) == admitted['binding_sha256'], 'baseline binding changed')
        exclusive(output / 'binding/_mojolearn_byte_lm.so', binary)
    require_default_shape()
    handoff = dict(schema=SCHEMA, kind='head64' if head is not None else 'baseline128',
        profile=_state.PROFILE, vendor=vendor, binding_sha256=capture['runtime']['binding_sha256'],
        binding_file=None if head is not None else 'binding/_mojolearn_byte_lm.so',
        source=capture['source'], runtime=capture['runtime'], metadata=capture['metadata'],
        summary_sha256=capture['summary_sha256'], baseline_summary_sha256=base['summary_sha256'],
        schedule=capture['summary']['schedule'], initial_state=capture['summary']['initial_state'],
        final_state=capture['summary']['final_state'],
        checkpoint=dict(file='head64.checkpoint.json' if head is not None else None,
                        bytes=len(capture['checkpoint']), sha256=sha(capture['checkpoint'])),
        steps=[dict(step=number, manifest_sha256=record['sha256'], manifest=record['manifest'])
               for number, record in sorted(capture['steps'].items())],
        root_admission=admitted, receipt_chain=chains, primary_receipt=primary,
        author_source_sha256=sha(read(__file__)),
        boundary='Root admitted full raw baseline locally. Remote hash agreement is diagnostic only; '
                 'final all-raw cross-vendor comparator and successful guard exit remain mandatory.')
    raw = canonical(handoff)
    require(len(raw) <= MAX_HANDOFF, 'handoff manifest exceeds two MiB')
    exclusive(output / 'handoff.json', raw)
    bundle_bound(output)
    return dict(handoff=str(output / 'handoff.json'), sha256=sha(raw), bytes=len(raw),
                kind=handoff['kind'], checkpoint_sha256=handoff['checkpoint']['sha256'])


def load_handoff(directory, expected_sha, *, kind=None, vendor=None):
    """Verify pinned compact bytes/receipt linkage; does not re-admit raw run."""
    directory = bundle_bound(directory)
    require(digest(expected_sha), 'root-pinned handoff SHA required')
    raw = read(directory / 'handoff.json', limit=MAX_HANDOFF)
    require(sha(raw) == expected_sha, 'handoff differs from root-pinned SHA')
    item = parse(raw)
    require_default_shape()
    require(item['schema'] == SCHEMA and item['profile'] == _state.PROFILE and
            item['kind'] in ('baseline128', 'head64') and item['vendor'] in ('cuda', 'hip') and
            (kind is None or item['kind'] == kind) and (vendor is None or item['vendor'] == vendor),
            'wrong compact handoff profile/kind/vendor')
    require(item['root_admission']['passed'] is True and
            item['root_admission']['vendor'] == item['vendor'] and
            item['root_admission']['binding_sha256'] == item['binding_sha256'] and
            item['runtime']['binding_sha256'] == item['binding_sha256'] and
            item['runtime']['native_vendor'] == item['vendor'] and
            item['runtime']['native_profile'] == _state.PROFILE and item['runtime']['native_numeric_mode'] == 1,
            'root admission/runtime handoff mismatch')
    check_metadata(item['metadata'], item['schedule'])
    end = 128 if item['kind'] == 'baseline128' else 64
    require(item['metadata']['completed_steps'] == item['metadata']['next_batch_index'] == end and
            len(item['steps']) == end, 'incomplete compact step inventory')
    previous = item['initial_state']
    for number, record in enumerate(item['steps'], 1):
        manifest = record['manifest']
        require(record['step'] == number and sha(canonical(manifest)) == record['manifest_sha256'] and
                manifest['schema'] == 'mojolearn.byte-lm.gradient-capture.v1' and
                manifest['registry'] == registry() and set(manifest['arrays']) == set(_state.COUNTS),
                'malformed compact step manifest')
        require(manifest['config'] == dict(profile=_state.PROFILE, numeric_mode='identical', vendor=item['vendor'],
                completed_steps=number - 1, post_completed_steps=number, optimizer=item['metadata']['config']) and
                manifest['input_state'] == previous and manifest['output_state']['completed_steps'] == number,
                'compact configuration/state hash chain differs')
        for key, count in _state.COUNTS.items():
            require(manifest['arrays'][key]['count'] == count and digest(manifest['arrays'][key]['sha256']),
                    'malformed compact raw array witness')
        previous = manifest['output_state']
    require(previous == item['final_state'], 'compact final state mismatch')
    require(digest(item['checkpoint']['sha256']) and type(item['checkpoint']['bytes']) is int and
            0 < item['checkpoint']['bytes'] <= MAX_HANDOFF, 'invalid compact checkpoint witness')
    require(3 <= len(item['receipt_chain']) <= 4, 'wrong receipt chain length')
    seen = set()
    for chain in item['receipt_chain']:
        name = relative(chain['file'])
        require(str(name) not in seen, 'duplicate compact receipt')
        seen.add(str(name))
        require(chain['vendor'] == item['vendor'], 'compact receipt vendor differs')
        body = read(directory / name, limit=65536)
        require(sha(body) == chain['receipt_sha256'], 'compact receipt changed')
        receipt(directory / name, chain['artifact_sha256'], chain['vendor'], chain['job_kind'])
    require(item['primary_receipt'] in seen, 'missing primary root receipt')
    primary = parse(read(directory / relative(item['primary_receipt']), limit=65536))
    require(primary['result']['sha256'] == item['summary_sha256'] and primary['vendor'] == item['vendor'],
            'primary receipt does not bind capture summary')
    summary_path = (directory / relative(item['primary_receipt'])).parent / relative(primary['result']['file'])
    summary = parse(read(summary_path, limit=MAX_HANDOFF))
    require(summary['runtime'] == item['runtime'] and summary['schedule'] == item['schedule'] and
            summary['initial_state'] == item['initial_state'] and summary['final_state'] == item['final_state'] and
            summary['checkpoint']['sha256'] == item['checkpoint']['sha256'] and
            summary['checkpoint']['bytes'] == item['checkpoint']['bytes'] and
            len(summary['records']) == end, 'compact witness differs from original receipted summary')
    for number, record in enumerate(summary['records'], 1):
        require(record['step'] == number and record['capture'] == f'step{number:06}' and
                record['capture_sha256'] == item['steps'][number - 1]['manifest_sha256'],
                'compact step inventory differs from original receipted summary')
    checkpoint_raw, checkpoint_arrays = None, None
    if item['kind'] == 'head64':
        require(item['checkpoint']['file'] == 'head64.checkpoint.json', 'wrong foreign checkpoint name')
        checkpoint_raw, metadata, arrays = checkpoint(directory, 'head64.checkpoint.json', item['checkpoint'])
        require(metadata == item['metadata'] and signature(metadata, arrays, 64) == item['final_state'],
                'actual foreign checkpoint differs from handed state')
        checkpoint_arrays = arrays
    else:
        require(item['binding_file'] == 'binding/_mojolearn_byte_lm.so', 'wrong compact binding path')
        require(sha(read(directory / item['binding_file'])) == item['binding_sha256'], 'compact baseline binary changed')
    return dict(directory=directory, data=item, sha256=sha(raw), checkpoint=checkpoint_raw,
                checkpoint_arrays=checkpoint_arrays)


def compatible_handoffs(a, b):
    left, right = a['data'], b['data']
    require(left['source'] == right['source'] and left['schedule'] == right['schedule'] and
            dict(left['metadata'], completed_steps=0, next_batch_index=0) ==
            dict(right['metadata'], completed_steps=0, next_batch_index=0), 'compact source/config/schedule mismatch')
    if left['vendor'] == right['vendor']:
        require(left['binding_sha256'] == right['binding_sha256'], 'same-vendor compact binary mismatch')
    for x, y in zip(left['steps'], right['steps']):
        require(x['step'] == y['step'] and x['manifest']['arrays'] == y['manifest']['arrays'] and
                x['manifest']['input_state'] == y['manifest']['input_state'] and
                x['manifest']['output_state'] == y['manifest']['output_state'], 'compact cross-vendor hashes differ')


def compare_capture_hashes(capture, baseline):
    """All actual capture bytes are validated, but reference side is hashes."""
    reference = baseline['data']
    require(capture['source'] == reference['source'] and
            capture['runtime']['native_vendor'] == reference['vendor'] and
            capture['runtime']['binding_sha256'] == reference['binding_sha256'] and
            capture['summary']['schedule'] == reference['schedule'], 'capture/handoff provenance mismatch')
    for number, actual in capture['steps'].items():
        expected = reference['steps'][number - 1]
        require(actual['sha256'] == expected['manifest_sha256'] and actual['manifest'] == expected['manifest'],
                'actual step differs from compact reference witness')
    if capture['summary']['completed_steps'] == 128:
        require(sha(capture['checkpoint']) == reference['checkpoint']['sha256'], 'terminal checkpoint SHA differs')


def compact_control(head, resumed, control, baseline):
    """Actual local control bytes versus actual resume bytes; baseline hashed."""
    compare_capture_hashes(resumed, baseline)
    reference = baseline['data']
    require(control['source'] == reference['source'] and
            control['runtime']['native_vendor'] == reference['vendor'] and
            control['runtime']['binding_sha256'] == reference['binding_sha256'] and
            control['summary']['schedule'] == reference['schedule'] and
            head['data']['vendor'] != reference['vendor'] and
            control['incoming'] == resumed['incoming'] == head['checkpoint'],
            'compact control provenance or actual transferred bytes differ')
    legitimate = initial_raw(control['root'] / 'legitimate-head64')
    require(legitimate == head['checkpoint_arrays'] and
            control['initial']['parameters'] == legitimate['parameters'] and
            control['initial']['flags'] == legitimate['flags'] and
            all(any(legitimate[k]) for k in ('m', 'v')) and
            all(not any(control['initial'][k]) for k in ('m', 'v')), 'malformed planted moments')
    evidence = control['summary']['control']
    require(evidence['name'] == 'zero-moments65' and evidence['changed_fields'] == ['m', 'v'] and
            evidence['legitimate_state_directory'] == 'legitimate-head64' and
            evidence['legitimate_state'] == head['data']['final_state'] ==
            parse(read(control['root'] / 'legitimate-head64/state.json', limit=65536)) and
            evidence['altered_state'] == control['summary']['initial_state'], 'control state witness mismatch')
    good, bad = resumed['steps'][65], control['steps'][65]
    require(good['manifest']['config'] == bad['manifest']['config'], 'control config/cursor changed')
    for key in ('initial_p', 'initial_flags', 'ids', 'loss', 'grad', 'post_flags'):
        require(read(good['directory'] / filename(key)) == read(bad['directory'] / filename(key)),
                'control changed forward/backward bytes: ' + key)
    for key in ('post_p', 'post_m', 'post_v'):
        require(read(good['directory'] / filename(key)) != read(bad['directory'] / filename(key)),
                'ineffective missing-moments control: ' + key)
    return dict(effective=True, step=65, changed_post_arrays=['post_p', 'post_m', 'post_v'],
                boundary='actual local control/resume byte comparison; foreign baseline remains compact witness')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True, help='complete locally retained baseline campaign')
    parser.add_argument('--head', type=Path, help='complete locally retained head campaign; omit for baseline handoff')
    parser.add_argument('--output', type=Path, required=True, help='new bundle directory, parent must exist')
    args = parser.parse_args()
    print(canonical(author(args.baseline, args.head, args.output)).decode(), end='')


if __name__ == '__main__':
    main()
