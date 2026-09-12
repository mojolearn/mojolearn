#!/usr/bin/env python3
"""Bounded, standard-library-only comparison of retained byte-LM evidence.

ROOT EXECUTION ONLY. This program never imports NumPy, MojoLearn or a GPU
runtime and never executes a model. Authored evidence is not qualification.
Both continuous captures are required; the head/resume/control and external
guard/oracle receipts are required for admission. Missing prerequisites leave
an agreement-only diagnostic and a nonzero exit status.

Optional --metal adds a continuous128 trajectory and requires guard metal=FILE.
It retains both CUDA/HIP oracles and their resume/control requirements. Metal
must identify Darwin/arm64 and match the complete source inventory and raw
trajectory; this does not admit a Metal FP64 oracle or Metal checkpoint resume.

Example (all inputs are existing files):
  python tools/byte_lm_state_compare.py --cuda CUDA128 --hip HIP128 \
    --head CUDA64 --resume HIP_RESUME128 --control HIP_ZERO65 \
    --guard cuda=cuda.receipt.json --guard hip=hip.receipt.json \
    --guard head=head.receipt.json --guard resume=resume.receipt.json \
    --guard control=control.receipt.json \
    --oracle cuda=cuda.oracle.json --oracle hip=hip.oracle.json \
    --oracle-guard cuda=cuda.oracle.receipt.json \
    --oracle-guard hip=hip.oracle.receipt.json --output comparison.json

Receipts use tools/root_job_receipt.py's mojolearn.root-job-receipt.v1
schema. Retained command/log/result files are SHA-bound under the receipt's
parent directory. The result must identify the capture summary or oracle JSON;
the log must end in the successful vendor guard JSON. This verifies supplied
root-observed linkage, not signed attestation.
Outputs are exclusive; paths must be regular files without symlink components.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import importlib.util

# Load the adjacent standard-library-only policy without depending on cwd or
# PYTHONPATH (this comparator is also imported by file-path fixture drivers).
_receipt_spec = importlib.util.spec_from_file_location(
    '_byte_lm_receipt_policy', Path(__file__).with_name('root_job_receipt.py'))
_receipt_policy = importlib.util.module_from_spec(_receipt_spec)
_receipt_spec.loader.exec_module(_receipt_policy)
validate_guard_terminal = _receipt_policy.validate_guard_terminal

# DEVIATION 2682. The shape was literals here. It now comes from the shared
# module, loaded the same way and for the same reason as the policy above, so
# this comparator and the capture harness cannot describe two different models
# to each other. The helper asserts at import that its default derivation equals
# the certified literals, so the b2-l32 path cannot move underneath this.
_shape_spec = importlib.util.spec_from_file_location(
    '_byte_lm_shape', Path(__file__).with_name('byte_lm_shape.py'))
byte_lm_shape = importlib.util.module_from_spec(_shape_spec)
_shape_spec.loader.exec_module(byte_lm_shape)

#: The shape this run is reading. It starts as the certified default, which is
#: what every capture written before DEVIATION 2682 is, and `use_shape` replaces
#: it once a run says otherwise. One binding point, because every reader below
#: needs the same answer and a second source would be a second opinion.
SHAPE = byte_lm_shape.Shape()
PROFILE = SHAPE.profile
CORPUS_SHA = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
INIT_ID = 'u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1'
N = SHAPE.n_total
STATE_NAMES = {'parameters': 'p', 'm': 'm', 'v': 'v', 'flags': 'flags'}
COUNTS = SHAPE.counts()
_BOUND = False
MAX_FILE = 16 * 1024 * 1024
MAX_TREE = 256 * 1024 * 1024
MAX_FILES = 2048
ORACLE_TOLERANCES = {
    'gradient': {'atol': 2e-6, 'rtol': 2e-4},
    'loss': {'atol': 2e-6, 'rtol': 2e-6},
    'post_p': {'atol': 2e-6, 'rtol': 2e-5},
    'post_m': {'atol': 2e-7, 'rtol': 2e-5},
    'post_v': {'atol': 2e-9, 'rtol': 2e-5},
}


def use_shape(shape):
    """Bind this run to one shape, once, and refuse a second one.

    One run compares one trajectory against another, so a second shape inside it
    would mean the counts and the registry changed underneath a comparison that
    had already started."""
    global SHAPE, PROFILE, N, COUNTS, _BOUND
    if _BOUND and shape != SHAPE:
        raise ValueError('one run compares one shape; this run is bound to '
                         f'{SHAPE.profile} and was asked for {shape.profile}')
    _BOUND = True
    SHAPE, PROFILE, N, COUNTS = shape, shape.profile, shape.n_total, shape.counts()
    return shape


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value, newline=True):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'),
                       ensure_ascii=True, allow_nan=False) + ('\n' if newline else '')).encode('ascii')


def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'duplicate JSON key')
        result[key] = value
    return result


def parse(raw):
    def finite_float(text):
        value = float(text)
        require(math.isfinite(value), 'nonfinite JSON number')
        return value
    value = json.loads(raw, object_pairs_hook=unique_pairs,
                       parse_float=finite_float,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
    # Bounded text size is enforced before parsing; recursion is bounded by the
    # JSON decoder and rejected as an invalid artifact by main.
    return value


def safe_path(path):
    path = Path(os.path.abspath(path))
    for component in (path, *path.parents):
        require(not component.is_symlink(), 'symlink component refused: ' + str(component))
    return path


def read(path, limit=MAX_FILE, size=None):
    path = safe_path(path)
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode) and before.st_size <= limit,
                'nonregular/oversized artifact: ' + str(path))
        require(size is None or before.st_size == size, 'wrong artifact size: ' + str(path))
        with os.fdopen(fd, 'rb', closefd=False) as handle:
            raw = handle.read(limit + 1)
        after = os.fstat(fd)
        require(len(raw) == before.st_size and (before.st_dev, before.st_ino, before.st_size,
                before.st_mtime_ns, before.st_ctime_ns) == (after.st_dev, after.st_ino,
                after.st_size, after.st_mtime_ns, after.st_ctime_ns), 'artifact changed during read')
        return raw
    finally:
        os.close(fd)


def tree_bound(root):
    root = safe_path(root)
    require(root.is_dir(), 'capture is not a directory')
    count = total = 0
    pending = [(root, 0)]
    while pending:
        parent, depth = pending.pop()
        # Stream entries so a pathological directory cannot allocate an
        # unbounded os.walk list before the count limit is checked.
        with os.scandir(parent) as entries:
            for entry in entries:
                count += 1
                require(count <= MAX_FILES, 'artifact count exceeds bound')
                info = entry.stat(follow_symlinks=False)
                require(not stat.S_ISLNK(info.st_mode), 'symlink artifact refused')
                if stat.S_ISDIR(info.st_mode):
                    require(depth < 2, 'unexpected artifact depth')
                    pending.append((Path(entry.path), depth + 1))
                else:
                    require(stat.S_ISREG(info.st_mode) and info.st_size <= MAX_FILE, 'bad artifact file')
                    total += info.st_size
                    require(total <= MAX_TREE, 'capture exceeds logical byte bound')
    return root


def registry():
    # The parameter tensors of the bound shape, in flat order. A capture's own
    # registry is admitted against this, and the capture harness admits the same
    # derivation against the native one, so what is checked here is still the
    # native layout rather than this file's opinion of it.
    result = SHAPE.registry()
    require(result[-1]['offset'] + result[-1]['count'] == N, 'internal registry mismatch')
    return result


def filename(key):
    return key + ('.i32' if key == 'ids' or key.endswith('flags') else '.f32')


def frozen_initial_bytes():
    """Reproduce only the declared host initialization, never model math."""
    values = bytearray(N * 4)
    for index in range(N):
        value = (index + 1) ^ 0x42595445
        value = (value ^ (value >> 16)) & 0xffffffff
        value = (value * 0x85ebca6b) & 0xffffffff
        value = (value ^ (value >> 13)) & 0xffffffff
        value = (value * 0xc2b2ae35) & 0xffffffff
        value = (value ^ (value >> 16)) & 0xffffffff
        struct.pack_into('<f', values, index * 4, ((value >> 24) - 128) / 1024.0)
    for entry in registry():
        if entry['name'].endswith(('norm1_w', 'norm2_w')):
            for index in range(entry['offset'], entry['offset'] + entry['count']):
                struct.pack_into('<f', values, index * 4, 1.0)
    return bytes(values)


def array_read(directory, key, descriptor=None):
    raw = read(directory / filename(key), size=COUNTS[key] * 4)
    if descriptor is not None:
        require(descriptor == dict(file=filename(key), count=COUNTS[key], sha256=sha(raw)),
                'array descriptor mismatch: ' + key)
    code = '<i' if key == 'ids' or key.endswith('flags') else '<f'
    for (value,) in struct.iter_unpack(code, raw):
        if key.endswith('flags'):
            require(value in (0, 1), 'invalid state flag')
        elif key == 'ids':
            require(0 <= value < 256, 'invalid byte token')
        else:
            require(math.isfinite(value), 'nonfinite FP32 artifact')
            if key.endswith('_v') or key == 'loss':
                require(value >= 0, 'negative variance/loss')
    return raw


def signature(metadata, raw_arrays, step):
    meta = dict(metadata, completed_steps=step, next_batch_index=step)
    return dict(arrays={key: sha(value) for key, value in raw_arrays.items()},
                metadata_sha256=sha(canonical(meta)), completed_steps=step)


def checkpoint(directory, name, descriptor):
    require(descriptor.get('file') == name, 'checkpoint filename mismatch')
    raw = read(directory / name, limit=2 * 1024 * 1024)
    require(descriptor.get('sha256') == sha(raw) and descriptor.get('bytes') == len(raw),
            'checkpoint byte witness mismatch')
    envelope = parse(raw)
    require(set(envelope) == {'schema', 'payload', 'payload_sha256'} and envelope['schema'] ==
            'mojolearn.small-byte-lm-json-checkpoint.v1', 'wrong checkpoint schema')
    payload = envelope['payload']
    require(sha(canonical(payload, False)) == envelope['payload_sha256'] and
            raw == canonical(envelope), 'noncanonical/corrupt checkpoint')
    arrays = {}
    metadata = dict(payload)
    for key in STATE_NAMES:
        item = metadata.pop(key)
        count = SHAPE.n_flags if key == 'flags' else N
        require(set(item) == {'dtype', 'shape', 'hex'} and item['shape'] == [count] and
                item['dtype'] == ('<i4' if key == 'flags' else '<f4') and
                isinstance(item['hex'], str) and len(item['hex']) == count * 8,
                'checkpoint tensor descriptor mismatch')
        arrays[key] = bytes.fromhex(item['hex'])
        require(len(arrays[key]) == count * 4, 'checkpoint hex length mismatch')
    return raw, metadata, arrays


def check_metadata(metadata, schedule):
    reg = registry()
    require(set(metadata) == {'schema', 'profile', 'numeric_mode', 'parameter_names',
            'parameter_shapes', 'parameter_offsets', 'completed_steps', 'next_batch_index',
            'config', 'data_schedule'}, 'wrong state metadata keys')
    require(metadata['schema'] == 'mojolearn.small-byte-lm-state.v1' and
            metadata['profile'] == PROFILE and metadata['numeric_mode'] == 'identical' and
            metadata['parameter_names'] == [r['name'] for r in reg] and
            metadata['parameter_shapes'] == [r['shape'] for r in reg] and
            metadata['parameter_offsets'] == [r['offset'] for r in reg] + [N] and
            metadata['data_schedule'] == schedule, 'wrong state metadata profile/schedule')
    f32 = lambda x: struct.unpack('<f', struct.pack('<f', x))[0]
    expected = dict(kind=2, lr=f32(.003), beta1=f32(.9), beta2=f32(.999), eps=f32(1e-8),
                    weight_decay=f32(.01), momentum=0., dampening=0., nesterov=False, max_norm=0.)
    require(metadata['config'] == expected, 'optimizer differs from fixed capture contract')


def initial_raw(directory):
    return {name: array_read(directory, 'initial_' + suffix) for name, suffix in STATE_NAMES.items()}


def heldout(root, name, expected, state, schedule_raw):
    # Held-out batches, their byte size and their starts all follow from the
    # bound shape. Every shape reads the same held-out target bytes, as fewer
    # batches of more rows or the reverse.
    batches, size, starts = SHAPE.validation_batches, SHAPE.n_ids * 4, SHAPE.validation_starts
    report = parse(read(root / name / 'evaluation.json', limit=65536))
    require(report == expected and report['state_before'] == state == report['state_after'] and
            report['state_unchanged'] is True and len(report['batches']) == batches,
            'heldout state/report mismatch')
    losses = []
    for index, batch in enumerate(report['batches']):
        ids = read(root / name / f'batch{index:02}.ids.i32', size=size)
        loss = read(root / name / f'batch{index:02}.loss.f32', size=4)
        value = struct.unpack('<f', loss)[0]
        require(math.isfinite(value) and value > 0 and batch['loss'] == value and
                batch['start'] == starts[index] and
                batch['ids_sha256'] == sha(ids) and batch['loss_sha256'] == sha(loss) and
                ids == schedule_raw[index * size:(index + 1) * size], 'heldout raw witness mismatch')
        losses.append(value)
    require(report['mean_loss'] == math.fsum(losses) / batches, 'heldout aggregation mismatch')
    return report


def validate_metal_host_runtime(runtime):
    host = runtime.get('host_runtime')
    require(isinstance(host, dict) and host.get('system') == 'Darwin' and host.get('machine') == 'arm64'
            and all(isinstance(host.get(k), str) and host[k] for k in ('release', 'python', 'macos_version')),
            'Metal capture requires explicit Darwin/arm64 host runtime')


def load_capture(path, expected_action, expected_vendor=None):
    root = tree_bound(path)
    summary_raw = read(root / 'summary.json', limit=1024 * 1024)
    summary = parse(summary_raw)
    require(summary['schema'] == 'mojolearn.byte-lm.real-text-capture.v1' and
            summary['action'] == expected_action, 'wrong capture action/schema')
    start, end = {'continuous': (0, 128), 'head64': (0, 64),
                  'resume128': (64, 128), 'zero-moments65': (64, 65)}[expected_action]
    require(summary['completed_steps'] == summary['expected_steps'] == end and
            len(summary['records']) == end - start, 'incomplete capture')
    runtime = parse(read(root / 'runtime.json', limit=65536))
    source = parse(read(root / 'source.json', limit=1024 * 1024))
    # TWO PROFILES, AND THEY ARE NOT THE SAME THING. `native_profile` is the
    # BINARY's identity, a constant compiled into the shared object, and it does
    # not vary with the shape because one binary runs every shape. The RUN's
    # identity is `profile`. Comparing the first against the bound shape is true
    # only at the default, which is why this held for every retained tree and
    # then refused a four-row capture that was in fact correct. Both now.
    require(runtime == summary['runtime']
            and runtime['native_profile'] == byte_lm_shape.DEFAULT_PROFILE
            and runtime.get('profile') == PROFILE and
            runtime['native_numeric_mode'] == 1 and (runtime['native_vendor'] in ('cuda', 'hip') or
                (runtime['native_vendor'] == expected_vendor == 'metal' and expected_action == 'continuous')) and
            (expected_vendor is None or runtime['native_vendor'] == expected_vendor), 'runtime witness mismatch')
    if runtime['native_vendor'] == 'metal':
        validate_metal_host_runtime(runtime)
    require(isinstance(source, dict) and 1 <= len(source) <= 2000 and
            all(isinstance(k, str) and len(k) <= 1024 and isinstance(v, str) and
                len(v) == 64 and all(c in '0123456789abcdef' for c in v) for k, v in source.items()) and
            {'tools/byte_lm_gradient_oracle.py', 'tools/byte_lm_real_text_capture.py',
             'training/byte_lm.mojo', 'bindings/_mojolearn_byte_lm.mojo'} <= source.keys(),
            'invalid/incomplete source inventory')
    require(isinstance(runtime['binding_sha256'], str) and len(runtime['binding_sha256']) == 64,
            'missing binding hash')
    loaded_sources = runtime['source_sha256']
    require(loaded_sources.get('loaded_python_wrapper') == source.get('python/mojolearn/_byte_lm_impl.py') and
            all(key == 'loaded_python_wrapper' or source.get(key) == value
                for key, value in loaded_sources.items()), 'loaded Python/direct source witness mismatch')
    schedule = summary['schedule']
    ids_bytes = SHAPE.n_ids * 4
    train = read(root / 'train-schedule.i32', size=128 * ids_bytes)
    evaluation = read(root / 'heldout-schedule.i32', size=SHAPE.validation_batches * ids_bytes)
    init = read(root / 'frozen-initial-parameters.f32', size=N * 4)
    require(init == frozen_initial_bytes(), 'initial parameters differ from declared deterministic formula')
    corpus = read(root / 'corpus-manifest.json', limit=65536)
    manifest = parse(corpus)
    require(summary['corpus_sha256'] == CORPUS_SHA == manifest['sha256'] and
            manifest['schema'] == 'mojolearn.byte-lm.corpus.v1' and
            manifest['bytes'] == 1115394 and manifest['train_range'] == [0, 65536] and
            manifest['validation_range'] == [65536, 73728] and
            manifest['vocabulary'] == SHAPE.vocab_size and manifest['batch'] == SHAPE.batch and
            manifest['context'] == SHAPE.length and
            manifest['planned_steps'] == 128 and
            manifest['validation_batch_starts'] == SHAPE.validation_starts and
            manifest['learning_gate']['heldout_mean_loss_ratio_max'] == .9,
            'wrong pinned corpus/heldout threshold')
    require(schedule == dict(schema='mojolearn.byte-lm.real-text-schedule.v1',
            corpus_sha256=CORPUS_SHA, corpus_manifest_sha256=sha(corpus),
            train_schedule_sha256=sha(train), heldout_schedule_sha256=sha(evaluation),
            planned_steps=128, initialization=INIT_ID, initial_parameters_sha256=sha(init)),
            'schedule witnesses mismatch')
    cp_raw, metadata, cp_arrays = checkpoint(root, 'final.checkpoint.json', summary['checkpoint'])
    check_metadata(metadata, schedule)
    require(runtime['profile'] == PROFILE and runtime['data_schedule'] == schedule and
            runtime['config'] == metadata['config'] and
            runtime['completed_steps'] == runtime['next_batch_index'] == start,
            'runtime state/config witness mismatch')
    require(metadata['completed_steps'] == metadata['next_batch_index'] == end,
            'checkpoint counters mismatch')
    initial = initial_raw(root / 'initial')
    initial_sig = signature(metadata, initial, start)
    require(initial_sig == summary['initial_state'] == parse(read(root / 'initial/state.json', limit=65536)),
            'initial state mismatch')
    if start == 0:
        require(initial['parameters'] == init and all(not any(initial[k]) for k in ('m', 'v', 'flags')),
                'fresh initialization/moments mismatch')
    incoming = None
    if start == 64:
        desc = summary['incoming_checkpoint']
        validate_checkpoint_capture_provenance(desc)
        incoming, in_meta, in_arrays = checkpoint(root, 'incoming.checkpoint.json', desc)
        require(in_meta == dict(metadata, completed_steps=64, next_batch_index=64), 'incoming metadata mismatch')
        if expected_action == 'resume128':
            require(in_arrays == initial, 'resume discarded incoming state')
    else:
        require(summary['incoming_checkpoint'] is None, 'unexpected incoming checkpoint')
    previous = initial
    steps = {}
    for number, record in zip(range(start + 1, end + 1), summary['records']):
        folder = f'step{number:06}'
        require(record['step'] == number and record['capture'] == folder, 'step sequence/path mismatch')
        raw = read(root / folder / 'capture.json', limit=65536)
        capture = parse(raw)
        require(sha(raw) == record['capture_sha256'] and capture['schema'] ==
                'mojolearn.byte-lm.gradient-capture.v1' and capture['registry'] == registry() and
                set(capture['arrays']) == set(COUNTS), 'step manifest mismatch')
        # DEVIATION 2682. A step may record `model_shape`, and when it does it
        # must be this run's shape, which the profile alone already pins because
        # the profile spells out all nine dimensions. A step that records none is
        # the default by construction, since that was the only shape that existed
        # when it was written.
        expected_config = dict(profile=PROFILE, numeric_mode='identical',
                vendor=runtime['native_vendor'], completed_steps=number - 1,
                post_completed_steps=number, optimizer=metadata['config'])
        if isinstance(capture['config'], dict) and 'model_shape' in capture['config']:
            expected_config['model_shape'] = SHAPE.to_json()
        require(capture['config'] == expected_config, 'step config mismatch')
        arrays = {key: array_read(root / folder, key, capture['arrays'][key]) for key in COUNTS}
        before = {k: arrays['initial_' + suffix] for k, suffix in STATE_NAMES.items()}
        after = {k: arrays['post_' + suffix] for k, suffix in STATE_NAMES.items()}
        require(before == previous and capture['input_state'] == signature(metadata, before, number - 1) and
                capture['output_state'] == signature(metadata, after, number), 'broken full-state chain')
        require(arrays['ids'] == train[(number - 1) * ids_bytes:number * ids_bytes] and
                record['loss'] == struct.unpack('<f', arrays['loss'])[0], 'token/loss record mismatch')
        steps[number] = dict(directory=root / folder, manifest=capture, sha256=sha(raw))
        previous = after
    require(previous == cp_arrays and summary['final_state'] == signature(metadata, previous, end),
            'terminal checkpoint differs from retained state')
    if expected_action != 'zero-moments65':
        first = heldout(root, 'heldout-initial', summary['initial_heldout'], initial_sig, evaluation)
        last = heldout(root, 'heldout-final', summary['final_heldout'], summary['final_state'], evaluation)
        ratio = last['mean_loss'] / first['mean_loss']
        require(summary['learning']['ratio'] == ratio and summary['learning']['threshold'] == .9,
                'learning ratio/threshold mismatch')
    else:
        require(summary['initial_heldout'] is None and summary['final_heldout'] is None,
                'control should have no heldout claims')
        ratio = None
    return dict(root=root, summary=summary, summary_sha256=sha(summary_raw), runtime=runtime,
                source=source, metadata=metadata, steps=steps, initial=initial,
                checkpoint=cp_raw, incoming=incoming, ratio=ratio)


def compatible(a, b):
    require(a['source'] == b['source'], 'source inventories differ')
    if a['runtime']['native_vendor'] == b['runtime']['native_vendor']:
        require(a['runtime']['binding_sha256'] == b['runtime']['binding_sha256'],
                'same-vendor binding hashes differ')
    require(a['summary']['schedule'] == b['summary']['schedule'], 'dataset/token/init witnesses differ')
    require(dict(a['metadata'], completed_steps=0, next_batch_index=0) ==
            dict(b['metadata'], completed_steps=0, next_batch_index=0), 'state configuration differs')
    for filename_ in ('corpus-manifest.json', 'train-schedule.i32',
                      'heldout-schedule.i32', 'frozen-initial-parameters.f32'):
        require(read(a['root'] / filename_) == read(b['root'] / filename_), 'raw inputs differ')


def same_steps(a, b):
    compatible(a, b)
    common = sorted(a['steps'].keys() & b['steps'].keys())
    require(common, 'no overlapping steps')
    for number in common:
        left, right = a['steps'][number], b['steps'][number]
        for key in COUNTS:
            require(read(left['directory'] / filename(key), size=COUNTS[key] * 4) ==
                    read(right['directory'] / filename(key), size=COUNTS[key] * 4),
                    f'raw state mismatch at step {number}: {key}')
        require(left['manifest']['input_state'] == right['manifest']['input_state'] and
                left['manifest']['output_state'] == right['manifest']['output_state'], 'step state metadata differs')
    return len(common)


def validate_checkpoint_capture_provenance(desc):
    """Accept original sealed Linux captures or the explicit immutable-bytes v1 capture."""
    if not isinstance(desc, dict):
        raise ValueError('Missing incoming checkpoint provenance')
    if desc.get('loaded_from_sealed_capture') is True:
        require('loaded_from_immutable_bytes' not in desc and 'capture_method' not in desc,
                'contradictory sealed checkpoint provenance')
    else:
        require(desc.get('loaded_from_sealed_capture') is False
                and desc.get('loaded_from_immutable_bytes') is True
                and desc.get('capture_method') == 'bounded-read-once-immutable-bytes.v1',
                'missing explicit immutable checkpoint provenance')


def receipt(path, artifact_hash, vendor, job_kind='capture'):
    raw = read(path, limit=65536)
    report = parse(raw)
    require(report['schema'] == 'mojolearn.root-job-receipt.v1' and
            report['vendor'] == vendor and report['job_kind'] == job_kind and
            type(report['exit_code']) is int and report['exit_code'] == 0,
            'guard receipt linkage/exit mismatch')
    retained = {}
    for key, limit in (('command', 16384), ('guard_log', 8 * 1024 * 1024), ('result', 2 * 1024 * 1024)):
        descriptor = report[key]
        name = descriptor['file']
        require(isinstance(name, str) and 0 < len(name) <= 1024 and not Path(name).is_absolute() and
                all(part not in ('.', '..') for part in name.split('/')) and
                len(Path(name).parts) <= 8, 'unsafe receipt artifact path')
        retained[key] = read(Path(path).parent / name, limit=limit)
        require(set(descriptor) == {'file', 'sha256', 'bytes'} and
                descriptor['sha256'] == sha(retained[key]) and descriptor['bytes'] == len(retained[key]),
                'receipt retained artifact mismatch: ' + key)
    require(retained['command'].strip() and sha(retained['result']) == artifact_hash,
            'empty command/wrong receipt result')
    log = retained['guard_log']
    require(log.strip(), 'empty guard log')
    final = parse(log.rstrip().split(b'\n')[-1])
    require(report['guard_terminal'] == final, 'terminal guard differs from bound receipt')
    validate_guard_terminal(final, vendor, report['exit_code'], job_kind)
    return dict(receipt_sha256=sha(raw), guard_log_sha256=sha(log), final=final,
                command=report['command'], result=report['result'],
                linkage='root-supplied artifact/command pairing; not a signed attestation')


def oracle(path, capture, guard_path):
    raw = read(path, limit=2 * 1024 * 1024)
    report = parse(raw)
    vendor = capture['runtime']['native_vendor']
    require(report['schema'] == 'mojolearn.byte-lm.gradient-oracle.v1' and report['passed'] is True and
            report['profile'] == PROFILE and report['vendor'] == vendor and
            report['oracle_source_sha256'] == capture['source']['tools/byte_lm_gradient_oracle.py'],
            'independent oracle failed/source mismatch')
    require(report['tolerances'] == ORACLE_TOLERANCES, 'independent oracle tolerances changed')
    step = report['config']['post_completed_steps']
    require(step in capture['steps'], 'oracle step not in capture')
    witness = capture['steps'][step]
    require(report['capture_manifest_sha256'] == witness['sha256'] and
            report['config'] == witness['manifest']['config'] and
            report['input_array_sha256'] == {k: d['sha256'] for k, d in witness['manifest']['arrays'].items()},
            'oracle capture linkage mismatch')
    names = {r['name'] for r in registry()}
    require(set(report['gradients']) == names and all(v['passed'] is True for v in report['gradients'].values()) and
            report['loss']['passed'] is True and report['flags_preserved'] is True and
            report['parameters_moved'] is True and report['controls']['sign_effective'] is True and
            report['controls']['nonlinear_effective'] is True, 'incomplete independent gradient gate')
    require(set(report['optimizer_updates']) == {'post_p', 'post_m', 'post_v'} and
            all(set(group) == names and all(v['passed'] is True for v in group.values())
                for group in report['optimizer_updates'].values()), 'incomplete independent optimizer gate')
    reference = report['reference_file']
    name = reference['file']
    require(isinstance(name, str) and Path(name).name == name and name not in ('.', '..'), 'unsafe oracle reference path')
    require(sha(read(Path(path).parent / name)) == reference['sha256'], 'missing/corrupt raw oracle references')
    return dict(oracle_sha256=sha(raw), step=step, reference=reference,
                guard=receipt(guard_path, sha(raw), vendor, 'oracle'))


def resume_control(head, resumed, control, continuous):
    require(head['runtime']['native_vendor'] != resumed['runtime']['native_vendor'],
            'resume must transfer across vendors')
    require(resumed['runtime']['native_vendor'] == control['runtime']['native_vendor'],
            'control vendor differs from legitimate resume')
    for other in (resumed, control):
        compatible(head, other)
        require(other['incoming'] == head['checkpoint'], 'actual transferred checkpoint bytes differ')
    require(resumed['checkpoint'] == continuous['checkpoint'], 'resumed terminal checkpoint differs')
    same_steps(continuous, head)
    same_steps(continuous, resumed)
    for left, left_folder, right, right_folder in (
            (head, 'heldout-initial', continuous, 'heldout-initial'),
            (head, 'heldout-final', resumed, 'heldout-initial'),
            (resumed, 'heldout-final', continuous, 'heldout-final')):
        for index in range(SHAPE.validation_batches):
            for suffix in ('ids.i32', 'loss.f32'):
                name = f'batch{index:02}.{suffix}'
                require(read(left['root'] / left_folder / name) == read(right['root'] / right_folder / name),
                        'resume heldout byte chain differs')
    legit = initial_raw(control['root'] / 'legitimate-head64')
    head_final = {k: read(head['steps'][64]['directory'] / filename('post_' + v),
                          size=(SHAPE.n_flags if k == 'flags' else N) * 4) for k, v in STATE_NAMES.items()}
    require(legit == head_final and control['initial']['parameters'] == legit['parameters'] and
            control['initial']['flags'] == legit['flags'] and all(any(legit[k]) for k in ('m', 'v')) and
            all(not any(control['initial'][k]) for k in ('m', 'v')), 'ineffective/malformed planted moment control')
    evidence = control['summary']['control']
    require(evidence['name'] == 'zero-moments65' and evidence['changed_fields'] == ['m', 'v'] and
            evidence['legitimate_state_directory'] == 'legitimate-head64' and
            evidence['legitimate_state'] == head['summary']['final_state'] ==
            parse(read(control['root'] / 'legitimate-head64/state.json', limit=65536)) and
            evidence['altered_state'] == control['summary']['initial_state'], 'control state provenance mismatch')
    good, bad = resumed['steps'][65]['directory'], control['steps'][65]['directory']
    for key in ('initial_p', 'initial_flags', 'ids', 'loss', 'grad', 'post_flags'):
        require(read(good / filename(key)) == read(bad / filename(key)), 'control changed forward/backward/counters')
    for key in ('post_p', 'post_m', 'post_v'):
        require(read(good / filename(key)) != read(bad / filename(key)), 'missing-moments control ineffective: ' + key)
    return dict(transferred_checkpoint_sha256=sha(head['checkpoint']), effective=True,
                compared_legitimate_step=65, changed_post_arrays=['post_p', 'post_m', 'post_v'])


def mappings(values, allowed):
    result = {}
    for value in values:
        label, separator, path = value.partition('=')
        require(separator and label in allowed and label not in result and path, 'invalid/duplicate receipt mapping')
        result[label] = Path(path)
    return result


def compare_continuous(a, b):
    count = same_steps(a, b)  # includes full source-inventory/input/config equality
    require(count == 128 and a['checkpoint'] == b['checkpoint'],
            'full continuous checkpoint comparison failed')
    # Heldout raw bytes are included, independently of learning admission.
    for folder in ('heldout-initial', 'heldout-final'):
        for index in range(SHAPE.validation_batches):
            for suffix in ('ids.i32', 'loss.f32'):
                name = f'batch{index:02}.{suffix}'
                require(read(a['root'] / folder / name) ==
                        read(b['root'] / folder / name), 'heldout raw bytes differ')
    return count


def compare(args):
    # DEVIATION 2682. One run compares one shape, because identity is claimed per
    # shape. Absent, it is the certified default, which is every retained tree.
    use_shape(byte_lm_shape.parse(getattr(args, 'shape', None)))
    captures = {name: load_capture(getattr(args, name), 'continuous', name) for name in ('cuda', 'hip')}
    continuous_vendors = ['cuda', 'hip']
    count = compare_continuous(captures['cuda'], captures['hip'])
    if getattr(args, 'metal', None) is not None:
        captures['metal'] = load_capture(args.metal, 'continuous', 'metal')
        compare_continuous(captures['cuda'], captures['metal'])
        continuous_vendors.append('metal')
    missing = []
    control_result = None
    for name, action in (('head', 'head64'), ('resume', 'resume128'), ('control', 'zero-moments65')):
        if getattr(args, name):
            captures[name] = load_capture(getattr(args, name), action)
            compatible(captures[name], captures[captures[name]['runtime']['native_vendor']])
        else:
            missing.append('missing ' + name + ' capture')
    if all(name in captures for name in ('head', 'resume', 'control')):
        control_result = resume_control(captures['head'], captures['resume'], captures['control'],
                                        captures[captures['resume']['runtime']['native_vendor']])
    guards = mappings(args.guard, set(captures))
    guard_results = {}
    for name, capture in captures.items():
        if name not in guards:
            missing.append('missing successful guard receipt: ' + name)
        else:
            guard_results[name] = receipt(guards[name], capture['summary_sha256'], capture['runtime']['native_vendor'])
    oracles = mappings(args.oracle, {'cuda', 'hip'})
    oracle_guards = mappings(args.oracle_guard, {'cuda', 'hip'})
    oracle_results = {}
    for vendor in ('cuda', 'hip'):
        if vendor not in oracles or vendor not in oracle_guards:
            missing.append('missing independently guarded gradient oracle: ' + vendor)
        else:
            oracle_results[vendor] = oracle(oracles[vendor], captures[vendor], oracle_guards[vendor])
    learning_observed = all(captures[v]['ratio'] <= .9 for v in continuous_vendors)
    admitted = not missing
    return dict(schema='mojolearn.byte-lm.state-comparison.v1',
                status='QUALIFIED_BOUNDED_CAPTURE' if admitted and learning_observed else 'AGREEMENT_ONLY_DIAGNOSTIC',
                profile=PROFILE, compared_continuous_steps=count, raw_continuous_agreement=True,
                continuous_vendors=continuous_vendors,
                independent_oracle_vendors=list(oracle_results),
                resume_direction=(dict(source=captures['head']['runtime']['native_vendor'],
                                       destination=captures['resume']['runtime']['native_vendor'])
                                  if control_result is not None else None),
                metal_checkpoint_resume_admitted=False,
                identity_admitted=admitted, learning_observed_gate_passed=learning_observed,
                learning_admitted=admitted and learning_observed, learning_ratio_max=.9,
                prerequisites_missing=missing, control=control_result, guards=guard_results, oracles=oracle_results,
                captures={name: dict(summary_sha256=c['summary_sha256'], runtime=c['runtime'],
                                    source=c['source'], ratio=c['ratio']) for name, c in captures.items()},
                scope='Only these retained FP32 byte-LM trajectories and heldout batches; no universal certificate, '
                      'external bitwise claim, generation-quality claim, or speed measurement. Optional Metal '
                      'covers continuous agreement only; resume_direction names the separately checked transfer.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--hip', type=Path, required=True)
    parser.add_argument('--shape', default=None,
                        help='batch,length or the nine dimensions; the default is the certified b2-l32')
    parser.add_argument('--metal', type=Path,
                        help='optional continuous128 Metal capture; requires guard metal=FILE, no Metal oracle/resume claim')
    for name in ('head', 'resume', 'control'):
        parser.add_argument('--' + name, type=Path)
    for name in ('guard', 'oracle', 'oracle-guard'):
        parser.add_argument('--' + name, action='append', default=[], metavar='LABEL=FILE')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    # Reserve before reading artifacts, retaining a clear failure report.
    with safe_path(args.output).open('xb') as output:
        try:
            result = compare(args)
        except (ValueError, OSError, KeyError, TypeError, AttributeError, IndexError,
                RecursionError, OverflowError) as error:
            result = dict(schema='mojolearn.byte-lm.state-comparison.v1', status='REJECTED',
                          identity_admitted=False, learning_admitted=False, error=str(error))
        result['comparator_source_sha256'] = sha(read(__file__, limit=1024 * 1024))
        result['receipt_policy_source_sha256'] = sha(read(
            Path(__file__).with_name('root_job_receipt.py'), limit=1024 * 1024))
        output.write(canonical(result))
        output.flush()
        os.fsync(output.fileno())
    return 0 if result.get('identity_admitted') and result.get('learning_admitted') else 1


if __name__ == '__main__':
    raise SystemExit(main())
