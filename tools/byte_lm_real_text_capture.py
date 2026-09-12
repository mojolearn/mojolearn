#!/usr/bin/env python3
"""Root-only bounded real-byte LM capture; authored, not executed.

No provisioning, compiler, subprocess, timing, or inline reference model.
Invoke under the corresponding NVIDIA/AMD/macOS serial guard, IDENTICAL mode selected.
Steps1 produces one independent-oracle-compatible capture without a learning
claim. Full128 records initial/final fixed heldout loss and its preset gate.
Head64/resume128 retain actual checkpoint transfer bytes but do not alone
establish cross-vendor identity or admit a full initial-to-final learning claim.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
from pathlib import Path
import stat
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
CORPUS_SHA = '86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed'
CORPUS_BYTES = 1115394
INIT_ID = 'u32-avalanche-index-xor-42595445-top8-centered128-div1024-norm1.v1'

# DEVIATION 2682. The shape was nine literals spread through this file and four
# verifiers. It now comes from one module, loaded by path because this tool runs
# from a leased host where the tools directory is not on the import path. The
# helper asserts at import that its default derivation equals the literals the
# certified b2-l32 run was admitted with, so threading a second shape through
# here cannot move the first one.
_shape_spec = importlib.util.spec_from_file_location(
    '_byte_lm_shape', Path(__file__).with_name('byte_lm_shape.py'))
byte_lm_shape = importlib.util.module_from_spec(_shape_spec)
_shape_spec.loader.exec_module(byte_lm_shape)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False) + '\n').encode()


def exclusive(path, raw):
    with Path(path).open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def read_corpus(shape):
    """The pinned corpus and the schedule manifest OF THIS SHAPE.

    Each shape has its own manifest file, so neither run can read the other's
    schedule, and the corpus bytes are the same pinned file for both."""
    path = ROOT / 'training/corpus/tinyshakespeare/input.txt'
    manifest_path = path.with_name(shape.manifest_name)
    with manifest_path.open('rb') as stream:
        manifest_raw = stream.read(65537)
    if len(manifest_raw) > 65536:
        raise ValueError('corpus manifest exceeds bound')
    manifest = json.loads(manifest_raw)
    expected = shape.manifest_fields(corpus_sha=CORPUS_SHA, corpus_bytes=CORPUS_BYTES)
    if any(manifest.get(k) != v for k, v in expected.items()):
        raise ValueError('pinned corpus/schedule manifest differs')
    if manifest.get('learning_gate', {}).get('heldout_mean_loss_ratio_max') != .9:
        raise ValueError('predeclared learning threshold changed')
    with path.open('rb') as stream:
        raw = stream.read(CORPUS_BYTES + 1)
    if len(raw) != CORPUS_BYTES or sha(raw) != CORPUS_SHA:
        raise ValueError('pinned corpus length/SHA mismatch')
    return raw, manifest, manifest_raw


def train_ids(raw, step, shape):
    import numpy as np
    rows = []
    for b in range(shape.batch):
        start = shape.train_start(step, b)
        width = shape.length + 1
        rows.append(np.frombuffer(raw[start:start + width], dtype=np.uint8).astype(np.int32))
    return np.stack(rows)


def heldout_ids(raw, start, shape):
    import numpy as np
    width = shape.length + 1
    return np.stack([np.frombuffer(raw[start + b * shape.length:start + b * shape.length + width],
                                  dtype=np.uint8).astype(np.int32) for b in range(shape.batch)])


def validate_registry(registry, shape):
    """The public registry must equal the one derived from the nine dimensions.

    This is the cross-check of the native surface against an independent
    derivation, so the comparison stays, with the derivation moved into the
    shared module rather than spelled out a second time here."""
    expected = shape.registry()
    if registry != expected or sum(e['count'] for e in expected) != shape.n_total:
        raise ValueError(f'public registry differs from the {shape.profile} profile')


def initialize(registry, shape):
    """Exact integer/dyadic initializer; every resulting FP32 bit is retained."""
    import numpy as np
    values = np.empty(shape.n_total, dtype=np.float32)
    for i in range(shape.n_total):
        h = (i + 1) ^ 0x42595445
        h = (h ^ (h >> 16)) & 0xffffffff
        h = (h * 0x85ebca6b) & 0xffffffff
        h = (h ^ (h >> 13)) & 0xffffffff
        h = (h * 0xc2b2ae35) & 0xffffffff
        h = (h ^ (h >> 16)) & 0xffffffff
        values[i] = ((h >> 24) - 128) / 1024.0
    for entry in registry:
        if entry['name'].endswith(('norm1_w', 'norm2_w')):
            values[entry['offset']:entry['offset'] + entry['count']] = 1.0
    return values


def array_bytes(value, integer=False):
    import numpy as np
    dtype = np.dtype('int32' if integer else 'float32')
    # DEVIATION 2464: trainer buffers are mojolearn.Array; the zero-copy NumPy
    # view keeps the dtype test exact (no conversion happens in asarray).
    if hasattr(value, '__array_interface__') and not isinstance(value, np.ndarray):
        value = np.asarray(value)
    if not isinstance(value, np.ndarray) or value.dtype != dtype:
        raise TypeError('capture requires actual int32/float32 arrays; no silent dtype conversion')
    if not integer and not np.isfinite(value).all():
        raise ValueError('capture refuses nonfinite FP32 cells')
    return np.asarray(value, dtype='<i4' if integer else '<f4', order='C').tobytes()


def state_signature(state, shape):
    import numpy as np
    for key in ('parameters', 'm', 'v', 'flags'):
        expected_shape = (shape.n_flags,) if key == 'flags' else (shape.n_total,)
        if hasattr(state[key], '__array_interface__'):  # DEVIATION 2464
            state[key] = np.asarray(state[key])
        if not isinstance(state[key], np.ndarray) or state[key].shape != expected_shape:
            raise ValueError('state array shape differs from fixed profile')
    arrays = {key: sha(array_bytes(state[key], key == 'flags'))
              for key in ('parameters', 'm', 'v', 'flags')}
    metadata = {key: value for key, value in state.items() if key not in arrays}
    return dict(arrays=arrays, metadata_sha256=sha(canonical(metadata)),
                completed_steps=state['completed_steps'])


def retain_initial(directory, state, shape):
    directory.mkdir()
    paths = {}
    for source, target in (('parameters', 'initial_p'), ('m', 'initial_m'),
                           ('v', 'initial_v'), ('flags', 'initial_flags')):
        path = directory / (target + ('.i32' if source == 'flags' else '.f32'))
        exclusive(path, array_bytes(state[source], source == 'flags'))
        paths[source] = path
    exclusive(directory / 'state.json', canonical(state_signature(state, shape)))
    return paths


def retain_step(directory, before, after, result, ids, previous, registry, vendor, shape):
    import numpy as np
    width = shape.length + 1
    if not isinstance(ids, np.ndarray) or ids.shape != (shape.batch, width) or ids.dtype != np.int32:
        raise ValueError(f'step capture requires actual int32[{shape.batch},{width}] IDs')
    gradient = result['flat_gradients']
    if hasattr(gradient, '__array_interface__'):  # DEVIATION 2464: zero-copy view
        gradient = np.asarray(gradient)
    if not isinstance(gradient, np.ndarray) or gradient.shape != (shape.n_total,) or gradient.dtype != np.float32:
        raise ValueError(f'step capture requires actual float32[{shape.n_total}] gradients')
    if not math.isfinite(result['loss']) or float(np.float32(result['loss'])) != result['loss']:
        raise ValueError('step loss is not an exact finite FP32 value')
    directory.mkdir()
    descriptors = {}
    for source, target in (('parameters', 'initial_p'), ('m', 'initial_m'),
                           ('v', 'initial_v'), ('flags', 'initial_flags')):
        raw = array_bytes(before[source], source == 'flags')
        with previous[source].open('rb') as stream:
            retained_raw = stream.read(len(raw) + 1)
        if retained_raw != raw:
            raise ValueError('retained previous state differs from actual step input')
        path = directory / (target + ('.i32' if source == 'flags' else '.f32'))
        os.link(previous[source], path)  # exclusive immutable evidence alias
        descriptors[target] = dict(file=path.name, count=len(raw) // 4, sha256=sha(raw))
    new_previous = {}
    arrays = dict(post_p=after['parameters'], post_m=after['m'], post_v=after['v'],
                  post_flags=after['flags'], grad=result['flat_gradients'],
                  loss=np.asarray([result['loss']], dtype=np.float32), ids=ids.reshape(-1))
    for key, value in arrays.items():
        integer = key in ('ids', 'post_flags')
        raw = array_bytes(value, integer)
        path = directory / (key + ('.i32' if integer else '.f32'))
        exclusive(path, raw)
        descriptors[key] = dict(file=path.name, count=len(raw) // 4, sha256=sha(raw))
        for source, target in (('parameters', 'post_p'), ('m', 'post_m'), ('v', 'post_v'), ('flags', 'post_flags')):
            if key == target:
                new_previous[source] = path
    # DEVIATION 2682: a non-default shape records `model_shape`, so a reader
    # knows which shape it is holding rather than inferring it from an array
    # length. THE DEFAULT RECORDS NOTHING, deliberately. Absence already means
    # the default, because that was the only shape that existed when every
    # retained capture was written, and writing the key for the default would
    # change the bytes of a file every existing reader already validates. A
    # certified shape's capture.json stays exactly what it was.
    config = dict(profile=shape.profile, numeric_mode='identical', vendor=vendor,
                  completed_steps=before['completed_steps'], post_completed_steps=after['completed_steps'],
                  optimizer=before['config'])
    if shape != byte_lm_shape.Shape():
        config['model_shape'] = shape.to_json()
    manifest = dict(schema='mojolearn.byte-lm.gradient-capture.v1', registry=registry,
                    config=config, arrays=descriptors,
                    input_state=state_signature(before, shape), output_state=state_signature(after, shape))
    exclusive(directory / 'capture.json', canonical(manifest))
    return new_previous, sha(canonical(manifest))


def evaluate_heldout(trainer, raw, directory, shape):
    """Fixed held-out batches; arithmetic mean of their FP32 batch losses.

    Every shape reads the same number of target bytes over the same region of
    the corpus, as fewer batches of more rows or the reverse, so two shapes'
    held-out losses are means over the same text."""
    import numpy as np
    directory.mkdir()
    starts = shape.validation_starts
    before = state_signature(trainer.state_dict(), shape)
    losses, records = [], []
    for index, start in enumerate(starts):
        ids = heldout_ids(raw, start, shape)
        value = trainer.evaluate(ids)
        if not math.isfinite(value):
            raise ValueError('nonfinite heldout loss')
        after = state_signature(trainer.state_dict(), shape)
        if after != before:
            raise ValueError('evaluation changed state/config/cursor')
        token_raw = array_bytes(ids, True)
        loss_raw = array_bytes(np.asarray([value], dtype=np.float32))
        exclusive(directory / f'batch{index:02d}.ids.i32', token_raw)
        exclusive(directory / f'batch{index:02d}.loss.f32', loss_raw)
        losses.append(value)
        records.append(dict(start=start, ids_sha256=sha(token_raw), loss_sha256=sha(loss_raw), loss=value))
    count = len(starts)
    result = dict(batches=records, state_before=before,
                  state_after=state_signature(trainer.state_dict(), shape),
                  state_unchanged=True, mean_loss=math.fsum(losses) / count,
                  aggregation=f'math.fsum of {count} FP32 batch means / {count}; '
                              f'{byte_lm_shape.VALIDATION_TARGETS} targets')
    exclusive(directory / 'evaluation.json', canonical(result))
    return result


def save_exclusive_checkpoint(trainer, path):
    # The public writer replaces its path. Give it a new private directory, then
    # publish by exclusive hard link so previous user evidence is never replaced.
    with tempfile.TemporaryDirectory(prefix='.checkpoint-', dir=path.parent) as temp:
        candidate = Path(temp) / 'state.json'
        trainer.save_checkpoint(candidate)
        raw = candidate.read_bytes()
        if len(raw) > 2 * 1024 * 1024:
            raise ValueError('checkpoint exceeds fixed public bound')
        os.link(candidate, path)
    return dict(file=path.name, bytes=len(raw), sha256=sha(raw))


def load_foreign_checkpoint(cls, source, destination, *, resident=False):
    # Capture one bounded regular inode into immutable bytes. Decode and hash
    # the exact same object without reopening a caller-controlled path.
    fd = os.open(source, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 2 * 1024 * 1024:
            raise ValueError('incoming checkpoint must be a bounded regular file')
        chunks, size = [], 0
        while size <= 2 * 1024 * 1024:
            chunk = os.read(fd, min(65536, 2 * 1024 * 1024 + 1 - size))
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
        after = os.fstat(fd)
        if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (after.st_size, after.st_mtime_ns, after.st_ctime_ns):
            raise ValueError('incoming checkpoint changed while captured')
        raw = b''.join(chunks)
        if len(raw) != before.st_size:
            raise ValueError('incoming checkpoint length changed')
    finally:
        os.close(fd)
    trainer = cls.from_checkpoint_bytes(raw, resident=resident)
    exclusive(destination, raw)
    return trainer, dict(file=destination.name, bytes=len(raw), sha256=sha(raw),
                        loaded_from_sealed_capture=False, loaded_from_immutable_bytes=True,
                        capture_method='bounded-read-once-immutable-bytes.v1')


def validate_platform_vendor(expected_vendor):
    admitted = (sys.platform == 'linux' and expected_vendor in ('cuda', 'hip')) or (
        sys.platform == 'darwin' and expected_vendor == 'metal')
    if not admitted or os.environ.get('MOJOLEARN_NUMERIC_MODE') != 'identical':
        raise ValueError('requires explicit IDENTICAL and matching Linux CUDA/HIP or Darwin Metal')


SOURCE_DIRECTORIES = ('checks', 'core', 'gemm', 'embedding', 'transformer', 'training', 'mamba')
REQUIRED_MAMBA_SOURCES = (
    'mamba/impl/modeling/modeling_mamba.mojo',
    'mamba/impl/ops/selective_scan_interface.mojo',
    'mamba/checks/mamba_fixture.mojo',
)


def source_inventory():
    # Llama forward/backward import pinned_mul/residual helpers from Mamba;
    # that module also imports its scan interface and fixture descriptors.
    # An absent transport subtree must not silently produce a smaller proof.
    for name in REQUIRED_MAMBA_SOURCES:
        if not (ROOT / name).is_file():
            raise ValueError('Missing required transitive byte-LM source: ' + name)
    paths = []
    for name in SOURCE_DIRECTORIES:
        included = list((ROOT / name).rglob('*.mojo'))
        if not included:
            raise ValueError('Missing or empty byte-LM source directory: ' + name)
        paths.extend(included)
    paths.extend(ROOT / name for name in (
        'bindings/_mojolearn_byte_lm.mojo', 'bindings/build_byte_lm.sh',
        'python/mojolearn/_byte_lm_impl.py', 'python/mojolearn/language_model.py',
        'tools/byte_lm_real_text_capture.py', 'tools/byte_lm_gradient_oracle.py',
        'tools/byte_lm_shape.py', 'pixi.toml', 'pixi.lock'))
    return {str(path.relative_to(ROOT)): sha(path.read_bytes()) for path in sorted(set(paths))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-vendor', choices=('cuda', 'hip', 'metal'), required=True)
    parser.add_argument('--steps', type=int, choices=(1, 128), default=128)
    parser.add_argument('--action', choices=('continuous', 'head64', 'resume128', 'zero-moments65'), default='continuous')
    parser.add_argument('--resume-checkpoint', type=Path)
    parser.add_argument('--resident', action=argparse.BooleanOptionalAction, default=None,
                        help='retain GPU model/optimizer across calls (default: on for Metal, off for CUDA/HIP)')
    parser.add_argument('--shape', default=None,
                        help='DEVIATION 2682: batch,length or nine dimensions. '
                             'Omitted is the certified b2-l32 profile. Each shape '
                             'is a SEPARATE certificate, because nine weight '
                             'gradients contract over the token count.')
    args = parser.parse_args()
    shape = byte_lm_shape.parse(args.shape)
    resident = args.expected_vendor == 'metal' if args.resident is None else args.resident
    validate_platform_vendor(args.expected_vendor)
    if (args.action in ('resume128', 'zero-moments65')) != (args.resume_checkpoint is not None) or (args.steps == 1 and args.action != 'continuous'):
        raise ValueError('head64 checkpoint required exactly for resume128/zero-moments65; steps1 only continuous')
    raw, corpus_manifest, corpus_manifest_raw = read_corpus(shape)
    args.output.mkdir(parents=False, exist_ok=False)
    exclusive(args.output / 'corpus-manifest.json', corpus_manifest_raw)
    import numpy as np
    from mojolearn import ByteLanguageModelConfig
    from mojolearn.language_model import SmallByteLanguageModelTrainer
    # The native surface takes its own config object. Building it from the same
    # nine integers is what makes the derived registry here and the registry the
    # trainer reports comparable rather than two independent guesses.
    native_shape = ByteLanguageModelConfig(**shape.to_json())
    registry = [dict(name=x['name'], shape=list(x['shape']), offset=x['offset'], count=x['size'])
                for x in SmallByteLanguageModelTrainer.parameter_registry(shape=native_shape)]
    validate_registry(registry, shape)
    initialization = initialize(registry, shape)
    init_raw = array_bytes(initialization)
    exclusive(args.output / 'frozen-initial-parameters.f32', init_raw)
    full_schedule = b''.join(array_bytes(train_ids(raw, step, shape), True) for step in range(128))
    heldout_schedule = b''.join(array_bytes(heldout_ids(raw, start, shape), True)
                                for start in shape.validation_starts)
    exclusive(args.output / 'train-schedule.i32', full_schedule)
    exclusive(args.output / 'heldout-schedule.i32', heldout_schedule)
    schedule = dict(schema='mojolearn.byte-lm.real-text-schedule.v1', corpus_sha256=CORPUS_SHA,
                    corpus_manifest_sha256=sha(corpus_manifest_raw), train_schedule_sha256=sha(full_schedule),
                    heldout_schedule_sha256=sha(heldout_schedule), planned_steps=128,
                    initialization=INIT_ID, initial_parameters_sha256=sha(init_raw))
    incoming = None
    if args.action in ('resume128', 'zero-moments65'):
        trainer, incoming = load_foreign_checkpoint(SmallByteLanguageModelTrainer,
                             args.resume_checkpoint, args.output / 'incoming.checkpoint.json', resident=resident)
    else:
        trainer = SmallByteLanguageModelTrainer(initialization, data_schedule=schedule,
                       lr=.003, betas=(.9, .999), eps=1e-8, weight_decay=.01,
                       shape=native_shape, resident=resident)
    state = trainer.state_dict()
    expected_opt = dict(kind=2, lr=float(np.float32(.003)), beta1=float(np.float32(.9)),
        beta2=float(np.float32(.999)), eps=float(np.float32(1e-8)), weight_decay=float(np.float32(.01)),
        momentum=0., dampening=0., nesterov=False, max_norm=0.)
    expected_start = 64 if args.action in ('resume128', 'zero-moments65') else 0
    if (state['profile'] != shape.profile or state['data_schedule'] != schedule or state['config'] != expected_opt
        or state['completed_steps'] != expected_start or state['next_batch_index'] != expected_start):
        raise ValueError('starting state/configuration/profile/data cursor differs from fixed run')
    control = None
    if args.action == 'zero-moments65':
        # The original loaded state/configuration/cursor have already passed
        # fixed-run admission. Retain it before constructing the deliberate
        # mutation. No weight, flag, counter or schedule field is changed.
        legitimate = state_signature(state, shape)
        retain_initial(args.output / 'legitimate-head64', state, shape)
        if not np.any(np.asarray(state['m']) != 0) or not np.any(np.asarray(state['v']) != 0):  # DEVIATION 2464
            raise ValueError('zero-moments control requires nonzero incoming m AND v')
        altered = trainer.state_dict()
        altered['m'] = np.zeros_like(altered['m'])
        altered['v'] = np.zeros_like(altered['v'])
        trainer.load_state_dict(altered)
        state = trainer.state_dict()
        changed = state_signature(state, shape)
        if (changed['metadata_sha256'] != legitimate['metadata_sha256']
            or any(changed['arrays'][key] != legitimate['arrays'][key] for key in ('parameters', 'flags'))
            or np.any(np.asarray(state['m']) != 0) or np.any(np.asarray(state['v']) != 0)):
            raise ValueError('control changed something besides zeroing moments')
        control = dict(name='zero-moments65', legitimate_state=legitimate,
            altered_state=changed, legitimate_state_directory='legitimate-head64',
            changed_fields=['m', 'v'], effective='PENDING_COMPARATOR')
    runtime = trainer.run_metadata()
    runtime['host_runtime'] = dict(system=platform.system(), release=platform.release(),
                                  machine=platform.machine(), python=sys.version,
                                  macos_version=platform.mac_ver()[0] if sys.platform == 'darwin' else None)
    if (runtime['native_vendor'] != args.expected_vendor or runtime['native_profile'] != shape.profile
            or runtime['native_numeric_mode'] != 1):
        raise ValueError('native runtime witness differs')
    if file_sha(runtime['binding_file']) != runtime['binding_sha256']:
        raise ValueError('loaded binding artifact changed before capture')
    sources = source_inventory()
    exclusive(args.output / 'runtime.json', canonical(runtime))
    exclusive(args.output / 'source.json', canonical(sources))
    previous = retain_initial(args.output / 'initial', state, shape)
    skip_evaluation = args.steps == 1 or args.action == 'zero-moments65'
    initial_eval = None if skip_evaluation else evaluate_heldout(
        trainer, raw, args.output / 'heldout-initial', shape)
    final_step = 65 if args.action == 'zero-moments65' else (64 if args.action == 'head64' else args.steps)
    records = []
    for step in range(expected_start, final_step):
        ids = train_ids(raw, step, shape)
        before = trainer.state_dict()
        result = trainer.train_step(ids)
        if runtime['step_result'] == 'lean':
            # DEVIATION 2514: the lean step leaves the gradient on the device;
            # export it (the last completed step's, before the next call) so
            # retain_step's `flat_gradients` read is the same bytes as 'full'.
            result = dict(result, **trainer.export_gradients())
        after = trainer.state_dict()
        if before['completed_steps'] != step or after['completed_steps'] != step + 1:
            raise ValueError('training cursor mismatch')
        folder = args.output / f'step{step + 1:06d}'
        previous, capture_sha = retain_step(folder, before, after, result, ids, previous,
                                            registry, args.expected_vendor, shape)
        records.append(dict(step=step + 1, capture=folder.name, capture_sha256=capture_sha, loss=result['loss']))
        if (step + 1) % 16 == 0:
            print(json.dumps(dict(completed_steps=step + 1, loss=result['loss'])), flush=True)
    final_eval = None if skip_evaluation else evaluate_heldout(
        trainer, raw, args.output / 'heldout-final', shape)
    checkpoint = save_exclusive_checkpoint(trainer, args.output / 'final.checkpoint.json')
    if source_inventory() != sources or file_sha(runtime['binding_file']) != runtime['binding_sha256']:
        raise ValueError('source changed during capture; cannot admit run')
    qualified_learning = args.action == 'continuous' and args.steps == 128
    ratio = None if initial_eval is None else final_eval['mean_loss'] / initial_eval['mean_loss']
    learning_pass = qualified_learning and ratio <= .9
    summary = dict(schema='mojolearn.byte-lm.real-text-capture.v1', action=args.action,
        completed_steps=trainer.step_, expected_steps=final_step, records=records, control=control,
        corpus_sha256=CORPUS_SHA, schedule=schedule, runtime=runtime, incoming_checkpoint=incoming,
        checkpoint=checkpoint, initial_state=state_signature(state), final_state=state_signature(trainer.state_dict()),
        initial_heldout=initial_eval, final_heldout=final_eval,
        learning=dict(admitted=False, observed_gate_passed=bool(learning_pass),
                      admission='pending successful root guard exit and independent gradient validation',
                      eligible=qualified_learning, ratio=ratio,
                      threshold=.9, claim='small fixed next-byte learning only; no useful-generation or reasoning claim'),
        independent_gradient_gate='NOT_RUN; execute separate byte_lm_gradient_oracle.py against retained step',
        cross_vendor_identity='NOT_ADMITTED; requires separate byte comparison and effective resume controls',
        guard_exit_evidence='REQUIRED_EXTERNALLY; this summary may precede teardown failure')
    exclusive(args.output / 'summary.json', canonical(summary))
    return 1 if qualified_learning and not learning_pass else 0


if __name__ == '__main__':
    raise SystemExit(main())
