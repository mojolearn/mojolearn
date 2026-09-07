#!/usr/bin/env python3
"""Root-only bounded SmallMLP capture and file-only comparison.

Capture requires remote Linux CUDA/HIP and process-selected IDENTICAL mode.
Run under the root's serial GPU/time/memory guard. There are no timers,
performance ratios, builds, rentals, or downloads in this tool. Fixed FP32
48x8 input, 8→16→3 ReLU/mean-CE/AdamW, 16 full-batch steps. The declared
learning gate is a >=10% loss decrease from step 1 to step 16, assessed
separately from raw identity. This is a toy training check, not generalization.

capture --out RUN                       continuous 16 steps
capture --out HEAD --head               8 steps and checkpoint-0008.json
capture --out TAIL --resume CHECKPOINT  transferred step-8 checkpoint to 16
compare --left RUN --right HEAD TAIL    file-only continuous vs resumed chain

Comparisons require matching source/profile/input/configuration witnesses,
all raw output/state arrays, and actual checkpoint SHA linkage for chains.
Different CUDA/HIP binding hashes are retained, never assumed equal. At step
9 a real zeroed-moments branch must diverge from the legitimate corresponding
update, while preserving its forward/gradient bytes and schedule counter.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import zipfile

PROFILE = 'small-mlp.identical.48x8.relu16.ce3.adamw.steps16.v1'
NAMES = ('weight1', 'bias1', 'weight2', 'bias2')
SHAPES = ((16, 8), (16,), (3, 16), (3,))
STATE_KEYS = (*NAMES, 'm', 'v', 'flags', 'step')
STEP_KEYS = ('loss', 'logits', 'input_grad',
             *('grad_' + name for name in NAMES), *STATE_KEYS)
CONFIG = dict(lr=0.03, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
POLICY = dict(first_step=1, final_step=16, maximum_loss_ratio=0.9)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False).encode('ascii')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def save_json(path, value):
    path.write_bytes(canonical(value) + b'\n')


def witness(array):
    # DEVIATION 2463: trainer buffers are mojolearn.Array; np.asarray is a
    # zero-copy view with the same typestr and bytes as the ndarray it replaces.
    import numpy as np
    array = np.asarray(array)
    return dict(shape=list(array.shape), dtype=array.dtype.str,
                raw_sha256=hashlib.sha256(array.tobytes(order='C')).hexdigest())


def fixture():
    import numpy as np
    rows = np.arange(48, dtype=np.int32)
    labels = np.asarray(rows % 3, dtype=np.int32)
    x = np.zeros((48, 8), dtype=np.float32)
    x[rows, labels] = np.float32(1)
    x[:, 3:] = (((rows[:, None] * 3 + np.arange(5)[None, :] * 5) % 9 - 4)
                 .astype(np.float32) / np.float32(32))
    weights = [(((np.arange(math.prod(shape), dtype=np.int32) * 7 + 3) % 17 - 8)
                 .astype(np.float32) / np.float32(64)).reshape(shape) for shape in SHAPES]
    weights[1].fill(np.float32(0.125))
    weights[3].fill(np.float32(0))
    return x, labels, weights


def schedule(x, y):
    return dict(schema='small-mlp.full-batch-schedule.v1',
                dataset_sha256=hashlib.sha256(x.tobytes() + y.tobytes()).hexdigest(),
                order='all 48 rows in stored order on every step', batch_size=48,
                total_steps=16, cursor='optimizer.step is the next zero-based full-batch ordinal')


def state_arrays(state):
    import numpy as np
    return dict(state['weights'], m=state['optimizer']['m'], v=state['optimizer']['v'],
                flags=state['optimizer']['flags'],
                step=np.array([state['optimizer']['step']], dtype=np.int64))


def step_arrays(result, state):
    import numpy as np
    return dict(state_arrays(state), loss=np.array([result['loss']], dtype=np.float32),
                logits=result['logits'], input_grad=result['input_grad'],
                **{'grad_' + name: result['gradients'][name] for name in NAMES})


def save_arrays(out, name, arrays):
    import numpy as np
    if any(not np.isfinite(value).all() for value in arrays.values()):
        raise RuntimeError('Non-finite capture array')
    np.savez(out / name, **arrays)
    return dict(file=name, file_sha256=sha(out / name),
                arrays={key: witness(value) for key, value in arrays.items()})


def sources(root):
    paths = set()
    for directory in ('training', 'gemm', 'core'):
        paths.update((root / directory).rglob('*.mojo'))
    for name in ('checks/numerics.mojo', 'checks/vendor.mojo',
                 'bindings/_mojolearn_training.mojo', 'bindings/_mojolearn_linalg.mojo',
                 'python/mojolearn/_mlp_impl.py', 'python/mojolearn/neural_network.py',
                 'python/mojolearn/_training_impl.py', 'python/mojolearn/_linalg_impl.py',
                 'python/mojolearn/_backend.py', 'python/mojolearn/_arrays.py',
                 'python/mojolearn/_mode.py', 'tools/small_mlp_training_capture.py'):
        paths.add(root / name)
    if any(not path.is_file() for path in paths):
        raise RuntimeError('Source capture requires the matching source checkout')
    return {str(path.relative_to(root)): sha(path) for path in sorted(paths)}


def loaded_libraries():
    return sorted({line.split(maxsplit=5)[5] for line in Path('/proc/self/maps').read_text().splitlines()
                   if len(line.split(maxsplit=5)) == 6 and '.so' in line.split(maxsplit=5)[5]})


def checkpoint(model, out, name):
    model.save_checkpoint(out / name)
    return dict(file=name, sha256=sha(out / name), step=model.step_)


def learning(steps):
    if 1 not in steps or 16 not in steps:
        return dict(status='FRAGMENT', policy=POLICY)
    first = float(steps[1]['loss'][0])
    final = float(steps[16]['loss'][0])
    passed = math.isfinite(first) and math.isfinite(final) and first > 0 and final <= first * .9
    return dict(status='PASS' if passed else 'FAIL', policy=POLICY,
                initial_loss=first, final_loss=final)


def control_differences(legitimate, control):
    forward = ('loss', 'logits', 'input_grad', *('grad_' + name for name in NAMES))
    if any(witness(legitimate[key]) != witness(control[key]) for key in forward):
        raise ValueError('Zeroed-moments control changed forward/gradient bytes')
    if any(witness(legitimate[key]) != witness(control[key]) for key in ('step', 'flags')):
        raise ValueError('Zeroed-moments control changed unrelated carried state')
    changed = [key for key in (*NAMES, 'm', 'v')
               if witness(legitimate[key]) != witness(control[key])]
    if not any(name in changed for name in NAMES):
        raise ValueError('Zeroed-moments control did not change any updated parameter')
    return changed


def capture(args):
    # Set bounded thread policy before loading numerical libraries. The
    # process's numeric mode must already be IDENTICAL; never change it here.
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                 'NUMEXPR_NUM_THREADS', 'MOJOLEARN_CPU_THREADS'):
        os.environ[name] = '1'
    if sys.platform != 'linux':
        raise RuntimeError('SmallMLP capture refuses Apple/non-Linux execution')
    if args.head and args.resume:
        raise ValueError('--head and --resume are mutually exclusive')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = dict(schema='small-mlp.capture.v1', profile=PROFILE, status='INITIALIZING',
                  scope='fixed toy training capture; no timing or universal identity certificate',
                  steps=[], checkpoints=[], resume_input=None, control=None)
    captured = {}
    try:
        import numpy as np
        import mojolearn
        from mojolearn import _backend, _linalg_impl, _training_impl
        if _backend.default_mode() != 'identical' or _backend.numeric_mode() != 'identical':
            raise RuntimeError('SmallMLP capture requires process-selected IDENTICAL')
        training = _training_impl._load('identical')
        _linalg_impl.require_identical()
        linalg = _linalg_impl._load()
        vendors = {_backend.read_vendor(binding) for binding in (training, linalg)}
        if len(vendors) != 1 or next(iter(vendors)) not in ('cuda', 'hip'):
            raise RuntimeError('SmallMLP capture requires matching CUDA or HIP bindings')
        record.update(vendor=next(iter(vendors)), numeric_mode='identical',
                      python=sys.version, numpy=np.__version__, mojolearn=mojolearn.__version__,
                      package_file=mojolearn.__file__, sources=sources(Path(__file__).resolve().parents[1]),
                      bindings={name: dict(file=binding.__file__, sha256=sha(binding.__file__),
                                          vendor=_backend.read_vendor(binding))
                                for name, binding in (('training', training), ('linalg', linalg))},
                      libraries=loaded_libraries(), training_mode=int(training.training_numeric_mode()),
                      linalg_mode=_linalg_impl.numeric_mode())
        x, y, weights = fixture()
        descriptor = schedule(x, y)
        record['inputs'] = save_arrays(out, 'inputs.npz', dict(X=x, targets=y,
                                      **{'initial_' + name: array for name, array in zip(NAMES, weights)}))
        fresh = mojolearn.SmallMLPTrainer(*weights, data_schedule=descriptor, **CONFIG)
        expected = fresh.state_dict()
        if args.resume:
            # Retain exactly the transferred bytes and restore that copy.
            with args.resume.open('rb') as stream:
                raw = stream.read(32769)
            if len(raw) > 32768:
                raise ValueError('Transferred MLP checkpoint exceeds its bound')
            (out / 'resume-input.json').write_bytes(raw)
            model = mojolearn.SmallMLPTrainer.from_checkpoint(out / 'resume-input.json')
            state = model.state_dict()
            if model.step_ != 8 or state['config'] != expected['config'] or state['data_schedule'] != descriptor:
                raise ValueError('Resume requires the matching step-8 checkpoint/configuration/schedule')
            record['resume_input'] = dict(file='resume-input.json', sha256=sha(out / 'resume-input.json'), step=8)
        else:
            model = fresh
        record.update(start_step=model.step_, end_step=8 if args.head else 16,
                      config=expected['config'], data_schedule=descriptor,
                      initial_state=save_arrays(out, 'initial-state.npz', state_arrays(model.state_dict())),
                      status='RUNNING')
        record['checkpoints'].append(checkpoint(model, out, 'checkpoint-%04d.json' % model.step_))
        save_json(out / 'metadata.json', record)
        while model.step_ < record['end_step']:
            control = None
            if model.step_ == 8:
                corrupted = model.state_dict()
                corrupted['optimizer']['m'].fill(0)
                corrupted['optimizer']['v'].fill(0)
                control_model = mojolearn.SmallMLPTrainer(*weights, data_schedule=descriptor, **CONFIG)
                control_model.load_state_dict(corrupted)
                control_result = control_model.train_step(x, y, return_input_grad=True)
                control = step_arrays(control_result, control_model.state_dict())
                record['control'] = dict(kind='zeroed-moments-at-step8', corresponding_step=9,
                                         archive=save_arrays(out, 'control-step-0009.npz', control))
                save_json(out / 'metadata.json', record)
            result = model.train_step(x, y, return_input_grad=True)
            step = model.step_
            arrays = step_arrays(result, model.state_dict())
            captured[step] = arrays
            record['steps'].append(dict(step=step, archive=save_arrays(out, 'step-%04d.npz' % step, arrays)))
            if control is not None:
                record['control']['changed_arrays'] = control_differences(arrays, control)
                record['control']['status'] = 'PASS'
            if step in (8, 16):
                record['checkpoints'].append(checkpoint(model, out, 'checkpoint-%04d.json' % step))
            save_json(out / 'metadata.json', record)
        record.update(status='CAPTURED', learning=learning(captured), libraries=loaded_libraries())
    except BaseException as exc:
        record.update(status='REFUSED', reason=repr(exc))
        raise
    finally:
        save_json(out / 'metadata.json', record)
    print(json.dumps(dict(status=record['status'], learning=record['learning'])), flush=True)


def require(value, message):
    if not value:
        raise ValueError(message)


def load_arrays(root, entry, keys):
    """File-only decoding; never imports MojoLearn or invokes a GPU."""
    import numpy as np
    shapes = dict(zip(NAMES, SHAPES))
    shapes.update({'initial_' + name: shape for name, shape in zip(NAMES, SHAPES)})
    shapes.update({'grad_' + name: shape for name, shape in zip(NAMES, SHAPES)})
    shapes.update(X=(48, 8), targets=(48,), loss=(1,), logits=(48, 3),
                  input_grad=(48, 8), m=(195,), v=(195,), flags=(4,), step=(1,))
    def expected_dtype(key):
        return np.dtype(np.int64 if key == 'step' else np.int32 if key in ('flags', 'targets') else np.float32)
    require(isinstance(entry['file'], str) and Path(entry['file']).name == entry['file'],
            'Archive path must be a local filename')
    path = root / entry['file']
    require(path.stat().st_size <= 65536, 'Oversized capture archive')
    require(sha(path) == entry['file_sha256'], 'Capture archive checksum mismatch')
    with zipfile.ZipFile(path) as container:
        members = container.infolist()
        require(len(members) == len(keys) and {item.filename for item in members}
                == {key + '.npy' for key in keys}, 'Unexpected archive members')
        require(all(item.file_size <= 16384 for item in members)
                and sum(item.file_size for item in members) <= 65536,
                'Oversized decompressed capture archive')
        # Validate shapes BEFORE np.load can allocate from an untrusted NPY
        # header, even when the zip member's actual byte count is tiny.
        for item in members:
            key = item.filename[:-4]
            with container.open(item) as header:
                require(np.lib.format.read_magic(header) == (1, 0), 'Unexpected NPY format')
                shape, fortran, dtype = np.lib.format.read_array_header_1_0(header)
                require(shape == shapes[key] and dtype == expected_dtype(key) and not fortran,
                        'Capture shape/dtype/layout differs from fixed profile: ' + key)
                require(header.tell() + math.prod(shape) * dtype.itemsize == item.file_size,
                        'Capture array payload size mismatch')
    with np.load(path, allow_pickle=False) as archive:
        require(set(archive.files) == set(keys) and len(archive.files) == len(keys), 'Missing/duplicate archive array')
        arrays = {key: archive[key] for key in keys}
    require(set(entry['arrays']) == set(keys), 'Missing raw array witness')
    for key, value in arrays.items():
        require(value.shape == shapes[key] and value.dtype == expected_dtype(key),
                'Capture shape/dtype differs from fixed profile: ' + key)
        require(np.isfinite(value).all() and witness(value) == entry['arrays'][key],
                'Raw array witness/nonfinite mismatch: ' + key)
    return arrays


def load_chain(paths):
    require(1 <= len(paths) <= 2, 'Require one continuous capture or a head/resume pair')
    chain = []
    all_steps = {}
    previous = None
    input_keys = ('X', 'targets', *('initial_' + name for name in NAMES))
    for directory in paths:
        root = Path(directory)
        require((root / 'metadata.json').stat().st_size <= 262144, 'Oversized capture metadata')
        meta = json.loads((root / 'metadata.json').read_text())
        require(meta['schema'] == 'small-mlp.capture.v1' and meta['profile'] == PROFILE
                and meta['status'] == 'CAPTURED', 'Incomplete or incompatible capture')
        require(meta['numeric_mode'] == 'identical' and meta['linalg_mode'] == 'identical'
                and meta['vendor'] in ('cuda', 'hip') and meta['training_mode'] == 1,
                'Missing IDENTICAL CUDA/HIP witnesses')
        require(set(meta['bindings']) == {'training', 'linalg'} and meta['sources'] and meta['libraries'],
                'Missing source/library inventory')
        for binding in meta['bindings'].values():
            require(binding['vendor'] == meta['vendor'] and len(binding['sha256']) == 64,
                    'Missing binding provenance')
        inputs = load_arrays(root, meta['inputs'], input_keys)
        initial = load_arrays(root, meta['initial_state'], STATE_KEYS)
        start, end = meta['start_step'], meta['end_step']
        require((start, end) in ((0, 8), (0, 16), (8, 16)), 'Unsupported schedule fragment')
        require(int(initial['step'][0]) == start, 'Initial cursor mismatch')
        checkpoints = {}
        for item in meta['checkpoints']:
            require(Path(item['file']).name == item['file'], 'Invalid checkpoint path')
            require((root / item['file']).stat().st_size <= 32768, 'Oversized checkpoint')
            require(sha(root / item['file']) == item['sha256'], 'Changed checkpoint bytes')
            require(item['step'] not in checkpoints, 'Duplicate checkpoint step')
            checkpoints[item['step']] = item
        require(start in checkpoints and end in checkpoints, 'Missing boundary checkpoint')
        if previous is None:
            require(start == 0 and meta['resume_input'] is None, 'Chain must begin at the fixed initialization')
            require(all(witness(initial[name]) == witness(inputs['initial_' + name]) for name in NAMES),
                    'Initial parameters differ from retained initialization bytes')
            require(all(not initial[key].tobytes().strip(b'\x00') for key in ('m', 'v', 'flags', 'step')),
                    'Initial moments/flags/cursor must be positive-zero bytes')
        else:
            prior_root, prior_meta, prior_last, prior_checkpoints = previous
            require(start == prior_meta['end_step'] == 8, 'Broken schedule cursor chain')
            resume = meta['resume_input']
            require(resume is not None and resume['step'] == 8
                    and Path(resume['file']).name == resume['file'], 'Missing transferred checkpoint')
            require((root / resume['file']).stat().st_size <= 32768, 'Oversized transferred checkpoint')
            require(sha(root / resume['file']) == resume['sha256']
                    == prior_checkpoints[8]['sha256'] == checkpoints[8]['sha256'],
                    'Transferred checkpoint SHA chain differs')
            require(all(witness(initial[key]) == witness(prior_last[key]) for key in STATE_KEYS),
                    'Restored state differs from transferred head state')
        require(len(meta['steps']) == end - start, 'Missing/duplicate step records')
        local_steps = {}
        for item in meta['steps']:
            step = item['step']
            require(type(step) is int and start < step <= end and step not in local_steps,
                    'Invalid/duplicate step ordinal')
            arrays = load_arrays(root, item['archive'], STEP_KEYS)
            require(int(arrays['step'][0]) == step, 'Raw optimizer step differs from schedule')
            local_steps[step] = arrays
        require(set(local_steps) == set(range(start + 1, end + 1)), 'Missing step')
        if 9 in local_steps:
            control = meta['control']
            require(control is not None and control['kind'] == 'zeroed-moments-at-step8'
                    and control['corresponding_step'] == 9 and control['status'] == 'PASS',
                    'Missing effective moment-reset control')
            changed = control_differences(local_steps[9], load_arrays(root, control['archive'], STEP_KEYS))
            require(changed == control['changed_arrays'], 'Control changed-array witness mismatch')
        all_steps.update(local_steps)
        chain.append(meta)
        previous = (root, meta, local_steps[end], checkpoints)
    require(set(all_steps) == set(range(1, 17)), 'Comparison requires all 16 steps')
    compatible = ('profile', 'sources', 'config', 'data_schedule', 'numeric_mode', 'numpy', 'mojolearn')
    for meta in chain[1:]:
        require(all(meta[key] == chain[0][key] for key in compatible), 'Incompatible resumed provenance')
        require(meta['inputs']['arrays'] == chain[0]['inputs']['arrays'], 'Resumed input bytes differ')
    return chain, all_steps


def compare(args):
    left_meta, left = load_chain(args.left)
    right_meta, right = load_chain(args.right)
    compatible = ('profile', 'sources', 'config', 'data_schedule', 'numeric_mode', 'numpy', 'mojolearn')
    require(all(left_meta[0][key] == right_meta[0][key] for key in compatible), 'Capture provenance differs')
    require(left_meta[0]['inputs']['arrays'] == right_meta[0]['inputs']['arrays'], 'Input bytes differ')
    require(left_meta[0]['initial_state']['arrays'] == right_meta[0]['initial_state']['arrays'],
            'Initial optimizer/parameter bytes differ')
    compared = 0
    for step in range(1, 17):
        for key in STEP_KEYS:
            a, b = left[step][key], right[step][key]
            require(witness(a) == witness(b) and a.tobytes() == b.tobytes(),
                    'Raw identity mismatch at step %d / %s' % (step, key))
            compared += a.size
    result = dict(schema='small-mlp.comparison.v1', profile=PROFILE, identity='PASS',
                  compared_cells=int(compared), learning=learning(left),
                  left=[str(path) for path in args.left], right=[str(path) for path in args.right],
                  vendors={'left': [meta['vendor'] for meta in left_meta],
                           'right': [meta['vendor'] for meta in right_meta]},
                  transferred_checkpoints={name: [dict(
                      sha256=meta['resume_input']['sha256'],
                      head_vendor=chain[index - 1]['vendor'], resume_vendor=meta['vendor'])
                      for index, meta in enumerate(chain) if index > 0]
                      for name, chain in (('left', left_meta), ('right', right_meta))},
                  scope='named fixed training/resume fixtures only; no universal certificate')
    if args.output:
        with args.output.open('xb') as stream:
            stream.write(canonical(result) + b'\n')
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result['learning']['status'] == 'PASS' else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    run = commands.add_parser('capture')
    run.add_argument('--out', type=Path, required=True)
    run.add_argument('--head', action='store_true')
    run.add_argument('--resume', type=Path)
    check = commands.add_parser('compare')
    check.add_argument('--left', nargs='+', type=Path, required=True)
    check.add_argument('--right', nargs='+', type=Path, required=True)
    check.add_argument('--output', type=Path)
    options = parser.parse_args()
    if options.command == 'capture':
        capture(options)
    else:
        raise SystemExit(compare(options))
