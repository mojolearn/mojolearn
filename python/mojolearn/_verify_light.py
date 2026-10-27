"""Tiny bitwise fixtures. No performance clocks or model-quality campaign.

One execution per case, at most sixteen input elements and one update per
optimizer. These fixtures have their own references: shrinking a historical
fixture must never silently reuse its hashes. Python here is test control.
"""
import hashlib
import inspect
import io
import json
import re
from pathlib import Path
import tempfile
import token
import tokenize

FORMAT = 'mojolearn.light-identity.v1'
EXPECTED_CASES = ('tape_elementwise', 'tape_layout', 'tape_gradients',
                  'tape_optimizer', 'tape_program', 'tape_convolution', 'numerical_edges',
                  'tape_losses')
OUTPUT_COUNTS = dict(zip(EXPECTED_CASES, (43, 23, 8, 15, 8, 7, 5, 9)))
MAX_INPUT_ELEMENTS = 16


def case_fingerprint(function):
    # ast.dump changes between supported Python versions (including empty
    # fields). Token names and lexemes compare source consistently on all
    # hosts without making whitespace/comments invalidate numerical evidence.
    tokens = tokenize.generate_tokens(io.StringIO(inspect.getsource(function)).readline)
    parts = [(token.tok_name[t.type], '' if t.type in (token.INDENT, token.DEDENT, token.NEWLINE) else t.string)
             for t in tokens if t.type not in (tokenize.COMMENT, tokenize.NL, token.ENDMARKER)]
    return hashlib.sha256(json.dumps(parts, separators=(',', ':')).encode()).hexdigest()


def tape_elementwise():
    import numpy as np
    from . import ops
    with ops.Session(numeric_mode='identical') as s:
        x = s.tensor(np.array([[-0.75, -0.25], [0.25, 0.75]], np.float32))
        p = s.tensor(np.array([[0.25, 0.5], [1, 2]], np.float32))
        unary = (s.relu, s.sigmoid, s.tanh, s.exp, s.erf, s.gelu, s.softplus,
                 s.silu, s.leaky_relu, s.elu, s.hard_sigmoid, s.hard_swish,
                 s.abs, s.floor, s.ceil, s.round, s.sin, s.cos, s.sign)
        outputs = [s.numpy(fn(x)) for fn in unary]
        outputs += [s.numpy(fn(p)) for fn in (s.log, s.sqrt, s.reciprocal)]
        outputs += [s.numpy(fn(x, p)) for fn in
                    (s.add, s.sub, s.mul, s.div, s.minimum, s.maximum,
                     s.equal, s.not_equal, s.greater, s.greater_equal, s.less, s.less_equal)]
        outputs += [s.numpy(s.pow(p, p)), s.numpy(s.pow_scalar(p, 2)),
                    s.numpy(s.scale(x, 0.5)), s.numpy(s.clip(x, -0.5, 0.5))]
        mask = s.greater(x, s.full((2, 2), 0))
        outputs += [s.numpy(s.where(mask, x, p)), s.numpy(s.logical_not(mask)),
                    s.numpy(s.logical_and(mask, mask)), s.numpy(s.logical_or(mask, mask)),
                    s.numpy(s.detach(s.copy(x)))]
        return outputs


def tape_layout():
    import numpy as np
    from . import ops
    with ops.Session(numeric_mode='identical') as s:
        x = s.tensor(np.arange(6, dtype=np.float32).reshape(2, 3) / 8)
        outputs = [s.numpy(s.transpose(x)), s.numpy(s.permute(x, (1, 0))),
                   s.numpy(s.reshape(x, (3, 2))), s.numpy(s.flatten(x, 0)),
                   s.numpy(s.squeeze(s.unsqueeze(x, 0), 0)),
                   s.numpy(s.concat([x, x], axis=0))]
        outputs += [s.numpy(h) for h in s.split(x, [1, 2], axis=1)]
        outputs += [s.numpy(s.take(x, np.array([2, 0], np.int32), axis=1)),
                    s.numpy(s.expand(s.full((1, 3), 0.25), (2, 3))),
                    s.numpy(s.pad(x, [0, 0, 1, 1], value=-0.25)),
                    s.numpy(s.index(x, [s.tensor(np.array([[0, 1]], np.float32))], axes=[0]))]
        outputs += [s.numpy(fn(x, axis=1)) for fn in
                    (s.sum, s.mean, s.max, s.softmax, s.log_softmax, s.argmax, s.any, s.all)]
        outputs += [s.numpy(s.cumsum(x, axis=1)),
                    s.numpy(s.gather_rows(x, s.permutation(2, seed=7))),
                    s.numpy(s.embedding(x, s.ids(np.array([[1, 0]], np.int32))))]
        return outputs


def tape_gradients():
    import numpy as np
    from . import ops
    with ops.Session(record=True, numeric_mode='identical') as s:
        x = s.tensor(np.array([[0.25, 0.5], [0.75, 1]], np.float32), requires_grad=True)
        square = s.mul(x, x)
        loss = s.sum(square)
        s.backward(loss)
        outputs = [s.numpy(square), s.numpy(loss), s.grad(x)]
        s.backward(loss, accumulate=True)
        outputs.append(s.grad(x))
        s.zero_grad()
        s.backward(loss, create_graph=True)
        dx = s.grad_tensor(x)
        s.backward(s.sum(dx))
        outputs.append(s.grad(x))
        seed = s.full((2, 2), 0.5)
        outputs.append(s.numpy(s.vjp(square, [x], seed)[0]))
        outputs.append(s.numpy(s.jvp(square, [x], [seed])))
        outputs.append(s.numpy(s.hvp(loss, [x], [seed])[0]))
        return outputs


def tape_optimizer():
    import numpy as np
    from . import ops
    outputs = []
    for kind in ('sgd', 'adam', 'adamw'):
        with ops.Session(record=True, numeric_mode='identical') as s:
            x = s.tensor(np.array([[0.25, -0.5]], np.float32), requires_grad=True)
            loss = s.mean(s.mul(x, x))
            s.backward(loss)
            opt = s.optimizer(kind, lr=0.125, momentum=0.5, schedule='step', gamma=0.5)
            s.step(opt, [x])
            outputs.append(s.numpy(x))
            state = s.optimizer_state(opt, x)
            key = 'momentum_buffer' if kind == 'sgd' else 'exp_avg'
            outputs.append(s.numpy(state[key]))
            s.load_optimizer_state(opt, x, state)
            outputs.append(s.numpy(s.optimizer_state(opt, x)[key]))
            s.scheduler_step(opt)
            outputs.append(np.asarray([s.learning_rate(opt)], np.float64))
            with tempfile.TemporaryDirectory(prefix='mjl-light-') as directory:
                path = str(Path(directory) / 'state.mjl')
                s.save_state(path, [x], opt, seed=7, epoch=1, step=1)
                restored = s.load_state(path, optimizer=opt)
                outputs.append(s.numpy(restored['params'][0]))
    return outputs


def tape_program():
    import numpy as np
    from . import ops
    a = np.array([[0.25, 0.5], [0.75, 1]], np.float32)
    with ops.Session(numeric_mode='identical') as s:
        x = s.tensor(a)
        w = s.tensor(np.array([[0.5], [-0.25]], np.float32))
        p = s.capture(x, s.matmul(x, w))
        outputs = [p.run(a), s.run(p, a), p.run_dynamic(a[:1])]
        opt = s.optimizer('sgd', lr=0.125, momentum=0.5)
        p.fit(a, targets=np.zeros((2, 1), np.float32), optimizer=opt,
              parameters=[w], epochs=1, batch_size=2, shuffle=False)
        outputs += [s.numpy(w), p.run(a), np.asarray(p.loss_curve_, np.float32)]
        with tempfile.TemporaryDirectory(prefix='mjl-light-') as directory:
            path = str(Path(directory) / 'program.mjl')
            p.save(path)
            loaded = s.load_program(path)
            outputs += [loaded.run(a), loaded.run(a, optimize=True)]
        return outputs


def tape_convolution():
    import numpy as np
    from . import ops
    with ops.Session(numeric_mode='identical') as s:
        x = s.tensor(np.arange(9, dtype=np.float32).reshape(1, 1, 3, 3) / 8)
        w = s.tensor(np.array([[[[0.25, -0.5], [0.5, 0.25]]]], np.float32))
        outputs = [s.numpy(s.conv2d(x, w)), s.numpy(s.avg_pool2d(x, 2)), s.numpy(s.max_pool2d(x, 2))]
        h = s.tensor(np.array([[0.25, -0.5, 0.75, 1]], np.float32))
        scale, bias = s.full((1, 4), 1), s.full((1, 4), 0)
        outputs += [s.numpy(s.layer_norm(h, scale, bias)), s.numpy(s.rms_norm(h, scale)),
                    s.numpy(s.dropout(h, 0.25, seed=7)), s.numpy(s.rotary(h, 4))]
        return outputs


def numerical_edges():
    import numpy as np
    from . import ops
    with ops.Session(numeric_mode='identical') as s:
        # Exact FP32 words exercise signed zero, FTZ and cancellation. These
        # are fixture choices, never production dispatch dimensions.
        bits = np.array([[0, 0x80000000, 1, 0x80000001],
                         [0x00800000, 0x80800000, 0x3f800001, 0xbf800000]], np.uint32)
        x = s.tensor(bits.view(np.float32))
        outputs = [s.numpy(s.copy(x)), s.numpy(s.add(x, s.full((2, 4), 0))),
                   s.numpy(s.mul(x, s.full((2, 4), 0.5))), s.numpy(s.sum(x, axis=1))]
        a = s.tensor(np.array([[1e8, 1, -1e8, 0.25]], np.float32))
        outputs.append(s.numpy(s.matmul(a, s.full((4, 1), 1))))
        return outputs


def tape_losses():
    import numpy as np
    from . import ops
    with ops.Session(numeric_mode='identical') as s:
        prediction = s.tensor(np.array([[0.25], [0.75]], np.float32))
        target = s.tensor(np.array([[0], [1]], np.float32))
        logits = s.tensor(np.array([[0.5, -0.25], [-0.5, 0.75]], np.float32))
        outputs = [s.numpy(s.mse(prediction, target)), s.numpy(s.mae(prediction, target)),
                   s.numpy(s.huber(prediction, target)), s.numpy(s.bce_with_logits(prediction, target)),
                   s.numpy(s.cross_entropy(logits, target))]
        outputs += [np.asarray([s.accuracy(target, target)], np.float64),
                    np.asarray([s.log_loss(target, prediction)], np.float64),
                    np.asarray([s.roc_auc(target, prediction)], np.float64),
                    np.asarray([s.scalar(s.sum(prediction))], np.float64)]
        return outputs


def digest(value):
    import numpy as np
    a = np.ascontiguousarray(value)
    if a.dtype.kind not in 'biuf' or a.size > 64:
        raise ValueError('light fixture produced an unexpected dtype or size')
    raw = a.tobytes()
    return dict(shape=list(a.shape), dtype=a.dtype.str,
                sha256=hashlib.sha256(raw).hexdigest(), bytes_hex=raw.hex())


def capture(cases=EXPECTED_CASES):
    from . import _backend, _verify, __version__
    cases = tuple(cases)
    if not cases or len(set(cases)) != len(cases) or set(cases) - set(EXPECTED_CASES):
        raise ValueError('unknown, duplicate or empty light case selection')
    binding = _backend.binding('_mojolearn_x_sequence', 'identical')
    report = dict(format=FORMAT, version=__version__,
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  case_sources={name: case_fingerprint(globals()[name])
                                for name in cases},
                  case_source_protocol='python-tokens-v1',
                  digest_protocol='sha256-raw-shape-dtype-v1',
                  vendor=str(binding.x_sequence_vendor()), numeric_mode=int(binding.x_sequence_numeric_mode()),
                  expected_cases=list(cases), cases={}, max_input_elements=MAX_INPUT_ELEMENTS)
    if report['numeric_mode'] != 1:
        raise ValueError('light verification requires identical mode')
    for name in cases:
        try:
            values = globals()[name]()
            if len(values) != OUTPUT_COUNTS[name]:
                raise ValueError(f'{name}: expected {OUTPUT_COUNTS[name]} outputs, got {len(values)}')
            report['cases'][name] = dict(status='captured', outputs=[digest(x) for x in values])
        except Exception as exc:
            report['cases'][name] = dict(status='failed', error=str(exc), outputs=[])
    report['bindings'] = _verify.binding_artifacts()
    report['status'] = 'CAPTURED' if all(r['status'] == 'captured' for r in report['cases'].values()) else 'FAILED'
    return report


def compare(actual, expected):
    errors = []
    names = actual.get('expected_cases', [])
    for label, report in (('actual', actual), ('expected', expected)):
        if (report.get('format') != FORMAT or report.get('numeric_mode') != 1
                or report.get('status') != 'CAPTURED' or not report.get('bindings')
                or not names or len(set(names)) != len(names) or set(names) - set(EXPECTED_CASES)
                or report.get('expected_cases') != names or set(report.get('cases', {})) != set(names)):
            errors.append(label + ': incomplete or invalid capture')
            continue
        for name in names:
            row = report['cases'][name]
            if row.get('status') != 'captured' or len(row.get('outputs', [])) != OUTPUT_COUNTS[name]:
                errors.append(label + ': missing outputs for ' + name)
            for output in row.get('outputs', []):
                shape = output.get('shape')
                if (not isinstance(shape, list) or not shape
                        or any(type(n) is not int or n <= 0 for n in shape)
                        or not re.fullmatch(r'[<|][bifu][1248]', str(output.get('dtype', '')))
                        or not re.fullmatch(r'[0-9a-f]{64}', str(output.get('sha256', '')))):
                    errors.append(label + ': malformed output for ' + name)
                    continue
                try:
                    raw = bytes.fromhex(output['bytes_hex'])
                    elements = 1
                    for dim in shape:
                        elements *= dim
                    valid = (elements <= 64 and len(raw) == elements * int(output['dtype'][-1])
                             and hashlib.sha256(raw).hexdigest() == output['sha256'])
                except (KeyError, TypeError, ValueError):
                    valid = False
                if not valid:
                    errors.append(label + ': absent or inconsistent raw output for ' + name)
        for binding in report.get('bindings', []):
            if not re.fullmatch(r'[0-9a-f]{64}', str(binding.get('sha256', ''))):
                errors.append(label + ': missing binary hash')
        if (set(report.get('case_sources', {})) != set(names)
                or any(not re.fullmatch(r'[0-9a-f]{64}', str(h)) for h in report.get('case_sources', {}).values())
                or report.get('digest_protocol') != 'sha256-raw-shape-dtype-v1'):
            errors.append(label + ': missing fixture contract')
        if report.get('case_source_protocol') != 'python-tokens-v1':
            errors.append(label + ': unsupported fixture fingerprint protocol')
    if actual.get('case_sources') != expected.get('case_sources'):
        errors.append('different numerical fixture source')
    if not errors:
        for name in names:
            if actual['cases'][name]['outputs'] != expected['cases'][name]['outputs']:
                errors.append('bits differ: ' + name)
    return errors
