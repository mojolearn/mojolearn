"""Generate autograd/checks/higher.mojo: higher-order gradient checks of the tape.

Python and NumPy are oracle generators only (float64, run with ``python3 -I``):
they prepare expected values for the committed Mojo suite. The suite records
the forward, the recorded backward (create_graph), the second and third
backward and every gradient in compiled Mojo on the executor under test.

Second derivatives are float64 nested central differences of the float64
forward along the direction v: with inner and outer step h,
``(f(t + h v + h e) - f(t + h v - h e) - f(t - h v + h e) + f(t - h v - h e)) / 4h^2``
is the central difference along v of the central-difference gradient. A
case is accepted only when h = 1e-3 and h = 1e-4 agree to 1e-6 (relative to
max(1, |value|)); each is Richardson-extrapolated from steps h and h/2
(the h^2 term cancels), and the h = 1e-3 value is the oracle. Closed forms are used
where they are trivial (x^3, the softmax Hessian, the tanh network's input
derivatives) and are cross-checked against the differences here.

Usage:
    python3 -I autograd/checks/generate_higher_oracles.py [--fused]

``--fused`` emits the linear, loss and dropout cases (the fused-kind
lowerings); without it those three kinds get refusal cases instead. The
first line of the output names the expected counts that
tools/check_autograd.py --only higher requires.
"""
import argparse
import json
from pathlib import Path

import numpy as np

FNV = 1469598103934665603


def f32(a):
    return np.asarray(a, dtype=np.float64).astype(np.float32).astype(np.float64)


def softmax(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(x):
    s = x - x.max(axis=1, keepdims=True)
    return s - np.log(np.exp(s).sum(axis=1, keepdims=True))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def direction(shape, k):
    n = int(np.prod(shape))
    return f32(np.round(np.cos(np.arange(n) * 1.3 + 0.7 * k), 3).reshape(shape))


# ---------------------------------------------------------------- differences

def _with(inputs, steps):
    out = {k: v.copy() for k, v in inputs.items()}
    for k, (d, s) in steps.items():
        out[k] = out[k] + s * d
    return out


def grad_fd(fn, inputs, names):
    """Central-difference gradient of the scalar fn, two steps cross-checked."""
    result = {}
    for k in names:
        g = np.zeros_like(inputs[k])
        for idx in np.ndindex(inputs[k].shape):
            vals = []
            for rel in (1e-5, 1e-6):
                h = rel * max(1.0, abs(inputs[k][idx]))
                e = np.zeros_like(inputs[k])
                e[idx] = 1
                plus = fn(_with(inputs, {k: (e, h)}))
                minus = fn(_with(inputs, {k: (e, -h)}))
                vals.append((plus - minus) / (2 * h))
            if abs(vals[0] - vals[1]) > 1e-6 * max(1.0, abs(vals[1])):
                raise SystemExit(f'gradient differences disagree for {k}{idx}: {vals}')
            g[idx] = vals[1]
        result[k] = g
    return result


def hvp_fd(fn, inputs, v):
    """Nested central differences of H v; v maps input name -> direction."""
    names = list(v)

    def at(h):
        result = {}
        for q in names:
            out = np.zeros_like(inputs[q])
            for idx in np.ndindex(inputs[q].shape):
                e = np.zeros_like(inputs[q])
                e[idx] = 1

                def f(sv, se):
                    steps = {k: (v[k], sv * h) for k in names}
                    shifted = _with(inputs, steps)
                    shifted[q] = shifted[q] + se * h * e
                    return fn(shifted)
                out[idx] = (f(1, 1) - f(1, -1) - f(-1, 1) + f(-1, -1)) / (4 * h * h)
            result[q] = out
        return result
    def richardson(h):
        # (4 D(h/2) - D(h)) / 3 cancels the h^2 term of the central difference
        full, half = at(h), at(h / 2)
        return {q: (4 * half[q] - full[q]) / 3 for q in names}
    coarse, fine = richardson(1e-3), richardson(1e-4)
    for q in names:
        diff = np.abs(coarse[q] - fine[q])
        if (diff > 1e-6 * np.maximum(1.0, np.abs(coarse[q]))).any():
            raise SystemExit(f'second differences disagree for {q}: max {diff.max():.3g}')
    return coarse


def jacobian_fd(fn, inputs, names):
    """Central-difference Jacobian of an array-valued fn: name -> (out.size, input.size)."""
    out = np.asarray(fn(inputs))
    result = {}
    for k in names:
        jac = np.zeros((out.size, inputs[k].size))
        for j, idx in enumerate(np.ndindex(inputs[k].shape)):
            h = 1e-6 * max(1.0, abs(inputs[k][idx]))
            e = np.zeros_like(inputs[k])
            e[idx] = 1
            jac[:, j] = ((np.asarray(fn(_with(inputs, {k: (e, h)}))) - np.asarray(fn(_with(inputs, {k: (e, -h)}))))
                         / (2 * h)).ravel()
        result[k] = jac
    return result


def agree(a, b, what, tol=1e-6):
    a, b = np.asarray(a), np.asarray(b)
    if (np.abs(a - b) > tol * np.maximum(1.0, np.abs(b))).any():
        raise SystemExit(f'closed form and differences disagree: {what}: {np.abs(a - b).max():.3g}')


# ---------------------------------------------------------------- the fused layer's tape semantics

def act_vjp(act, z, g, mask=None):
    """The linear node's activation VJP written in its output z (both backward passes).

    RELU's indicator is a constant (the tape records it without a graph), so
    a caller differentiating through this map passes the base point's mask."""
    if act == 'none':
        return g
    if act == 'relu':
        return g * ((z > 0) if mask is None else mask)
    if act == 'sigmoid':
        return g * z * (1 - z)
    if act == 'tanh':
        return g * (1 - z * z)
    if act == 'softmax':
        return z * (g - (g * z).sum(axis=1, keepdims=True))
    raise ValueError(act)


def act_fn(act, a):
    return {'none': lambda q: q, 'relu': lambda q: np.maximum(q, 0), 'sigmoid': sigmoid,
            'tanh': np.tanh, 'softmax': softmax}[act](a)


def linear_mse_oracle(inputs, v, act, bias, l2, divisor):
    """First-order gradients and the second backward's gradients of
    mse(linear(x, w, b, act, l2, divisor), y) as the tape defines them.

    With divisor > 0 the linear node's weight gradient is (x^T dZ + l2 w) /
    divisor and its bias gradient colsum(dZ) / divisor, in the recorded
    backward and in the second (first-order) backward alike, so the second
    pass is not a plain Hessian: phi = sum_p <G_p, v_p> is a function of
    the operands read directly (x, w) and of the node output z; its
    gradient flows to x, w directly and through the node's own modified
    rule from dphi/dz. With l2 = 0 and divisor = 0 this is the Hessian
    (checked against the differences in main)."""
    x, w, y = inputs['x'], inputs['w'], inputs['y']
    b = inputs['b'] if bias else 0.0
    n = y.size
    den = divisor if divisor > 0 else 1.0
    z = act_fn(act, x @ w + b)
    mask = z > 0

    def dz_of(zz):
        return act_vjp(act, zz, 2 * (zz - y) / n, mask)

    def grads(xd, wd, zz):
        dz = dz_of(zz)
        g = {'w': (xd.T @ dz + (l2 * wd if l2 > 0 else 0)) / den, 'x': dz @ wd.T}
        if bias:
            g['b'] = dz.sum(axis=0, keepdims=True) / den
        return g

    first = {k: val for k, val in grads(x, w, z).items() if k in v}

    def phi(d):
        g = grads(d['x'], d['w'], d['z'])
        return float(sum(np.sum(g[k] * v[k]) for k in v))
    point = {'x': x, 'w': w, 'z': z}
    part = grad_fd(phi, point, ['x', 'w', 'z'])
    dz2 = act_vjp(act, z, part['z'], mask)
    second = {'w': part['w'] + (x.T @ dz2 + (l2 * w if l2 > 0 else 0)) / den, 'x': part['x'] + dz2 @ w.T}
    if bias:
        second['b'] = dz2.sum(axis=0, keepdims=True) / den
    return first, {k: val for k, val in second.items() if k in v}


# ---------------------------------------------------------------- cases

def huber(r, delta):
    q = np.minimum(np.abs(r), delta)
    return 0.5 * q * q + delta * (np.abs(r) - q)


def bce_logits(p, t):
    return np.maximum(p, 0) - p * t + np.log1p(np.exp(-np.abs(p)))


def build_cases(fused):
    X = f32([[0.3, -0.8, 1.2], [0.7, 0.2, -0.4]])
    POS = f32(X + 2)
    ROW = f32([[0.9, 1.3, 0.6]])
    COL = f32([[1.1], [0.8]])
    W = f32([[0.7, -0.2], [0.1, 0.4], [-0.5, 0.8]])
    T = f32([[0.2, 0.5, 0.3], [0.6, 0.1, 0.3]])
    cases = []

    def hvp(name, inputs, code, fn, kind='hvp', **extra):
        cases.append(dict(name=name, inputs=inputs, code=code, fn=fn, kind=kind, **extra))

    # 1. x*x*x by two multiplies, with a third backward
    hvp('cube_third', {'x': (f32([[0.7, -1.2, 1.5]]), True)},
        'var x2 = tape.multiply(ex, x, x)\nvar out = tape.sum(ex, tape.multiply(ex, x2, x))',
        lambda d: float(np.sum(d['x'] ** 3)), kind='third')
    # 2. unary
    for m, a, fn in [('exp', X, np.exp), ('log', POS, np.log), ('sqrt', POS, np.sqrt),
                     ('tanh', X, np.tanh), ('sigmoid', X, sigmoid), ('relu', X, lambda q: np.maximum(q, 0))]:
        hvp(f'unary_{m}', {'x': (a, True)}, f'var out = tape.sum(ex, tape.{m}(ex, x))',
            lambda d, fn=fn: float(np.sum(fn(d['x']))))
    hvp('unary_scale', {'x': (X, True)}, 'var out = tape.sum(ex, tape.scale(ex, x, Float32(-1.7)))',
        lambda d: float(np.sum(d['x'] * np.float64(np.float32(-1.7)))))
    hvp('relu_times_x', {'x': (X, True)}, 'var out = tape.sum(ex, tape.multiply(ex, tape.relu(ex, x), x))',
        lambda d: float(np.sum(np.maximum(d['x'], 0) * d['x'])))
    hvp('divide_joint', {'x': (X, True), 'y': (POS, True)}, 'var out = tape.sum(ex, tape.divide(ex, x, y))',
        lambda d: float(np.sum(d['x'] / d['y'])))
    hvp('divide_row', {'x': (X, True), 'y': (f32(ROW + 0.5), True)}, 'var out = tape.sum(ex, tape.divide(ex, x, y))',
        lambda d: float(np.sum(d['x'] / d['y'])))
    # 3. broadcast folds and expand
    sq = 'var out = tape.sum(ex, tape.multiply(ex, a, a))'
    hvp('add_row_square', {'x': (X, True), 'b': (ROW, True)}, 'var a = tape.add(ex, x, b)\n' + sq,
        lambda d: float(np.sum((d['x'] + d['b']) ** 2)))
    hvp('subtract_col_square', {'x': (X, True), 'b': (COL, True)}, 'var a = tape.subtract(ex, x, b)\n' + sq,
        lambda d: float(np.sum((d['x'] - d['b']) ** 2)))
    hvp('multiply_col_square', {'x': (X, True), 'b': (COL, True)}, 'var a = tape.multiply(ex, x, b)\n' + sq,
        lambda d: float(np.sum((d['x'] * d['b']) ** 2)))
    hvp('multiply_scalar', {'x': (X, True), 's': (f32([[1.7]]), True)},
        'var out = tape.sum(ex, tape.multiply(ex, x, s))', lambda d: float(np.sum(d['x'] * d['s'])))
    # 4. matmul, shared operand, transpose / reshape, mean and sum over each axis
    hvp('matmul_square', {'x': (X, True), 'w': (W, True)}, 'var a = tape.matmul(ex, x, w)\n' + sq,
        lambda d: float(np.sum((d['x'] @ d['w']) ** 2)))
    hvp('shared_matmul', {'x': (X, True)}, 'var out = tape.sum(ex, tape.matmul(ex, x, tape.transpose(ex, x)))',
        lambda d: float(np.sum(d['x'] @ d['x'].T)))
    hvp('transpose_reshape', {'x': (X, True)},
        'var r = tape.reshape(ex, tape.transpose(ex, x), 2, 3)\nvar out = tape.sum(ex, tape.multiply(ex, r, x))',
        lambda d: float(np.sum(d['x'].T.reshape(2, 3) * d['x'])))
    for axis in (-1, 0, 1):
        label = 'all' if axis == -1 else str(axis)
        hvp(f'mean_axis_{label}_square', {'x': (X, True)}, f'var a = tape.mean(ex, x, {axis})\n' + sq,
            lambda d, axis=axis: float(np.sum(np.mean(d['x'], axis=None if axis == -1 else axis) ** 2)))
    hvp('sum_axis_1_exp', {'x': (X, True)}, 'var out = tape.sum(ex, tape.exp(ex, tape.sum(ex, x, 1)))',
        lambda d: float(np.sum(np.exp(d['x'].sum(axis=1)))))
    # 5. softmax rows (analytic Hessian, rows of t sum to 1)
    hvp('log_softmax_dot', {'x': (X, True), 't': (T, False)},
        'var out = tape.sum(ex, tape.multiply(ex, t, tape.log_softmax(ex, x)))',
        lambda d: float(np.sum(d['t'] * log_softmax(d['x']))), analytic='log_softmax')
    hvp('softmax_dot', {'x': (X, True), 't': (T, False)},
        'var out = tape.sum(ex, tape.multiply(ex, t, tape.softmax(ex, x)))',
        lambda d: float(np.sum(d['t'] * softmax(d['x']))))
    hvp('cross_entropy_dense', {'x': (X, True), 't': (T, False)}, 'var out = tape.cross_entropy(ex, x, t)',
        lambda d: float(-np.sum(log_softmax(d['x']) * d['t']) / d['x'].shape[0]))
    # 9. the adjoint identity on a two-layer tanh network
    hvp('adjoint_tanh2', {'x': (f32([[0.4, -0.6], [0.9, 0.3], [-0.2, 0.5]]), True),
                          'w1': (f32([[0.7, -0.4, 0.2], [0.3, 0.8, -0.5]]), True),
                          'w2': (f32([[0.6, -0.3], [-0.2, 0.9], [0.5, 0.4]]), False)},
        'var out = tape.tanh(ex, tape.matmul(ex, tape.tanh(ex, tape.matmul(ex, x, w1)), w2))',
        lambda d: np.tanh(np.tanh(d['x'] @ d['w1']) @ d['w2']), kind='adjoint')
    # 10. physics: per-row d2u/dx2 of a one-input tanh network, then a residual loss
    hvp('physics_tanh', {'x': (f32([[-0.9], [-0.2], [0.4], [1.1]]), True),
                         'w1': (f32([[0.8, -1.3, 0.5]]), True), 'b1': (f32([[0.1, 0.3, -0.2]]), True),
                         'w2': (f32([[0.9], [0.4], [-0.7]]), True)},
        'var a = tape.add(ex, tape.matmul(ex, x, w1), b1)\nvar u = tape.matmul(ex, tape.tanh(ex, a), w2)',
        None, kind='physics')
    if fused:
        # 6. the linear node
        LX = f32([[0.5, -0.3], [0.2, 0.9], [-0.7, 0.4]])
        LW = f32([[0.6, -0.4, 0.3], [0.2, 0.7, -0.5]])
        LB = f32([[0.1, -0.2, 0.3]])
        LY = f32([[0.3, -0.1, 0.5], [0.2, 0.4, -0.3], [0.6, 0.1, 0.2]])
        act_code = {'none': '0', 'relu': 'RELU', 'sigmoid': 'SIGMOID', 'tanh': 'TANH', 'softmax': 'SOFTMAX'}
        variants = [(a, True, 0.0, 0.0, True) for a in ('none', 'relu', 'sigmoid', 'tanh', 'softmax')]
        variants += [('tanh', False, 0.0, 0.0, True), ('relu', False, 0.0, 0.0, True),
                     ('tanh', True, 0.0, 4.0, True), ('tanh', True, 0.1, 4.0, True),
                     ('none', True, 0.1, 4.0, False)]
        for act, bias, l2, div, x_grad in variants:
            name = f'linear_{act}' + ('' if bias else '_nobias') + (f'_l2_{l2:g}' if l2 else '') \
                + (f'_div_{div:g}' if div else '') + ('' if x_grad else '_data')
            name = name.replace('.', 'p')
            inputs = {'x': (LX, x_grad), 'w': (LW, True)}
            if bias:
                inputs['b'] = (LB, True)
            inputs['y'] = (LY, False)
            bias_code = 'Optional[Tensor](b)' if bias else 'Optional[Tensor]()'
            code = (f'var z = tape.linear(ex, x, w, {bias_code}, {act_code[act]}, Float32({l2:g}), Float32({div:g}))\n'
                    'var out = tape.mse_loss(ex, z, y)')

            def fn(d, act=act, bias=bias):
                z = act_fn(act, d['x'] @ d['w'] + (d['b'] if bias else 0.0))
                return float(np.mean((z - d['y']) ** 2))
            hvp(name, inputs, code, fn, linear=(act, bias, l2, div))
        LC = f32([[2.0], [0.0], [1.0]])

        def sce(d):
            lsm = log_softmax(d['x'] @ d['w'] + d['b'])
            return float(-np.mean(lsm[np.arange(3), d['c'].ravel().astype(int)]))
        hvp('linear_softmax_cross_entropy', {'x': (LX, False), 'w': (LW, True), 'b': (LB, True), 'c': (LC, False)},
            'var z = tape.linear(ex, x, w, Optional[Tensor](b))\nvar out = tape.softmax_cross_entropy(ex, z, c)', sce)
        # 7. losses
        P = f32([[0.3, -0.8, 1.2], [0.7, 0.2, -1.5]])
        TT = f32([[0.0, 1.1, -0.6], [0.4, 0.9, 0.3]])  # r = P - TT: |r| in (0.3, 1.9), away from 0 and 1
        kinds = [('mse', 'LS_MSE', lambda p, t: (p - t) ** 2), ('mae', 'LS_MAE', lambda p, t: np.abs(p - t)),
                 ('huber', 'LS_HUBER', lambda p, t: huber(p - t, 1.0)),
                 ('bce_logits', 'LS_BCE_LOGITS', bce_logits)]
        for label, kc, fn in kinds:
            t_values = f32([[0.0, 1.0, 0.3], [1.0, 0.6, 0.0]]) if label == 'bce_logits' else TT
            inputs = {'p': (P, True), 't': (t_values, True)}
            hvp(f'loss_terms_{label}_mean', inputs,
                f'var out = tape.mean(ex, tape.loss_terms(ex, p, t, {kc}, Float32(1)))',
                lambda d, fn=fn: float(np.mean(fn(d['p'], d['t']))))
            hvp(f'loss_reduced_{label}_mean', inputs,
                f'var out = tape.loss_reduced(ex, p, t, {kc}, Float32(1))',
                lambda d, fn=fn: float(np.mean(fn(d['p'], d['t']))))
        hvp('loss_reduced_mse_sum', {'p': (P, True), 't': (TT, True)},
            'var out = tape.loss_reduced(ex, p, t, LS_MSE, Float32(1), False)',
            lambda d: float(np.sum((d['p'] - d['t']) ** 2)))
        CODES = f32([[2.0], [0.0]])
        hvp('cross_entropy_rows_mean', {'x': (X, True), 'c': (CODES, False)},
            'var out = tape.mean(ex, tape.cross_entropy_rows(ex, x, c))',
            lambda d: float(-np.mean(log_softmax(d['x'])[[0, 1], [2, 0]])))
        hvp('softmax_cross_entropy', {'x': (X, True), 'c': (CODES, False)},
            'var out = tape.softmax_cross_entropy(ex, x, c)',
            lambda d: float(-np.mean(log_softmax(d['x'])[[0, 1], [2, 0]])))
        # 8. dropout (the oracle reads the mask back from the forward value)
        hvp('dropout_tanh_square', {'x': (X, True)}, None, None, kind='dropout')
    return cases


# ---------------------------------------------------------------- Mojo text

def lit(values):
    return '[' + ', '.join(f'Float32({float(v):.12g})' for v in np.asarray(values).flat) + ']'


def tensor_line(name, value, grad):
    r, c = value.shape
    return f'    var {name} = tape.tensor(ex, {lit(value)}, {r}, {c}{", True" if grad else ""})'


def body(code):
    return ['    ' + line for line in code.splitlines()]


def emit_hvp(case, lines, records):
    name, inputs = case['name'], case['inputs']
    vals = {k: v for k, (v, _) in inputs.items()}
    params = [k for k, (_, g) in inputs.items() if g]
    v = {k: direction(vals[k].shape, i + 1) for i, k in enumerate(params)}
    if 'linear' in case:
        act, bias, l2, div = case['linear']
        grads, second = linear_mse_oracle(vals, v, act, bias, l2, div)
        if l2 == 0 and div == 0:
            agree(np.concatenate([grads[k].ravel() for k in params]),
                  np.concatenate([g.ravel() for g in grad_fd(case['fn'], vals, params).values()]), name + ' grad', 1e-5)
            plain = hvp_fd(case['fn'], vals, v)
            agree(np.concatenate([second[k].ravel() for k in params]),
                  np.concatenate([plain[k].ravel() for k in params]), name + ' hvp', 1e-5)
    else:
        grads = grad_fd(case['fn'], vals, params)
        second = hvp_fd(case['fn'], vals, v)
    if case.get('analytic') == 'log_softmax':
        # d/dz of sum(t log_softmax(z)) per row: t - p sum(t); its Hessian
        # with sum(t) = 1 is p p^T - diag(p): H v = p (p . v) - p v
        p = softmax(vals['x'])
        vx = v['x']
        closed = p * (p * vx).sum(axis=1, keepdims=True) - p * vx
        agree(closed, second['x'], name + ' analytic')
        second = {'x': closed}
    lines += ['', f'def _c_{name}[E: Exec](mut ex: E, mut tape: Tape) raises -> Bool:', '    tape.reset()']
    lines += [tensor_line(k, val, g) for k, (val, g) in inputs.items()]
    lines += body(case['code'])
    lines.append('    tape.backward(ex, out)')
    lines += [f'    var fo_{k} = tape.gradients(ex, {k})' for k in params]
    lines.append('    backward_scalar_recorded(tape, ex, out)')
    lines += ['    var fdig = UInt64(1469598103934665603)', '    var hdig = UInt64(1469598103934665603)',
              '    var same = True']
    for k in params:
        lines += [f'    var gr_{k} = tape.grad_of({k})',
                  f'    var rv_{k} = tape.values(ex, gr_{k})',
                  f'    _check("{name}/grad/{k}", rv_{k}, {lit(grads[k])}, fdig)',
                  f'    same = same and _same_bits(rv_{k}, fo_{k})']
    for k in params:
        r, c = vals[k].shape
        lines.append(f'    var dv_{k} = tape.tensor(ex, {lit(v[k])}, {r}, {c})')
    lines.append(f'    var dot = _dot_sum(ex, tape, [{", ".join("gr_" + k for k in params)}], '
                 f'[{", ".join("dv_" + k for k in params)}])')
    lines.append(f'    var hv = _second(ex, tape, dot, [{", ".join(params)}])')
    for i, k in enumerate(params):
        lines.append(f'    _check("{name}/hvp/{k}", hv[{i}], {lit(second[k])}, hdig)')
    lines += [f'    print("HIGHER_DIGEST {name}", hdig)', '    return same']
    records.append({'case': name, 'kind': 'hvp', 'params': params,
                    'second_order_elements': int(sum(vals[k].size for k in params))})


def emit_third(case, lines, records):
    name = case['name']
    x = case['inputs']['x'][0]
    g1, g2, g3 = 3 * x * x, 6 * x, np.full_like(x, 6.0)
    agree(g1, grad_fd(case['fn'], {'x': x}, ['x'])['x'], name + ' grad')
    ones = np.ones_like(x)
    agree(g2, hvp_fd(case['fn'], {'x': x}, {'x': ones})['x'], name + ' second')
    lines += ['', f'def _c_{name}[E: Exec](mut ex: E, mut tape: Tape) raises -> Bool:', '    tape.reset()',
              tensor_line('x', x, True)]
    lines += body(case['code'])
    lines += ['    tape.backward(ex, out)', '    var fo_x = tape.gradients(ex, x)',
              '    backward_scalar_recorded(tape, ex, out)',
              '    var fdig = UInt64(1469598103934665603)', '    var hdig = UInt64(1469598103934665603)',
              '    var g1 = tape.grad_of(x)', '    var rv_x = tape.values(ex, g1)',
              f'    _check("{name}/grad/x", rv_x, {lit(g1)}, fdig)',
              '    var same = _same_bits(rv_x, fo_x)',
              '    backward_scalar_recorded(tape, ex, tape.sum(ex, g1))',
              '    var g2 = tape.grad_of(x)',
              f'    _check("{name}/second/x", tape.values(ex, g2), {lit(g2)}, hdig)',
              '    tape.backward(ex, tape.sum(ex, g2))',
              f'    _check("{name}/third/x", tape.gradients(ex, x), {lit(g3)}, hdig)',
              f'    print("HIGHER_DIGEST {name}", hdig)', '    return same']
    records.append({'case': name, 'kind': 'third', 'second_order_elements': int(2 * x.size)})


def emit_adjoint(case, lines, records):
    name, inputs = case['name'], case['inputs']
    vals = {k: v for k, (v, _) in inputs.items()}
    params = ['x', 'w1']
    out = np.asarray(case['fn'](vals))
    vseed = direction(out.shape, 7)
    wseed = direction(out.shape, 9)
    u = {k: direction(vals[k].shape, 11 + i) for i, k in enumerate(params)}
    jac = jacobian_fd(case['fn'], vals, params)
    jtv = {k: (jac[k].T @ vseed.ravel()).reshape(vals[k].shape) for k in params}
    ju = sum(jac[k] @ u[k].ravel() for k in params).reshape(out.shape)
    r, c = out.shape
    lines += ['', f'def _c_{name}[E: Exec](mut ex: E, mut tape: Tape) raises -> Bool:', '    tape.reset()']
    lines += [tensor_line(k, val, g) for k, (val, g) in inputs.items()]
    lines += body(case['code'])
    lines += [f'    var vseed = tape.tensor(ex, {lit(vseed)}, {r}, {c})',
              '    tape.backward_from(ex, out, vseed)',
              '    var fo_x = tape.gradients(ex, x)', '    var fo_w1 = tape.gradients(ex, w1)',
              '    backward_recorded(tape, ex, out, vseed)',
              '    var fdig = UInt64(1469598103934665603)', '    var hdig = UInt64(1469598103934665603)',
              '    var jtv_x = tape.values(ex, tape.grad_of(x))', '    var jtv_w1 = tape.values(ex, tape.grad_of(w1))',
              f'    _check("{name}/vjp/x", jtv_x, {lit(jtv["x"])}, fdig)',
              f'    _check("{name}/vjp/w1", jtv_w1, {lit(jtv["w1"])}, fdig)',
              '    var same = _same_bits(jtv_x, fo_x) and _same_bits(jtv_w1, fo_w1)',
              '    # jvp by the double-VJP identity: the gradient of <J^T w, u> in w is J u',
              f'    var wseed = tape.tensor(ex, {lit(wseed)}, {r}, {c}, True)',
              '    backward_recorded(tape, ex, out, wseed)']
    for k in params:
        rr, cc = vals[k].shape
        lines.append(f'    var dv_{k} = tape.tensor(ex, {lit(u[k])}, {rr}, {cc})')
    lines += ['    var dot = _dot_sum(ex, tape, [tape.grad_of(x), tape.grad_of(w1)], [dv_x, dv_w1])',
              '    tape.backward(ex, dot)',
              '    var ju = tape.gradients(ex, wseed)',
              f'    _check("{name}/jvp", ju, {lit(ju)}, hdig)',
              '    # <v, J u> = <J^T v, u>, both sides from the tape words',
              f'    var lhs = _inner(ju, {lit(vseed)})',
              f'    var rhs = _inner(jtv_x, {lit(u["x"])}) + _inner(jtv_w1, {lit(u["w1"])})',
              '    if abs(lhs - rhs) > 1.0e-4 * (1.0 + abs(lhs)):',
              f'        raise Error("{name}: <v, J u> != <J^T v, u>")',
              f'    print("PASS {name}/identity lhs", lhs, "rhs", rhs)',
              f'    print("HIGHER_DIGEST {name}", hdig)', '    return same']
    records.append({'case': name, 'kind': 'adjoint', 'second_order_elements': int(out.size)})


def emit_physics(case, lines, records):
    name, inputs = case['name'], case['inputs']
    vals = {k: v for k, (v, _) in inputs.items()}

    def parts(d):
        t = np.tanh(d['x'] @ d['w1'] + d['b1'])
        w1, w2 = d['w1'].ravel(), d['w2'].ravel()
        du = ((1 - t * t) * w1 * w2).sum(axis=1, keepdims=True)
        d2u = (-2 * t * (1 - t * t) * w1 * w1 * w2).sum(axis=1, keepdims=True)
        return t, du, d2u

    def u_sum(d):
        return float(np.sum(np.tanh(d['x'] @ d['w1'] + d['b1']) @ d['w2']))

    def du_sum(d):
        return float(np.sum(parts(d)[1]))

    def residual(d):
        return float(np.mean((parts(d)[2] + d['x']) ** 2))
    _, du, d2u = parts(vals)
    agree(du, grad_fd(u_sum, vals, ['x'])['x'], name + ' du/dx')
    agree(d2u, grad_fd(du_sum, vals, ['x'])['x'], name + ' d2u/dx2', 1e-5)
    params = ['x', 'w1', 'b1', 'w2']
    res_grad = grad_fd(residual, vals, params)
    lines += ['', f'def _c_{name}[E: Exec](mut ex: E, mut tape: Tape) raises -> Bool:', '    tape.reset()']
    lines += [tensor_line(k, val, g) for k, (val, g) in inputs.items()]
    lines += body(case['code'])
    lines += ['    var su = tape.sum(ex, u)', '    tape.backward(ex, su)', '    var fo_x = tape.gradients(ex, x)',
              '    backward_scalar_recorded(tape, ex, su)',
              '    var fdig = UInt64(1469598103934665603)', '    var hdig = UInt64(1469598103934665603)',
              '    var du = tape.grad_of(x)', '    var rv_x = tape.values(ex, du)',
              f'    _check("{name}/du_dx", rv_x, {lit(du)}, fdig)',
              '    var same = _same_bits(rv_x, fo_x)',
              '    # rows are independent: the gradient of sum(du/dx) in x is d2u/dx2 per row',
              '    backward_scalar_recorded(tape, ex, tape.sum(ex, du))',
              '    var d2u = tape.grad_of(x)',
              f'    _check("{name}/d2u_dx2", tape.values(ex, d2u), {lit(d2u)}, hdig)',
              '    # a residual d2u/dx2 + x, trained with a third backward',
              '    var res = tape.add(ex, d2u, x)',
              '    tape.backward(ex, tape.mean(ex, tape.multiply(ex, res, res)))']
    for k in params:
        lines.append(f'    _check("{name}/residual/{k}", tape.gradients(ex, {k}), {lit(res_grad[k])}, hdig)')
    lines += [f'    print("HIGHER_DIGEST {name}", hdig)', '    return same']
    records.append({'case': name, 'kind': 'physics', 'second_order_elements': int(vals['x'].size + 10)})


def emit_dropout(case, lines, records):
    name = case['name']
    x = case['inputs']['x'][0]
    v = direction(x.shape, 1)
    r, c = x.shape
    lines += ['', f'def _c_{name}[E: Exec](mut ex: E, mut tape: Tape) raises -> Bool:', '    tape.reset()',
              tensor_line('x', x, True),
              '    var y = tape.dropout(ex, x, Float32(0.3), UInt64(7), 3)',
              '    var th = tape.tanh(ex, y)',
              '    var out = tape.sum(ex, tape.multiply(ex, th, th))',
              '    tape.backward(ex, out)', '    var fo_x = tape.gradients(ex, x)',
              '    backward_scalar_recorded(tape, ex, out)',
              '    var fdig = UInt64(1469598103934665603)', '    var hdig = UInt64(1469598103934665603)',
              '    var gr_x = tape.grad_of(x)', '    var rv_x = tape.values(ex, gr_x)',
              '    # the mask is the forward value: y = m s x, m s = y / x (x has no zero)',
              f'    var xs: List[Float32] = {lit(x)}', f'    var vs: List[Float32] = {lit(v)}',
              '    var ys = tape.values(ex, y)',
              '    var want_g = List[Float32]()', '    var want_h = List[Float32]()', '    var kept = 0',
              '    for i in range(len(xs)):',
              '        var ms = Float64(ys[i]) / Float64(xs[i])',
              '        var t = tanh(Float64(ys[i]))',
              '        if ys[i] != 0:', '            kept += 1',
              '        want_g.append(Float32(ms * 2.0 * t * (1.0 - t * t)))',
              '        want_h.append(Float32(ms * ms * 2.0 * (1.0 - t * t) * (1.0 - 3.0 * t * t) * Float64(vs[i])))',
              f'    _check("{name}/grad/x", rv_x, want_g, fdig)',
              '    var same = _same_bits(rv_x, fo_x)',
              f'    var dv_x = tape.tensor(ex, vs, {r}, {c})',
              '    var hv = _second(ex, tape, _dot_sum(ex, tape, [gr_x], [dv_x]), [x])',
              f'    _check("{name}/hvp/x", hv[0], want_h, hdig)',
              f'    print("PASS {name}/mask kept", kept, "of", len(xs))',
              f'    print("HIGHER_DIGEST {name}", hdig)', '    return same']
    records.append({'case': name, 'kind': 'dropout', 'second_order_elements': int(x.size)})


PRELUDE = '''
comptime _FNV_OFFSET = UInt64(1469598103934665603)
comptime _FNV_PRIME = UInt64(1099511628211)


def _check(name: String, actual: List[Float32], expected: List[Float32], mut digest: UInt64) raises:
    """check_values' tolerance and FNV digest; also folds the words into the case digest."""
    if len(actual) != len(expected):
        raise Error(name + ": wrong result length")
    var own = _FNV_OFFSET
    for i in range(len(actual)):
        var tolerance = Float32(0.0003) + Float32(0.0003) * abs(expected[i])
        if not (abs(actual[i] - expected[i]) <= tolerance):
            print("FAIL", name, "index", i, "actual", actual[i], "expected", expected[i])
            raise Error(name + ": higher-order oracle check failed")
        var word = UInt64(bitcast[DType.uint32](actual[i]))
        own = (own ^ word) * _FNV_PRIME
        digest = (digest ^ word) * _FNV_PRIME
    print("PASS", name, "elements", len(actual), "digest", own)


def _same_bits(a: List[Float32], b: List[Float32]) -> Bool:
    if len(a) != len(b):
        return False
    for i in range(len(a)):
        if bitcast[DType.uint32](a[i]) != bitcast[DType.uint32](b[i]):
            return False
    return True


def _inner(a: List[Float32], b: List[Float32]) -> Float64:
    var total = Float64(0)
    for i in range(len(a)):
        total += Float64(a[i]) * Float64(b[i])
    return total


def _dot_sum[E: Exec](mut ex: E, mut tape: Tape, gs: List[Tensor], vs: List[Tensor]) raises -> Tensor:
    """sum_i sum(gs[i] * vs[i]) as tape operations, added in list order."""
    var total = tape.sum(ex, tape.multiply(ex, gs[0], vs[0]))
    for i in range(1, len(gs)):
        total = tape.add(ex, total, tape.sum(ex, tape.multiply(ex, gs[i], vs[i])))
    return total


def _second[E: Exec](mut ex: E, mut tape: Tape, dot: Tensor, wrt: List[Tensor]) raises -> List[List[Float32]]:
    """The gradients of `dot` (a first-order backward); zeros when the
    recorded gradients carry no graph (zero curvature: relu, scale, mae)."""
    var result = List[List[Float32]]()
    if not tape.requires_grad(dot):
        for i in range(len(wrt)):
            result.append(List[Float32](length=tape.rows(wrt[i]) * tape.cols(wrt[i]), fill=0))
        return result^
    tape.backward(ex, dot)
    for i in range(len(wrt)):
        result.append(tape.gradients(ex, wrt[i]))
    return result^


def _refused(err: String, want: String, what: String) raises -> Int:
    if want not in err:
        raise Error("refusal " + what + " gave the wrong error: " + err)
    print("PASS refusal", what)
    return 1
'''


def emit_refusals(fused, lines):
    """Refusal cases; returns their count. Each must raise with the given text."""
    X6 = lit(f32([[0.3, -0.8, 1.2], [0.7, 0.2, -0.4]]))
    R3 = lit(f32([[0.9, 1.3, 0.6]]))
    W6 = lit(f32([[0.7, -0.2], [0.1, 0.4], [-0.5, 0.8]]))
    blocks = []

    def refusal(what, setup, call, want):
        blocks.append((what, setup, call, want))
    graph = 'tape.reset()\nvar x = tape.tensor(ex, ' + X6 + ', 2, 3, True)\n'
    refusal('custom_tag', graph + 'var cn = tape.custom(ex, [x], tape.value(x), 2, 3, 123, Args())\n'
            'var root = tape.sum(ex, cn)',
            'backward_scalar_recorded(tape, ex, root)', 'create_graph cannot differentiate through')
    refusal('no_differentiable_ancestors', 'tape.reset()\nvar root = tape.scalar(ex, Float32(3))',
            'backward_scalar_recorded(tape, ex, root)', 'no differentiable ancestors')
    refusal('grad_of_before_recorded', graph + 'tape.backward(ex, tape.sum(ex, tape.multiply(ex, x, x)))',
            '_ = tape.grad_of(x)', 'no recorded gradient')
    refusal('grad_of_unreached', graph + 'var unreached = tape.exp(ex, x)\n'
            'backward_scalar_recorded(tape, ex, tape.sum(ex, tape.multiply(ex, x, x)))',
            '_ = tape.grad_of(unreached)', '')
    refusal('inferred_root', graph + 'tape.no_grad()\nvar root = tape.sum(ex, tape.multiply(ex, x, x))\ntape.record()',
            'backward_scalar_recorded(tape, ex, root)', 'inference mode')
    refusal('grad_of_after_preactivation', 'tape.reset()\nvar x = tape.tensor(ex, ' + X6 + ', 2, 3)\n'
            'var w = tape.tensor(ex, ' + W6 + ', 3, 2, True)\nvar z = tape.linear(ex, x, w)\n'
            'var seed = tape.tensor(ex, ' + lit(np.ones(4)) + ', 2, 2)\n'
            'tape.backward_from_preactivation(ex, z, seed)',
            '_ = tape.grad_of(w)', 'no recorded gradient')
    if fused:
        refusal('preactivation_after_recorded', 'tape.reset()\nvar x = tape.tensor(ex, ' + X6 + ', 2, 3)\n'
                'var w = tape.tensor(ex, ' + W6 + ', 3, 2, True)\nvar z = tape.linear(ex, x, w)\n'
                'backward_scalar_recorded(tape, ex, tape.sum(ex, tape.multiply(ex, z, z)))',
                '_ = tape.preactivation_gradient(z)', '')
    else:
        # the fused kinds before their lowerings land (lane H2)
        refusal('linear_first_order_only', 'tape.reset()\nvar x = tape.tensor(ex, ' + X6 + ', 2, 3)\n'
                'var w = tape.tensor(ex, ' + W6 + ', 3, 2, True)\nvar root = tape.sum(ex, tape.linear(ex, x, w))',
                'backward_scalar_recorded(tape, ex, root)', 'create_graph cannot differentiate through')
        refusal('softmax_cross_entropy_first_order_only', graph +
                'var c = tape.tensor(ex, [Float32(2), Float32(0)], 2, 1)\n'
                'var root = tape.softmax_cross_entropy(ex, x, c)',
                'backward_scalar_recorded(tape, ex, root)', 'create_graph cannot differentiate through')
        refusal('dropout_first_order_only', graph +
                'var root = tape.sum(ex, tape.dropout(ex, x, Float32(0.3), UInt64(7), 3))',
                'backward_scalar_recorded(tape, ex, root)', 'create_graph cannot differentiate through')
    for i, (what, setup, call, want) in enumerate(blocks):
        lines += ['', f'def _r_{what}[E: Exec](mut ex: E, mut tape: Tape) raises -> Int:']
        lines += body(setup)
        lines += ['    var refused = 0', '    try:', '        ' + call, '    except e:',
                  f'        refused = _refused(String(e), "{want}", "{what}")']
        lines += ['    if refused == 0:', f'        raise Error("{what}: not refused")', '    return refused']
    lines += ['', 'def _refusals[E: Exec](mut ex: E, mut tape: Tape) raises -> Int:', '    var count = 0']
    lines += [f'    count += _r_{what}(ex, tape)' for what, *_ in blocks]
    lines += ['    return count']
    return len(blocks)


LIFECYCLE = '''

def _lifecycle[E: Exec](mut ex: E, mut tape: Tape) raises:
    """30 x (reset, record, recorded backward, second backward) with no
    storage growth after the second iteration, the same words every time,
    and the same words again after release()."""
    var slots = 0
    var capacity = 0
    var first = List[Float32]()
    for iteration in range(31):
        if iteration == 30:
            tape.release(ex)
        tape.reset()
        var w = tape.tensor(ex, [Float32(0.7), Float32(-0.2), Float32(0.1), Float32(0.4), Float32(-0.5), Float32(0.8)], 2, 3, True)
        var x = tape.tensor(ex, [Float32(0.3), Float32(-0.8), Float32(1.2), Float32(0.7), Float32(0.2), Float32(-0.4)], 3, 2)
        var h = tape.tanh(ex, tape.matmul(ex, x, w))
        var out = tape.mean(ex, tape.multiply(ex, h, h))
        backward_scalar_recorded(tape, ex, out)
        var v = tape.tensor(ex, [Float32(0.5), Float32(-0.3), Float32(0.9), Float32(0.1), Float32(-0.7), Float32(0.4)], 2, 3)
        var dot = tape.sum(ex, tape.multiply(ex, tape.grad_of(w), v))
        tape.backward(ex, dot)
        var hv = tape.gradients(ex, w)
        if iteration == 0:
            first = hv.copy()
        elif not _same_bits(hv, first):
            raise Error("repeated higher-order graph changed its words at iteration " + String(iteration))
        if iteration == 30:
            break
        var total = 0
        for n in tape._capacities:
            total += n
        if iteration == 1:
            slots = len(tape._buffers)
            capacity = total
        elif iteration > 1 and (slots != len(tape._buffers) or capacity != total):
            raise Error("repeated higher-order graph grows tape storage")
    print("PASS higher lifecycle repeated_graphs=30 release=1")
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--fused', action='store_true',
                        help='emit the linear, loss and dropout cases (else their refusal cases)')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'higher.mojo')
    args = parser.parse_args()
    cases = build_cases(args.fused)
    body_lines, records = [], []
    emit = {'hvp': emit_hvp, 'third': emit_third, 'adjoint': emit_adjoint, 'physics': emit_physics,
            'dropout': emit_dropout}
    for case in cases:
        emit[case['kind']](case, body_lines, records)
    refusal_lines = []
    refusals = emit_refusals(args.fused, refusal_lines)
    header = [f'# HIGHER_EXPECT cases={len(cases)} refusals={refusals} fused={int(args.fused)}',
              '# SPDX-License-Identifier: Apache-2.0',
              '# Generated by autograd/checks/generate_higher_oracles.py' + (' --fused' if args.fused else '')
              + '; do not edit.',
              '# Expected first-order values: float64 central differences; second and third order:',
              '# float64 nested central differences, Richardson-extrapolated (h = 1e-3 and 1e-4 agree to 1e-6), or closed forms.',
              '# Every case prints PASS lines and one HIGHER_DIGEST line over its second- and',
              '# third-order words; FIRST_ORDER_MATCH counts the cases whose recorded first-order',
              '# gradient equals the fused first-order path bit for bit (information, not a gate).',
              'from std.math import tanh',
              'from std.memory import bitcast',
              'from autograd import Tape, Tensor',
              'from autograd.higher import backward_recorded, backward_scalar_recorded',
              'from autograd.layers import layer_norm',
              'from autograd.ops import RELU, SIGMOID, TANH, SOFTMAX',
              'from autograd.loss_ops import LS_MSE, LS_MAE, LS_HUBER, LS_BCE_LOGITS',
              'from sequence.exec_trait import Exec',
              'from sequence.ops import Args']
    run = ['', '', 'def run_higher_checks[E: Exec](mut ex: E) raises:', '    var tape = Tape()', '    var matches = 0']
    run += [f'    if _c_{c["name"]}(ex, tape):\n        matches += 1' for c in cases]
    run += ['    var refusals = _refusals(ex, tape)',
            f'    if refusals != {refusals}:',
            '        raise Error("higher-order refusals incomplete")',
            '    _lifecycle(ex, tape)',
            f'    print("FIRST_ORDER_MATCH " + String(matches) + "/{len(cases)}")',
            f'    print("PASS higher refusals=" + String(refusals))',
            f'    print("PASS higher-order cases={len(cases)}")']
    text = '\n'.join(header) + '\n' + PRELUDE + '\n'.join(body_lines) + '\n' + '\n'.join(refusal_lines) \
        + LIFECYCLE + '\n'.join(run) + '\n'
    args.output.write_text(text)
    manifest = {'method': 'NumPy float64 nested central differences, Richardson-extrapolated, h=1e-3 accepted when h=1e-4 agrees to 1e-6; '
                          'closed forms where trivial; float32-representable inputs and directions',
                'numpy': np.__version__, 'fused': args.fused, 'cases': records, 'refusals': refusals}
    print(json.dumps({'cases': len(cases), 'refusals': refusals, 'fused': args.fused,
                      'second_order_elements': sum(r['second_order_elements'] for r in records)}))
    return manifest


if __name__ == '__main__':
    main()
