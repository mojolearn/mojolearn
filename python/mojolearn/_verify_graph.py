# SPDX-License-Identifier: Apache-2.0
"""Test-only installed-package qualification for captured graph execution.

Python makes fixtures, independent oracles and reports. Product graph, control
flow, input conversion and profiling execute through native binding calls.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path
import sys
import tempfile

FORMAT = "mojolearn-graph-verification/1"
EXPECTED_CASES = (
    "native_inputs", "dynamic_shapes_and_recovery", "lazy_conditional",
    "bounded_while", "shape_growing_while", "exact_duplicate_elimination",
    "training_after_optimization", "node_diagnostics_and_profile",
    "saved_component_roundtrip", "native_session_boundary", "native_checkpoint_boundary", "native_weight_boundary",
    "variable_rank_replay",
    "executed_control_vjp", "executor_accounting", "control_checkpoint",
)
OUTPUT_COUNTS = dict(zip(EXPECTED_CASES, (4, 1, 1, 4, 1, 4, 4, 1, 1, 3, 2, 4, 6, 21, 2, 9)))


_AUDITS = []


def _invoke(operation, *args, **kwargs):
    from ._verify_native_runtime import audit_native_call
    audit = audit_native_call(operation, *args, **kwargs)
    _AUDITS.append(audit.report())
    if audit.exception is not None:
        if audit.frames or audit.forbidden_native or audit.unobserved_threads or audit.wrapper_work:
            audit.assert_native()
        raise audit.exception
    return audit.assert_native()


def _equal(actual, expected):
    import numpy as np
    a, b = np.asarray(actual), np.asarray(expected)
    assert a.shape == b.shape, (a.shape, b.shape)
    assert a.dtype == b.dtype, (a.dtype, b.dtype)
    assert a.tobytes() == b.tobytes(), "output words differ"


def _refuses(call, text, error_type=Exception):
    try:
        call()
    except Exception as error:
        assert isinstance(error, error_type), (error_type.__name__, type(error).__name__)
        assert text in str(error), (text, str(error))
        if _AUDITS and _AUDITS[-1]["exception"] is not None:
            _AUDITS[-1]["expected_refusal"] = True
        return
    raise AssertionError("invalid operation unexpectedly succeeded: " + text)


def native_inputs():
    import numpy as np
    from . import ops
    x = np.arange(12, dtype=np.float32).reshape(4, 3)
    with ops.Session() as s:
        h = s.tensor(x)
        p = s.capture(h, s.scale(h, 2.0))
        outputs = []
        for a in (x, x.astype(np.float64), x[:, ::-1], x.tolist()):
            out = _invoke(p.run, a)
            _equal(out, np.asarray(a, dtype=np.float32) * np.float32(2))
            outputs.append(out)
        _refuses(lambda: _invoke(p.run, x, x), "input arrays")
        _refuses(lambda: _invoke(p.run, [[1.0], [2.0, 3.0]]), "rectangular")
        _refuses(lambda: _invoke(p.run, x.astype(">f4")), "native byte order")
        integer = s.ids(np.array([[1, 2]], dtype=np.int64))
        id_program = s.capture(integer, s.embedding(s.tensor(np.ones((4, 2), np.float32)), integer))
        _refuses(lambda: _invoke(id_program.run, np.ones((1, 2), np.float32)), "integer", TypeError)
        _equal(_invoke(p.run, x), outputs[0])
        return outputs


def dynamic_shapes_and_recovery():
    import numpy as np
    from . import ops
    w = np.array([[1, 0], [0, 1], [1, -1]], dtype=np.float32)
    with ops.Session() as s:
        x = s.tensor(np.ones((2, 4, 3), dtype=np.float32))
        p = s.capture(x, s.matmul(x, s.tensor(w)))
        a = np.arange(18, dtype=np.float32).reshape(3, 2, 3)
        out = _invoke(p.run, a, dynamic_shapes=True)
        _equal(out, a @ w)
        _refuses(lambda: _invoke(p.run, a), "shape", ValueError)
        _refuses(lambda: _invoke(p.run, np.ones((2, 3), np.float32), dynamic_shapes=True), "rank")
        _refuses(lambda: _invoke(p.run, np.ones((3, 2, 5), np.float32), dynamic_shapes=True), "program node 0")
        _equal(_invoke(p.run, a, dynamic_shapes=True), out)
        return [out]


def variable_rank_replay():
    import numpy as np
    from . import ops
    w = np.array([[1, 0], [0, 1], [1, -1]], np.float32)
    outputs = []
    with tempfile.TemporaryDirectory(prefix="mojolearn-ranks-") as d, ops.Session() as s:
        x = s.tensor(np.zeros((2, 2, 3), np.float32))
        product = s.matmul(x, s.tensor(w))
        p = s.capture(x, product)
        reduced = s.capture(x, s.sum(product, axis=-1, keepdims=True))
        path = Path(d) / "rank-program.mjl"
        _invoke(p.save, path)
        loaded = _invoke(s.load_program, path)
        for shape in ((2, 3), (2, 2, 3), (2, 1, 2, 3), (1, 1, 1, 1, 1, 1, 2, 3)):
            a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            out = _invoke(p.run_dynamic, a)
            _equal(out, a @ w)
            _equal(_invoke(loaded.run_dynamic, a), out)
            outputs.append(out)
        for shape in ((2, 3), (2, 1, 2, 3)):
            a = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            out = _invoke(reduced.run_dynamic, a)
            _equal(out, np.sum(a @ w, axis=-1, keepdims=True))
            outputs.append(out)
        _refuses(lambda: _invoke(p.run_dynamic, np.ones((2, 5), np.float32)), "program node 0")
        _refuses(lambda: _invoke(p.run_dynamic, np.ones((1,) * 9, np.float32)), "rank", ValueError)
        _equal(_invoke(p.run_dynamic, np.arange(6, dtype=np.float32).reshape(2, 3)), outputs[0])
    return outputs


def executed_control_vjp():
    import numpy as np
    from . import ops
    outputs = []
    seed = np.array([[0.5]], np.float32)

    # Independent scalar-double oracle, including the stopping decision.
    # Perturbations stay away from predicate boundaries in all five fixtures.
    def oracle(x, w, loop, maximum):
        if loop:
            iterations = 0
            while x < 8 and iterations < maximum:
                x *= w
                iterations += 1
            return 0.5 * x
        return 0.5 * (x * w if x > 0 else x * x)

    with tempfile.TemporaryDirectory(prefix="mojolearn-control-vjp-") as d, ops.Session() as s:
        x = s.tensor([[1.25]])
        w = s.tensor([[2.0]])
        condition = s.capture(x, s.greater(x, s.full((1, 1), 0)))
        loop_condition = s.capture(x, s.less(x, s.full((1, 1), 8)))
        body = s.capture(x, s.mul(x, w))
        otherwise = s.capture(x, s.mul(x, x))
        for initial, loop, maximum in ((1.25, False, 0), (-1.25, False, 0),
                                       (1.25, True, 8), (9.0, True, 8), (1.25, True, 2)):
            a = np.array([[initial]], np.float32)
            if loop:
                report = _invoke(body.vjp_while, a, condition=loop_condition,
                                 max_iterations=maximum, seeds=seed, parameters=[w])
            else:
                report = _invoke(body.vjp_if, condition, otherwise, a, seeds=seed, parameters=[w])
            value, dx, dw = report["outputs"], report["input_gradients"][0], report["parameter_gradients"][w]
            expected = oracle(initial, 2.0, loop, maximum)
            _equal(value, np.array([[expected * 2]], np.float32))
            epsilon = 1e-5
            expected_dx = (oracle(initial + epsilon, 2.0, loop, maximum) -
                           oracle(initial - epsilon, 2.0, loop, maximum)) / (2 * epsilon)
            expected_dw = (oracle(initial, 2.0 + epsilon, loop, maximum) -
                           oracle(initial, 2.0 - epsilon, loop, maximum)) / (2 * epsilon)
            np.testing.assert_allclose(dx, [[expected_dx]], rtol=1e-6, atol=1e-7)
            np.testing.assert_allclose(dw, [[expected_dw]], rtol=1e-6, atol=1e-7)
            outputs.extend((value, dx, dw))
            if loop:
                assert report["iterations"] == (0 if initial == 9 else min(3, maximum))
                assert report["stopped"] is (initial == 9 or maximum >= 3)
        zero = _invoke(body.vjp_while, [[1.25]], condition=loop_condition,
                       max_iterations=0, seeds=seed, parameters=[w])
        assert zero["iterations"] == 0 and zero["stopped"] is False
        _equal(zero["input_gradients"][0], seed)
        _equal(zero["parameter_gradients"][w], np.zeros((1, 1), np.float32))
        _refuses(lambda: _invoke(body.vjp_if, condition, otherwise, [[1.25]],
                                seeds=np.ones((2, 1), np.float32)), "seed must have", ValueError)
        _refuses(lambda: _invoke(body.vjp_if, condition, otherwise, [[1.25]],
                                seeds=seed, parameters=[w, w]), "twice", ValueError)
        dead = s.tensor([[3.0]])
        dead_branch = s.capture(x, s.mul(x, dead))
        s.free(dead)
        lazy = _invoke(body.vjp_if, condition, dead_branch, [[1.25]], seeds=seed)
        _equal(lazy["outputs"], outputs[0])
        for name, program in (("body", body), ("condition", condition),
                              ("loop", loop_condition), ("otherwise", otherwise)):
            _invoke(program.save, Path(d) / (name + ".mjl"))
        with ops.Session() as restored:
            rb = _invoke(restored.load_program, Path(d) / "body.mjl")
            rc = _invoke(restored.load_program, Path(d) / "condition.mjl")
            rl = _invoke(restored.load_program, Path(d) / "loop.mjl")
            ro = _invoke(restored.load_program, Path(d) / "otherwise.mjl")
            rw, = rb.params
            branch = _invoke(rb.vjp_if, rc, ro, [[1.25]], seeds=seed, parameters=[rw])
            loop = _invoke(rb.vjp_while, [[1.25]], condition=rl, max_iterations=8,
                           seeds=seed, parameters=[rw])
            for report, offset in ((branch, 0), (loop, 6)):
                _equal(report["outputs"], outputs[offset])
                _equal(report["input_gradients"][0], outputs[offset + 1])
                _equal(report["parameter_gradients"][rw], outputs[offset + 2])
        multi = s.capture(x, [s.mul(x, w), s.mul(x, x)])
        both = _invoke(multi.vjp_if, condition, multi, [[1.25]],
                       seeds=(seed, np.array([[0.25]], np.float32)), parameters=[w])
        _equal(both["input_gradients"][0], np.array([[1.625]], np.float32))
        _equal(both["parameter_gradients"][w], np.array([[0.625]], np.float32))
        outputs.extend((*both["outputs"], both["input_gradients"][0], both["parameter_gradients"][w]))
        growing = s.capture(x, s.concat([x, x], axis=0))
        growing_condition = s.capture(x, s.less(s.sum(x), s.full((1, 1), 4)))
        grown = _invoke(growing.vjp_while, [[1.25]], condition=growing_condition,
                        max_iterations=3, seeds=np.ones((4, 1), np.float32), parameters=[])
        assert grown["iterations"] == 2 and grown["stopped"] is True
        _equal(grown["outputs"], np.full((4, 1), 1.25, np.float32))
        _equal(grown["input_gradients"][0], np.array([[4]], np.float32))
        outputs.extend((grown["outputs"], grown["input_gradients"][0]))
        integer_body = s.capture(x, s.argmax(x, keepdims=True, int32=True))
        _refuses(lambda: _invoke(integer_body.vjp_while, [[1.25]], condition=loop_condition,
                                max_iterations=1, seeds=seed), "outputs must hold floating", ValueError)
        integer_x = s.ids(np.array([[0]], np.int32))
        integer_identity = s.capture(integer_x, integer_x)
        _refuses(lambda: _invoke(integer_identity.vjp_while, np.array([[0]], np.int32),
                                condition=integer_identity, max_iterations=1, seeds=seed),
                 "inputs must hold floating", ValueError)
        _equal(_invoke(body.vjp_if, condition, otherwise, [[1.25]], seeds=seed)["outputs"], outputs[0])
    return outputs


def lazy_conditional():
    import numpy as np
    from . import ops
    a = np.ones((2, 3), np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        condition = s.capture(x, s.greater(s.sum(x), s.full((1, 1), 0)))
        body = s.capture(x, s.scale(x, 2))
        weight = s.tensor(np.ones((3, 2), np.float32))
        other = s.capture(x, s.matmul(x, weight))
        s.free(weight)
        out = _invoke(body.run_if, condition, other, a)
        _equal(out, a * np.float32(2))
        # The freed parameter proves the unselected branch was not replayed
        # or eagerly validated. Selecting it must fail, then recovery works.
        _refuses(lambda: _invoke(body.run_if, condition, other, -a), "freed")
        _equal(_invoke(body.run_if, condition, other, a), out)
        return [out]


def bounded_while():
    import numpy as np
    from . import ops
    a = np.array([[0]], np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        condition = s.capture(x, s.less(x, s.full((1, 1), 3)))
        body = s.capture(x, s.add(x, s.full((1, 1), 1)))
        done = _invoke(body.while_loop, a, condition=condition, max_iterations=10)
        assert done["iterations"] == 3 and done["stopped"] is True
        _equal(done["outputs"], np.array([[3]], np.float32))
        bound = _invoke(body.while_loop, a, condition=condition, max_iterations=2)
        assert bound["iterations"] == 2 and bound["stopped"] is False
        _equal(bound["outputs"], np.array([[2]], np.float32))
        zero = _invoke(body.while_loop, a, condition=condition, max_iterations=0)
        assert zero["iterations"] == 0 and zero["stopped"] is False
        _equal(zero["outputs"], a)
        stopped = _invoke(body.while_loop, np.array([[5]], np.float32), condition=condition, max_iterations=0)
        assert stopped["iterations"] == 0 and stopped["stopped"] is True
        nonfinite = s.capture(x, x)
        _refuses(lambda: _invoke(body.while_loop, np.array([[np.nan]], np.float32), condition=nonfinite,
                                        max_iterations=3), "finite")
        _refuses(lambda: _invoke(body.while_loop, a, condition=condition, max_iterations=-1), "must be >= 0")
        _refuses(lambda: _invoke(body.while_loop, a, condition=condition, max_iterations=1.5), "integer")
        _equal(_invoke(body.while_loop, a, condition=condition, max_iterations=3)["outputs"], done["outputs"])
        return [done["outputs"], bound["outputs"], zero["outputs"], stopped["outputs"]]


def shape_growing_while():
    import numpy as np
    from . import ops
    a = np.ones((1, 1), np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        condition = s.capture(x, s.less(s.sum(x), s.full((1, 1), 4)))
        body = s.capture(x, s.concat([x, x], axis=0))
        out = _invoke(body.while_loop, a, condition=condition, max_iterations=5)
        assert out["iterations"] == 2 and out["stopped"] is True
        _equal(out["outputs"], np.ones((4, 1), np.float32))
        invalid = s.capture(x, s.reshape(x, (1, 1, 1)))
        _refuses(lambda: _invoke(invalid.while_loop, a, condition=condition, max_iterations=1), "preserve")
        _equal(_invoke(body.while_loop, a, condition=condition, max_iterations=5)["outputs"], out["outputs"])
        return [out["outputs"]]


def exact_duplicate_elimination():
    import numpy as np
    from . import ops
    a = np.array([[-0.0, 1, -2], [3, -4, 0.0]], np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        u, v = s.relu(x), s.relu(x)
        p = s.capture(x, s.add(u, v))
        before = _invoke(p.inspect, )
        candidate = _invoke(p.inspect, optimize=True)
        assert len(before["nodes"]) == 3 and candidate["nodes_removed"] == 1
        eager, reduced = _invoke(p.run, a), _invoke(p.run, a, optimize=True)
        _equal(eager, reduced)
        assert _invoke(p.inspect, ) == before, "optimization must not mutate the captured program"
        plus, minus = s.scale(x, 0.0), s.scale(x, -0.0)
        zeros = s.capture(x, [plus, minus])
        assert _invoke(zeros.inspect, optimize=True)["nodes_removed"] == 0
        zo = _invoke(zeros.run, a, optimize=True)
        _equal(zo[0], a * np.float32(0))
        _equal(zo[1], a * np.float32(-0.0))
        d = s.capture(x, [s.dropout(x, 0.25, seed=3), s.dropout(x, 0.25, seed=3)])
        assert _invoke(d.inspect, optimize=True)["nodes_removed"] == 0
        return [eager, reduced, *zo]


def training_after_optimization():
    import numpy as np
    from . import ops
    a = np.arange(12, dtype=np.float32).reshape(4, 3) / np.float32(8)
    target = np.ones((4, 1), np.float32)
    states = []
    for optimize_first in (False, True):
        with ops.Session() as s:
            x = s.tensor(a)
            w = s.tensor(np.zeros((3, 1), np.float32))
            p = s.capture(x, s.add(s.matmul(x, w), s.matmul(x, w)))
            _invoke(p.run, a, optimize=optimize_first)
            opt = s.optimizer("adam", lr=0.01)
            _invoke(p.fit, a, targets=target, optimizer=opt, epochs=3, batch_size=2, shuffle=False, seed=2**63 + 9)
            state = s.optimizer_state(opt, w)
            assert state["step"] == 6
            states.append([s.numpy(w), np.asarray(p.loss_curve_, np.float32),
                           s.numpy(state["exp_avg"]), s.numpy(state["exp_avg_sq"])])
    for a, b in zip(*states):
        _equal(a, b)
    return states[0]


def node_diagnostics_and_profile():
    import numpy as np
    from . import ops
    a = np.ones((2, 3), np.float32)
    with ops.Session() as s:
        x = s.tensor(a)
        w = s.tensor(np.ones((3, 2), np.float32))
        p = s.capture(x, s.relu(s.matmul(x, w)))
        report = _invoke(p.profile, a)
        out, profile = report["outputs"], report["profile"]
        _equal(out, a @ np.ones((3, 2), np.float32))
        assert profile["executed_nodes"] == len(profile["nodes"]) == 2
        assert profile["input_logical_bytes"] == a.nbytes
        assert profile["output_bytes"] == out.nbytes
        assert profile["parameter_bytes"] == 24
        assert profile["node_sync_count"] == 4
        assert profile["peak_tape_reserved_bytes"] >= out.nbytes
        executor = profile["executor"]
        assert executor["complete"] and executor["operations"]
        assert executor["sync_calls"] >= profile["node_sync_count"] + 1
        assert executor["retired_bytes"] == executor["staging_bytes"] == 0
        assert executor["peak_retained_bytes"] >= executor["retained_bytes"]
        _invoke(s.execution_profile_begin)
        _refuses(lambda: _invoke(p.profile, a), "already active", ValueError)
        _invoke(s.execution_profile_end)
        _refuses(lambda: _invoke(p.profile, np.ones((2, 5), np.float32), dynamic_shapes=True),
                 "program node 0 (matmul)")
        _equal(_invoke(p.profile, a)["outputs"], out)
        for row in profile["nodes"]:
            assert row["enqueue_ns"] >= 0 and row["wait_ns"] >= 0
        _refuses(lambda: _invoke(p.run, np.ones((2, 5), np.float32), dynamic_shapes=True),
                 "program node 0 (matmul) inputs=[(2, 5), (3, 2)]")
        _equal(_invoke(p.run, a), out)
        return [out]


def control_checkpoint():
    import struct
    import numpy as np
    from . import ops
    outputs = []
    seed = np.array([[0.5]], np.float32)
    with tempfile.TemporaryDirectory(prefix="mojolearn-control-checkpoint-") as d, ops.Session() as s:
        root = Path(d)
        x, w = s.tensor([[1.25]]), s.tensor([[2.0]])
        condition = s.capture(x, s.less(x, w))
        body = s.capture(x, s.mul(x, w))
        other = s.capture(x, s.add(x, w))
        branch_file, loop_file = root / "conditional.mjl", root / "loop.mjl"
        _invoke(body.save_control, branch_file, condition=condition, otherwise=other)
        _invoke(body.save_control, loop_file, condition=condition, max_iterations=3)
        assert sorted(path.name for path in root.iterdir()) == ["conditional.mjl", "loop.mjl"]
        _refuses(lambda: _invoke(body.save_control, root / "bad.mjl", condition=condition,
                                otherwise=other, max_iterations=2), "does not take", ValueError)
        _refuses(lambda: _invoke(body.save_control, root / "bad.mjl", condition=condition,
                                max_iterations=-1), "must be >=", ValueError)
        with ops.Session() as restored:
            branch = _invoke(restored.load_control, branch_file)
            loop = _invoke(restored.load_control, loop_file)
            rb, rc, ro = branch["body"], branch["condition"], branch["otherwise"]
            rw, = rb.params
            assert rc.params == ro.params == (rw,), "shared parameters must restore to one handle"
            assert branch["kind"] == "if" and branch["max_iterations"] is None
            assert loop["kind"] == "while" and loop["max_iterations"] == 3 and loop["otherwise"] is None
            assert loop["body"].params == loop["condition"].params
            result = _invoke(rb.vjp_if, rc, ro, [[1.25]], seeds=seed, parameters=[rw])
            _equal(result["outputs"], np.array([[2.5]], np.float32))
            _equal(result["input_gradients"][0], np.array([[1]], np.float32))
            _equal(result["parameter_gradients"][rw], np.array([[0.625]], np.float32))
            outputs.extend((result["outputs"], result["input_gradients"][0], result["parameter_gradients"][rw]))
            lw, = loop["body"].params
            result = _invoke(loop["body"].vjp_while, [[1.25]], condition=loop["condition"],
                             max_iterations=loop["max_iterations"], seeds=seed, parameters=[lw])
            assert result["iterations"] == 1 and result["stopped"]
            outputs.extend((result["outputs"], result["input_gradients"][0], result["parameter_gradients"][lw]))
            for a, b in zip(outputs[:3], outputs[3:]):
                _equal(a, b)
            _invoke(restored.load_weights, {"w": np.array([[4]], np.float32)}, [(rw, "w")])
            selected = _invoke(rb.run_if, rc, ro, [[3]])
            _equal(selected, np.array([[12]], np.float32))
            _invoke(restored.load_weights, {"w": np.array([[1]], np.float32)}, [(rw, "w")])
            alternate = _invoke(rb.run_if, rc, ro, [[3]])
            _equal(alternate, np.array([[4]], np.float32))
            outputs.extend((selected, alternate))
            # Deliberately damage the control record while retaining a valid
            # outer container hash: these exercise semantic parser refusals.
            original = branch_file.read_bytes()
            extension = sum(struct.unpack_from("<4I", original, 12))
            ni, _ = struct.unpack_from("<2I", original, extension)
            ints = list(struct.unpack_from("<" + "q" * ni, original, extension + 8))
            cursor = 4
            for _ in range(ints[3]):
                cursor += ints[cursor] + 2  # rank, dimensions, dtype kind
            first_reference = cursor + 3  # component lengths precede refs
            for tag, index, bad_value, reason in (("kind", 1, 7, "invalid kind"),
                                                  ("reference", first_reference, ints[3], "out of range")):
                raw = bytearray(original)
                struct.pack_into("<q", raw, extension + 8 + index * 8, bad_value)
                digest = 0xCBF29CE484222325
                for byte in raw[:-8]:
                    digest = ((digest ^ byte) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
                struct.pack_into("<Q", raw, len(raw) - 8, digest)
                damaged = root / (tag + ".mjl")
                damaged.write_bytes(raw)
                _refuses(lambda: _invoke(restored.load_control, damaged), reason, ValueError)
            truncated = root / "truncated.mjl"
            truncated.write_bytes(original[:-9])
            _refuses(lambda: _invoke(restored.load_control, truncated), "CKPT_")
            _equal(_invoke(rb.run_if, rc, ro, [[3]]), alternate)
        empty = s.capture(x, x)
        _invoke(empty.save_control, root / "empty.mjl", condition=empty, max_iterations=0)
        with ops.Session() as restored:
            bundle = _invoke(restored.load_control, root / "empty.mjl")
            assert bundle["body"].params == bundle["condition"].params == ()
            value = _invoke(bundle["body"].while_loop, [[1]], condition=bundle["condition"],
                            max_iterations=bundle["max_iterations"])["outputs"]
            _equal(value, np.array([[1]], np.float32))
            outputs.append(value)
        _invoke(body.save, root / "ordinary.mjl")
        _refuses(lambda: _invoke(s.load_control, root / "ordinary.mjl"), "not a control-flow", ValueError)
    return outputs


def executor_accounting():
    import numpy as np
    from . import ops
    a = np.array([[1, -2], [3, -4]], np.float32)
    with ops.Session(record=True) as s:
        _refuses(lambda: _invoke(s.execution_profile_end), "no execution profile", ValueError)
        _invoke(s.execution_profile_begin)
        _refuses(lambda: _invoke(s.execution_profile_begin), "already active", ValueError)
        x = _invoke(s.tensor, a, requires_grad=True)
        y = _invoke(s.sum, _invoke(s.mul, x, x))
        _invoke(s.backward, y)
        gradient = _invoke(s.grad, x)
        value = _invoke(s.numpy, y)
        _equal(gradient, a * np.float32(2))
        _equal(value, np.array([[30]], np.float32))
        _invoke(s.free, x)
        _invoke(s.reset)  # Session retirement returns owned storage at this boundary.
        report = _invoke(s.execution_profile_end)
        assert report["format"] == "mojolearn-execution-profile/1" and report["complete"]
        assert report["allocations"] > 0 and report["allocated_bytes"] >= a.nbytes
        assert report["freed_bytes"] >= a.nbytes and report["frees"] >= 1
        assert report["retained_bytes"] == report["live_bytes"]
        assert report["peak_retained_bytes"] >= report["retained_bytes"]
        assert report["retired_bytes"] == report["staging_bytes"] == 0
        assert report["sync_calls"] > 0 and report["sync_ns"] >= 0
        assert report["operations"] and all(row["calls"] > 0 for row in report["operations"])
        assert all(row["host_call_ns"] >= 0 for row in report["operations"])
        assert report["upload_host_ns"] >= 0 and report["download_host_ns"] >= 0
        if report["device"]:
            assert report["kernel_enqueues"] > 0
            assert report["upload_bytes"] >= a.nbytes
            assert report["download_bytes"] >= value.nbytes + gradient.nbytes
            assert report["peak_staging_bytes"] > 0
        else:
            assert report["kernel_enqueues"] == report["upload_bytes"] == report["download_bytes"] == 0
            assert report["host_copy_bytes"] >= a.nbytes + value.nbytes + gradient.nbytes
        _refuses(lambda: _invoke(s.execution_profile_end), "no execution profile", ValueError)
        _invoke(s.execution_profile_begin)
        _refuses(lambda: _invoke(s.numpy, x), "not a live", ValueError)
        empty = _invoke(s.execution_profile_end)
        assert empty["allocations"] == empty["frees"] == 0 and empty["operations"] == []
        assert empty["retained_bytes"] == report["retained_bytes"]
    return [value, gradient]


def saved_component_roundtrip():
    import numpy as np
    from . import ops
    a = np.array([[0]], np.float32)
    with tempfile.TemporaryDirectory(prefix="mojolearn-graph-") as d:
        with ops.Session() as s:
            x = s.tensor(a)
            condition = s.capture(x, s.less(x, s.full((1, 1), 3)))
            body = s.capture(x, s.add(x, s.full((1, 1), 1)))
            _invoke(condition.save, Path(d) / "condition.mjl")
            _invoke(body.save, Path(d) / "body.mjl")
            for name in ("condition", "body"):
                path = Path(d) / (name + ".mjl")
                report = json.loads(Path(str(path) + ".verify.json").read_text())
                assert report["format"] == "mojolearn-model-verify/1"
                assert report["model"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
                assert report["model"]["bytes"] == path.stat().st_size
                assert report["inputs"][0]["shape"] == [1, 1]
                assert report["outputs"][0]["shape"] == [1, 1]
                from ._verify_model import verify, compare
                assert compare(verify(str(path), expect=report), report)[0] == "MATCH"
        with ops.Session() as s:
            condition = _invoke(s.load_program, Path(d) / "condition.mjl")
            body = _invoke(s.load_program, Path(d) / "body.mjl")
            out = _invoke(body.while_loop, a, condition=condition, max_iterations=4)
            assert out["iterations"] == 3 and out["stopped"] is True
            _equal(out["outputs"], np.array([[3]], np.float32))
            return [out["outputs"]]


def native_session_boundary():
    import numpy as np
    import operator
    from . import ops
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    _refuses(lambda: _invoke(ops.Session, numeric_mode="invalid"), "numeric_mode", ValueError)
    held = [_invoke(ops.Session)]
    import weakref
    released_session = weakref.ref(held[0])
    released_sid = held[0]._id
    released_tensor = held[0].tensor([[1.0]])
    native_shape = held[0]._b.ops_shape
    _invoke(operator.delitem, held, 0)
    assert released_session() is None, "native setup must not retain a Session reference"
    _refuses(lambda: _invoke(native_shape, released_sid, released_tensor), "session")
    with tempfile.TemporaryDirectory(prefix="mojolearn-session-lifetime-") as d:
        with ops.Session() as source:
            x = source.tensor([[2.0]])
            program = source.capture(x, source.scale(x, 3))
            program.save(Path(d) / "program.mjl")
        loaded_session = _invoke(ops.Session)
        loaded = _invoke(loaded_session.load_program, Path(d) / "program.mjl")
        owner_ref, program_ref = weakref.ref(loaded_session), weakref.ref(loaded)
        loaded_sid = loaded_session._id
        loaded_tensor = loaded_session.tensor([[1]])
        native_loaded_shape = loaded_session._b.ops_shape
        del loaded_session
        assert owner_ref() is not None, "a live Program must retain its owning Session"
        _equal(_invoke(loaded.run, [[2]]), np.array([[6]], np.float32))
        programs = [loaded]
        del loaded
        _invoke(operator.delitem, programs, 0)
        assert program_ref() is None and owner_ref() is None, "native load must not leak Program or Session references"
        _refuses(lambda: _invoke(native_loaded_shape, loaded_sid, loaded_tensor), "session")
    with _invoke(ops.Session, numeric_mode=" IDENTICAL ") as s:
        assert _invoke(s.__enter__) is s
        x = _invoke(s.tensor, a.tolist())
        y = _invoke(s.scale, x, 2.0)
        pieces = _invoke(s.split, y, [1, 2], axis=1)
        joined = _invoke(s.concat, pieces, axis=1)
        out = _invoke(s.numpy, joined)
        _equal(out, a * np.float32(2))
        assert _invoke(s.shape, joined) == (2, 3)
        p = _invoke(s.capture, x, joined)
        _equal(_invoke(p.run, a), out)
        _refuses(lambda: _invoke(s.reshape, x, (True, 3)), "integer", TypeError)
        _refuses(lambda: _invoke(s.ids, a), "integer", TypeError)
        _refuses(lambda: _invoke(s.tensor, np.zeros((1,) * 9, np.float32)), "rank", ValueError)
        _refuses(lambda: _invoke(s.tensor, np.zeros((2, 3, 4), np.float32).transpose(2, 1, 0)), "C-contiguous", ValueError)
        table = _invoke(s.tensor, a)
        ids = np.array([1, 0], np.int64)
        _equal(_invoke(s.numpy, _invoke(s.embedding, table, ids)), a[ids])
        _equal(_invoke(s.numpy, _invoke(s.take, table, ids, axis=0)), a[ids])
        _refuses(lambda: _invoke(s.gelu, x, approximate="bad"), "approximate", ValueError)
    with _invoke(ops.Session, record=True) as s:
        x = _invoke(s.tensor, [[2.0]], requires_grad=True)
        loss = _invoke(s.mul, x, x)
        _invoke(s.backward, loss, create_graph=True)
        first = _invoke(s.grad, x)
        _equal(first, np.array([[4]], np.float32))
        gradient = _invoke(s.grad_tensor, x)
        _invoke(s.backward, gradient)
        second = _invoke(s.grad, x)
        _equal(second, np.array([[2]], np.float32))
        opt = _invoke(s.optimizer, "sgd", lr=0.25)
        _refuses(lambda: _invoke(s.step, opt + 1, x), "optimizer", ValueError)
        _invoke(s.step, opt, x)
        _equal(_invoke(s.numpy, x), np.array([[1.5]], np.float32))
        state = _invoke(s.optimizer_state, opt, x)
        assert state["step"] == 1
        _invoke(s.load_optimizer_state, opt, x, state)
        _invoke(s.zero_grad)
        _invoke(s.reset)
        _invoke(s.free, x)
        assert _invoke(s.__exit__, None, None, None) is False
        _invoke(s.close)
    return [out, first, second]


def native_checkpoint_boundary():
    import numpy as np
    from . import ops
    with tempfile.TemporaryDirectory(prefix="mojolearn-state-") as d:
        path = Path(d) / "state.ckpt"
        with ops.Session(record=True) as s:
            x = s.tensor([[2.0]], requires_grad=True)
            opt = s.optimizer("sgd", lr=0.25, momentum=0.5)
            s.backward(s.mul(x, x))
            s.step(opt, x)
            saved = s.numpy(x)
            _equal(saved, np.array([[1]], np.float32))
            _refuses(lambda: _invoke(s.save_state, path, [x], seed=-1), "seed must be in", ValueError)
            _refuses(lambda: _invoke(s.save_state, path, [x], seed=2**64), "seed must be in", ValueError)
            _refuses(lambda: _invoke(s.save_state, path, [x], seed=True), "seed must be an integer", TypeError)
            _invoke(s.save_state, path, [x], optimizer=opt, seed=2**63 + 1, epoch=3, step=1)
            s.reset()
            s.backward(s.mul(x, x))
            s.step(opt, x)
            expected = s.numpy(x)
        with ops.Session(record=True) as s:
            opt = s.optimizer("sgd", lr=0.25, momentum=0.5)
            state = _invoke(s.load_state, path, optimizer=opt)
            assert state["seed"] == 2**63 + 1 and state["epoch"] == 3 and state["step"] == 1
            x, = state["params"]
            _equal(s.numpy(x), saved)
            s.backward(s.mul(x, x))
            s.step(opt, x)
            restored = s.numpy(x)
            _equal(restored, expected)
            _equal(restored, np.array([[0]], np.float32))
    return [saved, restored]


def native_weight_boundary():
    import numpy as np
    import struct
    from . import ops
    weights = np.arange(6, dtype=np.float16).reshape(3, 2)
    bias = np.array([0.5, -0.5], np.float64)
    with ops.Session() as s:
        w = s.tensor(np.zeros((2, 3), np.float32))
        b = s.tensor(np.zeros((1, 2), np.float32))
        mapping = {"weight": weights, "bias": bias}
        assert _invoke(s.load_weights, mapping, [(w, "weight", "transpose"), (b, "bias")]) == [w, b]
        first, second = s.numpy(w), s.numpy(b)
        _equal(first, weights.T.astype(np.float32))
        _equal(second, bias.astype(np.float32).reshape(1, 2))
        bad = {"weight": weights + np.float16(10), "bias": np.zeros(7, np.float32)}
        _refuses(lambda: _invoke(s.load_weights, bad, [(w, "weight", "transpose"), (b, "bias")]), "shape")
        _equal(s.numpy(w), first)
        _refuses(lambda: _invoke(s.load_weights, mapping, [(w, "missing")]), "no tensor", ValueError)
        with tempfile.TemporaryDirectory(prefix="mojolearn-weights-") as d:
            root = Path(d)
            _refuses(lambda: _invoke(s.load_weights, root / "missing.safetensors", [(w, "weight")]),
                     "does not exist", FileNotFoundError)
            _refuses(lambda: _invoke(s.load_weights, root, [(w, "weight")]),
                     "checkpoint directory", FileNotFoundError)
            for filename, name, array, dtype in (("a.safetensors", "weight", weights, "F16"),
                                                   ("b.safetensors", "bias", bias, "F64")):
                header = json.dumps({name: {"dtype": dtype, "shape": list(array.shape),
                                           "data_offsets": [0, array.nbytes]}}).encode()
                (root / filename).write_bytes(struct.pack("<Q", len(header)) + header + array.tobytes())
            (root / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                "weight": "a.safetensors", "bias": "b.safetensors"}}))
            _invoke(s.load_weights, root, [(w, "weight", "transpose"), (b, "bias")])
            third, fourth = s.numpy(w), s.numpy(b)
            _equal(third, first)
            _equal(fourth, second)
    return [first, second, third, fourth]


def _digest(a):
    import numpy as np
    x = np.ascontiguousarray(a)
    return {"shape": list(x.shape), "dtype": x.dtype.str,
            "sha256": hashlib.sha256(x.tobytes()).hexdigest()}


def compare_reports(actual, expected):
    errors = []
    for label, report in (("actual", actual), ("expected", expected)):
        if report.get("format") != FORMAT or report.get("expected_cases") != list(EXPECTED_CASES):
            errors.append(label + ": schema or coverage differs")
        if report.get("backend", {}).get("numeric_mode") != 1:
            errors.append(label + ": native identical mode is unproven")
        rows = report.get("cases", {})
        if set(rows) != set(EXPECTED_CASES):
            errors.append(label + ": missing or unexpected cases")
        for name in EXPECTED_CASES:
            row = rows.get(name, {})
            if row.get("status") != "passed" or len(row.get("outputs", [])) != OUTPUT_COUNTS[name]:
                errors.append(label + ": incomplete " + name)
            for output in row.get("outputs", []):
                if (not re.fullmatch(r"[0-9a-f]{64}", str(output.get("sha256", "")))
                        or output.get("dtype") != "<f4"
                        or not isinstance(output.get("shape"), list)
                        or not output["shape"]
                        or not all(isinstance(d, int) and d > 0 for d in output["shape"])):
                    errors.append(label + ": malformed output evidence " + name)
            audits = row.get("native_calls", [])
            if not audits or any(a.get("python_frames") or a.get("forbidden_native_calls")
                                 or a.get("wrapper_work") or a.get("unobserved_existing_threads")
                                 or not (a.get("passed") or a.get("expected_refusal")) for a in audits):
                errors.append(label + ": native execution unproven " + name)
    if errors:
        return errors
    for name in EXPECTED_CASES:
        if actual["cases"][name]["outputs"] != expected["cases"][name]["outputs"]:
            errors.append("output words differ: " + name)
    return errors


def comparator_self_test(report):
    assert not compare_reports(report, report)
    corrupt = copy.deepcopy(report)
    first = EXPECTED_CASES[0]
    corrupt["cases"][first]["outputs"][0]["sha256"] = "0" * 64
    assert compare_reports(report, corrupt), "one changed digest must fail"
    missing = copy.deepcopy(report)
    del missing["cases"][first]
    assert compare_reports(report, missing), "missing evidence must fail"
    failed = copy.deepcopy(report)
    failed["cases"][first]["status"] = "failed"
    assert compare_reports(report, failed), "failed execution must fail comparison"
    fast = copy.deepcopy(report)
    fast["backend"]["numeric_mode"] = 0
    assert compare_reports(report, fast), "wrong numeric mode must fail comparison"
    return {"same_report": "passed", "changed_digest": "rejected", "missing_case": "rejected",
            "failed_case": "rejected", "wrong_mode": "rejected"}


def run_checks(vendor, require_installed=False):
    from . import _backend
    import mojolearn
    report = {"format": FORMAT, "expected_cases": list(EXPECTED_CASES), "cases": {}, "backend": {}}
    try:
        b = _backend.binding("_mojolearn_x_sequence", "identical")
        report["backend"] = {"vendor": str(b.x_sequence_vendor()), "numeric_mode": int(b.x_sequence_numeric_mode()),
                             "binding_file": str(getattr(b, "__file__", "")), "binding_type": type(b).__name__,
                             "package_file": str(Path(mojolearn.__file__).resolve()), "python": sys.executable}
        assert report["backend"]["vendor"] == vendor, report["backend"]
        assert report["backend"]["numeric_mode"] == 1, report["backend"]
        for entry in ("ops_program_execute", "ops_program_control", "ops_program_inspect"):
            assert hasattr(b, entry), "missing native entry " + entry
        if require_installed:
            package_path = Path(mojolearn.__file__).resolve()
            assert "site-packages" in package_path.parts or "dist-packages" in package_path.parts, package_path
            assert package_path.is_relative_to(Path(sys.prefix).resolve()), (package_path, sys.prefix)
    except Exception as error:
        for name in EXPECTED_CASES:
            report["cases"][name] = {"status": "cannot_run", "error": str(error), "outputs": []}
        return report
    for name in EXPECTED_CASES:
        _AUDITS.clear()
        try:
            values = globals()[name]()
            report["cases"][name] = {"status": "passed", "outputs": [_digest(a) for a in values],
                                      "native_calls": list(_AUDITS)}
        except Exception as error:
            report["cases"][name] = {"status": "failed", "error": str(error), "outputs": [],
                                      "native_calls": list(_AUDITS)}
    if all(row["status"] == "passed" for row in report["cases"].values()):
        report["comparator_controls"] = comparator_self_test(report)
    return report


def cmd_verify_graph(args):
    report = run_checks(args.vendor, args.require_installed)
    rc = 0 if all(row["status"] == "passed" for row in report["cases"].values()) else 4
    if args.expect:
        try:
            expected = json.loads(Path(args.expect).read_text())
            errors = compare_reports(report, expected)
        except Exception as error:
            errors = ["reference could not be read: " + str(error)]
        report["comparison"] = {"expected": args.expect, "errors": errors, "matched": not errors}
        if errors and rc == 0:
            rc = 1
    report["exit_code"] = rc
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        counts = {status: sum(row["status"] == status for row in report["cases"].values())
                  for status in ("passed", "failed", "cannot_run")}
        print("GRAPH", report["backend"].get("vendor", "unknown"), counts, "exit", rc)
    return rc


def add_parser(sub):
    p = sub.add_parser("verify-graph", help="qualify this install's native captured graph execution")
    p.add_argument("--vendor", required=True, choices=("cpu", "metal", "cuda", "hip"))
    p.add_argument("--require-installed", action="store_true")
    p.add_argument("--expect", help="compare all output bits with another vendor's report")
    p.add_argument("--out", help="retain the complete report at this path")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_verify_graph)
