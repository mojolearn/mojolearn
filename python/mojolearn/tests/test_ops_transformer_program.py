# SPDX-License-Identifier: Apache-2.0
"""Independent primitive derivatives and native whole-block capture checks."""
import numpy as np
import pytest

from mojolearn import ops
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn import _verify_graph as verify


def native(call, *args, **kwargs):
    return audit_native_call(call, *args, **kwargs).assert_native()


# The rms_norm, rotary, projection-vjp and transformer-softmax values,
# derivatives and reduction contracts are checked natively in
# checks/native_tests/transformer_ops_program_*.mojo (2026-11-14). The
# refusal and the captured-program checks below stay at the Python API.


def test_projection_first_vjp_refuses_create_graph():
    x = np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 16
    w = np.arange(20, dtype=np.float32).reshape(5, 4) / 32 - .2
    seed = np.linspace(-.3, .7, 30, dtype=np.float32).reshape(2, 3, 5)
    with ops.Session(record=True) as s:
        hx, hw = s.tensor(x, requires_grad=True), s.tensor(w, requires_grad=True)
        y = native(s._transformer_linear, hx, hw)
        seed_handle = s.tensor(seed)
        # The established recorded-custom-rule contract raises Exception,
        # with a named first-order-only refusal before any backward launch.
        with pytest.raises(Exception, match=r"create_graph cannot differentiate.*tag 210.*first-order rule only") as refusal:
            s.vjp(y, [hx, hw], seed_handle, create_graph=True)
        assert type(refusal.value) is Exception


@pytest.mark.parametrize("columns", [5, 33, 65])
def test_transformer_softmax_program_run_and_checkpoint_keep_the_serial_contract(columns, tmp_path):
    rng = np.random.default_rng(718 + columns)
    x = rng.normal(0, .7, (2, 3, columns)).astype(np.float32)
    with ops.Session() as s:
        h = s.tensor(x)
        # A separate native exp operation establishes the rounded terms;
        # the test controller computes the serial sum independently.
        exp = native(s.numpy, s.exp(s.sub(h, s.max(h, axis=-1, keepdims=True))))
        serial = np.zeros(x.shape[:-1], np.float32)
        for k in range(columns):
            serial = np.add(serial, exp[..., k], dtype=np.float32)
        expected = exp / serial[..., None]
        p = native(s.capture, h, s._transformer_softmax(h))
        np.testing.assert_array_equal(native(p.run, x, optimize=True).view(np.uint32), expected.view(np.uint32))
        checkpoint = tmp_path / "transformer-softmax.mjl"
        native(p.save, checkpoint)
        restored = native(s.load_program, checkpoint)
        report = native(restored.inspect)
        assert any(node["op"] == "transformer_softmax" for node in report["nodes"])
        np.testing.assert_array_equal(native(restored.run, x).view(np.uint32), expected.view(np.uint32))


@pytest.mark.parametrize("bad", ["rms_weight", "rms_epsilon", "rms_nan", "rotary_odd", "rotary_width", "rotary_domain", "rotary_theta", "integer"])
def test_native_primitive_refusals_and_recovery(bad):
    with ops.Session() as s:
        x = s.tensor(np.ones((1, 3, 8), np.float32))
        w = s.tensor(np.ones(8, np.float32))
        function, args, kwargs = s.rms_norm, (x, w), {}
        if bad == "rms_weight":
            args = (x, s.tensor(np.ones(7, np.float32)))
        elif bad in ("rms_epsilon", "rms_nan"):
            kwargs = {"eps": 0 if bad == "rms_epsilon" else float("nan")}
        elif bad == "integer":
            args = (s.ids(np.ones((1, 3, 8), np.int64)), w)
        else:
            function, args = s.rotary, (x, 3 if bad == "rotary_odd" else (6 if bad == "rotary_width" else 4))
            if bad == "rotary_domain":
                kwargs = {"offset": 8191}
            elif bad == "rotary_theta":
                kwargs = {"theta": -1.}
        audit = audit_native_call(function, *args, **kwargs)
        report = audit.report()
        assert isinstance(audit.exception, ValueError), report
        assert not any(report[k] for k in ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads")), report
        native(s.numpy, native(s.rms_norm, x, w))


@pytest.mark.parametrize("where", ["epsilon", "theta", "head_dim", "handle"])
def test_primitive_scalar_hooks_refused_before_dispatch(where):
    seen = []
    class FloatHook(float):
        def __float__(self):
            seen.append("float")
            raise AssertionError("Python scalar callback must not execute")
    class IntHook(int):
        def __int__(self):
            seen.append("int")
            raise AssertionError("Python scalar callback must not execute")
        def __index__(self):
            seen.append("index")
            raise AssertionError("Python scalar callback must not execute")
    with ops.Session() as s:
        x = s.tensor(np.ones((1, 3, 8), np.float32))
        w = s.tensor(np.ones(8, np.float32))
        call, args, kwargs = s.rms_norm, (x, w), {"eps": FloatHook(.01)}
        if where == "theta":
            call, args, kwargs = s.rotary, (x, 4), {"theta": FloatHook(10000.)}
        elif where == "head_dim":
            call, args, kwargs = s.rotary, (x, IntHook(4)), {}
        elif where == "handle":
            call, args, kwargs = s._transformer_linear, (IntHook(x), w), {}
        audit = audit_native_call(call, *args, **kwargs)
        report = audit.report()
        assert isinstance(audit.exception, TypeError), report
        assert not seen
        assert not any(report[k] for k in ("python_frames", "wrapper_work", "forbidden_native_calls", "unobserved_existing_threads")), report


@pytest.mark.parametrize("order", ["permute_swap", "swap_permute"])
def test_mixed_permutation_composition_rank_and_alias_guards(order):
    a = np.arange(48, dtype=np.float32).reshape(1, 3, 2, 8)
    with ops.Session() as s:
        x = s.tensor(a)
        if order == "permute_swap":
            intermediate = s.permute(x, (0, 2, 1, 3))
            result = s.transpose(intermediate, -1, -2)
        else:
            intermediate = s.transpose(x, -1, -2)
            result = s.permute(intermediate, (0, 2, 1, 3))
        p = s.capture(x, [intermediate, result])
        before = native(p.run, a)
        after = native(p.run, a, optimize=True)
        for actual, expected in zip(after, before):
            assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
        counts = {r["name"]: r["rewrites"] for r in native(p.inspect, optimize=True)["pass_report"]["passes"]}
        assert counts["permutation_composition"] == 1
        assert counts["dead_intermediates"] == 0, "returned intermediate remains live"
        for optimize in (False, True):
            with pytest.raises(ValueError, match="rank|axes"):
                p.run_dynamic(a.reshape(3, 2, 8), optimize=optimize)
        native(p.run, a, optimize=True)


def test_complete_transformer_composition_and_checkpoint():
    outputs = verify.transformer_program_composition()
    assert len(outputs) == verify.OUTPUT_COUNTS["transformer_program_composition"]
