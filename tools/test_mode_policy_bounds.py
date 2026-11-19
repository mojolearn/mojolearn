"""tools/mode_policy_bounds.py and the gates that read it (no wheel, no GPU)."""
import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import board_quality as Q  # noqa: E402
import mode_policy_bounds as P  # noqa: E402

ATOL, RTOL = 1e-6, 1e-5


def _package(entry=None, backend=True):
    binding = types.SimpleNamespace()
    if entry is not None:
        setattr(binding, P.MASK_ENTRY, entry)
    ml = types.SimpleNamespace(__name__="fake_mojolearn")
    if backend:
        ml._backend = types.SimpleNamespace(binding=lambda name: binding)
    return ml


def test_mask_is_read_through_the_binding_export():
    assert P.wheel_policy_mask(_package(lambda: 0)) == 0
    assert P.wheel_policy_mask(_package(lambda: 16)) == 16


def test_a_wheel_without_the_entry_is_mask_zero():
    assert P.wheel_policy_mask(_package(None)) == 0
    assert P.wheel_policy_mask(_package(lambda: 1 / 0)) == 0
    assert P.wheel_policy_mask(_package(lambda: 16, backend=False)) == 0  # binding not loaded, no selector


def test_rows_follow_the_table_of_the_policy_module():
    table = (Path(__file__).resolve().parent.parent / "checks" / "mode_policy.mojo").read_text()
    for bit, row in enumerate(P.POLICY_ROWS):
        assert f"comptime POLICY_ROW_{row.upper()} = {bit}\n" in table, row
    assert P.MATRIX_UNITS_BIT == 16 and P.relaxed_rows(16 + 1) == ["fold_free", "matrix_units"]


def test_mask_zero_is_the_fp32_bound_unchanged():
    ref = np.array([[1.0, -2.0], [0.0, 1e3]])
    for mask in (0, None, 1, 2, 4, 8, 32, 64, 79):       # every row but matrix units: nothing widens
        assert np.array_equal(P.forward_reference_bound(np, ref, mask, ATOL, RTOL), ATOL + RTOL * np.abs(ref))
        assert P.forward_reference_rms_passed(np, ref * 0, ref, mask) == (True, None)
        assert P.reference_margin("grad_max_rel_diff_vs_ours", 1e-4, mask) == 1e-4


def test_matrix_units_bound_is_the_derived_one():
    assert P.MATRIX_UNITS_CELL_REL == 2 * 2.0 ** -10 + 2.0 ** -20
    assert P.MATRIX_UNITS_RMS_REL == 1.25 * 2.0 ** -10
    assert P.matrix_units_cell_bound(8.0, fp32_allowance=1e-6) == P.MATRIX_UNITS_CELL_REL * 8.0 + 1e-6
    ref = np.array([1.0, -4.0, 0.5])
    bound = P.forward_reference_bound(np, ref, 16, ATOL, RTOL)
    assert np.allclose(bound, ATOL + RTOL * np.abs(ref) + P.MATRIX_UNITS_CELL_REL * 4.0, rtol=0, atol=1e-18)
    assert P.reference_margin("grad_max_rel_diff_vs_ours", 1e-4, 16) == 1e-4 + P.MATRIX_UNITS_CELL_REL
    assert P.reference_margin("norm_rel_diff_vs_ours", 1e-4, 16) == 1e-4   # not a product: not widened


def test_rms_test_passes_truncation_and_fails_a_precision_collapse():
    rng = np.random.default_rng(0)
    ref = rng.standard_normal(4096)
    ok, ratio = P.forward_reference_rms_passed(np, ref * 1.08 * 2.0 ** -10, ref, 16)
    assert ok and abs(ratio - 1.08 * 2.0 ** -10) < 1e-12
    bad, ratio = P.forward_reference_rms_passed(np, ref * 1.6 * 2.0 ** -10, ref, 16)
    assert not bad and ratio > P.MATRIX_UNITS_RMS_REL


def _pair(ours_quality, grad, **cell):
    return Q.compare(dict(arm="ours", quality=ours_quality, **cell),
                     dict(arm="torch-eager-fp32", quality={"grad_max_rel_diff_vs_ours": grad}))


def _grad_check(result):
    return [c for c in result["checks"] if c["metric"] == "grad_max_rel_diff_vs_ours"][0]


def test_grad_gate_is_unchanged_without_a_mask():
    check = _grad_check(_pair({}, 5e-4))
    assert check["margin"] == 1e-4 and check["verdict"] == "FAIL"
    assert _grad_check(_pair({}, 5e-5))["verdict"] == "PASS"
    assert _grad_check(_pair({"mode_policy_mask": 0}, 5e-4))["verdict"] == "FAIL"
    assert _grad_check(_pair({"mode_policy_mask": 4}, 5e-4))["verdict"] == "FAIL"    # a row with no bound


def test_grad_gate_takes_the_matrix_units_bound_from_our_cell():
    for result in (_pair({"mode_policy_mask": 16}, 5e-4), _pair({}, 5e-4, mode_policy_mask=16)):
        check = _grad_check(result)
        assert check["margin"] == 1e-4 + P.MATRIX_UNITS_CELL_REL and check["verdict"] == "PASS"
    assert _grad_check(_pair({"mode_policy_mask": 16}, 5e-3))["verdict"] == "FAIL"
    assert Q.policy_mask({"quality": {"mode_policy_mask": True}}) == 0
