"""SVM score label equality, refusal order and live program-context seams."""
import numpy as np
import pytest

from mojolearn.svm import LinearSVC

__mojolearn_user_boundary__ = True


@pytest.fixture
def fitted():
    model = LinearSVC(numeric_mode="identical").fit(
        np.array([[-1], [1]] * 4, dtype=np.float32), [-5, 9] * 4)
    np.asarray(model._w)[:] = [1, 0]
    return model


def _query():
    return np.array([[-1], [1]], dtype=np.float32)


@pytest.mark.parametrize("label", [7, 10**80, "\U0001f642\ud800"])
def test_svc_score_duplicate_mutated_classes_compare_label_values(fitted, label):
    fitted.classes_ = [label, label]
    assert fitted.score(_query(), [label, label]) == 1.0


def test_svc_score_union_keeps_unknown_truth_as_miss(fitted):
    assert fitted.score(_query(), [-5, 999]) == 0.5


@pytest.mark.parametrize("bias,truth", [(-1, 7), (1, 10**80)])
def test_svc_score_selected_only_huge_integer_pack_behavior(fitted, bias, truth):
    fitted.classes_ = [7, 10**80]
    np.asarray(fitted._w)[:] = [0, bias]
    assert fitted.score(_query(), [truth, truth]) == 1.0


def test_svc_score_bool_labels_keep_integer_equality(fitted):
    fitted.classes_ = [False, True]
    assert fitted.score(_query(), [0, 1]) == 1.0


@pytest.mark.parametrize("truth,name", [([0.5, 0.5], "y_true"), ([0, 0], "y_pred")])
def test_svc_score_truth_type_refusal_precedes_decoded_float_prediction(fitted, truth, name):
    fitted.classes_ = [-5, 9.0]
    with pytest.raises(TypeError, match=name + " must contain only strings or only integers"):
        fitted.score(_query(), truth)


def test_svc_score_row_count_refusal_precedes_label_refusal(fitted):
    fitted.classes_ = [-5, 9.0]
    with pytest.raises(ValueError, match="y must be 1-D with one entry per row of X"):
        fitted.score(_query(), [0.5])


def test_svc_score_reads_rebound_live_metrics_program_context(fitted, monkeypatch):
    from mojolearn import _expansion_metrics

    monkeypatch.setattr(_expansion_metrics, "_PROG_CONTEXT", None)
    with pytest.raises(ImportError, match="metrics programs are not connected"):
        fitted.score(_query(), [-5, 9])


def test_svc_score_native_metrics_factory_refuses_python_callback(fitted, monkeypatch):
    from mojolearn import _expansion_metrics

    context = _expansion_metrics._PROG_CONTEXT
    entries = context["prog_bindings"][context["backend_fields"]["_DEFAULT_MODE"]]
    calls = []

    def forbidden():
        calls.append(True)
        raise AssertionError("Python metrics factory invoked")

    monkeypatch.setitem(entries, "metrics_program_native_api_v1", forbidden)
    with pytest.raises(TypeError, match="compiled extension entry"):
        fitted.score(_query(), [-5, 9])
    assert calls == []


@pytest.mark.parametrize("ranges,out_entry", [("1", True), ("0", True), ("0", False)])
def test_svc_score_preserves_count_transfer_switches(fitted, monkeypatch, ranges, out_entry):
    from mojolearn import _expansion_metrics

    monkeypatch.setenv("MOJOLEARN_ARENA_RANGES", ranges)
    if not out_entry:
        context = _expansion_metrics._PROG_CONTEXT
        entries = context["prog_bindings"][context["backend_fields"]["_DEFAULT_MODE"]]
        monkeypatch.setitem(entries, "x_metrics_run_out", None)
    assert fitted.score(_query(), [-5, 999]) == 0.5
