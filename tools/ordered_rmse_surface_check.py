#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Installed native OrderedRMSE gate; executes directly or via runpy as __main__.

No test framework or reference library required. Run serially with CPU thread
limits set by the controller. JSON contains all model bytes and prediction
bits so independent GPU runs can be compared without reconstructing results.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expect_refusal(call, exception, label):
    try:
        call()
    except exception:
        return label
    raise AssertionError(f"{label}: unsupported input was accepted")


def prepare_restore_default(mode, selected_modes, installed_modes, set_mode, read_mode):
    """Select only a requested, physically installed tier for the contrast.

    Missing files are coverage information, not a caught load failure. When an
    alternate tier is selected and present its actual load/readback must work;
    errors propagate instead of silently downgrading the check.
    """
    known = {"identical", "fast"}
    require(mode in known, "unknown saved numeric mode")
    for label, modes in (("selected", selected_modes), ("installed", installed_modes)):
        require(bool(modes) and len(modes) == len(set(modes)) and set(modes) <= known,
                f"invalid {label} numeric modes")
        require(mode in modes, f"saved mode is not {label}")
    alternate = "fast" if mode == "identical" else "identical"
    if alternate not in installed_modes:
        reason = "alternate_tier_not_installed"
    elif alternate not in selected_modes:
        reason = "alternate_tier_not_selected"
    else:
        reason = None
    before_load = mode if reason else alternate
    set_mode(before_load)
    require(read_mode() == before_load, "process default mode readback differs before load")
    return {"selected_modes": list(selected_modes), "installed_gbdt_tiers": list(installed_modes),
            "alternate_mode": alternate, "process_default_before_load": before_load,
            "alternate_default_contrast": {"status": "NOT_EXERCISED" if reason else "PASS",
                                           "reason": reason}}


def main():
    import numpy as np
    from mojolearn import OrderedRMSE, set_numeric_mode, numeric_mode, _backend

    mode = os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical").lower()
    require(mode in ("fast", "identical"), "unknown numeric mode")
    set_numeric_mode(mode)
    # Respect the release's requested tiers. Direct invocation defaults to this
    # run's mode, so an identical-only check never opts into fast implicitly.
    selected_modes = os.environ.get("MOJOLEARN_RELEASE_MODES", mode).split()
    installed_modes = [tier for tier in ("identical", "fast")
                       if _backend._merged.binding_exists(_backend.tier_dir(tier), "_mojolearn_gbdt")]
    options = dict(n_estimators=3, max_depth=2, border_count=7,
                   learning_rate=0.2, l2_leaf_reg=1.0, numeric_mode=mode)
    # Dyadic features/targets/weights avoid fixture construction differences.
    rows = np.arange(32, dtype=np.int32)
    X = np.asfortranarray(np.column_stack((
        (rows % 8).astype(np.float32) * np.float32(0.5) - np.float32(1.75),
        ((rows * 5) % 11).astype(np.float32) * np.float32(0.25),
        ((rows * 3) % 7).astype(np.float32) * np.float32(0.125),
    )))
    y = (X[:, 0] * np.float32(2) - X[:, 1] * np.float32(0.5)
         + ((rows % 3) - 1).astype(np.float32) * np.float32(0.125))
    weights = ((rows % 4) + 1).astype(np.float32) * np.float32(0.5)
    weights[::7] = 0
    permutation = ((rows * 13 + 7) % 32).astype(np.uint32)
    query = np.asfortranarray(X[[1, 4, 9, 12, 17, 22, 28, 31]].copy())
    query[:, 0] += np.float32(0.125)

    first = OrderedRMSE(**options)
    binding = first._bind("_mojolearn_gbdt")
    actual_code = int(binding.gbdt_numeric_mode())
    actual_vendor = str(binding.gbdt_vendor())
    require(actual_code == {"fast": 0, "identical": 1}[mode],
            "native numeric-mode readback does not match requested tier")
    require(actual_vendor in ("metal", "hip", "cuda"), "native vendor readback missing")
    print(f"ORDERED_PYTHON_NATIVE {mode} {actual_vendor}", flush=True)

    refusals = [
        expect_refusal(lambda: OrderedRMSE(max_depth=9), ValueError, "depth"),
        expect_refusal(lambda: OrderedRMSE(loss="Logloss"), TypeError, "objective"),
        expect_refusal(lambda: OrderedRMSE(cat_features=[0]), TypeError, "categorical"),
        expect_refusal(lambda: first.fit(X, y, permutation=np.zeros(32, np.uint32)),
                       ValueError, "duplicate_permutation"),
        expect_refusal(lambda: first.fit(X, y, permutation=permutation,
                                         sample_weight=np.zeros(32, np.float32)),
                       ValueError, "zero_weight_mass"),
        expect_refusal(lambda: first.fit(X, y, permutation=permutation,
                                         eval_set=(X, y)), TypeError, "eval_set"),
    ]
    first.fit(X, y, permutation=permutation, sample_weight=weights)
    model_text = str(first.model_)
    require("trees 3\n" in model_text, "wrong exported tree count")
    # Estimators return mojolearn.Array (no NumPy at runtime); this gate
    # compares bits, so view every prediction through NumPy once.
    prediction = np.asarray(first.predict(X))
    query_prediction = np.asarray(first.predict(query))
    require(prediction.dtype == np.float32 and prediction.shape == (32,),
            "prediction dtype/shape mismatch")
    require(np.isfinite(prediction).all() and np.isfinite(query_prediction).all(),
            "non-finite predictions")
    require(np.ptp(prediction) > 0, "training returned a constant placeholder")
    require(np.mean((prediction.astype(np.float64) - y)**2)
            < np.mean(y.astype(np.float64)**2), "fit did not improve zero-bias RMSE")
    require(first.best_iteration_ is None and first.loss_curve_ is None,
            "ordered estimator invented tracking attributes")

    # Native fit returns fully-owned text after its context is synchronized
    # and closed; repeat prediction opens separate native contexts. Temporary
    # arrays can therefore leave scope without surviving GPU work borrowing
    # them. This gate deliberately uses two fits plus a deserialized model.
    second = OrderedRMSE(**options).fit(
        X.copy(order="C"), y.copy(), permutation=permutation.copy(),
        sample_weight=weights.copy(),
    )
    repeat_prediction = np.asarray(second.predict(X))
    repeat_exact = (model_text == str(second.model_)
                    and np.array_equal(prediction.view(np.uint32),
                                       repeat_prediction.view(np.uint32)))
    if mode == "identical":
        require(repeat_exact, "pinned mode repeat fit/model prediction bits differ")
    with tempfile.TemporaryDirectory(prefix="mojolearn-ordered-") as directory:
        path = Path(directory) / "ordered.npz"
        first.save(path)
        restoration = prepare_restore_default(
            mode, selected_modes, installed_modes, set_numeric_mode, numeric_mode)
        restored = OrderedRMSE.load(path)
        require(restored.numeric_mode == mode, "saved numeric mode was not restored")
        require(numeric_mode() == restoration["process_default_before_load"],
                "loading the saved model changed the process default")
        restored_binding = restored._bind("_mojolearn_gbdt")
        restored_code = int(restored_binding.gbdt_numeric_mode())
        restored_vendor = str(restored_binding.gbdt_vendor())
        require(restored_code == actual_code and restored_vendor == actual_vendor,
                "restored model native mode/vendor differs from saved model")
        restored_prediction = np.asarray(restored.predict(X))
        restored_query = np.asarray(restored.predict(query))
        require(str(restored.model_) == model_text, "serialized model text changed")
        require(np.array_equal(prediction.view(np.uint32), restored_prediction.view(np.uint32)),
                "save/load prediction bits differ")
        require(np.array_equal(query_prediction.view(np.uint32), restored_query.view(np.uint32)),
                "save/load unseen-row prediction bits differ")

    # The pass below covers save/load and exact predictions in the saved mode.
    # An unavailable/unselected alternate default is separately NOT_EXERCISED;
    # it is never reported as a successful two-tier contrast.
    restoration.update(saved_mode=mode, restored_mode=restored.numeric_mode,
                       restored_native_mode_code=restored_code,
                       restored_native_vendor=restored_vendor,
                       model_text_equal=True, training_predictions_equal=True,
                       query_predictions_equal=True)
    set_numeric_mode(mode)
    require(numeric_mode() == mode, "process default was not restored after the fixture")
    evidence = {
        "schema": "mojolearn.ordered_python.v1", "status": "PASS",
        "requested_mode": mode, "native_mode_code": actual_code,
        "native_vendor": actual_vendor, "rows": 32, "query_rows": 8,
        "trees": 3, "repeat_exact": bool(repeat_exact),
        "serialization": restoration,
        "refusals": refusals, "model_text": model_text,
        "model_sha256": hashlib.sha256(model_text.encode()).hexdigest(),
        "prediction_bits": prediction.view(np.uint32).tolist(),
        "query_prediction_bits": query_prediction.view(np.uint32).tolist(),
        "repeat_prediction_bits": repeat_prediction.view(np.uint32).tolist(),
    }
    print("ORDERED_PYTHON_JSON " + json.dumps(evidence, sort_keys=True), flush=True)
    print("ORDERED PYTHON SURFACE PASS", flush=True)


if __name__ == "__main__":
    main()
