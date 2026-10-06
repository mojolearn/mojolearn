"""Explicit full saved-estimator variants needing additional input buffers.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
No weights, dataset dimensions or estimator settings are synthesized here.
"""


def make_runner(module, workload, arrays, record):
    if workload.get("adapter") != "classical_weighted_cv" or workload["lane"] != "logreg-cv":
        raise ValueError("Unknown classical full-workload variant")
    key = workload.get("fit_sample_weight_key")
    if not key or key not in arrays:
        raise ValueError("Weighted LogisticRegressionCV needs a saved full-length weight artifact")
    if workload.get("inference") != "separate":
        raise ValueError("Weighted CV requires separate full-query inference")
    attrs = workload.get("fitted_output_attributes")
    if not isinstance(attrs, list) or not attrs:
        raise ValueError("CV selected parameters/scores/fitted coefficient output schema pending")
    make, name, params = module._est_factory("logreg-cv", "ours", arrays)
    probe = make()
    info = module._ours_info("logreg-cv", probe)
    info.update(adapter="classical_weighted_cv", source_status=__doc__,
                config=f"{name}({params!r}) with saved full sample_weight buffer")
    state = {}
    outputs = {}

    def fit():
        outputs.clear()
        est = state["est"] = make()
        est.fit(arrays["X"], arrays["y"], sample_weight=arrays[key])
        for attr in attrs:  # named fitted outputs only; no processing over data
            outputs["model_" + attr] = getattr(est, attr)

    def infer():
        outputs["predictions"] = state["est"].predict(arrays["Xq"])
        outputs["probabilities"] = state["est"].predict_proba(arrays["Xq"])

    return module.Runner(info, fit, lambda: outputs, infer=infer, record=probe)
