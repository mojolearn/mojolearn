"""Full saved-estimator workloads exercising C08/C09/C10/C12 consumers.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Written only. No synthetic/reduced data, worker jobs or candidate imports ran.
The caller resolves the saved expanded lane's dataset/version/hash/dimensions,
settings, caps and output schema before constructing either adapter. There is
no standalone saved metrics dataset recipe: these are declared report extensions
of the existing full estimator recipes, not fabricated metric board rows.
"""
from __future__ import annotations

ADAPTERS = {
    "classical_regression_report": "C09",
    "classical_ranking_report": "C10",
    "classical_ordinal_fit_transform": "C08",
    "classical_onehot_fit_transform": "C08",
    "classical_classification_report": "C12",
    "classical_lda_outputs": "C04+C56",
}
STATUS = "NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED"


def make_runner(module, workload, arrays, record):
    """Return the existing expanded Runner interface: fit/infer/outputs.

    Preparation belongs to full_workload.load_inputs and stays inside its
    whole-operation clock. fit includes estimator fit, full-fit-row prediction,
    report synchronization and model output retrieval. infer covers the complete
    saved query split separately. outputs only transfers already-consumed native
    results for the harness digest, whose cost remains inside the outer clock.
    """
    adapter = workload.get("adapter")
    if adapter not in ADAPTERS or workload.get("family") != "expanded":
        raise ValueError("C09/C10 need a named expanded saved-estimator adapter")
    if adapter in {"classical_ordinal_fit_transform", "classical_onehot_fit_transform"}:
        return ordinal_runner(module, workload, arrays, record)
    if adapter == "classical_lda_outputs":
        return lda_runner(module, workload, arrays, record)
    lane = workload["lane"]
    spec = module.LANES[lane]
    expected_task = "reg" if adapter == "classical_regression_report" else "clf"
    if spec.get("task") != expected_task or spec.get("kind") != "est":
        raise ValueError("The chosen saved lane is outside this report adapter")
    if spec.get("xlane") in {"trees", "cnn"}:
        raise ValueError("Classical IDENTICAL report workloads exclude trees/neural lanes")
    if any(name not in arrays for name in ("X", "y", "Xq", "yq")):
        raise ValueError("Full training and query arrays/targets are required; no substitute")
    report = workload.get("report_settings")
    if adapter == "classical_classification_report":
        if lane != "sgd-clf":
            raise ValueError("C12 adapter uses the saved full SGDClassifier recipe")
        required_settings = ("labels", "target_names", "zero_division", "kappa_weights",
                             "replace_undefined_by", "sample_weight_key", "query_sample_weight_key")
    elif expected_task == "reg":
        required_settings = ("multioutput", "force_finite", "sample_weight_key", "query_sample_weight_key")
    else:
        required_settings = ("pos_label", "include_curves", "sample_weight_key", "query_sample_weight_key",
                             "score_method", "score_column", "classes")
    if not isinstance(report, dict) or any(key not in report for key in required_settings):
        raise ValueError("Full metric report settings remain pending")
    if adapter == "classical_ranking_report" and report["classes"] != 2:
        raise ValueError("C10 binary report is integrated; multiclass report remains pending")
    if workload.get("inference") != "separate":
        raise ValueError("Report adapters require separately recorded inference")
    attributes = workload.get("fitted_output_attributes")
    if not isinstance(attributes, list):
        raise ValueError("The estimator's admitted fitted-output schema remains pending")
    for key in (report["sample_weight_key"], report["query_sample_weight_key"]):
        if key is not None and key not in arrays:
            raise ValueError("A requested full sample-weight array is missing")
    make, class_name, params = module._est_factory(lane, "ours", arrays)
    state = {"outputs": {}}
    info = module._ours_info(lane)
    info.update(config=f"{class_name}({params!r}) + {adapter}({report!r})",
                adapter=adapter, candidate=ADAPTERS[adapter], mode="identical",
                report_status=STATUS, recipe_scope="saved estimator plus explicitly requested full report",
                preparation="saved expanded lane_arrays; all declared rows; cap audit required",
                fit_boundary="fit + predict/score all fit rows + native report + synchronized consumed fitted outputs",
                inference_boundary="predict/score all saved query rows + native report + synchronized consumed outputs")

    def f32_buffer(value, name):
        # API transport: select a native float32 buffer, no row arithmetic.
        from mojolearn._array import Array
        from mojolearn._buffer import as_f32_c
        if isinstance(value, Array) and value.dtype != "<f4":
            value = value.astype("<f4")
        return as_f32_c(value, ndim=None, name=name)[0]

    def produce(prefix, X, y, weight_key):
        from mojolearn import metrics
        est = state["est"]
        weights = None if weight_key is None else arrays[weight_key]
        prediction = est.predict(X)
        state["outputs"][prefix + "predictions"] = prediction
        if expected_task == "reg":
            answer = metrics.regression_report(
                f32_buffer(y, "target"), f32_buffer(prediction, "prediction"),
                sample_weight=weights, multioutput=report["multioutput"],
                force_finite=report["force_finite"], numeric_mode="identical")
        elif adapter == "classical_classification_report":
            # Public reports on every saved row. Kappa's occupied class-pair
            # table goes through the actual x_metrics group_sort planner.
            # Its scratch-cost predicate can keep B on low-cardinality data;
            # source reach is not evidence of candidate branch activation.
            table = metrics.classification_report(
                y, prediction, labels=report["labels"], target_names=report["target_names"],
                sample_weight=weights, zero_division=report["zero_division"],
                output_dict=True, numeric_mode="identical")
            answer = {}
            for label, row in table.items():  # glue: named class/aggregate report fields
                if isinstance(row, dict):
                    for metric, value in row.items():  # glue: fixed report fields
                        answer[str(label) + "_" + metric] = value
                else:
                    answer[str(label)] = row
            answer["cohen_kappa"] = metrics.cohen_kappa_score(
                y, prediction, labels=report["labels"], weights=report["kappa_weights"],
                sample_weight=weights, replace_undefined_by=report["replace_undefined_by"],
                numeric_mode="identical")
        else:
            method = report["score_method"]
            if method not in {"decision_function", "predict_proba"}:
                raise ValueError("C10 score_method must be an explicit estimator score method")
            scores = getattr(est, method)(X)
            if report["score_column"] is not None:
                scores = scores[:, report["score_column"]]
            answer = metrics.ranking_report(
                y, f32_buffer(scores, "scores"), pos_label=report["pos_label"],
                sample_weight=weights, include_curves=report["include_curves"],
                numeric_mode="identical")
            state["outputs"][prefix + "scores"] = scores
        for key, value in answer.items():  # glue: fixed named report outputs, no row work
            if isinstance(value, tuple):
                for component, array in enumerate(value):  # glue: three named curve arrays
                    state["outputs"][prefix + key + "_" + str(component)] = array
            else:
                state["outputs"][prefix + key] = value

    def fit():
        state["outputs"] = {}
        est = state["est"] = make()
        est.fit(arrays["X"], arrays["y"])
        produce("fit_", arrays["X"], arrays["y"], report["sample_weight_key"])
        for name in attributes:  # glue: explicitly admitted fitted attributes
            state["outputs"]["model_" + name] = getattr(est, name)

    def infer():
        produce("inference_", arrays["Xq"], arrays["yq"], report["query_sample_weight_key"])

    def outputs():
        return state["outputs"]

    # Public Mojo calls return consumed host buffers and synchronize at their
    # binding boundary. The outer Runner's timer includes these completions.
    return module.Runner(info, fit, outputs, infer=infer,
                         record={"__library__": "mojolearn", "class": class_name,
                                 "params": params, "report_settings": report})



def ordinal_runner(module, workload, arrays, record):
    """C08 saved ordinal recipe with its full fit_transform operation declared.

    The saved ordinary runner's separate fit/transform cannot exercise fit-local
    inverse ownership. This explicit boundary includes all fit rows and returns
    the full transformed matrix in A and B; inference remains a separate call.
    """
    lane = workload["lane"]
    expected = "onehot" if workload["adapter"] == "classical_onehot_fit_transform" else "ordinal"
    if lane != expected or workload.get("inference") != "separate":
        raise ValueError("C08 needs the corresponding saved encoder lane and separate inference")
    make, name, params = module._est_factory(lane, "ours", arrays)
    state = {"outputs": {}}
    info = module._ours_info(lane)
    info.update(config=f"{name}({params!r}).fit_transform(full X)",
                adapter=workload["adapter"], report_status=STATUS,
                fit_boundary="preparation + fit_transform all fit rows + synchronized dense output/categories",
                inference_boundary="transform all saved query rows + synchronized dense output")

    def fit():
        est = state["est"] = make()
        result = est.fit_transform(arrays["X"])
        state["outputs"] = {"fit_transform": result}
        for column, categories in enumerate(est.categories_):  # glue: per-feature output owners
            state["outputs"]["categories_" + str(column)] = categories

    def infer():
        state["outputs"]["transform"] = state["est"].transform(arrays["Xq"])

    def outputs():
        return state["outputs"]

    return module.Runner(info, fit, outputs, infer=infer,
                         record={"__library__": "mojolearn", "class": name, "params": params})


def lda_runner(module, workload, arrays, record):
    """Saved lda-clf recipe plus explicitly requested full decision/transform.

    The ordinary saved classifier runner consumes predict/proba only. C04/C56
    require these additional public outputs to be requested in both A and B.
    """
    if workload["lane"] != "lda-clf" or workload.get("inference") != "separate":
        raise ValueError("LDA candidate needs the saved lda-clf recipe and separate inference")
    methods = workload.get("output_methods")
    if methods != ["decision_function", "transform"]:
        raise ValueError("LDA decision_function and transform output contract remains pending")
    attributes = workload.get("fitted_output_attributes")
    if not isinstance(attributes, list) or any(k not in arrays for k in ("X", "y", "Xq")):
        raise ValueError("LDA full arrays or fitted output schema remain pending")
    make, name, params = module._est_factory("lda-clf", "ours", arrays)
    state = {"outputs": {}}
    info = module._ours_info("lda-clf")
    info.update(adapter="classical_lda_outputs", report_status=STATUS,
                config=f"{name}({params!r}) plus full decision_function/transform",
                fit_boundary="fit full X/y + full fit-row decisions/projections + fitted state + synchronization",
                inference_boundary="full saved query decisions/projections + synchronization")

    def produce(prefix, X):
        for method in methods:  # glue: two requested native public operations
            state["outputs"][prefix + method] = getattr(state["est"], method)(X)

    def fit():
        state["outputs"] = {}
        est = state["est"] = make()
        est.fit(arrays["X"], arrays["y"])
        produce("fit_", arrays["X"])
        for name in attributes:  # glue: explicitly named fitted output attributes
            state["outputs"]["model_" + name] = getattr(est, name)

    def infer():
        produce("inference_", arrays["Xq"])

    def outputs():
        return state["outputs"]

    return module.Runner(info, fit, outputs, infer=infer,
                         record={"__library__": "mojolearn", "class": name, "params": params})
