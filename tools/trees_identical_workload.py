"""Whole-operation extension of forest_speed_arm; opt in through its arguments.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
Only benchmark orchestration and buffer consumption live here. Estimator runtime
work remains in the public Mojo bindings. Nothing in this file runs on import.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import trees_identical_ideas as ideas


def add_arguments(parser):
    parser.add_argument("--trees-experiment", help="Txx[:subarm][@A|@B], C45–C51, or X01–X10; frozen binding required")
    parser.add_argument("--trees-arm", choices=("A", "B"))
    parser.add_argument("--trees-recipe")
    parser.add_argument("--trees-artifact", help="frozen source/compiler/binary/define manifest")
    parser.add_argument("--trees-recipe-facts", help="full saved recipe provenance; unknown/capped facts refuse")
    parser.add_argument("--trees-result", help="fresh result path; never a measured-board writer")


def file_hash(path):
    value = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def consume(value):
    """Encode already-consumed public buffers after the operation timer.

    Bytes are evidence transport, not estimator data computation. No list of
    row values, Python reductions, label processing, or runtime arithmetic.
    """
    if isinstance(value, tuple):
        return {"outputs": [consume(buffer) for buffer in value]}  # output buffers, not rows
    if hasattr(value, "tobytes"):
        raw = value.tobytes()
        return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw),
                "shape": list(getattr(value, "shape", ())), "dtype": str(getattr(value, "dtype", "unknown"))}
    if isinstance(value, (int, float)):
        return {"scalar": value if math.isfinite(value) else None,
                "nonfinite": None if math.isfinite(value) else repr(value)}
    raise ValueError("consumer must expose a native output buffer or scalar")


def loaded_provenance(artifact):
    """Hash this worker's actually imported extensions, never a controller's."""
    loaded = []
    for name, module in tuple(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name.startswith("mojolearn") and path and path.endswith((".so", ".dylib", ".pyd")):
            path = str(Path(path).resolve())
            actual = file_hash(path)
            if artifact["files"].get(path) != actual:
                raise ValueError("loaded native extension absent from frozen artifact: " + path)
            loaded.append({"module": name, "path": path, "sha256": actual})
    if not loaded:
        raise ValueError("no native tree binding was observed in this worker")
    return loaded


def prepare_request(args, spec):
    """Before dataset loading/imports; this is future execution admission only."""
    required = ("trees_arm", "trees_recipe", "trees_artifact", "trees_recipe_facts", "trees_result")
    if any(not getattr(args, key) for key in required):
        raise ValueError("tree experiment requires arm, recipe, artifact, full recipe facts and fresh result")
    if args.rows is not None or spec.size_tag() != "shipped" or spec.rounds() != 1:
        raise ValueError("tree experiments require saved full workload, one excluded warmup and one scored sample")
    if any((args.opponents_only, args.ours_ab, args.with_opponents, args.arms,
            args.opponents_first, args.host_digest, args.ours_cpu, args.infer,
            args.params_only, args.list_arms, args.save_scored_models, args.mem)):
        raise ValueError("tree experiment is its own full-operation route; legacy modes cannot be mixed")
    if os.environ.get("MOJOLEARN_NUMERIC_MODE") != "identical":
        raise ValueError("tree experiment requires IDENTICAL")
    if os.environ.get("MOJOLEARN_SPEED_FORTRAN", "0") not in ("", "0"):
        raise ValueError("tree experiment preparation belongs inside the timer")
    destination = Path(args.trees_result)
    if destination.exists():
        raise ValueError("preserve previous evidence: result path must be fresh")
    plan = ideas.make_plan(args.trees_experiment, args.trees_arm, args.trees_recipe,
                           args.trees_artifact, args.trees_recipe_facts, args.trees_result)
    recipe = plan["recipe"]
    if recipe.get("source_blockers"):
        raise ValueError("recipe source prerequisites remain pending: " + "; ".join(recipe["source_blockers"]))
    if recipe["lane"] != args.lane or recipe["dataset"] != args.dataset:
        raise ValueError("driver lane/dataset differ from selected saved recipe")
    controls, _, blockers = ideas.selected_controls(args.trees_experiment, args.trees_arm)
    if blockers or not controls:
        raise ValueError("unimplemented selection: " + "; ".join(blockers))
    # A combination may span several estimators. This cell must have at least
    # one intended route; other affected recipe cells stay separately pending.
    recipe_tags = set(recipe["groups"]) | {recipe["id"], recipe["lane"],
                  recipe["lane"] + ":" + recipe["dataset"], recipe.get("estimator_class", ""),
                  recipe["id"].split(":", 1)[0]}
    eligible = [c for c in controls if set(c.get("recipe_keys", [])).intersection(recipe_tags)]
    if not eligible:
        raise ValueError("selected candidates do not affect this recipe")
    artifact = ideas.read_json(args.trees_artifact)
    facts = ideas.read_json(args.trees_recipe_facts)
    for key in ("source_commit", "compiler", "hardware", "files", "defines", "source_hashes", "harness_hashes"):
        if key not in artifact or (key != "defines" and not artifact[key]):
            raise ValueError("missing artifact provenance: " + key)
    if artifact.get("numeric_mode") != "identical" or artifact.get("arm") != args.trees_arm:
        raise ValueError("artifact mode/arm mismatch")
    if artifact.get("selection") != args.trees_experiment:
        raise ValueError("artifact selection mismatch")
    if set(artifact["defines"]) != set(plan["defines"]):
        raise ValueError("artifact defines differ from selected A/B controls")
    for group in ("source_hashes", "harness_hashes"):
        for relative, expected in artifact[group].items():
            if file_hash(ideas.ROOT / relative) != expected:
                raise ValueError("frozen " + group + " differ: " + relative)
    for key in ("dataset_version", "dataset_files", "split", "dimensions", "settings",
                "intrinsic_caps", "full_coverage_basis", "prepared_hashes"):
        if key not in facts or facts[key] in (None, "", "PENDING"):
            raise ValueError("full recipe fact remains pending: " + key)
    if facts.get("recipe") != args.trees_recipe or facts.get("full_dataset") is not True:
        raise ValueError("recipe not attested as the full intended dataset")
    if facts.get("estimator_overrides", {}) != recipe.get("estimator_overrides", {}):
        raise ValueError("saved estimator overrides differ from the declared recipe extension")
    if facts.get("fit_options", {}) != recipe.get("fit_options", {}):
        raise ValueError("saved fit-output requests differ from the declared recipe")
    if facts.get("unresolved_caps") or facts.get("pending"):
        raise ValueError("missing/capped full-workload coverage remains pending")
    if not facts["dataset_files"]:
        raise ValueError("dataset files or saved synthetic recipe bytes must be pinned")
    for path, expected in facts["dataset_files"].items():
        if file_hash(path) != expected:
            raise ValueError("dataset bytes differ from saved recipe: " + path)
    if {"T14", "T26", "T34"}.issubset({c["id"] for c in controls}) and not facts.get("accepted_individual_numerical_evidence"):
        raise ValueError("X10 requires accepted T14/T26/T34 individual all-column evidence")
    return dict(plan=plan, recipe=recipe, artifact=artifact, facts=facts, destination=destination)


def confirm_loaded_recipe(request, data, cfg):
    """No `--rows full` inference: actual split buffers must match pinned facts."""
    facts = request["facts"]
    shape = {"train_rows": int(data.X_train.shape[0]), "test_rows": int(data.X_test.shape[0]),
             "features": int(data.X_train.shape[1]), "classes": int(data.n_classes)}
    if data.name != request["recipe"]["dataset"]:
        raise ValueError("loader substituted another dataset")
    if shape != facts["dimensions"] or cfg != facts["settings"]:
        raise ValueError("actual loaded dimensions/settings differ from saved full recipe")
    for name in ("X_train", "X_test", "y_train", "y_test"):
        if hashlib.sha256(getattr(data, name).tobytes()).hexdigest() != facts["prepared_hashes"].get(name):
            raise ValueError("prepared split digest mismatch: " + name)
    if data.task == "ranking":
        for name in ("qid_train", "qid_test"):
            if hashlib.sha256(getattr(data, name).tobytes()).hexdigest() != facts["prepared_hashes"].get(name):
                raise ValueError("query boundaries not pinned: " + name)
    return shape


def consumer(model, data, recipe):
    """Return a public output callable and retain any fit-owned explanation state."""
    kind = recipe["consumer"]
    if kind == "shap":
        from mojolearn import TreeExplainer
        # Full background construction is timed with the cold consumer.
        explainer = TreeExplainer(model, data=data._ours_X)
        return lambda: explainer.shap_values(data._ours_Xtest)
    if kind == "importance":
        return lambda: model.feature_importances_
    if kind in ("oob", "oob_importance"):
        attr = "oob_prediction_" if data.task == "regression" else "oob_decision_function_"
        if kind == "oob_importance":
            return lambda: (getattr(model, attr), model.oob_score_, model.feature_importances_)
        return lambda: (getattr(model, attr), model.oob_score_)
    if kind == "isolation":
        return lambda: model.score_samples(data._ours_Xtest)
    if kind == "isolation_decision":
        return lambda: model.decision_function(data._ours_Xtest)
    if kind == "isolation_predict":
        return lambda: model.predict(data._ours_Xtest)
    if kind == "apply":
        return lambda: model.apply(data._ours_Xtest)
    if kind == "staged":
        return lambda: model.staged_predict(data._ours_Xtest)
    if kind == "predict_apply":
        return lambda: model.predict_with_leaves(data._ours_Xtest)
    if kind == "labels":
        return lambda: model.predict(data._ours_Xtest)
    if data.task in ("binary", "multiclass"):
        return lambda: model.predict_proba(data._ours_Xtest)
    return lambda: model.predict(data._ours_Xtest)


def make_arm(request, args, cfg, data, driver):
    recipe = request["recipe"]
    public = recipe.get("estimator_class")
    if public is None:
        return driver.OUR_BUILDERS[args.lane](
            args.lane, cfg, data, extra=recipe.get("estimator_overrides", {}))
    # A saved full-data extension supplies every constructor knob; the forest
    # recipe is a data source, never permission to replace a DT with a forest
    # or to shorten DART/AdaBoost settings to fit a screening fixture.
    allowed = {"DecisionTreeClassifier", "DecisionTreeRegressor", "DARTClassifier",
               "DARTRegressor", "AdaBoostClassifier", "AdaBoostRegressor"}
    if public not in allowed:
        raise ValueError("unsupported tree recipe constructor")
    constructor = request["facts"].get("constructor")
    if not constructor or "random_state" not in constructor:
        raise ValueError("full extension constructor/settings/seed remain pending")
    if constructor.get("numeric_mode", "identical") != "identical":
        raise ValueError("extension constructor must preserve IDENTICAL")
    for key, expected in recipe.get("constructor_requirements", {}).items():
        if key not in constructor or constructor[key] != expected:
            raise ValueError("saved constructor cannot reach the declared recipe route: " + key)
    import mojolearn
    cls = getattr(mojolearn, public)
    return driver.spec.Arm("ours", lambda: cls(**constructor), driver._our_fit,
                           driver._our_score, sync=driver._our_sync, library="mojolearn")


def scored_metrics(raw, model, arm, data, recipe, driver):
    """Benchmark diagnostics from the already-consumed scored output.

    These use the existing benchmark metric functions after the timed region;
    they do not move any estimator runtime work into Python.
    """
    kind = recipe["consumer"]
    if kind in ("importance", "shap", "oob", "oob_importance", "apply", "isolation_predict"):
        return "separate task diagnostic; auxiliary exactness remains pending", [
            (name, value) for name, value, _ in arm.score(model, data)]
    value = raw[0] if kind == "predict_apply" else raw
    if kind == "staged":
        value = driver.np.asarray(value)[:, -1, :]
    if kind == "labels" and data.task in ("binary", "multiclass"):
        return "consumed scored labels", [("accuracy", driver.spec.accuracy(data.y_test, driver.np.asarray(value)))]
    if data.task == "binary":
        value = driver._p1(value)
    elif data.task == "multiclass":
        value = driver._mat(value)
    else:
        value = driver._vec(value)
    return "consumed scored output", driver.infer_metrics(recipe["lane"], data, value)


def run(request, args, data, cfg, driver):
    """One excluded warmup plus one scored complete operation; no promotion.

    The repeated inference uses the scored model and owned output. Dataset/file
    decoding is an explicit CPU-only input step outside runtime, and preparation
    of public buffers/model state stays inside every complete-operation timer.
    """
    record = dict(schema=1, selection=args.trees_experiment, arm=args.trees_arm,
                  recipe=args.trees_recipe, numeric_mode="identical", status="STARTED",
                  artifact=request["artifact"], facts=request["facts"], samples=[],
                  identity="NOT VERIFIED", quality="NOT VERIFIED", runtime_reach="NOT DEMONSTRATED",
                  admitted_measurement=False, board_update=None,
                  cold_scope="first consumer of each newly fitted model; process/binding import is outside clocks",
                  timed_boundary="public buffer preparation + constructor + fit + sync + requested output consumption")
    destination = request["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation preserves original partial failures and completed cells.
    with destination.open("x") as stream:
        stream.write(json.dumps(record, indent=2) + "\n")
    try:
        record["loaded_dimensions"] = confirm_loaded_recipe(request, data, cfg)
        arm = make_arm(request, args, cfg, data, driver)
        record["binding"] = driver.verify_our_arm(arm)
        record["loaded_before"] = loaded_provenance(request["artifact"])
        for phase in ("warmup_excluded", "scored"):
            start = time.perf_counter()
            driver.prepare_our_inputs(data)
            prepared = time.perf_counter()
            model = arm.make()
            fit_options = request["recipe"].get("fit_options", {})
            if fit_options:
                if args.lane != "rf" or fit_options != {"feature_importances": True}:
                    raise ValueError("unsupported requested native fit outputs")
                model.fit(data._ours_X, data._ours_y, **fit_options)
            else:
                arm.fit(model, data)
            arm.sync()
            fitted = time.perf_counter()
            output_call = consumer(model, data, request["recipe"])
            raw = output_call()
            arm.sync()
            finished = time.perf_counter()
            output = consume(raw)
            sample = dict(phase=phase, preparation_ms=(prepared-start)*1000,
                          fit_ms=(fitted-prepared)*1000,
                          cold_inference_ms=(finished-fitted)*1000,
                          whole_operation_ms=(finished-start)*1000, output=output)
            record["samples"].append(sample)
            if phase == "scored":
                repeated_start = time.perf_counter()
                repeated_raw = output_call()
                arm.sync()
                repeated_finished = time.perf_counter()
                sample["repeated_inference_ms"] = (repeated_finished-repeated_start)*1000
                repeated = consume(repeated_raw)
                sample["repeated_output"] = repeated
                # Read the earlier output after reuse: no extra model invocation.
                sample["retained_cold_output"] = consume(raw)
                # Quality diagnostics consume the scored output where the
                # requested consumer produces task scores, without another fit.
                # They are not accepted same-version identity evidence.
                record["quality_source"], metrics = scored_metrics(
                    raw, model, arm, data, request["recipe"], driver)
                record["quality_metrics"] = [{"metric": metric,
                    "value": float(value) if math.isfinite(float(value)) else None,
                    "nonfinite": None if math.isfinite(float(value)) else repr(value)}
                    for metric, value in metrics]
        record["loaded_after"] = loaded_provenance(request["artifact"])
        record["status"] = "CAPTURED_PENDING_IDENTITY_QUALITY_AND_REACH"
        record["resource_policy"] = {"arms_serial": True, "opponents": False,
            "thread_environment": {key: os.environ.get(key) for key in
                ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}}
        return 0
    except BaseException as exc:
        record["status"] = "FAILED"
        record["failure"] = type(exc).__name__ + ": " + str(exc)
        raise
    finally:
        destination.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")
