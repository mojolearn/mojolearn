"""Saved 'more' board recipes with fit and inference boundaries separated.

NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
This is harness glue: estimator construction/settings remain in the saved
factory, and every fit/inference calculation remains in its Mojo binding.
"""


def make_runner(module, workload, arrays, record):
    state = {}
    separate = workload["inference"] == "separate"
    variant = None
    if workload.get("adapter") == "classical_agglomerative":
        if workload["lane"] != "agglomerative":
            raise ValueError("C42 linkage variant must retain the saved agglomerative dataset recipe")
        variant = workload["estimator_settings"]
        if variant.get("linkage") == "single" and not variant.get("compute_distances"):
            raise ValueError("C42 needs the public x_cluster route; legacy hierarchy single-linkage is a different path")
        if not workload.get("fitted_output_attributes"):
            raise ValueError("C42 needs the full fitted linkage-tree output schema")
    inner = module._build_ours(workload["lane"], arrays, record, state,
                               separate_inference=separate, classical_variant=variant)
    attrs = workload.get("fitted_output_attributes")
    requests = workload.get("inference_operations")
    if separate and (not isinstance(attrs, list) or not isinstance(requests, list) or not requests):
        raise ValueError("Separate full inference needs explicit fitted attributes and output operations")
    outputs = {}

    class Runner:
        info = inner.info
        params = inner.params

        def call(self):
            outputs.clear()
            inner.call()
            inner.sync()
            if separate:
                if "est" not in state:
                    raise ValueError("Saved recipe has no retained estimator for separate inference")
                for attr in attrs:  # fixed fitted-output schema, never row arithmetic
                    outputs["model_" + attr] = getattr(state["est"], attr)
            else:
                outputs.update(inner.outputs())
                for attr in attrs or []:
                    outputs["model_" + attr] = getattr(state["est"], attr)

        def sync(self):
            inner.sync()

        def infer(self):
            if not separate:
                return False
            for request in requests:  # fixed public calls specified by the saved recipe
                method = request["method"]
                if method not in {"predict", "predict_proba", "decision_function", "transform",
                                  "forecast", "search", "kneighbors", "score_samples"}:
                    raise ValueError("Unsupported classical inference method")
                args = [arrays[key] for key in request.get("array_arguments", [])]
                args.extend(request.get("literal_arguments", []))
                value = getattr(state["est"], method)(*args, **request.get("kwargs", {}))
                names = request["outputs"]
                if len(names) == 1:
                    outputs[names[0]] = value
                else:
                    if not isinstance(value, tuple) or len(value) != len(names):
                        raise ValueError("Declared inference output schema differs")
                    outputs.update(zip(names, value))
            inner.sync()
            return True

        def outputs(self):
            return outputs

    return Runner()
