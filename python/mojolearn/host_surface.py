# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE CPU SURFACE OF MOJOLEARN, DECLARED ONCE (the host surface manifest
lane, 2026-09-14).

Before this file the CPU surface had no single statement in code. The
covered training lanes lived in the CPU identity gate's YAML env, the
routing table in `_backend._HOST_MODULES`, the inference lanes in two gate
tools, the recording directories in the workflow, and the README restated
all of it by hand, which is how it came to say k-NN had no CPU path the day
after the k-NN host lane merged. This file is the one source; everything
else READS it:

  `_backend._HOST_MODULES`           is `routed_modules()`
  .github/workflows/cpu-identity-gate.yml
                                     fills COVERED_LANES, HOST_FAMILIES,
                                     HOST_BINDINGS and CLASSICAL_RECORDED
                                     from the command line below, in one step
  tools/identity_break.py            records `host.surface` beside
                                     `host.families` in every CPU column
  tools/docs_facts.py                checks the marked spans of README.md,
                                     SUPPORT_MATRIX.md and
                                     docs/BYTE_LM_CPU_TRAINING.md against it
  python/mojolearn/tests/test_host_surface.py
                                     fails when a binding exports a name
                                     this file does not list, when this file
                                     lists a family with no binding source,
                                     when a build shim does not exec
                                     bindings/build_host_family.sh, or when a lane
                                     named here is unknown to the gate that
                                     is supposed to run it

Per host family the manifest declares: the binding basename under
mojolearn/host/, the build shim, the GPU family it routes on a CPU-only
install (None for the two bindings loaded by path), the sabotage define its
gate's negative control passes, the identity_break lanes it covers for
TRAINING (the CPU column must read STABLE and IDENTICAL x4 on them), the
lanes and public classes it serves for INFERENCE from a saved model, the
Mojo host modules that ship inside it, the function names it exports, and
whether it ships in a wheel.

This file imports nothing from the package on purpose. It runs by path
before the package can import (the gate runner has no binding built yet):

    python3 python/mojolearn/host_surface.py --covered-lanes
    python3 python/mojolearn/host_surface.py --routed-families
    python3 python/mojolearn/host_surface.py --bindings --sep ,
    python3 python/mojolearn/host_surface.py --classical-recorded
    python3 python/mojolearn/host_surface.py --markdown
    python3 python/mojolearn/host_surface.py --json

and `python3 -m mojolearn.host_surface ...` says the same thing on a box
where the package imports.
"""
import argparse
import json
import sys

#: Where this manifest lives, recorded into every CPU column.
SOURCE = "python/mojolearn/host_surface.py"

#: The one builder every family compiles through; bindings/build_<family>_host.sh
#: is a two-line shim that execs it with the family name.
BUILDER = "bindings/build_host_family.sh"

#: The GPU columns the TRAINING gate diffs the CPU column against
#: (cpu-identity-gate.yml, --require-columns 4 on the covered lanes).
TRAINING_GPU_COLUMNS = (
    "bench/results/identity_break/2026-09-13_three-columns/apple-m4.json",
    "bench/results/identity_break/2026-09-13_three-columns/nvidia-h100-sm_90a.json",
    "bench/results/identity_break/2026-09-13_three-columns/amd-mi325x-gfx942.json",
)

#: The GPU columns the classical INFERENCE gate compares each host identity
#: hash against (tools/classical_host_gate.py check --gpu-column).
CLASSICAL_GPU_COLUMNS = (
    "bench/results/identity_break/2026-09-14_46-lanes/apple-m4.json",
    "bench/results/identity_break/2026-09-14_46-lanes/nvidia-h100-sm_90a.json",
    "bench/results/identity_break/2026-09-14_46-lanes/amd-mi300x-gfx942.json",
)

#: The classical inference recordings (one directory per GPU box and lane
#: set; every fixture under each must be RECORDED or the gate exits 2).
CLASSICAL_RECORDED = (
    "bench/results/classical_host/2026-09-13-apple-m4",
    "bench/results/classical_host/2026-09-14-nvidia-h100",
    "bench/results/classical_host/2026-09-14-amd-mi300x",
    "bench/results/classical_host/2026-09-14-apple-m4-kde-svc",
    "bench/results/classical_host/2026-09-14-apple-m4-knn",
)

#: The forest inference recordings: every directory under this root whose
#: expected.json says RECORDED (the workflows sort them at run time).
FOREST_RECORDED_ROOT = "bench/results/forest_host"

#: The identity_break lanes with a CPU TRAINING path, in the gate's order,
#: with the name the docs use for each.
TRAINING_LANE_NAMES = {
    "gemm-pinned": "pinned GEMM",
    "kde": "kernel density",
    "holtwinters": "Holt-Winters",
    "lasso": "lasso",
    "elasticnet": "elasticnet",
    "svc": "SVC",
}

#: The lanes with NO CPU path of any kind, as the README states them. A
#: lane leaves this list the day its host lane merges; docs_facts fails the
#: README until the marked span is rewritten.
NO_CPU_PATH = (
    "k-means",
    "DBSCAN",
    "isolation forest",
    "agglomerative and spectral clustering",
    "UMAP",
    "the Gaussian process",
    "ARIMA",
    "the neural blocks",
    "training for the forests, gradient boosting and k-NN",
)

#: The read-back trio every host binding exports under its own prefix,
#: plus the sabotage flag: `<prefix>_numeric_mode()` must answer 1,
#: `<prefix>_vendor()` "cpu", `<prefix>_column()` "cpu" (the kernel
#: matrix's CPU column, asserted at build time), `<prefix>_sabotage()` False
#: outside the gate.
READBACK = ("numeric_mode", "vendor", "column", "sabotage")

FAMILIES = (
    dict(
        family="byte_lm",
        binding="_mojolearn_byte_lm_host",
        routes=None,
        loaded_by="python/mojolearn/_byte_lm_host.py",
        sabotage_define="MOJOLEARN_BYTE_LM_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=(),
        forest_kinds=(),
        classes=("LanguageModelInference", "LanguageModelHostTrainer"),
        display="the byte LM forward pass and one training step",
        host_modules=(
            "training/byte_lm_host.mojo",
            "training/byte_lm_host_backward.mojo",
            "training/byte_lm_host_kernels.mojo",
            "gemm/host/gemm_oracle.mojo",
        ),
        exports=(
            "byte_lm_host_numeric_mode", "byte_lm_host_vendor",
            "byte_lm_host_column", "byte_lm_host_sabotage",
            "byte_lm_host_profile", "byte_lm_host_logits", "byte_lm_host_loss",
            "byte_lm_host_train_step", "all_finite_f32", "all_finite_f64",
            "cast_f64_to_f32",
        ),
        gate=".github/workflows/byte-lm-cpu-gate.yml",
        ships_in_wheel=True,
    ),
    dict(
        family="forest",
        binding="_mojolearn_forest_host",
        routes=None,
        loaded_by="python/mojolearn/_forest_host.py, python/mojolearn/_gbdt_host.py",
        sabotage_define="MOJOLEARN_FOREST_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=(),
        forest_kinds=(
            "rf_classifier", "rf_regressor", "et_classifier", "et_regressor",
            "gbdt_symmetric", "gbdt_depthwise", "gbdt_lossguide", "gbdt_rmse",
        ),
        classes=(
            "RandomForestClassifier", "RandomForestRegressor",
            "ExtraTreesClassifier", "ExtraTreesRegressor", "GradientBoosting",
        ),
        display="random forests, Extra Trees and the four gradient boosting variants",
        host_modules=("core/forest_host_predict.mojo", "core/gbdt_host_predict.mojo"),
        exports=(
            "forest_host_numeric_mode", "forest_host_vendor", "forest_host_column",
            "forest_host_sabotage", "forest_host_rf_predict_proba",
            "forest_host_rf_predict_reg", "forest_host_et_predict",
            "forest_host_gbdt_predict", "forest_host_gbdt_sigmoid",
            "all_finite_f32", "all_finite_f64", "cast_f64_to_f32",
            "argmax_rows_f32", "argmax_rows_f64", "gather_i64", "gather_f64",
        ),
        gate="tools/forest_host_gate.py (.github/workflows/forest-host-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        family="core",
        binding="_mojolearn_core_host",
        routes="_mojolearn",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=(),
        inference_lanes=("knn", "knn-clf", "knn-reg"),
        forest_kinds=(),
        classes=("NearestNeighbors", "KNeighborsClassifier", "KNeighborsRegressor"),
        display="nearest neighbors, k-NN classification and k-NN regression",
        host_modules=("core/knn_host_predict.mojo", "bindings/host_helpers.mojo"),
        exports=(
            "core_host_numeric_mode", "core_host_vendor", "core_host_column",
            "core_host_sabotage", "mojolearn_vendor", "mojolearn_numeric_mode",
            "knn_search", "knn_classify", "knn_regress", "transpose_f32",
            "cast_colmajor_f64_to_f32", "cast_f64_to_f32", "all_finite_f32",
            "all_finite_f64", "gather_i64", "gather_f64", "argmax_rows_f32",
            "argmax_rows_f64",
        ),
        gate="tools/classical_host_gate.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        family="linalg",
        binding="_mojolearn_linalg_host",
        routes="_mojolearn_linalg",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("gemm-pinned",),
        inference_lanes=(),
        forest_kinds=(),
        classes=("linalg.gemm", "linalg.gemv"),
        display="pinned GEMM",
        host_modules=("gemm/host/gemm_oracle.mojo",),
        exports=(
            "linalg_host_numeric_mode", "linalg_host_vendor", "linalg_host_column",
            "linalg_host_sabotage", "linalg_vendor", "linalg_numeric_mode",
            "linalg_profile_version", "gemm",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        family="estimators",
        binding="_mojolearn_estimators_host",
        routes="_mojolearn_estimators",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("kde",),
        inference_lanes=("ols", "ridge", "tsvd", "logistic", "pca", "pca-whiten", "kde"),
        forest_kinds=(),
        classes=(
            "LinearRegression", "Ridge", "TruncatedSVD", "LogisticRegression",
            "PCA", "KernelDensity",
        ),
        display="linear regression, ridge, truncated SVD, logistic regression, PCA with and without whitening and kernel density",
        host_modules=("kde/host/kde_oracle.mojo", "core/classical_host_predict.mojo"),
        exports=(
            "estimators_host_numeric_mode", "estimators_host_vendor",
            "estimators_host_column", "estimators_host_sabotage",
            "estimators_vendor", "estimators_numeric_mode", "kde_score_samples",
            "ols_predict", "tsvd_transform", "pca_transform",
            "pca_whiten_transform", "pca_whiten_inverse_transform",
            "qn_decision_function", "qn_sigmoid",
        ),
        gate="tools/identity_break.py and tools/classical_host_gate.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        family="tsa",
        binding="_mojolearn_tsa_host",
        routes="_mojolearn_tsa",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("holtwinters",),
        inference_lanes=(),
        forest_kinds=(),
        classes=("ExponentialSmoothing",),
        display="Holt-Winters",
        host_modules=("holtwinters/host/hw_oracle.mojo",),
        exports=(
            "tsa_host_numeric_mode", "tsa_host_vendor", "tsa_host_column",
            "tsa_host_sabotage", "tsa_vendor", "holtwinters_fit",
            "holtwinters_forecast",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        family="solver",
        binding="_mojolearn_solver_host",
        routes="_mojolearn_solver",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("lasso", "elasticnet"),
        inference_lanes=(),
        forest_kinds=(),
        classes=("Lasso", "ElasticNet"),
        display="lasso and elasticnet",
        host_modules=("solver/host/cd_oracle.mojo", "gemm/host/gemm_oracle.mojo"),
        exports=(
            "solver_host_numeric_mode", "solver_host_vendor", "solver_host_column",
            "solver_host_sabotage", "solver_vendor", "cd_fit", "cd_predict",
        ),
        gate="tools/identity_break.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
    dict(
        family="svm",
        binding="_mojolearn_svm_host",
        routes="_mojolearn_svm",
        loaded_by="_backend._HOST_MODULES",
        sabotage_define="MOJOLEARN_HOST_SABOTAGE",
        training_lanes=("svc",),
        inference_lanes=("svc",),
        forest_kinds=(),
        classes=("SVC",),
        display="SVC",
        host_modules=("svm/host/smo_oracle.mojo", "gemm/host/gemm_oracle.mojo"),
        exports=(
            "svm_host_numeric_mode", "svm_host_vendor", "svm_host_column",
            "svm_host_sabotage", "svm_vendor", "svm_numeric_mode", "svc_fit",
            "svc_predict",
        ),
        gate="tools/identity_break.py and tools/classical_host_gate.py (cpu-identity-gate.yml)",
        ships_in_wheel=False,
    ),
)


def families():
    """Every host family name, in build order."""
    return [f["family"] for f in FAMILIES]


def family(name):
    for f in FAMILIES:
        if f["family"] == name:
            return f
    raise KeyError(f"no host family named {name!r}; the manifest lists {families()}")


def bindings():
    """Every host binding basename, in the same order."""
    return [f["binding"] for f in FAMILIES]


def binding_source(name):
    """The Mojo source of a family's binding."""
    return f"bindings/_mojolearn_{name}_host.mojo"


def build_shim(name):
    """The two-line shim that execs BUILDER for a family."""
    return f"bindings/build_{name}_host.sh"


def routed_modules():
    """`_MODULES` name -> host binding basename: the table `_backend` routes a
    CPU-only install through. The two bindings loaded by path (byte_lm,
    forest) are deliberately absent."""
    return {f["routes"]: f["binding"] for f in FAMILIES if f["routes"]}


def routed_families():
    """The families with a route, the phase 1 set the gate builds in a loop."""
    return [f["family"] for f in FAMILIES if f["routes"]]


def routed_bindings():
    return [f["binding"] for f in FAMILIES if f["routes"]]


def covered_lanes():
    """The identity_break lanes with a CPU TRAINING path, in gate order."""
    out = []
    for f in FAMILIES:
        for lane in f["training_lanes"]:
            if lane not in out:
                out.append(lane)
    return out


def inference_lanes():
    """The classical gate lanes served from a saved model, in gate order."""
    out = []
    for f in FAMILIES:
        for lane in f["inference_lanes"]:
            if lane not in out:
                out.append(lane)
    return out


def forest_kinds():
    return list(family("forest")["forest_kinds"])


def sabotage_define(name):
    return family(name)["sabotage_define"]


def training_sentence():
    """The training list as the README states it."""
    return _join([TRAINING_LANE_NAMES[lane] for lane in covered_lanes()])


def inference_sentence():
    """The inference list as the README states it, one clause per family
    that serves a saved model."""
    parts = [f["display"] for f in FAMILIES if f["inference_lanes"] or f["forest_kinds"]]
    return "; ".join(parts)


def no_cpu_path_sentence():
    return _join(list(NO_CPU_PATH))


def _join(items):
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def markdown_table():
    """The CPU surface as one table, for the marked spans in
    SUPPORT_MATRIX.md and docs/BYTE_LM_CPU_TRAINING.md."""
    rows = [
        "| family | binding under `mojolearn/host/` | routes (CPU-only install) | trains on a CPU (identity_break lanes) | predicts on a CPU from a saved model | gate | in a wheel |",
        "|---|---|---|---|---|---|---|",
    ]
    for f in FAMILIES:
        trains = ", ".join(f["training_lanes"]) or "no"
        if f["forest_kinds"]:
            predicts = ", ".join(f["classes"]) + " (" + ", ".join(f["forest_kinds"]) + ")"
        elif f["inference_lanes"]:
            predicts = ", ".join(f["classes"]) + " (" + ", ".join(f["inference_lanes"]) + ")"
        elif f["family"] == "byte_lm":
            predicts = ", ".join(f["classes"])
        else:
            predicts = "no"
        rows.append(
            f"| {f['family']} | `{f['binding']}.so` | {('`' + f['routes'] + '`') if f['routes'] else 'loaded by path'} "
            f"| {trains} | {predicts} | {f['gate']} | {'yes' if f['ships_in_wheel'] else 'no, `' + build_shim(f['family']) + '`'} |"
        )
    return "\n".join(rows)


def as_dict():
    return dict(
        source=SOURCE,
        builder=BUILDER,
        families=[dict(f) for f in FAMILIES],
        routed=routed_modules(),
        covered_lanes=covered_lanes(),
        inference_lanes=inference_lanes(),
        forest_kinds=forest_kinds(),
        classical_recorded=list(CLASSICAL_RECORDED),
        classical_gpu_columns=list(CLASSICAL_GPU_COLUMNS),
        training_gpu_columns=list(TRAINING_GPU_COLUMNS),
        forest_recorded_root=FOREST_RECORDED_ROOT,
        no_cpu_path=list(NO_CPU_PATH),
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="the CPU surface manifest; one flag per list")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--families", action="store_true", help="every host family")
    g.add_argument("--routed-families", action="store_true", help="families routed by _backend._HOST_MODULES")
    g.add_argument("--bindings", action="store_true", help="every host binding basename")
    g.add_argument("--routed-bindings", action="store_true", help="the routed families' basenames")
    g.add_argument("--covered-lanes", action="store_true", help="identity_break lanes with a CPU training path (comma separated)")
    g.add_argument("--inference-lanes", action="store_true", help="classical gate lanes served from a saved model (comma separated)")
    g.add_argument("--forest-kinds", action="store_true", help="forest gate kinds (comma separated)")
    g.add_argument("--classical-recorded", action="store_true", help="classical gate recording directories")
    g.add_argument("--classical-gpu-columns", action="store_true", help="the GPU columns the classical gate compares against")
    g.add_argument("--training-gpu-columns", action="store_true", help="the GPU columns the training gate diffs against")
    g.add_argument("--markdown", action="store_true", help="the surface as a Markdown table")
    g.add_argument("--json", action="store_true", help="the whole manifest as JSON")
    p.add_argument("--sep", default=None, help="separator for list output (default: comma for lanes and kinds, space otherwise)")
    args = p.parse_args(argv)
    if args.json:
        print(json.dumps(as_dict(), indent=2, sort_keys=True))
        return 0
    if args.markdown:
        print(markdown_table())
        return 0
    comma = (args.covered_lanes or args.inference_lanes or args.forest_kinds)
    sep = args.sep if args.sep is not None else ("," if comma else " ")
    if args.families:
        items = families()
    elif args.routed_families:
        items = routed_families()
    elif args.bindings:
        items = bindings()
    elif args.routed_bindings:
        items = routed_bindings()
    elif args.covered_lanes:
        items = covered_lanes()
    elif args.inference_lanes:
        items = inference_lanes()
    elif args.forest_kinds:
        items = forest_kinds()
    elif args.classical_recorded:
        items = list(CLASSICAL_RECORDED)
    elif args.classical_gpu_columns:
        items = list(CLASSICAL_GPU_COLUMNS)
    else:
        items = list(TRAINING_GPU_COLUMNS)
    print(sep.join(items))
    return 0


if __name__ == "__main__":
    sys.exit(main())
