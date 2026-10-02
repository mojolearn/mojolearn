#!/usr/bin/env python3
"""Split the no_host_routes debt rows into file-disjoint sub-lanes.

Writes docs/plans/cpu-gpu-cleanup/rows/<lane>.tsv. First matching prefix wins.
"""
import collections, os, sys

LANES = [
    # family, lane, path prefixes
    ("neural", "n-gemm", ["gemm/", "bindings/_mojolearn_linalg.mojo"]),
    ("neural", "n-train-mamba", ["training/", "mamba/", "transformer/", "bindings/_mojolearn_training.mojo",
                                 "bindings/_mojolearn_mamba.mojo", "bindings/_mojolearn_transformer.mojo"]),
    ("neural", "n-seq", ["python/mojolearn/_x_sequence_", "arima/", "sequence/", "holtwinters/", "tsa/", "bindings/holtwinters_host_predict.mojo",
                         "bindings/arima_exog_layout.mojo", "bindings/_mojolearn_arima", "bindings/_mojolearn_sequence",
                         "bindings/_mojolearn_holtwinters", "bindings/_mojolearn_tsa"]),
    ("neural", "n-pyneural", ["python/mojolearn/neural_inference.py", "python/mojolearn/lm_corpus.py", "python/mojolearn/language_model.py", "python/mojolearn/_tokenizer_synthetic.py", "python/mojolearn/_causal_lm_fixtures.py", "python/mojolearn/_bpe_trainer.py", "x_cnn/", "tokenizer/", "bindings/_mojolearn_x_cnn.mojo", "python/mojolearn/_expansion_cnn.py",
                              "python/mojolearn/_byte_lm_impl.py", "python/mojolearn/lowbit.py",
                              "python/mojolearn/tokenizer.py", "python/mojolearn/models", "python/mojolearn/_cnn",
                              "python/mojolearn/byte_lm", "bindings/_mojolearn_tokenizer", "bindings/_mojolearn_byte_lm"]),
    ("classical", "c-xneighbors-iter", ["x_neighbors/iter_device.mojo"]),
    ("classical", "c-xneighbors", ["x_neighbors/", "neighbors/", "python/mojolearn/_expansion_neighbors.py",
                                   "bindings/_mojolearn_x_neighbors", "bindings/_mojolearn_neighbors"]),
    ("classical", "c-ann", ["x_ann/", "ivf/", "bindings/_mojolearn_ivf.mojo", "bindings/ivf_index_arrays.mojo",
                            "bindings/_mojolearn_x_ann"]),
    ("classical", "c-core", ["python/mojolearn/cross_vendor.py", "python/mojolearn/_mode.py", "python/mojolearn/_labels.py", "python/mojolearn/_buffer.py", "python/mojolearn/_array.py", "core/", "bindings/_mojolearn.mojo", "bindings/hotpath_helpers.mojo", "bindings/hostptr.mojo",
                             "python/mojolearn/_bufcheck.py", "python/mojolearn/__init__.py",
                             "python/mojolearn/_parallel_pool.py", "python/mojolearn/_portable_math.py"]),
    ("classical", "c-decomp", ["decomposition/", "x_decomp/", "python/mojolearn/_expansion_decomp.py",
                               "bindings/_mojolearn_decomposition", "bindings/_mojolearn_x_decomp"]),
    ("classical", "c-gp-kernel", ["python/mojolearn/_gp_impl.py", "gaussian_process/", "kernel_methods/", "bindings/_mojolearn_gp.mojo",
                                  "bindings/_mojolearn_kernel_methods.mojo", "python/mojolearn/_gpc_impl.py",
                                  "python/mojolearn/parallel_gaussian_process.py"]),
    ("classical", "c-svm", ["svm/", "bindings/_mojolearn_svm.mojo", "python/mojolearn/_svm_impl.py"]),
    ("classical", "c-linear", ["python/mojolearn/_cholesky_impl.py", "x_linear/", "glm/", "solver/", "cholesky/", "python/mojolearn/_expansion_linear.py",
                               "python/mojolearn/_linalg_impl.py", "bindings/_mojolearn_x_linear", "bindings/_mojolearn_glm",
                               "bindings/_mojolearn_solver"]),
    ("classical", "c-cluster", ["cluster/", "hdbscan/", "dbscan/", "hierarchy/", "mixture/", "spectral/", "umap/",
                                "embedding/", "x_cluster/", "bindings/_mojolearn_hdbscan.mojo",
                                "bindings/_mojolearn_mixture.mojo", "bindings/_mojolearn_x_cluster.mojo",
                                "bindings/_mojolearn_embedding.mojo", "python/mojolearn/_expansion_cluster.py",
                                "bindings/_mojolearn_umap", "bindings/_mojolearn_spectral", "bindings/_mojolearn_cluster"]),
    ("classical", "c-metrics-prep", ["bindings/_mojolearn_preprocessing.mojo", "bindings/_mojolearn_metrics.mojo", "python/mojolearn/parallel_preprocessing.py", "x_metrics/", "metrics/", "x_prep/", "preprocessing/", "resample/", "naive_bayes/",
                                     "kde/", "bindings/_mojolearn_x_metrics.mojo", "bindings/_mojolearn_x_prep.mojo",
                                     "bindings/_mojolearn_resample.mojo", "python/mojolearn/preprocessing.py",
                                     "python/mojolearn/_expansion_metrics.py", "python/mojolearn/_expansion_prep.py",
                                     "python/mojolearn/_metrics_impl.py", "python/mojolearn/model_selection.py"]),
    ("trees", "t-forest", ["python/mojolearn/_forest_protocol.py", "extratrees/", "ensemble/", "isolation_forest/", "bindings/_mojolearn_rf.mojo",
                           "python/mojolearn/ensemble.py", "bindings/_mojolearn_isolation"]),
    ("trees", "t-gbdt", ["gbdt/", "xtrees/", "bindings/_mojolearn_trees.mojo", "bindings/_mojolearn_estimators.mojo",
                         "python/mojolearn/_expansion_trees.py", "python/mojolearn/_gbdt_adapters.py",
                         "bindings/_mojolearn_gbdt", "bindings/_mojolearn_xtrees"]),
]

here = os.path.dirname(os.path.abspath(__file__))
root = os.path.abspath(os.path.join(here, "../../.."))
rows = collections.defaultdict(list)
header = None
for line in open(os.path.join(root, "tools/hooks/host_routes_baseline.tsv")):
    if line.startswith("#"):
        continue
    f = line.rstrip("\n").split("\t")
    if f[0] == "rule" or f[4] == "path":
        header = line
        continue
    if f[3] != "debt":
        continue
    lane = "unassigned"
    for fam, name, prefixes in LANES:
        if any(f[4].startswith(p) for p in prefixes):
            lane = name
            break
    rows[lane].append(line)
os.makedirs(os.path.join(here, "rows"), exist_ok=True)
fam_of = {n: fam for fam, n, _ in LANES}
tot = collections.Counter()
for lane, ls in sorted(rows.items()):
    with open(os.path.join(here, "rows", lane + ".tsv"), "w") as o:
        o.write(header)
        o.writelines(ls)
    tot[fam_of.get(lane, "?")] += len(ls)
    print(f"{fam_of.get(lane, '?'):10s} {lane:20s} {len(ls):4d}")
print(dict(tot))
