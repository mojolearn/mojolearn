#!/usr/bin/env python3
"""Held-out UMAP transform quality gate; only the main lane executes it.

Training and query rows are disjoint. Fit sees training rows only. Quality
compares query-to-training neighborhoods before and after embedding, without
requiring coordinate equality or sklearn. It does not certify upstream parity,
cross-vendor identity, or performance.

Declared before the first run: k=5, bipartite trustworthiness >=0.85,
neighbor retention >=0.35, and >=0.15 improvement in BOTH metrics over EACH
of two deterministic correspondence-breaking controls. Do not tune these
thresholds to a failing result; retain that result and diagnose the feature.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

# Bound this standalone process before importing NumPy or MojoLearn.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np

import mojolearn
from mojolearn import UMAP


NEIGHBORS = 5
THRESHOLDS = {"trustworthiness": 0.85, "retention": 0.35,
              "minimum_control_margin": 0.15}
SOURCE_FILES = (
    "umap/transform.mojo", "umap/estimator.mojo", "umap/params.mojo",
    "umap/curve.mojo", "umap/optimizer.mojo", "umap/graph.mojo",
    "umap/spectral_init.mojo", "neighbors/estimator.mojo",
    "neighbors/checks/pinned_distance_tile.mojo",
    "neighbors/checks/select_radix_identical.mojo", "core/row_norms.mojo",
    "checks/numerics.mojo", "bindings/_mojolearn_metrics.mojo",
    "python/mojolearn/_umap_impl.py",
)


def fixtures():
    # Dyadic coordinates make the fixture bytes independent of libm/RNG.
    t = (np.arange(128, dtype=np.float32) - 64) / np.float32(64)
    curve = np.column_stack((t, t * t, t * t * t)).astype(np.float32)
    yield "cubic128_interleaved64_64", curve[::2].copy(), curve[1::2].copy(), 2, 19

    i = np.arange(64, dtype=np.float32)
    u = (i % 8 - np.float32(3.5)) / np.float32(4)
    v = (i // 8 - np.float32(3.5)) / np.float32(4)
    train = np.column_stack((u, v, (u * u - v * v) / np.float32(2)))
    j = np.arange(49, dtype=np.float32)
    qu = (j % 7 - np.float32(3)) / np.float32(4)
    qv = (j // 7 - np.float32(3)) / np.float32(4)
    query = np.column_stack((qu, qv, (qu * qu - qv * qv) / np.float32(2)))
    yield "saddle_grid64_cellcenters49", train.astype(np.float32), query.astype(np.float32), 3, 7


def neighbor_order(query, train):
    """Euclidean ranks; exact ties take lowest training-row index."""
    query = np.asarray(query, dtype=np.float64)
    train = np.asarray(train, dtype=np.float64)
    if (query.ndim != 2 or train.ndim != 2 or query.shape[1] != train.shape[1]
            or not 0 < len(query) <= 128 or not 0 < len(train) <= 256
            or not np.isfinite(query).all() or not np.isfinite(train).all()):
        raise ValueError("quality metric requires finite bounded 2D arrays")
    delta = query[:, None, :] - train[None, :, :]
    distance2 = np.sum(delta * delta, axis=2)
    return np.argsort(distance2, axis=1, kind="stable")


def quality(original_order, query_embedding, training_embedding, k=NEIGHBORS):
    """Bipartite rank-penalty trustworthiness and mean top-k overlap.

    There is no excluded self row: each query has N training candidates.
    An intrusion with original rank r>k costs r-k. Maximum per-query
    penalty is k*(2*N-3*k+1)/2, attained by selecting the k worst ranks.
    This explains the +1 (ordinary self-excluding trustworthiness uses -1).
    """
    q, n = original_order.shape
    if not 1 <= k <= n // 2:
        raise ValueError("quality k must lie in [1, n_train//2]")
    embedded = neighbor_order(query_embedding, training_embedding)[:, :k]
    rank = np.empty_like(original_order)
    np.put_along_axis(rank, original_order, np.arange(1, n + 1)[None, :], axis=1)
    selected_ranks = np.take_along_axis(rank, embedded, axis=1)
    penalty = int(np.maximum(selected_ranks - k, 0).sum())
    trust = 1.0 - (2.0 * penalty) / (q * k * (2 * n - 3 * k + 1))
    retention = float(np.mean(selected_ranks <= k))
    return {"trustworthiness": float(trust), "retention": retention,
            "rank_penalty": penalty}


def bits(array):
    array = np.ascontiguousarray(array, dtype=np.float32)
    return {"shape": list(array.shape),
            "float32_le_sha256": hashlib.sha256(array.astype("<f4").tobytes()).hexdigest(),
            "uint32": array.view(np.uint32).tolist()}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance(root):
    result = {"source_root": str(root),
              "files_sha256": {name: sha(root / name) for name in SOURCE_FILES
                               if (root / name).is_file()}}
    try:
        result["git_commit"] = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
        result["tracked_dirty_paths"] = subprocess.check_output(
            ["git", "-C", str(root), "diff", "HEAD", "--name-only"], text=True).splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        result["git_error"] = str(exc)
    return result


def save(path, record):
    path.write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("fast", "deterministic", "identical"), required=True)
    parser.add_argument("--device", required=True, help="Actual device reported by the main operator")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    record = {"schema": "umap.transform.heldout-quality.v1", "status": "RUNNING",
              "started_utc": datetime.now(timezone.utc).isoformat(),
              "mode": args.mode, "device": args.device, "platform": platform.platform(),
              "python": platform.python_version(), "numpy": np.__version__,
              "mojolearn": mojolearn.__version__, "package_file": mojolearn.__file__,
              "harness_sha256": sha(Path(__file__)), "source": provenance(args.source_root.resolve()),
              "threads": {name: os.environ[name] for name in
                          ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")},
              "metric": "query-to-training Euclidean rank trustworthiness and top-k retention",
              "k": NEIGHBORS, "thresholds": THRESHOLDS,
              "scope": "Named held-out quality cases; not identity, upstream parity or performance",
              "results": []}
    save(args.output, record)
    for name, train, query, dimensions, seed in fixtures():
        config = dict(n_neighbors=8, n_components=dimensions, n_epochs=200,
                      random_state=seed, min_dist=0.1, spread=1.0,
                      set_op_mix_ratio=1.0, local_connectivity=1.0,
                      metric="euclidean", init="spectral")
        row = {"profile": name, "parameters": config, "passed": False,
               "training_input": bits(train), "query_input": bits(query),
               "transform_schedule": {"epochs": 66, "initial_learning_rate": 0.25,
                                      "negative_sample_rate": 5, "seed": seed}}
        record["results"].append(row)
        try:
            # The query fixture must never occur among training rows.
            if np.any(np.all(query[:, None, :] == train[None, :, :], axis=2)):
                raise RuntimeError("held-out fixture contains a training row")
            train_before, query_before = train.tobytes(), query.tobytes()
            original = neighbor_order(query, train)
            model = UMAP(numeric_mode=args.mode, **config).fit(train)
            fitted = model.embedding_.copy()
            row["training_embedding"] = bits(fitted)
            row["fitted_config"] = list(model._transform_config)
            row["fitted_mode"] = model._transform_mode
            binding = model._bind()
            row["binding_file"] = binding.__file__
            row["binding_sha256"] = sha(Path(binding.__file__))
            row["binding_mode_code"] = int(binding.umap_numeric_mode())
            save(args.output, record)
            transformed = model.transform(query)
            row["query_embedding"] = bits(transformed)
            if transformed.shape != (len(query), dimensions):
                raise RuntimeError("transform output shape mismatch")
            if (train.tobytes() != train_before or query.tobytes() != query_before
                    or model.embedding_.tobytes() != fitted.tobytes()
                    or model._transform_embedding.tobytes() != fitted.tobytes()):
                raise RuntimeError("transform mutated training/query data or fitted coordinates")
            measured = quality(original, transformed, fitted)
            # Both lengths (64 and 49) are coprime to 37; these are bijections.
            qp = (np.arange(len(query)) * 37 + 11) % len(query)
            tp = (np.arange(len(train)) * 37 + 11) % len(train)
            controls = {
                "query_embedding_permutation": quality(original, transformed[qp], fitted),
                "training_embedding_permutation": quality(original, transformed, fitted[tp]),
            }
            margins = {metric: measured[metric] - max(c[metric] for c in controls.values())
                       for metric in ("trustworthiness", "retention")}
            row.update({"quality": measured, "controls": controls, "control_margins": margins,
                        "query_control_permutation": qp.tolist(),
                        "training_control_permutation": tp.tolist(),
                        "passed": all(measured[m] >= THRESHOLDS[m]
                                      and margins[m] >= THRESHOLDS["minimum_control_margin"]
                                      for m in ("trustworthiness", "retention"))})
            print(name, args.mode, measured, "control_margins", margins,
                  "PASS" if row["passed"] else "FAIL", flush=True)
        except Exception as exc:
            row["error"] = {"type": type(exc).__name__, "message": str(exc)}
            print(name, "ERROR", row["error"], flush=True)
        save(args.output, record)
    record["status"] = "PASS" if all(r["passed"] for r in record["results"]) else "FAIL"
    record["completed_utc"] = datetime.now(timezone.utc).isoformat()
    save(args.output, record)
    return 0 if record["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
