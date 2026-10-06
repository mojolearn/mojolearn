"""Opt-in metadata capture for the existing forest benchmark, not ML runtime."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys


def packet(data, cfg, retained_scores, states, records) -> dict:
    from bench_board_state import canonical_hash
    # Hash the actual arrays loaded by the caller, including labels and query
    # metadata. Declaring a cache-file hash alone does not prove it was loaded.
    inputs = {name: value for name, value in vars(data).items()
              if not name.startswith("_") and hasattr(value, "shape") and hasattr(value, "dtype")}
    loaded = {}
    for module in tuple(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if not name:
            continue
        path = Path(name)
        if path.name.startswith("_mojolearn") and path.suffix == ".so":
            path = path.resolve()
            relative = "identical/" + path.name if path.parent.name == "identical" else path.name
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for block in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(block)
            loaded[relative] = {"path": str(path), "sha256": digest.hexdigest()}
    metrics = {}
    for arm, (_, triples, _) in retained_scores.items():
        metrics[arm] = {name: float(value) for name, value, _ in triples}
    models = {}
    for arm, model in records.items():
        getter = getattr(model, "get_params", None)
        models[arm] = getter(deep=False) if callable(getter) else {"unavailable": True}
    pools = None
    try:
        from threadpoolctl import threadpool_info
        pools = threadpool_info()
    except ImportError:
        pass  # Explicitly unknown; an environment value is not pool evidence.
    return {
        "dataset": data.name,
        "dimensions": {"train": list(data.X_train.shape), "test": list(data.X_test.shape)},
        "input_array_sha256": canonical_hash(inputs),
        "estimator_settings": cfg, "actual_model_parameters": models,
        "metrics": metrics, "states": states, "loaded_bindings": loaded,
        "resource_policy": {"cpu_libraries": "unrestricted", "cpu_count": os.cpu_count(),
                            "effective_pools": pools, "thread_environment": {
                                key: os.environ.get(key) for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MOJOLEARN_BENCH_THREADS")}},
        "mode": os.environ.get("MOJOLEARN_NUMERIC_MODE"),
        "vendor": os.environ.get("MOJOLEARN_VENDOR"),
        "full_dataset_coverage": "must match separately frozen dataset hash/split and intrinsic-cap recipe",
    }
