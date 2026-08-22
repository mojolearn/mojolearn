#!/usr/bin/env python3
"""Make a gbm-bench checkout runnable on a machine with no CUDA, and
register the mojolearn arms.

gbm-bench is NVIDIA's harness and assumes NVIDIA hardware: `algorithms.py`
imports `dask_cuda` and `xgboost` at module scope, so on an Apple silicon Mac
it fails at import before any benchmark runs. This script makes exactly those
imports optional and adds five names to the algorithm factory. It changes no
timing code, no metric, no dataset, and no parameter belonging to another
library's EXISTING arm, which is the property that makes a number from this
harness worth more than a number from ours. (The lgbm forest arms are ADDED
arms with their own parameters; adding an arm is not the same act as
changing one.)

Every edit is anchored to an exact upstream string and asserted. If upstream
moves, this fails loudly rather than silently patching the wrong thing. Run
it as many times as you like; it detects its own work and stops.

Derived from mojotrees' bench/external/patch_gbm_bench.py (same repo family,
same restraint rules).
"""

import io
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARKER = "# --- mojolearn arm (bench/external/patch_gbm_bench.py) ---"


def _read(path):
    with io.open(path, encoding="utf-8") as handle:
        return handle.read()


def _write(path, text):
    with io.open(path, "w", encoding="utf-8") as handle:
        handle.write(text)


def _replace_once(text, old, new, what):
    if old not in text:
        raise SystemExit(
            "gbm-bench has moved: could not find the anchor for " + what +
            ".\nExpected to find:\n" + old +
            "\nRe-read algorithms.py and update patch_gbm_bench.py rather "
            "than forcing this through."
        )
    if text.count(old) != 1:
        raise SystemExit("anchor for " + what + " is not unique; refusing")
    return text.replace(old, new, 1)


CUDA_IMPORTS_OLD = (
    "import dask.dataframe as dd\n"
    "import dask.array as da\n"
    "from dask.distributed import Client\n"
    "from dask_cuda import LocalCUDACluster\n"
    "import xgboost as xgb\n"
)

CUDA_IMPORTS_NEW = (
    "# These are CUDA-only or CUDA-adjacent and are not installable on\n"
    "# every machine this harness now runs on. Made optional so the module\n"
    "# imports; every algorithm that needs one still fails at construction\n"
    "# time if it is missing.\n"
    "try:\n"
    "    import dask.dataframe as dd\n"
    "    import dask.array as da\n"
    "    from dask.distributed import Client\n"
    "except ImportError:\n"
    "    dd = da = Client = None\n"
    "try:\n"
    "    from dask_cuda import LocalCUDACluster\n"
    "except ImportError:\n"
    "    LocalCUDACluster = None\n"
    "try:\n"
    "    import xgboost as xgb\n"
    "except ImportError:\n"
    "    xgb = None\n"
)


def patch_algorithms(root):
    path = os.path.join(root, "algorithms.py")
    text = _read(path)
    if MARKER in text:
        print("algorithms.py already patched (mojolearn)")
        return

    # 1. CUDA-only imports become optional so the module loads off-NVIDIA.
    #    A checkout already patched by mojotrees' script has this edit
    #    applied under its marker; skip it there.
    if CUDA_IMPORTS_OLD in text:
        text = _replace_once(
            text, CUDA_IMPORTS_OLD, MARKER + "\n" + CUDA_IMPORTS_NEW,
            "the CUDA-only import block",
        )
    elif "except ImportError:" not in text:
        raise SystemExit(
            "algorithms.py has neither the stock CUDA import block nor an "
            "optional one; re-read it before patching"
        )

    # 2. Register the mojolearn arms in the factory.
    text = _replace_once(
        text,
        "        raise ValueError(\"Unknown algorithm: \" + name)",
        "        " + MARKER + "\n"
        "        if name == 'mojolearn-gbdt-gpu':\n"
        "            return MojolearnGbdtGPUAlgorithm()\n"
        "        if name == 'mojolearn-et-gpu':\n"
        "            return MojolearnExtraTreesGPUAlgorithm()\n"
        "        if name == 'mojolearn-rf-gpu':\n"
        "            return MojolearnRandomForestGPUAlgorithm()\n"
        "        if name == 'skl-et-cpu':\n"
        "            return SkExtraTreesCPUAlgorithm()\n"
        "        if name == 'skl-rf-cpu':\n"
        "            return SkRandomForestCPUAllCoresAlgorithm()\n"
        "        if name == 'lgbm-et-cpu':\n"
        "            return LgbmExtraTreesCPUAlgorithm()\n"
        "        if name == 'lgbm-rf-cpu':\n"
        "            return LgbmRandomForestCPUAlgorithm()\n"
        "        if name == 'lgbm-et-gpu':\n"
        "            return LgbmExtraTreesGPUAlgorithm()\n"
        "        if name == 'lgbm-rf-gpu':\n"
        "            return LgbmRandomForestGPUAlgorithm()\n"
        "        raise ValueError(\"Unknown algorithm: \" + name)",
        "the algorithm factory",
    )

    # 3. Import the adapters. They subclass Algorithm, so this goes at the
    #    end of the file, after the base class and shared_params exist.
    text = text.rstrip("\n") + "\n\n\n" + MARKER + "\n" + \
        "from mojolearn_algorithm import (  # noqa: E402\n" \
        "    MojolearnGbdtGPUAlgorithm,\n" \
        "    MojolearnExtraTreesGPUAlgorithm,\n" \
        "    MojolearnRandomForestGPUAlgorithm,\n" \
        "    SkRandomForestCPUAllCoresAlgorithm,\n" \
        "    SkExtraTreesCPUAlgorithm,\n" \
        "    LgbmExtraTreesCPUAlgorithm,\n" \
        "    LgbmRandomForestCPUAlgorithm,\n" \
        "    LgbmExtraTreesGPUAlgorithm,\n" \
        "    LgbmRandomForestGPUAlgorithm,\n" \
        ")\n"

    _write(path, text)
    print("patched algorithms.py (mojolearn arms)")


def patch_metrics(root):
    """ONE compatibility edit to their metric code, semantics preserved.

    `classification_metrics` calls `sklm.log_loss(real, y_prob, eps=1e-5)`;
    sklearn removed `eps` in 1.5, so every BINARY dataset crashes in their
    metric under a current sklearn. `eps` clipped the probabilities to
    [eps, 1-eps] before the log; `np.clip` is that exact arithmetic, spelled
    without the removed keyword. Same numbers on any sklearn that still
    accepts `eps`, and the reserved-in-README compatibility-only exception
    to the no-edits rule."""
    path = os.path.join(root, "metrics.py")
    text = _read(path)
    if MARKER in text:
        print("metrics.py already patched (log_loss eps compat)")
        return
    text = _replace_once(
        text,
        "sklm.log_loss(real, y_prob, eps=1e-5),",
        "sklm.log_loss(real, np.clip(y_prob, 1e-5, 1 - 1e-5)),  " + MARKER,
        "the binary log_loss eps call",
    )
    _write(path, text)
    print("patched metrics.py (log_loss eps compat, semantics preserved)")


RUNME_HELPER = '''

''' + MARKER + '''
def _mojolearn_result_hashes(data, pred):
    """Bit-exact fingerprints for the results JSON.

    Two runs whose "predictions" hashes match produced BYTE-IDENTICAL
    prediction vectors -- a stronger statement than any rounded accuracy
    column, and the one the cross-device identity claim is made of: the
    same fit on a second vendor's GPU must reproduce this hash, not an
    approximation of the metric. "test_target" pins the eval rows the
    predictions were scored against, so a hash match can never be two
    different subsets agreeing by luck. The hash covers dtype, shape and
    raw bytes; it costs one pass over arrays that already exist and runs
    AFTER both timed phases, so it cannot perturb a timing.
    """
    import hashlib

    import numpy as np

    def one(arr):
        a = np.ascontiguousarray(np.asarray(arr))
        digest = hashlib.sha256()
        digest.update(str(a.dtype).encode())
        digest.update(str(a.shape).encode())
        digest.update(a.tobytes())
        return digest.hexdigest()

    try:
        return {"predictions": one(pred), "test_target": one(data.y_test)}
    except Exception as exc:  # a new arm returning an exotic type
        return {"error": "unhashable: %s" % exc}


'''


def patch_runme(root):
    """Add the hash section to every arm's results entry.

    Timing-neutral by construction: the hash runs after `fit` and `test`
    have both returned and their times are already captured.
    """
    path = os.path.join(root, "runme.py")
    text = _read(path)
    if MARKER in text:
        print("runme.py already patched (result hashes)")
        return
    text = _replace_once(
        text,
        "# benchmarks a single dataset\ndef benchmark(",
        RUNME_HELPER.lstrip("\n") + "\n# benchmarks a single dataset\ndef benchmark(",
        "the benchmark() definition",
    )
    text = _replace_once(
        text,
        "            results[alg] = {\n"
        "                \"train_time\": train_time,\n"
        "                \"accuracy\": get_metrics(data, pred),\n"
        "            }",
        "            results[alg] = {\n"
        "                \"train_time\": train_time,\n"
        "                \"accuracy\": get_metrics(data, pred),\n"
        "                \"hashes\": _mojolearn_result_hashes(data, pred),\n"
        "            }",
        "the per-arm results dict",
    )
    _write(path, text)
    print("patched runme.py (bit-exact result hashes)")


def install_adapter(root):
    src = os.path.join(HERE, "gbm_bench", "mojolearn_algorithm.py")
    dst = os.path.join(root, "mojolearn_algorithm.py")
    shutil.copyfile(src, dst)
    print("installed mojolearn_algorithm.py")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: patch_gbm_bench.py <path-to-gbm-bench>")
    root = os.path.abspath(sys.argv[1])
    if not os.path.isfile(os.path.join(root, "algorithms.py")):
        raise SystemExit("not a gbm-bench checkout: " + root)
    install_adapter(root)
    patch_algorithms(root)
    patch_metrics(root)
    patch_runme(root)
    print("\nready. mojolearn-gbdt-gpu, mojolearn-et-gpu, skl-et-cpu, "
          "lgbm-et-cpu and lgbm-rf-cpu are now valid -algorithm values.")


if __name__ == "__main__":
    main()
