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
        "        if name == 'lgbm-et-cpu':\n"
        "            return LgbmExtraTreesCPUAlgorithm()\n"
        "        if name == 'lgbm-rf-cpu':\n"
        "            return LgbmRandomForestCPUAlgorithm()\n"
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
        "    SkExtraTreesCPUAlgorithm,\n" \
        "    LgbmExtraTreesCPUAlgorithm,\n" \
        "    LgbmRandomForestCPUAlgorithm,\n" \
        ")\n"

    _write(path, text)
    print("patched algorithms.py (mojolearn arms)")


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
    print("\nready. mojolearn-gbdt-gpu, mojolearn-et-gpu, skl-et-cpu, "
          "lgbm-et-cpu and lgbm-rf-cpu are now valid -algorithm values.")


if __name__ == "__main__":
    main()
