#!/usr/bin/env python3
"""One pass that says what a box is MISSING, instead of one lease per discovery.

WHY THIS EXISTS. Four H100 leases on 2026-09-03 died one import at a time:
the system python3 had no numpy, then the environment that had numpy had no
pandas, then the environment that had both had to be installed before the job
that imported from it. Each fix was correct and each was incomplete, because
each run only ever reported the FIRST thing wrong with it.

The vendor comparison legs have a much larger install surface -- cuML, cuVS,
CatBoost GPU, LightGBM CUDA, PyTorch, mamba-ssm -- and discovering it the same
way would cost several more leases. This probe imports everything, catches
every failure, and prints one line per (interpreter, module). It never raises
on a missing opponent: a missing opponent is the ANSWER, not a crash.

  PREFLIGHT <tag> <module> <OK|MISSING|ERROR> <detail>

Exit is non-zero only when a REQUIRED module is missing. Opponents are
optional by construction -- the point is to learn which ones a box has.
"""
import importlib
import os
import sys

TAG = os.environ.get("PREFLIGHT_TAG", "unknown")

# (module, required?, what it is for)
PROBES = [
    ("numpy",       True,  "every arm"),
    ("pandas",      False, "the higgs decode in speed_gbdt_arm"),
    ("sklearn",     False, "the cpu opponents"),
    ("catboost",    False, "the gbdt-symmetric opponent (port lineage)"),
    ("lightgbm",    False, "the depthwise/lossguide opponent"),
    ("xgboost",     False, "a second gbdt opponent"),
    ("cuml",        False, "the classical + rf opponents"),
    ("cuvs",        False, "the knn/kmeans opponents"),
    ("torch",       False, "the transformer opponent"),
    ("mamba_ssm",   False, "the mamba opponent"),
]


def probe(name):
    try:
        m = importlib.import_module(name)
    except ImportError as exc:
        return "MISSING", str(exc)[:80]
    except Exception as exc:                      # a broken install, not an absent one
        return "ERROR", f"{type(exc).__name__}: {str(exc)[:70]}"
    return "OK", getattr(m, "__version__", "?")


def main():
    print(f"PREFLIGHT {TAG} python {sys.version.split()[0]} at {sys.executable}")
    missing_required = []
    for name, required, why in PROBES:
        status, detail = probe(name)
        req = "REQUIRED" if required else "optional"
        print(f"PREFLIGHT {TAG} {name} {status} [{req}] {detail}  ({why})")
        if required and status != "OK":
            missing_required.append(name)

    # OUR OWN EXTENSION, which is the one import that is never optional and the
    # one a fresh box is most likely to get wrong: it is built against an
    # interpreter and loaded by another.
    sys.path.insert(0, "python")
    try:
        import mojolearn
        print(f"PREFLIGHT {TAG} mojolearn OK [REQUIRED] mode={mojolearn.numeric_mode()}")
    except Exception as exc:
        print(f"PREFLIGHT {TAG} mojolearn ERROR [REQUIRED] "
              f"{type(exc).__name__}: {str(exc)[:70]}")
        missing_required.append("mojolearn")

    # DEVICE VISIBILITY, asked of the opponent stack rather than of nvidia-smi:
    # a box can show a GPU to the driver and not to torch or cuML.
    try:
        import torch
        print(f"PREFLIGHT {TAG} torch.cuda {torch.cuda.is_available()} "
              f"devices={torch.cuda.device_count()}")
    except Exception:
        pass

    if missing_required:
        print(f"PREFLIGHT {TAG} FAILED: missing required {','.join(missing_required)}")
        return 1
    print(f"PREFLIGHT {TAG} OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
