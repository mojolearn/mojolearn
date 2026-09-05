#!/usr/bin/env python3
"""Opt-in call-boundary diagnostics for the unchanged Transformer API checks.

Use the Python/environment containing the binding to investigate. --source-root
selects the test file, not the imported package; set PYTHONPATH explicitly when
investigating a source build. Redirect both stdout and stderr to retain call
events, test results and periodic faulthandler stacks. Apply an external timeout.
This is diagnostic instrumentation, not a benchmark or an additional certificate.
"""

import argparse
import faulthandler
import functools
import hashlib
import itertools
import json
import os
from pathlib import Path
import runpy
import sys
import time
import traceback


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("fast", "deterministic", "identical"),
                        required=True)
    parser.add_argument("--stack-interval", type=float, default=30.0)
    args = parser.parse_args()
    if not 0 < args.stack_interval < float("inf"):
        parser.error("--stack-interval must be finite and positive")
    test = (args.source_root.resolve() / "python/mojolearn/tests/"
            "test_transformer_surface.py")
    if not test.is_file():
        parser.error(f"missing test: {test}")

    os.environ["MOJOLEARN_NUMERIC_MODE"] = args.mode
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    # Start before package import so an import stall also leaves Python stacks.
    faulthandler.enable(file=sys.stderr, all_threads=True)
    faulthandler.dump_traceback_later(args.stack_interval, repeat=True,
                                    file=sys.stderr)
    started = time.monotonic()

    def emit(event, **fields):
        print("TRANSFORMER_API_DIAGNOSTIC " + json.dumps(
            {"event": event, "elapsed_s": time.monotonic() - started,
             "pid": os.getpid(), **fields}, sort_keys=True),
            file=sys.stderr, flush=True)

    original = None
    block_type = None
    old_argv = sys.argv[:]
    try:
        emit("IMPORT_ENTER", mode=args.mode, test=str(test),
             test_sha256=hashlib.sha256(test.read_bytes()).hexdigest(),
             python=sys.executable)
        import numpy as np
        import mojolearn
        from mojolearn._transformer_impl import TransformerBlock

        block_type = TransformerBlock
        original = block_type._call
        emit("IMPORT_RETURN", package=mojolearn.__file__,
             version=getattr(mojolearn, "__version__", None),
             implementation=sys.modules[block_type.__module__].__file__)

        def array_meta(value):
            # Never coerce, copy, inspect values or invoke GPU array protocols.
            if type(value) is np.ndarray:
                return {"type": "numpy.ndarray", "shape": list(value.shape),
                        "dtype": str(value.dtype), "strides": list(value.strides)}
            return {"type": type(value).__name__}

        def scalar(value):
            return value if type(value) in (str, int, float, bool, type(None)) else {
                "type": type(value).__name__}

        def state_meta(state):
            if state is None:
                return {"type": "None", "allocation": "inside _call"}
            fields = vars(state) if hasattr(state, "__dict__") else {}
            result = {"type": type(state).__name__, "id": id(state)}
            for key in ("batch_size", "n_kv_heads", "head_dim", "max_tokens",
                        "cached_tokens"):
                result[key] = scalar(fields.get(key))
            for key in ("k_cache", "v_cache"):
                result[key] = array_meta(fields.get(key))
            return result

        sequence = itertools.count(1)

        @functools.wraps(original)
        def traced(self, x, state, step):
            call_id = next(sequence)
            caller = traceback.extract_stack(limit=3)[0]
            fields = vars(self)
            emit("ENTER", call_id=call_id, step=scalar(step),
                 caller={"file": caller.filename, "line": caller.lineno,
                         "function": caller.name}, input=array_meta(x),
                 state=state_meta(state), block_id=id(self),
                 block={key: scalar(fields.get(key)) for key in (
                     "d_model", "n_heads", "n_kv_heads", "head_dim",
                     "intermediate", "numeric_mode")})
            try:
                result = original(self, x, state, step)
            except BaseException as exc:
                emit("RAISE", call_id=call_id, exception=type(exc).__name__,
                     message=str(exc), state=state_meta(state))
                raise
            emit("RETURN", call_id=call_id, output=array_meta(result),
                 state=state_meta(state))
            return result

        block_type._call = traced
        sys.argv = [str(test)]
        emit("TEST_ENTER")
        try:
            runpy.run_path(str(test), run_name="__main__")
        except SystemExit as exc:
            emit("TEST_EXIT", code=scalar(exc.code))
            raise
        except BaseException as exc:
            emit("TEST_RAISE", exception=type(exc).__name__, message=str(exc))
            raise
        else:
            emit("TEST_RETURN")
    finally:
        if original is not None:
            block_type._call = original
        sys.argv = old_argv
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
