"""Finite native ABI proof; dependent lifecycle steps run in order.

The two independent scenarios run in separate processes. Compilation belongs
to the orchestrator and must produce two separate shared libraries first.
"""
import argparse
import concurrent.futures
import ctypes
import json
from pathlib import Path
import subprocess
import sys


def scenario(provider, consumer, mode):
    # Keep both images pinned until every call and native lease has finished.
    images = [ctypes.CDLL(str(provider)), ctypes.CDLL(str(consumer))]
    signatures = {
        "publish": (0, []), "state": (0, [ctypes.c_int64, ctypes.c_int64]),
        "release": (0, [ctypes.c_int64, ctypes.c_uint64]),
        "query": (1, [ctypes.c_int64, ctypes.c_uint64, ctypes.c_int64]),
        "hold": (1, [ctypes.c_int64, ctypes.c_uint64]),
        "held": (1, [ctypes.c_int64]), "drop": (1, []),
        "schema_refusal": (1, []), "layout_abort": (1, []),
    }
    calls = {}
    for name, (owner, arguments) in signatures.items():
        fn = getattr(images[owner], "native_entry_image_" + name)
        fn.argtypes, fn.restype = arguments, ctypes.c_int64
        calls[name] = fn
    if mode == "layout":
        assert calls["publish"]() == 0
        calls["layout_abort"]()
        raise AssertionError("incompatible SharedSlot layout returned instead of aborting")
    generation = ctypes.c_uint64()
    index = calls["state"](37, ctypes.addressof(generation))
    assert index >= 0
    assert calls["query"](index, generation.value, 3) == -700
    assert calls["publish"]() == calls["publish"]() == 0
    assert calls["query"](index, generation.value, 3) == 3703
    assert calls["hold"](index, generation.value) == 0
    assert calls["release"](index, generation.value) == 0
    assert calls["query"](index, generation.value, 3) == -700
    assert calls["held"](4) == 3704
    newer = ctypes.c_uint64()
    replacement = calls["state"](91, ctypes.addressof(newer))
    assert replacement == index and newer.value != generation.value
    assert calls["query"](replacement, newer.value, 5) == 9105
    assert calls["held"](6) == 3706
    assert calls["schema_refusal"]() == 1
    assert calls["release"](replacement, newer.value) == 0
    assert calls["drop"]() == 0
    assert calls["held"](0) == -700
    print("PASS native cross-image lifecycle", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", type=Path)
    parser.add_argument("consumer", type=Path)
    parser.add_argument("evidence", type=Path)
    parser.add_argument("--scenario", choices=("lifecycle", "layout"))
    args = parser.parse_args()
    if args.scenario:
        scenario(args.provider.resolve(), args.consumer.resolve(), args.scenario)
        return 0
    args.evidence.mkdir(parents=True, exist_ok=True)

    def run(mode):
        log = args.evidence / (mode + ".log")
        with log.open("x") as output:
            try:
                result = subprocess.run(
                    [sys.executable, __file__, str(args.provider.resolve()),
                     str(args.consumer.resolve()), str(args.evidence.resolve()),
                     "--scenario", mode], stdout=output, stderr=subprocess.STDOUT,
                    timeout=120)
                code = result.returncode
            except subprocess.TimeoutExpired:
                return {"scenario": mode, "passed": False, "timeout": True, "log": str(log)}
        body = log.read_text()
        passed = (code == 0 and "PASS native cross-image lifecycle" in body) if mode == "lifecycle" else (
            code < 0 and "mixed binary versions" in body)
        return {"scenario": mode, "passed": passed, "exit_code": code, "log": str(log)}

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(run, ("lifecycle", "layout")))
    passed = all(row["passed"] for row in results)
    report = {"passed": passed, "results": results,
              "scope": "CPU native ABI proof only; no product runtime, GPU, identity or speed claim"}
    (args.evidence / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
