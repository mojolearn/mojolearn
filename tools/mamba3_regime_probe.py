#!/usr/bin/env python3
"""Diagnostic only: repeated same-binary Mamba3 calls with explicit shape order.

Loads explicit retained helper/spec files without overwriting tracked tools. No source changes, compiler invocation, or timing claim.
JSON event lines accompany fixture witnesses; native M3_PHASE lines can be interleaved when a separately
built phase-instrumented library is supplied. That library is a different arm.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time


def emit(kind, **fields):
    print(json.dumps(dict(kind=kind, **fields)), flush=True)


def sha_file(path):
    with open(path, "rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def usage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return dict(user_s=r.ru_utime, system_s=r.ru_stime, minor_faults=r.ru_minflt,
                major_faults=r.ru_majflt, voluntary_switches=r.ru_nvcsw,
                involuntary_switches=r.ru_nivcsw, maxrss_native=r.ru_maxrss)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--harness", type=Path, required=True)
    p.add_argument("--spec", type=Path, required=True, help="retained speed_torch_seq.py with Mamba3 fixture rows")
    p.add_argument("--order", default="narrow,wide,narrow,tiny,narrow,wide")
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--passes", type=int, default=2)
    p.add_argument("--release-output-before-call", action="store_true")
    p.add_argument("--reference-json", type=Path, help="retained mamba-repeat summary.json; enforce its complete output hashes")
    p.add_argument("--telemetry", type=Path, help="fresh file for continuous nvidia-smi CSV; optional")
    args = p.parse_args()
    assert 1 <= args.rounds <= 50 and 1 <= args.passes <= 5
    order = args.order.split(",")
    assert 1 <= len(order) <= 12 and set(order) <= {"tiny", "narrow", "wide"}
    harness = args.harness.resolve(strict=True)
    checkout = Path(__file__).resolve().parents[1]
    fixture_spec_path = args.spec.resolve(strict=True)
    driver_path = (checkout / "bench/speed/seq_speed_main.mojo").resolve(strict=True)
    os.environ["MOJOLEARN_NUMERIC_MODE"] = "identical"
    # Historical helpers derive paths from their own archive directory.
    # Bind the explicit spec first, with its data/source roots overridden to
    # THIS checkout. No tracked module is copied over or edited.
    for path in (checkout / "tools", checkout / "python"):
        sys.path.insert(0, str(path))
    controlled_path = sys.path.copy()
    spec_loader = importlib.util.spec_from_file_location("speed_torch_seq", fixture_spec_path)
    fixture_spec = importlib.util.module_from_spec(spec_loader)
    sys.modules["speed_torch_seq"] = fixture_spec
    spec_loader.loader.exec_module(fixture_spec)
    original_roots = dict(REPO=str(fixture_spec.REPO), DRIVER_MOJO=str(fixture_spec.DRIVER_MOJO))
    fixture_spec.REPO = str(checkout)
    fixture_spec.DRIVER_MOJO = str(driver_path)
    harness_loader = importlib.util.spec_from_file_location("mamba_regime_fixture", harness)
    fixture = importlib.util.module_from_spec(harness_loader)
    harness_loader.loader.exec_module(fixture)
    assert fixture.seqspec is fixture_spec, "historical helper imported a different fixture spec"
    fixture._ROOT = str(checkout)
    # Remove archive-relative paths added during helper import before any
    # public API import; selected backend paths are checked below as well.
    sys.path[:] = controlled_path
    import numpy as np
    import mojolearn._mamba_impl as implementation
    assert Path(implementation.__file__).resolve().is_relative_to(checkout), "wrong Python implementation checkout"
    _, constants = fixture.seqspec.load_shapes()
    rows = fixture.seqspec.py_rows("mamba3")
    chosen = {}
    for row in rows:
        key = "tiny" if row["name"].startswith("lane.") else row["name"].split(".")[0]
        if key in order:
            chosen[key] = row
    assert set(order) <= chosen.keys(), "retained helper lacks requested fixture"
    emit("environment", pid=os.getpid(), python=sys.version, executable=sys.executable,
         numpy=np.__version__, harness_path=str(harness), harness_sha256=sha_file(harness),
         fixture_spec_path=str(fixture_spec_path), fixture_spec_sha256=sha_file(fixture_spec_path),
         original_spec_roots=original_roots, effective_spec_roots=dict(REPO=str(checkout), DRIVER_MOJO=str(driver_path)),
         driver_sha256=sha_file(driver_path), implementation=str(implementation.__file__),
         affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
         order=order, rounds=args.rounds, passes=args.passes,
         release_output_before_call=args.release_output_before_call,
         note="hashing/resource/telemetry are diagnostics outside call timer; no opponent ratio")
    telemetry = None
    telemetry_file = None
    try:
        if args.telemetry:
            telemetry_file = args.telemetry.open("x")
            telemetry = subprocess.Popen([
                "nvidia-smi", "--query-gpu=timestamp,uuid,pstate,clocks.current.sm,clocks.current.memory,power.draw,temperature.gpu,memory.used,utilization.gpu,utilization.memory",
                "--format=csv", "--loop-ms=250"], stdout=telemetry_file, stderr=telemetry_file)
        loaded = {}
        expected = {}
        if args.reference_json:
            reference = json.loads(args.reference_json.read_text())["hashes"]
            expected = {key: reference["seq.mamba3." + row["name"] + ".f32.bin"]
                        for key, row in chosen.items()}
        library = None
        previous = None
        for trial in range(args.passes):
            for position, key in enumerate(order):
                if key not in loaded:
                    row = chosen[key]
                    weights, x, _ = fixture.build_mamba("mamba3", row, "mamba3", row["name"], constants["witness_samples"])
                    block = fixture.make_block("mamba3", weights, row)
                    ext = block._extension()
                    assert int(ext.mamba_numeric_mode()) == 1
                    binary = Path(ext.__file__).resolve()
                    identity = (str(binary), sha_file(binary))
                    assert library is None or library == identity, "binding changed inside process"
                    library = identity
                    emit("fixture", shape=key, binary=identity[0], binary_sha256=identity[1],
                         vendor=str(ext.mamba_vendor()), input_shape=list(x.shape),
                         input_sha256=hashlib.sha256(memoryview(x).cast("B")).hexdigest(),
                         input_address=int(x.ctypes.data), input_mod_2m=int(x.ctypes.data) % (2**21))
                    loaded[key] = block, x
                block, x = loaded[key]
                for call in range(args.rounds):
                    if args.release_output_before_call:
                        previous = None
                    started_wall = time.time_ns()
                    emit("call_begin", trial=trial, position=position, shape=key, call=call, wall_ns=started_wall)
                    before = usage()
                    start = time.perf_counter_ns()
                    output = block.forward(x)
                    elapsed = (time.perf_counter_ns() - start) / 1e6
                    ended_wall = time.time_ns()
                    after = usage()
                    raw = np.asarray(output)
                    digest = hashlib.sha256(memoryview(raw).cast("B")).hexdigest()
                    if key in expected:
                        assert digest == expected[key], f"complete output changed for {key}"
                    else:
                        expected[key] = digest
                    emit("call_end", trial=trial, position=position, shape=key, call=call,
                         wall_ns=ended_wall, elapsed_ms=elapsed, output_sha256=digest,
                         output_address=int(raw.ctypes.data), output_mod_2m=int(raw.ctypes.data) % (2**21),
                         resource_delta={name: after[name] - before[name] for name in before if name != "maxrss_native"},
                         maxrss_native=after["maxrss_native"])
                    previous = output
                    del raw, output
        assert library and sha_file(library[0]) == library[1], "binding file changed during probe"
        emit("pass", output_hashes=expected, binary_sha256=library[1])
    finally:
        if telemetry is not None:
            telemetry.terminate()
            telemetry.wait(timeout=10)
            emit("telemetry_exit", returncode=telemetry.returncode, note="negative15 is expected termination; inspect CSV for unsupported fields/errors")
        if telemetry_file is not None:
            telemetry_file.close()


if __name__ == "__main__":
    main()
