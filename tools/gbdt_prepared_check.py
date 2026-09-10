#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build and verify reusable native datasets in all numeric modes."""
import argparse
from pathlib import Path
import statistics
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gbdt-prepared-") as tmp:
        for mode in ("fast", "identical", "deterministic"):
            binary = str(Path(tmp) / mode)
            cmd = ["tools/with_build_lock.sh", "pixi", "run", "mojo", "build", "-I", "."]
            if mode != "fast":
                cmd += ["-D", f"MOJOLEARN_NUMERIC_{mode.upper()}=1"]
            cmd += ["checks/gbdt_prepared_check.mojo", "-o", binary]
            run(cmd, out / f"{mode}.build.log")
            run(["tools/with_build_lock.sh", binary], out / f"{mode}.check.log")
            content = (out / f"{mode}.check.log").read_text()
            if f"numeric_mode {mode.upper()}" not in content or "PASS prepared comparisons 72 owned snapshot and invalid bootstrap" not in content:
                raise RuntimeError(f"{mode}: missing mode/completeness witness")
            print(f"PASS {mode}: 72 full comparisons, owned snapshot, invalid bootstrap", flush=True)
            if mode == "fast":
                run(["tools/with_build_lock.sh", "env", "MOJOLEARN_PREPARED_BENCH=1", binary], out / "fast.bench.log")
                values = {"0": [], "1": []}
                for line in (out / "fast.bench.log").read_text().splitlines():
                    fields = line.split()
                    if fields[:1] == ["fit_ms"]:
                        values[fields[1]].append(float(fields[3]))
                if any(len(v) != 6 for v in values.values()):
                    raise RuntimeError("incomplete timing samples")
                before, after = [statistics.median(values[k]) for k in ("0", "1")]
                print(f"FAST repeated fits median ms {before:.3f} -> {after:.3f}; {before/after:.3f}x; preparation excluded", flush=True)


def run(cmd, path):
    with path.open("w") as log:
        subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)


if __name__ == "__main__":
    main()
