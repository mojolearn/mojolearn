"""Full-model identity gate; launch-logged correctness runs, never timings."""
from pathlib import Path
import os
import subprocess
import sys

root = Path(__file__).resolve().parents[2]
os.chdir(root)
if os.environ.get("MOJOLEARN_BUILD_LOCK_HELD") != "1":
    raise SystemExit("Use tools/check_rf_column_tiles.sh to hold the shared build lock")
out = Path(sys.argv[1] if len(sys.argv) > 1 else "bench/results/rf_column_tiles_gate").resolve()
out.mkdir(parents=True, exist_ok=True)
source = "ensemble/checks/rf_column_tiles_fingerprint.mojo"
subprocess.run(["pixi", "run", "mojo", "--version"], stdout=(out / "toolchain.txt").open("w"), check=True)
cases = {"c2_boot", "c5_no_boot", "c5_weighted", "c2_weighted_boot", "c17_capacity", "c5_search", "c2_bins256", "c2_leaf_cap", "reg_boot", "reg_weighted", "reg_weighted_boot"}
summary = []
for mode in ("fast", "deterministic", "identical"):
    expected = None
    for arm in ("reference", "columns2", "columns4"):
        name = f"{mode}-{arm}"
        binary = out / name
        command = ["pixi", "run", "mojo", "build", "-I", "."]
        if mode != "fast":
            command += ["-D", f"MOJOLEARN_NUMERIC_{mode.upper()}=1"]
        if arm != "reference":
            command += ["-D", f"MOJOLEARN_RF_HIST_{arm.upper()}=1"]
        command += [source, "-o", str(binary)]
        print("BUILD", name, flush=True)
        with (out / f"{name}.build.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        launches = out / f"{name}.launches.log"
        launches.write_text("")
        print("RUN", name, flush=True)
        with (out / f"{name}.log").open("w") as log:
            subprocess.run([str(binary)], env={**os.environ, "RF_LAUNCH_LOG": str(launches)}, stdout=log, stderr=subprocess.STDOUT, check=True)
        lines = (out / f"{name}.log").read_text().splitlines()
        assert f"numeric_mode {mode.upper()}" in lines, (name, "wrong mode")
        assert f"tile2 {'True' if arm == 'columns2' else 'False'} tile4 {'True' if arm == 'columns4' else 'False'}" in lines, (name, "wrong flags")
        hashes = dict(line.split()[1:] for line in lines if line.startswith("fingerprint "))
        assert set(hashes) == cases, (name, hashes)
        if expected is None:
            expected = hashes
        assert hashes == expected, (name, "model/prediction fingerprint mismatch", hashes, expected)
        case_launches = {}
        current = None
        for line in launches.read_text().splitlines():
            if line.startswith("CASE_BEGIN "):
                current = line.split()[1]
                case_launches[current] = []
            elif line.startswith("CASE_END "):
                current = None
            elif current:
                case_launches[current].append(line)
        assert set(case_launches) == cases
        reach = {case: sum(line.startswith("histogram_binned_columns") for line in logs) for case, logs in case_launches.items()}
        if arm == "reference":
            assert not any(reach.values()), (name, "reference unexpectedly tiled")
        else:
            for case in cases - {"c17_capacity", "c5_search"}:
                assert reach[case] > 0, (name, case, "candidate did not run")
            for case in ("c17_capacity", "c5_search"):
                assert reach[case] == 0, (name, case, "fallback unexpectedly tiled")
        message = f"PASS {name}: {len(hashes)} full-model/prediction fingerprints; tile launches {reach}"
        print(message, flush=True)
        summary.append(message)
        (out / "summary.txt").write_text("\n".join(summary) + "\n")
print("PASS all three numeric modes", flush=True)
