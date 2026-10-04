#!/usr/bin/env python3
"""Quality-only eigh cache admission and separate manifest-gated M3 timing."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

SOURCE = "14764dbb846b882443858cff3c05c3d95b1b9876"
REPAIRED = "131a0d78a74146815aca44e2b23e445f64daf448"
DEFINE = "MOJOLEARN_EIGH_TANGENT_CACHE"
BIND = "x_decomp"
METRICS = ("relative_residual", "max_eigenvalue_error", "orthogonality_error")
SMALL = {f"{kind}:{n}" for n in (31, 128, 257)
         for kind in ("board", "indefinite", "repeated", "gram")}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise SystemExit(message)


def checked_metrics(path, expected, strict=True):
    data = json.loads(path.read_text())
    require(set(data) == expected, "fixture mismatch: " + str(path))
    for case, metrics in data.items():
        require(set(metrics) == set(METRICS), "metric mismatch: " + case)
        require(all(math.isfinite(x) and 0 <= x and (not strict or x <= 2e-4) for x in metrics.values()),
                "invalid residual/eigenvalue/orthogonality: " + case)
    return data


def captured_metrics(out, name, expected, strict=True):
    """Validate saved metrics against the complete checker output, no GPU run."""
    data = checked_metrics(out / (name + ".json"), expected, strict=strict)
    text = (out / (name + ".log")).read_text()
    require("Traceback (most recent call last)" not in text, "traceback in " + name)
    rows = [json.loads(line[len("EIGH-QUALITY "):]) for line in text.splitlines()
            if line.startswith("EIGH-QUALITY ")]
    require(len(rows) == len(expected) and {r["case"] for r in rows} == expected,
            "incomplete checker log: " + name)
    for row in rows:
        require({k: row[k] for k in METRICS} == data[row["case"]],
                "JSON/log mismatch: " + name)
        require(row["status"] in ("OK", "FAIL"), "unknown checker status")
        if strict:
            require(row["status"] == "OK", "candidate/reference checker failure: " + name)
        elif row["status"] != "OK":
            print("EIGH-BASELINE-FAILURE " + json.dumps(row, sort_keys=True), flush=True)
    return data


def compare(candidate, reference, label):
    for case, metrics in candidate.items():
        for metric, value in metrics.items():
            ref = reference[case][metric]
            require(value <= max(ref * 1.1, ref + 5e-8),
                    f"EIGH-CACHE-QUALITY FAIL {label} {case} {metric}: {value} vs {ref}")
    print("EIGH-CACHE-QUALITY PASS comparison=" + label, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mode", choices=("quality", "resume", "timing"))
    ap.add_argument("quality_tag")
    ap.add_argument("timing_tag", nargs="?")
    ap.add_argument("--repaired-small", type=Path,
                    default=Path.home() / "mq/out/gap26-eigh-rayleigh-quality/B.json")
    ap.add_argument("--repaired-board", type=Path)
    args = ap.parse_args()
    require(all(re.fullmatch(r"[A-Za-z0-9_-]+", t) for t in
                (args.quality_tag, args.timing_tag or "unused")), "invalid tag")
    require(bool(args.timing_tag) == (args.mode == "timing"), "timing mode needs timing_tag only")
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    home = Path.home()
    py = home / "board-0834/cache/venv/bin/python"
    validator = home / "mq/verified_arms.py"
    checker = root / "tools/apple_fast_eigh_quality.py"
    require(py.is_file() and (root / "python/mojolearn/_mojolearn.so").is_file(),
            "provision FAST base binding and board venv first")
    env = os.environ.copy()
    env.update(MOJOLEARN_VENDOR="apple", MOJOLEARN_NUMERIC_MODE="fast",
               MOJOLEARN_BENCH_INSTALLED="0", PYTHONPATH=str(root / "python"),
               OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", AB_MULTI_RUN="0")
    compiled = home / "mq/verified-arms" / SOURCE / BIND
    signature = {"source": SOURCE, "repaired_source": REPAIRED, "define": DEFINE,
                 "manifest": sha(compiled / "manifest.json"), "checker": sha(checker),
                 "helper": sha(Path(__file__).resolve())}
    out = home / "mq/out" / (args.quality_tag + "-quality")
    marker = out / "PASS.json"
    if args.mode == "timing":
        require(marker.is_file(), "SKIP: no complete small+4096 quality PASS")
        evidence = json.loads(marker.read_text())
        require(evidence["signature"] == signature, "SKIP: quality source/checker/manifest mismatch")
        for name, digest in evidence["artifacts"].items():
            require(sha(out / name) == digest, "SKIP: changed quality artifact " + name)
        cmd = [str(py), str(validator), SOURCE, BIND, DEFINE, args.timing_tag,
               "bash", "tools/afc_ab_def.sh", args.timing_tag, BIND, "eigh",
               "synthetic", "1", "1", "", "-D " + DEFINE]
        os.execve(str(py), cmd, env)

    # Existing repaired JSON is manager-supplied historical quality evidence.
    # It is copied and hashed in the receipt, never modified or re-timed.
    repaired_small = checked_metrics(args.repaired_small, SMALL)
    repaired_dir = home / "mq/verified-arms" / REPAIRED / BIND
    repaired_manifest = None
    if args.repaired_board:
        checked_metrics(args.repaired_board, {"board:4096"})
    else:
        repaired_manifest = json.loads((repaired_dir / "manifest.json").read_text())
        require(repaired_manifest["source_sha"] == REPAIRED
                and repaired_manifest["binding"] == BIND
                and repaired_manifest["numeric_mode"] == "fast"
                and repaired_manifest["defines_B"] == "-D MOJOLEARN_EIGH_FAST_TANGENT"
                and sha(repaired_dir / "B.so") == repaired_manifest["hashes"]["B"],
                "repaired quality binding manifest/hash mismatch")
        # Running the old binary is deliberate, quality-only. Its checker
        # and linalg import surface must still match; kernel source differs
        # because this checkout contains the new experiment.
        subprocess.run(["git", "diff", "--quiet", REPAIRED + "..HEAD", "--",
                        "tools/apple_fast_eigh_quality.py", "python/mojolearn/linalg.py",
                        "python/mojolearn/_backend.py"], check=True)
    if args.mode == "resume":
        require(out.is_dir() and not marker.exists(), "resume needs existing incomplete quality directory")
        # Legacy helper did not write a start receipt. Manager explicitly
        # authorizes these captured artifacts; verify source-scoped staged
        # binaries, both manifests, unchanged checker, matching log/JSON,
        # then freeze all inherited bytes before any new GPU work.
        current = json.loads((compiled / "manifest.json").read_text())
        staged_existing = home / "afc-def" / args.quality_tag
        for arm in ("A", "B"):
            require(sha(staged_existing / (arm + ".so")) == current["hashes"][arm],
                    "resume staged binary mismatch: " + arm)
        require(sha(out / "repaired-small.json") == sha(args.repaired_small),
                "resume repaired-small reference mismatch")
        require(sha(out / "repaired-manifest.json") == sha(repaired_dir / "manifest.json"),
                "resume repaired manifest mismatch")
        subprocess.run(["git", "diff", "--quiet", SOURCE + "..HEAD", "--",
                        "tools/apple_fast_eigh_quality.py"], check=True)
        for name, expected, strict in (("A-small", SMALL, False), ("B-small", SMALL, True),
                                       ("repaired-board", {"board:4096"}, True),
                                       ("A-board", {"board:4096"}, False)):
            captured_metrics(out, name, expected, strict=strict)
        require(not (out / "B-board.json").exists() and not (out / "B-board.log").exists(),
                "resume only supports missing B-board; never repeat a captured candidate")
        resume_receipt = out / "RESUME.json"
        inherited = {p.name: sha(p) for p in out.iterdir() if p.is_file()}
        require(not resume_receipt.exists(), "resume already claimed; inspect previous attempt")
        resume_receipt.write_text(json.dumps({"signature": signature, "inherited": inherited,
            "provenance": "manager-authorized legacy capture; validated staged binary and artifact hashes"}, indent=2) + "\n")
    else:
        out.mkdir(parents=True, exist_ok=False)
        shutil.copy2(args.repaired_small, out / "repaired-small.json")
    subprocess.run([str(py), str(validator), SOURCE, BIND, DEFINE,
                    args.quality_tag, "--stage-only"], check=True, env=env)
    staged = home / "afc-def" / args.quality_tag
    dest = root / "python/mojolearn/_mojolearn_x_decomp.so"

    def install(so):
        shutil.copy2(so, dest.with_suffix(".so.next"))
        os.replace(dest.with_suffix(".so.next"), dest)

    def probe(name, board=False, baseline=False):
        cmd = [str(py), str(checker), "--output", str(out / (name + ".json"))]
        if board:
            cmd += ["--sizes", "4096", "--kinds", "board"]
        with (out / (name + ".log")).open("w") as log:
            run = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)
        require(run.returncode == 0 or (baseline and run.returncode == 1),
                "quality checker failed; inspect " + str(out / (name + ".log")))
        return captured_metrics(out, name, {"board:4096"} if board else SMALL, strict=not baseline)

    small = {}
    for arm in ("A", "B"):
        if args.mode == "resume":
            small[arm] = captured_metrics(out, arm + "-small", SMALL, strict=arm != "A")
        else:
            install(staged / (arm + ".so"))
            small[arm] = probe(arm + "-small", baseline=arm == "A")
    compare(small["B"], small["A"], "small-vs-current-main")
    compare(small["B"], repaired_small, "small-vs-repaired")
    if args.mode == "resume":
        repaired_board = captured_metrics(out, "repaired-board", {"board:4096"})
    elif args.repaired_board:
        shutil.copy2(args.repaired_board, out / "repaired-board.json")
        repaired_board = checked_metrics(out / "repaired-board.json", {"board:4096"})
    else:
        install(repaired_dir / "B.so")
        repaired_board = probe("repaired-board", board=True)
        shutil.copy2(repaired_dir / "manifest.json", out / "repaired-manifest.json")
    board = {}
    for arm in ("A", "B"):
        if args.mode == "resume" and arm == "A":
            board[arm] = captured_metrics(out, "A-board", {"board:4096"}, strict=False)
        else:
            install(staged / (arm + ".so"))
            board[arm] = probe(arm + "-board", board=True, baseline=arm == "A")
    compare(board["B"], board["A"], "4096-vs-current-main")
    compare(board["B"], repaired_board, "4096-vs-repaired")
    artifacts = {p.name: sha(p) for p in out.iterdir()
                 if p.is_file() and p.suffix in (".json", ".log")}
    evidence = {"signature": signature, "artifacts": artifacts,
                "gate": "candidate all 3 metrics per fixture <= max(reference*1.1, reference+5e-8) against both references; candidate finite <=2e-4 and sorted; main failure retained as reference evidence"}
    tmp = marker.with_suffix(".json.next")
    tmp.write_text(json.dumps(evidence, indent=2) + "\n")
    os.replace(tmp, marker)
    print("EIGH-CACHE-QUALITY-PAIR PASS source=" + SOURCE + " logs=" + str(out), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
