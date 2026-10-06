#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""Inspect and execute source-frozen performance experiment recipes.

This is experiment orchestration, not ML runtime computation. A successful
process is recorded as COMPLETED; only a harness's declared gate evidence can
admit timing. The tool never promotes defaults or updates benchmark boards.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
EXPECTED = tuple(
    [f"I{n:02d}" for n in range(1, 25)]
    + [f"A{n:02d}" for n in range(1, 9)]
    + [f"N{n:02d}" for n in range(1, 9)]
    + [f"F{n:02d}" for n in range(1, 21)]
    + [f"C{n:02d}" for n in range(1, 61)]
)
STATUSES = {"source_draft", "source_ready", "build_passed", "blocked_toolchain", "blocked_prerequisite"}
STAGES = {"build": "build_argv", "validate": "validation_argv", "time": "timing_argv", "run": "run_argv"}


class ExperimentError(ValueError):
    """An invalid recipe or evidence prerequisite."""


def mode_for(idea: str) -> str:
    if idea not in EXPECTED:
        raise ExperimentError(f"Unknown idea: {idea}")
    return "fast" if idea.startswith("F") else "identical"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(root: Path, name: str) -> Path:
    path = root / name
    if not name or Path(name).is_absolute() or not path.resolve().is_relative_to(root.resolve()):
        raise ExperimentError(f"Source path must remain inside the repository: {name!r}")
    return path


def require_string_list(value: Any, name: str, *, allow_empty: bool = True) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value):
        raise ExperimentError(f"{name} must be an array of nonempty strings")
    if not allow_empty and not value:
        raise ExperimentError(f"{name} must not be empty")
    return value


def read_manifest(path: Path, root: Path = ROOT) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Cannot read {path}: {exc}") from exc
    if not isinstance(record, dict) or record.get("schema") != 1:
        raise ExperimentError(f"{path}: expected schema 1 object")
    idea = record.get("id")
    if idea not in EXPECTED or path.parent.name != idea:
        raise ExperimentError(f"{path}: ID does not match its idea directory")
    if record.get("mode") != mode_for(idea):
        raise ExperimentError(f"{idea}: incorrect numeric mode")
    if not isinstance(record.get("title"), str) or not record["title"].strip():
        raise ExperimentError(f"{idea}: title is required")
    if record.get("status") not in STATUSES:
        raise ExperimentError(f"{idea}: unsupported implementation status")
    vendors = require_string_list(record.get("vendors"), f"{idea}.vendors", allow_empty=False)
    allowed = {"apple"} if record["mode"] == "fast" else {"nvidia", "amd", "apple", "host"}
    if len(vendors) != len(set(vendors)) or not set(vendors).issubset(allowed):
        raise ExperimentError(f"{idea}: vendors do not match its numeric mode")
    blocked = record["status"].startswith("blocked_")
    if blocked:
        if not isinstance(record.get("blocker"), str) or not record["blocker"].strip():
            raise ExperimentError(f"{idea}: blocked status needs a factual blocker")
    elif record.get("blocker") is not None:
        raise ExperimentError(f"{idea}: ready status cannot carry a blocker")
    paths = require_string_list(record.get("implementation_paths"), f"{idea}.implementation_paths")
    if not paths and not blocked:
        raise ExperimentError(f"{idea}: ready status requires an executable implementation")
    for name in paths:
        source = inside(root, name)
        if source.suffix not in {".mojo", ".py", ".sh", ".cpp", ".cc", ".c", ".metal"}:
            raise ExperimentError(f"{idea}: implementation path is not executable source: {name}")
        if not source.is_file():
            raise ExperimentError(f"{idea}: missing implementation source: {name}")
    for key in ("validation_paths", "candidate_defines", "baseline_defines", "depends_on", "quality_gates"):
        values = require_string_list(record.get(key), f"{idea}.{key}")
        if key == "validation_paths":
            for name in values:
                if not inside(root, name).is_file():
                    raise ExperimentError(f"{idea}: missing validation source: {name}")
        elif key == "depends_on":
            if any(value not in EXPECTED or value == idea for value in values):
                raise ExperimentError(f"{idea}: invalid dependency")
        elif key.endswith("_defines"):
            if any(not re.fullmatch(r"MOJOLEARN_[A-Z0-9_]+(?:=[A-Za-z0-9_.+-]+)?", x) for x in values):
                raise ExperimentError(f"{idea}: invalid compiler define")
    for key in ("build_argv", "run_argv"):
        require_string_list(record.get(key), f"{idea}.{key}")
    for key in ("validation_argv", "timing_argv", "baseline_build_argv"):
        if key in record:
            require_string_list(record[key], f"{idea}.{key}")
    if not isinstance(record.get("build_uses_compile_slot", False), bool):
        raise ExperimentError(f"{idea}: build_uses_compile_slot must be boolean")
    if not isinstance(record.get("paired_build", False), bool):
        raise ExperimentError(f"{idea}: paired_build must be boolean")
    if record.get("output_kind", "file") not in {"file", "directory"}:
        raise ExperimentError(f"{idea}: invalid output_kind")
    if not isinstance(record.get("timing_contract"), str) or not record["timing_contract"].strip():
        raise ExperimentError(f"{idea}: timing contract is required")
    return record


def catalog(root: Path = ROOT) -> tuple[dict[str, dict[str, Any]], list[str]]:
    records: dict[str, dict[str, Any]] = {}
    problems: list[str] = []
    paths = list((root / "experiments/performance_ideas").glob("*/manifest.json"))
    paths += list((root / "experiments/classical_identical_ideas").glob("*/manifest.json"))
    for path in sorted(paths):
        try:
            record = read_manifest(path, root)
            records[record["id"]] = record
        except ExperimentError as exc:
            problems.append(str(exc))
    return records, problems


def validate_dependencies(records: dict[str, dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    visited: set[str] = set()
    active: set[str] = set()

    def visit(idea: str) -> None:
        if idea in active:
            errors.append(f"Cyclic dependency involving {idea}")
            return
        if idea in visited:
            return
        active.add(idea)
        for dependency in records[idea]["depends_on"]:
            if dependency not in records:
                errors.append(f"{idea}: missing dependency {dependency}")
            else:
                visit(dependency)
        active.remove(idea)
        visited.add(idea)

    for idea in records:
        visit(idea)
    return errors


def git(root: Path, *args: str) -> str:
    result = subprocess.run(["git", *args], cwd=root, text=True, capture_output=True)
    if result.returncode:
        raise ExperimentError(result.stderr.strip() or "git command failed")
    return result.stdout.strip()


def mojo_path(root: Path = ROOT) -> str:
    configured = os.environ.get("MOJOLEARN_MOJO")
    if configured:
        return configured
    local = root / ".pixi/envs/default/bin/mojo"
    if local.is_file():
        return str(local)
    located = shutil.which("mojo")
    if located:
        return located
    shared = Path.home() / "CascadeProjects/mojolearn/.pixi/envs/default/bin/mojo"
    return str(shared) if shared.is_file() else "mojo"


def command_for(record: dict[str, Any], stage: str, vendor: str, output: Path, source: str,
                root: Path = ROOT, arm: str = "candidate") -> list[str]:
    if vendor not in record["vendors"]:
        raise ExperimentError(f"{record['id']}: vendor {vendor} is outside this recipe")
    if record["status"].startswith("blocked_"):
        raise ExperimentError(f"{record['id']}: {record['blocker']}")
    if stage == "build" and arm == "baseline" and record.get("paired_build", False):
        raise ExperimentError(f"{record['id']}: this recipe builds both arms once; use the default build arm")
    key = "baseline_build_argv" if stage == "build" and arm == "baseline" else STAGES[stage]
    command = record.get(key, [])
    if not command:
        raise ExperimentError(f"{record['id']}: no executable {stage} recipe")
    variables = {
        "repo": str(root), "python": sys.executable, "mojo": mojo_path(root),
        "compile_slot": str(Path.home() / "mojolearn-evidence/compile_slot.sh"),
        "vendor": vendor, "output": str(output), "source_sha": source, "mode": record["mode"], "arm": arm, "configuration": record.get("configuration", "default"),
    }
    try:
        return [argument.format_map(variables) for argument in command]
    except (KeyError, ValueError) as exc:
        raise ExperimentError(f"{record['id']}: invalid command template: {exc}") from exc


def select_configuration(record: dict[str, Any], name: str | None) -> dict[str, Any]:
    """Select a programmed C-card sub-arm/interaction without changing incumbents.

    NOT COMPILED — NOT TESTED — IDENTITY NOT VERIFIED — QUALITY NOT VERIFIED — NOT MEASURED.
    Selection changes compile defines and the full-workload coverage requirement.
    It never imports a candidate or establishes runtime reach.
    """
    if not record["id"].startswith("C"):
        if name is not None:
            raise ExperimentError("--configuration is a classical C-card selector")
        return record
    if record["status"].startswith("blocked_"):
        raise ExperimentError(f"{record['id']}: {record['blocker']}")
    selected = name or record.get("default_configuration")
    configs = record.get("configurations", {})
    if selected not in configs:
        raise ExperimentError(f"{record['id']}: configuration is pending or unknown: {selected}")
    cfg = configs[selected]
    if not cfg.get("selectable", False):
        raise ExperimentError(f"{record['id']}/{selected}: source integration pending")
    return {**record, "configuration": selected,
            "candidate_defines": cfg["candidate_defines"], "baseline_defines": cfg["baseline_defines"]}


def admitted_quality(path: Path, record: dict[str, Any], source: str, vendor: str, manifest_digest: str) -> None:
    try:
        evidence = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Cannot read quality evidence: {exc}") from exc
    if not isinstance(evidence, dict):
        raise ExperimentError("Quality evidence must be an object")
    expected = {
        "id": record["id"], "mode": record["mode"], "vendor": vendor,
        "source_sha": source, "manifest_sha256": manifest_digest, "status": "PASS",
    }
    for key, value in expected.items():
        if evidence.get(key) != value:
            raise ExperimentError(f"Quality evidence mismatches {key}")
    gates = evidence.get("gates")
    if not isinstance(gates, dict) or not record["quality_gates"]:
        raise ExperimentError("Quality evidence needs the recipe's declared gates")
    for name in record["quality_gates"]:
        if gates.get(name) != "PASS":
            raise ExperimentError(f"Quality gate has not passed: {name}")
    if record["id"].startswith("C"):
        if evidence.get("configuration") != record.get("configuration"):
            raise ExperimentError("Quality receipt does not name the selected classical configuration")
        if evidence.get("candidate_defines") != record["candidate_defines"]:
            raise ExperimentError("Quality receipt does not match the selected classical defines")
    # The actual harness is responsible for task metrics and identity witnesses.
    # Merely exiting zero is never converted into a quality PASS here.


def require_queue_host(stage: str, record: dict[str, Any], vendor: str) -> None:
    """Device execution belongs to the established queues, not this laptop."""
    if stage == "build":
        return
    if os.environ.get("MOJOLEARN_PERFORMANCE_QUEUE_JOB") != "1":
        raise ExperimentError("Device validation and execution must run as a queued job")
    if vendor == "host" and stage in {"time", "run"}:
        raise ExperimentError("The host column supplies verification, not performance timing")
    if sys.platform == "darwin" and vendor not in {"apple", "host"}:
        raise ExperimentError("AMD and NVIDIA device execution requires its matching Linux queue")
    if vendor == "apple" and sys.platform != "darwin":
        raise ExperimentError("Apple device execution requires its matching macOS queue")
    if sys.platform == "darwin":
        brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                               text=True, capture_output=True, check=True).stdout.strip()
        expected = "Apple M3 Ultra" if record["mode"] == "fast" else "Apple M2"
        if expected not in brand:
            raise ExperimentError(f"This recipe requires the queued {expected} host")
        if record["mode"] == "identical" and stage in {"time", "run"}:
            raise ExperimentError("Apple is an identity witness, not an IDENTICAL timing target")


def execute(record: dict[str, Any], path: Path, stage: str, vendor: str, output: Path,
            evidence_dir: Path, quality: Path | None, timeout: float,
            root: Path = ROOT, arm: str = "candidate") -> dict[str, Any]:
    if any(p.resolve().is_relative_to(root.resolve()) for p in (output, evidence_dir)):
        raise ExperimentError("Save run evidence outside the source worktree")
    if git(root, "status", "--porcelain"):
        raise ExperimentError("Commit the source before executing a frozen experiment")
    require_queue_host(stage, record, vendor)
    source = git(root, "rev-parse", "HEAD")
    manifest_digest = digest(path)
    if stage in {"time", "run"}:
        if quality is None:
            raise ExperimentError("A source-matched quality receipt is required before execution")
        admitted_quality(quality, record, source, vendor, manifest_digest)
    command = command_for(record, stage, vendor, output, source, root, arm)
    if stage == "build" and not record.get("build_uses_compile_slot", False):
        semaphore = Path.home() / "mojolearn-evidence/compile_slot.sh"
        if not semaphore.is_file():
            raise ExperimentError("The required compile-slot semaphore is unavailable")
        command = ["bash", str(semaphore), *command]
    evidence_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    if record.get("output_kind") == "directory":
        output.mkdir(parents=True, exist_ok=True)
    receipt_path = evidence_dir / f"{record['id']}-{vendor}-{stage}-{arm}-receipt.json"
    log_path = evidence_dir / f"{record['id']}-{vendor}-{stage}-{arm}.log"
    if receipt_path.exists() or log_path.exists():
        raise ExperimentError("Choose a fresh evidence directory; this run's paths already exist")
    environment = os.environ.copy()
    environment["MOJOLEARN_COMPILE_JOBS"] = "1"
    environment["MOJOLEARN_NUMERIC_MODE"] = record["mode"]
    environment["MOJOLEARN_VENDOR"] = vendor
    if quality is not None and record["id"].startswith("C"):
        environment["MOJOLEARN_CLASSICAL_QUALITY_RECEIPT"] = str(quality.resolve())
    if stage == "build":
        defines = record["baseline_defines" if arm == "baseline" else "candidate_defines"]
        flags = " ".join(
            shlex.quote(value) for flag in defines for value in ("-D", flag)
        )
        # Every declared binding builder accepts this input. Some also append
        # EXTRA_DEFINES; populating both defines the same Mojo name twice.
        environment["MOJOLEARN_BUILD_EXTRA_DEFINES"] = ""
        environment["MOJOLEARN_MOJO_BUILD_FLAGS"] = flags
        # Build-only execution must not invoke a builder's device smoke gate.
        # Those checks belong to the explicit queued validation stage.
        environment["MOJOLEARN_SKIP_BUILD_GATE"] = "1"
    compiler = Path(mojo_path(root))
    if compiler.is_file() and (compiler.parent.parent / "share/max/modular.cfg").is_file():
        environment["MODULAR_HOME"] = str(compiler.parent.parent / "share/max")
        environment["PATH"] = str(compiler.parent) + os.pathsep + environment.get("PATH", "")
    receipt: dict[str, Any] = {
        "schema": 1, "id": record["id"], "mode": record["mode"], "vendor": vendor,
        "stage": stage, "arm": arm, "source_sha": source, "manifest_sha256": manifest_digest,
        "paired_build": stage == "build" and record.get("paired_build", False),
        "argv": command, "log": str(log_path), "status": "RUNNING",
        "quality_receipt": str(quality) if quality else None,
        "configuration": record.get("configuration"),
        "candidate_defines": record["candidate_defines"],
        "baseline_defines": record["baseline_defines"],
    }
    receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    start = time.monotonic()
    process = None
    try:
        with log_path.open("x") as log:
            process = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
            returncode = process.wait(timeout=timeout)
        receipt["returncode"] = returncode
        receipt["status"] = "COMPLETED" if returncode == 0 else "FAILED"
    except subprocess.TimeoutExpired:
        receipt.update(status="TIMEOUT", returncode=124)
    except OSError as exc:
        receipt.update(status="FAILED", returncode=127, error=str(exc))
    except KeyboardInterrupt:
        receipt.update(status="INTERRUPTED", returncode=130)
    finally:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        receipt["elapsed_seconds"] = round(time.monotonic() - start, 6)
        if git(root, "rev-parse", "HEAD") != source or digest(path) != manifest_digest or git(root, "status", "--porcelain"):
            receipt.update(status="SOURCE_CHANGED", returncode=125)
        receipt_path.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def parser() -> argparse.ArgumentParser:
    argument = argparse.ArgumentParser(
        description=__doc__,
        epilog="Neural source ideas: performance_ideas.py neural list|show|plan|build-plan|queue-template ...",
    )
    argument.add_argument("--root", type=Path, default=ROOT)
    commands = argument.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="List every idea and its honest implementation state")
    listing.add_argument("--mode", choices=["identical", "fast"])
    listing.add_argument("--json", action="store_true")
    check = commands.add_parser("check", help="Validate source references and dependencies")
    check.add_argument("--require-all", action="store_true")
    check.add_argument("--require-ready", action="store_true")
    for name in ["plan", "execute"]:
        command = commands.add_parser(name)
        command.add_argument("id", choices=EXPECTED)
        command.add_argument("--configuration", help="Classical candidate sub-arm or interaction name")
        command.add_argument("--workload-recipe", type=Path, help="Resolved full-dataset recipe for classical C cards")
        command.add_argument("--stage", choices=STAGES, required=True)
        command.add_argument("--vendor", choices=["nvidia", "amd", "apple", "host"], required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--arm", choices=["baseline", "candidate"], default="candidate")
        if name == "execute":
            command.add_argument("--evidence", type=Path, required=True)
            command.add_argument("--quality-receipt", type=Path)
            command.add_argument("--timeout", type=float, default=3600)
    return argument


def main(argv: list[str] | None = None) -> int:
    # The neural catalog keeps its per-lane source records, numerical-version
    # rules and full-operation queue adapter. Share that implementation instead
    # of copying 60 manifests and letting their controls drift from the kernels.
    arguments = sys.argv[1:] if argv is None else argv
    if arguments and arguments[0] == "neural":
        from neural_identical_ideas import main as neural_main
        return neural_main(arguments[1:])
    args = parser().parse_args(arguments)
    root = args.root.resolve()
    records, errors = catalog(root)
    errors.extend(validate_dependencies(records))
    if args.command == "check":
        missing = [idea for idea in EXPECTED if idea not in records]
        if args.require_all and missing:
            errors.append("Missing ideas: " + ", ".join(missing))
        if args.require_ready:
            blocked = [idea for idea, r in records.items() if r["status"].startswith("blocked_")]
            if blocked:
                errors.append("Unimplemented prerequisites: " + ", ".join(blocked))
        print(json.dumps({"records": len(records), "missing": missing, "errors": errors}, indent=2))
        return int(bool(errors))
    if errors:
        raise ExperimentError("; ".join(errors))
    if args.command == "list":
        rows = [{"id": idea, "mode": mode_for(idea), **records.get(idea, {"status": "missing"})}
                for idea in EXPECTED if args.mode is None or mode_for(idea) == args.mode]
        if args.json:
            print(json.dumps(rows, indent=2))
        else:
            for row in rows:
                print(f"{row['id']} {row['mode']:9} {row['status']:20} {row.get('title', '')}")
        return 0
    if args.id not in records:
        raise ExperimentError(f"No implementation manifest for {args.id}")
    record = select_configuration(records[args.id], args.configuration)
    family = "classical_identical_ideas" if args.id.startswith("C") else "performance_ideas"
    manifest = root / f"experiments/{family}/{args.id}/manifest.json"
    if args.workload_recipe is not None:
        os.environ["MOJOLEARN_CLASSICAL_WORKLOAD_RECIPE"] = str(args.workload_recipe.resolve())
    if args.command == "plan":
        source = git(root, "rev-parse", "HEAD")
        command = command_for(record, args.stage, args.vendor, args.output.resolve(), source, root, args.arm)
        print(json.dumps({"id": args.id, "mode": record["mode"], "source_sha": source,
                          "stage": args.stage, "vendor": args.vendor, "configuration": record.get("configuration"),
                          "candidate_defines": record["candidate_defines"], "baseline_defines": record["baseline_defines"],
                          "argv": command}, indent=2))
        return 0
    if args.timeout <= 0:
        raise ExperimentError("Timeout must be positive")
    receipt = execute(record, manifest, args.stage, args.vendor, args.output.resolve(),
                      args.evidence.resolve(), args.quality_receipt, args.timeout, root, args.arm)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "COMPLETED" else int(receipt["returncode"])


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExperimentError as exc:
        print(f"performance-ideas: {exc}", file=sys.stderr)
        raise SystemExit(2)
