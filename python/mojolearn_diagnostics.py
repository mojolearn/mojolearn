"""Dependency-light diagnostics for the installed MojoLearn distribution.

This module intentionally lives outside ``mojolearn``. Importing a package
submodule first executes ``mojolearn.__init__``, which selects and loads the
native extensions. A diagnostic that cannot run when that load fails is not
a diagnostic for the failure users are most likely to encounter.
"""

from __future__ import annotations

import argparse
import datetime as _datetime
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import textwrap
import zipfile
from pathlib import Path
from typing import Any, Sequence


SCHEMA_VERSION = 1
SAFE_ENVIRONMENT_KEYS = (
    "MOJOLEARN_NUMERIC_MODE",
    "PYTHONHASHSEED",
)
# SET-OR-NOT, NEVER THE VALUE. Each of these is a FILESYSTEM PATH the caller
# chose, so its value routinely contains a username or a home directory --
# exactly the two facts the `privacy` block at the bottom of this file
# promises are omitted. `MOJOLEARN_IDENTITY_TRACE` was in the list above
# until 2026-08-29 and shipped its own value into every bundle, which made
# that promise false. Whether identity tracing is on is the diagnostic fact;
# where the card was written is not.
PRESENCE_ONLY_ENVIRONMENT_KEYS = (
    "MOJOLEARN_IDENTITY_TRACE",
    "MOJOLEARN_IDENTITY_TRACE_DUMP",
    "MOJOLEARN_IDENTITY_TRACE_DIFF",
)
# The numeric tiers, in the order the ladder runs. `fast` is the package
# directory itself and every other tier is a subdirectory of the same name;
# see python/mojolearn/_backend.py, which is what actually loads them.
NUMERIC_TIERS = ("fast", "deterministic", "identical")
RELEVANT_DISTRIBUTIONS = (
    "mojolearn",
    "numpy",
    "mojo",
    "max",
)
COMMAND_TIMEOUT_SECONDS = 12


def _run(command: Sequence[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False}
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=COMMAND_TIMEOUT_SECONDS,
            env=os.environ.copy(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return {
            "available": True,
            "error": type(exc).__name__,
        }
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip()[:12000],
        "stderr": completed.stderr.strip()[:4000],
    }


def _distribution_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for name in RELEVANT_DISTRIBUTIONS:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _native_files() -> list[dict[str, Any]]:
    """Describe packaged native files without exposing their absolute paths."""
    try:
        distribution = importlib.metadata.distribution("mojolearn")
    except importlib.metadata.PackageNotFoundError:
        return []

    records: list[dict[str, Any]] = []
    for relative in distribution.files or ():
        relative_text = str(relative)
        if not relative_text.endswith((".so", ".dylib")):
            continue
        absolute = Path(distribution.locate_file(relative))
        record: dict[str, Any] = {
            "relative_path": relative_text,
            "exists": absolute.is_file(),
        }
        if absolute.is_file():
            digest = hashlib.sha256()
            with absolute.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            record.update(size=absolute.stat().st_size, sha256=digest.hexdigest())
        records.append(record)
    return records


def _installed_tiers(native_files: list[dict[str, Any]]) -> dict[str, Any]:
    """Which numeric tiers this install actually carries.

    A wheel that shipped fewer tiers than the caller expects is the single
    most confusing failure this library has: `numeric_mode="deterministic"`
    raises from a missing-binary stub BY NAME, which is the correct
    behaviour, but the answer to "why" lives in the wheel's contents rather
    than in the traceback. This puts it in the report.

    Read off the installed file list rather than by importing, so it still
    answers when the extensions do not load at all -- the same reason this
    module lives outside the package.
    """
    counts: dict[str, int] = {tier: 0 for tier in NUMERIC_TIERS}
    for record in native_files:
        relative = record["relative_path"]
        if not relative.endswith(".so") or not record.get("exists"):
            continue
        parts = Path(relative).parts
        # mojolearn/_mojolearn_rf.so -> fast; mojolearn/identical/... -> identical
        parent = parts[-2] if len(parts) >= 2 else ""
        tier = parent if parent in counts else "fast"
        counts[tier] += 1
    return {
        "expected_extensions_per_tier": 10,
        "present": [tier for tier in NUMERIC_TIERS if counts[tier] > 0],
        "absent": [tier for tier in NUMERIC_TIERS if counts[tier] == 0],
        "extensions_by_tier": counts,
        # A tier with SOME of its extensions is worse than one with none: the
        # package imports and only the estimators needing the missing binding
        # raise, so it is worth naming as its own state rather than as present.
        "incomplete": [
            tier for tier in NUMERIC_TIERS if 0 < counts[tier] < 10
        ],
    }


def _gpu_facts() -> dict[str, Any]:
    facts: dict[str, Any] = {}
    system = platform.system()
    if system == "Darwin":
        # system_profiler's hardware JSON also contains the machine serial,
        # hardware UUID and provisioning UDID. Query only these named fields.
        facts["apple"] = {
            "model": _run(["sysctl", "-n", "hw.model"]),
            "chip": _run(["sysctl", "-n", "machdep.cpu.brand_string"]),
            "cpu_count": _run(["sysctl", "-n", "hw.ncpu"]),
            "memory_bytes": _run(["sysctl", "-n", "hw.memsize"]),
            "macos": _run(["sw_vers", "-productVersion"]),
        }
    facts["nvidia"] = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,compute_cap",
            "--format=csv,noheader",
        ]
    )
    rocm = _run(["rocm-smi", "--showproductname", "--showdriverversion"])
    if not rocm.get("available"):
        rocm = _run(["amd-smi", "static", "--asic", "--driver"])
    facts["amd"] = rocm
    return facts


def _support_classification(gpu: dict[str, Any]) -> dict[str, str]:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine == "arm64":
        return {
            "level": "supported",
            "reason": "macOS arm64 is the released wheel platform; exact certification depends on the release support matrix",
        }
    if gpu["nvidia"].get("available"):
        return {
            "level": "source-build",
            "reason": "CUDA is supported from source on the release's tested GPU columns; no Linux wheel is currently published",
        }
    if gpu["amd"].get("available"):
        return {
            "level": "source-build",
            "reason": "HIP is supported from source on the release's tested GPU columns; no Linux wheel is currently published",
        }
    return {
        "level": "compatibility-report",
        "reason": "no released GPU column was detected; reports are welcome but this environment is not certified",
    }


def _import_probe() -> dict[str, Any]:
    code = textwrap.dedent(
        """
        import json
        try:
            import mojolearn
        except BaseException as exc:
            print(json.dumps({"ok": False, "type": type(exc).__name__, "message": str(exc)}))
            raise SystemExit(1)
        # CALLED, not read. `mojolearn.numeric_mode` is a FUNCTION, and
        # putting the function object in this dict made json.dumps raise
        # "Object of type function is not JSON serializable" -- inside the
        # probe, so a healthy install reported `import probe: failed` and
        # `mojolearn doctor` exited 1. Fixed 2026-08-29; found by running it.
        # The call also reads the mode back OUT OF THE LOADED BINARY, which
        # is the fact worth having: it catches a wheel that answered with a
        # tier other than the one that was asked for.
        try:
            mode = mojolearn.numeric_mode()
        except BaseException as exc:
            mode = "unreadable: " + type(exc).__name__
        print(json.dumps({
            "ok": True,
            "version": getattr(mojolearn, "__version__", None),
            "numeric_mode": mode,
        }))
        """
    )
    raw = _run([sys.executable, "-c", code])
    result: dict[str, Any] = {
        "available": raw.get("available", False),
        "returncode": raw.get("returncode"),
    }
    # Loader messages frequently contain absolute installation and home
    # paths. Retain the exception class and successful public package facts,
    # but never the raw message or stderr.
    stdout = raw.get("stdout", "")
    if stdout:
        try:
            payload = json.loads(stdout.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            result["result"] = {"ok": False, "type": "UnparsedProbeFailure"}
        else:
            result["result"] = {
                key: payload[key]
                for key in ("ok", "type", "version", "numeric_mode")
                if key in payload
            }
    elif raw.get("error"):
        result["result"] = {"ok": False, "type": raw["error"]}
    return result


def collect_diagnostics() -> dict[str, Any]:
    gpu = _gpu_facts()
    native_files = _native_files()
    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_at_utc": _datetime.datetime.now(_datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python_implementation": platform.python_implementation(),
            "python_version": platform.python_version(),
        },
        "environment": dict(
            {
                key: os.environ[key]
                for key in SAFE_ENVIRONMENT_KEYS
                if key in os.environ
            },
            **{
                key + "_set": True
                for key in PRESENCE_ONLY_ENVIRONMENT_KEYS
                if os.environ.get(key, "").strip()
            },
        ),
        "distributions": _distribution_versions(),
        "native_files": native_files,
        "numeric_tiers": _installed_tiers(native_files),
        "gpu": gpu,
        "support": _support_classification(gpu),
        "mojo_version": _run(["mojo", "--version"]),
        "import_probe": _import_probe(),
        "privacy": {
            "policy": "allowlist",
            "omitted": [
                "username",
                "home_directory",
                "working_directory",
                "network_configuration",
                "arbitrary_environment_variables",
                "credentials",
            ],
        },
    }


def _summary(report: dict[str, Any]) -> str:
    versions = report["distributions"]
    support = report["support"]
    probe = report["import_probe"]
    lines = [
        "MojoLearn doctor",
        f"recorded: {report['recorded_at_utc']}",
        "platform: {system} {release} {machine}".format(**report["platform"]),
        f"python: {report['platform']['python_version']}",
        f"mojolearn: {versions.get('mojolearn') or 'not installed'}",
        f"numeric mode requested: {report['environment'].get('MOJOLEARN_NUMERIC_MODE', 'fast (default)')}",
        f"numeric mode loaded: {report['import_probe'].get('result', {}).get('numeric_mode') or 'unknown (import probe did not report one)'}",
        "numeric tiers installed: {}".format(
            ", ".join(report["numeric_tiers"]["present"]) or "none"
        ),
        f"support classification: {support['level']}",
        f"support note: {support['reason']}",
        f"import probe: {'passed' if probe.get('returncode') == 0 else 'failed'}",
        "privacy: allowlisted facts only; review every bundle before uploading",
    ]
    tiers = report["numeric_tiers"]
    # Named, not merely omitted from the "installed" line. `numeric_mode=` on
    # an absent tier raises from a missing-binary stub, and this is where the
    # reader finds out that is the wheel's doing rather than their own.
    if tiers["absent"]:
        lines.append(
            "numeric tiers NOT installed: {} (asking for one raises by name)".format(
                ", ".join(tiers["absent"])
            )
        )
    if tiers["incomplete"]:
        lines.append(
            "numeric tiers INCOMPLETE: {} (some estimators will raise)".format(
                ", ".join(tiers["incomplete"])
            )
        )
    probe_result = probe.get("result", {})
    if not probe_result.get("ok") and probe_result.get("type"):
        lines.append(f"import error type: {probe_result['type']}")
    return "\n".join(lines) + "\n"


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def command_doctor(args: argparse.Namespace) -> int:
    report = collect_diagnostics()
    summary = _summary(report)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(summary, end="")
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        _write_json(output, report)
    return 0 if report["import_probe"].get("returncode") == 0 else 1


def command_bug_report(args: argparse.Namespace) -> int:
    reproducer = Path(args.reproducer).expanduser()
    if not reproducer.is_file():
        raise SystemExit(f"reproducer does not exist or is not a file: {reproducer}")

    stamp = _datetime.datetime.now(_datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    destination = Path(args.output or f"mojolearn-bug-{stamp}.zip")
    if destination.exists() and not args.force:
        raise SystemExit(f"refusing to overwrite {destination}; pass --force")

    report = collect_diagnostics()
    with tempfile.TemporaryDirectory(prefix="mojolearn-report-") as temporary:
        root = Path(temporary) / "mojolearn-bug-report"
        root.mkdir()
        _write_json(root / "doctor.json", report)
        (root / "doctor.txt").write_text(_summary(report), encoding="utf-8")
        shutil.copyfile(reproducer, root / ("reproducer" + reproducer.suffix))

        attachment_names: list[str] = []
        for index, attachment_text in enumerate(args.attach or (), start=1):
            attachment = Path(attachment_text).expanduser()
            if not attachment.is_file():
                raise SystemExit(f"attachment does not exist or is not a file: {attachment}")
            safe_name = f"attachment-{index}{attachment.suffix}"
            shutil.copyfile(attachment, root / safe_name)
            attachment_names.append(safe_name)

        manifest = {
            "schema_version": SCHEMA_VERSION,
            "reproducer": "reproducer" + reproducer.suffix,
            "attachments": attachment_names,
            "user_review_required": True,
        }
        _write_json(root / "manifest.json", manifest)
        (root / "README.txt").write_text(
            "Review every file before uploading this archive. The diagnostic "
            "collector uses an allowlist, but the reproducer and attachments "
            "are user-provided and may contain private data.\n",
            encoding="utf-8",
        )

        destination.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.iterdir()):
                archive.write(path, path.relative_to(root.parent))

    print(f"wrote {destination}")
    print("Review the archive before attaching it to a public issue.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="mojolearn",
        description="MojoLearn diagnostics and support-bundle tools.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="inspect an installation without exposing private environment data")
    doctor.add_argument("--json", action="store_true", help="print the full JSON report")
    doctor.add_argument("--output", help="also write the JSON report to this path")
    doctor.set_defaults(handler=command_doctor)

    bug_report = subparsers.add_parser("bug-report", help="create a review-before-upload support bundle")
    bug_report.add_argument("reproducer", help="minimal reproducer file to include")
    bug_report.add_argument("--attach", action="append", help="identity trace or log to include; may be repeated")
    bug_report.add_argument("--output", help="archive path (default: timestamped zip in the current directory)")
    bug_report.add_argument("--force", action="store_true", help="overwrite an existing output archive")
    bug_report.set_defaults(handler=command_bug_report)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
