# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Dependency-light diagnostics for the installed Mojolearn distribution.

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
import re
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
    "MOJOLEARN_VENDOR",
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
NUMERIC_TIERS = ("fast", "identical")
NATIVE_DISTRIBUTIONS = (
    "mojolearn",
    "mojolearn-core",
    "mojolearn-nvidia",
    "mojolearn-amd",
)
RELEVANT_DISTRIBUTIONS = (
    *NATIVE_DISTRIBUTIONS,
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
    records: list[dict[str, Any]] = []
    for name in NATIVE_DISTRIBUTIONS:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for relative in distribution.files or ():
            relative_text = str(relative)
            if not relative_text.endswith((".so", ".dylib")):
                continue
            absolute = Path(distribution.locate_file(relative))
            record: dict[str, Any] = {
                "distribution": name,
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


#: Architecture directory names under a vendor (the architecture axis,
#: 2026-08-30): sm_80, sm_90a, gfx90a, gfx942, ...
_ARCH_RE = re.compile(r"^(sm_[0-9]+a?|gfx[0-9a-f]+)$")
_GPU_DIRECTORIES = ("cuda", "hip", "cuda_native", "cuda_ptx", "hip_native")


def _installed_gpu_sets(native_files: list[dict[str, Any]]) -> list[str]:
    """Report carried GPU sets, without claiming they can run on this device."""
    sets = set()
    for record in native_files:
        parts = Path(record["relative_path"]).parts
        if (record.get("exists") and len(parts) >= 4
                and parts[0] == "mojolearn" and parts[1] in _GPU_DIRECTORIES
                and _ARCH_RE.fullmatch(parts[2])
                and parts[-1].startswith("_mojolearn") and parts[-1].endswith(".so")):
            sets.add("/".join(parts[1:3]))
    return sorted(sets)


#: The index every merged set directory ships beside its native files
#: (`{"format": 1, "files": {"<stem>.so": [member modules]}}`), written by
#: bindings/merged_build.sh, held to bindings/merged_groups.json by
#: packaging/linux/merged_layout.py and read by python/mojolearn/_merged.py to
#: load each member. It is the wheel's own statement of which native files a
#: directory must hold, so it is what completeness is judged against.
BINDING_INDEX = "_mojolearn_merged.json"
#: The CPU set directory (mojolearn/host/). It serves `identical` on the CPU
#: (packaging/linux/pack_wheel.py records it supported_modes=['identical']).
HOST_SET = "host"
_SET_ROLES = (*NUMERIC_TIERS, HOST_SET)


def _binding_indexes() -> list[dict[str, Any]]:
    """Read every shipped binding index (file names only, no absolute paths).

    A file read, not an import, so it answers when the extensions do not load.
    """
    records: list[dict[str, Any]] = []
    for name in NATIVE_DISTRIBUTIONS:
        try:
            distribution = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        for relative in distribution.files or ():
            relative_text = str(relative)
            if Path(relative_text).name != BINDING_INDEX:
                continue
            absolute = Path(distribution.locate_file(relative))
            record: dict[str, Any] = {
                "distribution": name,
                "relative_path": relative_text,
                "exists": absolute.is_file(),
            }
            if record["exists"]:
                try:
                    document = json.loads(absolute.read_text(encoding="utf-8"))
                    files = document.get("files") if isinstance(document, dict) else None
                    if not isinstance(files, dict) or not files or not all(
                            isinstance(f, str) and f.endswith(".so") and "/" not in f
                            for f in files):
                        raise ValueError("not a binding index")
                    record["files"] = sorted(files)
                except (OSError, ValueError):
                    record["unreadable"] = True
            records.append(record)
    return records


def _set_vendor(parts: tuple[str, ...]) -> str:
    """'flat' for the package-level directories, else `<vendor>[/<arch>]`."""
    vendor = "flat"
    for i, part in enumerate(parts):
        if part in _GPU_DIRECTORIES:
            vendor = part
            # THE ARCHITECTURE AXIS (2026-08-30): one directory per GPU
            # architecture under the vendor.
            if i + 1 < len(parts) and _ARCH_RE.match(parts[i + 1]):
                vendor = f"{part}/{parts[i + 1]}"
    return vendor


def _set_role(parts: tuple[str, ...]) -> str:
    """The set a directory holds: mojolearn/[<vendor>/<arch>/]identical ->
    identical, mojolearn/host -> host, the package or <vendor>/<arch>
    directory itself -> fast (the legacy fast tier directory)."""
    last = parts[-1] if parts else ""
    return last if last in _SET_ROLES else "fast"


def _installed_tiers(native_files: list[dict[str, Any]],
                     indexes: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Which numeric tiers this install carries, and whether each native set
    directory holds every file the wheel says it ships.

    Read off the installed file list and the shipped binding indexes rather
    than by importing, so it still answers when the extensions do not load at
    all -- the same reason this module lives outside the package.

    COMPLETENESS IS WHAT THE WHEEL SAYS, NEVER A COUNT (2026-11-20). Until
    then a set directory was called INCOMPLETE when it held between 1 and 14
    extensions, a constant from before the bindings were merged into groups
    (bindings/merged_groups.json, 2026-10-20). A merged directory ships a few
    group files, so every healthy Linux install read "numeric tiers
    INCOMPLETE: identical"; and mojolearn/host/ was counted as the fast tier.
    Now a directory is:

      complete     it ships a binding index and holds every file it names
      incomplete   a file its index names, or the wheel RECORD lists, is not
                   on disk (each one named), or its index is listed but absent
                   or unreadable
      unknown      no index (a per-binding or older wheel): the doctor cannot
                   tell whether a binding is missing, and says so
    """
    directories: dict[str, dict[str, Any]] = {}

    def entry(parts: tuple[str, ...]) -> dict[str, Any]:
        key = "/".join(parts)
        if key not in directories:
            directories[key] = {"parts": parts, "present": set(), "listed_absent": set(),
                                "expected": None, "index_problem": None}
        return directories[key]

    for record in native_files:
        relative = record["relative_path"]
        if not relative.endswith(".so"):
            continue
        parts = Path(relative).parts
        if not parts[-1].startswith("_mojolearn"):
            continue  # a staged runtime library, not an extension
        d = entry(parts[:-1])
        (d["present"] if record.get("exists") else d["listed_absent"]).add(parts[-1])
    for record in indexes or ():
        parts = Path(record["relative_path"]).parts
        d = entry(parts[:-1])
        if not record.get("exists"):
            d["index_problem"] = "listed in the wheel RECORD, absent on disk"
        elif record.get("unreadable"):
            d["index_problem"] = "unreadable"
        else:
            d["expected"] = set(record["files"])

    counts: dict[str, int] = {role: 0 for role in _SET_ROLES}
    by_vendor: dict[str, dict[str, int]] = {}
    sets: list[dict[str, Any]] = []
    missing_files: list[str] = []
    incomplete: list[str] = []
    cannot_tell: list[str] = []
    for key in sorted(directories):
        d = directories[key]
        parts = d["parts"]
        role = _set_role(parts)
        vendor = _set_vendor(parts)
        present = d["present"]
        counts[role] += len(present)
        if present:
            by_vendor.setdefault(vendor, {r: 0 for r in _SET_ROLES})[role] += len(present)
        missing = set(d["listed_absent"])
        if d["expected"] is not None:
            missing |= d["expected"] - present
        named = [f"{key}/{name}" for name in sorted(missing)]
        if d["index_problem"]:
            named.append(f"{key}/{BINDING_INDEX} ({d['index_problem']})")
        if named:
            status = "incomplete"
            missing_files.extend(named)
            tier = "identical" if role == HOST_SET else role
            if tier not in incomplete:
                incomplete.append(tier)
        elif d["expected"] is not None:
            status = "complete"
        else:
            status = "unknown"
            cannot_tell.append(key)
        sets.append({"directory": key, "vendor": vendor, "set": role,
                     "status": status, "files": len(present), "missing": named})

    # The host set is the identical set on the CPU, so it makes `identical`
    # present on a CPU-only install.
    tier_files = {"fast": counts["fast"], "identical": counts["identical"] + counts[HOST_SET]}
    present_tiers = [tier for tier in NUMERIC_TIERS if tier_files[tier] > 0]
    # OFF APPLE THE NAME `fast` IS ANSWERED BY THE IDENTICAL SET when no fast
    # set is installed (python/mojolearn/_backend.py fast_answer), so it is
    # not reported as a tier whose request raises.
    answered_by = {}
    if (platform.system() != "Darwin" and "fast" not in present_tiers
            and "identical" in present_tiers):
        answered_by["fast"] = "identical"
    return {
        "present": present_tiers,
        "absent": [tier for tier in NUMERIC_TIERS
                   if tier not in present_tiers and tier not in answered_by],
        "answered_by": answered_by,
        "extensions_by_tier": counts,
        "extensions_by_vendor_and_tier": by_vendor,
        "sets": sets,
        # A tier with SOME of its files is worse than one with none: the
        # package imports and only the estimators needing the missing file
        # raise, so it is named as its own state, with the files.
        "incomplete": [tier for tier in NUMERIC_TIERS if tier in incomplete],
        "missing_files": missing_files,
        "cannot_tell": cannot_tell,
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


def _support_classification(probe: dict[str, Any]) -> dict[str, str]:
    """Separate a published wheel platform from a successful backend import.

    GPU utility availability is not evidence of compatible binaries. Let the
    loader probe report failure for a missing set, driver or runtime instead.
    """
    system = platform.system()
    machine = platform.machine().lower()
    wheel_platform = {("Darwin", "arm64"): "macOS arm64",
                      ("Linux", "x86_64"): "Linux x86-64"}.get((system, machine))
    if wheel_platform is None:
        return {
            "level": "compatibility-report",
            "reason": "no published wheel platform matches this OS and architecture; consult the release support matrix for source builds",
        }
    result = probe.get("result", {})
    if probe.get("returncode") != 0 or not result.get("ok"):
        return {
            "level": "load-failed",
            "reason": f"{wheel_platform} wheels are published, but this installation failed its import probe; check installed GPU sets, package versions and driver/runtime requirements",
        }
    vendor = result.get("vendor")
    backend = {"cuda": "NVIDIA CUDA", "hip": "AMD HIP", "metal": "Apple Metal",
               "cpu": "CPU"}.get(vendor)
    if backend is None:
        return {
            "level": "backend-unconfirmed",
            "reason": f"{wheel_platform} wheels are published and the package imports, but the loaded backend could not be confirmed",
        }
    return {
        "level": "imported",
        "reason": f"{wheel_platform} wheels are published; the {backend} backend loaded. An import pass does not certify every algorithm or numerical result; consult the release support matrix and run verify for the selected checks",
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
        # THE VENDOR, read back the same way (2026-08-29): on Linux one
        # wheel carries CUDA and HIP sets and the one that loaded is a
        # fact about this install worth having in every doctor bundle.
        try:
            vendor = mojolearn.vendor()
        except BaseException as exc:
            vendor = "unreadable: " + type(exc).__name__
        # THE ARCHITECTURE (2026-08-30): which sm_*/gfx* set loaded, and
        # why -- None on the flat and arch-less layouts.
        try:
            gpu_arch = mojolearn.gpu_arch()
        except BaseException as exc:
            gpu_arch = "unreadable: " + type(exc).__name__
        print(json.dumps({
            "ok": True,
            "version": getattr(mojolearn, "__version__", None),
            "numeric_mode": mode,
            "vendor": vendor,
            "gpu_arch": gpu_arch,
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
                for key in ("ok", "type", "version", "numeric_mode", "vendor",
                            "gpu_arch")
                if key in payload
            }
    elif raw.get("error"):
        result["result"] = {"ok": False, "type": raw["error"]}
    return result


def collect_diagnostics() -> dict[str, Any]:
    gpu = _gpu_facts()
    native_files = _native_files()
    probe = _import_probe()
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
        "gpu_sets": _installed_gpu_sets(native_files),
        "numeric_tiers": _installed_tiers(native_files, _binding_indexes()),
        "gpu": gpu,
        "support": _support_classification(probe),
        "mojo_version": _run(["mojo", "--version"]),
        "import_probe": probe,
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
        "Mojolearn doctor",
        f"recorded: {report['recorded_at_utc']}",
        "platform: {system} {release} {machine}".format(**report["platform"]),
        f"python: {report['platform']['python_version']}",
        f"mojolearn: {versions.get('mojolearn') or 'not installed'}",
        f"mojolearn-core: {versions.get('mojolearn-core') or 'not installed'}",
        f"mojolearn-nvidia: {versions.get('mojolearn-nvidia') or 'not installed'}",
        f"mojolearn-amd: {versions.get('mojolearn-amd') or 'not installed'}",
        "GPU architecture sets installed: {}".format(
            ", ".join(report["gpu_sets"]) or "none (Apple Metal uses a flat layout)"
        ),
        f"numeric mode requested: {report['environment'].get('MOJOLEARN_NUMERIC_MODE', 'identical (default)')}",
        f"numeric mode loaded: {report['import_probe'].get('result', {}).get('numeric_mode') or 'unknown (import probe did not report one)'}",
        f"vendor loaded: {report['import_probe'].get('result', {}).get('vendor') or 'unknown (import probe did not report one)'}",
        f"gpu architecture loaded: {report['import_probe'].get('result', {}).get('gpu_arch') or 'none (flat or arch-less layout, or probe did not report one)'}",
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
    for tier, answer in tiers.get("answered_by", {}).items():
        lines.append(f"numeric mode {tier}: answered by the {answer} set (no separate {tier} set on this platform)")
    if tiers["incomplete"]:
        lines.append(
            "numeric tiers INCOMPLETE: {} (estimators served by a missing file will raise)".format(
                ", ".join(tiers["incomplete"])
            )
        )
        lines.extend(f"  missing: {name}" for name in tiers["missing_files"])
    if tiers.get("cannot_tell"):
        lines.append(
            "native set completeness: cannot tell for {} (the wheel ships no binding index there)".format(
                ", ".join(tiers["cannot_tell"])
            )
        )
    elif tiers.get("sets") and not tiers["incomplete"]:
        lines.append("native sets complete: every file each binding index names is installed")
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
        description="Mojolearn diagnostics and support-bundle tools.",
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
