# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""`python -m mojolearn verify --qualify-gpu`: qualify this NVIDIA device and
driver for IDENTICAL mode through the PTX fallback.

Native kernels ship for a few NVIDIA architectures. Every other NVIDIA GPU
runs the PTX fallback, which the driver compiles on the machine, so a release
can only admit the device/driver pairs it measured. This command lets any
other configuration qualify itself. It runs the complete identity suite the
wheel already ships (`verify --all`, then `verify --all --neural-training`:
every lane, every fixture, every part) through the PTX payload on this device,
in fresh verifier processes, and compares every part with the shipped
reference table (`mojolearn/verify_reference/table.json`, the cross-vendor
record). Only a run in which every applicable lane, fixture and part reads
IDENTICAL writes a local admission (`ptx_admission.LOCAL_SCHEMA`) bound to this
exact wheel, PTX payload, device, driver and reference. The pinned exclusions
(`ptx_admission.LOCAL_EXCLUSIONS`), which the shipped table cannot judge on
any device, must each read absent in exactly their pinned way; the admission
lists them and counts none as a match. Anything else writes
no admission and reports what differed, what the wheel ships no reference for,
and what did not run.

This is glue: it launches the existing verifier, reads its reports and writes
one JSON record. It computes no numeric result.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

from . import ptx_admission

EXIT_QUALIFIED = 0
EXIT_DIFFERS = 1
EXIT_CANNOT_RUN = 4
EXIT_NO_REFERENCE = 5
_SHOWN = 20


def _emit(text, stream=None):
    print(text, file=stream or sys.stdout, flush=True)


def _write_token(path, key):
    """The one-run token this command's verifier processes present to the
    loader. Private to this user, bound to this process and configuration."""
    import secrets
    doc = ptx_admission.qualification_token(key, os.getpid(), secrets.token_hex(32))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(doc, stream, sort_keys=True)
    return path


def _run_verifier(profile, report_path, log_path, args, token=None):
    """One full-depth verifier scope in a fresh process under the PTX payload."""
    command = [sys.executable, "-m", "mojolearn", "verify", "--all"]
    if profile == "neural-training":
        command.append("--neural-training")
    command += ["--json-out", report_path, "--cell-timeout", str(getattr(args, "cell_timeout", 120.0)),
                "--cpu-threads", str(getattr(args, "cpu_threads", 1))]
    env = dict(os.environ)
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env["PYTHONPATH"] = package_parent + os.pathsep + env.get("PYTHONPATH", "")
    env["MOJOLEARN_NUMERIC_MODE"] = "identical"
    env.pop(ptx_admission.QUALIFY_TOKEN_ENV, None)
    if token is not None:
        env[ptx_admission.QUALIFY_TOKEN_ENV] = os.path.abspath(token)
    with open(log_path, "wb") as log:
        return subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT).returncode


def _atomic_json(path, value):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = f"{path}.{os.getpid()}.tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=1, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _finish(args, code, headline, detail, **extra):
    if getattr(args, "json", False):
        _emit(json.dumps(dict(format="mojolearn.ptx-qualification.v1", verdict=headline, exit=code,
                              detail=detail, **extra), indent=1, sort_keys=True))
    else:
        _emit(f"RESULT: {headline}: {detail}. exit {code}")
    return code


def _judge_scope(profile, evidence, run, args, token, log, reference, table_fixtures, ptx_identical):
    """Run one verifier scope and judge its report."""
    report_path = os.path.join(evidence, f"{profile}.json")
    log_path = os.path.join(evidence, f"{profile}.log")
    log(f"# running the {profile} scope (log: {log_path})")
    code = run(profile, report_path, log_path, args, token=token)
    try:
        with open(report_path, "rb") as stream:
            raw = stream.read()
        report, digest = json.loads(raw), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError):
        report, digest = None, None
    result = ptx_admission.judge_report(report, profile=profile, reference=reference,
        table_fixtures=table_fixtures, ptx_identical_sha256=ptx_identical, report_sha256=digest)
    if code != 0 and not (result["differing"] or result["missing"] or result["incomplete"]):
        result["incomplete"].append(f"{profile}: the verifier exited {code}")
    summary = result["summary"] or {}
    log(f"#   {profile}: verifier exit {code}, {summary.get('lanes_verified', 0)} of "
        f"{summary.get('lanes_in_scope', 0)} lanes verified, {len(summary.get('exclusions', []))} pinned "
        f"exclusions not compared, {len(result['differing'])} differing, "
        f"{len(result['missing'])} without a reference, {len(result['incomplete'])} incomplete")
    return result


def cmd_qualify_gpu(args, backend=None, run=None):
    """`backend` and `run` are the real loader and verifier unless a test supplies its own."""
    if backend is None:
        from . import _backend as backend
    run = run or _run_verifier
    log = (lambda s: _emit(s, sys.stderr)) if getattr(args, "json", False) else _emit
    if backend._CPU_ONLY is not None:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN",
                       "this process loaded no GPU set, so there is no GPU to qualify; nothing was written")
    receipt = backend.baseline_selection_receipt()
    if receipt is None:
        return _finish(args, EXIT_QUALIFIED, "NOT NEEDED",
                       f"this device runs native {backend.vendor()} kernels ({backend.gpu_arch()}); "
                       "IDENTICAL mode needs no local qualification")
    if receipt.get("requested") != "native-first":
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN",
                       "the experimental PTX route is forced (MOJOLEARN_CUDA_PATH); unset it and run again")
    if receipt.get("identical_qualified"):
        return _finish(args, EXIT_QUALIFIED, "ALREADY QUALIFIED",
                       f"a {receipt.get('admission')} admission covers this wheel, device and driver",
                       admission=receipt.get("admission"))
    configuration = receipt.get("configuration")
    if configuration is None:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN",
                       str((receipt.get("admission_detail") or {}).get("error")
                           or "the device configuration could not be read"))
    source, manifest = receipt["source_commit"], receipt["manifest_sha256"]
    pkg = backend._pkg_dir()
    try:
        reference = ptx_admission.reference_hashes(pkg)
        with open(os.path.join(pkg, "verify_reference", "table.json"), encoding="utf-8") as stream:
            table_fixtures = list(json.load(stream)["fixtures"])
        key = ptx_admission.local_admission_key(source, manifest, configuration, reference)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return _finish(args, EXIT_NO_REFERENCE, "NO REFERENCE", f"{exc}; nothing was written")
    directory = ptx_admission.local_admission_dir()
    admission_path = ptx_admission.local_admission_path(directory, key)
    evidence = os.path.join(directory, "evidence", f"{key[:32]}-{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}")
    os.makedirs(evidence, exist_ok=True)
    ptx_identical = {digest for name, digest in backend.ptx_payload_files().items()  # glue: select IDENTICAL-tier file digests from the manifest
                     if name.startswith("identical/")}
    log(f"# qualifying {configuration['device_name']} (compute capability "
        f"{'.'.join(str(v) for v in configuration['compute_capability'])}, driver "  # glue: format capability metadata
        f"{configuration['driver_version']}) for IDENTICAL mode through the PTX fallback")
    log(f"# every lane, fixture and part against the shipped reference table; evidence in {evidence}")
    judged = []
    token = _write_token(os.path.join(evidence, "qualification.token"), key)
    try:
        for profile in ptx_admission.QUALIFY_PROFILES:  # glue: launch the two verifier scopes in turn
            judged.append(_judge_scope(profile, evidence, run, args, token, log, reference, table_fixtures,
                                       ptx_identical))
    finally:
        # One run, one token: nothing later can present it.
        try:
            os.unlink(token)
        except OSError:
            pass
    differing = judged[0]["differing"] + judged[1]["differing"]
    missing = judged[0]["missing"] + judged[1]["missing"]
    incomplete = judged[0]["incomplete"] + judged[1]["incomplete"]
    if differing or missing or incomplete:
        groups = (("DIFFERING (this device does not reproduce the reference)", differing),
                  ("NO REFERENCE SHIPPED (the wheel cannot judge these)", missing),
                  ("DID NOT RUN OR NOT ATTRIBUTABLE", incomplete))
        for title, rows in groups:  # glue: print three message lists
            if rows:
                log(f"# {title}: {len(rows)}")
                for row in rows[:_SHOWN]:  # glue: print a bounded list of message strings
                    log(f"#   {row}")
                if len(rows) > _SHOWN:
                    log(f"#   ... {len(rows) - _SHOWN} more in the reports under {evidence}")
        code, headline = ((EXIT_DIFFERS, "NOT QUALIFIED") if differing else
                          (EXIT_NO_REFERENCE, "NOT QUALIFIED") if missing else
                          (EXIT_CANNOT_RUN, "INCOMPLETE"))
        return _finish(args, code, headline,
                       f"{len(differing)} differing, {len(missing)} without a shipped reference, "
                       f"{len(incomplete)} incomplete; no admission was written and IDENTICAL mode stays "
                       "refused on this configuration", differing=differing, missing_reference=missing,
                       incomplete=incomplete, evidence=evidence)
    from ._version import __version__
    doc = ptx_admission.build_local_admission(
        source_commit=source, manifest_sha256=manifest, configuration=configuration, reference=reference,
        summaries=[judged[0]["summary"], judged[1]["summary"]], core_version=__version__,
        created_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    _atomic_json(admission_path, doc)
    coverage = doc["coverage"]
    log(f"# compared equal: {coverage['parts_compared']} parts over {coverage['lanes_compared']} lanes. "
        f"Not compared: {len(coverage['exclusions'])} pinned exclusions the shipped reference cannot judge "
        "(listed in the admission; none is counted as a match)")
    return _finish(args, EXIT_QUALIFIED, "QUALIFIED",
                   f"{coverage['parts_compared']} parts over {coverage['lanes_compared']} lanes matched the "
                   f"shipped reference, and {len(coverage['exclusions'])} pinned exclusions it cannot judge were "
                   f"not compared; local admission written to {admission_path} for this wheel, device and "
                   "driver only", admission_path=admission_path, evidence=evidence,
                   coverage=dict(lanes_compared=coverage["lanes_compared"], parts_compared=coverage["parts_compared"],
                                 exclusions=len(coverage["exclusions"])))
