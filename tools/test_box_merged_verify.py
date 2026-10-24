"""Qualification-controller failure boundaries, with inert compiler/GPU shims.

No Mojo build, native extension import, hardware access, or network request.
The controller uses real git and the real merged-index checker.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
CONTROLLER = ROOT / "tools/box_merged_verify.sh"


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


SHIM = r'''
import json, os, pathlib, subprocess, sys
args = sys.argv[1:]
tool = pathlib.Path(sys.argv[0]).name
root = pathlib.Path(os.environ["BOX_REPO_DIR"])
with open(os.environ["FAKE_EVENTS"], "a") as f:
    f.write(json.dumps([tool, args, os.environ.get("MOJOLEARN_NUMERIC_MODE"),
                       os.environ.get("MOJOLEARN_COMPILE_JOBS")]) + "\n")
if tool == "nvidia-smi":
    if "-L" in args: print("GPU 0: Fake GPU")
    elif any("compute_cap" in a for a in args): print("8.9")
    else: print("NVIDIA L40S")
elif tool == "rocm-smi":
    if "--showid" in args: print("GPU[0]")
    else: print("MI325X")
elif tool == "flock":
    sys.exit(int(os.environ.get("FAKE_LOCK_RC", "0")))
elif tool == "mojo":
    assert args == ["--version"]
    print("inert compiler fixture")
elif tool == "sha256sum":
    import hashlib
    for name in args:
        print(hashlib.sha256(pathlib.Path(name).read_bytes()).hexdigest(), name)
elif tool == "pixi":
    if args[0] == "install": sys.exit(int(os.environ.get("FAKE_INSTALL_RC", "0")))
    if args[:2] == ["run", "-q"]: args = args[2:]
    if args == ["mojo", "--version"]:
        print("inert compiler fixture")
    elif args[:2] == ["python", "-c"]:
        if "sys.executable" in args[2]: print(os.environ["FAKE_PYTHON"])
        elif "shutil.which" in args[2]: print(pathlib.Path(os.environ["FAKE_PYTHON"]).with_name("mojo"))
        else:
            rc = int(os.environ.get("FAKE_MATH_RC", "0"))
            if not rc:
                output = pathlib.Path(args[3]); output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b"fresh math")
            sys.exit(rc)
    elif args[:2] == ["sh", "bindings/merged_build.sh"]:
        kind = args[2]; directory = "host" if kind == "host" else "identical"
        rc = int(os.environ.get("FAKE_BUILD_" + directory.upper() + "_RC", "0"))
        if rc: sys.exit(rc)
        sys.path.insert(0, str(root / "tools"))
        import merged_bindings
        out = pathlib.Path(args[args.index("--outdir") + 1]); out.mkdir(parents=True)
        files = {}
        for stem, _, members in merged_bindings.plan(directory, "linux", True):
            (out / (stem + ".so")).write_bytes(b"fresh " + directory.encode())
            files[stem + ".so"] = members
        if os.environ.get("FAKE_INCOMPLETE") == directory:
            (out / next(iter(files))).unlink()
        (out / "_mojolearn_merged.json").write_text(json.dumps({"format": 1, "files": files}))
    else: sys.exit("unexpected pixi call: " + repr(args))
elif tool == "fake-python":
    if args[:3] == ["-m", "mojolearn", "verify"]:
        print("RESULT inert verification")
        sys.exit(int(os.environ.get("FAKE_VERIFY_RC", "0")))
    if args[0] == "tools/merged_bindings.py":
        sys.exit(subprocess.call([os.environ["REAL_PYTHON"], *args]))
    sys.exit("unexpected Python call: " + repr(args))
else: sys.exit("unexpected tool: " + tool)
'''


@pytest.fixture
def box(tmp_path):
    repo = tmp_path / "repo"
    (repo / "tools").mkdir(parents=True)
    (repo / "bindings").mkdir()
    (repo / "python/mojolearn/.libs").mkdir(parents=True)
    shutil.copyfile(ROOT / "tools/merged_bindings.py", repo / "tools/merged_bindings.py")
    shutil.copyfile(ROOT / "bindings/merged_groups.json", repo / "bindings/merged_groups.json")
    for name in ("pixi.toml", "pixi.lock", "bindings/merged_build.sh"):
        (repo / name).write_text("inert test fixture\n")
    (repo / ".gitignore").write_text("__pycache__/\npython/mojolearn/identical/\npython/mojolearn/host/\n")
    (repo / "tracked.txt").write_text("frozen source\n")
    (repo / "tools/record_identity_column.sh").write_text(
        '#!/bin/bash\nprintf "record\\n" >> "$FAKE_EVENTS"\nexit "${FAKE_RECORD_RC:-0}"\n'
    )
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
        "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture")
    sha = git(repo, "rev-parse", "HEAD")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    for tool in ("pixi", "nvidia-smi", "rocm-smi", "fake-python", "sha256sum", "flock", "mojo"):
        p = bindir / tool
        p.write_text("#!" + sys.executable + "\n" + SHIM)
        p.chmod(0o755)
    # These are already installed artifacts, including the untracked math helper
    # found on the AMD recovery box. They must survive failed qualification.
    for directory in ("identical", "host"):
        d = repo / "python/mojolearn" / directory
        d.mkdir()
        (d / "old.so").write_bytes(b"previous " + directory.encode())
    (repo / "python/mojolearn/.libs/libMojolearnMath.so").write_bytes(b"previous math")
    env = dict(os.environ)
    for key in tuple(env):
        if key.startswith(("MOJOLEARN_", "BOX_", "FAKE_")):
            del env[key]
    # BOX_TOOL_PATH: on a GPU pod a real /root/.pixi/bin/pixi would otherwise
    # shadow the shim (L4 runs: "pixi rc=1 ... REFUSED pixi install failed")
    env.update(BOX_REPO_DIR=str(repo), BOX_OUT_DIR=str(tmp_path / "run"), BOX_RECORD="0",
               BOX_TOOL_PATH=str(bindir),
               PATH=str(bindir) + os.pathsep + env["PATH"], FAKE_EVENTS=str(tmp_path / "events"),
               FAKE_PYTHON=str(bindir / "fake-python"), REAL_PYTHON=sys.executable)
    return repo, sha, env


def run(box, vendor="nvidia", **overrides):
    repo, sha, env = box
    env = dict(env, **overrides)
    result = subprocess.run(["bash", str(CONTROLLER), vendor, sha], env=env,
                            text=True, capture_output=True, timeout=20)
    output = Path(env["BOX_OUT_DIR"])
    log = (output / "job.log").read_text() if (output / "job.log").exists() else result.stderr
    events = Path(env["FAKE_EVENTS"])
    return result.returncode, log, events.read_text() if events.exists() else "", output


def test_default_builds_identical_host_serially_and_retains_previous(box):
    repo, _, _ = box
    rc, log, events, out = run(box)
    assert rc == 0, log
    assert (out / "build-status.tsv").read_text().splitlines() == [
        "build\texit_code", "gpu-identical\t0", "host\t0"]
    assert "fast" not in events
    assert "JOB " not in log
    assert "RECORD SKIP BOX_RECORD=0" in log
    for d in ("identical", "host"):
        assert (out / "previous" / d / "old.so").read_bytes() == b"previous " + d.encode()
        assert not (repo / "python/mojolearn" / d / "old.so").exists()
    assert (out / "previous/libMojolearnMath.so").read_bytes() == b"previous math"
    assert (out / "controller.sh").read_bytes() == CONTROLLER.read_bytes()
    assert "inert compiler fixture" in (out / "compiler-version.txt").read_text()
    assert "pixi.lock" in (out / "toolchain-input-sha256.txt").read_text()
    assert "MOJOLEARN_TARGET_COLUMN=cpu" in (out / "build-host.command").read_text()
    assert (out / "exit-code").read_text() == "0\n"


@pytest.mark.parametrize("failed", ["IDENTICAL", "HOST"])
def test_build_failure_never_verifies_or_activates(box, failed):
    repo, _, _ = box
    rc, log, events, out = run(box, **{f"FAKE_BUILD_{failed}_RC": "137"})
    assert rc != 0, log
    assert "137" in (out / "build-status.tsv").read_text()
    assert '"verify"' not in events and "record\n" not in events
    for d in ("identical", "host"):
        assert (repo / "python/mojolearn" / d / "old.so").exists()
    assert not (out / "previous").exists()


@pytest.mark.parametrize("setting,value", [("FAKE_INSTALL_RC", "5"), ("FAKE_MATH_RC", "5"),
                                          ("FAKE_INCOMPLETE", "identical"), ("FAKE_LOCK_RC", "1")])
def test_failed_prerequisite_never_verifies(box, setting, value):
    rc, log, events, out = run(box, **{setting: value})
    assert rc != 0, log
    assert '"verify"' not in events
    assert not (out / "previous").exists()


@pytest.mark.parametrize("dirt", ["tracked", "index", "untracked", "wrong-commit"])
def test_checkout_is_refused_without_mutation(box, dirt):
    repo, _, _ = box
    path = repo / ("new.mojo" if dirt == "untracked" else "tracked.txt")
    path.write_text("preserve this\n")
    if dirt in {"index", "wrong-commit"}:
        git(repo, "add", str(path))
    if dirt == "wrong-commit":
        git(repo, "-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
            "-c", "core.hooksPath=/dev/null", "commit", "-qm", "other commit")
    head = git(repo, "rev-parse", "HEAD")
    rc, log, events, _ = run(box)
    assert rc != 0, log
    assert not events or '"install"' not in events
    assert path.read_text() == "preserve this\n"
    assert git(repo, "rev-parse", "HEAD") == head


def test_existing_output_directory_is_untouched(box):
    _, _, env = box
    out = Path(env["BOX_OUT_DIR"])
    out.mkdir()
    (out / "job.log").write_text("previous run\n")
    rc, _, events, _ = run(box)
    assert rc != 0
    assert (out / "job.log").read_text() == "previous run\n"
    assert not events


@pytest.mark.parametrize("builds", ["host", "gpu-identical", "gpu-fast", "host host gpu-identical"])
def test_incomplete_or_invalid_selection_is_refused(box, builds):
    rc, log, events, _ = run(box, BOX_BUILDS=builds)
    assert rc != 0, log
    assert '"install"' not in events


def test_verify_failure_stops_recording_and_jobs(box):
    rc, log, events, out = run(box, FAKE_VERIFY_RC="7", BOX_RECORD="1", BOX_JOBS="dist")
    assert rc != 0, log
    assert (out / "verify-status.tsv").read_text() == "identical\t7\n"
    assert "record\n" not in events and "JOB dist start" not in log


def test_record_failure_is_nonzero(box):
    rc, log, events, out = run(box, BOX_RECORD="1", FAKE_RECORD_RC="8")
    assert rc != 0, log
    assert "record\n" in events
    assert (out / "record-exit-code").read_text() == "8\n"


def test_explicit_job_remains_opt_in(box):
    rc, log, _, _ = run(box, BOX_JOBS="dist")
    assert rc == 0, log
    assert "JOB dist start" in log and "DIST SKIP gpus=1" in log


@pytest.mark.parametrize("vendor,gpu_workers,overrides", [
    ("nvidia", "2", {}),
    ("amd", "8", {}),
    ("amd", "4", {"MOJOLEARN_AMD_COMPILE_JOBS": "4"}),
    ("amd", "6", {"BOX_GPU_COMPILE_JOBS": "6", "BOX_HOST_COMPILE_JOBS": "3"}),
])
def test_per_set_compiler_workers_follow_release_policy(box, vendor, gpu_workers, overrides):
    # A generic inherited one-worker cap reproduced the release builder's known
    # AMD merged compiler crash. It must not override per-set defaults.
    rc, log, events, out = run(box, vendor=vendor, MOJOLEARN_COMPILE_JOBS="1", **overrides)
    assert rc == 0, log
    builds = [json.loads(line) for line in events.splitlines() if line.startswith("[")]
    builds = [row for row in builds if row[0] == "pixi" and "bindings/merged_build.sh" in row[1]]
    host_workers = overrides.get("BOX_HOST_COMPILE_JOBS", "2")
    assert [row[3] for row in builds] == [gpu_workers, host_workers]
    for tag, workers in (("gpu-identical", gpu_workers), ("host", host_workers)):
        assert f"MOJOLEARN_COMPILE_JOBS={workers}" in (out / f"build-{tag}.command").read_text()
    assert f"gpu_compile_jobs\t{gpu_workers}\n" in (out / "run.tsv").read_text()
    assert f"host_compile_jobs\t{host_workers}\n" in (out / "run.tsv").read_text()


@pytest.mark.parametrize("setting,value", [("BOX_GPU_COMPILE_JOBS", "0"), ("BOX_HOST_COMPILE_JOBS", "33")])
def test_invalid_per_set_worker_count_refuses_before_install(box, setting, value):
    rc, log, events, _ = run(box, **{setting: value})
    assert rc != 0, log
    assert '"install"' not in events
