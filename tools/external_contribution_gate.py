#!/usr/bin/env python3
"""Trusted-base external-PR admission and isolated hosted CPU checks.

`plan` reads GitHub metadata only. It never checks out/imports PR code,
dispatches a workflow, leases hardware, comments, or merges. Its typed GPU
request is an integration hook, not evidence that a controller exists.
`cpu` must run only in the unprivileged pull_request hosted job. It runs
trusted-base checks against a separate candidate checkout. Never invoke
`cpu` from pull_request_target or on a trusted/self-hosted GPU machine.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request

POLICY_VERSION = "external-performance.v1"
TRUSTED_ASSOCIATIONS = {"OWNER", "MEMBER", "COLLABORATOR"}
SHA = re.compile(r"[0-9a-f]{40}\Z")
REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+\Z")
MAX_FILES = 500
MAX_SOURCE_BYTES = 2 * 1024 * 1024
EXACT_OPTIMIZATIONS = {
    "core/gemm.mojo": "matrix", "core/gram_splitk.mojo": "matrix",
    "neighbors/checks/pinned_distance_tile.mojo": "knn",
    "neighbors/impl/neighbors/detail/knn_brute_force.mojo": "knn",
    "neighbors/impl/neighbors/detail/fused_l2_knn.mojo": "knn",
    "umap/optimizer.mojo": "umap", "umap/optimizer_fast.mojo": "umap",
    "umap/spectral_init.mojo": "umap", "umap/transform.mojo": "umap",
}
IMPLEMENTATION_PREFIXES = {"mamba/impl/": "mamba", "transformer/impl/": "transformer"}
RECIPES = {
    "matrix": ["core/gemm_identity_check.mojo", "bench/linalg_price_main.mojo"],
    "knn": ["neighbors/checks/knn_identity_check.mojo", "bench/identity_price_main.mojo"],
    "umap": ["umap/checks/identity_check.mojo", "umap/checks/identity_broader_check.mojo",
             "umap/checks/transform_check.mojo", "tools/umap_transform_quality_check.py"],
    "mamba": ["tools/mamba_backward_certify.sh", "python/mojolearn/tests/test_mamba_surface.py"],
    "transformer": ["python/mojolearn/tests/test_transformer_surface.py"],
}
HELPER_TESTS = (
    ("umap_identity_compare.py", "test_umap_identity_compare.py"),
    ("mamba_backward_identity.py", "test_mamba_backward_identity.py"),
)


def policy_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def checked_sha(value):
    if not isinstance(value, str) or not SHA.fullmatch(value):
        raise ValueError("Invalid commit SHA")
    return value


def safe_path(name):
    if not isinstance(name, str) or not name or "\\" in name or any(ord(c) < 32 for c in name):
        raise ValueError("Invalid repository path")
    path = PurePosixPath(name)
    if path.is_absolute() or any(p in (".", "..", "") for p in name.split("/")):
        raise ValueError("Unsafe repository path")
    return name


def classify(files):
    recipes, review = set(), []
    for item in files:
        name = safe_path(item["filename"])
        if item.get("status") != "modified":
            review.append({"path": name, "reason": "added, removed, renamed or unknown-status file"})
            continue
        recipe = EXACT_OPTIMIZATIONS.get(name)
        if recipe is None and name.endswith(".mojo"):
            recipe = next((v for p, v in IMPLEMENTATION_PREFIXES.items() if name.startswith(p)), None)
        if recipe is None:
            review.append({"path": name, "reason": "outside existing-implementation optimization allowlist"})
        else:
            recipes.add(recipe)
    return sorted(recipes), review


def build_plan(event, live, files, repository):
    """Fail closed on stale heads, incomplete file lists and another repository."""
    if not REPOSITORY.fullmatch(repository):
        raise ValueError("Invalid repository")
    original = event["pull_request"]
    if live["number"] != original["number"] or live["base"]["repo"]["full_name"] != repository:
        raise ValueError("PR/repository mismatch")
    head = checked_sha(live["head"]["sha"])
    base = checked_sha(live["base"]["sha"])
    if head != checked_sha(original["head"]["sha"]):
        raise ValueError("PR head changed; wait for the newer event")
    if base != checked_sha(original["base"]["sha"]):
        raise ValueError("PR base changed; regenerate against the current base")
    if live.get("state") != "open":
        raise ValueError("PR is no longer open")
    association = live.get("author_association", "NONE")
    exempt = association in TRUSTED_ASSOCIATIONS
    count = live.get("changed_files")
    if not isinstance(count, int) or count < 1 or count > MAX_FILES or len(files) != count:
        raise ValueError("Changed-file list missing, too large or incomplete")
    if len({f["filename"] for f in files}) != len(files):
        raise ValueError("Duplicate changed-file metadata")
    recipes, review = classify(files)
    if live["base"]["ref"] != event["repository"]["default_branch"]:
        review.append({"reason": "target is not the default branch"})
    if live.get("draft"):
        review.append({"reason": "draft pull request"})
    eligible = bool(recipes) and not review and not exempt
    state = "MAINTAINER_EXEMPT" if exempt else ("GPU_PENDING" if eligible else "REVIEW_REQUIRED")
    return {
        "schema": POLICY_VERSION, "policy_sha256": policy_hash(), "state": state,
        "repository": repository, "pull_request": live["number"], "base_sha": base, "head_sha": head,
        "author_association": association, "changed_files": files, "review_reasons": review,
        "eligible_for_gpu_evaluation": eligible,
        "cpu_evidence": "separate unprivileged external CPU job; this metadata job executes no candidate code",
        "gpu_evidence": "NOT_RUN", "automerge_enabled": False,
        "gpu_request": {
            "schema": "mojolearn.external-gpu-request.v1", "enabled": False,
            "repository": repository, "pull_request": live["number"], "base_sha": base, "head_sha": head,
            "policy_sha256": policy_hash(), "recipe_ids": recipes,
            "trusted_base_entrypoints": {r: RECIPES[r] for r in recipes},
            "required_devices": ["apple", "nvidia", "amd"], "modes": ["fast", "identical"],
            "required_evidence": ["exact source and binary hashes", "correctness and negative controls",
                                  "legacy IDENTICAL bytes unchanged", "cross-vendor certificates",
                                  "same-device interleaved base/head samples >= 9",
                                  "retained raw outputs, timings and worker destruction receipt"],
            "execution_constraints": {"disposable_vm": True, "provider_credentials_in_worker": False,
                                      "github_write_token_in_worker": False, "release_credentials": False,
                                      "shared_cache": False, "persistent_self_hosted_runner": False,
                                      "workers_concurrent": 1, "cpu_threads": 4},
            "missing_configuration": ["isolated GPU controller and disposable vendor workers",
                                      "administrator-defined recurring budget and quotas",
                                      "trusted GitHub App result publisher and exact-SHA verification",
                                      "hosted/disposable Apple GPU capacity", "required-check/ruleset policy"],
        },
        "scope": "Admission is not numerical approval, a performance result or merge authorization",
    }


def api(path):
    token = os.environ.get("GH_TOKEN")
    if not token:
        raise ValueError("Read-only GH_TOKEN required for metadata admission")
    request = urllib.request.Request("https://api.github.com/" + path,
        headers={"Authorization": "Bearer " + token, "Accept": "application/vnd.github+json",
                 "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "mojolearn-external-admission"})
    with urllib.request.urlopen(request, timeout=30) as response:
        data = response.read(8 * 1024 * 1024 + 1)
    if len(data) > 8 * 1024 * 1024:
        raise ValueError("Oversized GitHub metadata response")
    return json.loads(data)


def admission(args):
    event = json.loads(args.event.read_text())
    repository = os.environ["GITHUB_REPOSITORY"]
    if not REPOSITORY.fullmatch(repository):
        raise ValueError("Invalid repository")
    number = event["pull_request"]["number"]
    if not isinstance(number, int) or number < 1:
        raise ValueError("Invalid PR number")
    live = api(f"repos/{repository}/pulls/{number}")
    if not 1 <= live.get("changed_files", 0) <= MAX_FILES:
        raise ValueError("PR exceeds bounded automated admission")
    files = []
    for page in range(1, 7):
        items = api(f"repos/{repository}/pulls/{number}/files?per_page=100&page={page}")
        if not isinstance(items, list):
            raise ValueError("Malformed changed-file response")
        files.extend({k: item[k] for k in ("filename", "status", "sha") if k in item} for item in items)
        if len(items) < 100:
            break
    plan = build_plan(event, live, files, repository)
    args.output.write_text(json.dumps(plan, indent=2) + "\n")
    # Controller hook: a data-only request. A future controller must independently
    # refresh GitHub metadata and trusted policy; never execute artifact strings.
    if args.request_output:
        args.request_output.write_text(json.dumps(plan["gpu_request"], indent=2) + "\n")
    if args.summary:
        with args.summary.open("a") as out:
            out.write("## External contribution admission\n\n")
            out.write(f"State: **{plan['state']}**. Changed files: {len(files)}.\n\n")
            out.write(f"Base `{plan['base_sha']}`; head `{plan['head_sha']}`.\n\n")
            out.write("GPU testing: **not run**. Automatic merge: **disabled**. "
                      "The separate hosted CPU job reports its own checks.\n")
    print(json.dumps({k: plan[k] for k in ("state", "base_sha", "head_sha", "gpu_evidence")}))


def candidate_file(root, name):
    path = root / safe_path(name)
    resolved = path.resolve()
    if not resolved.is_relative_to(root.resolve()) or path.is_symlink():
        raise ValueError("Candidate symlink escapes or replaces a checked file")
    if path.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Candidate checked file exceeds 2 MiB")
    return path


def cpu_checks(args):
    if os.environ.get("GITHUB_EVENT_NAME") != "pull_request":
        raise ValueError("CPU candidate checks require the unprivileged pull_request event")
    trusted, candidate = args.trusted.resolve(), args.candidate.resolve()
    if trusted == candidate or not Path(__file__).resolve().is_relative_to(trusted):
        raise ValueError("Gate must execute from the separate trusted-base checkout")
    event = json.loads(args.event.read_text())
    expected = checked_sha(event["pull_request"]["head"]["sha"])
    actual = subprocess.check_output(["git", "-C", str(candidate), "rev-parse", "HEAD"], text=True).strip()
    if actual != expected:
        raise ValueError("Candidate checkout differs from event head")
    base = checked_sha(event["pull_request"]["base"]["sha"])
    trusted_head = subprocess.check_output(["git", "-C", str(trusted), "rev-parse", "HEAD"], text=True).strip()
    if trusted_head != base:
        raise ValueError("Trusted checkout differs from event base")
    env = dict(os.environ)
    for name in list(env):
        if name.startswith(("GITHUB_TOKEN", "GH_TOKEN", "ACTIONS_RUNTIME_", "ACTIONS_ID_TOKEN_")) or name in ("PYTHONPATH", "PYTHONHOME"):
            env.pop(name)
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONNOUSERSITE="1")
    records = []

    def run(label, command, cwd):
        result = subprocess.run(command, cwd=cwd, env=env, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
        records.append({"check": label, "exit_code": result.returncode, "output": result.stdout[-40000:]})

    report = {"schema": POLICY_VERSION, "base_sha": base, "head_sha": expected,
              "policy_sha256": policy_hash(), "cpu_checks": records,
              "gpu_evidence": "NOT_RUN", "automerge_enabled": False, "status": "RUNNING"}
    try:
        # Reject escaping symlinks before trusted parsers walk candidate source.
        for directory in ("python", "bindings", "packaging"):
            root = candidate / directory
            if root.is_symlink():
                raise ValueError("Candidate source directory is a symlink")
            for path in root.rglob("*"):
                if path.is_symlink():
                    raise ValueError("Candidate checked source tree contains a symlink")
        for name in ("CITATION.cff", "python/pyproject.toml", "python/mojolearn/_version.py",
                     "python/setup.py", "python/mojolearn_diagnostics.py"):
            candidate_file(candidate, name)
        # Execute the base revision of existing static gates, never PR edits to
        # wheel_ci.py. Those checks parse/read candidate files without importing.
        gate = trusted / "packaging/wheel_ci.py"
        run("version agreement", [sys.executable, "-I", str(gate), "versions", str(candidate)], trusted)
        run("CPU build baselines", [sys.executable, "-I", str(gate), "pins", str(candidate)], trusted)
        run("package import inventory", [sys.executable, "-I", str(gate), "inventory",
            str(candidate / "python/mojolearn"), str(candidate / "python/mojolearn_diagnostics.py")], trusted)
        # Read candidate Python/TOML; do not import the GPU package on this host.
        import tomllib
        for path in (candidate / "python/mojolearn").rglob("*.py"):
            ast.parse(candidate_file(candidate, str(path.relative_to(candidate))).read_text(), filename=str(path))
        tomllib.loads(candidate_file(candidate, "python/pyproject.toml").read_text())
        records.append({"check": "Python AST and package TOML", "exit_code": 0})
        for path in sorted((candidate / "bindings").glob("build*.sh")):
            candidate_file(candidate, str(path.relative_to(candidate)))
            run("shell syntax " + path.name, ["bash", "-n", str(path)], trusted)
        # Candidate helper code executes ONLY in this hosted, no-secrets job.
        # Copy trusted-base negative-control suites alongside candidate helpers;
        # candidate edits to the tests cannot turn a red helper into green.
        for helper, suite in HELPER_TESTS:
            with tempfile.TemporaryDirectory(prefix="external-helper-") as tmp:
                staging = Path(tmp)
                shutil.copyfile(candidate_file(candidate, "tools/" + helper), staging / helper)
                shutil.copyfile(trusted / "tools" / suite, staging / suite)
                # unittest must not see bootstrap's positional arguments.
                bootstrap = "import runpy,sys; p,f=sys.argv[1:]; sys.path.insert(0,p); sys.argv=[f]; runpy.run_path(f,run_name='__main__')"
                run("trusted negative controls: " + helper,
                    [sys.executable, "-I", "-c", bootstrap, str(staging), str(staging / suite)], staging)
        report["status"] = "PASS" if all(r["exit_code"] == 0 for r in records) else "FAIL"
    except Exception as exc:
        report["status"] = "FAIL"
        report["error"] = str(exc)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    if args.summary:
        with args.summary.open("a") as out:
            out.write("## External hosted CPU checks\n\n")
            out.write(f"**{report['status']}**, {len(records)} checks. GPU testing: **not run**.\n\n")
            out.write("Source/packaging and helper checks are not GPU correctness or performance evidence. "
                      "No automatic merge is authorized. See the JSON artifact for individual results.\n")
    return 0 if report["status"] == "PASS" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("plan", "cpu"):
        p = sub.add_parser(command)
        p.add_argument("--event", type=Path, required=True)
        p.add_argument("--output", type=Path, required=True)
        p.add_argument("--summary", type=Path)
        if command == "plan":
            p.add_argument("--request-output", type=Path)
        else:
            p.add_argument("--trusted", type=Path, required=True)
            p.add_argument("--candidate", type=Path, required=True)
    args = parser.parse_args()
    return admission(args) if args.command == "plan" else cpu_checks(args)


if __name__ == "__main__":
    raise SystemExit(main())
