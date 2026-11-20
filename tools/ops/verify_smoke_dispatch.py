#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Start .github/workflows/verify-smoke.yml on one measurement build's packed core wheel.

    python3 tools/ops/verify_smoke_dispatch.py --wheel-dir <work dir>/<commit12>/linux/final [--ref BRANCH] [--dry-run]
    python3 tools/ops/verify_smoke_dispatch.py --workflow fast-trial --wheel-dir ... [--families 0xfffffff]
        [--exact-families MASK] [--shards N]     (the per-estimator fast trial, .github/workflows/fast-trial.yml)
    python3 tools/ops/verify_smoke_dispatch.py --workflow cpu-coverage --wheel-dir ... [--shards 4]
        (the CPU coverage smoke of every public class: fast-trial.yml with mode=coverage)

The wheels are packed on this Mac (tools/release.py candidate, linux-pack); release-linux-build.yml uploads no wheel.
So, as tools/release_identity_wheels.py does for cpu-identity-gate.yml, the core wheel goes to R2 (credentials stay
on this Mac) and the workflow gets a presigned GET, its name and its sha256. --ref is the pushed branch whose
workflow file and smoke script run (default: this checkout's branch); the source tested is the wheel's commit
(candidate.json source_commit). tools/release_candidate.py calls dispatch() after the pack, fail soft: the smoke
reports and gates nothing. Prints `VERIFY-SMOKE dispatched ...` with the token its run-name carries.

THE CENSUS SKIP RULE (fb4-census, 2026-11-17). The smoke's `frames` step (the Python frames census) is skipped, with a
SUMMARY line naming the wheel whose census stands, when no file under python/mojolearn (at the wheel's commit) and none
of the census's own files (CENSUS_FILES, at --ref, whose tooling runs) changed since the last census dispatched for
that --ref: their git object ids are compared with the record in CENSUS_RECORDS. Default: the census runs whenever
anything changed (or no record exists). --census forces it; --no-census skips it. A run that takes the census is
recorded when it is dispatched; a run that skips it leaves the record as it was.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = "verify-smoke.yml"
FAST_TRIAL_WORKFLOW = "fast-trial.yml"
#: the CPU coverage smoke's matrix runners: about 250 public classes, one process each, on 4-core runners
COVERAGE_SHARDS = 4
EVIDENCE = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", os.path.expanduser("~/mojolearn-evidence")))
#: the last census dispatched per --ref: {ref: {commit, wheel, sha256, inputs, token, made}}
CENSUS_RECORDS = EVIDENCE / "verify-smoke" / "census-records.json"
#: the census's own files (read at --ref): the tool, its fixtures, the smoke step that runs it, the surface it reads
CENSUS_FILES = ("tools/ops/python_frames_census.py", "tools/ops/census_fixtures", "tools/ops/wheel_verify_smoke.py",
                "tools/ops/wheel_verify_smoke.sh", "tools/public_api_surface.py")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def core_wheel(wheel_dir):
    found = sorted(Path(wheel_dir).glob("mojolearn_core-*.whl"))
    if len(found) != 1:
        raise SystemExit(f"verify smoke: expected one mojolearn_core-*.whl in {wheel_dir}, found {len(found)}")
    return found[0]


def current_branch():
    return subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, capture_output=True,
                          text=True).stdout.strip()


def object_id(rev, path, repo=ROOT):
    """The git object id of PATH at REV (a tree or a blob), or "" when it does not exist there."""
    p = subprocess.run(["git", "rev-parse", "-q", "--verify", f"{rev}:{path}"], cwd=repo, capture_output=True,
                       text=True)
    return p.stdout.strip() if p.returncode == 0 else ""


def tooling_rev(ref, repo=ROOT):
    """The commit the workflow's tooling checkout will run: origin/<ref> when the remote has it, else <ref>."""
    for rev in ((ref,) if ref == "HEAD" else (f"origin/{ref}", ref)):
        p = subprocess.run(["git", "rev-parse", "-q", "--verify", f"{rev}^{{commit}}"], cwd=repo,
                           capture_output=True, text=True)
        if p.returncode == 0:
            return p.stdout.strip()
    return ref


def census_inputs(commit, ref, repo=ROOT):
    """{path: object id}: python/mojolearn at the wheel's commit and the census's own files at --ref."""
    inputs = {"python/mojolearn": object_id(commit, "python/mojolearn", repo)}
    rev = tooling_rev(ref, repo)
    inputs.update({path: object_id(rev, path, repo) for path in CENSUS_FILES})
    return inputs


def load_records(path=CENSUS_RECORDS):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return {}


def census_decision(ref, inputs, census=None, records_path=CENSUS_RECORDS):
    """(run, the SUMMARY line the smoke prints when it skips). census: True forces, False skips, None decides."""
    last = load_records(records_path).get(ref)
    stands = (f"{last['wheel']} ({last['commit'][:12]}, run token {last.get('token', '?')}, {last.get('made', '?')})"
              if last else "none recorded")
    if census is False:
        return False, f"--no-census given; the census of {stands} on {ref} stands"
    if census is True:
        return True, ""
    if last and all(inputs.get(k) for k in inputs) and last.get("inputs") == inputs:
        return False, (f"no file under python/mojolearn and none of the census's files changed since the census of "
                       f"{stands} on {ref}; that census stands (--census forces it)")
    return True, ""


def record_census(ref, entry, records_path=CENSUS_RECORDS):
    records = load_records(records_path)
    records[ref] = entry
    Path(records_path).parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(records_path).with_suffix(".tmp")
    tmp.write_text(json.dumps(records, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, records_path)


def upload_wheel(wheel, key, days):
    """PUT the wheel to R2 at KEY (credentials stay on this Mac); returns a presigned GET valid for DAYS. The one
    upload path of every workflow that tests a packed measurement wheel (verify-smoke.yml, fast-trial.yml)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import bincache
    import box_persist
    creds = box_persist.load_creds()
    status = bincache.http_put(bincache.presign("PUT", key, 3600, creds), str(wheel))
    if status not in (200, 201):
        raise SystemExit(f"verify smoke: R2 upload of {wheel.name} failed (HTTP {status})")
    return bincache.presign("GET", key, int(days * 86400), creds)


def fast_trial_argv(ref, commit, wheel_name, url, digest, token, families, exact="", shards=1, tests_from="source",
                    mode="trial"):
    """The `gh workflow run fast-trial.yml` command (inputs as verify-smoke.yml's, plus the trial's). `mode` is sent
    only when it is not the default, so a --ref whose workflow predates the input still takes a trial."""
    return (["gh", "workflow", "run", FAST_TRIAL_WORKFLOW, "--ref", ref, "-f", f"ref={commit}",
             "-f", f"wheel_url={url}", "-f", f"wheel_name={wheel_name}", "-f", f"wheel_sha256={digest}",
             "-f", f"families={families}", "-f", f"exact_families={exact}", "-f", f"shards={int(shards)}",
             "-f", f"tests_from={tests_from}", "-f", f"token={token}"]
            + ([] if mode == "trial" else ["-f", f"mode={mode}"]))


def dispatch_fast_trial(wheel_dir, ref="", commit="", families="0xfffffff", exact="", shards=1, dry=False, days=2,
                        tests_from="source", mode="trial"):
    """Dispatch .github/workflows/fast-trial.yml on the core wheel (same upload and key as the smoke). mode="coverage"
    is the CPU coverage smoke (every public class and method, identical; artifact cpu-coverage-<commit>)."""
    label = "CPU-COVERAGE" if mode == "coverage" else "FAST-TRIAL"
    run_label = "cpu-coverage" if mode == "coverage" else "fast-trial"
    wheel = core_wheel(wheel_dir)
    if not commit:
        commit = json.loads((Path(wheel_dir) / "candidate.json").read_text())["source_commit"]
    ref = ref or current_branch()
    digest = sha256(wheel)
    token = hashlib.sha256(f"{run_label} {commit} {time.time()}".encode()).hexdigest()[:12]
    key = f"verify-smoke/{commit[:12]}/{wheel.name}"
    if dry:
        print(f"{label} would upload " + wheel.name + " to R2 " + key + " and run: "
              + " ".join(fast_trial_argv(ref, commit, wheel.name, "<presigned>", digest, token, families, exact,
                                         shards, tests_from, mode)))
        return token
    url = upload_wheel(wheel, key, days)
    p = subprocess.run(fast_trial_argv(ref, commit, wheel.name, url, digest, token, families, exact, shards,
                                       tests_from, mode),
                       cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"{run_label}: gh workflow run failed: {(p.stderr or p.stdout).strip()[-400:]}")
    print(f"{label} dispatched {FAST_TRIAL_WORKFLOW} (mode {mode}) on --ref {ref} for {commit[:12]} ({wheel.name}) "
          f"families {families} exact {exact or '-'} shards {shards}; run-name '{run_label} {commit} {families} {exact} "
          f"{token}'; artifact {run_label}-{commit}; find it: gh run list --workflow {FAST_TRIAL_WORKFLOW} -L 5")
    return token


def dispatch(wheel_dir, ref="", commit="", dry=False, days=2, census=None, record=True, python=None):
    """Upload the core wheel, dispatch the workflow; returns the token its run-name carries. census: None applies
    the skip rule, True forces the frames census, False skips it. record=False (the pre-build probe wheel, whose
    native code is an earlier build's) never records its census as the branch's."""
    wheel = core_wheel(wheel_dir)
    if not commit:
        commit = json.loads((Path(wheel_dir) / "candidate.json").read_text())["source_commit"]
    ref = ref or current_branch()
    digest = sha256(wheel)
    token = hashlib.sha256(f"{commit} {time.time()}".encode()).hexdigest()[:12]
    key = f"verify-smoke/{commit[:12]}/{wheel.name}"
    inputs = census_inputs(commit, ref)
    run_census, stands = census_decision(ref, inputs, census)
    census_args = [] if run_census else ["-f", f"census_stands={stands}"]
    print(f"VERIFY-SMOKE census: {'runs' if run_census else 'skipped: ' + stands}")
    if dry:
        print(f"VERIFY-SMOKE would upload {wheel.name} to R2 {key} and run: gh workflow run {WORKFLOW} --ref {ref} "
              f"-f ref={commit} -f wheel_url=<presigned> -f wheel_name={wheel.name} -f wheel_sha256={digest} "
              f"-f token={token}" + (" -f census_stands=..." if census_args else ""))
        return token
    url = upload_wheel(wheel, key, days)
    argv = ["gh", "workflow", "run", WORKFLOW, "--ref", ref, "-f", f"ref={commit}", "-f", f"wheel_url={url}",
            "-f", f"wheel_name={wheel.name}", "-f", f"wheel_sha256={digest}", "-f", f"token={token}"] + census_args
    if python:   # the workflow's own input (default: the release CPU column's)
        argv += ["-f", f"python={python}"]
    p = subprocess.run(argv, cwd=ROOT, capture_output=True, text=True)
    if p.returncode != 0:
        raise SystemExit(f"verify smoke: gh workflow run failed: {(p.stderr or p.stdout).strip()[-400:]}")
    if run_census and record:
        record_census(ref, dict(commit=commit, wheel=wheel.name, sha256=digest, inputs=inputs, token=token,
                                made=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    print(f"VERIFY-SMOKE dispatched {WORKFLOW} on --ref {ref} for {commit[:12]} ({wheel.name}); run-name "
          f"'verify-smoke {commit} {token}'; find it: gh run list --workflow {WORKFLOW} -L 5")
    return token


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--wheel-dir", required=True, help="<work dir>/<commit12>/linux/final of a measurement build")
    ap.add_argument("--ref", default="", help="pushed branch holding the workflow (default: this checkout's branch)")
    ap.add_argument("--commit", default="", help="the wheel's source commit (default: candidate.json)")
    ap.add_argument("--dry-run", action="store_true")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--census", dest="census", action="store_true", default=None,
                   help="take the frames census even when nothing it reads changed")
    g.add_argument("--no-census", dest="census", action="store_false", help="skip the frames census")
    ap.add_argument("--workflow", choices=("verify-smoke", "fast-trial", "cpu-coverage"), default="verify-smoke",
                    help="fast-trial: the per-estimator fast trial (.github/workflows/fast-trial.yml) on the same wheel; "
                         "cpu-coverage: the CPU coverage smoke of every public class (fast-trial.yml mode=coverage)")
    ap.add_argument("--families", default="0xfffffff", help="fast-trial: trial mask of door families")
    ap.add_argument("--exact-families", default="", help="fast-trial: a second trial with exactly this set")
    ap.add_argument("--shards", type=int, default=None,
                    help="fast-trial / cpu-coverage: matrix runners (default 1 for the trial, COVERAGE_SHARDS for coverage)")
    ap.add_argument("--python", default=None, help="verify-smoke only: the Python version to install the wheel under (e.g. 3.10)")
    ap.add_argument("--tests-from", choices=("source", "tooling"), default="source",
                    help="fast-trial: run the wheel commit's test file (source) or --ref's copy (tooling)")
    a = ap.parse_args(argv)
    if a.workflow == "fast-trial":
        dispatch_fast_trial(a.wheel_dir, a.ref, a.commit, a.families, a.exact_families, a.shards or 1, a.dry_run,
                            tests_from=a.tests_from)
        return 0
    if a.workflow == "cpu-coverage":
        dispatch_fast_trial(a.wheel_dir, a.ref, a.commit, shards=a.shards or COVERAGE_SHARDS, dry=a.dry_run,
                            mode="coverage")
        return 0
    dispatch(a.wheel_dir, a.ref, a.commit, a.dry_run, census=a.census, python=a.python)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
