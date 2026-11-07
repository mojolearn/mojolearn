#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE COMMAND FOR A RELEASE (2026-09-22): docs/RELEASE_CHECKLIST.md, walked.

    pixi run release 0.8.15 --dry-run          the plan and what is already done
    pixi run release 0.8.15 --status           per pipeline, leg and column: done, failed, owed
    pixi run release 0.8.15 --report           one row per requirement and step: passed, reused, failed,
                                               missing-execution, blocked-provenance, not-applicable, waived
    pixi run release 0.8.15                    everything up to publication
    pixi run release 0.8.15 --publish pypi     ... and the TestPyPI rehearsal, publish, finish line (hashes), record

NO GPU BY DEFAULT (Andrew 2026-10-13): identity and behavior are checked
before merge on NVIDIA, AMD, Apple and CPU; the release only builds, checks
artifact integrity and publishes. Linux builds where the source's binding
layout fits (see LINUX BUILD BACKENDS below), gpu-column-* are SKIPPED, macos-build packs
with --build-only, macos-smoke is tools/wheel_self_test.py (fresh venv,
`verify --self-test`), every publish takes release_linux_publish.sh
--preverified (validation_profile=preverified, qualification_reuse in the
manifest, no receipt), and the rehearsal skips python-tests and the GPU-box
leg dry-runs. --gpu-smoke restores the GPU route the step table below
describes (wheel smokes, light-smoke receipts, any build backend).

LINUX BUILD BACKENDS (--build-backend; default `github` for either binding
layout, Andrew 2026-10-26: compiling never uses paid rentals):
  layout       default   also allowed          without --gpu-smoke
  merged       github    cpu-box, gpu-legs     github, cpu-box
  per-binding  github    cpu-box, gpu-legs     github, cpu-box
  github    tools/release_github_build.py on GitHub's free runners (4 vCPU,
            15 GB), no rental. Merged: one shard job per merged compile (the
            cpu box's exact compile, so its bytes and keys), with swap sized
            from the compile's recorded peak (UNIT_PEAK_GIB); per-binding:
            each set's builds split over shard jobs.
  cpu-box   tools/release_linux_build.sh, one RunPod CPU pod per set, no GPU
            (a paid rental, opt-in by name). Merged: 16 vCPU cpu5m/cpu3m
            (>= 64 GB), one merged compile at a time under a 64 GiB cap;
            per-binding: 32 vCPU cpu5g..cpu3m.
  gpu-legs  the RunPod NVIDIA and DigitalOcean/Hot Aisle AMD GPU boxes.

SEVERABLE PIPELINES (2026-09-25). After the common steps a release is five
build pipelines that run at once in one invocation. Before any publication,
both vendor candidate gates must pass for the actual packed wheel hashes:
  macos       macos-build -> macos-smoke -> publish-macos
  core-linux  linux-builds -> linux-wait -> linux-assemble -> linux-pack
              -> publish-core-linux (mojolearn-core; FIRST on Linux)
  nvidia      gpu-column-nvidia -> publish-nvidia (after publish-core-linux; the one
              NVIDIA wheel carries native sm_89 and PTX sm_80, Andrew 2026-11-05)
  amd         gpu-column-amd -> publish-amd (after publish-core-linux)
  meta-linux  publish-meta-linux (the file-free Linux `mojolearn`; LAST, after
              publish-core-linux, publish-nvidia AND publish-amd)
ONE IDENTITY CHECK PER RELEASE (Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun).
The admitted reference table (tools/record_identity_column.sh on NVIDIA and AMD,
tools/admit_identity_columns.sh: NVIDIA == AMD) IS the release's identity check,
and every wheel smoke verifies the installed wheel against that table. The
release no longer re-fits the changed lanes on Apple (release-check), on the
installed GPU wheels (the column pass) or diffs them again (linux-joint-diff,
the macOS cross-check), and macos-self-test is gone: macos-smoke's self-test
job runs the same `verify --self-test` on the same wheel. The finish line and the record
cover whichever platforms are published, and run again when another is. The
run ends with every outcome and exits non-zero if any did not publish. A
pipeline is PIPELINES' named builds, checks and one publish.

THE SHIPPED SOURCE AND THE RELEASE TOOLING ARE TWO COMMITS. freeze-commit pins
the source commit in state.json and it never moves because main moved:
publication ships exactly it, and only `--refreeze` moves it to HEAD. This
script and everything it drives (the leg runners, the guards, the wheel smoke,
the publisher) run from THIS checkout, the tooling checkout, normally current
main. Steps that read the source (the macOS build, the pack,
the rehearsal) run in the source checkout: this one when
HEAD is the source commit, else a detached worktree of it under
<version>/source/<commit12>/ (or --source-checkout). A build leg's box gets
the SOURCE archive plus the route overlay (tools/release_tooling.py,
tools/route_overlay_lib.sh): the tooling copy of each box-side tool that
differs, never a file of the source inventory. Every leg, column and record
names both commits (source_commit, tooling_commit and the overlay digests).

RESULTS ARE KEYED TO WHAT THEY BUILT. A completed build leg of an earlier
freeze is taken under a new one when, and only when, its set identity
(release_reuse.set_identity: every binding identity of the set, the host
bindings and the runtime closure) and its build tooling digest equal this
freeze's; its bytes then enter the pack through reuse.json with their origin
(built from X, admitted for Y), and anything unequal or unreadable rebuilds.
A GPU column is taken only for a byte-identical wheel (sha256) and the same
lane selection, and is diffed again against this release's references. Every
leg and column result is recorded once in the PASS LEDGER in R2
(tools/release_ledger.py); a ledger PASS is admitted only when its evidence is
on this machine and verifies.

State lives in ~/mojolearn-evidence/release/<version>/state.json
(MOJOLEARN_EVIDENCE_ROOT moves it); everything after the freeze is under
<version>/<commit12>/. A step is done when its OUTPUT checks out (a wheel of the
right commit, a PASSED receipt for that wheel's sha256, the file on PyPI), not merely because the state file says
so. A rerun resumes only what is not done.

THE STEPS
  freeze-version     _version.py and pyproject.toml say <version> (edited if not)
  freeze-changelog   CHANGELOG.md has `## <version> (published YYYY-MM-DD)`; the
                     prose is the releaser's, so a missing entry stops here
  freeze-docs-facts  pixi run write-docs-facts (CITATION and marked spans)
  freeze-commit      commit exactly those files if they changed, push, and
                     freeze HEAD (the tree must be clean and pushed). Once
                     frozen, the four freeze steps only check the frozen source
  identity-at-freeze IDENTITY ONCE, ON THE FROZEN SOURCE (Andrew 2026-11-05), before anything is
                     built or rented: tools/identity_at_freeze.py compares the freeze with the commit
                     each identity column of the table at the freeze was last recorded at. Source-equal
                     (equal, or only record outputs differ): ADOPTED, no selector. Otherwise the
                     freeze's lane_select counts the owed lanes and, on nvidia, amd, cpu or apple, the
                     release HOLDS here: "IDENTITY RECORDED AT X, FREEZE IS Y: N lane(s) owed". Record
                     once at the freeze (tools/reference_refresh.sh), merge ref/<sha12>, --refreeze
  rehearsal          pixi run release-rehearsal in the source checkout, logs kept
  cross-compile      (core-linux, beside the rehearsal) dispatch
                     .github/workflows/cross-compile-check.yml on the frozen
                     commit against the previous release's source commit and
                     wait: every changed Linux GPU binding x tier x arch
                     compiled on GitHub's free runners in parallel. Any FAIL,
                     TIMEOUT, STALLED or MISSING job stops the Linux pipelines
                     by name before linux-builds rents a box
  reuse-plan         tools/release_reuse.py: for every binding of every set
                     (cuda sm_89, the cuda sm_80 PTX slot, hip gfx942, the host bindings,
                     the runtime closure, the macOS wheel) REUSE the bytes the
                     last PUBLISHED release shipped when the binding's identity
                     (source closure, toolchain, flags, builder scripts as they
                     run after the overlay, box image or Apple toolchain) is
                     unchanged, else BUILD
  linux-builds       a leg per Linux set that has a binding to BUILD, all
                     launched AT ONCE, detached, unless the leg is done at this
                     freeze or a completed leg of an earlier freeze has the same
                     set identity and build tooling (taken, not launched), on
                     the backend LINUX BUILD BACKENDS above names: `github`
                     (GitHub's free runners) by default, `cpu-box` (RunPod CPU
                     pods) by name. The
                     --gpu-smoke backend `gpu-legs` builds on the GPU boxes: RunPod
                     NVIDIA (walking NVIDIA_WALK on no stock) and a DigitalOcean
                     MI325X, or a Hot Aisle 1x MI300X when DigitalOcean has a GPU
                     droplet live or no token (--amd-build-provider do|hotaisle
                     or MOJOLEARN_AMD_PROVIDER pins one)
  macos-build        mac_slot --slots 4 run -- build_release_wheel.sh, byte LM on
  macos-smoke        default: wheel_self_test.py (fresh venv, verify --self-test);
                     --gpu-smoke: qualify_verifier_wheel.py --scope expanded under the Metal lock;
                     its self-test job is the bundled-table gate (0.8.35 shipped a
                     stale table): no PASSED receipt for these wheel bytes, no
                     publish-macos
  gate-apple         the installed-wheel framework adapter smoke on this Mac (tools/adapter_gate.py): the
                     macOS wheel in a fresh venv, torch and max installed from the shipped pins,
                     tools/adapter_wheel_smoke.py --device mps under the Metal lock; only PASSED passes
                     (NO_FRAMEWORK fails); receipt <version>/<commit12>/gate-apple/adapters/;
                     MOJOLEARN_RELEASE_ADAPTERS=0 skips it
  linux-wait         every launched leg finished, its proof complete for this
                     commit, its route overlay verified on the box, and every
                     host binding byte-identical across the legs (taken ones too);
                     a host byte mismatch with identical recorded inputs is
                     composed from the host owner leg (release_reuse.HOST_LEG)
                     by tools/release_host_composition.py into
                     <commit12>/host-composition/, receipt in the state
  linux-assemble     the set directories the packer packs: a leg's set, a taken
                     leg's set, or one synthesized from the published wheel
  linux-pack         pixi run -e pkg pack-linux-wheel --profile release-split
                     (--host-composition <receipt> when linux-wait composed)
                     (the core, the plugins and the Linux mojolearn), audit.sh,
                     strip, split_audit
  gpu-column-nvidia  } SKIPPED by default; with --gpu-smoke the NVIDIA and AMD wheel smokes AT ONCE (detached
  gpu-column-amd     } tools/release_wheel_smoke.sh, the core and both plugins
                       installed, verified against the admitted table; no
                       second identity pass of the changed lanes)
  testpypi-rehearsal BEFORE THE FIRST PUBLISH, inside --publish pypi only (Andrew 2026-11-06: everything that
                     can be checked before the real publish is checked before it). After both gates, the one
                     verification round and every check a publish needs: the exact gated wheels go to TestPyPI
                     through the release's own publisher (Trusted Publishing, environments testpypi[-core|
                     -nvidia|-amd]; no token); a file TestPyPI already holds with the same sha256 is not
                     uploaded again, the same file name with other bytes FAILS (TestPyPI never takes a name
                     twice: a re-freeze needs the next version). Then, with our project names served by
                     TestPyPI ONLY (a per-project routing index on 127.0.0.1; third-party projects from PyPI):
                     pip resolves `mojolearn`, every `mojolearn[<extra>]`, `mojolearn-core`, `mojolearn-nvidia`
                     and `mojolearn-amd` for each platform tag and Python version the wheels declare, to exactly
                     the expected set at the gated sha256; every file downloaded from TestPyPI hashes to the
                     gated wheel; the macOS wheel installs (hash-pinned) into a fresh venv on this Mac and
                     imports. A file over TestPyPI's upload limit is reported OWED and checked from the local
                     file. Any failure stops here: the real PyPI has seen nothing. Report at
                     <commit12>/testpypi-rehearsal/rehearsal.json (tools/release_testpypi_rehearsal.py)
  publish-core-linux }
  publish-nvidia     } tools/release_linux_publish.sh ... --preverified|--light-smoke, only
  publish-amd        }   with --publish (--preverified by default); the core first,
  publish-meta-linux }   the vendors next, the Linux mojolearn last; with --publish pypi each
  publish-macos      }   needs a passed testpypi-rehearsal of its exact bytes
  finish-line        HASH ONLY: per published platform, `pip download --no-deps` from the real PyPI, the
                     sha256 compared with the shipped wheel. NOTHING AFTER THE FIRST PUBLISH INSTALLS OR RUNS
                     ANYTHING: the install, the import and the package-set resolution ran in testpypi-rehearsal
  gate-boxes         ORCHESTRATOR ONLY (MOJOLEARN_ORCHESTRATOR=1; it rents): the boxes the selected box
                     steps (gate-*) still need: the shared NVIDIA pod
                     (tools/nvidia_central.sh up 1, out of stock retried there; lane release-gate synced to
                     the frozen source) and the DigitalOcean AMD droplet (tools/do_amd_steward.sh up 240), or
                     the live ones adopted; recorded in the state. While a box step is pending the release
                     holds the pod busy (one `nvidia_central.sh sh release-gate` hold, renewed) and renews
                     the droplet's lease (do_amd_steward.sh extend 120 every 85 minutes). --dry-run prints
                     what it would rent and for how long. Never an Apple machine
  gate-nvidia        } THE RELEASE GATE, after linux-pack and before any publish: tools/release_lq_smoke.py
  gate-amd           }   --gate runs tools/gate_smoke_box.py <vendor> <version> ... all ON the held lq box
                         (never the Mac, nothing rented): release-light-v1 completes eleven installed
                         correctness records, 24 autograd entries and three adapter phases against the
                         exact candidate wheels. A 30-minute total ceiling includes setup; timeouts fail.
                         No opponent/performance campaign or native compilation is part of publication.
                         Validated unchanged autograd artifact inputs reuse their original binaries.
                         Reports come home to <version>/<commit12>/gate-<vendor>/<artifact-id>/.
  published-apple    } after publish-<platform>, with --publish pypi: tools/release_published_install.py
  published-nvidia   }   check, on this Mac, no device, no box, nothing installed: pip resolves each
  published-amd      }   published package for the candidate's platform tags and downloads it; its
                         sha256 must equal the gated candidate wheel's (different bytes FAIL with both
                         hashes; nothing reruns), its RECORD must hold, and the gate's receipts are the
                         evidence by reference. Report at <commit12>/published-<vendor>/published-report.json,
                         where the ledger reads it (Andrew 2026-11-02: no re-verification of green work)
  published-smoke-*  REMOVED 2026-11-05 (--published-smoke refuses): no release step runs identity lanes
                     on a device after the first publish. Every identity-lane step (DEVICE_LANE_STEPS:
                     macos-smoke, gpu-column-*, gate-nvidia, gate-amd) settles before any publish starts
                     and refuses to launch once a platform of the release is published
  boxes-down         right after the gates (no published-* step needs a box): every box job's results home first
                     (tools/box_persist.py fetch --tag; none missing, or nothing is taken down), then what
                     gate-boxes rented taken down (nvidia_central.sh down, do_amd_steward.sh down); an
                     adopted box is released, left up
  record             THE RELEASE EVIDENCE SET (tools/release_evidence_set.py): deletes the previous content of the ONE
                     fixed directory bench/results/release_evidence/ and writes this release's evidence there
                     (manifest.json: version, frozen commit, wheel names and sha256, tools, every item's sha256,
                     producing step and the wheel bytes it checked; README.md; SUMMARY.md, the human-readable
                     account), commits it as "Release evidence for
                     <version>", pushes, and tags that commit v<version>. A missing required item fails the step by
                     name. An older release's evidence: `git show v<version>:bench/results/release_evidence/
                     manifest.json`. Waits for both gates and the published-* producers

Nothing is published without `--publish none|testpypi|pypi`; without it both
pipelines stop at their publish step and say so.

THE SPLIT LINUX PACKAGES (python/mojolearn/gpu_plugins.py). Linux ships as
four PyPI projects (Andrew 2026-10-20; mojolearn-nvidia-ptx folded back into
mojolearn-nvidia 2026-11-05): mojolearn-core (the Python package, host bindings
and runtime; no vendor requirement), mojolearn-nvidia (native sm_89 and PTX
sm_80 in one wheel) and mojolearn-amd (each requires mojolearn-core at its
version), and mojolearn, a
file-free Linux wheel requiring the core and both vendors. `pip install
mojolearn` installs everything; `pip install mojolearn-nvidia` or
`mojolearn-amd` installs the core and that vendor. THE CORE PUBLISHES FIRST,
then the VENDORS, and the Linux mojolearn LAST (it resolves only
once all of them are on the index). Each column gates its own plugin (its
wheel smoke against the admitted reference table); a failed candidate gate
holds every publication. Each package is its
own GitHub release and workflow dispatch (the workflow uploads it to its own
PyPI project); column and ledger entries stay keyed to one wheel's sha256
(the plugin's, with the core's recorded beside it). macOS is unchanged: the
one complete `mojolearn` wheel.
"""
import os
import argparse
import ssl as _ssl
# The pixi env's OpenSSL may carry a default CA path from the worktree the env was cloned from (2026-10-12: it
# pointed at a removed ~/mojolearn-wt/af-sym-hist/... and every GitHub call failed CERTIFICATE_VERIFY_FAILED).
# When the interpreter sees no CA bundle, point it at the system one so the release's own urllib calls verify.
if _ssl.get_default_verify_paths().cafile is None and not os.environ.get("SSL_CERT_FILE"):
    for _candidate in ("/private/etc/ssl/cert.pem", "/etc/ssl/cert.pem", "/etc/ssl/certs/ca-certificates.crt"):
        if os.path.exists(_candidate):
            os.environ["SSL_CERT_FILE"] = _candidate
            break

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import platform
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable


def ignore_hangup():
    """A real run survives its terminal closing (2026-11-05, interruption recovery): SIGHUP is ignored by
    the controller and, inherited, by the compiles and uploads it runs in the foreground, whose output it
    keeps teeing into their logs (say() drops what it cannot print). A kill or a reboot still stops it;
    every step then resumes from the state (docs/release/RELEASE_PLAN.md, interruption recovery)."""
    import signal
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
    except (ValueError, OSError, AttributeError):   # not the main thread, or no SIGHUP on this platform
        return False
    return True


def release_report_module():
    """tools/release_report.py of THIS tooling tree (docs/release/STATUS_REPORT.md)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import release_report
    return release_report


def release_plan_module():
    """tools/release_plan.py of THIS tooling tree (the decision table is tooling, not the frozen source)."""
    sys.path.insert(0, str(ROOT / "tools"))
    import release_plan
    return release_plan
PUBLISH_CHOICES = ("none", "testpypi", "pypi")
sys.path.insert(0, str(ROOT / "tools"))
import release_ledger  # noqa: E402
import release_reuse  # noqa: E402
import release_requirements  # noqa: E402
import release_tooling  # noqa: E402
import release_github_build  # noqa: E402
import runpy
GPU_PACKAGES = runpy.run_path(str(ROOT / "python/mojolearn/gpu_plugins.py"))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args, check=True):
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True,
                          check=check).stdout.strip()


def git_at(root, *args):
    p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    return p.stdout.strip() if p.returncode == 0 else None


def wheel_commit(path):
    """The source commit a wheel records (mojolearn/identity_columns/COMMIT), or None."""
    try:
        with zipfile.ZipFile(path) as z:
            return z.read("mojolearn/identity_columns/COMMIT").decode().strip()
    except (OSError, KeyError, zipfile.BadZipFile):
        return None


def adapter_gate_module():
    """tools/adapter_gate.py: the installed-wheel framework adapter smoke the gates share."""
    sys.path.insert(0, str(ROOT / "tools"))
    import adapter_gate
    return adapter_gate


def release_gate_module():
    """tools/ops/release_gate.py: the NVIDIA/AMD gate's box job through the supported wrappers (run_gate)."""
    import importlib.util
    module = sys.modules.get("release_gate_ops")
    if module is None:
        spec = importlib.util.spec_from_file_location("release_gate_ops", ROOT / "tools" / "ops" / "release_gate.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules["release_gate_ops"] = module
        spec.loader.exec_module(module)
    return module


def smoke_passed(results, wheel):
    """A light-smoke receipt that is PASSED, expanded, and about THIS wheel."""
    try:
        d = json.loads(Path(results).read_text())
    except (OSError, ValueError):
        return False
    return (d.get("status") == "PASSED" and d.get("scope") == "expanded"
            and d.get("wheel_sha256") == sha256(wheel)
            and d.get("source_commit") == wheel_commit(wheel))


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    tmp.replace(path)


# ---------------------------------------------------------------- freeze helpers
VERSION_RE = re.compile(r"^\d+\.\d+\.\d+(?:\.\d+)?(?:(?:a|b|rc)\d+)?$")


def set_version(root, version):
    """Write <version> into _version.py and pyproject.toml; return the files changed."""
    changed = []
    for rel, pattern, repl in (
            ("python/mojolearn/_version.py", r'^__version__ = "[^"]*"$', f'__version__ = "{version}"'),
            ("python/pyproject.toml", r'^version = "[^"]*"$', f'version = "{version}"')):
        path = Path(root) / rel
        text = path.read_text()
        new, n = re.subn(pattern, repl, text, count=1, flags=re.M)
        if n != 1:
            raise SystemExit(f"release: no version line in {rel}")
        if new != text:
            path.write_text(new)
            changed.append(rel)
    return changed


def changelog_date(root, version, text=None):
    """The `published YYYY-MM-DD` date of <version>'s CHANGELOG heading, or None."""
    text = text if text is not None else (Path(root) / "CHANGELOG.md").read_text()
    m = re.search(r"^## " + re.escape(version) + r" \(published (\d{4}-\d{2}-\d{2})\)\s*$", text, re.M)
    return m.group(1) if m else None


def file_at(commit, rel):
    """REL's text at COMMIT (the frozen source), or None."""
    p = subprocess.run(["git", "-C", str(ROOT), "show", f"{commit}:{rel}"], capture_output=True, text=True)
    return p.stdout if p.returncode == 0 else None


def release_files():
    """What a version bump edits by hand or by write-docs-facts' citation sync."""
    return ["python/mojolearn/_version.py", "python/pyproject.toml", "CHANGELOG.md", "CITATION.cff"]


def docs_fact_files():
    """The documents `write-docs-facts` rewrites (tools/docs_facts.py DOCS)."""
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        import docs_facts
        return list(docs_facts.DOCS)
    finally:
        sys.path.pop(0)


# ---------------------------------------------------------------- Linux build legs
class Leg:
    """One detached build process and where its release-build tree lands."""

    def __init__(self, name, vendor, arch, command, env, release_build, workdir, out_dir):
        self.name, self.vendor, self.arch = name, vendor, arch
        self.command, self.env = command, env
        self.release_build = Path(release_build)
        self.workdir = Path(workdir)
        #: what a relaunch moves aside (the leg refuses an existing output)
        self.out_dir = Path(out_dir)
        self.log = self.workdir / f"{name}.log"
        self.exit_file = self.workdir / f"{name}.exit"
        self.pid_file = self.workdir / f"{name}.pid"
        #: written beside the log at launch: both commits, the overlay, the identities
        self.provenance_file = self.workdir / f"{name}.provenance.json"
        self.provenance = None

    def describe(self):
        env = " ".join(f"{k}={shlex.quote(v)}" for k, v in self.env.items())
        return f"{env} {' '.join(shlex.quote(c) for c in self.command)}"

    def pid(self):
        try:
            return int(self.pid_file.read_text())
        except (OSError, ValueError):
            return None

    def running(self):
        pid = self.pid()
        if pid is None or self.exit_file.exists():
            return False
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            return True

    def exit_code(self):
        try:
            return int(self.exit_file.read_text().strip())
        except (OSError, ValueError):
            return None

    def proof_ok(self, commit):
        """The leg's build proof: complete, exit 0, this commit, this architecture's set."""
        try:
            proof = json.loads((self.release_build / "build" / "build-provenance.json").read_text())
        except (OSError, ValueError):
            return False
        from staged_native_admission import SCHEMA as STAGED_SCHEMA, complete_native_proof
        admitted = (complete_native_proof(proof) if proof.get("schema") == STAGED_SCHEMA else
                    proof.get("complete") is True and proof.get("build_exit") == 0)
        return (admitted and proof.get("source_commit") == commit
                and (self.release_build / "build" / "sets" / self.vendor / self.arch).is_dir())

    def done(self, commit):
        """Finished and its output checks out, so a (re)launch leaves it alone."""
        return self.exit_code() == 0 and self.proof_ok(commit)


class ColumnLeg(Leg):
    """One GPU wheel column (tools/release_wheel_smoke.sh), run detached like a
    build leg: its own session, its exit code written last, a failed attempt
    moved aside before a relaunch. `check` says whether its output (a PASSED
    receipt for this wheel, a column with no DIVERGENT cell) is already there,
    whatever the exit file says."""

    def __init__(self, name, vendor, command, workdir, out_dir, check):
        super().__init__(name, vendor, "", command, {}, out_dir, workdir, out_dir)
        self.check = check

    def done(self, commit):
        return bool(self.check())


def verify_leg_tree(release_build, vendor, arch, commit, exit_file=None):
    """(ok, why, proof sha256) for a finished build leg's output: exit 0, a
    complete proof of COMMIT, its set directory, and every binary the proof
    names on disk with the proof's sha256. Read-only."""
    rb = Path(release_build)
    if exit_file is not None:
        try:
            if int(Path(exit_file).read_text().strip()) != 0:
                return False, "it did not exit 0", None
        except (OSError, ValueError):
            return False, "it has no exit code (not finished)", None
    proof_path = rb / "build" / "build-provenance.json"
    proof = read_json(proof_path)
    if not proof:
        return False, f"no build proof at {proof_path}", None
    from staged_native_admission import SCHEMA as STAGED_SCHEMA, complete_native_proof
    admitted = (complete_native_proof(proof) if proof.get("schema") == STAGED_SCHEMA else
                proof.get("complete") is True and proof.get("build_exit") == 0)
    if not admitted or proof.get("source_commit") != commit:
        return False, "its native proof is incomplete, invalid or not of " + commit[:12], None
    if not (rb / "build" / "sets" / vendor / arch).is_dir():
        return False, "no set directory", None
    ext = proof.get("extensions") or {}
    if not ext:
        return False, "its proof names no binary", None
    for name, digest in ext.items():
        parts = Path(name).parts
        if parts[:3] != ("mojolearn", vendor, arch) or ".." in parts:
            return False, f"its proof names {name}, outside {vendor}/{arch}", None
        p = rb / "build" / "sets" / Path(*parts[1:])
        try:
            if sha256(p) != digest:
                return False, f"{name} on disk is not the proof's bytes", None
        except OSError:
            return False, f"{name} is missing", None
    return True, "", sha256(proof_path)


def overlay_verified(prov, out_dir):
    """(ok, why): every file the leg's route overlay carried is, per the box's
    own after-sha256 in <out>/route-overlay.txt, the tooling's bytes."""
    files = ((prov or {}).get("overlay") or {}).get("files") or {}
    if not files:
        return True, ""
    try:
        text = (Path(out_dir) / "route-overlay.txt").read_text()
    except OSError:
        return False, f"no route-overlay.txt in {out_dir} (the overlay was not applied)"
    for rel, row in files.items():
        if f"after {rel} {row['tooling_sha256']}" not in text.splitlines():
            return False, f"{rel} on the box was not the overlaid bytes ({out_dir}/route-overlay.txt)"
    return True, ""


def gate_runner_verified(out_dir, source_commit):
    """(ok, why): the gate runner that wrote <out_dir>/gate-smoke.json is the source commit's own (no
    route-overlay.txt and no route-overlay.json: no overlay, or a report from before the gate overlay existed)
    or the gate runner overlay (tools/ops/release_gate.py GATE_OVERLAY) the box verified: every file its
    manifest names is a GATE_OVERLAY file, its bytes are that file's at the manifest's tooling commit, and the
    box's route-overlay.txt shows those bytes after the overlay (overlay_verified)."""
    out_dir = Path(out_dir)
    doc = read_json(out_dir / "route-overlay.json")
    if not doc:
        if (out_dir / "route-overlay.txt").is_file():
            return False, f"{out_dir}/route-overlay.txt without its route-overlay.json manifest"
        return True, ""
    if doc.get("source_commit") != source_commit:
        return False, "the gate runner overlay was made for another source commit"
    tooling = doc.get("tooling_commit") or ""
    if not re.fullmatch(r"[0-9a-f]{40}", tooling):
        return False, "the gate runner overlay names no tooling commit"
    allowed = release_gate_module().GATE_OVERLAY
    for rel, row in (doc.get("files") or {}).items():
        if rel not in allowed or release_tooling.refused(rel):
            return False, f"the gate runner overlay carries {rel}, not a gate runner file"
        if release_tooling.sha(release_tooling.blob(ROOT, tooling, rel)) != row.get("tooling_sha256"):
            return False, f"the gate runner overlay's {rel} is not its bytes at {tooling[:12]}"
    return overlay_verified(dict(overlay=doc), out_dir)


def leg_layout(legs_dir, backend, name):
    """(release_build, out_dir) of a leg named `vendor-arch` on a backend."""
    legs_dir = Path(legs_dir)
    if backend in ("cpu-box", "github"):
        out = legs_dir / name
        return out / name / "release-build", out
    if name.startswith("cuda-"):
        out = legs_dir / name
        return out / "remote" / "release-build", out
    return legs_dir / name / "release-build", legs_dir / name


def leg_common_env(ctx):
    """What every Linux build leg is told about the source and the tooling:
    the source checkout its inventory is compared with, and the route overlay
    (Release.leg_env). A bare context carries none."""
    fn = getattr(ctx, "leg_env", None)
    return fn() if callable(fn) else {}


def gpu_legs(ctx):
    """THE DEFAULT ROUTE: two RunPod NVIDIA legs and one DigitalOcean MI325X leg,
    exactly docs/RELEASE_CHECKLIST.md section 2's commands, output directories
    pinned under the release directory so this script can find them. The AMD
    leg does not wait for NVIDIA: its core-host probe is skipped and
    pack_wheel.py compares every host binding across the legs (linux-wait
    names a mismatch first)."""
    legs_dir = ctx.rel / "legs"
    runpod = dict(MOJOLEARN_RUNPOD_KEY_FILE=os.path.expanduser(
        os.environ.get("MOJOLEARN_RUNPOD_KEY_FILE", "~/.mojolearn_runpod_key")),
        MOJOLEARN_NVIDIA_CAMPAIGN="7")
    common = leg_common_env(ctx)
    legs = []
    for arch, gpu in ((a, NVIDIA_WALK[a][0]) for a in ("sm_90a", "sm_89")):
        rb, out = leg_layout(legs_dir, "gpu-legs", f"cuda-{arch}")
        legs.append(Leg(f"cuda-{arch}", "cuda", arch,
                        ["sh", "tools/gemm_remote_leg.sh", "nvidia", "--payload", "mamba",
                         "--source-ref", ctx.commit, "--gpu", gpu, "--allow-concurrent", "--rent",
                         "--segment-lease", "120", "--dollar-cap", "15"],
                        dict(runpod, MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_GEMM_LEG_OUT=str(out), **common),
                        rb, legs_dir, out))
    # THE PTX SLOT needs no GPU to build (Andrew 2026-10-10: PTX is a normal target; no
    # flag): its leg is the CPU box route, like cpu_legs, with the same set layout.
    rb, out = leg_layout(legs_dir, "cpu-box", "cuda-sm_80")
    legs.append(Leg("cuda-sm_80", "cuda", "sm_80",
                    ["bash", "tools/release_linux_build.sh", ctx.commit, "--rent", "--archs", "sm_80",
                     *cpu_box_pod(ctx), "--out", str(out)],
                    {}, rb, legs_dir, out))
    route, why = amd_build_route(ctx)
    if route == "cpu-box":
        rb, out = leg_layout(legs_dir, "cpu-box", "hip-gfx942")
        command = ["bash", "tools/release_linux_build.sh", ctx.commit, "--rent", "--archs", "gfx942",
                   *cpu_box_pod(ctx, amd_small=True), "--out", str(out)]
        leg = Leg("hip-gfx942", "hip", "gfx942", command, {}, rb, legs_dir, out)
    else:
        command, env = amd_leg_command(ctx, route, legs_dir)
        rb, out = leg_layout(legs_dir, "gpu-legs", "hip-gfx942")
        leg = Leg("hip-gfx942", "hip", "gfx942", command, env, rb, legs_dir, out)
    leg.provider, leg.provider_reason = route, why
    legs.append(leg)
    return legs


# ---------------------------------------------------------------- the AMD build route
#: Where the AMD (hip gfx942) build leg rents. do: tools/do_release061_leg.sh on
#: a DigitalOcean MI325X, the route every release through 0.8.18 used.
#: hotaisle: tools/hotaisle_release_leg.sh on a Hot Aisle 1x MI300X, the same
#: remote build and the same output tree (2026-09-25). auto: do, unless
#: DigitalOcean has a GPU droplet live (the DigitalOcean leg refuses a rental
#: then, one GPU droplet at a time on the account) or no usable token.
#: cpu-box: tools/release_linux_build.sh --archs gfx942 on a RunPod CPU pod,
#: the opt-in CPU route for the AMD set alone (2026-09-26, 0.8.23:
#: DigitalOcean was unavailable and Hot Aisle had no MI300X). gfx942 is
#: compiled ahead of time from --target-accelerator, so no AMD GPU is needed to
#: build it; the AMD column still runs the wheel on a real MI300X. Never auto.
AMD_BUILD_PROVIDERS = ("auto", "do", "hotaisle", "cpu-box")
DO_LIVE_REFUSAL = "GPU droplet(s) already live"


def amd_build_want(args):
    return getattr(args, "amd_build_provider", None) or os.environ.get("MOJOLEARN_AMD_PROVIDER") or "auto"


def do_gpu_busy(token_file=None, api=None):
    """(busy, why) for the DigitalOcean AMD route, from one free GET. Never raises."""
    token_file = os.path.expanduser(token_file or os.environ.get("MOJOLEARN_DO_TOKEN_FILE", "~/.mojolearn_do_token"))
    api = api or os.environ.get("MOJOLEARN_DO_API", "https://api.digitalocean.com/v2")
    try:
        token = Path(token_file).read_text().strip()
    except OSError:
        return True, f"no DigitalOcean token at {token_file}"
    if not token:
        return True, f"the DigitalOcean token file {token_file} is empty"
    req = urllib.request.Request(api + "/droplets?per_page=200", headers={"Authorization": "Bearer " + token})
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            droplets = json.load(r).get("droplets") or []
    except Exception as exc:  # the leg would refuse the same way
        return True, f"the DigitalOcean droplet listing failed ({type(exc).__name__})"
    live = [f"{d.get('id')}:{d.get('name')}:{d.get('size_slug', '')}" for d in droplets
            if str(d.get("size_slug") or "").startswith("gpu-")]
    if live:
        return True, "DigitalOcean GPU droplet(s) live, so its leg would refuse: " + " ".join(live)
    return False, "no DigitalOcean GPU droplet live"


def amd_build_route(ctx):
    """(route, why): the provider the AMD build leg rents from, decided once per run."""
    cached = getattr(ctx, "_amd_route", None)
    if cached:
        return cached
    want = amd_build_want(ctx.args)
    if want not in AMD_BUILD_PROVIDERS:
        raise StepFailed(f"the AMD build provider must be one of {', '.join(AMD_BUILD_PROVIDERS)}, not {want!r}")
    if want != "auto":
        route = (want, f"pinned ({want})")
    else:
        probe = getattr(ctx, "amd_do_probe", None) or do_gpu_busy
        busy, why = probe()
        route = ("hotaisle", "auto: " + why) if busy else ("do", "auto: " + why)
    ctx._amd_route = route
    ctx.say(f"  AMD build leg: {route[0]} ({route[1]})")
    return route


def amd_leg_command(ctx, route, legs_dir):
    """The AMD build leg's command and environment on ROUTE; the same output tree either way."""
    if route == "hotaisle":
        cmd = ["bash", "tools/hotaisle_release_leg.sh", ctx.commit, "--rent"]
        env = dict(MOJOLEARN_RELEASE_RESULTS_ROOT=str(legs_dir))
    else:
        token = os.path.expanduser(os.environ.get("MOJOLEARN_DO_TOKEN_FILE", "~/.mojolearn_do_token"))
        cmd = ["bash", "tools/do_release061_leg.sh", ctx.commit, token, "--rent"]
        env = dict(MOJOLEARN_RELEASE_UBUNTU22="1", MOJOLEARN_RELEASE_RESULTS_ROOT=str(legs_dir))
    if ctx.args.amd_expect_from:
        cmd += ["--expect-from", ctx.args.amd_expect_from]
    env.update(leg_common_env(ctx))
    return cmd, env


def do_refused_live(log):
    try:
        return DO_LIVE_REFUSAL in log.read_text(errors="replace")
    except OSError:
        return False


#: RunPod stock comes and goes by GPU type. A leg or a smoke whose create is
#: answered "no instances currently available" walks to the next type of the
#: same architecture (0.8.16, 2026-09-23: the L40S leg failed twice on stock
#: and the RTX 4090 smoke once, each stopping the release).
NVIDIA_WALK = {
    "sm_90a": ["NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H100 PCIe", "NVIDIA H200"],
    "sm_89": ["NVIDIA L40S", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation", "NVIDIA GeForce RTX 4090"],
}
#: The NVIDIA column needs any NVIDIA GPU (it tests the wheel's set for the
#: GPU it lands on); the H100 and H200 close the walk when no sm_89 box has
#: stock (0.8.19: none did for hours).
SMOKE_WALK = ["NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation",
              "NVIDIA H100 80GB HBM3", "NVIDIA H200"]
NO_STOCK = "no instances currently available"
#: RunPod create failures that say nothing about our request, so the walk
#: tries the next GPU type (0.8.19: "Something went wrong" ended the sm_89
#: walk one type short of the RTX 4090).
CREATE_TRANSIENT = ("no longer any instances available", "Something went wrong",
                    "Please try again later")


def no_stock(log, *dirs):
    """Whether RunPod answered the create with no stock. The leg's log carries
    only "create returned HTTP 500"; RunPod's words are in the
    create_response.json it keeps in its output directory (0.8.19,
    2026-09-25: both NVIDIA legs failed on stock and never walked)."""
    texts = [log] + [p for d in dirs if d for p in Path(d).glob("**/create_response.json")]
    for t in texts:
        try:
            text = Path(t).read_text(errors="replace")
            if NO_STOCK in text or any(phrase in text for phrase in CREATE_TRANSIENT):
                return True
        except OSError:
            pass
    return False


#: Build routes by name. A route returns the Leg list for ctx; launch, wait,
#: pack and resume are route-independent. The GPU legs are the default (the
#: release builds on the GPUs it ships for); the CPU pods are opt-in by name.
#: RunPod CPU flavors a release build pod may land on, 32 vCPU each: general
#: purpose (4 GiB per vCPU) then memory optimized (8). 0.8.19's first CPU
#: launch found neither general-purpose flavor in stock; compute optimized
#: (2 GiB per vCPU) is left out, too little memory for 16 compile jobs.
CPU_BOX_FLAVORS = "cpu5g,cpu3g,cpu5m,cpu3m"
#: THE MERGED LAYOUT'S POD (2026-10-21): one merged compile per tier, measured on
#: the DO MI325X 2026-10-20 at up to 46.4 GB (identical + byte LM; fast 36.6,
#: host 6.9), so a merged set builds one file at a time under a 64 GiB cap on a
#: memory-optimized pod of >= 64 GB: 16 vCPU at RunPod's >= 4 GB per vCPU.
#: tools/release_linux_build.sh has the same defaults (a test holds them equal).
CPU_BOX_MERGED_FLAVORS = "cpu5m,cpu3m"
CPU_BOX_MERGED_VCPU = 16


def source_layout_env():
    """The environment the cpu-box route decides the layout under: this
    process's MOJOLEARN_MERGED_BINDINGS (release_linux_build.sh passes it on
    to the box when set)."""
    return {k: v for k, v in os.environ.items() if k == "MOJOLEARN_MERGED_BINDINGS"}


def linux_merged(ctx):
    """Whether the source commit's Linux sets build the merged layout on the
    cpu-box route (a Release caches it; a bare context reads the commit)."""
    known = getattr(ctx, "linux_merged", None)
    if isinstance(known, bool):
        return known
    return release_github_build.commit_merged(ctx.commit, source_layout_env())


def cpu_box_pod(ctx, amd_small=False):
    """tools/release_linux_build.sh's pod options for one set: the merged
    layout's memory pod, else the per-binding flavors (32 vCPU; amd_small: the
    gpu-legs route's 16-vCPU gfx942 pod, compute flavors allowed)."""
    if linux_merged(ctx):
        return ["--vcpu", str(CPU_BOX_MERGED_VCPU), "--flavors", CPU_BOX_MERGED_FLAVORS]
    if amd_small:
        # 16 vCPU: one set, 8 builds at a time; RunPod had no 32-vCPU CPU pod
        # when this route was first used (0.8.23), and smaller pods stock more.
        # Compute-optimized flavors too (2 GiB/vCPU = 32 GiB): one gfx942 set
        # peaked at 11.6 GiB (0.8.23's leg), so they fit, and they widen stock.
        return ["--vcpu", "16", "--flavors", CPU_BOX_FLAVORS + ",cpu5c,cpu3c"]
    return ["--flavors", CPU_BOX_FLAVORS]


def default_build_backend(commit):
    """`github` for either binding layout (Andrew 2026-10-26: compiling never
    uses paid rentals; GitHub's free runners build programmatically). The
    merged layout builds there one compile per shard job with swap sized from
    its recorded peak (tools/release_github_build.py UNIT_PEAK_GIB); `cpu-box`
    stays selectable by name. COMMIT is kept for callers and the record."""
    return "github"


#: The Linux build legs, one per release set (release_reuse.LINUX_SETS): native sm_89, the
#: PTX slot sm_80 and gfx942. sm_90a out for 0.8.37 (gpu_plugins.py). Andrew 2026-10-10:
#: PTX is a normal target; no flag. The PTX leg builds through the same route as any arch
#: (release061_remote_build.sh gives cuda/sm_80 the ptx code format); it needs no GPU.
RELEASE_LEG_SETS = tuple(release_reuse.LINUX_SETS)


def cpu_legs(ctx):
    """THE RENTED CPU ROUTE, opt-in by name (--build-backend cpu-box; until
    2026-10-26 the merged layout's default, before the GitHub route built one
    merged compile per job). The Linux sets compile on
    RunPod CPU pods, one pod per set, all three at once, with no GPU present
    (tools/release_linux_build.sh --archs <one>: Mojo compiles each set ahead
    of time from --target-accelerator alone). CPU pods do not wait on GPU
    stock. Proven at d181d9792 (0.8.14): cuda/sm_90a and cuda/sm_89 132 of 132
    binaries byte-identical to the GPU-box builds; gfx942 codegen varies run to
    run on ANY box, so the binding cache freezes it either way. Identity is
    still read back on the silicon: the NVIDIA and AMD wheel columns run the
    built wheel on real GPUs before anything publishes. The route overlays its
    own files from this checkout (its route.txt), so it takes no route overlay.
    The pod follows the layout (cpu_box_pod)."""
    legs_dir = ctx.rel / "legs"
    legs = []
    pod = cpu_box_pod(ctx)
    for vendor, arch in RELEASE_LEG_SETS:
        name = f"{vendor}-{arch}"
        rb, out = leg_layout(legs_dir, "cpu-box", name)
        legs.append(Leg(name, vendor, arch,
                        ["bash", "tools/release_linux_build.sh", ctx.commit, "--rent", "--archs", arch,
                         *pod, "--out", str(out)],
                        {}, rb, legs_dir, out))
    return legs


def github_legs(ctx):
    """THE DEFAULT ROUTE FOR EITHER LAYOUT (--build-backend github, 2026-09-27;
    the merged layout since 2026-10-26, default_build_backend): the Linux sets
    compile on GitHub's free ubuntu-24.04 runners, no rental. Each leg is
    tools/release_github_build.py run: it mints the binding-cache URL map with
    this Mac's R2 credentials, dispatches .github/workflows/release-linux-build.yml
    for the frozen commit (each set split across shard jobs, then one assemble
    job running the CPU box's unchanged route in its pinned image), polls the
    run, downloads the set into the cpu-box layout and verifies it against its
    proof; a failure names the job. The route overlays its own files from the
    pushed tooling commit (its GHA/assemble/overlay.txt), so it takes no route
    overlay, like cpu-box."""
    legs_dir = ctx.rel / "legs"
    legs = []
    for vendor, arch in RELEASE_LEG_SETS:
        name = f"{vendor}-{arch}"
        rb, out = leg_layout(legs_dir, "github", name)
        legs.append(Leg(name, vendor, arch,
                        ["python3", "tools/release_github_build.py", "run", "--commit", ctx.commit,
                         "--arch", arch, "--out", str(out)],
                        {}, rb, legs_dir, out))
    return legs


#: The default is "github" (default_build_backend; 2026-09-27: GitHub's free runners
#: build, proven byte-identical to 0.8.24 on all three sets; 2026-10-26: the merged
#: layout too, one compile per shard job with swap); "cpu-box" and "gpu-legs" are
#: opt-in by name.
BUILD_BACKENDS = {"cpu-box": cpu_legs, "github": github_legs, "gpu-legs": gpu_legs}
#: Routes that overlay their own box-side files and take no route overlay.
SELF_OVERLAID = ("cpu-box", "github")


def linux_legs(ctx):
    """The legs the reuse plan says to launch: only the sets with a binding
    to BUILD (plus the cheapest leg when only a host binding or the runtime
    needs one). Every leg of the backend when there is no plan."""
    legs = BUILD_BACKENDS[ctx.args.build_backend](ctx)
    plan = ctx.plan()
    if plan is None:
        return legs
    return [l for l in legs if l.name in plan["legs"]]


def launch_detached(ctx, legs):
    """THE ONE LAUNCH POINT for the Linux builds and the GPU columns. Every leg
    is launched at once, with no stagger (2026-09-25: a release is
    parallel only). Each runs detached (its own session, so this script can
    exit or be interrupted without killing a paid rental) and writes its exit
    code last. A leg that is running is left alone, a leg whose output checks
    out is not rerun, and a leg that finished but failed is moved aside (never
    deleted) and relaunched. The leg's provenance (both commits, the route
    overlay, the identities it was asked for) is written beside its log."""
    started = 0
    for leg in legs:
        if leg.running():
            ctx.say(f"  {leg.name}: already running (pid {leg.pid()}), log {leg.log}")
            continue
        if leg.done(ctx.commit):
            ctx.say(f"  {leg.name}: already done")
            continue
        compile_requirements = None
        if type(leg) is Leg and callable(getattr(ctx, "begin_compile_requirements", None)):
            compile_requirements = ctx.begin_compile_requirements(leg)
        leg.workdir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        for p in (leg.log, leg.exit_file, leg.pid_file, leg.provenance_file):
            if p.exists():
                p.rename(p.with_name(p.name + f".failed-{stamp}"))
        out = leg.out_dir
        if out.exists():
            out.rename(out.with_name(out.name + f".failed-{stamp}"))
            ctx.say(f"  {leg.name}: previous attempt moved aside to {out.name}.failed-{stamp}")
        prov = leg.provenance
        if prov is None and type(leg) is Leg and callable(getattr(ctx, "leg_provenance", None)):
            prov = ctx.leg_provenance(leg)
        if compile_requirements is not None:
            prov = dict(prov or {}, compile_requirements=compile_requirements)
        if prov is not None:
            write_json(leg.provenance_file, dict(prov, command=[str(c) for c in leg.command], launched_at=now()))
        wrapped = (f"{' '.join(shlex.quote(c) for c in leg.command)} > {shlex.quote(str(leg.log))} 2>&1; "
                   f"echo $? > {shlex.quote(str(leg.exit_file))}")
        proc = ctx.spawn(["sh", "-c", wrapped], dict(os.environ, **leg.env))
        leg.pid_file.write_text(str(proc))
        ctx.say(f"  {leg.name}: launched pid {proc}, log {leg.log}")
        started += 1
    return started


def host_digest_mismatches(legs):
    """{binding: {leg: sha256}} for every host binding whose bytes differ across
    legs, or that a leg lacks. pack_wheel.py refuses the same thing; this names
    the leg before the packer does."""
    table = {}
    for leg in legs:
        host = leg.release_build / "build" / "sets" / leg.vendor / leg.arch / "host"
        for so in sorted(host.glob("*.so")):
            table.setdefault(so.name, {})[leg.name] = sha256(so)
    return {name: row for name, row in table.items()
            if len(row) != len(legs) or len(set(row.values())) != 1}


#: Where linux-wait writes a retained host composition (under the release dir).
HOST_COMPOSITION_DIR = "host-composition"


def host_owner_leg():
    """The leg that owns the merged CPU host bindings (release_reuse.HOST_LEG)."""
    return "%s-%s" % release_reuse.HOST_LEG


def compose_host_mismatch(legs, mismatch, source_root, out_dir):
    """THE HOST OWNER RULE FOR A HOST BYTE MISMATCH (0.8.63): the Mojo compiler's
    host output can differ by build machine CPU with identical recorded inputs
    (MOJOLEARN_LINUX_CPU=x86-64-v3 on every leg; the Modular ask is in
    docs/MODULAR_ASKS.md). Instead of stopping, every set ships the owner leg's
    retained host bytes through tools/release_host_composition.py, a witnessed
    packaging choice that leaves every original proof untouched.

    Strict: the owner leg must be one of `legs`; every mismatching file must be a
    planned merged host file present on every leg; every leg's proof, capture,
    cache witness, source closure, compiler, flags, index and runtime libraries
    must agree (release_host_composition.validate_document); device artifacts must
    still match each leg's own proof. Returns the record linux-wait keeps in the
    release state; raises StepFailed otherwise."""
    import release_host_composition as hc
    owner = host_owner_leg()
    by = {l.name: l for l in legs}
    if owner not in by:
        raise StepFailed(f"host bindings differ and the host owner leg {owner} is not among the built legs "
                         f"({', '.join(sorted(by)) or 'none'}); nothing to compose from")
    planned = set(hc.host_files())
    stray = sorted(set(mismatch) - planned)
    if stray:
        raise StepFailed("differing files under host/ are not planned merged host bindings: " + ", ".join(stray))
    absent = {n: sorted(set(by) - set(row)) for n, row in mismatch.items() if set(row) != set(by)}
    if absent:
        raise StepFailed("host bindings missing on a leg (a composition only replaces bytes, never fills gaps): "
                         + "; ".join(f"{n}: {', '.join(m)}" for n, m in sorted(absent.items())))
    owner_leg = by[owner]
    selected = f"{owner_leg.vendor}/{owner_leg.arch}"
    roots = [l.release_build / "build" for l in legs]
    compiler = (read_json(owner_leg.release_build / "build" / "compile-inputs.json") or {}).get("compiler_sha256")
    out_dir = Path(out_dir)
    receipt = out_dir / "host-composition.json"
    try:
        if out_dir.exists():
            # A rerun: the retained composition must still be this one, over these legs.
            doc = read_json(receipt)
            if not doc:
                raise ValueError(f"{out_dir} exists without a readable host-composition.json; inspect and remove it")
            hc.validate_document(doc)
            if (doc.get("selected_leg") != selected or doc.get("expected_compiler_sha256") != compiler
                    or sorted(s["proof"]["sha256"] for s in doc["sources"])
                    != sorted(sha256(r / "build-provenance.json") for r in roots)):
                raise ValueError(f"{receipt} composes other legs, compiler or owner; inspect and remove {out_dir}")
            sets = {s["leg"]: out_dir / "sets" / s["leg"] for s in doc["sources"]}
            hc.validate_assembly(doc, sets, [r / "build-provenance.json" for r in roots], source_root)
        else:
            doc = hc.prepare([str(r) for r in roots], selected, compiler, out_dir, source_root)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise StepFailed(f"host bindings differ and the retained host composition from {owner} is refused: {exc}")
    why = ("bytes differ with identical recorded source closure, compiler, flags and cache inputs on every leg "
           "(build-machine-dependent host codegen); the host owner leg's retained bytes ship in every set")
    return dict(receipt=str(receipt), receipt_sha256=sha256(receipt), sets=str(out_dir / "sets"),
                selected_leg=selected, owner=owner, host_contract_sha256=doc["host_contract_sha256"],
                taken={n: dict(from_leg=owner, sha256=row[owner], legs=dict(sorted(row.items())))
                       for n, row in sorted(mismatch.items())},
                why=why, at=now())


# ---------------------------------------------------------------- the pipelines
#: (step, pipeline, needs, resource). A step runs once every step it NEEDS is
#: done; a failed or blocked need blocks it. Steps sharing a RESOURCE run one
#: at a time ("mac": the Mac's cores and its one Metal GPU; the order of
#: STEPS decides who goes first). Everything else runs at once.
#: THE LINUX PIPELINE IS ONE PER PACKAGE (mojolearn, mojolearn-nvidia,
#: mojolearn-amd; the only Linux layout since 0.8.22, the combined wheel is
#: gone). Every publish shares the "dispatch" resource, so two workflow
#: dispatches never race for `gh run list`'s newest run.
STEP_TABLE = [
    ("freeze-version", "common", [], None),
    ("freeze-changelog", "common", ["freeze-version"], None),
    ("freeze-docs-facts", "common", ["freeze-changelog"], None),
    ("freeze-commit", "common", ["freeze-docs-facts"], None),
    ("rehearsal", "common", ["freeze-commit"], None),
    ("reuse-plan", "common", ["rehearsal"], None),
    ("cross-compile", "core-linux", ["freeze-commit"], None),
    ("linux-builds", "core-linux", ["reuse-plan", "cross-compile"], None),
    ("macos-build", "macos", ["reuse-plan"], "mac"),
    ("macos-smoke", "macos", ["macos-build"], "mac"),
    # THE APPLE ADAPTER GATE (2026-10-24, AGENTS.md "Framework adapters are the same API in another format"):
    # the Torch and MAX adapters of the built macOS wheel, installed into a fresh venv, on MPS/Metal
    ("gate-apple", "macos", ["macos-smoke"], "mac"),
    # Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun.
    # macos-self-test (the same `verify --self-test` macos-smoke runs on the same
    # wheel) and release-check (an Apple re-fit of the lanes the admitted table
    # already holds) are deleted: the table admit is the identity check.
    ("linux-wait", "core-linux", ["linux-builds"], None),
    ("linux-assemble", "core-linux", ["linux-wait"], None),
    ("linux-pack", "core-linux", ["linux-assemble"], "mac"),
    ("gpu-column-nvidia", "nvidia", ["linux-pack"], None),
    ("gpu-column-amd", "amd", ["linux-pack"], None),
    # linux-joint-diff is deleted (Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun):
    # it re-diffed a second identity pass of the changed lanes against the Apple
    # re-fit; NVIDIA == AMD is decided once, by the admitted table.
    # THE CORE FIRST, THE LINUX `mojolearn` LAST (2026-10-20): each vendor
    # requires mojolearn-core at its version, and the file-free mojolearn
    # requires the core and both vendors, so each is resolvable only once
    # what it requires is on the index
    ("publish-core-linux", "core-linux", ["linux-pack"], "dispatch"),
    ("publish-nvidia", "nvidia", ["gpu-column-nvidia", "publish-core-linux"], "dispatch"),
    ("publish-amd", "amd", ["gpu-column-amd", "publish-core-linux"], "dispatch"),
    ("publish-meta-linux", "meta-linux", ["publish-core-linux", "publish-nvidia", "publish-amd"], "dispatch"),
    ("publish-macos", "macos", ["macos-smoke", "gate-apple"], "dispatch"),
    ("finish-line", "finish", ["freeze-commit"], None),
    ("record", "finish", ["finish-line"], None),
]
#: Steps that wait for others to SETTLE (done, failed or blocked), not succeed.
AFTER = {"finish-line": ["publish-core-linux", "publish-nvidia", "publish-amd",
                         "publish-meta-linux", "publish-macos"],
         # with --gpu-smoke the core publishes on a column receipt: wait for both columns to settle
         "publish-core-linux": ["gpu-column-nvidia", "gpu-column-amd"]}
#: A PIPELINE: named builds, named checks, one publish, and the platform it
#: ships (what published_platforms() says: `linux` is the core's).
PIPELINES = {
    "macos": dict(builds=["macos-build"], checks=["macos-smoke", "gate-apple"],
                  publish="publish-macos",
                  platform="macos"),
    "core-linux": dict(builds=["linux-builds", "linux-wait", "linux-assemble", "linux-pack"],
                       checks=["cross-compile"], publish="publish-core-linux", platform="linux"),
    "nvidia": dict(builds=[], checks=["gpu-column-nvidia"], publish="publish-nvidia", platform="nvidia"),
    "amd": dict(builds=[], checks=["gpu-column-amd"], publish="publish-amd", platform="amd"),
    "meta-linux": dict(builds=[], checks=[], publish="publish-meta-linux", platform="linux-meta"),
}
#: split package -> (PyPI project, wheel-name prefix); python/mojolearn/gpu_plugins.py.
#: "linux" is mojolearn-core, "linux-meta" the file-free Linux mojolearn.
SPLIT_PACKAGES = {"linux": (GPU_PACKAGES["CORE_DISTRIBUTION"], GPU_PACKAGES["CORE_WHEEL_NAME"]), **{
    row["profile"]: (row["distribution"], row["wheel_name"])
    for row in GPU_PACKAGES["distribution_rows"]()},
    "linux-meta": (GPU_PACKAGES["META_DISTRIBUTION"], GPU_PACKAGES["META_WHEEL_NAME"])}
#: the split packages that carry GPU sets (every one but the core and the meta wheel)
PLUGIN_PACKAGES = tuple(k for k in SPLIT_PACKAGES if k not in ("linux", "linux-meta"))
#: split packages earlier releases published that this one does not: read only from a previous
#: release's record (its PTX bytes, release_reuse), never packed or published (2026-11-05).
LEGACY_SPLIT_PACKAGES = ("nvidia-ptx",)
#: what `pip install <project>` must resolve on Linux: package -> the split packages it installs
LINUX_RESOLVES = {"linux": ("linux",),
                  "linux-meta": tuple(SPLIT_PACKAGES),
                  **{k: ("linux", k) for k in PLUGIN_PACKAGES
                     if GPU_PACKAGES["PAYLOADS"][k]["role"] == "vendor"}}
# Vendor wheels retain separate native execution columns for every hardware slot.
NATIVE_COLUMNS = {"nvidia": ("nvidia", "nvidia-hopper"), "amd": ("amd",)}
if set(NATIVE_COLUMNS) != {r["profile"] for r in GPU_PACKAGES["distribution_rows"]() if r["role"] == "vendor"}:
    raise RuntimeError("every released vendor requires hardware qualification columns")
STEP_TABLE.insert(STEP_TABLE.index(next(r for r in STEP_TABLE if r[0] == "gpu-column-amd")),
                  ("gpu-column-nvidia-hopper", "nvidia", ["linux-pack"], None))
# RELEASES RENT NOTHING (2026-10-08, Andrew: "let's deprecate the per-architecture
# rented smoke"). The Ada and AMD columns run on the nv and amd boxes we hold
# (--smoke-via lq); the Hopper column runs only on a held Hopper box
# (--hopper-box), so publish-nvidia no longer NEEDS it: it waits for it to
# settle (AFTER) and requires its PASS only when a Hopper box was held
# (Release.hopper_required). Without one the sm_90a payload publishes on its
# GitHub build receipt and the alpha manifest says "smoke: not run".
PIPELINES["nvidia"]["checks"].append("gpu-column-nvidia-hopper")
AFTER["publish-nvidia"] = ["gpu-column-nvidia-hopper"]
#: where --smoke-via lq runs each column: the lq box and the column's tag stem
#: the held lq box per GPU column; MOJOLEARN_RELEASE_NV_BOX / _AMD_BOX pick another held box of the same
#: hardware (e.g. nv2, the second L40S, when nv is released)
LQ_COLUMN_BOX = {"nvidia": os.environ.get("MOJOLEARN_RELEASE_NV_BOX", "nv"),
                 "amd": os.environ.get("MOJOLEARN_RELEASE_AMD_BOX", "amd")}
#: the sm_90a set's line when no Hopper box is held
HOPPER_NOT_RUN = "not run (no Hopper box held)"
PUBLISH_STEPS = tuple(pipeline["publish"] for pipeline in PIPELINES.values())
AFTER["finish-line"] = list(PUBLISH_STEPS)
# Owner clarification 2026-10-25: publication uses release-light-v1 against
# the exact installed candidate wheels on held boxes, with no performance race.
# The explicit performance-campaign-v1 profile retains historical campaigns;
# performance evidence remains required for claims and default promotions.
# Every publication still waits for both actual candidate gate verdicts.
GATE_VENDORS = {"nvidia": "publish-nvidia", "amd": "publish-amd"}
STEP_TABLE[STEP_TABLE.index(next(r for r in STEP_TABLE if r[0] == "finish-line")):0] = [
    ("gate-" + v, "finish", ["linux-pack"], None) for v in GATE_VENDORS]
# There is no candidate-version PyPI prerequisite: both gates install the
# local core/vendor files emitted by linux-pack, including NVIDIA's PTX wheel.
STEP_TABLE = [(name, pipeline, needs + (["gate-"+v for v in GATE_VENDORS] if name.startswith('publish-') else []), resource)
              for name, pipeline, needs, resource in STEP_TABLE]
AFTER["record"] = ["gate-" + v for v in GATE_VENDORS]
# THE GATE BOXES AS STEPS (lane release-no-hands, 2026-11-04; ORCHESTRATOR ONLY, MOJOLEARN_ORCHESTRATOR=1):
# gate-boxes brings up what the box steps need (the shared NVIDIA pod, tools/nvidia_central.sh up 1, and the
# DigitalOcean AMD droplet, tools/do_amd_steward.sh up), records both in the state, holds the pod busy and
# renews the droplet's lease while a box step is pending (Release.BoxKeeper); boxes-down pulls every box job's
# results home (tools/box_persist.py fetch --tag) and takes down what gate-boxes rented once every box step
# has settled. Never an Apple machine.
#: box step -> the vendor box it runs on
BOX_CONSUMERS = {"gate-" + v: v for v in GATE_VENDORS}
STEP_TABLE.insert(STEP_TABLE.index(next(r for r in STEP_TABLE if r[0] == "gate-nvidia")),
                  ("gate-boxes", "finish", ["linux-pack"], "boxes"))
STEP_TABLE = [(name, pipeline, needs + (["gate-boxes"] if name in BOX_CONSUMERS else []), resource)
              for name, pipeline, needs, resource in STEP_TABLE]
# THE PUBLISHED PRODUCERS (lane published-no-reverify, 2026-11-04; Andrew 2026-11-02: the gates verified the
# exact wheel bytes, nothing reinstalls them from PyPI to verify again). After each platform's publish,
# published-<vendor> runs tools/release_published_install.py check HERE, on no device and no box: pip resolves
# each published package for the candidate's platform tags, the downloaded file's sha256 must equal the gated
# candidate's and its RECORD must hold, and the gate's own receipts are its evidence by reference. The report
# lands at <commit12>/published-<vendor>/published-report.json (retained_producers admits it into the ledger).
# No published-<vendor> step needs gate-boxes, so boxes-down follows the gates.
PUBLISHED_STEPS = {"published-apple": ("apple", "publish-macos"), "published-nvidia": ("nvidia", "publish-nvidia"),
                   "published-amd": ("amd", "publish-amd")}
# REMOVED (lane identity-once-on-frozen-source, Andrew 2026-11-05: identity runs ONCE per release, BEFORE
# publish, and NEVER after publish): the published-smoke-<vendor> steps. Behind --published-smoke they
# reinstalled the published wheels and ran tools/qualify_verifier_wheel.py --scope expanded (identity lanes on
# the vendor device) AFTER publish-<platform>. `--published-smoke` now refuses; published-<vendor> (the bytes
# check on the Mac, no device) is the post-publication read. PUBLISHED_SMOKE_STEPS stays as an empty map for
# the modules and tests that iterate it.
PUBLISHED_SMOKE_STEPS = {}
STEP_TABLE.insert(STEP_TABLE.index(next(r for r in STEP_TABLE if r[0] == "record")),
                  ("boxes-down", "finish", [], "boxes"))
STEP_TABLE[STEP_TABLE.index(next(r for r in STEP_TABLE if r[0] == "boxes-down")):0] = [
    (step, "finish", [publish], None) for step, (_, publish) in PUBLISHED_STEPS.items()]
AFTER["boxes-down"] = ["gate-boxes", *BOX_CONSUMERS]
AFTER["record"] = AFTER["record"] + list(PUBLISHED_STEPS)

# IDENTITY ONCE, ON THE FROZEN SOURCE (lane identity-once-on-frozen-source, Andrew 2026-11-05). identity-at-freeze
# (tools/identity_at_freeze.py) compares the frozen source with the commit each identity column was last recorded
# at, right after freeze-commit and BEFORE anything is built or rented: rehearsal and cross-compile (so every build,
# box and publish) need it. Records made at a source-equal commit are adopted; an owed lane HOLDS the release there.
STEP_TABLE.insert(STEP_TABLE.index(next(r for r in STEP_TABLE if r[0] == "freeze-commit")) + 1,
                  ("identity-at-freeze", "common", ["freeze-commit"], None))
STEP_TABLE = [(name, pipeline, [("identity-at-freeze" if n == "freeze-commit" else n) for n in needs]
               if name in ("rehearsal", "cross-compile") else needs, resource)
              for name, pipeline, needs, resource in STEP_TABLE]

# --identity-on-wheels (default off): identity-cpu (the cpu column and the CPU checks on GitHub) and identity-admit
# (every column admitted, the table repacked). Without the option both return "skipped" at once and order nothing:
# the ordering they need under the option is added per run (Release.__init__), so the default graph is unchanged.
STEP_TABLE.insert(STEP_TABLE.index(next(r for r in STEP_TABLE if r[0] == "finish-line")),
                  ("identity-cpu", "finish", ["linux-pack"], None))
STEP_TABLE.insert(STEP_TABLE.index(next(r for r in STEP_TABLE if r[0] == "finish-line")),
                  ("identity-admit", "finish", ["linux-pack"], None))

#: NO IDENTITY LANE AFTER THE FIRST PUBLISH (Andrew 2026-11-05). The steps that execute identity lanes on a device:
#: the macOS wheel self-test / smoke (`verify --self-test` or qualify_verifier_wheel on Metal), the GPU wheel columns
#: (verify against the admitted table on the NVIDIA/AMD box) and the NVIDIA/AMD candidate gates (installed correctness
#: records). gate-apple is not here: it runs the framework adapters and the runtime qualifier, no identity lane.
#: Every publish waits for each of them to settle (AFTER), and each refuses to launch once any platform of this
#: release is published (Release.refuse_device_lanes_after_publish); tools/test_identity_once.py holds both.
DEVICE_LANE_STEPS = ("macos-smoke", "gpu-column-nvidia", "gpu-column-nvidia-hopper", "gpu-column-amd",
                     "gate-nvidia", "gate-amd")
for _publish in PUBLISH_STEPS:
    AFTER[_publish] = list(dict.fromkeys(AFTER.get(_publish, []) + list(DEVICE_LANE_STEPS)))

# THE TESTPYPI REHEARSAL (lane release-testpypi-rehearsal, Andrew 2026-11-06: everything that can be checked
# before the real publish is checked before it). Inside --publish pypi, testpypi-rehearsal runs after every check
# a publish needs or waits for (the gates, the one verification round) and BEFORE the first publish step; every
# publish step needs it. It uploads the exact gated wheels to TestPyPI, resolves every package set there with our
# project names pinned to TestPyPI (tools/release_testpypi_rehearsal.py), compares every file's sha256 with the
# gated wheel's and installs and imports the macOS wheel on this Mac. Any failure stops the release before the
# real PyPI sees anything. It needs what EVERY publish needs (both gates) and waits for the rest to settle, so a
# failed macOS pipeline still lets the Linux packages rehearse and publish (each publish then checks that the
# rehearsal covered its own bytes: Release.publish).
REHEARSAL_STEP = "testpypi-rehearsal"
STEP_TABLE.insert(STEP_TABLE.index(next(r for r in STEP_TABLE if r[0] == PUBLISH_STEPS[0])),
                  (REHEARSAL_STEP, "finish", [], "dispatch"))


def wire_rehearsal(needs, after):
    """(needs, after) with testpypi-rehearsal needing what every publish needs, waiting for everything else a
    publish needs or waits for, and every publish needing it. Idempotent: rerun after the per-run graph changes
    (--identity-on-wheels, a held Hopper box)."""
    publishes = [s for s in PUBLISH_STEPS if s in needs]
    def own(p):
        return [d for d in needs[p] if d not in PUBLISH_STEPS and d != REHEARSAL_STEP]
    common = [d for d in own(publishes[0]) if all(d in own(p) for p in publishes)]
    settle = list(dict.fromkeys(d for p in publishes for d in own(p) + list(after.get(p, ()))
                                if d not in PUBLISH_STEPS and d != REHEARSAL_STEP and d not in common))
    needs = dict(needs, **{REHEARSAL_STEP: common},
                 **{p: needs[p] + ([] if REHEARSAL_STEP in needs[p] else [REHEARSAL_STEP]) for p in publishes})
    return needs, dict(after, **{REHEARSAL_STEP: settle})


_needs, AFTER = wire_rehearsal({s: n for s, _, n, _ in STEP_TABLE}, AFTER)
STEP_TABLE = [(name, pipeline, _needs[name], resource) for name, pipeline, _, resource in STEP_TABLE]

PIPELINE_OF = {s: p for s, p, _, _ in STEP_TABLE}
NEEDS = {s: n for s, _, n, _ in STEP_TABLE}
RESOURCE = {s: r for s, _, _, r in STEP_TABLE}


def receipt_plugins(results):
    """{wheel file name: sha256} of the split plugins a smoke receipt installed."""
    d = read_json(results) or {}
    return {Path(p.get("wheel", "")).name: p.get("wheel_sha256") for p in d.get("plugins") or [] if isinstance(p, dict)}


def merge_split(wheels, out):
    """The combined wheel a published split set is a partition of, for the
    reuse assembly (release_reuse.assemble_linux reads one wheel): every
    payload member of every wheel, and the core's .dist-info with its
    LINUX_PAYLOAD.json's `split` role dropped. A plugin that was not
    published leaves its members out; assembly then refuses to take them."""
    # the core: mojolearn_core-* since 2026-10-20, mojolearn-* (with files) before
    core = next((w for w in wheels if Path(w).name.startswith(GPU_PACKAGES["CORE_WHEEL_NAME"] + "-")), None) \
        or next(w for w in wheels if Path(w).name.startswith("mojolearn-"))
    members = {}
    for w in wheels:
        with zipfile.ZipFile(w) as z:
            for n in z.namelist():
                top = n.split("/")[0]
                if n.endswith("/") or (top.endswith(".dist-info") and w != core):
                    continue
                data = z.read(n)
                if top.endswith(".dist-info") and n.endswith("/LINUX_PAYLOAD.json"):
                    doc = json.loads(data)
                    doc.pop("split", None)
                    for mapping in ("extensions", "binding_origin"):
                        if mapping in doc:
                            doc[mapping] = {release_reuse.legacy_archive_path(k): v
                                            for k, v in doc[mapping].items()}
                    data = (json.dumps(doc, sort_keys=True, indent=2) + "\n").encode()
                n = release_reuse.legacy_archive_path(n)
                if n in members and members[n] != data:
                    raise StepFailed(f"conflicting split-wheel member: {n}")
                members[n] = data
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w") as z:
        for n, data in members.items():
            z.writestr(n, data)
    return Path(out)


class StepFailed(Exception):
    pass


class StepHeld(StepFailed):
    """Stopped on purpose (publication without --publish), not broken."""


# THE ONE RELEASE RUN ON THE BUILT WHEELS (--identity-on-wheels, default off; tools/release_identity_wheels.py):
# its steps and hooks live in that module as a mixin of Release; it raises this module's StepFailed.
import release_identity_wheels  # noqa: E402
release_identity_wheels.StepFailed = StepFailed
release_identity_wheels.write_json = lambda path, doc: write_json(path, doc)


# ---------------------------------------------------------------- the run

def wheel_marker(wheel):
    """The parsed gpu_plugin.json of a vendor plugin wheel."""
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/gpu_plugin.json"))
        return json.loads(archive.read(name))

class BoxKeeper:
    """KEEPS THE RELEASE'S BOXES while a box step is pending (lane release-no-hands, 2026-11-04): a thread of
    the release process (an active controller, never a background service; it ends with the run).

    NVIDIA: the shared pod deletes itself 60 idle minutes after it was last busy, and a pod is busy while an
    `nvidia_central.sh sh` command runs, so one hold runs there (`sh <lane> <wait for the release file>`, at
    most Release.HOLD_MAX_S, started again while still needed), never a hand `sleep` on the Mac. It ends when
    nothing is pending: the release file is touched through the same wrapper and the hold exits.
    AMD: the droplet's lease is renewed (`do_amd_steward.sh extend AMD_RENEW_MIN`) every AMD_RENEW_EVERY_S."""

    def __init__(self, release):
        self.r = release
        self.hold = None
        self._stop = threading.Event()
        self.thread = None
        self.release_file = f"/root/release-hold/{release.version}-{release.commit[:12]}.release"

    def hold_command(self):
        f = self.release_file
        return (f"mkdir -p {shlex.quote(str(Path(f).parent))}; rm -f {shlex.quote(f)}; "
                f"end=$(( $(date +%s) + {self.r.HOLD_MAX_S} )); "
                f"while [ ! -e {shlex.quote(f)} ] && [ $(date +%s) -lt $end ]; do sleep 30; done; echo RELEASE-HOLD-END")

    def start_hold(self):
        log = self.r.rel / "boxes" / "nvidia-hold.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = ["bash", self.r.nvc(), "sh", self.r.BOX_LANE, self.hold_command()]
        self.r.say(f"  box keeper: holding the NVIDIA pod busy: {' '.join(shlex.quote(c) for c in cmd[:4])} <hold>")
        with open(log, "a") as fh:
            self.hold = self.r.box_popen(cmd, cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=fh,
                                         stderr=subprocess.STDOUT)

    def end_hold(self):
        if self.hold is None:
            return
        if self.hold.poll() is None:
            self.r.run(["bash", self.r.nvc(), "sh", self.r.BOX_LANE, f"mkdir -p /root/release-hold && touch "
                        f"{shlex.quote(self.release_file)}"], log=self.r.rel / "boxes" / "nvidia-hold-release.log")
            try:
                self.hold.wait(timeout=120)
            except subprocess.TimeoutExpired:
                self.hold.terminate()
        self.hold = None

    def renew_amd(self, rec):
        if self.r.clock() - float(rec.get("lease_extended_at") or 0) < self.r.AMD_RENEW_EVERY_S:
            return
        rc = self.r.run(["bash", self.r.do_amd(), "extend", str(self.r.AMD_RENEW_MIN)],
                        log=self.r.rel / "boxes" / "amd-extend.log")
        with self.r._lock:
            # a failed renewal is retried five minutes later, before the lease can run out
            rec["lease_extended_at"] = self.r.clock() - (0 if rc == 0 else self.r.AMD_RENEW_EVERY_S - 300)
            self.r.save()

    def tick(self):
        """One pass; the vendors still needed."""
        need = self.r.boxes_pending()
        held = self.r.held_boxes()
        if "nvidia" in need and "nvidia" in held:
            if self.hold is None or self.hold.poll() is not None:
                self.start_hold()
        else:
            self.end_hold()
        if "amd" in need and "amd" in held:
            self.renew_amd(held["amd"])
        return need & set(held)

    def loop(self):
        try:
            while not self._stop.is_set() and self.tick():
                self._stop.wait(self.r.KEEPER_TICK_S)
        finally:
            self.end_hold()

    def start(self):
        self.thread = threading.Thread(target=self.loop, name="release-box-keeper", daemon=True)
        self.thread.start()

    def alive(self):
        return self.thread is not None and self.thread.is_alive()

    def stop(self):
        self._stop.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=300)
        self.end_hold()


class Release(release_identity_wheels.IdentityOnWheels):
    STEPS = [s for s, _, _, _ in STEP_TABLE]
    #: Every other step checks its own OUTPUT each time and returns at once when
    #: it is already there (a wheel of this commit, a PASSED receipt for that
    #: wheel); these are skipped on their record
    #: (finish-line and record: when the platforms they covered are still the
    #: published ones).
    SKIP_IF_RECORDED = {"identity-at-freeze", "rehearsal", "cross-compile", "finish-line", "record",
                        "identity-cpu", "identity-admit", *PUBLISH_STEPS}

    def __init__(self, args, runner=None):
        self.args = args
        self.version = args.version
        self.release_modes = release_reuse.parse_modes(getattr(args, "release_modes", None)
                                                       or release_reuse.release_modes())
        ev = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", os.path.expanduser("~/mojolearn-evidence")))
        self.base = Path(args.state_dir) if args.state_dir else ev / "release" / self.version
        self.state_path = self.base / "state.json"
        self.state = self.load()
        if self.state.get("steps") and release_reuse.record_modes(self.state) != self.release_modes:
            raise SystemExit("release: tier selection differs from retained release state; use a new "
                             "--state-dir to preserve the previous artifacts and qualification")
        self.state["release_modes"] = list(self.release_modes)
        # A release packed as the one combined wheel (0.8.20 and earlier) is
        # not resumable: the combined layout is gone.
        redo = set((getattr(args, "redo", "") or "").split(","))
        if self.state.get("linux_layout") == "combined" and "linux-pack" not in redo:
            raise SystemExit("release: this release packed the combined Linux wheel, which the release no "
                             "longer builds; --redo linux-pack to repack it as the split packages")
        (self.STEPS, self.PIPELINE_OF, self.NEEDS, self.RESOURCE, self.AFTER, self.PIPELINES) = (
            [t[0] for t in STEP_TABLE], PIPELINE_OF, NEEDS, RESOURCE, AFTER, PIPELINES)
        if self.single():
            # --identity-on-wheels: the one run settles before any publish. identity-admit waits for every column
            # (the box jobs, the Apple column in gate-apple, the GitHub CPU run); every publish waits for it, and
            # gate-apple and identity-cpu run identity lanes, so they join the no-lane-after-publish guard.
            self.AFTER = {k: list(v) for k, v in AFTER.items()}
            self.AFTER["identity-admit"] = ["gate-nvidia", "gate-amd", "gate-apple", "identity-cpu"]
            # THE CPU COLUMN FIRST (Andrew 2026-11-07): the CPU column is required to agree with every GPU column
            # (admit_identity_columns.sh), and it runs on GitHub for free, so nothing is rented until it has
            # recorded: a lane that errors or drifts on the exact wheel bytes stops the release before the bill.
            self.NEEDS = dict(self.NEEDS, **{"gate-boxes": list(dict.fromkeys(self.NEEDS["gate-boxes"] + ["identity-cpu"]))})
            for publish in PUBLISH_STEPS:
                self.AFTER[publish] = list(dict.fromkeys(self.AFTER.get(publish, []) + [
                    "gate-apple", "identity-cpu", "identity-admit"]))
            self.reference_env()
        if self.hopper_required():
            # a held Hopper box (or the rented route): its column gates publish-nvidia as before
            self.NEEDS = dict(NEEDS, **{"publish-nvidia": NEEDS["publish-nvidia"] + ["gpu-column-nvidia-hopper"]})
        # the rehearsal follows the per-run graph: it waits for whatever a publish needs or waits for
        needs, after = wire_rehearsal(self.NEEDS, self.AFTER)
        self.NEEDS = self.NEEDS if needs == self.NEEDS else needs   # unchanged graph: the module's own tables
        self.AFTER = self.AFTER if after == self.AFTER else after
        self.runner = runner
        self._linux_merged = None
        if getattr(args, "build_backend", None) is None:
            args.build_backend = default_build_backend(self.commit)
        self.dry = args.dry_run
        self.readonly = bool(getattr(args, "status", False) or getattr(args, "report", False))
        self.commands_shown = 0
        self.evidence = ev
        self._plan = None
        self._overlay = None
        self._tooling = None
        self._leg_env = None
        self._ledger = None
        self._lock = threading.RLock()
        self._out = threading.Lock()
        self._tl = threading.local()
        self.errors, self.blocked_by = {}, {}

    # state
    def load(self):
        try:
            return json.loads(self.state_path.read_text())
        except (OSError, ValueError):
            return {"version": self.version, "commit": None, "steps": {}}

    def save(self):
        if self.dry or self.readonly:
            return
        with self._lock:
            self.base.mkdir(parents=True, exist_ok=True)
            tmp = self.state_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.state, indent=2, sort_keys=True) + "\n")
            tmp.replace(self.state_path)

    def head(self):
        return git("rev-parse", "HEAD")

    def head_of(self, checkout):
        return self.head() if Path(checkout) == ROOT else git_at(checkout, "rev-parse", "HEAD")

    @property
    def commit(self):
        """THE SOURCE COMMIT: pinned at freeze, HEAD only before the first freeze."""
        return self.state.get("commit") or self.head()

    @property
    def linux_merged(self):
        """Whether the source's Linux sets build the merged layout (cpu-box route env)."""
        if self._linux_merged is None:
            self._linux_merged = release_github_build.commit_merged(self.commit, source_layout_env())
        return self._linux_merged

    @property
    def tooling_commit(self):
        return self.head()

    @property
    def rel(self):
        return self.base / self.commit[:12]

    def mark(self, step, **data):
        with self._lock:
            self.state["steps"][step] = dict(done=True, commit=self.state.get("commit"), at=now(),
                                             tooling_commit=git_at(ROOT, "rev-parse", "HEAD"), **data)
            self.state.setdefault("failures", {}).pop(step, None)
            self.save()

    def fail(self, step, exc, held=False):
        with self._lock:
            self.errors[step] = str(exc)
            self.state.setdefault("failures", {})[step] = dict(at=now(), commit=self.state.get("commit"),
                                                              error=str(exc), held=held)
            self.save()

    def recorded(self, step):
        rec = self.state["steps"].get(step)
        return rec if rec and rec.get("done") and rec.get("commit") == self.state.get("commit") else None

    def published_platforms(self):
        return [self.PIPELINES[p]["platform"] for p in self.PIPELINES if self.recorded(self.PIPELINES[p]["publish"])]

    def skip_recorded(self, step, rec):
        if step not in self.SKIP_IF_RECORDED:
            return False
        if step in ("finish-line", "record") and "platforms" in rec:
            return sorted(rec["platforms"]) == sorted(self.published_platforms())
        return True

    # execution
    def say(self, msg):
        tag = getattr(self._tl, "tag", "")
        with self._out:
            if tag:
                msg = "\n".join(f"[{tag}] {line}" for line in (str(msg).splitlines() or [""]))
            try:
                print(msg, flush=True)
            except (OSError, ValueError):
                # A disconnected controller terminal (EIO, a closed pipe): the step logs and the state carry
                # everything, so the run goes on instead of crashing the step that printed (2026-11-05).
                pass

    def sleep(self, seconds):
        if not self.dry:
            time.sleep(seconds)

    def run(self, cmd, env=None, log=None, cwd=None):
        """Run one command to completion; its output teed to `log` when given."""
        env = {"MOJOLEARN_RELEASE_MODES": " ".join(self.release_modes), **(env or {})}
        shown = " ".join(shlex.quote(str(c)) for c in cmd)
        extra = " ".join(f"{k}={shlex.quote(str(v))}" for k, v in (env or {}).items())
        where = f"(in {cwd}) " if cwd and Path(cwd) != ROOT else ""
        self.say(f"  $ {where}{extra + ' ' if extra else ''}{shown}")
        self.commands_shown += 1
        if self.dry:
            return 0
        if self.runner:
            return self.runner(cmd, env, log)
        full_env = dict(os.environ, **(env or {}))
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "w") as fh:
                proc = subprocess.Popen([str(c) for c in cmd], cwd=cwd or ROOT, env=full_env,
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                for line in proc.stdout:
                    self.say("    " + line.rstrip("\n"))
                    fh.write(line)
                return proc.wait()
        return subprocess.run([str(c) for c in cmd], cwd=cwd or ROOT, env=full_env).returncode

    def must(self, cmd, env=None, log=None, what=None, cwd=None):
        rc = self.run(cmd, env, log, cwd=cwd)
        if rc != 0:
            raise StepFailed(f"{what or cmd[0]} exited {rc}" + (f"; log {log}" if log else ""))

    def spawn(self, cmd, env):
        env = dict(env, MOJOLEARN_RELEASE_MODES=" ".join(self.release_modes))
        if self.runner:
            return self.runner(cmd, env, None, detach=True)
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True)
        return proc.pid

    # ------------------------------------------------------------ source and tooling
    def source_checkout(self, create=True):
        """Where the steps that read the SOURCE run: this checkout when its HEAD
        is the source commit, else --source-checkout, else a detached worktree
        of the source commit under <version>/source/<commit12>/ (made once).

        Made atomically and single-flight: the pipelines run concurrently and
        two of them once asked for it in the same second, one reading the half
        checked-out tree (0.8.66). The worktree is added at a temporary sibling
        `<commit12>.tmp-<pid>`, verified, then moved to its final path, under a
        thread lock (this process) and an fcntl lock beside the path (other
        processes). A final path made earlier is verified (HEAD at the commit,
        no tracked change) before it is used; a wrong one is refused."""
        c = self.commit
        if getattr(self.args, "source_checkout", ""):
            p = Path(self.args.source_checkout).expanduser().resolve()
            if not self.dry and not self.readonly and git_at(p, "rev-parse", "HEAD") != c:
                raise StepFailed(f"--source-checkout {p} is not at the source commit {c}")
            return p
        if self.head_of(ROOT) == c:
            return ROOT
        p = self.base / "source" / c[:12]
        if self.dry or self.readonly or not create:
            if (p / ".git").exists() and not self.readonly and git_at(p, "rev-parse", "HEAD") != c:
                raise StepFailed(f"the source checkout {p} is not at {c}; remove it (git worktree remove) and rerun")
            return p
        lock = self.__dict__.setdefault("_source_lock", threading.Lock())
        with lock:
            if self.__dict__.get("_source_ready") == (p, c):
                return p
            p.parent.mkdir(parents=True, exist_ok=True)
            import fcntl
            with open(p.parent / f".{p.name}.lock", "a+") as fh:
                fcntl.flock(fh, fcntl.LOCK_EX)
                try:
                    self._sweep_source_tmp(p)
                    if p.exists():
                        self._verify_source(p, c)
                    else:
                        self._add_source(p, c)
                finally:
                    fcntl.flock(fh, fcntl.LOCK_UN)
            self._source_ready = (p, c)
        return p

    @staticmethod
    def _source_problem(p, c):
        """Why the tree at P is not a complete checkout of C, or None."""
        if not (Path(p) / ".git").exists():
            return "it is not a git worktree"
        head = git_at(p, "rev-parse", "HEAD")
        if head != c:
            return f"its HEAD is {head or '(unreadable)'}"
        dirty = git_at(p, "status", "--porcelain", "--untracked-files=no")
        if dirty is None:
            return "git status failed in it"
        if dirty:
            return "it has tracked changes (a partial or edited checkout): " + dirty.splitlines()[0]
        return None

    def _verify_source(self, p, c):
        why = self._source_problem(p, c)
        if why:
            raise StepFailed(f"the source checkout {p} is not a clean checkout of {c}: {why}; "
                             f"remove it (git worktree remove --force {p}) and rerun")

    def _add_source(self, p, c):
        tmp = p.parent / f"{p.name}.tmp-{os.getpid()}"
        self.say(f"  source checkout: a detached worktree of {c[:12]} at {p} (the tooling runs from {ROOT})")
        rc = subprocess.run(["git", "-C", str(ROOT), "worktree", "add", "--detach", str(tmp), c],
                            capture_output=True, text=True)
        if rc.returncode != 0:
            self._drop_worktree(tmp)
            raise StepFailed(f"could not add a worktree of {c} at {tmp}: {rc.stderr.strip()}")
        why = self._source_problem(tmp, c)
        if why:
            self._drop_worktree(tmp)
            raise StepFailed(f"the new worktree of {c} at {tmp} is incomplete: {why}")
        rc = subprocess.run(["git", "-C", str(ROOT), "worktree", "move", str(tmp), str(p)],
                            capture_output=True, text=True)
        if rc.returncode != 0:
            self._drop_worktree(tmp)
            raise StepFailed(f"could not move the worktree {tmp} to {p}: {rc.stderr.strip()}")

    @staticmethod
    def _drop_worktree(tree):
        """Remove a worktree of ROOT and its registration (targeted, so other
        sessions' missing worktrees are not pruned)."""
        tree = Path(tree)
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(tree)],
                       capture_output=True, text=True)
        shutil.rmtree(tree, ignore_errors=True)
        common = git_at(ROOT, "rev-parse", "--git-common-dir")
        admin = (ROOT / common if common else ROOT / ".git") / "worktrees"
        for gitdir in admin.glob("*/gitdir") if admin.is_dir() else []:
            try:
                if Path(gitdir.read_text().strip()).parent in (tree, tree.resolve()):
                    shutil.rmtree(gitdir.parent, ignore_errors=True)
            except OSError:
                continue

    def _sweep_source_tmp(self, p):
        """Remove leftover `<commit12>.tmp-<pid>` trees beside P. Called under
        the file lock, so no live process is making one."""
        for tmp in sorted(p.parent.glob(f"{p.name}.tmp-*")):
            self.say(f"  source checkout: removing the leftover {tmp}")
            self._drop_worktree(tmp)

    @property
    def src(self):
        return self.source_checkout()

    def overlay(self):
        """The route overlay manifest: the tooling copy of each box-side tool
        that differs from the source commit's (tools/release_tooling.py)."""
        with self._lock:
            if self._overlay is None:
                try:
                    self._overlay = release_tooling.overlay_manifest(ROOT, self.tooling_commit, self.commit)
                except release_tooling.OverlayRefused as exc:
                    raise StepFailed(str(exc))
            return self._overlay

    def builders_override(self):
        """The binding-identity builders the box runs from the tooling checkout
        (the route overlay; tools/release_linux_build.sh overlays the same
        builder from this checkout on the cpu-box route)."""
        return release_tooling.effective_builders(self.overlay())

    def tooling(self):
        """The build tooling that runs: its commit, its digest (None when a
        tooling file is uncommitted, so nothing built with it is reused across
        freezes), the files and the uncommitted ones."""
        with self._lock:
            if self._tooling is None:
                commit = self.tooling_commit
                digest, rows = release_tooling.tooling_digest(ROOT, commit)
                dirty = release_tooling.dirty(ROOT, release_tooling.BUILD_TOOLING)
                self._tooling = dict(commit=commit, digest=None if dirty else digest, files=rows, dirty=dirty)
            return self._tooling

    def leg_env(self):
        """The environment every build leg gets: the source checkout its
        inventory is compared with, and the route overlay when there is one
        (the tarball is written once per overlay digest and never rewritten)."""
        with self._lock:
            if self._leg_env is None:
                env = {"MOJOLEARN_SOURCE_CHECKOUT": str(self.source_checkout(create=not self.dry))}
                m = self.overlay()
                if m["files"] and self.args.build_backend not in SELF_OVERLAID:
                    if self.dry:
                        env.update(MOJOLEARN_ROUTE_OVERLAY="<route-overlay.tgz of " + ", ".join(sorted(m["files"])) + ">",
                                   MOJOLEARN_ROUTE_OVERLAY_SHA256="<sha256>")
                    else:
                        dest = self.rel / "legs" / "route-overlay" / m["digest"][:16]
                        doc = read_json(dest / "route-overlay.json")
                        tgz = dest / "route-overlay.tgz"
                        if doc and doc.get("digest") == m["digest"] and tgz.is_file() \
                                and sha256(tgz) == doc.get("tarball_sha256"):
                            digest = doc["tarball_sha256"]
                        else:
                            try:
                                tgz, digest = release_tooling.write_overlay(ROOT, m, dest)
                            except release_tooling.OverlayRefused as exc:
                                raise StepFailed(str(exc))
                        env.update(MOJOLEARN_ROUTE_OVERLAY=str(tgz), MOJOLEARN_ROUTE_OVERLAY_SHA256=digest)
                self._leg_env = env
            return dict(self._leg_env)

    def set_identity(self, vendor, arch):
        try:
            plan = self.plan()
            return release_reuse.set_identity(plan, vendor, arch) if plan else None
        except (Exception, SystemExit):
            return None

    def leg_provenance(self, leg):
        """What a build leg is, recorded at launch: both commits, the tooling
        digest, the route overlay, and the set identity it was asked for."""
        try:
            t = self.tooling()
        except (Exception, SystemExit):
            t = dict(commit=None, digest=None, dirty=[])
        try:
            m = self.overlay() if self.args.build_backend not in SELF_OVERLAID else None
        except (Exception, SystemExit):
            m = None
        return dict(schema="mojolearn.release-leg-provenance.v1", leg=leg.name, vendor=leg.vendor, arch=leg.arch,
                    source_commit=self.commit, tooling_commit=t["commit"], tooling_digest=t["digest"],
                    tooling_dirty=t["dirty"], overlay=m if m and m.get("files") else None,
                    route_overlay_env={k: v for k, v in leg.env.items() if k.startswith("MOJOLEARN_ROUTE_OVERLAY")},
                    set_identity=self.set_identity(leg.vendor, leg.arch), backend=self.args.build_backend,
                    release_build=str(leg.release_build), out_dir=str(leg.out_dir), log=str(leg.log),
                    exit_file=str(leg.exit_file))

    def ledger(self):
        """The PASS ledger: R2 for a real run (not a dry run, not a test's
        injected runner, not a --state-dir run), always the local mirror."""
        with self._lock:
            if self._ledger is None:
                if self.args.state_dir:
                    local = Path(self.args.state_dir) / "ledger"
                    self._ledger = release_ledger.Ledger(local, None, say=self.say)
                elif self.runner is None and not self.dry:
                    self._ledger = release_ledger.Ledger.open(self.evidence, say=self.say)
                else:
                    self._ledger = release_ledger.Ledger(self.evidence / "release" / "ledger", None, say=self.say)
                if self.readonly:
                    self._ledger.cache = False
            return self._ledger

    def ledger_put(self, key, entry):
        if self.dry or self.readonly:
            return None
        try:
            return self.ledger().put(key, dict(entry, recorded_at=now()))
        except Exception as exc:  # the ledger is an index; its failure costs a rerun, never a release
            self.say(f"  ledger: could not record {key} ({type(exc).__name__}: {exc})")
            return None

    def waived(self, verdict):
        """Apply the owner's explicit waiver (--owner-waiver): blocked rows whose requirement
        id starts with a waived prefix are reported WAIVED and logged in the release state;
        everything else still blocks. Andrew 2026-10-25: 0.8.57 ships on identity plus the
        light installed gate; runtime-per-vendor, multi-GPU, coverage and adapter rows are owed."""
        ready, reasons = verdict
        path = getattr(self.args, "owner_waiver", None)
        if ready or not path:
            return verdict
        waiver = json.loads(Path(path).read_text())
        prefixes = tuple(waiver["prefixes"])
        kept = [r for r in reasons if not r.split(":", 1)[0].startswith(prefixes)]
        waived = [r.split(":", 1)[0] for r in reasons if r.split(":", 1)[0].startswith(prefixes)]
        for name in waived:
            self.say("   WAIVED " + name + " (" + waiver["approved_by"] + ": " + waiver["reason"] + ")")
        if not self.dry:
            self.state.setdefault("owner_waivers", []).append(dict(
                file=str(path), sha256=sha256(Path(path)), approved_by=waiver["approved_by"],
                reason=waiver["reason"], waived=sorted(waived), at=now()))
        return not kept, kept

    def readiness(self):
        """One persistent readiness ledger across freezes of this version."""
        return release_requirements.RequirementsLedger(self.base / "requirements.json")

    DRY_READINESS = "not evaluated in a dry run: the readiness analysis runs dependency workers"

    def readiness_manifest(self):
        # A dry run is read-only and prints the plan: the frozen dependency analysis (release_requirements ->
        # release_requirement_dependencies.expand_catalog) clones the tracked tree and runs snapshot workers,
        # which took the two dry-run tests in tools/test_release.py past their 600 s timeout. Never run it here.
        if self.dry:
            raise StepFailed(self.DRY_READINESS)
        configuration = dict(self.readiness().read().get("configuration", {}))
        configuration_path = getattr(self.args, "requirements_configuration", None)
        if configuration_path:
            supplied = json.loads(Path(configuration_path).read_text())
            if not isinstance(supplied, dict):
                raise ValueError("requirements configuration must be a JSON object")
            configuration = release_requirements.merge_configuration(configuration, supplied)
        configuration.update(release_modes=list(self.release_modes),
                             build_flags=release_requirements.build_flags())
        # runtime-<vendor> (tools/release_installed_evidence.py) runs the frozen source's own tests against
        # the installed wheels: the test revision is the frozen source unless a configuration names another.
        # 0.8.64: no configuration ever supplied it, so every runtime row was "missing input" for good.
        configuration.setdefault("runtime_test_source_revision", self.commit)
        context = dict(configuration.get("dependency_context", {}))
        apple_toolchain = release_reuse.darwin_toolchain()
        if apple_toolchain is not None:
            context["apple_toolchain"] = apple_toolchain
        configuration["dependency_context"] = context
        for vendor in GATE_VENDORS:
            configuration.pop("artifacts:" + vendor, None)
            try:
                identity = self.candidate_gate_identity(vendor)
                # Source revision is provenance, not an invalidation input.
                configuration["artifacts:" + vendor] = identity["wheels"]
            except (StepFailed, OSError, ValueError, KeyError, zipfile.BadZipFile):
                pass
        try:
            configuration.pop("artifacts:apple", None)
            wheel = self.macos_wheel()
            if wheel and Path(wheel).is_file():
                configuration["artifacts:apple"] = {Path(wheel).name: sha256(wheel)}
        except (StepFailed, OSError, ValueError):
            pass
        # One publication controller checks the same frozen inputs for every
        # split wheel. Reuse that source analysis; any candidate wheel hash,
        # configuration, freeze, tooling commit or catalog change invalidates it.
        key = (self.commit, self.tooling_commit, sha256(ROOT / 'tools/release_requirements.json'),
               json.dumps(configuration, sort_keys=True))
        cached = getattr(self, '_readiness_manifest_cache', None)
        if cached is None or cached[0] != key:
            manifest = release_requirements.requirements(ROOT, self.commit, self.version,
                                                        configuration=configuration)
            presence = self.optional_native_presence(manifest, context)
            if presence:
                # Re-expand with the derived layout contract; the frozen
                # analyzer accepts it only for source-proven optional loaders.
                configuration["dependency_context"] = dict(context, optional_native_files=presence)
                manifest = release_requirements.requirements(ROOT, self.commit, self.version,
                                                            configuration=configuration)
            self._readiness_manifest_cache = (key, manifest)
        return json.loads(json.dumps(self._readiness_manifest_cache[1]))

    UNMAPPED_BINDING = "no native binding context mapped for binding: "

    def optional_native_presence(self, manifest, context):
        """{target: {binding: present}} for unmapped bindings, read from the candidate wheels.

        An optional native file (one the frozen loader connects only when it is
        installed, e.g. the fast tokenizer an identical-only release does not
        ship) leaves every identity unit unresolved until its presence in the
        shipped layout is stated. That is a fact of the candidate artifacts, so
        it is read from their file lists: the macOS wheel for osx-arm64, the
        packed Linux wheels for linux-64. A target whose artifacts are not all
        present gets no contract, and an explicit configuration always wins.
        """
        names = sorted({item[len(self.UNMAPPED_BINDING):] for wanted in manifest["requirements"].values()
                        for item in wanted.get("missing_inputs", []) if item.startswith(self.UNMAPPED_BINDING)})
        if not names:
            return None
        listings = {}
        try:
            wheel = self.macos_wheel()
            if wheel and Path(wheel).is_file():
                with zipfile.ZipFile(wheel) as z:
                    listings["osx-arm64"] = {Path(n).name for n in z.namelist()}
        except (OSError, zipfile.BadZipFile, StepFailed):
            pass
        try:
            finals = [self.split_final(p) for p in SPLIT_PACKAGES if p != "linux-meta"]
            if finals and all(f and Path(f).is_file() for f in finals):
                members = set()
                for final in finals:
                    with zipfile.ZipFile(final) as z:
                        members |= {Path(n).name for n in z.namelist()}
                listings["linux-64"] = members
        except (OSError, zipfile.BadZipFile, StepFailed):
            pass
        declared = context.get("optional_native_files", {})
        presence = {}
        for target, members in listings.items():
            row = dict(declared.get(target, {}))
            for name in names:
                row.setdefault(name, name + ".so" in members)
            presence[target] = row
        return presence if presence != {t: declared.get(t) for t in presence} else None

    def evidence_root(self):
        """Retained-producer admission reports for this freeze (release_requirements_retained)."""
        return self.rel / "ledger-evidence"

    def macos_bincache(self):
        return self.rel / "macos-bincache"

    def macos_bincache_seeds(self):
        """Every earlier release's macOS cache, newest first (read only; archives are copied)."""
        mine = self.macos_bincache().resolve()
        found = [d for d in self.base.parent.glob("*/*/macos-bincache")
                 if d.is_dir() and d.resolve() != mine and any(d.glob("*.tar.gz"))]
        return sorted(found, key=lambda d: d.stat().st_mtime, reverse=True)

    def macos_cache_plan(self, env):
        """One line per merged macOS group, before anything compiles: `hit (<cache>)`
        or `compile: <reason>` (tools/bincache.py explain in the frozen tree, under
        the build's environment). Empty when that tree's bincache has no explain."""
        src = self.src
        tool = src / "tools" / "bincache.py"
        if not tool.is_file() or "def cmd_explain" not in tool.read_text(errors="replace"):
            return []
        base = dict(os.environ, **env)
        out = []
        modes = [m for m in self.release_modes] + (["host"] if env.get("MOJOLEARN_PACKAGE_BYTE_LM") == "1" else [])
        for mode in modes:
            plan = subprocess.run([PY, str(src / "tools" / "merged_bindings.py"), "plan", "--dir", mode,
                                   "--platform", "darwin", "--byte-lm", env.get("MOJOLEARN_PACKAGE_BYTE_LM", "1")],
                                  cwd=src, capture_output=True, text=True)
            for row in plan.stdout.splitlines():
                stem = row.split()[0]
                if mode == "host":
                    output, kind = f"python/mojolearn/host/{stem}.so", "host"
                    genv = dict(base, MOJOLEARN_NUMERIC_MODE="identical", MOJOLEARN_TARGET_COLUMN="cpu")
                    genv.pop("MOJOLEARN_GPU_ARCHS", None)
                    genv.pop("MOJOLEARN_HOST_OUTDIR", None)
                else:
                    output = f"python/mojolearn/{stem}.so" if mode == "fast" else f"python/mojolearn/{mode}/{stem}.so"
                    kind, genv = "gpu", dict(base, MOJOLEARN_NUMERIC_MODE=mode)
                genv.update(MOJOLEARN_BINCACHE_OUTPUTS=output, MOJOLEARN_BINCACHE_SHELL="sh")
                r = subprocess.run([PY, str(tool), "explain", "./bindings/merged_build.sh", kind, "--group", stem],
                                   cwd=src, env=genv, capture_output=True, text=True)
                line = (r.stdout.strip().splitlines() or [r.stderr.strip()[-200:]])[-1]
                out.append(f"macos {mode}/{stem}: {line}")
        return out

    def retained_producers(self):
        """(adapter, producer paths) of this freeze's retained outputs that never wrote the
        ledger: the macOS release build (its binary cache and log; the cache of a build
        made before step_macos_build kept one is named by MOJOLEARN_BINCACHE_DIR) and the
        Apple installed-adapter smoke of the macOS wheel (gate-apple)."""
        out = []
        wheel = self.macos_wheel()
        log = self.rel / "macos-build.log"
        caches = [self.macos_bincache()] + ([Path(os.environ["MOJOLEARN_BINCACHE_DIR"])]
                                            if os.environ.get("MOJOLEARN_BINCACHE_DIR") else [])
        cache = next((c for c in caches if (c / "provenance" / "provenance.tsv").is_file()), None)
        if wheel and log.is_file() and cache is not None:
            out.append(("macos-release-build", dict(wheel=str(wheel), log=str(log), cache=str(cache))))
        if wheel and (self.apple_adapters_out() / "wheel.json").is_file():
            out.append(("apple-gate-adapters", dict(adapters=str(self.apple_adapters_out()), wheel=str(wheel))))
        # The NVIDIA and AMD candidate gates (adapters-<vendor>): every gate directory of this freeze
        # that holds a report; the adapter admits only a PASS for the ledger's qualified wheels.
        # 0.8.64: these PASSED on the boxes and still blocked publication because nothing named them here.
        for vendor in GATE_VENDORS:
            for directory in sorted((self.rel / f"gate-{vendor}").glob("*/")):
                if (directory / "gate-smoke.json").is_file() and (directory / "adapter-smoke").is_dir():
                    out.append(("gate-adapters", dict(gate=str(directory))))
        # runtime-<vendor>: the installed-runtime qualifier the gate job runs after its smoke (a passed run only;
        # the adapter re-derives the counts from the raw junit and the qualifier receipt).
        for vendor in GATE_VENDORS:
            for report in sorted((self.rel / f"gate-{vendor}").glob(f"*/runtime-{vendor}/runtime-report.json")):
                out.append(("installed-runtime", dict(report=str(report))))
        # runtime-cpu: the same producer with --vendor cpu, run by the NVIDIA gate job after runtime-nvidia
        # (GATE_RUNTIME_CPU); runtime-apple: the Mac's gate-apple step (step_gate_apple).
        for report in sorted((self.rel / "gate-nvidia").glob("*/runtime-cpu/runtime-report.json")):
            out.append(("installed-runtime", dict(report=str(report))))
        if (self.apple_runtime_out() / "runtime-report.json").is_file():
            out.append(("installed-runtime", dict(report=str(self.apple_runtime_out() / "runtime-report.json"))))
        # published-<vendor> (post stage): tools/release_published_install.py reports under published-<vendor>/.
        # Only published-<vendor>/: an explicit --published-smoke report (published-smoke-<vendor>/) is not this
        # row's evidence.
        for vendor, _ in PUBLISHED_STEPS.values():
            report = self.published_out(vendor) / "published-report.json"
            if report.is_file():
                out.append(("published-install", dict(report=str(report))))
        # The Linux release legs (compile-<vendor>/runtime, compile-<vendor>/adapters/<arch>, compile-cpu/runtime
        # on the host leg): a leg built in this freeze, or the earlier freeze's leg a <leg>.reused.json names.
        for directory in self.retained_leg_dirs():
            out.append(("linux-release-leg", dict(leg=str(directory))))
        return out

    def retained_leg_dirs(self):
        """Leg directories whose build this freeze ships: built here (legs/<leg>/ with its .provenance.json
        and .exit beside it) or reused byte for byte from an earlier freeze (legs/<leg>.reused.json names
        that freeze's provenance file; the leg directory sits beside it)."""
        legs = self.rel / "legs"
        found = []
        for provenance in sorted(legs.glob("*.provenance.json")):
            name = provenance.name[:-len(".provenance.json")]
            if (legs / name).is_dir() and (legs / (name + ".exit")).is_file():
                found.append(legs / name)
        for reused in sorted(legs.glob("*.reused.json")):
            doc = read_json(reused) or {}
            name = reused.name[:-len(".reused.json")]
            source = Path(str(doc.get("provenance") or ""))
            leg = source.parent / name
            if (doc.get("leg") == name and source.is_file() and leg.is_dir()
                    and (source.parent / (name + ".exit")).is_file() and leg not in found):
                found.append(leg)
        return found

    def readiness_status(self, write=False):
        manifest = self.readiness_manifest()
        ledger = self.readiness()
        import release_requirements_retained as retained
        if write:
            ledger.sync(manifest)
            # Evidence that already exists (the committed identity table
            # admission) is admitted before any gate reads the ledger, so
            # verified work is never rerun or waived (Andrew, 2026-10-31).
            _, notes = retained.admit_existing_evidence(ROOT, ledger, self.commit, self.evidence_root(),
                                                        producers=self.retained_producers(), say=self.say)
            for note in notes:
                self.say("  ledger evidence: " + note)
            return ledger.sync(manifest)
        # Read-only (--status, --dry-run, the gate's plan): nothing persistent
        # is written. Reports already kept for this freeze are reused; the rest
        # are prepared in scratch and evaluated in memory with the ledger.
        doc = ledger.read()
        doc.update(json.loads(json.dumps(manifest)))
        with retained.scratch_root() as scratch:
            reports, notes = retained.retained_reports(ROOT, doc, self.commit, Path(scratch),
                                                       prepared=self.evidence_root(),
                                                       producers=self.retained_producers(), say=self.say)
            attempts = retained.retained_attempts(reports, notes)
            for note in notes:
                self.say("   ledger evidence: " + note)
            return ledger._evaluate(retained.with_attempts(doc, attempts))

    def record_candidate_requirement(self, vendor, run_id, executed=False):
        if run_id is None:
            return
        import release_requirement_receipts as receipts
        ledger = self.readiness()
        run = next(row for row in ledger.read()["runs"] if row["id"] == run_id)
        report = self.gate_out(vendor) / "gate-smoke.json"
        provenance_path = report.parent / "producer-inputs.json"
        if executed:
            # Only the controller that actually launched this begun run writes
            # this stamp. A legacy report cannot acquire current inputs by fiat.
            write_json(provenance_path, dict(schema=receipts.PROVENANCE_SCHEMA, requirement_id="candidate-" + vendor,
                       kind="candidate", source_revision=run["source_revision"], input_fingerprint=run["fingerprint"],
                       report_sha256=sha256(report), run_id=run_id))
        try:
            receipt = receipts.candidate_gate_receipt("candidate-" + vendor,
                         dict(run["manifest"], inputs=run["inputs"]), report,
                         self.candidate_gate_identity(vendor), read_json(provenance_path))
        except ValueError as error:
            raise StepFailed("candidate evidence needs provenance/coverage reconciliation; no automatic rerun: "
                             + str(error)) from error
        counts = receipts.coverage_summary(receipt)
        ledger.record(run_id, "passed", [report, provenance_path],
                      "candidate producer passed all required coverage IDs", 0, receipt=receipt, **counts)

    def begin_compile_requirements(self, leg):
        """Persist exact inputs before a fresh release leg starts, never on reuse."""
        import release_compile_evidence as compile_evidence
        source_tool = file_at(self.commit, "tools/release_compile_evidence.py") or ""
        if compile_evidence.CAPTURE_SCHEMA not in source_tool:
            raise StepFailed("frozen Linux build producer lacks before-build provenance capture; "
                             "reconcile retained evidence before launching another build")
        ledger = self.readiness()
        ledger.sync(self.readiness_manifest())
        requirements = ledger.read()["requirements"]
        selected = {name: wanted for name, wanted in requirements.items()
                    if compile_evidence.belongs_to_leg(wanted, leg.vendor, leg.arch)}
        if not selected:
            raise StepFailed("no supported declared compile requirements for " + leg.name + "; no build launched")
        if getattr(leg, "compile_provider_retry", False):
            # An explicitly recognized provisioning refusal did not reach the
            # compiler. Continue that same begun attempt on the next provider;
            # never use this path for an ordinary compile failure or rerun.
            if any((leg.release_build / "build" / name).exists()
                   for name in ("compile-inputs.json", "build-provenance.json")):
                self.record_compile_failure(leg)
                raise StepFailed("provider retry reached build preparation/execution; reconcile its original outcome")
            prior = (read_json(leg.provenance_file) or {}).get("compile_requirements", {})
            document = ledger.read()
            completed = {row["run_id"] for row in document["attempts"]}
            active = {row["id"]: row for row in document["runs"] if row["id"] not in completed}
            active_selected = {row["requirement"] for row in active.values() if row["requirement"] in selected}
            if (prior.get("schema") != compile_evidence.LAUNCH_SCHEMA or not prior.get("runs")
                    or prior.get("selected") != sorted(selected)
                    or set(prior["runs"]) != active_selected):
                raise StepFailed("provider retry lacks the original compile launch; reconcile before relaunch")
            for name, saved in prior.get("runs", {}).items():
                run = active.get(saved.get("run_id"))
                if (name not in selected or run is None or run["requirement"] != name
                        or run["fingerprint"] != selected[name]["fingerprint"]
                        or run["source_revision"] != selected[name]["source_revision"]
                        or saved.get("fingerprint") != run["fingerprint"]
                        or saved.get("source_revision") != run["source_revision"]):
                    raise StepFailed("provider retry differs from the active begun compile inputs")
            return prior
        # An open begun run whose producer stopped (a failed or cancelled
        # GitHub run, or a launch that never dispatched) is closed here with
        # its retained evidence, so only this leg relaunches; a running or
        # queued producer still refuses below, and a succeeded one is
        # collected and recorded, never interrupted (0.8.64: sm_89 run
        # 41364664603 left 124 open runs that had to be closed by hand).
        groups = []
        if callable(getattr(self, "close_stopped_compile_producers", None)):
            groups = self.close_stopped_compile_producers(leg, selected)
        succeeded = [g for g in groups if g["state"] == "succeeded"]
        if succeeded:
            attached = self.attach_succeeded_producer(leg, selected, succeeded[0], groups)
            if attached is not None:
                return attached
        runs = {}
        try:
            begun = ledger.begin_many(selected)
        except ValueError as error:
            held = "; ".join(f"{len(g['run_ids'])} run(s) {g['state']}: {g['reason']}"
                             for g in groups if not g.get("closed"))
            raise StepFailed("compile requirements refused a new build: " + str(error)
                             + (f" [producer: {held}]" if held else "")) from error
        for name, run_id in begun.items():
            wanted = selected[name]
            # Normal begin reuses an unchanged pass and refuses active work or
            # missing provenance. No free-text reason forces a repeat here.
            if run_id is not None:
                runs[name] = dict(run_id=run_id, source_revision=wanted["source_revision"],
                                  fingerprint=wanted["fingerprint"])
        if not runs:
            raise StepFailed("all compile inputs already have retained passes; restore/reconcile their artifacts "
                             "instead of launching another build")
        return dict(schema=compile_evidence.LAUNCH_SCHEMA, selected=sorted(selected), runs=runs)

    def leg_compile_names(self, leg):
        import release_compile_evidence as compile_evidence
        return {name for name, wanted in self.readiness().read()["requirements"].items()
                if compile_evidence.belongs_to_leg(wanted, leg.vendor, leg.arch)}

    def close_stopped_compile_producers(self, leg, names=None):
        """Classify a GitHub build leg's open begun compile runs by their
        producer (tools/release_stopped_producers.py) and interrupt, with the
        gh run JSON and failed-log excerpt (or the run listing) retained under
        compile-interruptions/, those whose producer concluded failure,
        cancelled, timed out or never dispatched. Returns the groups; the
        running, succeeded and unknown ones are left open for the caller."""
        import release_stopped_producers as producers
        if self.dry or getattr(self, "readonly", False) or type(leg) is not Leg or not producers.is_github_leg(leg):
            return []
        ledger = self.readiness()
        names = set(names) if names is not None else self.leg_compile_names(leg)
        groups = producers.Producers(lambda *a: self.gh(*a), release_github_build.WORKFLOW).classify(
            leg, ledger.read(), names, self.commit)
        producers.close_stopped(ledger, groups, self.rel / "compile-interruptions", leg.name)
        for group in groups:
            self.say(f"  {leg.name}: {len(group['run_ids'])} open compile run(s), producer {group['state']}: "
                     + group["reason"] + ("; interrupted (producer stopped, no qualification claim)"
                                          if group.get("closed") else ""))
        return groups

    def attach_succeeded_producer(self, leg, selected, group, groups):
        """A GitHub run that succeeded for this leg's open begun runs: record
        the leg from its own finished output, or relaunch its collector
        attached to that run (--run-id: download and verify, no compile), so
        the runs are recorded from the producer that made them, never
        interrupted. Returns the launch to reuse, or None after recording."""
        import release_compile_evidence as compile_evidence
        prov = read_json(leg.provenance_file) or {}
        prior = prov.get("compile_requirements") or {}
        document = self.readiness().read()
        runs = {row["id"]: row for row in document["runs"]}
        saved = prior.get("runs") or {}
        if (len(groups) != 1 or prior.get("schema") != compile_evidence.LAUNCH_SCHEMA
                or prior.get("selected") != sorted(selected)
                or group["run_ids"] != sorted(row.get("run_id") for row in saved.values())
                or any(name not in selected or runs[row["run_id"]]["requirement"] != name
                       or row.get("fingerprint") != selected[name]["fingerprint"]
                       or row.get("source_revision") != selected[name]["source_revision"]
                       for name, row in saved.items())):
            raise StepFailed(f"{leg.name}: {group['reason']} succeeded, but its open begun compile runs differ "
                             "from the leg's launch; reconcile before another build")
        source = prov.get("source_commit")
        if source and leg.done(source):
            self.record_compile_requirements(leg)
            return None
        if source != self.commit:
            raise StepFailed(f"{leg.name}: {group['reason']} built {str(source)[:12]}, not this freeze; "
                             "its output is not here to record")
        command = list(leg.command)
        if "--run-id" in command:
            del command[command.index("--run-id"):command.index("--run-id") + 2]
        leg.command = command + ["--run-id", str(group["github_run"])]
        self.say(f"  {leg.name}: collecting the succeeded GitHub run {group['github_run']} (no new compile)")
        return dict(prior, attached_github_run=group["github_run"])

    def record_compile_requirements(self, leg):
        """Parse the actual complete producer output for its original begun runs."""
        import release_compile_evidence as compile_evidence
        import release_requirement_receipts as receipts
        provenance = read_json(leg.provenance_file) or {}
        launch = provenance.get("compile_requirements", {})
        if launch.get("schema") != compile_evidence.LAUNCH_SCHEMA:
            self.say(f"  {leg.name}: compile readiness requires original launch provenance reconciliation; no rerun")
            return
        ledger = self.readiness()
        document = ledger.read()
        recorded = {attempt["run_id"] for attempt in document["attempts"]}
        runs = {run["id"]: run for run in document["runs"]}
        build = leg.release_build / "build"
        set_root = build / "sets" / leg.vendor / leg.arch
        admitted = []
        for name, begun in launch.get("runs", {}).items():
            run_id = begun["run_id"]
            if run_id in recorded:
                continue
            run = runs[run_id]
            output = self.rel / "compile-receipts" / run_id
            output.mkdir(parents=True, exist_ok=True)
            report_path = output / "compile-report.json"
            report = dict(schema=compile_evidence.REPORT_SCHEMA, producer=compile_evidence.TOOL,
                build_report=compile_evidence.reference(build / "build-provenance.json"),
                capture_report=compile_evidence.reference(build / "compile-inputs.json"),
                launch_report=compile_evidence.reference(leg.provenance_file),
                exit_report=compile_evidence.reference(leg.exit_file),
                set_manifest=compile_evidence.reference(set_root / "manifest.json"), set_root=str(set_root.resolve()))
            write_json(report_path, report)
            stamp = dict(schema=receipts.PROVENANCE_SCHEMA, requirement_id=name, kind="compile",
                         source_revision=run["source_revision"], input_fingerprint=run["fingerprint"],
                         report_sha256=sha256(report_path), run_id=run_id)
            write_json(output / "producer-inputs.json", stamp)
            try:
                receipt = compile_evidence.producer_payload(name,
                    dict(run["manifest"], inputs=run["inputs"]), report_path, stamp)
            except (ValueError, OSError, KeyError, TypeError) as error:
                raise StepFailed("retained release compilation needs provenance reconciliation; no rerun: "
                                 + str(error)) from error
            admitted.append(dict(run_id=run_id, status="passed", paths=[report_path, output / "producer-inputs.json"],
                summary="complete release build; retained source, compiler, flags and artifact bytes verified",
                exit_code=0, receipt=receipt, **receipts.coverage_summary(receipt)))
        ledger.record_many(admitted)

    def record_compile_failure(self, leg):
        """Retain an actual failed compiler producer; never infer from provisioning."""
        import release_compile_evidence as compile_evidence
        provenance = read_json(leg.provenance_file) or {}
        launch = provenance.get("compile_requirements", {})
        proof_path = leg.release_build / "build/build-provenance.json"
        capture_path = leg.release_build / "build/compile-inputs.json"
        proof, capture = read_json(proof_path) or {}, read_json(capture_path) or {}
        code, build_code = leg.exit_code(), proof.get("build_exit")
        if (launch.get("schema") != compile_evidence.LAUNCH_SCHEMA or not launch.get("runs")
                or proof.get("schema") != compile_evidence.BUILD_SCHEMA or proof.get("action") != "build"
                or type(code) is not int or not 0 < code < 124
                or type(build_code) is not int or not 0 < build_code < 124
                or capture.get("schema") != compile_evidence.CAPTURE_SCHEMA or capture.get("phase") != "before-build"):
            return  # No proof that a compiler/build-producer attempt completed with failure.
        if (proof.get("compile_inputs_sha256") != sha256(capture_path)
                or not proof.get("source_commit")
                or proof["source_commit"] != provenance.get("source_commit")
                or proof["source_commit"] != capture.get("source_commit")):
            raise StepFailed("failed compile evidence differs from its original launch; reconcile provenance")
        ledger = self.readiness()
        document = ledger.read()
        completed = {row["run_id"] for row in document["attempts"]}
        runs = {row["id"]: row for row in document["runs"]}
        identity = compile_evidence.digest([leg.name, sha256(leg.provenance_file), sha256(proof_path), code])
        retained = self.rel / "compile-failures" / identity
        retained.mkdir(parents=True, exist_ok=True)
        paths = []
        for path in (leg.log, leg.exit_file, leg.provenance_file, proof_path, capture_path):
            dest = retained / path.name
            if dest.exists() and sha256(dest) != sha256(path):
                raise StepFailed("retained compile failure bytes changed: " + str(dest))
            if not dest.exists():
                shutil.copy2(path, dest)
            paths.append(dest)
        failures = []
        for name, begun in launch["runs"].items():
            if begun["run_id"] in completed:
                continue
            run = runs.get(begun["run_id"])
            if (run is None or run["requirement"] != name or run["fingerprint"] != begun.get("fingerprint")
                    or run["source_revision"] != proof["source_commit"]):
                raise StepFailed("failed compile outcome does not identify its original begun inputs")
            failures.append(dict(run_id=run["id"], status="failed", paths=paths,
                summary=f"release build producer exited {build_code}; leg exited {code}; no compile coverage admitted",
                exit_code=code, expected=len(run["manifest"]["coverage"]), completed=0, failed=0, skipped=0))
        ledger.record_many(failures)

    # ------------------------------------------------------------ steps
    def frozen(self):
        return bool(self.state.get("commit")) and not getattr(self.args, "refreeze", False)

    def step_freeze_version(self):
        files = ["python/mojolearn/_version.py", "python/pyproject.toml"]
        if self.frozen():
            text = file_at(self.commit, files[0]) or ""
            m = re.search(r'^__version__ = "([^"]*)"', text, re.M)
            if not m or m.group(1) != self.version:
                raise StepFailed(f"the frozen source {self.commit[:12]} says {m.group(1) if m else '?'}, not "
                                 f"{self.version}; --refreeze to freeze HEAD")
            return f"frozen source {self.commit[:12]} says {self.version}"
        current = re.search(r'^__version__ = "([^"]*)"', (ROOT / files[0]).read_text(), re.M).group(1)
        if current == self.version and f'version = "{self.version}"' in (ROOT / files[1]).read_text():
            return "already " + self.version
        if self.dry:
            return f"would set {current} -> {self.version} in {', '.join(files)}"
        return "set in " + ", ".join(set_version(ROOT, self.version))

    def step_freeze_changelog(self):
        if self.frozen():
            date = changelog_date(ROOT, self.version, text=file_at(self.commit, "CHANGELOG.md") or "")
            if not date:
                raise StepFailed(f"the frozen source {self.commit[:12]} has no `## {self.version} (published "
                                 "YYYY-MM-DD)` heading; --refreeze after writing it")
            return "published " + date + " (frozen source)"
        date = changelog_date(ROOT, self.version)
        if not date:
            raise StepFailed(f"CHANGELOG.md has no `## {self.version} (published YYYY-MM-DD)` heading. "
                             "Write the entry (publication date in UTC) and rerun.")
        return "published " + date

    def step_freeze_docs_facts(self):
        if self.frozen():
            return f"part of the frozen source {self.commit[:12]}"
        self.must(["pixi", "run", "write-docs-facts"], what="write-docs-facts")
        return "written"

    def step_freeze_commit(self):
        head = git("rev-parse", "HEAD")
        frozen = self.state.get("commit")
        refreeze = getattr(self.args, "refreeze", False)
        if frozen and not refreeze:
            # THE SOURCE NEVER MOVES BECAUSE MAIN MOVED (2026-09-25): a tooling
            # fix lands on main and this checkout runs it; the legs, columns and
            # receipts of the frozen source stay.
            return (f"source pinned at {frozen}" + (f"; tooling runs from HEAD {head[:12]}" if head != frozen else "")
                    + ("" if head == frozen else " (--refreeze moves the source to HEAD)"))
        if frozen and head != frozen and self.published_platforms():
            raise StepFailed(f"--refreeze refused: a wheel of {frozen[:12]} is already published; "
                             "a new source state needs a new version")
        allowed = set(release_files()) | set(docs_fact_files())
        # NOT through git(): it strips the output, and the first porcelain line
        # then loses its leading status blank, so ` M CITATION.cff` read as
        # `ITATION.cff` and the release's own citation sync was refused.
        porcelain = subprocess.run(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
                                   capture_output=True, text=True, check=True).stdout
        dirty = [line[3:] for line in porcelain.splitlines() if line.strip()]
        others = [p for p in dirty if p not in allowed]
        if others:
            raise StepFailed("tracked files changed outside the release bump; commit or revert them first: "
                             + " ".join(others[:10]))
        if dirty:
            self.must(["git", "-C", ROOT, "add", "--", *dirty], what="git add")
            self.must(["git", "-C", ROOT, "commit", "-q", "-m", f"Release {self.version}", "-m",
                       "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"],
                      what="git commit")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        if branch == "HEAD":
            raise StepFailed("detached HEAD; release from a branch (main) so the commit can be pushed")
        # A rerun after main moved on (a fix landed while the legs built) must
        # not push the frozen commit over the newer main, and needs no push at
        # all when origin already contains it (0.8.16's second run stopped
        # here on a rejected non-fast-forward push).
        self.must(["git", "-C", ROOT, "fetch", "-q", "origin", branch], what="git fetch")
        contained = subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", "HEAD", f"origin/{branch}"]).returncode == 0
        if not contained:
            self.must(["git", "-C", ROOT, "push", "origin", f"HEAD:refs/heads/{branch}"], what="git push")
        if self.dry:
            return f"would freeze {'the bump commit' if dirty else head}"
        head = git("rev-parse", "HEAD")
        if self.state.get("commit") != head:
            # A new source commit (--refreeze). Nothing is discarded: every
            # later step checks its output, and a completed leg or column of the
            # earlier freeze is taken when what it built or tested is unchanged.
            earlier = [c for c in self.state.get("earlier_commits", []) + [self.state.get("commit")] if c]
            self.state = {"version": self.version, "commit": head, "earlier_commits": earlier,
                          "release_modes": list(self.release_modes),
                          "frozen_at": now(), "steps": {k: v for k, v in self.state["steps"].items()
                                                        if k.startswith("freeze-")}}
            self._plan = self._overlay = self._leg_env = self._tooling = None
            self.__dict__.pop("_admitted", None)
            self.save()
        return "frozen at " + head

    def step_identity_at_freeze(self):
        """IDENTITY ONCE, ON THE FROZEN SOURCE (Andrew 2026-11-05): tools/identity_at_freeze.py in the source
        checkout at the freeze, before anything is built or rented. Every column whose newest admitted record
        is source-equal to the freeze is adopted (no selector, nothing owed); a held column (nvidia, amd, cpu,
        apple) that owes lanes since its record HOLDS the release here, loudly, so the operator records once
        at the frozen source and refreezes instead of building first and recording twice. It runs, rents and
        waives nothing; the ledger still judges every identity row (tools/release_identity_reference.py)."""
        out = self.rel / "identity-at-freeze" / "report.json"
        cmd = [PY, str(ROOT / "tools" / "identity_at_freeze.py"), "check",
               "--root", str(self.source_checkout(create=not self.dry)), "--freeze", self.commit, "--json", str(out)]
        rc = self.run(cmd, log=self.rel / "identity-at-freeze.log")
        if self.dry:
            return None
        doc = read_json(out)
        if rc not in (0, 4) or not doc or doc.get("schema") != "mojolearn.identity-at-freeze.v1":
            raise StepFailed(f"identity-at-freeze exited {rc} without a report; log {self.rel / 'identity-at-freeze.log'}")
        sys.path.insert(0, str(ROOT / "tools"))
        import identity_at_freeze
        for line in identity_at_freeze.lines(doc):
            self.say("  " + line)
        if (rc == 4 or not doc.get("adopted")) and self.single():
            # --identity-on-wheels records identity from the built wheels of THIS freeze: owed lanes are
            # the one run's work, reported here, never a hold
            return ("identity owed at the freeze on %s: recorded once from the built wheels (--identity-on-wheels)"
                    % ", ".join(doc.get("held") or [])), dict(identity_at_freeze=str(out))
        if rc == 4 or not doc.get("adopted"):
            raise StepHeld("identity is owed at the freeze %s on %s: record ONCE at this source "
                           "(tools/reference_refresh.sh %s), merge ref/%s, then --refreeze; report %s" % (
                               self.commit[:12], ", ".join(doc.get("held") or ["?"]), self.commit[:12],
                               self.commit[:12], out))
        cols = doc.get("columns") or {}
        return "identity adopted at the freeze: " + ", ".join(
            "%s %s" % (c, "recorded at " + str(r.get("recorded_at", ""))[:12] if r.get("state") == "adopted"
                       else r.get("state")) for c, r in sorted(cols.items())), dict(identity_at_freeze=str(out))

    def refuse_device_lanes_after_publish(self, step):
        """NO IDENTITY LANE AFTER THE FIRST PUBLISH (Andrew 2026-11-05). Called by every DEVICE_LANE_STEPS step
        right before it would launch lanes on a device: once any platform of this release is published, it
        refuses (a read of an existing receipt is never refused; only a launch is)."""
        for publish in PUBLISH_STEPS:
            # defensive: partial Releases (tests' fakes) carry no state, and then nothing is published
            rec = ((getattr(self, "state", None) or {}).get("steps") or {}).get(publish)
            if rec and rec.get("done") and not str(rec.get("result", "")).startswith("none"):
                raise StepFailed(f"{step} would run identity lanes on a device after {publish} published "
                                 f"({rec.get('at')}: {str(rec.get('result', ''))[:120]}); identity runs once, before "
                                 "any publish (Andrew 2026-11-05). This platform ships in the next version")

    #: the rehearsal without --gpu-smoke: no python-tests (merged main ran them) and no GPU-box leg
    #: dry-runs (no GPU box builds); named with --only so an older source checkout's rehearsal parses it
    REHEARSAL_NO_GPU = ("wheel-audit-tests", "docs-facts", "ext-lists", "ext-lists-host", "platform-math")

    def step_rehearsal(self):
        work = self.rel / "rehearsal"
        only = [] if self.gpu_smoke() else ["--only", ",".join(self.REHEARSAL_NO_GPU)]
        self.must(["pixi", "run", "release-rehearsal", "--keep", work, *only], log=self.rel / "rehearsal.log",
                  what="release-rehearsal", cwd=self.src)
        return "PASS, logs " + str(work)

    # ------------------------------------------------------------ cross-compile (GitHub)
    XCC_WORKFLOW = "cross-compile-check.yml"
    # sm_90a out for 0.8.37 (gpu_plugins.py); sm_80 is the PTX slot, compiled like any arch
    # (Andrew 2026-10-10: PTX is a normal target; no flag).
    XCC_ARCHS = ",".join(arch for _, arch in RELEASE_LEG_SETS)

    def gh(self, *args, log=None):
        """(exit code, stdout) of one `gh` call; stdout also to `log` when given."""
        p = subprocess.run(["gh", *[str(a) for a in args]], cwd=ROOT, capture_output=True, text=True)
        if log:
            Path(log).parent.mkdir(parents=True, exist_ok=True)
            with open(log, "a") as fh:
                fh.write(p.stdout + p.stderr)
        return p.returncode, p.stdout

    def xcc_find_run(self, token, tries=24):
        """The run whose title carries `token` (the workflow's run-name), as
        (id, url); a dispatch takes a few seconds to appear."""
        for _ in range(tries):
            rc, out = self.gh("run", "list", "--workflow", self.XCC_WORKFLOW, "--event", "workflow_dispatch",
                              "--limit", "30", "--json", "databaseId,displayTitle,url")
            for r in (json.loads(out or "[]") if rc == 0 else []):
                if token in r.get("displayTitle", ""):
                    return str(r["databaseId"]), r.get("url", "")
            self.sleep(5)
        raise StepFailed(f"the dispatched {self.XCC_WORKFLOW} run (token {token}) never appeared in gh run list")

    def step_cross_compile(self):
        """CROSS-COMPILE ON GITHUB, RELEASE ONLY. The frozen commit's changed
        Linux GPU bindings, every tier and arch, each on its own free runner;
        diffed against the previous release's source commit (the reuse plan's
        base). Refuses when there is no such commit: it never widens."""
        prev = release_reuse.previous_release() or {}
        base = prev.get("source_commit")
        if not base:
            raise StepFailed("cross-compile: no published release record names a source commit to diff "
                             "against (bench/results/release_evidence/alpha-manifest-linux.json)")
        reused = self.cross_compile_reuse(base)
        if reused:
            return reused
        token = f"release-{self.version}-{self.commit[:12]}-{int(time.time())}"
        wf_ref = os.environ.get("MOJOLEARN_XCC_WORKFLOW_REF", "main")
        self.must(["gh", "workflow", "run", self.XCC_WORKFLOW, "--ref", wf_ref, "-f", f"ref={base}",
                   "-f", f"commit={self.commit}", "-f", f"archs={self.XCC_ARCHS}",
                   "-f", "tiers=" + ",".join(self.release_modes), "-f", f"token={token}"],
                  what="gh workflow run " + self.XCC_WORKFLOW)
        if self.dry:
            return f"would dispatch {self.XCC_WORKFLOW} on {self.commit[:12]} vs {prev.get('version')} and wait"
        run_id, url = self.xcc_find_run(token)
        self.say(f"  cross-compile-check run {url} (waiting; log {self.rel / 'cross-compile.log'})")
        rc, _ = self.gh("run", "watch", run_id, "--exit-status", "--interval", "30",
                        log=self.rel / "cross-compile.log")
        if rc == 0:
            write_json(self.cross_compile_receipt(), dict(schema=self.XCC_SCHEMA, verdict="PASS", commit=self.commit,
                       base=base, archs=self.XCC_ARCHS, tiers=list(self.release_modes), run=url, at=now()))
            return f"PASS vs {prev.get('version')} ({base[:12]}): {url}", dict(run=url, base=base)
        _, out = self.gh("run", "view", run_id, "--json", "jobs", "--jq",
                         '.jobs[] | select(.conclusion != "success" and .conclusion != "skipped") | .name')
        names = [n for n in out.splitlines() if n.strip()]
        named = [n for n in names if n != "summary"] or names
        raise StepFailed(f"cross-compile-check on {self.commit[:12]} vs {prev.get('version')} ({base[:12]}) did "
                         f"not pass: {', '.join(named) or f'gh run watch exited {rc}'}; {url}")

    XCC_SCHEMA = "mojolearn.release-cross-compile.v1"
    #: Files outside the build scripts' source closures that still decide what
    #: cross-compile-check compiles and how: the toolchain pins, the workflow
    #: and its planner.
    XCC_INPUTS = ("pixi.toml", "pixi.lock", ".github/workflows/cross-compile-check.yml",
                  "tools/cross_compile_check.py")

    def cross_compile_receipt(self):
        return self.rel / "cross-compile.json"

    def kernel_changes(self, old, new):
        """Native build inputs that differ between two commits (None: unreadable).

        The build scripts' source closures (tools/release_reuse.tree_facts, the
        facts cross_compile_check.py selects its jobs from) and XCC_INPUTS.
        Empty means cross-compile-check at NEW compiles exactly what it compiled
        at OLD, from the same bytes."""
        import cross_compile_check as xcc
        cache = self.evidence / "release" / "identities"
        try:
            scripts = xcc.changed_scripts(release_reuse.facts_for_commit(old, cache),
                                          release_reuse.facts_for_commit(new, cache))
            names = git("diff", "--name-only", old, new, "--", *self.XCC_INPUTS).split()
        except (Exception, SystemExit):  # glue: an unreadable tree is never "unchanged"
            return None
        return sorted(["bindings/" + name for name in scripts] + names)

    def cross_compile_reuse(self, base):
        """A PASS of an earlier freeze of this version, against the same base, archs
        and tiers, when no native build input changed since that commit (plan item 2:
        0.8.63 reran it, about 30 minutes, on every freeze). Returns the step result or None."""
        for path in sorted(self.base.glob("*/cross-compile.json")):
            doc = read_json(path) or {}
            if (doc.get("schema") != self.XCC_SCHEMA or doc.get("verdict") != "PASS" or doc.get("base") != base
                    or doc.get("archs") != self.XCC_ARCHS or doc.get("tiers") != list(self.release_modes)
                    or not doc.get("commit")):
                continue
            passed = doc["commit"]
            changed = [] if passed == self.commit else self.kernel_changes(passed, self.commit)
            if changed is None or changed:
                self.say(f"  cross-compile: the PASS at {passed[:12]} is not reused: "
                         + ("its tree is unreadable" if changed is None else "changed " + ", ".join(changed[:5])))
                continue
            return (f"PASS reused from {passed[:12]} (no native build input changed since): {doc.get('run')}",
                    dict(run=doc.get("run"), base=base, reused_from=passed, receipt=str(path)))
        return None

    # ------------------------------------------------------------ reuse
    @property
    def plan_path(self):
        return self.rel / "reuse" / "plan.json"

    def plan(self):
        """The reuse plan of the frozen commit, as the builders run after the
        route overlay: read from the release directory when this commit's
        plan with this overlay is there, else computed (in memory only under
        --dry-run; never under --status)."""
        with self._lock:
            if self._plan is not None:
                return self._plan
            doc = read_json(self.plan_path)
            if (doc and doc.get("schema") == release_reuse.PLAN_SCHEMA and doc.get("commit") == self.commit
                    and release_reuse.record_modes(doc) == self.release_modes):
                if self.readonly:
                    self._plan = doc
                    return doc
                if (doc.get("builders_override") or {}) == self.builders_override():
                    self._plan = doc
                    return doc
            if self.readonly:
                return None
            override = self.builders_override()
            # evidence_root: a published candidate of this tree (a TestPyPI freeze under
            # <evidence>/release/<version>/<commit12>/) is weighed beside the PyPI record and
            # wins when it reuses more (release_reuse.candidate_builds, Andrew 2026-10-15)
            self._plan = release_reuse.make_plan(self.commit, self.evidence / "release" / "identities",
                                                 builders_override=override or None, evidence_root=self.evidence,
                                                 modes=self.release_modes)
            return self._plan

    def step_reuse_plan(self):
        plan = self.plan()
        m = self.overlay()
        if m["files"]:
            self.say(f"  route overlay from the tooling checkout ({self.tooling_commit[:12]}): "
                     + ", ".join(sorted(m["files"])))
        self.say(release_reuse.render(plan, full=self.dry))
        prev = plan.get("previous") or {}
        summary = "%d REUSE, %d BUILD, legs: %s" % (
            len(release_reuse.plan_rows(plan, decision="REUSE")), len(release_reuse.plan_rows(plan, decision="BUILD")),
            ", ".join(plan["legs"]) or "none")
        if self.dry:
            return "would write " + str(self.plan_path) + "; " + summary
        old = read_json(self.plan_path) or {}
        if (old.get("commit") != self.commit or old.get("builders_override") != plan.get("builders_override")
                or release_reuse.record_modes(old) != self.release_modes):
            write_json(self.plan_path, plan)
        return (f"against {prev.get('version', 'no published release')}"
                f"{' ' + (prev.get('channel') or 'pypi') if prev else ''}: {summary}; {self.plan_path}")

    def previous_wheel(self, platform):
        """The plan's source build's wheel for `platform` (the last PyPI
        release or a published candidate), verified against its record (a
        local copy, else the index it was published to)."""
        prev = (self.plan().get("previous") or {})
        if self.dry:
            info = prev.get(platform) or {}
            return Path("<published %s wheel %s>" % (platform, info.get("wheel", "?")))
        whl = release_reuse.published_wheel(prev, platform, self.evidence)
        # LEGACY_SPLIT_PACKAGES: a previous release up to 0.8.65 shipped its PTX set in
        # mojolearn-nvidia-ptx; its bytes stay reusable from that wheel.
        plugins = [k for k in PLUGIN_PACKAGES + LEGACY_SPLIT_PACKAGES if prev.get(k)] if platform == "linux" else []
        if plugins:
            # a split release: its Linux bytes are the core and its plugins
            wheels = [whl] + [release_reuse.published_wheel(prev, k, self.evidence) for k in plugins]
            return merge_split(wheels, self.rel / "reuse" / "previous-split" / whl.name)
        return whl

    # ------------------------------------------------------------ leg reuse by identity
    def admission_path(self, name):
        return self.rel / "legs" / f"{name}.reused.json"

    def admitted_leg(self, name):
        """The admission of a completed leg of an earlier freeze for NAME in this
        freeze, still true (its tree verifies), or None. Read-only."""
        doc = read_json(self.admission_path(name))
        if not doc or doc.get("admitted_for") != self.commit:
            return None
        cache = self.__dict__.setdefault("_admitted", {})
        key = (name, json.dumps(doc, sort_keys=True))
        if key in cache:
            return cache[key]
        vendor, arch = name.split("-", 1)
        ok, _, proof_sha = verify_leg_tree(doc["release_build"], vendor, arch, doc["built_from"], doc.get("exit_file"))
        cache[key] = doc if ok and proof_sha == doc.get("proof_sha256") else None
        return cache[key]

    def admitted_legs(self):
        out = {}
        for vendor, arch in release_reuse.LINUX_SETS:
            doc = self.admitted_leg(f"{vendor}-{arch}")
            if doc:
                out[f"{vendor}-{arch}"] = doc
        return out

    def leg_candidates(self, name, vendor, arch, ident):
        out = []
        here = self.rel.resolve()
        for p in sorted(self.base.glob(f"*/legs/{name}.provenance.json")):
            if p.parent.parent.resolve() == here:
                continue
            doc = read_json(p)
            if doc:
                out.append((str(p), doc))
        if ident:
            for e in self.ledger().find(f"build/{vendor}-{arch}/{ident['digest']}/"):
                if e.get("verdict") == "PASS" and e.get("provenance"):
                    out.append(("ledger " + e.get("key", "?"), e["provenance"]))
        return out

    def find_reusable_leg(self, name, vendor, arch, write=True):
        """(admission, why not): a completed leg of an earlier freeze whose set
        identity AND build tooling digest equal this freeze's, whose exit is 0,
        whose proof is complete for the commit it built, whose every binary is
        the proof's bytes and whose route overlay verified on the box. Anything
        else (unequal, unreadable, dirty tooling, missing evidence) is refused
        and the leg is built."""
        ident = self.set_identity(vendor, arch)
        if not ident:
            return None, "the set identity is unreadable (no plan, or a binding without a readable identity)"
        if not self.packer_takes_legs():
            return None, "the frozen source's packer predates leg origins in reuse.json"
        if (vendor, arch) == release_reuse.PTX_SET:
            packer = file_at(self.commit, "packaging/linux/pack_wheel.py") or ""
            if ("PTX_RELEASE_ADMISSION_SCHEMA" not in packer
                    or not file_at(self.commit, "packaging/linux/ptx_reuse.py")):
                return None, "the frozen packer cannot rebind retained PTX provenance; packaging support is missing"
        t = self.tooling()
        if not t["digest"]:
            return None, "the build tooling has uncommitted changes (" + ", ".join(t["dirty"]) + ")"
        why = []
        for where, c in self.leg_candidates(name, vendor, arch, ident):
            if c.get("source_commit") == self.commit:
                continue
            prior_ident = release_reuse.normalized_set_identity(c.get("set_identity"))
            neutral = None
            if prior_ident is not None and prior_ident["digest"] != ident["digest"]:
                neutral = self.byte_neutral_runtime_change(c["source_commit"], prior_ident, ident)
            if prior_ident is None or (prior_ident["digest"] != ident["digest"] and not neutral):
                why.append(f"{where}: set identity differs")
                continue
            if c.get("tooling_digest") != t["digest"]:
                why.append(f"{where}: build tooling differs")
                continue
            ok, reason, proof_sha = verify_leg_tree(c["release_build"], vendor, arch, c["source_commit"], c.get("exit_file"))
            if ok:
                ok, reason = overlay_verified(c, c.get("out_dir", ""))
            if not ok:
                why.append(f"{where}: {reason}")
                continue
            doc = dict(schema="mojolearn.release-leg-admission.v1", leg=name, set=f"{vendor}/{arch}",
                       built_from=c["source_commit"], built_with_tooling=c.get("tooling_commit"),
                       admitted_for=self.commit, admitted_with_tooling=t["commit"],
                       because=("the set identity (every binding identity of the set, the host bindings and the "
                                "runtime closure) and the build tooling digest equal this freeze's"
                                if not neutral else "every binding identity and the build tooling digest equal "
                                "this freeze's; the runtime closure differs only where no byte is built: " + neutral),
                       set_identity_digest=ident["digest"], tooling_digest=t["digest"],
                       original_set_identity=c.get("set_identity"),
                       release_build=c["release_build"], exit_file=c.get("exit_file"),
                       set_dir=str(Path(c["release_build"]) / "build" / "sets" / vendor / arch),
                       proof_sha256=proof_sha, provenance=where, at=now())
            if write and not self.dry and not self.readonly:
                write_json(self.admission_path(name), doc)
            return doc, None
        return None, "; ".join(why) or "no completed leg of an earlier freeze with this set identity"

    #: What the frozen source's PTX packer must carry before the PTX set can be
    #: taken with the host owner's bytes (packaging/linux/ptx_reuse.py binds the
    #: set's host files to the PTX leg's own original proof otherwise).
    PTX_HOST_OWNER_MARKER = "HOST_OWNER_ADMISSION"

    def requirement_units(self, vendor, arch):
        """(wanted requirements this leg owns, evaluated ledger rows, attempts
        by id), read without writing the ledger."""
        import release_compile_evidence as compile_evidence
        ledger = self.readiness()
        doc = ledger.read()
        rows = ledger.plan(self.readiness_manifest())
        wanted = {name: w for name, w in doc["requirements"].items()
                  if compile_evidence.belongs_to_leg(w, vendor, arch)}
        return wanted, rows, {a["id"]: a for a in doc.get("attempts", [])}

    def record_finished_legs(self, name, vendor, arch, ident):
        """Record, from its own output, each finished earlier-freeze leg whose begun
        compile runs were never recorded (0.8.63: cuda-sm_89 at 9e79e6e65085 needed
        record_compile_requirements by hand). A leg qualifies only when it exited 0,
        its proof is complete for the commit it built and its tree verifies; the
        receipt is parsed from that output exactly as linux-wait would have.
        Returns the provenance files recorded."""
        if self.dry or self.readonly:
            return []
        import release_compile_evidence as compile_evidence
        document = self.readiness().read()
        closed = ({a["run_id"] for a in document.get("attempts", [])}
                  | {i["run_id"] for i in document.get("interruptions", [])})
        begun = {r["id"] for r in document.get("runs", [])}
        recorded = []
        for where, c in self.leg_candidates(name, vendor, arch, ident):
            if where.startswith("ledger ") or not c.get("release_build") or not c.get("source_commit"):
                continue
            launch = c.get("compile_requirements") or {}
            open_runs = [r.get("run_id") for r in (launch.get("runs") or {}).values()
                         if r.get("run_id") in begun and r.get("run_id") not in closed]
            if launch.get("schema") != compile_evidence.LAUNCH_SCHEMA or not open_runs:
                continue
            workdir = Path(where).parent
            leg = Leg(name, vendor, arch, [], {}, c["release_build"], workdir, workdir / name)
            if leg.provenance_file.resolve() != Path(where).resolve() or not leg.done(c["source_commit"]):
                continue
            ok, reason, _ = verify_leg_tree(c["release_build"], vendor, arch, c["source_commit"], c.get("exit_file"))
            if not ok:
                self.say(f"  {name}: finished leg {where} is not recorded: {reason}")
                continue
            try:
                self.record_compile_requirements(leg)
            except StepFailed as error:
                self.say(f"  {name}: finished leg {where} is not recorded: {error}")
                continue
            self.say(f"  {name}: recorded the {len(open_runs)} begun compile run(s) of the finished leg {where}")
            recorded.append(where)
        return recorded

    #: packaging/portable_math files the build consumes: stage.py compiles
    #: portable_math.c (with powers_of_ten.h) into .libs. The rest of the
    #: directory (wheel.py's audit, tests, README) never reaches a compiled byte.
    PORTABLE_MATH_BUILD_INPUTS = ("packaging/portable_math/stage.py", "packaging/portable_math/portable_math.c",
                                  "packaging/portable_math/powers_of_ten.h")

    def byte_neutral_runtime_change(self, built_from, prior, ident):
        """Why a set identity change cannot alter the leg's compiled bytes, or None.

        Accepted only when every binding identity is equal, the runtime closure
        identities differ in nothing but the packaging/portable_math directory
        digest, and every file changed there since BUILT_FROM is not a build
        input (0.8.63: packaging/portable_math/wheel.py's audit logic). The
        earlier runtime identity is read from that freeze's own reuse plan and
        must hash to the runtime digest its leg recorded."""
        if prior.get("set") != ident.get("set") or prior.get("bindings") != ident.get("bindings"):
            return None
        old_plan = read_json(self.base / built_from[:12] / "reuse" / "plan.json") or {}
        old = (old_plan.get("runtime") or {}).get("identity")
        new = (self.plan().get("runtime") or {}).get("identity")
        if (old_plan.get("commit") != built_from or not isinstance(old, dict) or not isinstance(new, dict)
                or release_reuse.digest_of(old) != prior.get("runtime")
                or release_reuse.digest_of(new) != ident.get("runtime")):
            return None
        if {k: v for k, v in old.items() if k != "portable_math"} != {k: v for k, v in new.items() if k != "portable_math"}:
            return None
        try:
            changed = git("diff", "--name-only", built_from, self.commit, "--", "packaging/portable_math").split()
        except subprocess.CalledProcessError:
            return None
        if not changed or any(p in self.PORTABLE_MATH_BUILD_INPUTS for p in changed):
            return None
        return "only non-build files of packaging/portable_math changed (" + ", ".join(changed) + ")"

    def find_requirement_leg(self, name, vendor, arch, write=True):
        """(admission, why not, every owned unit retained) for a leg whose set
        identity changed ONLY in what it does not own.

        TAKE THE LEG WHOSE OWN UNITS PASSED (lane release-auto-reuse-2,
        2026-10-31; AGENTS.md "Reuse good old runs, automatically"). Every leg
        also stages the merged CPU host bindings and the runtime closure, so a
        host-only change (cluster/host/kmeans_oracle.mojo for 0.8.63) moved the
        set identity of cuda-sm_80 and hip-gfx942, and linux-builds then
        stopped: "all compile inputs already have retained passes". A leg is
        taken from an earlier freeze here when, and only when:
          * it is not the host owner (release_reuse.HOST_LEG), which builds;
          * every compile requirement it owns (release_compile_evidence
            .belongs_to_leg) reads `reuse` in the readiness ledger: the same
            declared inputs (binding identity, builders, flags, toolchain,
            image, compiler) with verified retained evidence;
          * that earlier leg was launched for exactly those units, and every
            unit it began is that unit's retained pass (the others it compiled
            under passes retained for the same inputs); the newest such leg wins;
          * its device bindings' identities equal this freeze's (only host
            rows and the runtime digest may differ);
          * its tree verifies (exit 0, a complete proof of the commit it
            built, every binary the proof's bytes) and its route overlay
            verified on the box.
        Its host bindings never ship: linux-assemble replaces them with the
        host owner's, built at this freeze, and refuses unless the runtime
        closure is byte-identical to the owner's. For the PTX set the frozen
        packer must support that (PTX_HOST_OWNER_MARKER). Anything else is
        refused with its reason."""
        # The host owner may be taken too (its host bindings are then its own build of
        # these exact host inputs: it owns those units); see the owner check below.
        owner = (vendor, arch) == release_reuse.HOST_LEG
        ident = self.set_identity(vendor, arch)
        if not ident:
            return None, "the set identity is unreadable", False
        if not self.packer_takes_legs():
            return None, "the frozen source's packer predates leg origins in reuse.json", False
        if write:
            self.record_finished_legs(name, vendor, arch, ident)
        try:
            wanted, rows, attempts = self.requirement_units(vendor, arch)
        except (Exception, SystemExit) as exc:  # glue: an unreadable ledger is not "retained"
            if self.dry:
                return None, self.DRY_READINESS, False
            return None, f"the readiness ledger cannot be read ({exc})", False
        if not wanted:
            return None, "it owns no declared compile requirement", False
        open_units = sorted(n for n in wanted if rows.get(n, {}).get("action") != "reuse")
        if open_units:
            return None, (f"{len(open_units)} of its {len(wanted)} compile requirement(s) have no retained pass "
                          f"for the current inputs (e.g. {open_units[0]})"), False
        passed = {}
        for n in wanted:
            attempt = attempts.get(rows[n].get("attempt"))
            if not attempt or attempt.get("status") != "passed" or attempt.get("fingerprint") != wanted[n]["fingerprint"]:
                return None, f"{n}: the retained pass is not an attempt of the current inputs", True
            passed[n] = attempt["run_id"]
        if (vendor, arch) == release_reuse.PTX_SET and self.PTX_HOST_OWNER_MARKER not in (
                file_at(self.commit, "packaging/linux/ptx_reuse.py") or ""):
            return None, ("the frozen source's PTX packer (packaging/linux/ptx_reuse.py) binds the PTX set's host "
                          "bytes and set identity to the PTX leg's own proof, so the set cannot ship the host owner's "
                          "bytes; it needs packer support and a refreeze"), True
        device = {k: v for k, v in ident["bindings"].items() if "/host/" not in k}
        why = []
        # newest launch first: the latest complete build of these units wins
        for where, c in sorted(self.leg_candidates(name, vendor, arch, ident),
                               key=lambda wc: str(wc[1].get("launched_at") or ""), reverse=True):
            if c.get("source_commit") == self.commit:
                continue
            # THE LEG WAS LAUNCHED FOR EXACTLY THESE UNITS (`selected`), and every
            # unit it began (`runs`) is the retained pass of that unit. A unit it
            # did not begin already had a retained pass for the same inputs when
            # it launched (begin reuses an unchanged pass), and the leg compiled
            # it with the rest of its set: its bytes are under the leg's proof.
            launch = c.get("compile_requirements") or {}
            runs = launch.get("runs") or {}
            selected = launch.get("selected") or sorted(runs)
            if sorted(selected) != sorted(wanted) or not set(runs) <= set(wanted):
                why.append(f"{where}: it was launched for other compile units")
                continue
            if any(runs[n].get("run_id") != passed[n] or runs[n].get("fingerprint") != wanted[n]["fingerprint"]
                   for n in runs):
                why.append(f"{where}: a unit it began is not that unit's retained pass")
                continue
            prior = release_reuse.normalized_set_identity(c.get("set_identity"))
            if prior is None or {k: v for k, v in prior["bindings"].items() if "/host/" not in k} != device \
                    or prior.get("set") != ident.get("set"):
                why.append(f"{where}: a device binding identity differs")
                continue
            # The owner's host bindings ship in every set, so for it every binding
            # identity (host rows included) must equal this freeze's.
            if owner and prior["bindings"] != ident["bindings"]:
                why.append(f"{where}: a host binding identity of the host owner differs")
                continue
            ok, reason, proof_sha = verify_leg_tree(c["release_build"], vendor, arch, c["source_commit"],
                                                    c.get("exit_file"))
            if ok:
                ok, reason = overlay_verified(c, c.get("out_dir", ""))
            if not ok:
                why.append(f"{where}: {reason}")
                continue
            t = self.tooling()
            doc = dict(schema="mojolearn.release-leg-admission.v1", kind="compile-requirements", leg=name,
                       set=f"{vendor}/{arch}", built_from=c["source_commit"], built_with_tooling=c.get("tooling_commit"),
                       admitted_for=self.commit, admitted_with_tooling=t.get("commit"),
                       because=(f"every compile requirement it owns ({len(wanted)}) has a retained pass of its own "
                                "runs for the current inputs, its device binding identities equal this freeze's, "
                                + ("and, as the host owner, its host binding identities too; its host bindings ship"
                                   if owner else "and its host bindings are replaced by the host owner's")),
                       host_from=None if owner else host_owner_leg(), requirements={n: dict(run_id=passed[n],
                                                                         fingerprint=wanted[n]["fingerprint"])
                                                                 for n in sorted(wanted)},
                       set_identity_digest=ident["digest"], tooling_digest=c.get("tooling_digest"),
                       admitted_with_tooling_digest=t.get("digest"), original_set_identity=c.get("set_identity"),
                       release_build=c["release_build"], exit_file=c.get("exit_file"),
                       set_dir=str(Path(c["release_build"]) / "build" / "sets" / vendor / arch),
                       proof_sha256=proof_sha, provenance=where, at=now())
            if write and not self.dry and not self.readonly:
                write_json(self.admission_path(name), doc)
            return doc, None, True
        return None, "; ".join(why) or "no completed leg of an earlier freeze began these units", True

    def admit_published_ptx(self, write=True):
        """(admission, why not) of THE PUBLISHED PTX SET, taken whole when no
        PTX binding builds (release_reuse._settle_ptx, lane ptx-reuse
        2026-11-01). Its leg is the one that built the previous release's PTX
        bytes, read from the retained proof the plan found (the record's
        ptx-proof/, else that leg's tree under the evidence root; a record-only
        proof is materialized from the published wheel's own bytes). It is
        admitted only when that tree verifies (a complete proof of the commit it
        built, every device file the proof's bytes), the proof's device bytes are
        the published wheel's, its device identities are this freeze's and the
        frozen packer carries HOST_OWNER_ADMISSION: its host bindings are then
        the host owner's, built at this freeze, and ptx_reuse.rebind_manifest
        rebinds its original PTX_BASELINE.json to this commit inside assembly."""
        name = "%s-%s" % release_reuse.PTX_SET
        vendor, arch = release_reuse.PTX_SET
        plan = self.plan()
        proof = (plan or {}).get("ptx_proof")
        if not proof:
            return None, "the plan takes no published PTX set"
        if self.PTX_HOST_OWNER_MARKER not in (file_at(self.commit, "packaging/linux/ptx_reuse.py") or ""):
            return None, "the frozen source's PTX packer cannot ship the host owner's bytes in the PTX set"
        if not self.packer_takes_legs():
            return None, "the frozen source's packer predates leg origins in reuse.json"
        ident = self.set_identity(vendor, arch)
        if not ident:
            return None, "the set identity is unreadable"
        if self.dry:
            return None, "dry run: would take the published PTX set with the proof of " + proof["source"]
        whl = self.previous_wheel("linux")
        rb = release_reuse.materialize_ptx_leg(proof, whl, self.rel / "reuse" / "ptx-published" / "release-build")
        ok, reason, proof_sha = verify_leg_tree(rb, vendor, arch, proof["built_from"])
        if not ok or proof_sha != proof["proof_sha256"]:
            return None, f"the published PTX set's leg tree does not verify: {reason or 'its proof changed'}"
        shipped = release_reuse.linux_payload(whl).get("extensions") or {}
        named = read_json(Path(rb) / release_reuse.PTX_WITNESSES["proof"]).get("extensions") or {}
        if any(shipped.get(k) != v for k, v in named.items()):
            return None, "the proof's PTX bytes are not the published wheel's"
        doc = dict(schema="mojolearn.release-leg-admission.v1", kind="published-ptx", leg=name,
                   set=f"{vendor}/{arch}", built_from=proof["built_from"], admitted_for=self.commit,
                   admitted_with_tooling=self.tooling().get("commit"),
                   because=("no PTX binding builds: the published PTX set is taken whole with the original proof of "
                            "the leg that built it (" + proof["source"] + "); its device binding identities equal "
                            "this freeze's and its host bindings are replaced by the host owner's"),
                   host_from=host_owner_leg(), set_identity_digest=ident["digest"],
                   tooling_digest=proof.get("tooling_digest") or proof["proof_sha256"],
                   original_set_identity=proof["original_set_identity"], release_build=str(rb), exit_file=None,
                   set_dir=str(Path(rb) / "build" / "sets" / vendor / arch), proof_sha256=proof_sha,
                   provenance=proof["source"], at=now())
        if write and not self.readonly:
            write_json(self.admission_path(name), doc)
        return doc, None

    def host_owner_origin(self):
        """{name, dir, origin} of the host owner leg whose host/ ships in every
        set: the one built at THIS freeze, or the one taken whole from an
        earlier freeze (its set identity, host rows included, equals this
        freeze's, so its host bytes are a build of this freeze's host inputs).
        A host owner taken by compile requirements has no current host bytes
        and is refused (find_requirement_leg never takes it)."""
        owner = host_owner_leg()
        doc = self.admitted_leg(owner)
        if doc:
            if doc.get("kind") == "compile-requirements" and doc.get("host_from") is not None:
                raise StepFailed(f"the host owner {owner} was taken with another leg's host bytes")
            return dict(dir=Path(doc["set_dir"]), name=owner, origin=self.leg_origin(doc))
        leg = next((l for l in linux_legs(self) if l.name == owner), None)
        if leg is None:
            raise StepFailed(f"a leg is taken with the host owner's bindings and {owner} was neither built at "
                             "this freeze nor taken whole")
        vendor, arch = owner.split("-", 1)
        proof = leg.release_build / "build" / "build-provenance.json"
        ident = self.set_identity(vendor, arch) or {}
        return dict(dir=leg.release_build / "build" / "sets" / vendor / arch, name=owner,
                    origin=dict(source_commit=self.commit, leg=owner, proof_sha256=sha256(proof),
                                admitted_for=self.commit, set_identity_digest=ident.get("digest"),
                                tooling_digest=self.tooling().get("digest"), release_build=str(leg.release_build)))

    def packer_takes_legs(self):
        """The pack runs the SOURCE's pack_wheel.py; one older than leg
        origins would pack a taken leg's files as the published release's."""
        return "LEG_ORIGIN_KEYS" in (file_at(self.commit, "packaging/linux/pack_wheel.py") or "")

    def leg_origin(self, doc):
        """reuse.json's origin of an admitted leg (pack_wheel.LEG_ORIGIN_KEYS)."""
        return dict(source_commit=doc["built_from"], leg=doc["leg"], proof_sha256=doc["proof_sha256"],
                    admitted_for=doc["admitted_for"], set_identity_digest=doc["set_identity_digest"],
                    tooling_digest=doc["tooling_digest"], release_build=doc["release_build"],
                    **({"original_set_identity": doc["original_set_identity"]}
                       if doc.get("original_set_identity") else {}))

    def reused_leg_objects(self):
        out = []
        for name, doc in self.admitted_legs().items():
            vendor, arch = name.split("-", 1)
            out.append(Leg(name, vendor, arch, [], {}, doc["release_build"], self.rel / "legs", doc["release_build"]))
        return out

    def linux_group_plan(self):
        """Per Linux merged group, before any leg compiles: `hit (closure unchanged
        since <previous release>)` or `compile: <reason>` (tools/compile_closures.py
        group_plan: two `git archive` trees, closure-v2 rule, nothing built)."""
        previous = (self.plan().get("previous") or {}).get("source_commit")
        if not previous:
            return ["linux groups: no previous release commit; every group compiles"]
        try:
            sys.path.insert(0, str(ROOT / "tools"))
            import compile_closures
            return compile_closures.group_plan(self.src, previous, self.commit, "linux")
        except Exception as exc:  # a plan line is advice; never fail the step over it
            return [f"linux groups: no plan ({type(exc).__name__}: {exc})"]

    def step_linux_builds(self):
        legs = linux_legs(self)
        for line in self.linux_group_plan():
            self.say("  " + line)
        prev = (self.plan().get("previous") or {}).get("version")
        if not legs:
            return f"no build leg: every Linux binding is taken from {prev} (nothing rented for builds)"
        launch, taken, refused = [], [], []
        for leg in legs:
            if leg.running():
                self.say(f"  {leg.name}: already running (pid {leg.pid()}), log {leg.log}")
                continue
            if leg.done(self.commit):
                self.say(f"  {leg.name}: already done at this freeze")
                continue
            doc = self.admitted_leg(leg.name)
            why = None
            if not doc:
                doc, why = self.find_reusable_leg(leg.name, leg.vendor, leg.arch)
            if not doc:
                # Not the same set: its own compile units may still all have
                # retained passes (find_requirement_leg). Then it is taken, or
                # refused by name: launching it would only stop at begin.
                doc, why_units, retained = self.find_requirement_leg(leg.name, leg.vendor, leg.arch)
                if not doc and retained:
                    refused.append(f"{leg.name}: every compile requirement it owns has a retained pass, and no "
                                   f"earlier leg can be taken for it: {why_units} (set identity: {why})")
                    self.say("  " + refused[-1])
                    continue
                why = f"{why}; by compile requirements: {why_units}" if why_units else why
            if doc:
                taken.append(leg.name)
                self.say(f"  {leg.name}: TAKEN, not launched: built from {doc['built_from'][:12]} "
                         f"({doc['provenance']}), admitted for {doc['admitted_for'][:12]} because {doc['because']}")
                continue
            self.say(f"  {leg.name}: {leg.describe()}  [{self.plan()['leg_reasons'].get(leg.name, '')}]"
                     + (f"  (not reusable: {why})" if why else ""))
            launch.append(leg)
        ptx_name = "%s-%s" % release_reuse.PTX_SET
        if ptx_name not in {l.name for l in legs} and (self.plan() or {}).get("ptx_proof") \
                and not self.admitted_leg(ptx_name):
            # NO PTX BINDING BUILDS: the published PTX set is taken whole
            doc, why = self.admit_published_ptx()
            if doc:
                taken.append(ptx_name)
                self.say(f"  {ptx_name}: TAKEN, not launched: the published set built from {doc['built_from'][:12]}, "
                         f"proved by {doc['provenance']}")
            elif self.dry:
                self.say(f"  {ptx_name}: {why}")
            else:
                refused.append(f"{ptx_name}: the plan takes the published PTX set and it cannot be admitted: {why}")
                self.say("  " + refused[-1])
        if self.dry:
            return (f"would launch {len(launch)} leg(s) ({self.args.build_backend})"
                    + (f"; would take {', '.join(taken)} from an earlier freeze" if taken else "")
                    + (f"; REFUSED {'; '.join(refused)}" if refused else ""))
        launch_detached(self, launch)
        if refused:
            # The other legs are launched; nothing is rebuilt for the refused one.
            raise StepFailed("; ".join(refused))
        return (f"{len(launch)} leg(s) launched or running ({self.args.build_backend})"
                + (f"; taken from an earlier freeze: {', '.join(taken)}" if taken else ""))

    # ------------------------------------------------------------ macOS
    def macos_wheel(self):
        found = sorted((self.rel / "macos").glob("mojolearn-*-macosx_*.whl"))
        return found[-1] if found else None

    def stale_preflight(self, phase):
        """THE STALE-ARTIFACT PRE-FLIGHT (tools/release_stale_artifacts.py): every untracked or ignored file in the
        source checkout's python/mojolearn/ that the wheel's package-data globs would ship and this build does not
        produce is moved (never deleted) to <commit12>/stale-artifacts/<timestamp>/, each path logged, before
        PHASE runs (0.8.65: three stray extensions each cost a 75-minute build before build_release_wheel.sh
        refused them). A dry run lists them and moves nothing."""
        import release_stale_artifacts as stale
        merged = os.environ.get("MOJOLEARN_MERGED_BINDINGS", "1") != "0"
        moved, receipt = stale.preflight(self.src, self.rel / "stale-artifacts", self.release_modes, byte_lm=True,
                                         merged=merged, phase=phase, dry=self.dry, log=lambda m: self.say("  " + m))
        if receipt:
            with self._lock:
                self.state.setdefault("stale_artifacts", []).append(dict(phase=phase, receipt=str(receipt),
                                                                         moved=len(moved), at=now()))
                self.save()
        return moved, receipt

    def step_macos_build(self):
        w = self.macos_wheel()
        if w and wheel_commit(w) == self.commit:
            return "have " + w.name
        self.stale_preflight("build")
        # Four builds at once, as measured on the 10-core M4, never more than
        # this Mac's slot ceiling (tools/mac_slot.py slot_count), which
        # mac_slot.py would otherwise refuse.
        sys.path.insert(0, str(ROOT / "tools"))
        import mac_slot
        slots = str(min(4, mac_slot.slot_count()))
        # The binary cache is the build's provenance (key = exact source closure,
        # flags, toolchain -> compiled bytes): keep it release-scoped so the
        # compile-apple rows are admitted from it (macos-release-build) and a
        # shared cache's pruning can never remove it.
        env = dict(MOJOLEARN_BINCACHE_DIR=str(self.macos_bincache()),
                   MOJOLEARN_PACKAGE_BYTE_LM="1", MOJOLEARN_BUILD_JOBS=slots, MOJOLEARN_COMPILE_JOBS="1",
                   MOJOLEARN_RELEASE_MODES=" ".join(self.release_modes))
        # CROSS-RELEASE HITS (2026-11-04). The cache stays release-scoped, but
        # an archive whose closure-v2 key an earlier release already built is
        # copied in from that release's cache (verified, tools/bincache.py
        # seed_archive) instead of compiled: a group compiles only when its
        # import closure, flags or toolchain changed since some earlier release.
        seeds = self.macos_bincache_seeds()
        if seeds:
            env["MOJOLEARN_BINCACHE_SEED_DIRS"] = os.pathsep.join(str(d) for d in seeds)
        for line in self.macos_cache_plan(env):
            self.say("  " + line)
        plan = self.plan()
        src = self.src
        reuse = release_reuse.plan_rows(plan, release_reuse.MACOS, "REUSE")
        # A MERGED macOS BUILD (MOJOLEARN_MERGED_BINDINGS, default 1) compiles
        # each merged file whole: per-binding reuse cannot place one member's
        # published bytes, and build_release_wheel.sh refuses a reuse plan in
        # that layout, so every macOS binding is built.
        merged = os.environ.get("MOJOLEARN_MERGED_BINDINGS", "1") != "0"
        env["MOJOLEARN_MERGED_BINDINGS"] = "0" if not merged else "1"
        if merged and reuse:
            self.say(f"  merged macOS bindings: {len(reuse)} reusable per-binding file(s) are built instead "
                     "(MOJOLEARN_MERGED_BINDINGS=0 reuses them in the per-binding layout)")
            reuse = []
        if reuse:
            store = self.rel / "reuse" / "macos"
            if self.dry:
                self.say(f"  {len(reuse)} macOS binding(s) would be placed from the published "
                         f"{plan['previous']['version']} wheel, verified against its RECORD")
            else:
                release_reuse.assemble_macos(plan, self.previous_wheel("macos"), store, say=self.say)
            env.update(MOJOLEARN_REUSE_PLAN=str(store / "macos-plan.json"), MOJOLEARN_REUSE_DIR=str(store))
        # without --gpu-smoke: --build-only (the checklist's preverified build; macos-smoke's self-test gates it)
        self.must([PY, str(ROOT / "tools" / "mac_slot.py"), "--slots", slots, "run", "--",
                   "./packaging/macos/build_release_wheel.sh", *([] if self.gpu_smoke() else ["--build-only"])],
                  env=env, log=self.rel / "macos-build.log", what="build_release_wheel.sh", cwd=src)
        if self.dry:
            return "would copy python/dist/*.whl to " + str(self.rel / "macos")
        built = sorted((src / "python" / "dist").glob(f"mojolearn-{self.version}-*-macosx_*.whl"))
        if len(built) != 1 or wheel_commit(built[0]) != self.commit:
            raise StepFailed(f"expected one macOS wheel of {self.commit} in {src}/python/dist, found {built}")
        (self.rel / "macos").mkdir(parents=True, exist_ok=True)
        dest = self.rel / "macos" / built[0].name
        shutil.copy2(built[0], dest)
        return f"{dest.name} sha256 {sha256(dest)}"

    def smoke_python(self):
        for name in ("python3.12", "python3.13", "python3.11", "python3.14", "python3.10"):
            if shutil.which(name):
                return shutil.which(name)
        raise StepFailed("no python3.10..3.14 on PATH for the macOS smoke venv")

    def macos_self_test(self):
        """Without --gpu-smoke, the receipt publish-macos needs: tools/wheel_self_test.py."""
        return self.rel / "self-test-macos" / "results.json"

    def macos_self_test_passed(self):
        sys.path.insert(0, str(ROOT / "tools"))
        import wheel_self_test
        w = self.macos_wheel()
        return bool(w) and wheel_self_test.passed(self.macos_self_test(), w)

    def step_macos_smoke(self):
        if self.single():
            # --identity-on-wheels: the self-test's lanes are part of the Apple column, recorded once from this
            # wheel in gate-apple; nothing runs here
            return "SKIPPED (--identity-on-wheels): the Apple column is recorded from this wheel in gate-apple"
        w = self.macos_wheel()
        if not self.gpu_smoke():
            # docs/RELEASE_CHECKLIST.md's preverified macOS gate: a fresh venv, `verify --self-test`
            # (the bundled reference table reproduces the wheel's own bits). Metal lock: it fits.
            out = self.macos_self_test().parent
            if self.macos_self_test_passed():
                return "WHEEL-SELF-TEST PASSED (receipt exists)"
            if not w and not self.dry:
                raise StepFailed("no macOS wheel; run macos-build")
            self.stale_preflight("smoke")
            self.refuse_device_lanes_after_publish("macos-smoke")
            if out.exists() and not self.dry:
                out.rename(out.with_name(out.name + ".failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
            self.must([PY, str(ROOT / "tools" / "mac_slot.py"), "--wait-timeout", "3600", "metal", "--",
                       PY, str(ROOT / "tools" / "wheel_self_test.py"), w or "<macos wheel>", "--out", out,
                       "--python", self.smoke_python() if not self.dry else "python3.12"],
                      log=self.rel / "macos-smoke.log", what="macOS wheel self-test")
            if not self.dry and not self.macos_self_test_passed():
                raise StepFailed(f"macOS wheel self-test is not PASSED for this wheel: {self.macos_self_test()}")
            return "WHEEL-SELF-TEST PASSED"
        out = self.rel / "smoke-macos"
        if w and smoke_passed(out / "results.json", w):
            return "PASSED (receipt exists)"
        if not w and not self.dry:
            raise StepFailed("no macOS wheel; run macos-build")
        self.stale_preflight("smoke")
        self.refuse_device_lanes_after_publish("macos-smoke")
        if out.exists() and not self.dry:
            out.rename(out.with_name(out.name + ".failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        # the smoke is stdlib-only tooling (the Linux column ships the same file)
        self.must([PY, str(ROOT / "tools" / "mac_slot.py"), "--wait-timeout", "3600", "metal", "--",
                   PY, str(ROOT / "tools" / "qualify_verifier_wheel.py"), w or "<macos wheel>", "--scope", "expanded",
                   "--python", self.smoke_python() if not self.dry else "python3.12",
                   "--expected-source-commit", self.commit, "--output", out],
                  log=self.rel / "macos-smoke.log", what="macOS smoke")
        if not self.dry and not smoke_passed(out / "results.json", w):
            raise StepFailed(f"macOS smoke receipt is not PASSED for this wheel: {out / 'results.json'}")
        return "PASSED"

    # ------------------------------------------------------------ Linux
    def step_linux_wait(self):
        legs = [l for l in linux_legs(self) if not self.admitted_leg(l.name)]
        taken = self.reused_leg_objects()
        if not legs:
            if taken:
                return "no leg to wait for: " + ", ".join(l.name for l in taken) + " taken from an earlier freeze"
            return "no leg to wait for (every Linux binding is reused)"
        if self.dry:
            return "would wait for " + ", ".join(l.name for l in legs)
        if not any(l.pid() or l.exit_code() is not None for l in legs):
            launch_detached(self, legs)
        tried = {l.name: 0 for l in legs}
        # A leg that failed on stock walks to its next GPU type AT ONCE, while
        # the other legs keep running (0.8.19: the sm_89 leg sat an hour behind
        # the AMD build before its walk).
        while True:
            again = []
            for l in legs:
                if l.running() or l.exit_code() is None:
                    continue
                if (l.vendor == "hip" and l.exit_code() != 0 and getattr(l, "provider", "") == "do"
                        and amd_build_want(self.args) == "auto" and do_refused_live(l.log)):
                    # a GPU droplet went live after the route was chosen: walk to Hot Aisle
                    self.say(f"  {l.name}: DigitalOcean refused (a GPU droplet is live); trying Hot Aisle")
                    l.command, l.env = amd_leg_command(self, "hotaisle", l.workdir)
                    l.compile_provider_retry = True
                    l.provider, l.provider_reason = "hotaisle", "walked: DigitalOcean refused a rental"
                    self._amd_route = ("hotaisle", l.provider_reason)
                    again.append(l)
                    continue
                walk = NVIDIA_WALK.get(l.arch) if l.vendor == "cuda" and "--gpu" in l.command else None
                if walk and l.exit_code() != 0 and no_stock(l.log, l.workdir / l.name, l.out_dir) and tried[l.name] + 1 < len(walk):
                    tried[l.name] += 1
                    i = l.command.index("--gpu")
                    self.say(f"  {l.name}: {l.command[i + 1]} has no stock; trying {walk[tried[l.name]]}")
                    l.command[i + 1] = walk[tried[l.name]]
                    l.compile_provider_retry = True
                    again.append(l)
            if again:
                launch_detached(self, again)
                continue
            if not any(l.running() for l in legs):
                break
            self.say("  waiting: " + ", ".join(f"{l.name}={'running' if l.running() else l.exit_code()}"
                                                for l in legs))
            self.sleep(60)
        bad = []
        for l in legs:
            if l.exit_code() != 0 or not l.proof_ok(self.commit):
                self.record_compile_failure(l)
                closed = []
                if callable(getattr(self, "close_stopped_compile_producers", None)):
                    # A stopped producer's open runs close now, so the next
                    # linux-builds relaunches only this leg without a hand step.
                    try:
                        closed = [g for g in self.close_stopped_compile_producers(l) if g.get("closed")]
                    except (StepFailed, ValueError, OSError) as error:
                        self.say(f"  {l.name}: open compile runs not closed: {error}")
                bad.append(f"{l.name} (exit {l.exit_code()}, log {l.log}"
                           + (f"; {sum(len(g['run_ids']) for g in closed)} open compile run(s) interrupted, "
                              f"producer {closed[0]['state']}; linux-builds relaunches it" if closed else "") + ")")
                continue
            ok, why = overlay_verified(read_json(l.provenance_file), l.out_dir)
            if not ok:
                bad.append(f"{l.name} ({why})")
            else:
                # Preserve independent completed compile units even if another
                # leg fails; cross-leg packaging checks below remain separate.
                self.record_compile_requirements(l)
        if bad:
            raise StepFailed("build leg(s) failed: " + ", ".join(bad)
                             + ". Inspect retained evidence; repair or reconcile before another build.")
        # A leg taken by its compile requirements, or the published PTX set
        # taken whole, ships the host owner's host bindings (linux-assemble
        # replaces its own), so its stale host bytes are not compared here.
        shipped = [l for l in taken if (self.admitted_leg(l.name) or {}).get("kind")
                   not in ("compile-requirements", "published-ptx")]
        owner_taken = any(l.name == host_owner_leg() for l in shipped)
        # A host owner taken whole ships its host/ in every set (linux-assemble,
        # release_reuse.assemble_linux host_owner); the built legs' own host
        # bytes are then never packed, so they are not compared.
        mismatch = {} if owner_taken else host_digest_mismatches(legs + shipped)
        composition = None
        if mismatch:
            lines = [f"{name}: " + ", ".join(f"{leg}={d[:12]}" for leg, d in row.items())
                     for name, row in sorted(mismatch.items())]
            if shipped:
                raise StepFailed("host bindings differ across legs and a leg is taken from an earlier freeze "
                                 "(a host composition needs every leg's original build):\n  " + "\n  ".join(lines[:12]))
            self.say("  host bindings differ across legs:\n    " + "\n    ".join(lines[:12]))
            composition = compose_host_mismatch(legs, mismatch, self.src, self.rel / HOST_COMPOSITION_DIR)
            for name, row in composition["taken"].items():
                self.say(f"  {name}: taken from {row['from_leg']} ({row['sha256'][:12]}) for every set: "
                         + composition["why"])
            self.say(f"  host composition receipt {composition['receipt']} sha256 {composition['receipt_sha256']}")
        with self._lock:
            if composition:
                self.state["host_composition"] = composition
            else:
                self.state.pop("host_composition", None)
            self.save()
        for l in legs:
            self.record_leg(l)
        if composition:
            return ("all legs built; host bindings composed from " + composition["owner"] + " ("
                    + ", ".join(composition["taken"]) + " differed; receipt " + composition["receipt"] + ")")
        return ("all legs built; host bindings byte-identical across " + ", ".join(l.name for l in legs + taken)
                + (f" ({', '.join(l.name for l in taken)} taken from an earlier freeze)" if taken else ""))

    def host_composition(self):
        """linux-wait's retained host composition record, or None. Its receipt
        must still be the bytes linux-wait recorded."""
        rec = self.state.get("host_composition")
        if not rec:
            return None
        receipt = Path(rec["receipt"])
        if not receipt.is_file() or sha256(receipt) != rec.get("receipt_sha256"):
            raise StepFailed(f"the host composition receipt {receipt} is missing or changed; --redo linux-wait")
        return rec

    def record_leg(self, leg):
        """A leg that built, in the PASS ledger, keyed by what it built."""
        prov = read_json(leg.provenance_file) or {}
        ident = prov.get("set_identity") or {}
        if not ident.get("digest") or not prov.get("tooling_digest"):
            return
        ok, _, proof_sha = verify_leg_tree(leg.release_build, leg.vendor, leg.arch, self.commit, leg.exit_file)
        if not ok:
            return
        self.ledger_put(release_ledger.build_key(leg.vendor, leg.arch, ident["digest"], prov["tooling_digest"]), dict(
            verdict="PASS", kind="build", set=f"{leg.vendor}/{leg.arch}", source_commit=self.commit,
            tooling_commit=prov.get("tooling_commit"), provenance=prov,
            evidence=dict(release_build=str(leg.release_build), proof_sha256=proof_sha, log=str(leg.log))))
        self.record_compile_requirements(leg)

    def linux_final(self):
        """The final Linux core wheel (mojolearn_core-*, beside its plugins)."""
        return self.split_final("linux")

    def split_final(self, package):
        """A final split wheel: package 'linux' (mojolearn-core), a plugin
        ('nvidia', 'amd') or 'linux-meta' (the Linux mojolearn)."""
        prefix = SPLIT_PACKAGES[package][1]
        # A corrected unpublished artifact gets a new immutable path. Original
        # candidate receipts must still resolve the exact bytes they admitted;
        # overwriting linux/final would destroy their historical provenance.
        retained = (getattr(self, "state", {}).get("final_wheel_overrides") or {}).get(package)
        if retained:
            path = Path(retained["path"])
            if (not path.is_file() or not path.name.startswith(f"{prefix}-{self.version}-")
                    or not path.name.endswith(".whl") or "-manylinux" not in path.name
                    or sha256(path) != retained.get("sha256")):
                raise StepFailed(f"corrected {package} artifact is missing, changed, or misidentified")
            return path
        found = sorted((self.rel / "linux" / "final").glob(f"{prefix}-*-manylinux*.whl"))
        return found[-1] if found else None

    def split_finals(self):
        return {k: self.split_final(k) for k in SPLIT_PACKAGES}

    def assembled_marker(self):
        return self.rel / "reuse" / "sets" / "assembled.json"

    def needs_published_linux(self):
        plan = self.plan()
        built = {l.name for l in linux_legs(self)} if not self.dry else set(plan["legs"])
        taken = set(self.admitted_legs())
        return (bool(release_reuse.plan_rows(plan, release_reuse.LINUX, "REUSE"))
                or plan["runtime"]["decision"] == "REUSE"
                or any(f"{v}-{a}" not in built | taken for v, a in release_reuse.LINUX_SETS))

    def assembly_needed(self):
        """True when the plan takes any Linux bytes from the published wheel or
        a leg of an earlier freeze is taken; otherwise the packer reads the
        legs' sets directly, as before."""
        plan = self.plan()
        return (bool(release_reuse.plan_rows(plan, release_reuse.LINUX, "REUSE"))
                or plan["runtime"]["decision"] == "REUSE" or bool(self.admitted_legs()))

    def assembled_ok(self):
        d = read_json(self.assembled_marker())
        if not d:
            return False
        return (d.get("commit") == self.commit and d.get("plan_sha256") == sha256(self.plan_path)
                and sorted(d.get("taken") or []) == sorted(self.admitted_legs()))

    def step_linux_assemble(self):
        # the assembled sets of this plan first: a done assembly is never refused again (0.8.65)
        if self.assembly_needed() and self.assembled_ok():
            return "have " + str(self.assembled_marker().parent)
        composition = self.host_composition()
        if not self.assembly_needed():
            if composition:
                return ("not needed: every Linux binding is built by the legs; the host set of every leg is "
                        + composition["owner"] + "'s, composed in " + composition["sets"])
            return "not needed: every Linux binding is built by the legs"
        taken = self.admitted_legs()
        built = [l for l in linux_legs(self) if l.name not in taken]
        legs = {l.name: l.release_build / "build" / "sets" / l.vendor / l.arch for l in built}
        leg_reuse = {n: dict(dir=Path(d["set_dir"]), origin=self.leg_origin(d)) for n, d in taken.items()}
        host_owner = None
        for n, d in taken.items():
            if (d.get("kind") == "compile-requirements" or (d.get("kind") == "published-ptx" and d.get("host_from"))) \
                    and not self.dry:
                leg_reuse[n]["host_from"] = self.host_owner_origin()
        owner_doc = taken.get(host_owner_leg())
        if owner_doc and owner_doc.get("kind") != "compile-requirements" and built and not self.dry:
            # The host owner is taken whole; the legs built here ship its host/.
            host_owner = self.host_owner_origin()
        if composition:
            # A RETAINED HOST COMPOSITION IN A REUSE ASSEMBLY (0.8.65): linux-wait
            # found the legs' host bytes differ (build-machine-dependent host
            # codegen) and selected the owner leg's. The assembly ships that
            # owner's host/ in every set it assembles, through the same path as
            # a host owner taken whole, and reuse.json names every host file with
            # its source leg; the packer validates the assembled sets through
            # reuse.json. The composition receipt stays the witness of why and
            # is named in the assembled marker; linux-pack does not pass it
            # again, because release_host_composition.validate_assembly binds
            # each set to its original manifest and proof, which a reuse
            # assembly replaces.
            if composition["owner"] != host_owner_leg():
                raise StepFailed("the host bindings are composed from " + composition["owner"]
                                 + ", not the host owner leg " + host_owner_leg())
            if host_owner is None and not self.dry:
                host_owner = self.host_owner_origin()
            self.say("  the retained host composition's owner " + composition["owner"]
                     + " ships its host bindings in every assembled set (receipt " + composition["receipt"] + ")")
        plan = self.plan()
        if self.dry:
            n = len(release_reuse.plan_rows(plan, release_reuse.LINUX, "REUSE"))
            return (f"would take {n} binding(s) and the runtime closure from the published "
                    f"{(plan['previous'] or {}).get('version')} Linux wheel into {self.rel / 'reuse' / 'sets'}"
                    + (f", over the sets of {', '.join(legs)}" if legs else " (no leg ran)")
                    + (f", and the sets of {', '.join(taken)} from an earlier freeze" if taken else "")
                    + (f", with the host bindings of {composition['owner']} (retained host composition)"
                       if composition else ""))
        whl = self.previous_wheel("linux") if self.needs_published_linux() else None
        for name, d in legs.items():
            if not d.is_dir():
                raise StepFailed(f"{name} has no set directory at {d}; run linux-wait")
        dest = self.rel / "reuse" / "sets"
        try:
            release_reuse.assemble_linux(plan, whl, legs, dest, say=self.say, leg_reuse=leg_reuse,
                                         host_owner=host_owner)
        except SystemExit as exc:
            raise StepFailed(str(exc))
        write_json(self.assembled_marker(), dict(
            commit=self.commit, plan_sha256=sha256(self.plan_path), wheel=str(whl) if whl else None,
            wheel_sha256=sha256(whl) if whl else None, legs=sorted(legs), taken=sorted(taken),
            taken_from={n: d["built_from"] for n, d in taken.items()}, at=now(),
            host_composition=(dict(owner=composition["owner"], receipt=composition["receipt"],
                                   receipt_sha256=composition["receipt_sha256"]) if composition else None)))
        return (f"{dest}: sets assembled" + (f" from {whl.name}" if whl else "")
                + (f" and legs {', '.join(sorted(legs))}" if legs else "")
                + (f"; taken from an earlier freeze: {', '.join(sorted(taken))}" if taken else "")
                + (f"; host bindings of {composition['owner']} in every set (retained host composition)"
                   if composition else ""))

    def pack_inputs(self):
        """(set directories, proofs, manifests) for pack_wheel.py and audit.sh:
        the assembled sets when the release reuses anything, else the legs'.
        A leg taken from an earlier freeze gives no proof (its set is named in
        reuse.json with its origin)."""
        taken = self.admitted_legs()
        legs = [l for l in linux_legs(self) if l.name not in taken]
        proofs = [leg.release_build / "build" / "build-provenance.json" for leg in legs]
        # A leg taken by its compile requirements gives no proof either: every
        # file of its set is named in reuse.json (its device bytes by the earlier
        # leg, its host/ by the host owner) and the PTX set's rebound manifest is
        # proved by ptx_reuse.validate_admission inside the packer.
        if self.assembly_needed():
            base = self.rel / "reuse" / "sets"
            sets = sorted({base / vendor for vendor, _ in release_reuse.LINUX_SETS})
            manifests = [base / vendor / arch / "manifest.json" for vendor, arch in release_reuse.LINUX_SETS]
            return sets, proofs, manifests
        composition = self.host_composition()
        if composition:
            # every leg's set with the owner leg's host bytes; the proofs stay the originals
            base = Path(composition["sets"])
            sets = sorted({base / leg.vendor for leg in legs})
            manifests = [base / leg.vendor / leg.arch / "manifest.json" for leg in legs]
            return sets, proofs, manifests
        sets = [leg.release_build / "build" / "sets" / leg.vendor for leg in legs]
        manifests = [leg.release_build / "build" / "sets" / leg.vendor / leg.arch / "manifest.json" for leg in legs]
        return sets, proofs, manifests

    def step_linux_pack(self):
        """--profile release-split: the core and both plugins from the same
        sets and proofs, each through audit.sh (the core first: a plugin's
        audit names the core's runtime from the core beside it) and the
        directory-entry strip, then split_audit over the final set."""
        finals = self.split_finals()
        if all(finals.values()) and wheel_commit(finals["linux"]) == self.commit:
            return "have " + ", ".join(w.name for w in finals.values())
        if self.assembly_needed() and not self.dry and not self.assembled_ok():
            raise StepFailed("the assembled sets are not there or not this plan's; run linux-assemble")
        sets, proofs, manifests = self.pack_inputs()
        src = self.src
        dist = self.rel / "linux"
        if dist.exists() and not self.dry:
            dist.rename(dist.with_name("linux.failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        args = ["pixi", "run", "-e", "pkg", "pack-linux-wheel", "--profile", "release-split"]
        for s_ in sets:
            args += ["--set", s_]
        for p_ in proofs:
            args += ["--build-proof", p_]
        composition = None if self.dry else self.host_composition()
        pack_cwd = src
        if composition and self.assembly_needed():
            # the assembled sets already carry the owner's host bindings, named in
            # each set's reuse.json (linux-assemble); the receipt is not re-validated
            self.say("  the assembled sets carry the host bindings of " + composition["owner"]
                     + " (retained host composition; receipt " + composition["receipt"] + ")")
            composition = None
        if composition:
            # The receipt is validated by this tooling's release_host_composition.py
            # (the frozen source's copy may predate the receipt it reads), so the
            # tooling checkout packs, reading the shipped source from the frozen one.
            args += ["--host-composition", composition["receipt"], "--source-root", src]
            pack_cwd = ROOT
            self.say("  packing with the retained host composition " + composition["receipt"]
                     + " (host bindings from " + composition["owner"] + ")")
        # THE FRAMEWORK ADAPTERS ship with every native set (packaging/adapters/);
        # MOJOLEARN_RELEASE_ADAPTERS=0 releases none, legs and pack alike.
        if os.environ.get("MOJOLEARN_RELEASE_ADAPTERS", "1") != "0":
            args += ["--require-adapters"]
        args += ["--out", dist]
        # the vendor wheel's README says which native set ships unsmoked (no Hopper box held)
        unsmoked = sorted(self.smoke_notes("nvidia"))
        self.must(args, env=dict(MOJOLEARN_RELEASE_UNSMOKED_ARCHES=",".join(unsmoked)) if unsmoked else None,
                  log=self.rel / "linux-pack.log", what="pack_wheel.py", cwd=pack_cwd)
        final_dir = dist / "final"
        if not self.dry:
            final_dir.mkdir(parents=True, exist_ok=True)
        for package, (project, prefix) in SPLIT_PACKAGES.items():
            packed = sorted(dist.glob(f"{prefix}-*.whl")) if not self.dry else [dist / f"<packed {prefix}>.whl"]
            if len(packed) != 1:
                raise StepFailed(f"expected one packed {project} wheel in {dist}, found {packed}")
            self.must(["bash", "packaging/linux/audit.sh", packed[0], *manifests],
                      log=self.rel / f"linux-audit-{package}.log", what=f"audit.sh {project}", cwd=src)
            repaired = sorted((dist / "audit" / "repaired").glob(f"{prefix}-*-manylinux*.whl")) if not self.dry \
                else [dist / "audit" / "repaired" / f"<repaired {prefix}>.whl"]
            if len(repaired) != 1:
                raise StepFailed(f"expected one repaired {project} wheel, found {repaired}")
            self.must([PY, str(ROOT / "tools" / "strip_wheel_dir_entries.py"), repaired[0],
                       final_dir / repaired[0].name, "--receipt", final_dir / f"dir-entry-strip-{package}.json"],
                      what=f"strip_wheel_dir_entries.py {project}")
        if self.dry:
            return "would pack the core, the plugins and the Linux mojolearn, audit and strip each"
        finals = self.split_finals()
        sys.path.insert(0, str(ROOT / "tools"))
        from wheel_api_audit import split_audit
        report = split_audit([w for w in finals.values() if w])
        write_json(final_dir / "split-audit.json", report)
        if report["problems"] or not all(finals.values()):
            raise StepFailed("the final split wheels fail their audit: " + "; ".join(report["problems"] or ["missing"]))
        self.state["linux_layout"] = "split"
        return ", ".join(f"{w.name} sha256 {sha256(w)}" for w in finals.values())

    # ------------------------------------------------------------ GPU columns
    def nvidia_column_ok(self):
        final, out = self.linux_final(), self.rel / "smoke-linux"
        return (bool(final) and smoke_passed(out / "results.json", final)
                and self.plugin_installed(out / "results.json", "nvidia")
                and self.column_arch_ok(out, ("sm_89",)))

    def column_arch_ok(self, out, arches):
        # Actual installed loader selection, not the requested rental model.
        installed = (read_json(out / "results.json") or {}).get("installed") or {}
        return installed.get("gpu_arch") in arches

    def hopper_column_ok(self):
        final, out = self.linux_final(), self.rel / "column-nvidia-hopper"
        return (bool(final) and smoke_passed(out / "results.json", final)
                and self.plugin_installed(out / "results.json", "nvidia")
                and self.column_arch_ok(out, ("sm_90", "sm_90a")))

    def amd_column_ok(self):
        out = self.rel / "column-amd"
        # the amd plugin publishes on its own receipt: the smoke runs on hip too
        final = self.linux_final()
        return (bool(final) and smoke_passed(out / "results.json", final)
                and self.plugin_installed(out / "results.json", "amd")
                and self.column_arch_ok(out, ("gfx942",)))

    def plugin_installed(self, results, package):
        """The receipt installed exactly this release's final plugins beside
        the core, all of them (the box installs what `pip install mojolearn`
        installs), among them `package`'s."""
        plugins = {k: self.split_final(k) for k in PLUGIN_PACKAGES}
        return (all(plugins.values()) and bool(plugins[package])
                and receipt_plugins(results) == {w.name: sha256(w) for w in plugins.values()})

    def column_wheel(self, vendor, name=None):
        """The wheel a column's results are keyed to (ledger, reuse): the
        plugin it installed."""
        return self.split_final("nvidia" if vendor == "cuda" else "amd")

    def column_specs(self):
        return [("nvidia", "cuda", self.nvidia_column_ok, self.rel / "smoke-linux"),
                ("nvidia-hopper", "cuda", self.hopper_column_ok, self.rel / "column-nvidia-hopper"),
                ("amd", "hip", self.amd_column_ok, self.rel / "column-amd")]

    def column_legs(self, names=("nvidia", "nvidia-hopper", "amd")):
        """The two GPU wheel columns as detached legs, each only when its
        record for this wheel is not already there and no PASSED column of a
        byte-identical wheel can be taken. The NVIDIA leg carries its GPU walk
        (SMOKE_WALK or --smoke-gpu; the position survives a rerun in
        <columns>/nvidia.gpu); the AMD leg's provider walk (RunPod, Hot Aisle,
        DigitalOcean) is release_wheel_smoke.sh's own --provider auto."""
        final_path = self.linux_final()
        final = str(final_path or "<final linux wheel>")
        work = self.rel / "columns"
        # Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun.
        # A column is the installed-wheel smoke only (verify against the
        # admitted table). The second identity pass of the changed lanes
        # (--column <selection> --ref-column <Apple column>) is deleted: the
        # table admit already decided NVIDIA == AMD for this source commit.
        legs = []
        for name, vendor, ok, out in self.column_specs():
            if name not in names or ok():
                continue
            if self.smoke_via() == "lq" and name in LQ_COLUMN_BOX:
                leg = ColumnLeg(name, vendor, self.lq_smoke_command(name, vendor, final, out, work),
                                work, out, ok)
            elif self.smoke_via() == "lq" and name == "nvidia-hopper":
                # a Hopper box we hold, reached by ssh; nothing is rented
                leg = ColumnLeg(name, "cuda",
                                ["bash", "tools/release_wheel_smoke.sh", final, "--expected-source-commit", self.commit,
                                 "--out", str(out), "--ssh", self.args.hopper_box,
                                 *self.plugin_args("nvidia")], work, out, ok)
            elif vendor == "cuda":
                arch = "sm_90a" if name == "nvidia-hopper" else "sm_89"
                requested = [g for g in self.args.smoke_gpu.split("|") if g]
                walk = [g for g in requested if g in NVIDIA_WALK[arch]] or NVIDIA_WALK[arch]
                try:
                    at = min(int((work / f"{name}.gpu").read_text()), len(walk) - 1)
                except (OSError, ValueError):
                    at = 0
                leg = ColumnLeg(name, "cuda",
                                ["bash", "tools/release_wheel_smoke.sh", final, "--expected-source-commit", self.commit,
                                 "--out", str(out), "--rent",
                                 *self.plugin_args("nvidia"), "--gpu", walk[at]], work, out, ok)
                leg.walk, leg.at = walk, at
            else:
                leg = ColumnLeg("amd", "hip",
                                ["bash", "tools/release_wheel_smoke.sh", final, "--expected-source-commit",
                                 self.commit, "--out", str(out), "--rent", "--vendor", "hip",
                                 "--provider", self.args.amd_provider, *self.plugin_args("amd")],
                                work, out, ok)
            keyed = self.column_wheel(vendor, name)
            leg.provenance = dict(schema="mojolearn.release-column-provenance.v1", column=name, vendor=vendor,
                                  wheel=str(keyed) if keyed else final,
                                  wheel_sha256=sha256(keyed) if keyed and not self.dry else None,
                                  source_commit=self.commit, tooling_commit=self.tooling_commit)
            leg.provenance["core_sha256"] = sha256(final_path) if final_path and not self.dry else None
            legs.append(leg)
        return legs

    def smoke_via(self):
        """lq (the default: the boxes we hold, through the lane queue) or rent
        (the old per-architecture rented smoke)."""
        return getattr(self.args, "smoke_via", None) or "lq"

    def hopper_required(self):
        """The Hopper column gates publish-nvidia only when it can run: a held
        Hopper box (--hopper-box), or the old rented route."""
        if not self.gpu_smoke():
            return False
        return self.smoke_via() == "rent" or bool(getattr(self.args, "hopper_box", ""))

    def gpu_smoke(self):
        """--gpu-smoke: the old GPU route. The CLI default is False (no GPU, Andrew 2026-10-13); an args
        namespace built without the field (the older unit tests) keeps the GPU route they exercise."""
        return bool(getattr(self.args, "gpu_smoke", True))

    def lq_smoke_command(self, name, vendor, final, out, work):
        """tools/release_lq_smoke.py for one column: the wheels, selection and
        reference columns copied to the box, `lq add --front <box> CMD` of the
        smoke with --local, the result polled, the out directory fetched."""
        box = LQ_COLUMN_BOX[name]
        return [PY, "tools/release_lq_smoke.py", "--box", box, "--tag", f"rel-{self.version}-{name}",
                "--remote-dir", f"/root/release-smoke/{self.version}/{name}", "--out", str(out),
                "--state", str(work / f"{name}.lq.json"),
                "--branch", getattr(self.args, "smoke_branch", None) or "main", "--",
                final, "--expected-source-commit", self.commit, "--vendor", vendor,
                *self.plugin_args("nvidia" if vendor == "cuda" else "amd")]

    def plugin_args(self, package):
        """`--plugin <final plugin>` for the column's smoke, the column's own
        plugin FIRST and then the others: the box installs the core and every
        plugin, as `pip install mojolearn` does."""
        out = []
        for k in (package,) + tuple(k for k in PLUGIN_PACKAGES if k != package):
            plugin = self.split_final(k)
            out += ["--plugin", str(plugin) if plugin else f"<final {SPLIT_PACKAGES[k][1]} wheel>"]
        return out

    NO_GPU_SKIP = "SKIPPED: no GPU smoke (checked before merge; --gpu-smoke runs it); nothing rented, no lq"

    def step_gpu_column_nvidia(self):
        """The NVIDIA column alone (core + both plugins, as pip installs them); it gates mojolearn-nvidia."""
        if not self.gpu_smoke():
            return self.NO_GPU_SKIP
        self.run_columns(("nvidia",))
        return "NVIDIA wheel smoke PASSED against the admitted reference table"

    def step_gpu_column_nvidia_hopper(self):
        if not self.gpu_smoke():
            return self.NO_GPU_SKIP
        if not self.hopper_required():
            self.say("  nvidia-hopper: no Hopper box held (--hopper-box <ssh> runs it); nothing is rented")
            return ("SKIPPED: no Hopper box held; the sm_90a payload publishes on its GitHub build receipt, "
                    "smoke " + HOPPER_NOT_RUN)
        self.run_columns(("nvidia-hopper",))
        return "NVIDIA Hopper column PASSED"

    def step_gpu_column_amd(self):
        """The AMD column alone (core + both plugins, with the smoke); it gates mojolearn-amd."""
        if not self.gpu_smoke():
            return self.NO_GPU_SKIP
        self.run_columns(("amd",))
        return "AMD wheel smoke PASSED against the admitted reference table"

    def run_columns(self, names):
        """Launch (or take) the named columns at once, wait for every one, and
        raise StepFailed naming each that did not PASS."""
        final = self.linux_final()
        if not final and not self.dry:
            raise StepFailed("no final Linux wheel; run linux-pack")
        legs = self.column_legs(names)
        for name in names:
            if not any(l.name == name for l in legs):
                self.say(f"  {name}: PASSED for this wheel (record exists), not rerun")
        for leg in legs:
            self.say(f"  {leg.name}: {leg.describe().strip()}")
        if self.dry:
            if legs:
                self.say(f"  would launch {', '.join(l.name for l in legs)} at once and wait for every one")
            return None
        if legs:
            for leg in legs:
                self.refuse_device_lanes_after_publish("gpu-column-" + leg.name)
            launch_detached(self, legs)
            self.wait_columns(legs)
        failed = []
        for name, vendor, ok, out in self.column_specs():
            label = name
            if name not in names:
                continue
            if ok():
                self.say(f"  {label} column: wheel smoke PASSED against the admitted reference table")
                continue
            leg = next((l for l in legs if l.name == name), None)
            failed.append(f"{label} column missing or not PASSED: {out}"
                          + (f" (exit {leg.exit_code()}, log {leg.log})" if leg else ""))
            self.say(f"  {failed[-1]}")
        if failed:
            raise StepFailed("; ".join(failed) + ". Rerun `release` to relaunch the failed column(s).")
        return True

    def wait_columns(self, legs):
        """Wait for every column leg; walk the NVIDIA leg to its next GPU type
        when RunPod had no stock, and wait again."""
        work = legs[0].workdir
        # the walk runs AT ONCE for a column out of stock, while the others run
        while True:
            again = []
            for l in legs:
                if l.running() or l.exit_code() is None:
                    continue
                walk = getattr(l, "walk", None)
                if (walk and not l.done(self.commit) and l.exit_code() != 0
                        and no_stock(l.log, l.out_dir) and l.at + 1 < len(walk)):
                    l.at += 1
                    self.say(f"  {l.name}: {walk[l.at - 1]} has no stock; trying {walk[l.at]}")
                    l.command[l.command.index("--gpu") + 1] = walk[l.at]
                    work.mkdir(parents=True, exist_ok=True)
                    (work / f"{l.name}.gpu").write_text(str(l.at))
                    again.append(l)
            if again:
                launch_detached(self, again)
                continue
            if not any(l.running() for l in legs):
                return
            self.say("  waiting: " + ", ".join(f"{l.name}={'running' if l.running() else l.exit_code()}"
                                                for l in legs))
            self.sleep(60)

    # ------------------------------------------------------------ publication
    def on_pypi(self, wheel):
        """True when PyPI already serves this exact file (name and sha256), in
        the project the wheel's name says (mojolearn, mojolearn-nvidia, mojolearn-amd)."""
        project = wheel.name.split("-")[0].replace("_", "-")
        url = f"https://pypi.org/pypi/{project}/{self.version}/json"
        try:
            with urllib.request.urlopen(url, timeout=20) as r:
                files = json.load(r).get("urls", [])
        except Exception:
            return False
        return any(f.get("filename") == wheel.name and f.get("digests", {}).get("sha256") == sha256(wheel)
                   for f in files)

    def published_marker(self, wheel):
        """gpu_plugin.json of the file PyPI already serves under this wheel's name, else None."""
        project = wheel.name.split("-")[0].replace("_", "-")
        try:
            with urllib.request.urlopen(f"https://pypi.org/pypi/{project}/{self.version}/json", timeout=20) as r:
                url = next((f["url"] for f in json.load(r).get("urls", []) if f.get("filename") == wheel.name), None)
            if url is None:
                return None
            local = self.rel / "published" / wheel.name
            local.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(url, local)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # this version is not on PyPI yet: nothing published to reuse
                return None
            raise StepFailed(f"cannot read the published {wheel.name}: {exc}") from exc
        except Exception as exc:
            raise StepFailed(f"cannot read the published {wheel.name}: {exc}") from exc
        return wheel_marker(local)

    def publish(self, platform, wheel, smoke):
        """Publish ONE platform's wheel, built from the frozen source. The
        publisher runs from the tooling checkout (its tag names the tooling;
        MOJOLEARN_ARTIFACT_SOURCE_COMMIT pins the wheel's source witness).
        A split plugin carries no source witness: the core it was packed and
        smoked with does, and the receipt names the plugin by sha256."""
        target = self.args.publish
        if target is None:
            raise StepHeld("publication needs an explicit --publish none|testpypi|pypi; stopping here")
        # testpypi-rehearsal uploads through this same path with the target set to TestPyPI for its thread
        rehearsing = getattr(self._tl, "publish_target", None) == "testpypi"
        if rehearsing:
            target = "testpypi"
        witness = self.linux_final() if platform in SPLIT_PACKAGES and platform != "linux" else wheel
        # --identity-on-wheels: the core and macOS wheels ship repacked with the admitted table (data only)
        wheel = self.shipped_wheel(platform, wheel) if not self.dry else wheel
        if not self.dry and wheel and witness and wheel_commit(witness) != self.commit:
            raise StepFailed(f"the {platform} wheel {wheel.name} records source {wheel_commit(witness)}, not the frozen "
                             f"{self.commit}")
        # A FILE ALREADY ON PYPI IS DONE (lane release-no-hands): checked before the pre-publication gate,
        # which only decides whether to upload; it never refuses what the index already serves.
        if not rehearsing and not self.dry and wheel and self.on_pypi(wheel):
            return "already on PyPI"
        if not rehearsing and not self.dry and wheel and platform in SPLIT_PACKAGES and platform != "linux":
            published = self.published_marker(wheel)
            # 0.8.37 (2026-10-10): a refreeze after a plugin published rebuilds that plugin with new bytes
            # (LINUX_PAYLOAD.json names the source commit), and PyPI never takes a filename twice. The loader
            # admits a plugin on version, arch directories and gpu_plugin.json only (_backend.py), so the
            # published plugin serves the new core exactly when its marker equals the new build's.
            if published is not None:
                if published != wheel_marker(wheel):
                    raise StepFailed(f"PyPI already has {wheel.name} with a different gpu_plugin.json; "
                                     "a new version is needed")
                return "already on PyPI (published from an earlier freeze; gpu_plugin.json equal)"
        # NOTHING REACHES PyPI UNREHEARSED (Andrew 2026-11-06): the TestPyPI rehearsal passed for these exact bytes
        if target == "pypi" and not rehearsing and not self.dry and wheel:
            covered = ((self.recorded(REHEARSAL_STEP) or {}).get("wheels") or {}).get(platform)
            if covered != sha256(wheel):
                raise StepFailed(f"the TestPyPI rehearsal has not passed for {wheel.name} (sha256 {sha256(wheel)[:12]}); "
                                 f"run {REHEARSAL_STEP} first; nothing went to PyPI")
        if not self.dry:
            held = self.plan_blocks_publication()
            if held:
                raise StepFailed("the release plan names evidence that is still missing: " + "; ".join(held[:8]))
            ready, reasons = self.release_gate(stage="pre")
            if not ready:
                raise StepFailed(("publication requires both candidate gates: " if reasons[0].startswith("candidate gate")
                                  else "release requirements ledger blocks publication: ") + "; ".join(reasons))
        else:
            try:  # a dry run reports the verdict; it never writes the ledger
                ready, reasons = self.release_gate(stage="pre")
                self.say(f"  pre-publication gate ({platform}): " + ("READY" if ready else
                         f"would block on {len(reasons)} row(s): " + "; ".join(r[:160] for r in reasons[:5])))
            except (ValueError, OSError, KeyError, subprocess.CalledProcessError, StepFailed) as error:
                self.say(f"  pre-publication gate ({platform}): not evaluated ({error})")
        if not self.dry and self.gpu_smoke() and platform in PLUGIN_PACKAGES \
                and receipt_plugins(smoke).get(wheel.name) != sha256(wheel):
            raise StepFailed(f"the {platform} smoke receipt {smoke} did not install {wheel.name}")
        day = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d")
        tag = f"alpha-api-{self.version}-{platform}-{day}"
        work = self.rel / (f"{REHEARSAL_STEP}/publish-{platform}" if rehearsing else f"publish-{platform}")
        env = dict(MOJOLEARN_ARTIFACT_SOURCE_COMMIT=self.commit,
                   MOJOLEARN_REQUIREMENTS_LEDGER=str(self.readiness().path))
        if getattr(self.args, "owner_waiver", None):
            env["MOJOLEARN_OWNER_WAIVER"] = str(Path(self.args.owner_waiver).resolve())
        if self.single() and (self.admitted() or {}).get("repacked"):
            # the publish admission accepts the data-only repack of a qualified wheel by its receipt
            env["MOJOLEARN_REPACK_RECEIPTS"] = ":".join(sorted(self.admitted()["repacked"].values()))
        notes = self.smoke_notes(platform)
        if notes:
            env["MOJOLEARN_ALPHA_SMOKE_NOTES"] = json.dumps(notes, sort_keys=True)
        # without --gpu-smoke: the checklist's preverified route (qualification_reuse, no receipt attached)
        # (the Linux mojolearn has no receipt of its own: it carries no files)
        route = ["--light-smoke", smoke] if self.gpu_smoke() and smoke is not None else ["--preverified"]
        self.must(["bash", "tools/release_linux_publish.sh", wheel or f"<{platform} wheel>", tag, target, work,
                   *route], env=env,
                  log=work.parent / f"publish-{platform}.log", what=f"publish {platform} to {target}")
        return f"{target} via {tag} (source {self.commit[:12]}, tooling {self.tooling_commit[:12]})"

    # ------------------------------------------------------------ testpypi-rehearsal (before the first publish)
    #: platform -> its publish step; the order TestPyPI takes them in (the core first, the vendors, the Linux
    #: mojolearn once all three are there, the macOS wheel), as the real publish does
    REHEARSAL_ORDER = ("linux", "nvidia", "amd", "linux-meta", "macos")
    #: pip resolution attempts while TestPyPI's simple pages catch up with an upload (30 s apart)
    REHEARSAL_ATTEMPTS = 6

    def platform_publish(self):
        return {p["platform"]: p["publish"] for p in self.PIPELINES.values()}

    def rehearsal_ready(self):
        """The platforms whose publish step could run now except for the rehearsal: every check it needs done,
        and every package it requires on the index ready too (in REHEARSAL_ORDER)."""
        by_step = {step: platform for platform, step in self.platform_publish().items()}
        ready = []
        for platform in self.REHEARSAL_ORDER:
            step = self.platform_publish().get(platform)
            if step is None:
                continue
            needs = self.NEEDS.get(step, [])
            own = [d for d in needs if d not in PUBLISH_STEPS and d != REHEARSAL_STEP]
            deps = [by_step[d] for d in needs if d in PUBLISH_STEPS]
            if all(self.recorded(d) for d in own) and all(d in ready for d in deps):
                ready.append(platform)
        return ready

    def rehearsal_wheel(self, platform):
        """The bytes the publish of PLATFORM ships (the gated wheel; under --identity-on-wheels its data-only repack)."""
        wheel = self.macos_wheel() if platform == "macos" else self.split_final(platform)
        return self.shipped_wheel(platform, wheel) if wheel else None

    @staticmethod
    def rehearsal_requires(platform):
        """The other platforms whose package PLATFORM's package requires on the index."""
        return tuple(d for d in LINUX_RESOLVES.get(platform, ()) if d != platform) if platform != "macos" else ()

    def rehearsal_upload(self, platform, wheel):
        """Put PLATFORM's gated wheel on TestPyPI through the release's own publisher, unless TestPyPI already
        holds exactly these bytes. Other bytes under the same file name FAIL: TestPyPI never takes a name twice."""
        import release_testpypi_rehearsal as TR
        project, digest = TR.wheel_project(wheel), sha256(wheel)
        try:
            state, served = TR.testpypi_file_state(project, self.version, wheel.name, digest)
        except Exception as error:   # noqa: BLE001 - any read failure stops before PyPI
            raise StepFailed(f"cannot read TestPyPI for {wheel.name}: {error}") from error
        if state == "same":
            return f"{wheel.name}: already on TestPyPI with the gated sha256 (not uploaded again)"
        if state == "different":
            raise StepFailed(TR.different_bytes_reason(wheel.name, served, digest, self.version))
        step = self.platform_publish()[platform]
        self._tl.publish_target = "testpypi"
        try:
            out = getattr(self, "step_" + step.replace("-", "_"))()
        finally:
            self._tl.publish_target = None
        result = out[0] if isinstance(out, tuple) else out
        for attempt in range(1, self.REHEARSAL_ATTEMPTS + 1):
            state, served = TR.testpypi_file_state(project, self.version, wheel.name, digest)
            if state == "same":
                return f"{wheel.name}: uploaded to TestPyPI ({result})"
            if state == "different":
                raise StepFailed(TR.different_bytes_reason(wheel.name, served, digest, self.version))
            self.say(f"  TestPyPI does not list {wheel.name} yet (attempt {attempt}/{self.REHEARSAL_ATTEMPTS})")
            self.sleep(30)
        raise StepFailed(f"{wheel.name} was dispatched to TestPyPI ({result}) but TestPyPI does not serve it")

    def rehearsal_checks(self, ready, wheels, local, out, index):
        """(problems, rows): (a) every package set resolved through INDEX (our names from TestPyPI only) for each
        platform tag and Python version the wheels declare, against the expected set and the gated sha256;
        (b) each rehearsed file downloaded and hashed; (c) the macOS wheel installed into a fresh venv and
        imported. LOCAL: platforms whose upload was skipped (their gated file is offered from out/local)."""
        import concurrent.futures
        import release_testpypi_rehearsal as TR
        pip_venv = out / "pip-venv"
        if pip_venv.exists():
            shutil.rmtree(pip_venv)
        self.must([self.smoke_python(), "-m", "venv", pip_venv], what="the rehearsal's pip venv")
        pip_python = pip_venv / "bin" / "python"
        links = None
        if local:
            links = out / "local"
            links.mkdir(parents=True, exist_ok=True)
            for platform in local:
                shutil.copy2(wheels[platform], links / wheels[platform].name)
        local_files = {wheels[p].name for p in local}
        expect = {p: (wheels[p].name, sha256(wheels[p])) for p in ready}
        # (a) the package sets: root project, the platforms it installs, its own wheel (extras, tags, Pythons)
        sets = [(SPLIT_PACKAGES[k][0], LINUX_RESOLVES[k], wheels[k]) for k in LINUX_RESOLVES
                if k in ready and all(d in ready for d in LINUX_RESOLVES[k])]
        if "macos" in ready:
            sets.append((TR.wheel_project(wheels["macos"]), ("macos",), wheels["macos"]))
        jobs = []
        for project, members, own in sets:
            meta = TR.wheel_metadata(own)
            want = {TR.wheel_project(wheels[m]): expect[m] for m in members}
            for root in [project] + [f"{project}[{e}]" for e in meta["extras"]]:
                for python in TR.python_versions(meta):
                    jobs.append((root, TR.wheel_platform(own), python, want))
        rows, problems = [], []

        def resolve(job):
            root, plat, python, want = job
            slug = re.sub(r"[^A-Za-z0-9.]+", "_", f"{root}-{plat}-py{python}")
            report = out / "resolve" / f"{slug}.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            for attempt in range(1, self.REHEARSAL_ATTEMPTS + 1):
                if report.exists():
                    report.unlink()
                rc = self.run(TR.resolve_command(pip_python, index.url, f"{root}=={self.version}", plat, python,
                                                 report, links), log=out / "resolve" / f"{slug}.log")
                if rc == 0:
                    break
                self.sleep(30)
            label = f"pip install {root}=={self.version} on {plat} py{python}"
            if rc != 0:
                return label, [f"pip could not resolve it (exit {rc}; {out / 'resolve' / (slug + '.log')})"]
            return label, TR.judge_resolution(read_json(report), want, self.version, index.served, local_files)

        def download(platform):
            wheel = wheels[platform]
            dest = out / "bytes" / platform
            if dest.exists():
                shutil.rmtree(dest)
            for attempt in range(1, self.REHEARSAL_ATTEMPTS + 1):
                rc = self.run(TR.download_command(pip_python, index.url, TR.wheel_project(wheel), self.version,
                                                  TR.wheel_platform(wheel), dest), log=out / "bytes" / f"{platform}.log")
                if rc == 0:
                    break
                self.sleep(30)
            got = sorted(dest.glob("*.whl")) if dest.exists() else []
            label = f"TestPyPI bytes of {wheel.name}"
            if rc != 0 or len(got) != 1:
                return label, [f"pip download did not fetch exactly one file (exit {rc}, {[g.name for g in got]})"]
            if got[0].name != wheel.name or sha256(got[0]) != sha256(wheel):
                return label, [f"TestPyPI served {got[0].name} sha256 {sha256(got[0])}, gated {sha256(wheel)}"]
            return label, []

        # every resolution and download is independent: all at once, one worker per core
        units = [(resolve, job) for job in jobs] + [(download, p) for p in ready if p not in local]
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, min(len(units), os.cpu_count() or 1))) as pool:
            done = list(pool.map(lambda unit: unit[0](unit[1]), units))
        for label, found in done:
            rows.append(dict(check=label, problems=found))
            problems += [f"{label}: {p}" for p in found]
        # (c) the macOS wheel from TestPyPI, pinned to the gated sha256, into a fresh venv; import only
        if "macos" in ready:
            wheel = wheels["macos"]
            venv = out / "macos-venv"
            if venv.exists():
                shutil.rmtree(venv)
            req = out / "macos-requirements.txt"
            req.write_text(TR.requirements_text(TR.wheel_project(wheel), self.version, sha256(wheel)))
            self.must([self.smoke_python(), "-m", "venv", venv], what="the rehearsal's macOS venv")
            label = f"pip install {wheel.name} from TestPyPI into a fresh venv and import"
            install = TR.install_command(venv, index.url, req) + (["--find-links", str(links)] if links else [])
            found = []
            for attempt in range(1, self.REHEARSAL_ATTEMPTS + 1):
                rc = self.run(install, log=out / "macos-install.log")
                if rc == 0:
                    break
                self.sleep(30)
            if rc != 0:
                found.append(f"pip install exited {rc} ({out / 'macos-install.log'})")
            else:
                rc = self.run(TR.import_command(venv, self.version), log=out / "macos-import.log", cwd=out)
                if rc != 0:
                    found.append(f"import mojolearn exited {rc} ({out / 'macos-import.log'})")
            rows.append(dict(check=label, problems=found))
            problems += [f"{label}: {p}" for p in found]
        return problems, rows

    def have_testpypi_rehearsal(self, rec):
        """Done when the recorded rehearsal covered every platform ready now, with the bytes they ship now."""
        if self.args.publish != "pypi" or "wheels" not in rec:
            return None
        ready = self.rehearsal_ready()
        wheels = rec.get("wheels") or {}
        if not ready or any(p not in wheels or wheels[p] != sha256(self.rehearsal_wheel(p)) for p in ready):
            return None
        return "have the TestPyPI rehearsal of " + ", ".join(ready)

    def step_testpypi_rehearsal(self):
        """THE TESTPYPI REHEARSAL (Andrew 2026-11-06), before the first real publish: the gated wheels uploaded
        to TestPyPI, every package set resolved there (our names from TestPyPI only), every file's sha256 equal to
        the gated wheel's, the macOS wheel installed and imported on this Mac. Any failure stops here: the real
        PyPI has seen nothing (tools/release_testpypi_rehearsal.py)."""
        import release_testpypi_rehearsal as TR
        if self.args.publish != "pypi":
            return f"skipped (--publish {self.args.publish}: the TestPyPI rehearsal runs inside --publish pypi)"
        ready = self.rehearsal_ready()
        if self.dry:
            self.say("  would upload to TestPyPI (same bytes already there: not again), resolve every package set "
                     "with our names from TestPyPI only, compare every sha256 and install + import the macOS wheel: "
                     + (", ".join(ready) or "nothing ready yet"))
            return "pending: would rehearse on TestPyPI"
        if not ready:
            return "nothing is ready to publish", dict(platforms=[], wheels={})
        wheels = {p: self.rehearsal_wheel(p) for p in ready}
        missing = [p for p, w in wheels.items() if w is None]
        if missing:
            raise StepFailed("TestPyPI rehearsal: no gated wheel for " + ", ".join(missing))
        out = self.rel / REHEARSAL_STEP
        out.mkdir(parents=True, exist_ok=True)
        uploads, owed, local = {}, [], []
        for platform in ready:
            wheel = wheels[platform]
            project = TR.wheel_project(wheel)
            limit = TR.testpypi_file_limit(project)
            blocked = [d for d in self.rehearsal_requires(platform) if d in local]
            if wheel.stat().st_size > limit:
                owed.append(f"{wheel.name} is {wheel.stat().st_size} bytes, over TestPyPI's {limit}-byte limit for "
                            f"{project}: OWED, ask TestPyPI for the limit PyPI granted (pypi/support, naming "
                            f"test.pypi.org); not uploaded, checked from the local gated file")
                local.append(platform)
            elif blocked:
                owed.append(f"{wheel.name} not uploaded: it requires {', '.join(blocked)} on the index, which TestPyPI "
                            f"could not take; checked from the local gated file")
                local.append(platform)
            else:
                uploads[platform] = self.rehearsal_upload(platform, wheel)
                self.say("  " + uploads[platform])
        for line in owed:
            self.say("  SKIPPED UPLOAD: " + line)
        with TR.RoutingIndex() as index:
            problems, rows = self.rehearsal_checks(ready, wheels, local, out, index)
            routed = sorted(set(index.requests))
        write_json(out / "rehearsal.json", dict(
            schema="mojolearn.release-testpypi-rehearsal.v1", version=self.version, commit=self.commit,
            platforms=ready, wheels={p: dict(file=w.name, sha256=sha256(w)) for p, w in wheels.items()},
            uploads=uploads, owed=owed, local=local, checks=rows, routed=routed, problems=problems))
        if problems:
            raise StepFailed(f"TestPyPI rehearsal: {problems[0]}"
                             + (f" (+{len(problems) - 1} more)" if len(problems) > 1 else "")
                             + f"; nothing went to PyPI ({out / 'rehearsal.json'})")
        checked = sum(1 for r in rows if r["check"].startswith("pip install ") and " on " in r["check"])
        return (f"TestPyPI rehearsal PASSED for {', '.join(ready)}: {checked} resolutions, "
                f"{len(ready) - len(local)} files' sha256, macOS install+import "
                + ("OK" if "macos" in ready else "not ready") + (f"; OWED: {'; '.join(owed)}" if owed else ""),
                dict(platforms=ready, wheels={p: sha256(w) for p, w in wheels.items()}, owed=owed, local=local))

    def smoke_notes(self, platform):
        """{native set: smoke line} for the alpha manifest of a vendor wheel
        whose set had no installed smoke (sm_90a without a held Hopper box)."""
        if platform == "nvidia" and self.gpu_smoke() and not self.hopper_required() and "sm_90a" in GPU_PACKAGES["PLUGINS"]["cuda"]["arches"]:
            return {"sm_90a": HOPPER_NOT_RUN}
        return {}

    def core_receipt(self):
        """The receipt mojolearn-core publishes on: a PASSED column that
        installed it (NVIDIA's, else AMD's). The core requires no vendor
        (2026-10-20), so one passing vendor is enough; a failed vendor holds
        only its own package and the Linux mojolearn."""
        if self.dry or not self.gpu_smoke():
            return self.rel / "smoke-linux" / "results.json"
        if self.nvidia_column_ok():
            return self.rel / "smoke-linux" / "results.json"
        if self.amd_column_ok():
            return self.rel / "column-amd" / "results.json"
        raise StepFailed("mojolearn-core publishes on a PASSED GPU column; neither NVIDIA nor AMD PASSED")

    def step_publish_meta_linux(self):
        """The file-free Linux `mojolearn`: no smoke installs it (it carries no
        files); it publishes after the core and both vendors (STEP_TABLE)."""
        return self.publish("linux-meta", self.split_final("linux-meta"), None), dict(smoke=None)

    def step_publish_core_linux(self):
        smoke = self.core_receipt()
        if not self.gpu_smoke():
            return self.publish("linux", self.split_final("linux"), None), dict(smoke=None)
        return self.publish("linux", self.split_final("linux"), smoke), dict(smoke=str(smoke))

    def step_publish_nvidia(self):
        if not self.gpu_smoke():
            return self.publish("nvidia", self.split_final("nvidia"), None), dict(smoke=None)
        if not self.dry and not self.nvidia_column_ok():
            raise StepFailed("NVIDIA vendor wheel requires the Ada column")
        if not self.dry and self.hopper_required() and not self.hopper_column_ok():
            raise StepFailed("NVIDIA vendor wheel requires the Hopper column (a Hopper box was held)")
        smoke = self.rel / "smoke-linux" / "results.json"
        return self.publish("nvidia", self.split_final("nvidia"), smoke), dict(smoke=str(smoke))

    def step_publish_amd(self):
        if not self.gpu_smoke():
            return self.publish("amd", self.split_final("amd"), None), dict(smoke=None)
        smoke = self.rel / "column-amd" / "results.json"
        return self.publish("amd", self.split_final("amd"), smoke), dict(smoke=str(smoke))

    # ------------------------------------------------------------ the Apple adapter gate
    def apple_adapters_out(self):
        return self.rel / "gate-apple" / "adapters"

    def apple_adapters_ok(self):
        """(ok, why): the installed-wheel adapter smoke PASSED for these macOS wheel bytes."""
        AG = adapter_gate_module()
        w, out = self.macos_wheel(), self.apple_adapters_out()
        reuse = read_json(out / "reuse.json")
        if reuse and w:
            from release_incremental_gate import apple_adapters
            try:
                apple_adapters(reuse, sha256(w))
                return True, 'PASSED: original installed adapters retained with unchanged runtime bytes; changed host qualified'
            except (ValueError, OSError, KeyError, TypeError) as error:
                return False, 'Apple adapter reuse refused: ' + str(error)
        stamp = read_json(out / "wheel.json") or {}
        if not w or stamp.get("wheel_sha256") != sha256(w):
            return False, f"no adapter smoke receipt for this macOS wheel ({out})"
        _, ok, why = AG.judge(out / AG.RECEIPT)
        return ok, why

    #: the Apple installed-runtime qualifier's outer cap (seconds); each unit has its own budget inside it
    APPLE_RUNTIME_SECONDS = int(os.environ.get("MOJOLEARN_APPLE_RUNTIME_SECONDS", "14400"))

    def apple_runtime_out(self):
        return self.rel / "gate-apple" / "runtime-apple"

    def apple_runtime_store(self):
        """The Apple unit store: appended as each unit ends, so a rerun of gate-apple resumes."""
        return self.rel / "gate-apple" / "runtime-units-apple"

    @staticmethod
    def apple_device_class():
        """apple-<chip> (e.g. apple-m2-pro): unit receipts are reused on the same chip only."""
        try:
            brand = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                   text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            brand = ""
        chip = re.sub(r"[^a-z0-9]+", "-", brand.lower().removeprefix("apple ")).strip("-")
        return "apple-" + (chip or platform.machine())

    def apple_runtime_ok(self):
        """(ok, why): runtime-apple passed for these macOS wheel bytes."""
        w = self.macos_wheel()
        error, cases = self.gate_runtime_error("apple", self.rel / "gate-apple")
        if error:
            return False, error
        report = read_json(self.apple_runtime_out() / "runtime-report.json") or {}
        if not w or (report.get("wheels") or {}).get(Path(w).name) != sha256(w):
            return False, f"runtime-apple ran other wheel bytes ({self.apple_runtime_out()})"
        units = report.get("units") or {}
        return True, (f"runtime-apple PASSED: {cases} installed-runtime cases "
                      f"(units reused={units.get('reused')} run={units.get('run')})")

    def apple_runtime_gate(self):
        """runtime-apple (tools/release_runtime_installed.py run --vendor apple): the public runtime registry
        against the installed macOS wheel, per unit, reusing unchanged units' receipts on this chip."""
        ok, why = self.apple_runtime_ok()
        if ok:
            return why + " (receipt exists)"
        w = self.macos_wheel()
        if not w and not self.dry:
            raise StepFailed("no macOS wheel; run macos-build")
        out = self.apple_runtime_out()
        if out.exists() and not self.dry:
            out.rename(out.with_name(out.name + ".failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        cmd = [PY, str(ROOT / "tools" / "mac_slot.py"), "--wait-timeout", "3600", "metal", "--",
               PY, str(ROOT / "tools" / "release_runtime_installed.py"), "run", "--vendor", "apple",
               "--source-root", str(self.source_checkout(create=not self.dry)), "--source-commit", self.commit,
               "--wheel", str(w or "<macos wheel>"),
               "--python", self.smoke_python() if not self.dry else "python3.12",
               "--device-class", self.apple_device_class(), "--unit-store", str(self.apple_runtime_store()),
               "--timeout", str(self.APPLE_RUNTIME_SECONDS), "--out", str(out)]
        if self.single() and not self.reuse():
            cmd.append("--redo")   # --identity-on-wheels runs every unit unless --reuse-unchanged opted in
        for directory in self.runtime_unit_sources("gate-apple"):
            if directory != self.apple_runtime_store():
                cmd += ["--prior-units", str(directory)]
        if not self.dry:
            out.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.must(cmd, log=self.rel / "gate-apple-runtime.log", what="the Apple installed-runtime qualifier")
        except StepFailed:
            pass  # the report under runtime-apple/ names the failed units; apple_runtime_ok reads it below
        if self.dry:
            return None
        ok, why = self.apple_runtime_ok()
        if not ok:
            raise StepFailed(f"the Apple installed-runtime qualifier did not pass: {why}")
        return why

    def step_gate_apple(self):
        """The Apple adapter smoke (when adapters ship) and runtime-apple, the installed-runtime qualifier; with
        --identity-on-wheels first the Apple identity column from the installed macOS wheel (one pass)."""
        identity = self.apple_identity_gate()
        # --identity-on-wheels: the adapter smoke and runtime-apple under the unit rule; an unchanged Apple column
        # runs nothing on this Mac (release_identity_wheels.apple_units_gate)
        adapters = self.apple_adapter_unit_gate() if self.single() else self.apple_adapter_gate()
        runtime = self.apple_runtime_reused() if self.single() else None
        runtime = runtime or self.apple_runtime_gate()
        return "; ".join(str(part) for part in (identity, adapters, runtime) if part) or None

    def apple_adapter_gate(self):
        AG = adapter_gate_module()
        if not AG.enabled():
            return "adapters skipped (MOJOLEARN_RELEASE_ADAPTERS=0: no adapters ship)"
        ok, why = self.apple_adapters_ok()
        if ok:
            return why + " (receipt exists)"
        w = self.macos_wheel()
        if not w and not self.dry:
            raise StepFailed("no macOS wheel; run macos-build")
        out = self.apple_adapters_out()
        if out.exists() and not self.dry:
            out.rename(out.with_name(out.name + ".failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        venv = out / "venv"
        base = self.smoke_python() if not self.dry else "python3.12"
        for k, cmd in enumerate(AG.venv_commands(base, venv, [w or "<macos wheel>"])):
            self.must(cmd, log=out / f"adapter-venv-{k}.log", what="the Apple adapter smoke venv")
        if not self.dry:
            write_json(out / "wheel.json", dict(wheel=Path(w).name, wheel_sha256=sha256(w)))
        self.must([PY, str(ROOT / "tools" / "mac_slot.py"), "--wait-timeout", "3600", "metal", "--",
                   *AG.smoke_command("apple", venv, out, runner_python=PY)],
                  log=self.rel / "gate-apple.log", what="the Apple adapter smoke")
        if self.dry:
            return None
        ok, why = self.apple_adapters_ok()
        if not ok:
            raise StepFailed(f"the Apple adapter smoke did not pass: {why} ({out / AG.RECEIPT})")
        return why

    def step_publish_macos(self):
        AG = adapter_gate_module()
        if AG.enabled() and not self.dry and not self.apple_adapters_ok()[0] and not (
                self.single() and self.apple_adapter_unit_ok()):
            raise StepFailed(f"no PASSED adapter smoke for this macOS wheel ({self.apple_adapters_ok()[1]}); "
                             "run gate-apple")
        # Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun.
        # The bundled-table gate is macos-smoke's own self-test job (same wheel,
        # same `verify --self-test`); its PASSED receipt for these bytes is
        # required even when publish-macos is run alone. The Apple-vs-previous
        # release cross-diff is deleted: identity is decided by the table admit.
        if self.single():
            # the Apple column recorded from this wheel (or every lane reused) replaces the self-test receipt
            if not self.dry and not (self.apple_identity_ok()[0] or (self.reuse() and self.lanes_owed("apple") == [])):
                raise StepFailed("no Apple column for this macOS wheel (%s); run gate-apple" % self.apple_identity_ok()[1])
            return self.publish("macos", self.macos_wheel(), None)
        if not self.gpu_smoke():
            if not self.dry and not self.macos_self_test_passed():
                raise StepFailed(f"no PASSED macOS wheel self-test for this wheel ({self.macos_self_test()}); "
                                 "run macos-smoke")
            return (self.publish("macos", self.macos_wheel(), None),
                    dict(smoke=str(self.macos_self_test())))
        smoke = self.rel / "smoke-macos" / "results.json"
        if not self.dry and not smoke_passed(smoke, self.macos_wheel()):
            raise StepFailed(f"no PASSED macOS smoke receipt for this wheel ({smoke}); run macos-smoke")
        return self.publish("macos", self.macos_wheel(), self.rel / "smoke-macos" / "results.json")

    def finish_bytes(self, platform):
        """THE FINISH LINE, HASH ONLY (lane release-testpypi-rehearsal, Andrew 2026-11-06): what the real PyPI
        serves for PLATFORM is the shipped bytes. `pip download --no-deps` from https://pypi.org/simple for the
        wheel's own platform tag, sha256 compared; nothing is installed or run. The install, the import and the
        package-set resolution ran before the first publish, in testpypi-rehearsal, against the same bytes."""
        import release_testpypi_rehearsal as TR
        final = self.rehearsal_wheel(platform) if not self.dry else None
        project = SPLIT_PACKAGES[platform][0] if platform in SPLIT_PACKAGES else "mojolearn"
        plat = TR.wheel_platform(final) if final else ("macosx_11_0_arm64" if platform == "macos"
                                                       else "manylinux_2_35_x86_64")
        dest = self.rel / ("finish-linux" if platform == "linux" else f"finish-{platform}")
        for attempt in range(1, 7):
            if dest.exists() and not self.dry:
                shutil.rmtree(dest)
            rc = self.run(TR.pip_base(self.smoke_python() if not self.dry else "python3.12") + [
                "download", "--no-deps", "--no-cache-dir", "--only-binary=:all:", "--platform", plat,
                "--implementation", "cp", "--index-url", TR.PYPI_SIMPLE, "--dest", dest,
                f"{project}=={self.version}"], log=self.rel / f"finish-{platform}-{attempt}.log")
            got = sorted(dest.glob("*.whl")) if not self.dry and dest.exists() else []
            if self.dry or (rc == 0 and final and len(got) == 1 and got[0].name == final.name
                            and sha256(got[0]) == sha256(final)):
                return f"{project}: PyPI serves {final.name if final else project} with the shipped sha256"
            self.say(f"  pip download failed or differs (attempt {attempt}/6; the index may still be propagating)")
            self.sleep(30)
        raise StepFailed(f"PyPI does not serve {project}=={self.version} for {plat} with the shipped sha256 "
                         "(six attempts)")

    def step_finish_line(self):
        """Hash-only: every published file on the real PyPI is the shipped bytes. Nothing after the first publish
        installs or runs anything; testpypi-rehearsal installed, imported and resolved before it."""
        if self.args.publish != "pypi":
            return "skipped (not published to PyPI)"
        published = self.published_platforms()
        if not published:
            return "nothing published yet", dict(platforms=[])
        done = [self.finish_bytes(p) for p in published]
        pending = [self.PIPELINES[p]["platform"] for p in self.PIPELINES
                   if self.PIPELINES[p]["platform"] not in published]
        return "; ".join(done) + (f"; not yet published: {', '.join(pending)}" if pending else ""), \
            dict(platforms=published)

    def shipped_ptx_leg(self):
        """record_ptx_proof's arguments for the leg whose PTX set this freeze
        shipped: the leg taken (its original build) or the leg built here."""
        name = "%s-%s" % release_reuse.PTX_SET
        doc = self.admitted_leg(name)
        if doc:
            return dict(leg_name=name, built_from=doc["built_from"], release_build=doc["release_build"],
                        set_identity=doc.get("original_set_identity"), tooling_digest=doc.get("tooling_digest"))
        prov = read_json(self.rel / "legs" / f"{name}.provenance.json")
        if prov and prov.get("release_build") and prov.get("source_commit") == self.commit:
            return dict(leg_name=name, built_from=prov["source_commit"], release_build=prov["release_build"],
                        set_identity=prov.get("set_identity"), tooling_digest=prov.get("tooling_digest"))
        return None

    def release_provenance(self):
        legs = {}
        for v, a in release_reuse.LINUX_SETS:
            name = f"{v}-{a}"
            doc = self.admitted_leg(name)
            prov = read_json(self.rel / "legs" / f"{name}.provenance.json")
            if doc:
                legs[name] = dict(origin="taken", built_from=doc["built_from"], admitted_for=doc["admitted_for"],
                                  because=doc["because"], set_identity_digest=doc["set_identity_digest"])
            elif prov:
                legs[name] = dict(origin="built", source_commit=prov.get("source_commit"),
                                  tooling_commit=prov.get("tooling_commit"), tooling_digest=prov.get("tooling_digest"),
                                  overlay=prov.get("overlay"),
                                  set_identity_digest=(prov.get("set_identity") or {}).get("digest"))
        cols = {name: read_json(self.rel / "columns" / f"{name}.provenance.json")
                for name, _, _, out in self.column_specs()}
        return dict(schema="mojolearn.release-provenance.v1", version=self.version, source_commit=self.commit,
                    tooling_commit=self.tooling_commit, earlier_commits=self.state.get("earlier_commits", []),
                    published=self.published_platforms(), legs=legs, columns=cols)

    # ------------------------------------------------------------ the release gate
    def candidate_gate_wheels(self, vendor):
        packages = ['linux', vendor]
        wheels = []
        for package in packages:
            path = self.split_final(package)
            if not path:
                if not self.dry:
                    raise StepFailed('candidate gate requires the packed '+package+' wheel')
                path = self.rel/'linux/final'/f'{SPLIT_PACKAGES[package][1]}-{self.version}-py3-none-manylinux_2_28_x86_64.whl'
            wheels.append(Path(path))
        return wheels

    def candidate_gate_identity(self, vendor):
        import candidate_wheel_install as C
        wheels = self.candidate_gate_wheels(vendor)
        if self.dry:
            return dict(source='candidate', version=self.version, vendor=vendor,
                        source_commit=self.commit, release_modes=list(self.release_modes),
                        wheels={path.name: 'DRY-RUN' for path in wheels})
        key = (vendor, self.version, self.commit, tuple((str(path), path.stat().st_size,
               path.stat().st_mtime_ns, path.stat().st_ctime_ns) for path in wheels))
        cache = getattr(self, '_candidate_gate_identities', {})
        if key not in cache:
            cache[key] = C.identity(C.build_manifest(self.version, vendor, wheels, self.commit))
            self._candidate_gate_identities = cache
        identity = cache[key]
        if release_reuse.record_modes(identity) != self.release_modes:
            raise StepFailed('candidate wheel release modes differ from the selected release modes')
        return identity

    def gate_out(self, vendor):
        identity = self.candidate_gate_identity(vendor)
        stamp = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:20]
        # A new profile must not overwrite an earlier full-campaign outcome.
        return self.rel / f"gate-{vendor}" / (stamp + "-release-light-v1")

    def gate_ok(self, vendor, measure=True):
        """(ok, why): the gate report that came home for this version and vendor says PASS, and (measure) the
        measurement smoke's own box job's summary passes. measure=False judges the gate job's part alone."""
        try:
            identity = self.candidate_gate_identity(vendor)
            output = self.gate_out(vendor)
        except (StepFailed, OSError, ValueError, KeyError, zipfile.BadZipFile) as error:
            return False, 'candidate artifacts are not ready: '+str(error)
        if self.single():
            # --identity-on-wheels: the one box job (identity + device-check units + runtime), not the gate stages
            return self.single_gate_ok(vendor, output, measure=measure)
        doc = read_json(output / "gate-smoke.json")
        if not doc:
            return False, f"no gate report at {output / 'gate-smoke.json'}"
        from release_incremental_gate import SCHEMA as incremental_schema, errors as incremental_errors
        if doc.get('schema') == incremental_schema:
            errors = incremental_errors(doc, identity)
            return (False, '; '.join(errors)) if errors else (True,
                'PASS: unchanged runtime and adapter evidence retained; changed host and verifier boundaries qualified')
        from release_gate_reuse import SCHEMA as component_schema, errors as component_errors
        if doc.get('schema') == component_schema:
            errors = component_errors(doc, identity)
            return (False, '; '.join(errors)) if errors else (True,
                'PASS: retained 11 correctness and 24 autograd records; final installed verifier and all three adapters passed')
        if doc.get('input_mode') != 'candidate' or doc.get('candidate') != identity:
            return False, 'gate report does not identify these candidate wheel hashes and source'
        if doc.get("version") != self.version or doc.get("vendor") != vendor:
            return False, f"the gate report is for {doc.get('vendor')} {doc.get('version')}"
        runner_ok, runner_why = gate_runner_verified(output, identity.get("source_commit"))
        if not runner_ok:
            return False, runner_why
        sys.path.insert(0, str(ROOT / "tools"))
        from gate_smoke_box import gate_verdict
        verdict, why = gate_verdict(doc)
        if verdict != "PASS" or doc.get("verdict") != "PASS":
            return False, f"verdict {verdict}: " + "; ".join(why or doc.get("verdict_reasons") or [])
        if adapter_gate_module().enabled() and doc.get("adapters_enabled") is False:
            return False, "the gate ran without its adapter smoke, but this release ships the adapters"
        from release_requirement_receipts import candidate_gate_errors
        errors = candidate_gate_errors(doc, identity)
        if errors:
            return False, "; ".join(errors)
        from release_gate_profiles import profile, LIGHT
        if profile(doc) == LIGHT:
            once = self.gate_identity_errors(vendor, output, doc.get("correctness"))
            if once:
                return False, "; ".join(once)
            # the measurement smoke runs in every box job (release_gate_job.sh phase measuresmoke, unconditional)
            from release_identity_wheels import measure_smoke_errors
            smoke = measure_smoke_errors(output) if measure else []
            if smoke:
                return False, "; ".join(smoke)
            runtime, cases = self.gate_runtime_error(vendor, output)
            if runtime:
                return False, runtime
            return True, (f"PASS: 11 installed correctness records, 24 autograd entries, three adapter phases, "
                          f"{cases} installed-runtime cases; "
                          f"elapsed {doc['budget']['elapsed_seconds']:.1f}s (no performance campaign)")
        m = doc["measure"]
        return True, f"PASS historical performance campaign: {m['expected']} measured lines"

    #: gate_ok's reason prefix for a gate whose smoke passed and whose installed-runtime qualifier did not
    RUNTIME_GATE_FAILED = "runtime qualifier "

    @staticmethod
    def gate_runtime_error(vendor, output):
        """(error or None, cases): the gate job's installed-runtime qualifier report under OUTPUT.

        0.8.65: the gate printed PASS while runtime-<vendor>/runtime-report.json said "failed" (the
        qualifier had timed out at 12-23%). The box job keeps the smoke's verdict; the release judges both,
        so a failed, stopped or missing qualifier fails the gate and names the report."""
        path = Path(output) / f"runtime-{vendor}" / "runtime-report.json"
        report = read_json(path)
        if not report:
            return f"{Release.RUNTIME_GATE_FAILED}left no report at {path}", None
        if report.get("status") != "passed" or report.get("exit_code") != 0:
            stop = report.get("timed_out") or {}
            partial = report.get("partial_tests") or {}
            detail = (f"stopped by its {stop.get('kind')} timeout after {stop.get('elapsed_s')} s "
                      f"({partial.get('passed')}/{partial.get('total')} finished tests passed)" if stop
                      else str(report.get("error") or report.get("qualifier_error") or "")[:300])
            return f"{Release.RUNTIME_GATE_FAILED}status {report.get('status')} in {path}: {detail}", None
        return None, report.get("cases")

    def run_gate(self, vendor):
        """THE TWO BOX JOBS OF A VENDOR, in order on the same box: the gate job (identity record, device checks,
        runtime qualifier; tools/ops/release_gate.py), then, once it has ended, the measurement smoke's own job
        (--job measuresmoke). They never share the GPU, and boxes-down waits for this step, so for both. A job whose
        part already passed is not launched again; the smoke has no reuse beyond its own landed result."""
        run_id = None
        if not self.dry:
            self.readiness_status(write=True)
            run_id = self.readiness().begin("candidate-" + vendor)
        ok, why = self.gate_ok(vendor)
        if ok:
            if not self.dry:
                self.record_candidate_requirement(vendor, run_id)
            return why
        job_ok, job_why = self.gate_ok(vendor, measure=False)
        if job_why.startswith(self.RUNTIME_GATE_FAILED):
            # The smoke passed and the qualifier ran: relaunching would only rerun green work. A qualifier
            # failure is fixed in code and qualified at the next freeze.
            raise StepFailed(f"the {vendor} release gate did not pass: {job_why}")
        work = self.rel / "gates"
        output = self.gate_out(vendor)
        if not job_ok:
            self.launch_gate_job(vendor, run_id, work, output)
            if self.dry:
                self.launch_measure_smoke_job(vendor, work, output)
                return None
            job_ok, job_why = self.gate_ok(vendor, measure=False)
            if not job_ok:
                report = output / "gate-smoke.json"
                if run_id and report.is_file():
                    self.readiness().record(run_id, "failed", [report], job_why, 1, 5, 0, failed=1)
                raise StepFailed(f"the {vendor} release gate did not pass: {job_why}")
        # the gate job has ended and passed: now the measurement smoke's own job on the same box
        self.launch_measure_smoke_job(vendor, work, output)
        if self.dry:
            return None
        ok, why = self.gate_ok(vendor)
        if not ok:
            if run_id:
                from release_identity_wheels import measure_smoke_summary
                reports = [p for p in (measure_smoke_summary(output), work / f"measuresmoke-{vendor}.log") if p.is_file()]
                if reports:
                    self.readiness().record(run_id, "failed", reports, why, 1, 5, 0, failed=1)
            raise StepFailed(f"the {vendor} release gate did not pass: {why}")
        self.record_candidate_requirement(vendor, run_id, executed=True)
        return why

    def launch_gate_job(self, vendor, run_id, work, output):
        # plan item 4 (docs/plans/POST_0863_RELEASE_TOOLING.md): one persisted box job per vendor through the
        # supported wrappers (tools/nvidia_central.sh submit, tools/do_amd_steward.sh run) that builds the eight
        # autograd executables on the box and runs the gate there; nothing outside this repository
        cmd = [PY, "tools/ops/release_gate.py", vendor, self.version, "--source-commit", self.commit,
               "--source-tree", str(self.source_checkout(create=not self.dry)),
               "--out", str(output), "--state", str(work / f"{vendor}-{output.name}.job.json"),
               "--profile", "release-light-v1", "--max-wait-hours", "72"]
        for wheel in self.candidate_gate_wheels(vendor):
            cmd += ['--candidate-wheel', str(wheel)]
        prior = self.gate_autograd_prior(vendor)
        if prior:
            cmd += ["--autograd-prior", str(prior)]  # the build reuses each proven-unchanged binary
        runtime_prior = self.gate_runtime_prior(vendor) if not self.dry else None
        if runtime_prior:
            cmd += ["--runtime-prior", str(runtime_prior)]  # unit receipts the qualifier may reuse
        if not adapter_gate_module().enabled():
            cmd.append("--no-adapters")  # no adapters ship: the gate records its adapter stage as disabled
        cmd += self.gate_identity_args(vendor)   # --identity-on-wheels: the one box job records the columns too
        self.refuse_device_lanes_after_publish("gate-" + vendor)
        if not self.dry:
            work.mkdir(parents=True, exist_ok=True)
        try:
            self.must(cmd, log=work / f"gate-{vendor}.log", what=f"the {vendor} release gate box job")
        except StepFailed as error:
            reports = [p for p in (output / "gate-smoke.json", work / f"gate-{vendor}.log") if p.is_file()]
            if run_id and reports:
                self.readiness().record(run_id, "failed", reports, str(error), 1, 5, 0, failed=1)
            raise
        if self.dry:
            # the job submission the command makes, wrapper by wrapper (nothing runs)
            for line in release_gate_module().Runner(release_gate_module().parse(cmd[2:])).dry_lines():
                self.say("    would: " + line)

    def measure_smoke_cmd(self, vendor, work, output):
        """tools/ops/release_gate.py --job measuresmoke: the measurement smoke's own persisted box job (its own
        NVIDIA cap, MEASURE_SMOKE_CAP_MIN), landing in <gate out>/measuresmoke-job."""
        from release_identity_wheels import MEASURE_JOB_DIR
        cmd = [PY, "tools/ops/release_gate.py", vendor, self.version, "--job", "measuresmoke",
               "--source-commit", self.commit, "--source-tree", str(self.source_checkout(create=not self.dry)),
               "--out", str(output / MEASURE_JOB_DIR),
               "--state", str(work / f"{vendor}-{output.name}-measuresmoke.job.json"),
               "--profile", "release-light-v1", "--max-wait-hours", "72", "--no-adapters"]
        for wheel in self.candidate_gate_wheels(vendor):
            cmd += ['--candidate-wheel', str(wheel)]
        return cmd

    def launch_measure_smoke_job(self, vendor, work, output):
        cmd = self.measure_smoke_cmd(vendor, work, output)
        self.refuse_device_lanes_after_publish("gate-" + vendor)
        if not self.dry:
            work.mkdir(parents=True, exist_ok=True)
        try:
            self.must(cmd, log=work / f"measuresmoke-{vendor}.log", what=f"the {vendor} measurement smoke box job")
        except StepFailed as error:
            self.say(f"the {vendor} measurement smoke job: {error}")   # the verdict that follows names why
        if self.dry:
            for line in release_gate_module().Runner(release_gate_module().parse(cmd[2:])).dry_lines():
                self.say("    would: " + line)

    def runtime_unit_sources(self, gate):
        """Every retained directory of installed-runtime unit receipts under GATE (gate-nvidia, gate-amd,
        gate-apple) of every version and freeze in the release state root: the unit stores the jobs append
        to (runtime-units-<class>/) and the per-run receipts (runtime-*/qualify/units/)."""
        root = self.base.parent
        found = sorted(root.glob(f"*/*/{gate}/*/runtime-units-*")) + sorted(root.glob(f"*/*/{gate}/runtime-units-*"))
        found += sorted(root.glob(f"*/*/{gate}/*/runtime-*/qualify/units"))
        found += sorted(root.glob(f"*/*/{gate}/runtime-*/qualify/units"))
        return [p for p in found if p.is_dir()]

    def gate_runtime_prior(self, vendor):
        """A staged directory of the newest passing unit receipt per (device class, unit) for this gate's
        qualifier runs (VENDOR, and cpu on the NVIDIA job), or None when there is none
        (docs/release/RUNTIME_QUALIFICATION.md). Staged anew per launch; the originals stay where they are."""
        import runtime_qualifier_units as units
        vendors = (vendor, "cpu") if vendor == "nvidia" else (vendor,)
        destination = self.rel / "gates" / ("runtime-prior-%s-%s" % (vendor, dt.datetime.now(
            dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
        staged = units.stage_prior(self.runtime_unit_sources(f"gate-{vendor}"), destination, vendors)
        if staged:
            self.say(f"  {staged} runtime unit receipt(s) staged for reuse: {destination}")
        return destination if staged else None

    def gate_autograd_prior(self, vendor):
        """A complete retained autograd set the gate job's build may reuse: --gate-autograd-root/<vendor>
        when it holds one, else the newest earlier gate's built set of this version (any freeze)."""
        root = getattr(self.args, 'gate_autograd_root', None)
        candidates = [Path(root) / vendor] if root else []
        candidates += sorted(self.base.glob(f"*/gate-{vendor}/*/autograd-bin"),
                             key=lambda p: (p / "build-manifest.json").stat().st_mtime
                             if (p / "build-manifest.json").is_file() else 0, reverse=True)
        for directory in candidates:
            doc = read_json(directory / "build-manifest.json") or {}
            if doc.get("state") == "built" and doc.get("vendor") == vendor:
                return directory
        return None

    def gate_waiver(self, vendor):
        """The owner's --owner-waiver reason when it waives this vendor's candidate
        gate (prefix `candidate-`), logged in the release state; else None."""
        path = getattr(self.args, "owner_waiver", None)
        if not path:
            return None
        waiver = json.loads(Path(path).read_text())
        if not any(("candidate-" + vendor).startswith(p) for p in waiver.get("prefixes", [])):
            return None
        self.state.setdefault("owner_waivers", []).append(dict(
            approved_by=waiver["approved_by"], reason=waiver["reason"], waived=["gate-" + vendor], at=now()))
        return f"WAIVED by owner ({waiver['approved_by']}): {waiver['reason']}"

    def candidate_verdict(self, vendor):
        """(ok, why) of one vendor's candidate gate: its PASS report, or the owner's
        --owner-waiver for candidate-<vendor> (logged in the release state)."""
        why = self.gate_waiver(vendor)
        return (True, why) if why else self.gate_ok(vendor)

    def release_gate(self, stage=None, vendors=GATE_VENDORS):
        """THE ONE GATE PREDICATE (plan item 2): the candidate gates of VENDORS and
        every required readiness row of STAGE (all stages when None), after the
        evidence that already exists is admitted. step_gate_<vendor> (through
        candidate_verdict), the publish preflight and the release record all
        decide with it. (ready, blocked reasons)."""
        reasons = [f"candidate gate {vendor}: {why}" for vendor in vendors
                   for ok, why in [self.candidate_verdict(vendor)] if not ok]
        if reasons:
            return False, reasons  # as before: a missing candidate gate blocks before the ledger is read
        rows = self.readiness_status(write=not self.dry and not self.readonly)
        ready, blocked = self.waived(release_requirements.gate_rows(rows, stage))
        return not reasons and ready, reasons + ([] if ready else blocked)

    def step_gate(self, vendor):
        ok, why = self.candidate_verdict(vendor)
        return why if ok else self.run_gate(vendor)

    def step_gate_nvidia(self):
        return self.step_gate("nvidia")

    def step_gate_amd(self):
        return self.step_gate("amd")

    # ------------------------------------------------------------ the gate boxes (ORCHESTRATOR ONLY)
    #: the gate lane whose tree the NVIDIA box jobs run in (tools/ops/release_gate.py LANE)
    BOX_LANE = "release-gate"
    #: the AMD droplet's first lease (minutes; RUNBOOK 0.8.65 section 11) and its renewal while a box step is pending
    AMD_LEASE_MIN = 240
    AMD_RENEW_MIN = 120
    AMD_RENEW_EVERY_S = 85 * 60   # renewed with margin before the 120-minute extension runs out
    #: one NVIDIA hold's on-box ceiling (nvidia_central.sh `sh` gives a command 4 h); the keeper renews it
    HOLD_MAX_S = 3 * 3600 + 1800
    KEEPER_TICK_S = 60
    clock = staticmethod(time.time)
    box_popen = staticmethod(subprocess.Popen)

    @staticmethod
    def nvc():
        return os.environ.get("MOJOLEARN_RELEASE_NVC", "tools/nvidia_central.sh")

    @staticmethod
    def do_amd():
        return os.environ.get("MOJOLEARN_RELEASE_DO_AMD", "tools/do_amd_steward.sh")

    @staticmethod
    def box_persist():
        return os.environ.get("MOJOLEARN_RELEASE_BOX_PERSIST", "tools/box_persist.py")

    @staticmethod
    def nvc_state_root():
        return Path(os.environ.get("MOJOLEARN_DEVPOD_STATE") or os.path.expanduser("~/mojolearn-evidence/devpods"))

    @staticmethod
    def do_state():
        return Path(os.environ.get("MOJOLEARN_STEWARD_DO_STATE")
                    or os.path.expanduser("~/mojolearn-evidence/do-amd-steward"))

    @staticmethod
    def orchestrator():
        return os.environ.get("MOJOLEARN_ORCHESTRATOR") == "1"

    def nvc_live(self):
        """The shared NVIDIA pods up now (tools/nvidia_central.sh live_pods: a state.env per pod key)."""
        return [k for k in ("nvc1", "nvc2", "nvc3") if (self.nvc_state_root() / k / "state.env").is_file()]

    def do_droplet(self):
        """The DigitalOcean droplet id of the live AMD steward, or None."""
        env = self.do_state() / "state.env"
        try:
            m = re.search(r"^DROPLET_ID=['\"]?([0-9]+)", env.read_text(), re.M)
        except OSError:
            return None
        return m.group(1) if m else "unknown"

    def selected_steps(self):
        return getattr(self, "selected", None) or self.STEPS

    def waiver_covers(self, row):
        """True when the owner's --owner-waiver names ROW by prefix (read only; nothing is logged)."""
        path = getattr(self.args, "owner_waiver", None)
        if not path:
            return False
        try:
            prefixes = json.loads(Path(path).read_text()).get("prefixes", [])
        except (OSError, ValueError):
            return False
        return any(row.startswith(p) for p in prefixes)

    def box_step_owed(self, step):
        """Why box step STEP still has work to do on its box, or None."""
        if step.startswith("gate-"):
            vendor = step[len("gate-"):]
            if self.waiver_covers("candidate-" + vendor):
                return None
            if self.single():
                # one box job: REUSED n / RUN m per vendor; no rental when nothing is owed
                return self.single_box_owed(vendor)
            ok, why = self.gate_ok(vendor)
            return None if ok else why
        owed = getattr(self, "box_step_owed_" + step.replace("-", "_"), None)
        return owed() if owed else None

    def box_needs(self):
        """{vendor: [step: why]} of the selected box steps with work to do on a box."""
        needs = {}
        selected = self.selected_steps()
        for step, vendor in BOX_CONSUMERS.items():
            if step in selected:
                why = self.box_step_owed(step)
                if why:
                    needs.setdefault(vendor, []).append(f"{step}: {why[:160]}")
        return needs

    def held_boxes(self):
        """{vendor: record} of the boxes this release holds (not taken down, not released)."""
        return {v: b for v, b in (self.state.get("boxes") or {}).items()
                if not b.get("down") and not b.get("released")}

    def box_alive(self, vendor, rec):
        if vendor == "nvidia":
            return bool(rec.get("pod")) and (self.nvc_state_root() / rec["pod"] / "state.env").is_file()
        return self.do_droplet() is not None and self.do_droplet() == rec.get("droplet")

    def boxes_pending(self):
        """The vendors with a selected box step still to run (the schedule's view when one runs)."""
        status = getattr(self, "_status", None)
        out = set()
        for step, vendor in BOX_CONSUMERS.items():
            if step not in self.selected_steps():
                continue
            if status is not None:
                if status.get(step) in ("pending", "running"):
                    out.add(vendor)
            elif not self.recorded(step):
                out.add(vendor)
        return out

    def box_plan(self, needs):
        """What gate-boxes rents (or adopts) and for how long, one line per box."""
        c12 = self.commit[:12]
        lines = []
        if "nvidia" in needs:
            live = self.nvc_live()
            what = (f"adopt the live shared NVIDIA pod {live[0]} (nothing rented)" if live else
                    f"rent the shared NVIDIA pod: bash {self.nvc()} up 1 (one pod, 1x sm_89 GPU: RTX 4090, L40S, "
                    "RTX 6000 Ada or L4, out of stock retried by nvidia_central.sh; it deletes itself 60 idle "
                    "minutes after its last job)")
            lines.append(f"would {what}, sync lane {self.BOX_LANE} to {c12}, and hold it busy (nvidia_central.sh "
                         f"sh {self.BOX_LANE}, at most {self.HOLD_MAX_S // 60} minutes a hold, renewed) while "
                         + ", ".join(n.split(':')[0] for n in needs['nvidia']) + " are pending")
        if "amd" in needs:
            what = ("adopt the live AMD droplet " + str(self.do_droplet()) + " (nothing rented; its lease extended to "
                    f"{self.AMD_LEASE_MIN} minutes)" if self.do_droplet() else
                    f"rent the AMD droplet: bash {self.do_amd()} up {self.AMD_LEASE_MIN} (gpu-mi300x1-192gb, else "
                    f"gpu-mi325x1-256gb; gfx942) for a {self.AMD_LEASE_MIN}-minute lease")
            lines.append(f"would {what}, renewed by {self.AMD_RENEW_MIN} minutes every {self.AMD_RENEW_EVERY_S // 60} "
                         "minutes while " + ", ".join(n.split(':')[0] for n in needs['amd']) + " are pending")
        lines.append("then boxes-down takes down what this release rented once every box step has settled")
        return lines

    def nvidia_box_up(self):
        boxes = self.state.setdefault("boxes", {})
        prev = boxes.get("nvidia") or {}
        logs = self.rel / "boxes"
        orch = dict(MOJOLEARN_ORCHESTRATOR="1")
        if prev and not prev.get("down") and not prev.get("released") and self.box_alive("nvidia", prev):
            rec = prev
        elif self.nvc_live():
            # a shared pod is already up (another release run's or a lane's): adopted, nothing rented
            rec = dict(pod=self.nvc_live()[0], rented=False, adopted=True, at=now(), lane=self.BOX_LANE)
            with self._lock:
                boxes["nvidia"] = rec
                self.save()
        else:
            before = self.nvc_live()
            self.must(["bash", self.nvc(), "up", "1"], env=orch, log=logs / "nvidia-up.log",
                      what="nvidia_central.sh up 1")
            after = self.nvc_live()
            if not after:
                raise StepFailed(f"nvidia_central.sh up 1 left no shared pod up ({logs / 'nvidia-up.log'})")
            new = [k for k in after if k not in before]
            rec = dict(pod=(new or after)[0], rented=bool(new), adopted=not new, at=now(), lane=self.BOX_LANE)
            with self._lock:
                boxes["nvidia"] = rec
                self.save()
        # the hold and the box jobs run in the gate lane's tree: the lane needs a pod and the frozen source
        self.must(["bash", self.nvc(), "sync", self.BOX_LANE, str(self.source_checkout())],
                  log=logs / "nvidia-sync.log", what=f"nvidia_central.sh sync {self.BOX_LANE}")
        return f"NVIDIA {rec['pod']} " + ("rented" if rec.get("rented") else "adopted (already up)")

    def amd_box_up(self):
        boxes = self.state.setdefault("boxes", {})
        prev = boxes.get("amd") or {}
        logs = self.rel / "boxes"
        if self.do_droplet():
            # already up: this release's own droplet, or adopted; its lease is extended to the first lease
            ours = bool(prev.get("rented") and not prev.get("down") and prev.get("droplet") == self.do_droplet())
            self.must(["bash", self.do_amd(), "extend", str(self.AMD_LEASE_MIN)], log=logs / "amd-extend.log",
                      what="do_amd_steward.sh extend")
            rec = dict(droplet=self.do_droplet(), rented=ours, adopted=not ours, at=prev.get("at") if ours else now())
        else:
            self.must(["bash", self.do_amd(), "up", str(self.AMD_LEASE_MIN)], log=logs / "amd-up.log",
                      what=f"do_amd_steward.sh up {self.AMD_LEASE_MIN}")
            if not self.do_droplet():
                raise StepFailed(f"do_amd_steward.sh up left no droplet state ({logs / 'amd-up.log'})")
            rec = dict(droplet=self.do_droplet(), rented=True, adopted=False, at=now())
        rec["lease_extended_at"] = self.clock()
        with self._lock:
            boxes["amd"] = rec
            self.save()
        return f"AMD droplet {rec['droplet']} " + ("rented" if rec["rented"] else "adopted (already up)")

    def step_gate_boxes(self):
        """GATE-BOXES (ORCHESTRATOR ONLY): the boxes the selected box steps need, up and recorded; held while
        they are pending (BoxKeeper). Never an Apple machine: only the NVIDIA pod and the AMD droplet."""
        needs = self.box_needs()
        if not needs:
            return "not needed: no selected box step has work to do on a box"
        planned = self.saved_plan()
        if planned and (planned.get("boxes") or {}).get("action") == "none":
            # THE PLAN OWES NO DEVICE WORK (tools/release_plan.py): rent nothing. A box step that still
            # reports work points at evidence to restore or a stale plan, never at a rental.
            raise StepFailed("the release plan (" + str(self.plan_json_path()) + ") owes no device work, yet "
                             + "; ".join(n for v in sorted(needs) for n in needs[v])
                             + ": restore the evidence it names or rerun --plan; gate-boxes rents nothing")
        for line in self.box_plan(needs):
            self.say("  " + line)
        if self.dry:
            if not self.orchestrator():
                self.say("  a real run refuses this step without MOJOLEARN_ORCHESTRATOR=1 (only the orchestrator rents)")
            return "would bring up: " + ", ".join(sorted(needs))
        if not self.orchestrator():
            raise StepFailed("gate-boxes rents or holds the " + " and ".join(sorted(needs)) + " box(es) for "
                             + "; ".join(n for v in sorted(needs) for n in needs[v])
                             + ": ORCHESTRATOR ONLY, rerun with MOJOLEARN_ORCHESTRATOR=1")
        done = []
        if "nvidia" in needs:
            done.append(self.nvidia_box_up())
        if "amd" in needs:
            done.append(self.amd_box_up())
        self.keep_boxes()
        return "; ".join(done), dict(boxes={v: dict(b) for v, b in self.held_boxes().items()})

    def have_gate_boxes(self, rec):
        needs = self.box_needs()
        if not needs:
            return "not needed: no selected box step has work to do on a box"
        held = self.held_boxes()
        if all(v in held and self.box_alive(v, held[v]) for v in needs):
            return "have " + ", ".join(f"{v} {held[v].get('pod') or held[v].get('droplet')}" for v in sorted(needs))
        return None

    # the keeper: a thread of the release process, never a background service; it ends with the run
    def keep_boxes(self):
        """Start the BoxKeeper unless it runs, when this release holds a box (a real run only)."""
        if self.dry or self.readonly or not self.held_boxes():
            return None
        with self._lock:
            keeper = getattr(self, "_keeper", None)
            if keeper is None or not keeper.alive():
                keeper = self._keeper = BoxKeeper(self)
                keeper.start()
        return keeper

    def stop_keeper(self):
        keeper = getattr(self, "_keeper", None)
        if keeper is not None:
            keeper.stop()

    BOX_TAG_RE = re.compile(r"box_persist: job (\S+) persists to R2|persisted as (\S+)|resuming (\S+) \(launched")

    def box_job_tags(self):
        """The R2 tags of every box job this freeze launched (the gate and producer logs name them)."""
        tags = []
        for log in sorted(self.rel.glob("gates/*.log")) + sorted(self.rel.glob("published/*.log")):
            try:
                text = log.read_text(errors="replace")
            except OSError:
                continue
            for m in self.BOX_TAG_RE.finditer(text):
                tag = next(g for g in m.groups() if g).rstrip(".,;)")
                if tag not in tags:
                    tags.append(tag)
        return tags

    def box_job_home(self, tag):
        return any((d / "final").is_dir() for d in (self.rel / "gates" / f"fetch-{tag}",
                                                    self.rel / "published" / f"fetch-{tag}",
                                                    self.rel / "box-fetch" / tag))

    def step_boxes_down(self):
        """BOXES-DOWN: every box job's results home first (tools/box_persist.py fetch --tag), then what
        gate-boxes rented taken down; an adopted box is released, never taken down."""
        held = self.held_boxes()
        self.stop_keeper()
        if not held:
            return "nothing to take down: this release holds no box"
        tags = self.box_job_tags()
        missing = [t for t in tags if not self.box_job_home(t)]
        failed = []
        for tag in missing:
            rc = self.run([PY, self.box_persist(), "fetch", "--tag", tag, "--dest", self.rel / "box-fetch" / tag,
                           "--extract"], log=self.rel / "boxes" / f"fetch-{tag}.log")
            if rc != 0 and not self.dry:
                failed.append(tag)
        if failed:
            raise StepFailed("the results of " + ", ".join(failed) + " are not home; nothing is taken down (the pod "
                             "deletes itself idle, the droplet at its lease); rerun boxes-down")
        rented = {v: b for v, b in held.items() if b.get("rented")}
        if self.dry:
            for v, b in sorted(held.items()):
                self.say(f"  would {'take down' if v in rented else 'release (adopted, left up)'} {v} "
                         f"{b.get('pod') or b.get('droplet')}")
            return f"would pull {len(missing)} job result(s) home and take down {', '.join(sorted(rented)) or 'nothing'}"
        if rented and not self.orchestrator():
            raise StepFailed("boxes-down deletes rented boxes: ORCHESTRATOR ONLY, rerun with MOJOLEARN_ORCHESTRATOR=1")
        done = []
        for vendor, rec in sorted(held.items()):
            if vendor in rented:
                if vendor == "nvidia":
                    self.must(["bash", self.nvc(), "down", rec["pod"]], env=dict(MOJOLEARN_ORCHESTRATOR="1"),
                              log=self.rel / "boxes" / "nvidia-down.log", what="nvidia_central.sh down")
                else:
                    self.must(["bash", self.do_amd(), "down"], log=self.rel / "boxes" / "amd-down.log",
                              what="do_amd_steward.sh down")
            with self._lock:
                rec["down" if vendor in rented else "released"] = now()
                self.save()
            done.append(f"{vendor} {rec.get('pod') or rec.get('droplet')} "
                        + ("taken down" if vendor in rented else "released (adopted, left up)"))
        return f"{len(tags)} box job(s) home ({len(missing)} fetched now); " + "; ".join(done)

    def have_boxes_down(self, rec):
        return None if self.held_boxes() else "have: this release holds no box"

    # ------------------------------------------------------------ published-<vendor> (post stage)
    PUBLISHED_SCHEMA = "mojolearn.release-published-install.v2"
    PUBLISHED_SMOKE_SCHEMA = "mojolearn.release-published-install.v1"

    def published_out(self, vendor):
        """Where retained_producers looks: <commit12>/published-<vendor>/published-report.json."""
        return self.rel / f"published-{vendor}"

    def published_smoke_out(self, vendor):
        """The explicit --published-smoke report: <commit12>/published-smoke-<vendor>/ (never ledger evidence)."""
        return self.rel / f"published-smoke-{vendor}"

    def published_wheels(self, vendor):
        """The qualified candidate wheels the published files must equal."""
        if vendor == "apple":
            return [self.shipped_wheel("macos", self.macos_wheel())]
        return [self.shipped_wheel("linux", self.split_final("linux"))] + (
            [self.split_final("nvidia")] if vendor == "nvidia" else [self.split_final("amd")])

    def published_index_url(self):
        """PyPI. MOJOLEARN_PUBLISHED_INDEX_URL names a mirror or a test index; a report made against any
        index but PyPI re-derives differently in the ledger and is refused there, so it is never evidence."""
        return os.environ.get("MOJOLEARN_PUBLISHED_INDEX_URL") or "https://pypi.org/simple"

    def published_gate(self, vendor):
        """(receipt paths, verdict): the gate that passed these candidate bytes, whose receipts are the
        published row's evidence by reference. StepFailed when that gate has not passed: the published
        bytes are then not verified work, and nothing here runs a device to make up for it."""
        if vendor == "apple":
            w = self.macos_wheel()
            if not w:
                raise StepFailed("published-apple: no macOS candidate wheel; run macos-build")
            receipts, why = [], []
            if self.macos_self_test_passed():
                receipts.append(self.macos_self_test())
                why.append("macOS wheel self-test PASSED")
            elif smoke_passed(self.rel / "smoke-macos" / "results.json", w):
                receipts.append(self.rel / "smoke-macos" / "results.json")
                why.append("macOS smoke PASSED")
            else:
                raise StepFailed("published-apple: the macOS wheel's smoke has not passed for these bytes "
                                 "(macos-smoke); the published bytes are judged against a passed gate only")
            if adapter_gate_module().enabled():
                ok, adapters = self.apple_adapters_ok()
                if not ok:
                    raise StepFailed("published-apple: the Apple adapter gate has not passed: " + adapters)
                out = self.apple_adapters_out()
                receipts += [p for p in (out / "wheel.json", out / adapter_gate_module().RECEIPT, out / "reuse.json")
                             if p.is_file()]
                why.append("adapters " + adapters)
            return receipts, "; ".join(why)
        ok, why = self.gate_ok(vendor)
        if not ok:
            raise StepFailed(f"published-{vendor}: gate-{vendor} has not passed for these candidate bytes: {why}")
        out = self.gate_out(vendor)
        receipts = [p for p in (out / "gate-smoke.json", out / "adapter-smoke" / adapter_gate_module().RECEIPT,
                                out / f"runtime-{vendor}" / "runtime-report.json") if p.is_file()]
        return receipts, f"gate-{vendor} " + why

    def published_ok(self, vendor):
        """(ok, why): a passed published-<vendor> bytes report for this version, vendor and these candidates."""
        out = self.published_out(vendor)
        doc, req = read_json(out / "published-report.json"), read_json(out / "request.json") or {}
        if not doc:
            return False, f"no published report at {out}"
        if (doc.get("schema") != self.PUBLISHED_SCHEMA or doc.get("vendor") != vendor
                or doc.get("version") != self.version):
            return False, (f"the report at {out} is a {doc.get('schema')} report for {doc.get('vendor')} "
                           f"{doc.get('version')}")
        if doc.get("status") != "passed" or doc.get("exit_code") != 0:
            differs = "; ".join("%s published %s, candidate %s" % (m.get("file"), m.get("published_sha256"),
                                                                  m.get("candidate_sha256"))
                                for m in doc.get("mismatch") or [])
            return False, (f"published-{vendor} status {doc.get('status')}: "
                           + (differs or str(doc.get("error") or "")[:200]
                              or ", ".join(r["id"] for r in doc.get("coverage", []) if r.get("status") != "passed")))
        wheels = self.published_wheels(vendor)
        if not all(wheels) or req.get("candidates") != {Path(w).name: sha256(w) for w in wheels}:
            return False, f"the published-{vendor} report compared other candidate wheels than this release's"
        return True, "PASSED " + " ".join("%s=%s/%s" % (r.get("id"), r.get("completed"), r.get("expected"))
                                          for r in doc.get("coverage", []))

    def step_published(self, vendor):
        """published-<vendor>: the published files resolved from PyPI for the candidates' platform tags, their
        bytes equal to the gated candidates', their RECORD intact, the gate's receipts as the evidence. No
        device, no box, no install. Different bytes FAIL with both hashes; nothing is rerun."""
        if self.args.publish != "pypi":
            return "skipped (not published to PyPI)"
        ok, why = self.published_ok(vendor)
        if ok:
            return why + " (report exists)"
        wheels = self.published_wheels(vendor)
        if not all(wheels) and not self.dry:
            raise StepFailed(f"published-{vendor}: a candidate wheel is missing; run linux-pack / macos-build")
        if self.dry:
            receipts, verdict = [], "<the gate's verdict>"
        else:
            receipts, verdict = self.published_gate(vendor)
        out, work = self.published_out(vendor), self.rel / "published"
        if out.exists() and not self.dry:
            # the producer refuses an existing output; a failed attempt is kept beside it, never deleted
            out.rename(out.with_name(out.name + ".failed-" + dt.datetime.now().strftime("%Y%m%d-%H%M%S")))
        names = [str(w) if w else "<candidate wheel>" for w in wheels]
        cmd = [PY, str(ROOT / "tools" / "release_published_install.py"), "check", "--vendor", vendor,
               "--version", self.version, "--source-commit", self.commit,
               *[a for n in names for a in ("--candidate-wheel", n)],
               *[a for r in receipts for a in ("--gate-evidence", str(r))], "--gate-verdict", verdict,
               "--index-url", self.published_index_url(),
               "--python", self.smoke_python() if not self.dry else "python3.12", "--out", str(out)]
        if not self.dry:
            work.mkdir(parents=True, exist_ok=True)
        rc = self.run(cmd, log=work / f"published-{vendor}.log")
        if self.dry:
            return None
        # the report names what failed (both hashes when the bytes differ); it is judged, never rerun
        ok, why = self.published_ok(vendor)
        if not ok or rc != 0:
            raise StepFailed(f"published-{vendor} did not pass (exit {rc}): {why} "
                             f"({work / f'published-{vendor}.log'})")
        return why

    def step_published_apple(self):
        return self.step_published("apple")

    def step_published_nvidia(self):
        return self.step_published("nvidia")

    def step_published_amd(self):
        return self.step_published("amd")

    def have_published(self, vendor):
        if self.args.publish != "pypi":
            return None
        ok, why = self.published_ok(vendor)
        return (why + " (report exists)") if ok else None

    def have_published_apple(self, rec):
        return self.have_published("apple")

    def have_published_nvidia(self, rec):
        return self.have_published("nvidia")

    def have_published_amd(self, rec):
        return self.have_published("amd")

    # ------------------------------------------------------------ --published-smoke (REMOVED 2026-11-05)
    def published_smoke_vendors(self):
        """{vendor: reason} of the --published-smoke requests of this invocation (each one is refused)."""
        vendors = list(getattr(self.args, "published_smoke", None) or [])
        reason = (getattr(self.args, "published_smoke_reason", None) or "").strip()
        return {v: reason for v in vendors}

    def record_published_smoke_requests(self):
        """REMOVED (lane identity-once-on-frozen-source, Andrew 2026-11-05: identity runs ONCE per release, before
        publish, NEVER after): the published-smoke-<vendor> steps ran the installed verifier (identity lanes) on a
        vendor device after publication. A request is refused; nothing is selected or recorded."""
        wanted = self.published_smoke_vendors()
        if wanted:
            raise SystemExit("release: --published-smoke is removed: no release step runs identity lanes on a device "
                             "after the first publish (Andrew 2026-11-05). published-<vendor> checks the published "
                             "bytes against the gated candidates on this Mac; a real fault is fixed in code and "
                             "ships in the next version")
        return {}

    def step_record(self):
        if self.args.publish != "pypi":
            return "skipped (not published to PyPI)"
        published = self.published_platforms()
        if not published:
            return "nothing published yet", dict(platforms=[])
        # THE RELEASE GATE cannot be skipped: every published Linux vendor needs its gate PASS
        # (or the owner's --owner-waiver), and every pre- and post-publication row (release_gate)
        if not self.dry:
            ready, reasons = self.release_gate(vendors=[v for v in GATE_VENDORS if v in published])
            if not ready:
                raise StepFailed("release record requires the candidate gates and the pre- and post-publication "
                                 "ledger checks: " + "; ".join(reasons))
        # THE RELEASE EVIDENCE SET (Andrew 2026-11-06; tools/release_evidence_set.py): ONE committed set at the
        # fixed bench/results/release_evidence/, replaced as a whole by every release, the release tag v<version>
        # on the commit that holds it; no dated directory per release. The record files the next release's
        # reuse plan reads (alpha manifests, admissions, provenance, binding identities, the PTX proof) are
        # staged here first and enter the set under the same names.
        import release_evidence_set
        tag = release_evidence_set.TAG_FORMAT.format(version=self.version)
        fixed = ROOT / release_evidence_set.EVIDENCE_DIR
        if self.dry:
            return "would replace %s with the %s evidence and tag %s" % (
                fixed.relative_to(ROOT), self.version, tag), dict(platforms=published)
        local_tag = git("rev-parse", "-q", "--verify", f"refs/tags/{tag}^{{commit}}", check=False)
        remote_tag = git("ls-remote", "--tags", "origin", f"refs/tags/{tag}", check=False)
        if local_tag or remote_tag.strip():
            raise StepFailed(f"the release tag {tag} already exists; a release tag is never moved")
        stage = self.stage_record_files(published)
        try:
            manifest = release_evidence_set.write_set(self.evidence_set_context(published, stage), ROOT)
        except release_evidence_set.EvidenceError as exc:
            raise StepFailed(str(exc))
        self.say(f"  wrote {fixed.relative_to(ROOT)}: {len(manifest['items'])} items, "
                 f"{len(manifest['absent_optional'])} optional absent, {len(manifest['references'])} references")
        # `git add -A` on the directory stages the previous set's removed files too: only this set remains
        self.must(["git", "-C", ROOT, "add", "-A", "--", fixed], what="git add record")
        self.must(["git", "-C", ROOT, "commit", "-q", "-m",
                   f"Release evidence for {self.version} ({', '.join(published)}): replaces the previous set",
                   "-m", "Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"],
                  what="git commit record")
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        # main has usually moved on during the builds; the record rides on top
        # of it (0.8.16's record push was rejected as non-fast-forward)
        self.must(["git", "-C", ROOT, "fetch", "-q", "origin", branch], what="git fetch")
        self.must(["git", "-C", ROOT, "rebase", "-q", f"origin/{branch}"], what="git rebase record onto origin")
        self.must(["git", "-C", ROOT, "push", "origin", f"HEAD:refs/heads/{branch}"], what="git push record")
        # THE RELEASE TAG on the commit that holds this set: `git show v<version>:<set>/manifest.json`
        self.must(["git", "-C", ROOT, "tag", "-a", tag, "-m", f"mojolearn {self.version}: release evidence", "HEAD"],
                  what="git tag release")
        self.must(["git", "-C", ROOT, "push", "origin", f"refs/tags/{tag}"], what="git push release tag")
        return str(fixed.relative_to(ROOT)) + f" ({', '.join(published)}) committed on {branch}, tagged {tag}", \
            dict(platforms=published)

    def stage_record_files(self, published):
        """The record files of this release, written fresh into <commit12>/record-staged/ (cleared first)."""
        rec = self.rel / "record-staged"
        if rec.exists():
            shutil.rmtree(rec)
        rec.mkdir(parents=True)
        for platform in published:
            art = self.rel / f"publish-{platform}"
            smoke = (self.recorded(self.publish_step(platform)) or {}).get("smoke") \
                or self.rel / f"smoke-{platform}" / "results.json"
            for src, name in ((art / "artifact" / "alpha-manifest.json", f"alpha-manifest-{platform}.json"),
                              (art / "file-admission.json", f"file-admission-{platform}.json"),
                              (Path(smoke), f"{'light-smoke' if self.gpu_smoke() else 'self-test'}-{platform}.json")):
                if src.is_file():
                    shutil.copy2(src, rec / name)
        write_json(rec / "release-provenance.json", self.release_provenance())
        # The identity of every shipped binding and what it shipped as, so the
        # next release decides REUSE or BUILD from this record alone (the
        # Apple toolchain that built the macOS wheel included).
        linux = [self.split_final(k) for k in SPLIT_PACKAGES if k in published]
        release_reuse.record_identities(self.plan(), linux, self.macos_wheel() if "macos" in published else None,
                                        rec / "binding-identities.json")
        # THE PTX SET'S ORIGINAL PROOF (lane ptx-reuse, 2026-11-01): the build
        # proof, PTX_BASELINE.json and stage manifest of the leg whose PTX bytes
        # shipped, so the next release can take the published PTX set from this
        # record alone (release_reuse.ptx_proof_source) when no PTX binding builds.
        if "nvidia" in published:  # the PTX set ships in mojolearn-nvidia (2026-11-05)
            leg = self.shipped_ptx_leg()
            if leg:
                try:
                    self.say("  retained the PTX proof: " + str(release_reuse.record_ptx_proof(rec, **leg)))
                except (OSError, ValueError, KeyError, SystemExit) as exc:
                    self.say(f"  NOTE the PTX proof is not retained ({exc}); the next release builds its PTX set")
            else:
                self.say("  NOTE no PTX leg named for the shipped PTX set; the next release builds its PTX set")
        return rec

    def evidence_set_context(self, published, stage):
        """tools/release_evidence_set.py's view of this release (paths, state, wheels; nothing is run)."""
        import release_evidence_set
        date = changelog_date(ROOT, self.version, text=file_at(self.commit, "CHANGELOG.md") or "") \
            or dt.date.today().isoformat()
        single = bool(self.single())
        gate_dirs = {}
        for vendor in GATE_VENDORS:
            if vendor in published:
                try:
                    gate_dirs[vendor] = self.gate_out(vendor)
                except (StepFailed, OSError, ValueError, KeyError, zipfile.BadZipFile) as exc:
                    self.say(f"  NOTE no gate directory for {vendor}: {exc}")
        admitted = self.state.get("identity_admitted")
        table, table_ref = None, ""
        admitted_table = self.rel / "identity-wheels" / "admitted-table.json"
        if single and admitted_table.is_file():
            table = admitted_table
            table_ref = "%s:%s" % ((admitted or {}).get("commit", "?"), "python/mojolearn/verify_reference/table.json")
        else:
            path = "python/mojolearn/verify_reference/table.json"
            text = file_at(self.commit, path)
            if text:
                table = stage.parent / "record-table.json"
                table.write_text(text)
                table_ref = "%s:%s" % (self.commit, path)
        wheels = {platform: wheel for platform, wheel, _ in self.platform_wheels() if wheel}
        git_version = subprocess.run(["git", "--version"], capture_output=True, text=True).stdout.strip()
        return release_evidence_set.Context(
            version=self.version, commit=self.commit, tooling_commit=self.tooling_commit, date=date, rel=self.rel,
            staged=stage, publish=self.args.publish, published=list(published), wheels=wheels, single=single,
            gate_dirs=gate_dirs, steps=dict(self.state.get("steps") or {}), identity_admitted=admitted,
            adapters=os.environ.get("MOJOLEARN_RELEASE_ADAPTERS", "1") != "0", table=table, table_ref=table_ref,
            box_tags=self.box_job_tags(),
            tools=dict(git=git_version, release_py_sha256=sha256(Path(__file__)), tooling_commit=self.tooling_commit))

    def publish_step(self, platform):
        """The publish step of the pipeline whose platform is `platform`."""
        return next(p["publish"] for p in self.PIPELINES.values() if p["platform"] == platform)

    def platform_wheels(self):
        """[(platform, wheel, smoke receipt)] of every wheel this release publishes."""
        rows = []
        for platform in SPLIT_PACKAGES:
            # the Linux mojolearn has no receipt (no smoke installs it): none, not NVIDIA's
            smoke = (self.recorded(self.publish_step(platform)) or {}).get("smoke") or (
                self.rel / ("column-amd" if platform == "amd" else "smoke-linux") / "results.json"
                if platform != "linux-meta" else self.rel / "no-smoke-linux-meta.json")
            rows.append((platform, self.split_final(platform), Path(smoke)))
        return rows + [("macos", self.macos_wheel(), self.rel / "smoke-macos" / "results.json" if self.gpu_smoke()
                        else self.macos_self_test())]

    # ------------------------------------------------------------ the run
    # ------------------------------------------------------------ done steps short-circuit
    # DONE STEPS SHORT-CIRCUIT (lane release-no-hands, 2026-11-04). A step recorded done at this
    # commit whose output still checks out returns its "have ..." result BEFORE its method runs, so no
    # refusal or recomputation at the top of a step can stop a resumed release (0.8.65: the publish
    # run re-executed a done linux-assemble, hit the refusal at the top of the method, and `--only`
    # juggling was the workaround). run_step asks have_<step>(rec): a string means "have it", None or a
    # raised error means "run the step", whose own checks then decide. A step with no have_<step> and
    # not in SKIP_IF_RECORDED (the freeze steps, which check the frozen source) runs as before.
    HAVE_ERRORS = (StepFailed, OSError, ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile,
                   subprocess.SubprocessError)

    def have(self, step, rec):
        """The "have ..." result of STEP recorded done (REC) at this commit, or None to run it."""
        check = getattr(self, "have_" + step.replace("-", "_"), None)
        if rec is None or check is None:
            return None
        try:
            got = check(rec)
        except self.HAVE_ERRORS:
            return None
        return str(got) if got else None

    def linux_packed(self):
        """Every final split wheel of this commit is there (linux-pack's output)."""
        finals = self.split_finals()
        return bool(finals) and all(finals.values()) and wheel_commit(finals["linux"]) == self.commit

    def have_reuse_plan(self, rec):
        doc = read_json(self.plan_path)
        if (doc and doc.get("schema") == release_reuse.PLAN_SCHEMA and doc.get("commit") == self.commit
                and release_reuse.record_modes(doc) == self.release_modes
                and (doc.get("builders_override") or {}) == self.builders_override()):
            return "have " + str(self.plan_path)
        return None

    def have_linux_builds(self, rec):
        if self.linux_packed():
            return "have the packed Linux wheels of this commit (the legs' output is packed)"
        if self.recorded("linux-wait") and self.have_linux_wait(self.recorded("linux-wait")):
            return "have every leg built and verified (linux-wait done " + self.recorded("linux-wait")["at"] + ")"
        return None

    def have_linux_wait(self, rec):
        if self.linux_packed():
            return "have the packed Linux wheels of this commit (the legs' output is packed)"
        self.host_composition()  # a changed composition receipt raises: the step itself decides
        legs = [l for l in linux_legs(self) if not self.admitted_leg(l.name)]
        if all(l.done(self.commit) for l in legs):
            return "have every leg done at this commit: " + (", ".join(l.name for l in legs) or "none to build")
        return None

    def have_linux_assemble(self, rec):
        if self.linux_packed():
            return "have the packed Linux wheels of this commit (the assembled sets are packed)"
        if not self.assembly_needed():
            return "not needed: every Linux binding is built by the legs"
        return ("have " + str(self.assembled_marker().parent)) if self.assembled_ok() else None

    def have_linux_pack(self, rec):
        return ("have " + ", ".join(w.name for w in self.split_finals().values())) if self.linux_packed() else None

    def have_macos_build(self, rec):
        w = self.macos_wheel()
        return ("have " + w.name) if w and wheel_commit(w) == self.commit else None

    def have_macos_smoke(self, rec):
        if not self.gpu_smoke():
            return "WHEEL-SELF-TEST PASSED (receipt exists)" if self.macos_self_test_passed() else None
        w = self.macos_wheel()
        return "PASSED (receipt exists)" if w and smoke_passed(self.rel / "smoke-macos" / "results.json", w) else None

    def have_gate_apple(self, rec):
        if not adapter_gate_module().enabled():
            return "skipped (MOJOLEARN_RELEASE_ADAPTERS=0: no adapters ship)"
        ok, why = self.apple_adapters_ok()
        return (why + " (receipt exists)") if ok else None

    def have_column(self, name):
        if not self.gpu_smoke():
            return self.NO_GPU_SKIP
        ok = next(ok for n, _, ok, _ in self.column_specs() if n == name)
        return f"have the {name} column PASSED for this wheel" if ok() else None

    def have_gpu_column_nvidia(self, rec):
        return self.have_column("nvidia")

    def have_gpu_column_nvidia_hopper(self, rec):
        return self.have_column("nvidia-hopper") if self.hopper_required() or not self.gpu_smoke() else rec.get("result")

    def have_gpu_column_amd(self, rec):
        return self.have_column("amd")

    def have_gate_nvidia(self, rec):
        ok, why = self.gate_ok("nvidia")
        return why if ok else None

    def have_gate_amd(self, rec):
        ok, why = self.gate_ok("amd")
        return why if ok else None

    def run_step(self, step):
        """Run one step; 'done', 'failed' or 'held'. Never raises in a real run:
        a crash is a failed step, reported, so the other pipeline goes on."""
        rec = self.recorded(step)
        if rec and self.skip_recorded(step, rec):
            self.say(f"-- {step}: done {rec['at']} ({rec.get('result', '')})")
            return "done"
        have = self.have(step, rec)
        if have:
            self.say(f"-- {step}: done {rec['at']}: {have}")
            if step == "gate-boxes":
                self.keep_boxes()
            return "done"
        if step in BOX_CONSUMERS:
            self.keep_boxes()   # a later run's box steps keep the boxes an earlier run brought up
        self.say(f"-- {step}")
        shown = self.commands_shown
        started = time.monotonic()   # waste.json: per-step seconds (tools/release_plan.py)
        try:
            out = getattr(self, "step_" + step.replace("-", "_"))()
            result, data = out if isinstance(out, tuple) else (out, {})
            if self.dry and self.commands_shown > shown:
                result = "pending: would run the command(s) above"
        except StepHeld as exc:
            if self.dry:
                self.say(f"   A REAL RUN WOULD STOP HERE: {exc}")
                return "done"
            self.say(f"   HELD at {step}: {exc}")
            self.fail(step, exc, held=True)
            return "held"
        except StepFailed as exc:
            if self.dry:
                self.say(f"   A REAL RUN WOULD STOP HERE: {exc}")
                return "done"
            self.say(f"   STOPPED at {step}: {exc}")
            self.fail(step, exc)
            return "failed"
        except (Exception, SystemExit) as exc:
            if self.dry:
                raise
            self.say(f"   CRASHED at {step}: {exc!r}\n" + traceback.format_exc(limit=6))
            self.fail(step, f"{type(exc).__name__}: {exc}")
            return "failed"
        self.say(f"   {result}")
        if not self.dry:
            data = dict(data)
            data.setdefault("seconds", round(time.monotonic() - started, 1))
            self.mark(step, result=result, **data)
        return "done"

    def schedule(self, selected):
        """THE PIPELINES, AT ONCE. A step starts when every step it needs is
        done (and its resource is free); a failed, held or blocked need blocks
        it, and nothing else. Returns {step: done|failed|held|blocked|skipped}."""
        status = {s: ("pending" if s in selected else "skipped") for s in self.STEPS}
        self._status = status   # the box keeper reads which box steps are still pending
        busy = set()
        events = queue.Queue()

        def worker(step):
            self._tl.tag = self.PIPELINE_OF[step] if self.PIPELINE_OF[step] in self.PIPELINES else ""
            try:
                events.put((step, self.run_step(step)))
            except BaseException as exc:  # run_step never raises; this is a last resort
                events.put((step, "failed"))
                self.errors[step] = repr(exc)

        while True:
            changed = True
            while changed:
                changed = False
                for s in self.STEPS:
                    if status[s] != "pending":
                        continue
                    bad = [d for d in self.NEEDS[s] if status[d] in ("failed", "held", "blocked")]
                    if bad:
                        status[s] = "blocked"
                        self.blocked_by[s] = bad[0]
                        changed = True
            for s in self.STEPS:
                if status[s] != "pending":
                    continue
                if any(status[d] not in ("done", "skipped") for d in self.NEEDS[s]):
                    continue
                if any(status[d] in ("pending", "running") for d in self.AFTER.get(s, ())):
                    continue
                res = self.RESOURCE[s]
                if res and res in busy:
                    continue
                status[s] = "running"
                if res:
                    busy.add(res)
                threading.Thread(target=worker, args=(s,), name="release-" + s, daemon=True).start()
            if not any(v == "running" for v in status.values()):
                if any(v == "pending" for v in status.values()):
                    raise RuntimeError("release: a step can never start: " + ", ".join(
                        s for s, v in status.items() if v == "pending"))
                return status
            step, outcome = events.get()
            status[step] = outcome
            busy.discard(self.RESOURCE[step])

    def report(self, status):
        """Both outcomes, one line each, and what a rerun does."""
        self.say("== pipelines")
        for s in self.STEPS:
            if self.PIPELINE_OF[s] == "common" and status[s] in ("failed", "held"):
                self.say(f"   common: {status[s].upper()} at {s}: {self.errors.get(s, '')}")
        lines = []
        for name, p in self.PIPELINES.items():
            steps = [s for s in self.STEPS if self.PIPELINE_OF[s] == name] + \
                [d for s in self.STEPS if self.PIPELINE_OF[s] == name for d in self.NEEDS[s]
                 if self.PIPELINE_OF[d] not in (name, "common")]
            pub = p["publish"]
            if status[pub] == "done" and self.recorded(pub):
                line = f"{name}: PUBLISHED ({self.recorded(pub).get('result', '')})"
            elif status[pub] == "done":
                line = f"{name}: done through {pub}"
            else:
                order = [s for s in self.STEPS if s in steps]
                first = next((s for s in order if status[s] in ("failed", "held")), None)
                blocked = next((s for s in order if status[s] == "blocked"), None)
                if first:
                    line = f"{name}: {status[first].upper()} at {first}: {self.errors.get(first, '')}"
                elif blocked:
                    by = self.blocked_by.get(blocked)
                    line = f"{name}: BLOCKED at {blocked}, which needs {by} ({status.get(by)})"
                else:
                    line = f"{name}: not published ({pub} {status[pub]})"
            lines.append(line)
            self.say("   " + line)
        for s in [s for s in self.STEPS if self.PIPELINE_OF[s] == "finish"]:
            if status[s] in ("done",) and self.recorded(s):
                self.say(f"   {s}: {self.recorded(s).get('result', '')}")
            elif status[s] not in ("skipped",):
                self.say(f"   {s}: {status[s]}" + (f": {self.errors[s]}" if s in self.errors else ""))
        ok = all(v in ("done", "skipped") for v in status.values())
        if not ok:
            self.say(f"   Rerun `pixi run release {self.version}"
                     + (f" --publish {self.args.publish}" if self.args.publish else "")
                     + "`: it resumes only what is not done (`--status` shows what that is).")
        return ok

    def banner(self):
        c = self.state.get("commit")
        head = git_at(ROOT, "rev-parse", "HEAD") or "?"
        self.say(f"   state {self.state_path}; source commit {c or '(not frozen yet: HEAD ' + head[:12] + ')'}")
        if c and head != c:
            self.say(f"   tooling: this checkout {ROOT} at {head[:12]}; the source is read from "
                     f"{self.source_checkout(create=False)}")

    # ------------------------------------------------------------ --plan (tools/release_plan.py)
    #: release step -> (gate kind in the decision table, device it runs on)
    PLAN_STEP_KINDS = {
        "freeze-version": ("metadata", None), "freeze-changelog": ("metadata", None),
        "freeze-docs-facts": ("metadata", None), "freeze-commit": ("metadata", None),
        "reuse-plan": ("metadata", None), "finish-line": ("metadata", None), "record": ("metadata", None),
        "rehearsal": ("package", None), "linux-wait": ("package", None), "linux-assemble": ("package", None),
        "linux-pack": ("package", None),
        "cross-compile": ("artifact", None), "linux-builds": ("artifact", None), "macos-build": ("artifact", None),
        "macos-smoke": ("loading", None),
        "gate-apple": ("numerical", None), "gate-nvidia": ("numerical", "nvidia"), "gate-amd": ("numerical", "amd"),
        "gpu-column-nvidia": ("numerical", "nvidia"), "gpu-column-nvidia-hopper": ("numerical", "nvidia"),
        "gpu-column-amd": ("numerical", "amd"),
        "published-apple": ("package", None), "published-nvidia": ("package", None),
        "published-amd": ("package", None), "identity-at-freeze": ("metadata", None),
        "testpypi-rehearsal": ("package", None),
        "identity-cpu": ("numerical", None), "identity-admit": ("package", None),
    }
    #: identity column -> the box its record runs on (Apple records on this Mac: never rented)
    PLAN_COLUMN_DEVICE = {"nvidia": "nvidia", "amd": "amd", "cpu": "nvidia", "ptx": "nvidia", "apple": None}

    def plan_json_path(self):
        return Path(getattr(self.args, "plan_json", "") or (self.rel / "plan.json"))

    def saved_plan(self):
        try:
            doc = read_json(self.plan_json_path())
        except AttributeError:   # a partial Release (tests, a publisher called directly): no release dir
            return None
        return doc if doc and doc.get("schema") == release_plan_module().SCHEMA and doc.get("commit") == self.commit \
            else None

    def plan_blocks_publication(self):
        """Rows of this commit's saved plan that still hold publication NOW: an unresolved failure whose step
        is not recorded since, and evidence to restore that is still not on this machine. Owed work is
        enforced by the step graph itself. No saved plan: nothing extra (the ledger gate still applies)."""
        plan = self.saved_plan()
        if not plan:
            return []
        held = []
        restore = {r["id"]: r for r in plan.get("restore") or []}
        for row in (plan.get("publish") or {}).get("blocked") or []:
            if row["reason"].startswith("unresolved failure") and not self.recorded(row["id"]):
                held.append(f"{row['id']}: {row['reason']}")
            elif row["id"] in restore and not Path(restore[row["id"]].get("local_path") or "/nonexistent").exists():
                held.append(f"{row['id']}: {row['reason']}")
        return held

    def plan_lane_facts(self, previous, columns):
        """{column: {lane: facts}} from tools/identity_columns.py changed-lanes, the selection each column's
        recorder makes (reference_refresh.SELECTION), over the supported surface; static, no box."""
        import reference_refresh as rr
        rp = release_plan_module()
        out, notes = {}, []
        work = self.rel / "plan"
        work.mkdir(parents=True, exist_ok=True)

        def is_ancestor(a, b):
            return subprocess.run(["git", "-C", str(self.src), "merge-base", "--is-ancestor", a, b],
                                  capture_output=True).returncode == 0
        for col in columns:
            if col not in rr.SELECTION:
                notes.append(f"identity column {col}: unknown (columns: {', '.join(rr.SELECTION)})")
                continue
            mode, backend = rr.SELECTION[col]
            summary = work / f"identity-{col}.json"
            cmd = [PY, "tools/identity_columns.py", "changed-lanes", "--scope", rr.SURFACE_SCOPE, "--backend", backend]
            if mode == "class":
                cmd += ["--class", col, "--records", str(Path(self.src) / "bench/results/identity_break")]
            cmd += ["--summary-json", str(summary)]
            with open(work / f"identity-{col}.log", "w") as log:
                rc = subprocess.run(cmd, cwd=str(self.src), stdout=subprocess.DEVNULL, stderr=log).returncode
            doc = read_json(summary)
            if rc != 0 or not doc:
                notes.append(f"identity column {col}: selection failed (rc={rc}; {work / f'identity-{col}.log'})")
                continue
            out[col] = rp.lane_facts(doc, previous, col, self.PLAN_COLUMN_DEVICE.get(col), is_ancestor)
        return out, notes

    def plan_facts(self):
        rp = release_plan_module()
        prev = (self.plan().get("previous") or {})
        previous = prev.get("source_commit")
        facts = dict(version=self.version, commit=self.commit, previous=previous, notes=[])
        facts["changes"] = rp.changed_paths(previous, self.commit, repo=self.src) if previous else []
        try:
            facts["groups"] = rp.group_facts(self.src, previous, self.commit)
        except Exception as exc:  # the plan says so; a group with no facts is never a hit
            facts["groups"] = {}
            facts["notes"].append(f"compile groups: no closure facts ({type(exc).__name__}: {exc})")
        columns = [c for c in (getattr(self.args, "plan_columns", "") or "").split(",") if c]
        facts["lanes"], notes = self.plan_lane_facts(previous, columns)
        facts["notes"] += notes
        prev_rel = (self.base.parent / str(prev.get("version")) / previous[:12]) if previous and prev.get("version") \
            else None
        gates = {}
        for step in self.selected_steps():
            if step not in self.PLAN_STEP_KINDS:
                continue
            k, device = self.PLAN_STEP_KINDS[step]
            st, detail = self.step_state(step)
            row = dict(kind=k, device=device, state="done" if st == "done" else ("failed" if st in ("failed", "held")
                                                                               else "owed"), reason=detail)
            if st != "done" and step in BOX_CONSUMERS:
                row["self_checked"] = True
                why = self.box_step_owed(step)
                row["reason"] = why or "nothing owed on its box"
                if not why:
                    row.update(state="done", device=None)
            if prev_rel is not None:
                evid = prev_rel / step
                row["previous_evidence"] = dict(local=evid.exists(), local_path=str(evid),
                                                retained=f"release ledger R2 receipts of {prev.get('version')}")
            gates[step] = row
        facts["gates"] = gates
        facts["failures"] = [dict(id=step, reason=f.get("error", ""), resolved=bool(self.recorded(step)))
                             for step, f in (self.state.get("failures") or {}).items()
                             if f.get("commit") == self.state.get("commit")]
        facts["leases"] = {"nvidia": self.HOLD_MAX_S // 60, "amd": self.AMD_LEASE_MIN}
        # the macOS build path, a key field: the previous release's (from its retained bincache keys) and ours
        prev_keys = sorted((prev_rel / "macos-bincache" / "provenance" / "keys").glob("*.json")) if prev_rel else []
        facts["macos_paths"] = dict(this=str(Path(self.src).resolve()),
                                    previous=(read_json(prev_keys[0]) or {}).get("repo_path") if prev_keys else None)
        facts["step_seconds"] = {s: r.get("seconds") for s, r in (self.state.get("steps") or {}).items()
                                 if isinstance(r, dict) and r.get("commit") == self.state.get("commit")
                                 and isinstance(r.get("seconds"), (int, float))}
        facts["previous_waste"] = rp.previous_waste(self.base.parent, self.version)
        return facts

    def release_plan(self):
        """--plan: gather the facts, apply the decision table (tools/release_plan.py), print one line per
        decision, write plan.json and plan-facts.json. Runs, rents and publishes nothing."""
        rp = release_plan_module()
        self.say(f"== mojolearn release {self.version} --plan (nothing runs, nothing is rented)")
        facts = self.plan_facts()
        plan = rp.decide(facts)
        plan["notes"] = facts.get("notes", [])
        out = self.plan_json_path()
        out.parent.mkdir(parents=True, exist_ok=True)
        (out.parent / "plan-facts.json").write_text(json.dumps(facts, indent=1, sort_keys=True, default=str) + "\n")
        # THE STATUS REPORT's totals line (tools/release_report.py), from the same inputs --report reads
        try:
            report = self.status_report(plan=plan)
            plan["report_totals"] = report["totals_line"]
        except Exception as exc:  # the plan still stands; the report says why it is missing
            plan["report_totals"] = None
            plan["notes"].append(f"status report unavailable ({type(exc).__name__}: {exc})")
        out.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        for line in rp.lines(plan) + ["note: " + n for n in plan["notes"]]:
            self.say(line)
        self.say(f"== plan.json: {out}")
        return 0 if plan["publish"]["status"] == "ready" else 3

    # ------------------------------------------------------------ --report (tools/release_report.py)
    def report_json_path(self):
        return Path(getattr(self.args, "report_json", "") or (self.rel / "report.json"))

    def report_waiver_paths(self):
        """--owner-waiver and the waivers committed at the frozen source (docs/release/<version>-*waiver*.json)."""
        paths = [getattr(self.args, "owner_waiver", None)]
        src = Path(getattr(self, "src", None) or ROOT)
        paths += sorted(str(p) for p in (src / "docs" / "release").glob(f"{self.version}-*waiver*.json"))
        return [p for p in paths if p]

    def report_inputs(self):
        """THE ONE LOADER of the status report (--report, and the totals line --plan embeds): the evaluated
        ledger rows (read-only, the rows --status and the gate read), every release step's state, the
        waivers and the saved plan. Writes nothing."""
        notes = []
        try:
            rows = self.readiness_status() if self.state.get("commit") else self.readiness().plan()
        except Exception as exc:   # an unreadable ledger is a provenance problem, never a silent pass
            rows = {"requirements-ledger": dict(status="untested", action="complete-provenance", stage="pre",
                                                required=True, evidence=[], reasons=[
                                                    f"ledger unreadable: {type(exc).__name__}: {exc}"])}
            notes.append(f"ledger unreadable ({type(exc).__name__}: {exc})")
        smoke = set(getattr(self.args, "published_smoke", None) or ())
        steps = {}
        for step in self.selected_steps():
            if step in PUBLISHED_SMOKE_STEPS and PUBLISHED_SMOKE_STEPS[step][0] not in smoke:
                continue
            st, detail = self.step_state(step)
            steps[step] = (st, detail, st == "held")
        waivers = release_report_module().waivers_from(self.report_waiver_paths(), self.state)
        return dict(rows=rows, steps=steps, waivers=waivers, plan=self.saved_plan(), notes=notes)

    def status_report(self, plan=None):
        rr = release_report_module()
        inputs = self.report_inputs()
        report = rr.build(self.version, self.commit, inputs["rows"], inputs["steps"], inputs["waivers"],
                          plan if plan is not None else inputs["plan"],
                          ledger_path=str(self.readiness().path))
        report["notes"] = inputs["notes"]
        return report

    def release_report(self):
        """--report: one row per ledger requirement and release step, each in exactly one of the seven states
        (docs/release/STATUS_REPORT.md), with its evidence, producing step and next action; writes
        report.json and report.md. Runs, rents and publishes nothing. Exit 0 when ready, 3 otherwise."""
        rr = release_report_module()
        self.say(f"== mojolearn release {self.version} --report (nothing runs, nothing is rented)")
        report = self.status_report()
        js, md = rr.write(report, self.report_json_path())
        for line in rr.lines(report) + ["note: " + n for n in report["notes"]]:
            self.say(line)
        self.say(f"== report.json: {js}; report.md: {md}")
        return 0 if report["ready"] else 3

    def write_waste(self):
        """waste.json beside the state after a real run (minimal: the saved plan's numbers + step seconds)."""
        if self.dry or getattr(self, "readonly", False):
            return None
        rp = release_plan_module()
        plan = self.saved_plan() or {}
        steps = {s: r.get("seconds") for s, r in (self.state.get("steps") or {}).items()
                 if isinstance(r, dict) and r.get("commit") == self.state.get("commit")
                 and isinstance(r.get("seconds"), (int, float))}
        waste = dict(plan.get("waste") or dict(schema=rp.WASTE_SCHEMA, version=self.version, commit=self.commit))
        waste.update(steps=steps, box_wait_seconds=steps.get("gate-boxes"), admission_seconds=steps.get("record"))
        try:
            return rp.write_waste(self.rel / "waste.json", waste)
        except OSError as exc:
            self.say(f"  waste.json not written: {exc}")
            return None

    def go(self):
        if getattr(self.args, "status", False):
            return self.status()
        if getattr(self.args, "plan", False):
            return self.release_plan()
        if getattr(self.args, "report", False):
            return self.release_report()
        steps = self.STEPS
        if self.args.only:
            wanted = self.args.only.split(",")
            unknown = [s for s in wanted if s not in steps]
            if unknown:
                raise SystemExit("release: unknown step(s) " + ", ".join(unknown) + "; steps: " + ", ".join(steps))
            steps = [s for s in steps if s in wanted]
        # --published-smoke is refused (removed 2026-11-05: no identity lane after the first publish)
        smoke = self.record_published_smoke_requests()
        if self.single() and self.gpu_smoke():
            raise SystemExit("release: --identity-on-wheels replaces the --gpu-smoke wheel columns (one run, nothing "
                             "twice); drop --gpu-smoke")
        steps = [s for s in steps if s not in PUBLISHED_SMOKE_STEPS or PUBLISHED_SMOKE_STEPS[s][0] in smoke]
        self.selected = list(steps)
        for redo in (self.args.redo.split(",") if self.args.redo else []):
            if redo not in self.STEPS:
                raise SystemExit("release: unknown --redo step " + redo)
            reason = (getattr(self.args, "rerun_reason", None) or "").strip()
            if not reason:
                raise SystemExit("release: --redo requires --rerun-reason naming the invalidated evidence or failed check")
            self.state.setdefault("rerun_requests", []).append(dict(step=redo, reason=reason, at=now()))
            self.state["steps"].pop(redo, None)
        if self.args.redo:
            self.save()
        self.say(f"== mojolearn release {self.version}{' (DRY RUN: nothing runs, nothing is rented)' if self.dry else ''}")
        self.banner()
        if self.dry:
            for step in steps:
                self.run_step(step)
            self.say("== end of plan (the " + " and ".join(self.PIPELINES)
                     + " pipelines run at once in a real run; see --status)")
            return 0
        ignore_hangup()
        try:
            status = self.schedule(steps)
        finally:
            self.stop_keeper()
            self.write_waste()
        ok = self.report(status)
        self.say("== done" if ok else "== stopped")
        return 0 if ok else 1

    # ------------------------------------------------------------ --status
    def step_state(self, step):
        rec = self.recorded(step)
        if rec:
            return "done", f"{rec['at']}: {rec.get('result', '')}"
        fail = (self.state.get("failures") or {}).get(step)
        if fail and fail.get("commit") == self.state.get("commit"):
            return ("held" if fail.get("held") else "failed"), f"{fail['at']}: {fail['error']}"
        return "owed", ""

    def rerun_action(self, step, st, states):
        if st == "done":
            if step in ("finish-line", "record"):
                rec = self.recorded(step) or {}
                if sorted(rec.get("platforms", [])) != sorted(self.published_platforms()):
                    return "run it again for the newly published platform(s)"
            return "skip"
        missing = [d for d in self.NEEDS[step] if states.get(d) != "done"]
        if st == "held":
            return "run it again (with --publish)"
        if st == "failed":
            return "run it again" + (f" after {', '.join(missing)}" if missing else "")
        return "run it" + (f" after {', '.join(missing)}" if missing else "")

    def leg_status(self, name):
        vendor, arch = name.split("-", 1)
        rb, out = leg_layout(self.rel / "legs", self.args.build_backend, name)
        leg = Leg(name, vendor, arch, [], {}, rb, self.rel / "legs", out)
        if leg.running():
            return "running", f"pid {leg.pid()}, log {leg.log}", "wait for it"
        if leg.done(self.commit):
            ok, why = overlay_verified(read_json(leg.provenance_file), leg.out_dir)
            if ok:
                return "done", f"exit 0, proof of {self.commit[:12]}", "skip"
            return "failed", why, "relaunch it"
        doc = self.admitted_leg(name)
        if doc:
            return ("taken", f"built from {doc['built_from'][:12]} ({doc['provenance']}), admitted for "
                    f"{doc['admitted_for'][:12]}: {doc['because']}", "skip")
        if leg.exit_code() is not None:
            return "failed", f"exit {leg.exit_code()}, log {leg.log}", "relaunch it (the failed attempt is moved aside)"
        doc, why = self.find_reusable_leg(name, vendor, arch, write=False)
        if doc:
            return "owed", "", f"take the completed leg built from {doc['built_from'][:12]} ({doc['provenance']})"
        return "owed", f"no reusable leg: {why}", "launch it"

    def column_status(self, name, vendor, ok, out):
        work = self.rel / "columns"
        leg = ColumnLeg(name, vendor, [], work, out, ok)
        final = self.linux_final()
        if ok():
            return "done", f"wheel smoke PASSED for wheel {sha256(final)[:12]}", "skip"
        if leg.running():
            return "running", f"pid {leg.pid()}, log {leg.log}", "wait for it"
        if leg.exit_code() is not None:
            return "failed", f"exit {leg.exit_code()}, log {leg.log}", "relaunch it (the failed attempt is moved aside)"
        return "owed", "" if final else "no final Linux wheel yet", "launch it" if final else "launch it after linux-pack"

    def status(self):
        """--status: per pipeline, per leg and column, done / failed (with its
        log) / owed, and what a rerun would do. Runs nothing and writes nothing."""
        say = self.say
        c = self.state.get("commit")
        head = git_at(ROOT, "rev-parse", "HEAD") or "?"
        say(f"== mojolearn release {self.version}: status (read-only)")
        say(f"   state {self.state_path}" + ("" if self.state_path.is_file() else " (no state yet)"))
        if c or self.readiness().path.exists():
            try:
                rows = self.readiness_status() if c else self.readiness().plan()
                counts = {s: sum(row["status"] == s for row in rows.values())
                          for s in ("passed", "failed", "untested")}
                say(f"   requirements ledger {self.readiness().path}: {counts}")
                assessed = sorted({row["source_revision"] for row in rows.values()})
                say("   assessed source: " + ", ".join(assessed))
                if not c and head not in assessed:
                    say("   HEAD has not been reconciled with this saved ledger; these results apply to the assessed source")
                families = {}
                for name, result in rows.items():
                    family = result.get("parent_requirement", name)
                    totals = families.setdefault(family, dict(passed=0, failed=0, untested=0))
                    totals[result["status"]] += 1
                for family, totals in families.items():
                    say(f"      {family}: {totals}")
                retained = sum(1 for result in rows.values() if result["status"] == "passed"
                               and result.get("tested_revision") not in (None, result["source_revision"]))
                say(f"   passed from evidence of an earlier source (retained, dependency-proven): {retained}")
                ready, blocked = self.waived(release_requirements.gate_rows(rows, "pre"))
                say("   pre-publication gate: " + ("READY" if ready else f"BLOCKED by {len(blocked)} required row(s)"))
                # Every row that still lacks evidence, grouped by family and reason.
                groups = {}
                for name, result in rows.items():
                    if result["status"] == "passed" or not result["required"] or result["stage"] != "pre":
                        continue
                    family = result.get("parent_requirement", name.split("/")[0])
                    reasons = result["reasons"] or ["(no reason recorded)"]
                    diffs = [r for r in reasons if r.startswith(("changed input:", "added input:", "removed input:"))]
                    reason = next((r for r in reasons if r not in diffs), reasons[0])
                    if diffs:
                        reason += " (inputs differ from the tested source)"
                    # One group per cause, not per lane: the lane is in the row name.
                    reason = re.sub(r"execution_environment/[^/]+/", "execution_environment/<lane>/", reason)
                    groups.setdefault((family, result["status"], result["action"], reason[:300]), []).append(name)
                for (family, st, action, reason), names in sorted(groups.items()):
                    say(f"      {family}: {len(names)} {st} / {action}: {reason}")
                    say("         e.g. " + ", ".join(names[:3]) + (f" (+{len(names) - 3})" if len(names) > 3 else ""))
            except (OSError, ValueError, subprocess.CalledProcessError) as error:
                say(f"   requirements ledger untested: {error}")
        if c:
            say(f"   source commit {c} (pinned at freeze; only --refreeze moves it)")
            say(f"   tooling: {ROOT} at {head[:12]}" + (" (the source commit)" if head == c else ""))
            if head != c:
                say(f"   source checkout: {self.source_checkout(create=False)}")
        else:
            say(f"   not frozen yet: a run freezes HEAD {head[:12]}")
        try:
            t = self.tooling()
            if t["dirty"]:
                say("   build tooling has uncommitted changes (no leg is reused across freezes): " + ", ".join(t["dirty"]))
        except Exception:
            pass
        states = {s: self.step_state(s)[0] for s in self.STEPS}
        actions = []

        def row(step, st, detail, action, indent="   "):
            say(f"{indent}{step:<18} {st:<8} {detail}")
            if action not in ("skip",):
                say(f"{indent}{'':<18} {'':<8} rerun: {action}")
                actions.append(f"{step}: {action}")

        groups = [("common", "common")] + [(p, p + " pipeline") for p in self.PIPELINES] + [("finish", "finish")]
        for group, title in groups:
            say(f"-- {title}")
            for step in self.STEPS:
                if self.PIPELINE_OF[step] != group:
                    continue
                st, detail = self.step_state(step)
                row(step, st, detail, self.rerun_action(step, st, states))
                if step == "linux-builds" and c:
                    plan = read_json(self.plan_path)
                    if not plan or plan.get("commit") != c:
                        say("      (the legs are decided by the reuse plan, not made for this commit yet)")
                    elif not plan.get("legs"):
                        say("      no build leg: every Linux binding is taken from the published wheel")
                    else:
                        for name in plan["legs"]:
                            row(name, *self.leg_status(name), indent="      ")
                shown = {"gpu-column-nvidia": ("nvidia",), "gpu-column-amd": ("amd",)}.get(step)
                if shown and c and self.gpu_smoke():
                    for name, vendor, ok, out in self.column_specs():
                        if name in shown:
                            row(name, *self.column_status(name, vendor, ok, out), indent="      ")
        summary = []
        for name, p in self.PIPELINES.items():
            steps = [s for s in self.STEPS if self.PIPELINE_OF[s] == name or any(
                s in self.NEEDS[t] and self.PIPELINE_OF[t] == name for t in self.STEPS)]
            bad = next((s for s in steps if states[s] in ("failed", "held")), None)
            if states[p["publish"]] == "done":
                summary.append(f"{name}: published")
            elif bad:
                summary.append(f"{name}: {states[bad]} at {bad}")
            else:
                summary.append(f"{name}: owed")
        say("== " + ", ".join(summary))
        say(f"   ledger: {self.ledger().where}")
        if actions:
            say("   a rerun would: " + "; ".join(actions[:12]) + (" ..." if len(actions) > 12 else ""))
        else:
            say("   a rerun would do nothing: everything is done")
        return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("version")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and each step's status; run nothing")
    ap.add_argument("--plan", action="store_true",
                    help="THE RELEASE PLAN (tools/release_plan.py): one decision with a reason per compile group, "
                         "identity lane x column, gate/producer, box and publish row; writes plan.json; runs nothing")
    ap.add_argument("--plan-json", default="", help="where --plan writes plan.json (default <release>/<commit12>/plan.json)")
    ap.add_argument("--plan-columns", default="nvidia,amd,cpu,ptx,apple",
                    help="identity columns --plan selects (tools/identity_columns.py changed-lanes per column)")
    ap.add_argument("--report", action="store_true",
                    help="THE STATUS REPORT (tools/release_report.py): one row per ledger requirement and release "
                         "step, each passed / reused / failed / missing-execution / blocked-provenance / "
                         "not-applicable / waived, with evidence and the next action; writes report.json and "
                         "report.md; runs nothing")
    ap.add_argument("--report-json", default="", help="where --report writes report.json (default <release>/<commit12>/report.json)")
    ap.add_argument("--status", action="store_true",
                    help="per pipeline, leg and column: done / failed (with its log) / owed, and what a rerun "
                         "would do; runs nothing, writes nothing")
    ap.add_argument("--publish", choices=PUBLISH_CHOICES,
                    help="publish the wheels: none = the workflow's checks without uploading; testpypi = publish "
                         "to TestPyPI only; pypi = rehearse every package on TestPyPI first (testpypi-rehearsal: "
                         "upload, resolve, hash, macOS install + import), then publish to PyPI and compare "
                         "hashes only")
    ap.add_argument("--only", default="", help="comma-separated step names to run (others are skipped)")
    ap.add_argument("--redo", default="", help="comma-separated steps whose record is discarded first")
    ap.add_argument("--rerun-reason", help="specific invalidation/failure reason required with --redo")
    ap.add_argument("--owner-waiver", metavar="JSON",
                    help="owner-approved waiver: {prefixes, reason, approved_by}; matching blocked ledger rows are "
                         "reported WAIVED and logged in the release state, every other row still blocks")
    ap.add_argument("--requirements-configuration", metavar="JSON",
                    help="explicit qualification inputs (CPU target, full-workload recipes); retained in the release ledger")
    ap.add_argument("--refreeze", action="store_true",
                    help="move the frozen source commit to HEAD (the only way it moves); completed legs and "
                         "columns are kept where what they built or tested is unchanged")
    ap.add_argument("--source-checkout", default="",
                    help="an existing checkout at the frozen source commit (default: this checkout when its HEAD "
                         "is the source commit, else a worktree under the release directory)")
    # default: github for either layout (default_build_backend); cpu-box and gpu-legs by name
    ap.add_argument("--build-backend", default=None, choices=sorted(BUILD_BACKENDS))
    # WHY NO GPU BY DEFAULT (Andrew 2026-10-13): identity and behavior are checked before merge on NVIDIA,
    # AMD, Apple and CPU; the release only builds, checks artifact integrity and publishes.
    ap.add_argument("--identity-on-wheels", action="store_true",
                    help="THE ONE RUN (default off until the orchestrator has run it once): identity recorded from "
                         "the built wheels on each device class (NVIDIA/AMD box jobs, Apple in gate-apple, CPU on "
                         "GitHub), admitted, the table repacked into the wheels, then publish; no gate repeats a lane")
    ap.add_argument("--reuse-unchanged", action="store_true",
                    help="with --identity-on-wheels: OPT IN to reusing unchanged lanes, check units and runtime units "
                         "(the selector decides). Default off: the one run runs every lane, unit and runtime unit")
    ap.add_argument("--nvidia-columns", default="",
                    help="with --identity-on-wheels: the NVIDIA columns of the box job, comma list of nvidia, ptx "
                         "(default nvidia,ptx); each on its own GPU when the pod has several")
    ap.add_argument("--gpu-smoke", action="store_true",
                    help="the old GPU route: the NVIDIA/AMD wheel smokes (lq or rented), the expanded macOS smoke, "
                         "light-smoke receipts and any --build-backend; default: no GPU (GitHub builds, the "
                         "preverified publication route, the macOS verify --self-test gate)")
    # Inert since 2026-10-10 (Andrew 2026-10-10: identity runs ONCE; a mismatch is a bug to fix, never a reason to rerun): the
    # Apple cross-diff and release-check they steered are deleted; kept so
    # existing invocations still parse.
    ap.add_argument("--published-smoke", action="append", default=[], choices=["apple", "nvidia", "amd"],
                    metavar="VENDOR",
                    help="REMOVED 2026-11-05 (refused): no release step runs identity lanes on a device after "
                         "the first publish")
    ap.add_argument("--published-smoke-reason", default="",
                    help="REMOVED 2026-11-05 with --published-smoke")
    ap.add_argument("--accept-moved", action="store_true", help="inert since 2026-10-10")
    ap.add_argument("--cpu-column", action="store_true", help="inert since 2026-10-10")
    ap.add_argument("--amd-expect-from", default="",
                    help="an NVIDIA release-build dir: run the AMD core-host probe against its STAGED copy")
    ap.add_argument("--smoke-via", default="lq", choices=["lq", "rent"],
                    help="lq (default): the NVIDIA and AMD columns run on the nv and amd boxes we hold, queued next "
                         "with `lq add --front` (tools/release_lq_smoke.py); rent: the old per-architecture rented smoke")
    ap.add_argument("--hopper-box", default="",
                    help="ssh target of a Hopper (sm_90a) box we hold; without it the Hopper column is SKIPPED and "
                         "the sm_90a payload publishes unsmoked on its GitHub build receipt (--smoke-via lq)")
    ap.add_argument("--smoke-branch", default="main",
                    help="the branch whose tree runs the smoke on the lq box (default main)")
    ap.add_argument('--gate-autograd-root', help='precompiled gate artifacts under nvidia/ and amd/, each with build-manifest.json (default release-dir/autograd-bin)')
    ap.add_argument("--smoke-gpu", default="", help="RunPod GPU(s) for the Linux smoke, |-separated, walked on no stock (default the 4090, L40S, L40, RTX 6000 Ada)")
    _amd_env = os.environ.get("MOJOLEARN_AMD_PROVIDER") or "auto"
    ap.add_argument("--amd-provider", default=_amd_env if _amd_env in ("auto", "runpod", "hotaisle", "do") else "auto",
                    choices=["auto", "runpod", "hotaisle", "do"],
                    help="where the AMD column rents: auto = RunPod MI300X, then Hot Aisle MI300X, then DigitalOcean "
                         "MI325X (default MOJOLEARN_AMD_PROVIDER, else auto)")
    ap.add_argument("--amd-build-provider", default=None, choices=list(AMD_BUILD_PROVIDERS),
                    help="where the AMD build leg rents: auto = DigitalOcean MI325X, Hot Aisle 1x MI300X when "
                         "DigitalOcean has a GPU droplet live (default MOJOLEARN_AMD_PROVIDER, else auto)")
    ap.add_argument("--state-dir", default="", help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.report and (args.plan or args.status or args.dry_run or args.only or args.redo or args.refreeze):
        ap.error("--report runs nothing; it takes no --plan, --status, --dry-run, --only, --redo or --refreeze")
    if args.plan:
        if args.status or args.only or args.redo or args.refreeze:
            ap.error("--plan runs nothing; it takes no --status, --only, --redo or --refreeze")
        args.dry_run = True
    # Current releases focus on identical. Explicit legacy selections remain
    # available for reproducing older layouts, and are recorded in every plan.
    try:
        modes = release_reuse.release_modes(default=("identical",))
    except ValueError as exc:
        ap.error(str(exc))
    args.release_modes = modes
    if not VERSION_RE.match(args.version):
        ap.error("version must look like 0.8.15 (or 0.9.0a1)")
    if args.status and (args.dry_run or args.only or args.redo or args.refreeze):
        ap.error("--status runs nothing; it takes no --dry-run, --only, --redo or --refreeze")
    if not args.gpu_smoke and (args.hopper_box or args.smoke_via != "lq"):
        ap.error("--hopper-box and --smoke-via steer the GPU wheel smokes; they need --gpu-smoke")
    explicit = args.build_backend
    r = Release(args)  # resolves an unset --build-backend from the source's layout
    if modes != release_reuse.TIERS and args.build_backend not in ("cpu-box", "github"):
        ap.error("selected release tiers require --build-backend github or cpu-box; the GPU-leg backend "
                 "does not forward a tier selection")
    # Without --gpu-smoke a release rents no GPU: the Linux sets build on GitHub's free runners
    # (the default for either layout) or, by name, on RunPod CPU pods (no GPU either).
    no_gpu_ok = ("github", "cpu-box")
    if not args.gpu_smoke and args.build_backend not in no_gpu_ok:
        ap.error(f"--build-backend {args.build_backend} uses rented GPU boxes; without --gpu-smoke the Linux sets "
                 "build on GitHub's free runners (github) or RunPod CPU pods (cpu-box)"
                 + (f" (the default for this source is {default_build_backend(r.commit)})" if explicit else ""))
    return r.go()


if __name__ == "__main__":
    raise SystemExit(main())
