#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""A MEASUREMENT BUILD: WHEELS BUILT FROM ONE COMMIT TO MEASURE THEM, NEVER A RELEASE CANDIDATE (lane
candidate-build, 2026-11-09; docs/plans/PRE_0867_MEASUREMENT.md; AGENTS.md "Fix everything before a release":
it never touches main, tags or PyPI, and no release happens until everything is fixed and measured clean). The
subcommand keeps the name `candidate`.

    python3 tools/release.py candidate --commit <sha40> [--targets sm_89,sm_80] [--vendors nvidia,amd]
        [--defines "-D A=1 -D B=2"] [--label NAME] [--base-version 0.8.67] [--ref <pushed tooling branch>] [--dry-run]

Builds the Linux wheels of one exact commit with the release's own steps and nothing else: tools/release.py's
reuse-plan, linux-builds (.github/workflows/release-linux-build.yml on GitHub's free runners for that commit,
through tools/release_github_build.py run, with the workflow's `defines` when --defines is given), linux-wait,
linux-assemble and linux-pack (packaging/linux/pack_wheel.py --local-version, audit.sh, the directory-entry strip,
split_audit). The Release runs with `--only` those five steps, so the scheduler counts every other step as skipped:
freeze-version, freeze-changelog, freeze-docs-facts, freeze-commit (the version bump commit and the push to the
release branch), identity-at-freeze, rehearsal, readiness, cross-compile, every gate, column, publish, finish-line
and record never run. The source commit is pinned in the candidate's own state.json, so nothing is frozen.

THE VERSION is local, `<base>.dev0+<commit12>[.<label>]` (PEP 440; PyPI refuses local versions, so these wheels can
never be uploaded). It exists only in the packed wheels under the candidate's work dir: the packer generates the
wheel's _version.py and still proves the committed one at the build commit. No commit, no push, no tag, no PyPI.

THE WORK DIR is <work root>/<version>/ (default ~/mojolearn-evidence/release-candidates): its state.json, its own
local ledger (Release with --state-dir never opens the shared R2 ledger), the legs, and a detached worktree of the
commit at source/<commit12> (the only tree it writes). It refuses a work dir or source tree inside the main
checkout. The wheels land in <work dir>/<commit12>/linux/final/, with candidate.json beside them, the path
tools/ops/measure_candidate.py stage reads for a local --version.

DEFINES (the A/B "B" wheel sets): --defines (or --owed-arm) makes a candidate of the same commit whose compiles get
MOJOLEARN_BUILD_DEFINES. The label defaults to `d` + 8 hex of the defines, so each define set has its own version
and work dir. RUN THE A SET FIRST: a B arm plans against its BASE BUILD, the finished A set of the same commit and
targets at <work root>/<base>.dev0+<commit12>/ (release_reuse.base_build; lane arm-build-reuse, 2026-11-16). Every
binding whose source closure names none of the defines is REUSE from the base (a guard module that only refuses a
define still names it, so it reads it); the shard units (merged groups) holding a reading binding BUILD, every other
unit gets no shard job (release_github_build.py run --skip-units) and the assemble job takes its archive from the
binding cache, keyed exactly as the base's because tools/bincache.py scope_build_defines drops the defines a closure
does not name; a leg runs only for a set with a unit to build. The pack places the base wheel's bytes for every
REUSE row (release_reuse.assemble_linux; an NVIDIA rebuild of the same unit must reproduce them), so the arm's wheel
is byte for byte what a full arm build packs. candidate.json `base` records the base and, per set, the units built
and reused. Without a finished base every unit builds (printed). A dry run plans against a base still building
(pending). A define read by every unit is a full build and the dry run says FULL BUILD.

SEVERAL ARMS (`--owed-arm a --owed-arm b ...`, run_arms): one measurement build per distinct define set, all started
at once as separate processes (their GitHub runs queue together; each dispatches only its BUILD units plus one
assemble job per launched leg, so the free runners, about 20 at a time, are what bound the wall time, not a fixed
worker count). A dry run plans them in parallel, one process per arm up to the Mac's cores. Each prints one line:
ARM <name>: units BUILD / REUSE on its legs, rows BUILD / REUSE.

--vendors names the vendor wheels validated and staged (tools/candidate_wheel_install.py build_manifest). The
packer packs the core and the plugins of the targets from one set of legs, so the legs the plan needs always run.

--targets (2026-11-15; AGENTS.md part 8: measure on the NVIDIA L40S only) scopes the build to some Linux sets
(release_reuse.scope_linux_sets): `--targets sm_89,sm_80` plans, dispatches, assembles and packs only the NVIDIA
sets (and the host bindings their legs build), packs the core and the mojolearn-nvidia wheel, and no AMD or meta
wheel. sm_89 alone is refused, for B arms too: the NVIDIA wheel does not import without its PTX slot sm_80, even on
an sm_89 GPU, and a B arm's sm_80 units that read the define must carry it (the base's sm_80 bytes would not).
Default: every set, as a release. candidate.json records the targets; tools/ops/measure_candidate.py stage refuses a
vendor the build did not pack. A release has no --targets.

--timed-on (lane fx-build-tools, 2026-11-18; B arms with a finished base only): `--targets sm_89,sm_80 --timed-on
sm_89` for an arm timed on the NVIDIA L40S, which loads the native sm_89 set and never the PTX slot. The define's
device units build in sm_89 only; the sm_80 set ships the base's device bytes (compiled without the define) with
the arm's host bindings, so the wheel imports and no sm_80 leg runs (49 of the 112 d652a4b0a1 arm legs were sm_80).
candidate.json records `timed_on` and `untimed_from_base`; an A/B on any other GPU must not use such an arm.
Default (no --timed-on): every set is built with the define, as before.

PRE-BUILD CHECKS (2026-11-15, tools/prebuild_check.py): before the Release dispatches anything, the commit's pack
source audits (NumPy/platform math, verification inputs, source/API payload) and the host-parse suite of
native-checks.yml (every release host group parsed and elaborated on free runners, or a passed run of the same sha
reused) run in parallel; a failure refuses the build. --probe-wheel <core.whl> adds the probe part (an earlier core
wheel with this commit's Python on verify-smoke.yml). Logs: <work dir>/prebuild/. A resume reuses their PASS.json.
--no-prebuild-check skips them. A dry run does not run them.

At the end it asserts that origin/main and the tag lists (origin and local) are what they were at the start
(`git ls-remote`, three tries with a pause between them; refs that stay unreadable after a finished build are
recorded as unknown, never a failed build).

THE PACK ON GITHUB (--pack github, the default for a new build; lane fx-pack-offload, 2026-11-19): the legs' trees
stay on GitHub (their collectors take only the release-<set>-gha records: `release_github_build.py run
--records-only`), and linux-wait's tree checks, linux-assemble and linux-pack run on a free runner beside them
(.github/workflows/candidate-pack.yml running this same command with --pack-here; tools/release_remote_pack.py). The
Mac uploads the build's records and downloads linux/final (130-170 MB a set) instead of 3.9 GB of leg trees, and runs
no Docker. candidate.json records the run as `pack`, and a B arm's pack job takes its base's wheels from it. --pack
local packs on the Mac as before (and is what a build whose legs are taken from an earlier freeze needs). A resumed
build keeps the mode it was started with.

DISK (2026-11-16: seven of eight arms died with ENOSPC at linux-assemble, 5.6 GB free). SEVERAL ARMS start only
when the work root's disk has arms x the measured need of one wheel set (arm_need: the source worktree plus
PER_LINUX_SET_BYTES per Linux set) plus a reserve; linux-assemble (tools/release.py) refuses, with the numbers,
when the measured need of the sets it is about to assemble is not free. After a set packs and its candidate.json is
written, prune_set deletes what the final wheels and the records do not need: the payload of the downloaded leg
trees, the host composition and assembled reuse sets, the audit's wheel copies, the unstripped wheels in linux/,
and the source worktree. Kept: linux/final (wheels, candidate.json, split and strip audits), every .json and .log
record (leg provenance and the PTX witnesses a B arm reads from its base, reuse/plan.json, audit reports, pack
logs), state.json, ledger/, prebuild/. --keep-intermediates keeps everything. A work dir made before pruning
existed is never pruned.
"""
import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

CANDIDATE_ROOT = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", os.path.expanduser("~/mojolearn-evidence"))) \
    / "release-candidates"
#: the only Release steps a candidate runs (STEP_TABLE order); every other step is skipped by --only
STEPS = ("reuse-plan", "linux-builds", "linux-wait", "linux-assemble", "linux-pack")
VENDORS = ("nvidia", "amd")
LABEL_RE = re.compile(r"[a-z0-9]+(\.[a-z0-9]+)*")
#: the owed B arms of the pre-0.8.67 measurement: {arm name: {define, status}} under "b_arms"
OWED_FILE = "bench/results/paper_compare/owed-pre0867.json"

GIB = 1 << 30
#: MEASURED DISK NEED (du -sk of finished sets under ~/mojolearn-evidence/release-candidates, 2026-11-16): a wheel set
#: of 23bc7a5ae358 (sm_89 + sm_80) held 4.2 GB as the A set and 5.2 GB as a B arm, fcc934a37cd4 (three sets) 5.3 GB.
#: Of that, the detached source worktree is 1.9 GB, and each Linux set takes about 0.69 GB of downloaded leg trees
#: (1.37 GB for two sets, 2.25 GB for three), 0.31 GB of host composition, 0.52 GB of assembled reuse sets (B arm:
#: 1.0 GB for two) and 0.17 GB of packed and audited wheels. On 2026-11-16 seven of eight arms died with ENOSPC at
#: linux-assemble with 5.6 GB free; these numbers make that a refusal before the work starts.
SOURCE_TREE_BYTES = 2 * GIB
PER_LINUX_SET_BYTES = int(1.7 * GIB)        # legs + host composition + assembled sets + packed wheels
ASSEMBLE_PER_LINUX_SET_BYTES = int(0.7 * GIB)   # what linux-assemble and linux-pack still write per set
RESERVE_BYTES = 1 * GIB                     # never fill the disk to the last byte
#: what one Linux set still lands on the Mac when it packs on GitHub (--pack github): its share of linux/final
#: (130-170 MB a set of wheels, measured 2026-11-19) and the records; the leg trees stay on GitHub
GITHUB_PACK_PER_LINUX_SET_BYTES = int(0.25 * GIB)
#: what a finished set keeps of the trees prune_set thins: records, never payload
KEEP_SUFFIXES = (".json", ".log", ".txt", ".md", ".exit", ".pid", ".sha256", ".tsv", ".csv")
#: the intermediate trees of a set (<work>/<commit12>/) that the final wheels and the records do not need: the
#: downloaded leg trees (their .so/.tar.gz/toolchain payload), the host composition and the assembled reuse sets
#: (copies the packer read) and the audit's repaired wheel copies. reuse/previous-split (the base's wheels merged
#: again, 464 MB an arm) and reuse/ptx-published (the published PTX set unpacked, 140 MB) are copies of the base
#: too: 31 GB over the 70 d652a4b0a1 arms, which stopped the arm controller on laptop disk for four hours
#: (lane fx-build-tools, 2026-11-18). A resume makes both again (Release.previous_wheel, the PTX admission
#: re-verifies its tree); their .json witnesses stay by KEEP_SUFFIXES.
PRUNE_TREES = ("legs", "host-composition/sets", "reuse/sets", "reuse/previous-split", "reuse/ptx-published",
               "linux/audit")


def free_bytes(path):
    """Free bytes on the file system that holds PATH (its nearest existing parent)."""
    import shutil
    p = Path(path).expanduser().resolve()
    while not p.exists() and p != p.parent:
        p = p.parent
    return shutil.disk_usage(str(p)).free


def space_refusal(path, need, what, free=None):
    """None when PATH's disk has NEED bytes plus RESERVE_BYTES free, else the refusal with the numbers."""
    free = free_bytes(path) if free is None else free
    if free >= need + RESERVE_BYTES:
        return None
    return (f"{what}: {free / GIB:.1f} GB free at {path}, {(need + RESERVE_BYTES) / GIB:.1f} GB needed "
            f"({need / GIB:.1f} GB measured need + {RESERVE_BYTES / GIB:.0f} GB reserve). Free space (finished "
            "sets of ~/mojolearn-evidence/release-candidates/ hold their pruned records only) and rerun")


def arm_need(n_linux_sets, pack="local"):
    """The measured peak bytes of one wheel set (one candidate run) over N_LINUX_SETS Linux sets: packed on this Mac
    (PER_LINUX_SET_BYTES a set: legs, host composition, assembled sets, wheels) or on GitHub (--pack github:
    GITHUB_PACK_PER_LINUX_SET_BYTES, the final wheels and records only)."""
    per = GITHUB_PACK_PER_LINUX_SET_BYTES if pack == "github" else PER_LINUX_SET_BYTES
    return SOURCE_TREE_BYTES + n_linux_sets * per


def prune_plan(set_dir):
    """[(path, bytes)] that prune_set deletes in SET_DIR (<work>/<commit12>): every file under PRUNE_TREES whose
    suffix is not in KEEP_SUFFIXES, and the unstripped wheels directly in linux/ (linux/final holds the shipped
    ones). Kept: linux/final (wheels, candidate.json, split and strip audits), every .json/.log record (leg
    provenance, the PTX witnesses a B arm reads, reuse/plan.json, audit reports), the pack logs."""
    d = Path(set_dir)
    out = []
    for rel in PRUNE_TREES:
        top = d / rel
        if not top.is_dir():
            continue
        for root, _, files in os.walk(top):
            for name in files:
                p = Path(root) / name
                if not name.endswith(KEEP_SUFFIXES):
                    out.append((p, p.lstat().st_size))
    for p in sorted((d / "linux").glob("*.whl")) if (d / "linux").is_dir() else ():
        out.append((p, p.lstat().st_size))
    return out


def drop_source(work, commit):
    """Remove the detached source worktree <work>/source/<commit12> (source_tree makes it again on a resume).
    Returns the bytes freed."""
    import shutil
    src = Path(work) / "source" / commit[:12]
    if not src.exists():
        return 0
    size = sum(f.lstat().st_size for f in src.rglob("*") if f.is_file() or f.is_symlink())
    sh("git", "worktree", "remove", "--force", str(src), check=False)
    if src.exists():
        shutil.rmtree(src, ignore_errors=True)
        sh("git", "worktree", "prune", check=False)
    return size


def drop_source_after_failure(work, commit, state, pack, keep=False, say=print):
    """A --pack github build that stopped after its legs were collected (linux-builds done: the trees are on GitHub,
    the pack job checks out the source itself) keeps nothing on the Mac that needs the 1.9 GB source worktree, so
    it is removed; a resume makes it again. 2026-11-19: twelve failed arms of 7f5eac7eab left 12 x 1.9 GB of source
    worktrees (86 -> 49 GB free); the pack runs' artifacts were 3.5 MB each. Returns the bytes freed."""
    if keep or pack != "github" or "linux-builds" not in ((state or {}).get("steps") or {}):
        return 0
    freed = drop_source(work, commit)
    if freed:
        say(f"measurement build: stopped after its legs ran on GitHub; removed the source worktree "
            f"({freed / GIB:.2f} GB, a resume makes it again)")
    return freed


#: the steps a finished --pack github build runs again when repair_finished_github_pack repairs it
REPACK_STEPS = ("linux-builds", "linux-wait", "linux-assemble", "linux-pack")


def repair_finished_github_pack(r, say=print):
    """A FINISHED --pack github build packed before lane fx-pack-arm (2026-11-19) is repaired on a plain re-run, with
    no compile: (a) a final wheel whose runtime bytes differ from its own LINUX_PAYLOAD (the pack runner rebuilt
    libMojolearnMath.so with its GCC 13.3: every B arm on it was refused at assemble) and (b) a GitHub leg with no
    legs/<leg>.provenance.json (RemoteLeg launched without one: an arm finds no PTX proof of the base and rebuilds
    sm_80 despite --timed-on). The final set moves aside to linux/final.repaired-<stamp>, the provenance-less legs'
    exit files move aside (their collectors re-attach to the finished runs by --run-id and write the provenance), and
    REPACK_STEPS are unrecorded so the build packs again on GitHub. Returns the reasons ([] when nothing to do)."""
    rel = Path(r.rel)
    final, legs = rel / "linux" / "final", rel / "legs"
    finals = [w for w in r.split_finals().values() if w] if final.is_dir() else []
    if not finals:
        return []
    import release_reuse
    why = []
    drift = [f"{w.name}: {line}" for w in finals for line in release_reuse.payload_runtime_mismatches(w)]
    if drift:
        why.append("the final wheels' runtime bytes differ from their LINUX_PAYLOAD (" + "; ".join(drift) + ")")
    missing = sorted(e.name[:-len(".exit")] for e in legs.glob("*.exit")
                     if not (legs / (e.name[:-len(".exit")] + ".provenance.json")).is_file())
    if missing:
        why.append("GitHub leg(s) " + ", ".join(missing) + " have no launch provenance (an arm finds no PTX proof)")
    if not why:
        return []
    import datetime as dt
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    final.rename(final.with_name("final.repaired-" + stamp))
    for name in missing:
        (legs / (name + ".exit")).rename(legs / (name + ".exit.repaired-" + stamp))
    with r._lock:
        for step in REPACK_STEPS:
            (r.state.get("steps") or {}).pop(step, None)
            (r.state.get("failures") or {}).pop(step, None)
        r.state.pop("remote_pack", None)
        r.state.pop("linux_layout", None)
        r.save()
    say("measurement build: REPAIRING a finished GitHub pack (no compile): " + "; ".join(why) + f". The final set "
        f"moved to {final.name}.repaired-{stamp}; " + (f"legs {', '.join(missing)} re-attach to their finished runs; "
                                                     if missing else "") + "it packs again on GitHub")
    return why


def replan_arm_on_repaired_base(r, base, say=print):
    """An unfinished B arm whose reuse plan predates its base's (re)pack is planned again: the plan decided which
    sets the arm builds from the base it saw (2026-11-19: arms planned on a base without leg provenance refused its
    PTX set and launched sm_80 despite --timed-on sm_89; after repair_finished_github_pack repacked that base, a
    resumed arm kept the stale two-leg plan). Its legs already built keep their records (a leg still in the new plan
    is done; the collectors re-attach by --run-id otherwise). Returns True when it re-plans."""
    if not base or base.get("pending"):
        return False
    rel = Path(r.rel)
    if any((rel / "linux" / "final").glob("*.whl")):
        return False
    plan_path = rel / "reuse" / "plan.json"
    try:
        plan = json.loads(plan_path.read_text())
        made = (json.loads((Path(base["build_dir"]) / "linux" / "final" / "candidate.json").read_text()) or {}).get("made")
    except (OSError, ValueError):
        return False
    if not made or not plan.get("made") or str(plan["made"]) >= str(made):
        return False
    import datetime as dt
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    plan_path.rename(plan_path.with_name("plan.json.stale-" + stamp))
    with r._lock:
        for step in ("reuse-plan",) + REPACK_STEPS:
            (r.state.get("steps") or {}).pop(step, None)
            (r.state.get("failures") or {}).pop(step, None)
        r.state.pop("remote_pack", None)
        r.save()
    say(f"measurement build: the arm's reuse plan ({plan['made']}) predates its base's pack ({made}); planned again "
        f"(the old plan is {plan_path.name}.stale-{stamp})")
    return True


def prune_set(work, commit, say=print):
    """After a set packed and its candidate.json was written: delete prune_plan's files, the directories they
    leave empty, and the detached source worktree <work>/source/<commit12> (a resume makes it again). Returns the
    bytes freed."""
    d = Path(work) / commit[:12]
    freed = 0
    for p, size in prune_plan(d):
        p.unlink(missing_ok=True)
        freed += size
    for rel in PRUNE_TREES:
        for root, _, _ in os.walk(d / rel, topdown=False):
            if not os.listdir(root):
                os.rmdir(root)
    freed += drop_source(work, commit)
    say(f"measurement build: pruned {freed / GIB:.2f} GB of intermediates from {Path(work)} (leg trees, host "
        "composition, assembled sets, audit wheel copies, unstripped wheels, the source worktree); kept "
        f"{d / 'linux' / 'final'} (wheels, candidate.json), every .json/.log record, state.json, ledger, prebuild. "
        "--keep-intermediates keeps them")
    return freed


def owed_arms(path=OWED_FILE):
    """{owed arm name: define} of the owed file: PATH on disk, else (a lean checkout leaves bench/results out)
    its committed bytes at this checkout's HEAD."""
    f = Path(path) if Path(path).is_absolute() else ROOT / path
    try:
        text = f.read_text() if f.is_file() else sh("git", "show", f"HEAD:{path}")
        arms = json.loads(text).get("b_arms") or {}
    except (OSError, ValueError, SystemExit) as exc:
        raise SystemExit(f"candidate: cannot read the owed file {path}: {exc}")
    return {name: (row or {}).get("define") or "" for name, row in arms.items()}


def owed_arm_refusal(arm, path=OWED_FILE):
    """Why the owed ARM gets no build, or None (lane fb8-owed-arms, 2026-11-17): an env arm (`env`) runs in the A
    set's venv, and an arm the file classifies other than RUN (`owed` DECIDED / PAUSED / NO-ROW) is not timed."""
    f = Path(path) if Path(path).is_absolute() else ROOT / path
    try:
        row = (json.loads(f.read_text() if f.is_file() else sh("git", "show", f"HEAD:{path}")).get("b_arms") or {}).get(arm)
    except (OSError, ValueError, SystemExit):
        return None
    row = row or {}
    if row.get("env"):
        return f"{arm} is an env arm ({row['env']}): no build; stage it as --b-arm {arm}=env:{row['env']}"
    if row.get("owed", "RUN") != "RUN":
        return f"{arm} is {row['owed']}, not owed: {str(row.get('evidence', ''))[:160]}"
    return None


def resolve_owed(label, owed_arm, defines, path=OWED_FILE):
    """(label, defines, owed arm name) of a measurement build. The owed file's arm names carry hyphens, which a
    PEP 440 local label cannot (they split wheel file names): a '-' becomes '.' (`small-mlp-generic` ->
    `small.mlp.generic`). --owed-arm NAME, or a --label that is an owed arm's name, takes that arm's define; a
    --defines that differs from it is refused (2026-11-13: B sets staged under other names were "not planned")."""
    from release_github_build import normalize_defines
    arm = owed_arm or ""
    if arm:
        arms = owed_arms(path)
        if arm not in arms:
            raise SystemExit(f"candidate: --owed-arm {arm} is not an owed arm of {path} ({', '.join(sorted(arms))})")
        why = owed_arm_refusal(arm, path)
        if why:
            raise SystemExit(f"candidate: --owed-arm {why}")
    elif label:
        try:
            arms = owed_arms(path)
        except SystemExit:
            arms = {}   # a plain label needs no owed file
        dotted = label.replace("-", ".").lower()
        arm = next((name for name in sorted(arms) if name.replace("-", ".").lower() == dotted), "")
    if arm:
        want = normalize_defines(arms[arm])
        if not want:
            raise SystemExit(f"candidate: the owed arm {arm} names no define in {path}")
        if defines and defines != want:
            raise SystemExit(f"candidate: --defines {defines} is not the owed arm {arm}'s define {want}")
        defines = want
        if label and label != arm and label.replace("-", ".").lower() != arm.replace("-", ".").lower():
            raise SystemExit(f"candidate: --label {label} and --owed-arm {arm} name different arms")
        label = arm
    return (label or "").replace("-", ".").lower(), defines, arm or None


def arm_summary(name, plan):
    """One line per arm: shard units BUILD / REUSE over the launched legs, Linux rows BUILD / REUSE, the legs."""
    import release_reuse
    rows = [r for r in plan["rows"] if r["target"] == release_reuse.LINUX]
    units = [u for s, us in (plan.get("units") or {}).items() if s.replace("/", "-") in plan["legs"] for u in us]
    b = sum(u["decision"] == "BUILD" for u in units)
    full = bool(units) and b == len(units)
    return (f"ARM {name}: units BUILD {b} / REUSE {len(units) - b} on legs {','.join(plan['legs']) or 'none'}"
            + f"; rows BUILD {sum(r['decision'] == 'BUILD' for r in rows)} / REUSE {sum(r['decision'] == 'REUSE' for r in rows)}"
            + f"; against {(plan.get('previous') or {}).get('version')}"
            + (" (FULL BUILD: every unit reads the define)" if full else ""))


def reused_from_base(plan):
    """candidate.json's record of what a B arm took from its base build: the base, and per set the units built
    here and the units whose bytes are the base's (assemble_linux verified them against the base wheel)."""
    prev = plan.get("previous") or {}
    return dict(version=prev.get("version"), source_commit=prev.get("source_commit"), build_dir=prev.get("build_dir"),
                wheels={k: prev[k] for k in ("linux", "nvidia", "amd") if prev.get(k)},
                legs=plan.get("legs"),
                units={s: dict(build=[u["unit"] for u in us if u["decision"] == "BUILD"],
                               reuse=[u["unit"] for u in us if u["decision"] == "REUSE"])
                       for s, us in (plan.get("units") or {}).items()})


#: a compiler diagnostic (Mojo: <path>.mojo:<line>:<col>: error: ...), then any other failure line of a binding log
_COMPILER_ERROR = re.compile(r"(\S+\.(?:mojo|py|sh|c|cc|cpp|h)):(\d+)(?::\d+)?: error: .*")
_OTHER_ERROR = re.compile(r"(?:\berror:|Error: |constraint failed|REFUSED|SystemExit: ).*")
_LEG_LOG = re.compile(r"log (\S+/legs/[^/\s]+\.log)")


def first_build_error(arm_log):
    """The first error of a failed arm build, for its ARM line (lane fb9-arm-builds, 2026-11-17: the nv-fma-asm
    arm's ARM line said only its plan): the first compiler diagnostic (file:line: error) in the binding logs of the
    legs the arm log names (the local leg tree and its GitHub shards), else their first other failure line, else the
    arm's own FAILED/STOPPED line. Paths inside the box's checkout are shown relative to it."""
    text = Path(arm_log).read_text(errors="replace").splitlines()
    legs = []
    for line in text:
        for m in _LEG_LOG.finditer(line):
            leg = Path(m.group(1)).with_suffix("")
            if leg not in legs:
                legs.append(leg)
    found = legs_first_error(legs)
    if found:
        return found
    stop = next((x.strip() for x in reversed(text) if re.search(r"\b(FAILED|STOPPED) at ", x)), None)
    return (stop or "no error line in " + str(arm_log))[:240]


def legs_first_error(leg_dirs):
    """The first compiler diagnostic (file:line: error) in the FAILED binding logs under LEG_DIRS (a leg's local
    tree and its GitHub shards), else their first other failure line, else None. Also read by tools/release.py
    linux-wait, which prints it as soon as a leg fails (lane fx-build-tools, 2026-11-18)."""
    logs = []
    for leg in leg_dirs:
        leg = Path(leg)
        if leg.is_dir():
            failed = [f for f in sorted(leg.rglob("build_logs/*.log"))
                      if "FAILED" in (f.read_text(errors="replace").strip().splitlines() or [""])[-1]]
            logs += [f for f in failed if f not in logs]
    other = None
    for f in logs:
        for line in f.read_text(errors="replace").splitlines():
            m = _COMPILER_ERROR.search(line)
            if m:
                return re.sub(r"^/root/[^/]+/", "", m.group(0))[:240]
            if other is None and _OTHER_ERROR.search(line):
                other = f"{f.stem}: {_OTHER_ERROR.search(line).group(0)}"
    if other:
        return re.sub(r"/root/[^/ ]+/", "", other)[:240]
    return None


def run_arms(a, argv):
    """SEVERAL ARMS AT ONCE (--owed-arm a --owed-arm b ...): one measurement build per distinct define set, every
    one started at once as its own `release.py candidate` process (its own work dir, state and log), so their
    GitHub runs queue together and keep the free runners full (GitHub runs about 20 jobs at once and queues the
    rest; each arm dispatches only its BUILD units, release_reuse.unit_plan). Arms naming the same defines build
    once. A dry run plans them in parallel too, one process per arm up to the Mac's cores. Logs:
    <work root>/arms-<time>/<arm>.log. Returns non-zero when any arm failed."""
    import concurrent.futures as cf
    import datetime as dt
    arms, seen = [], {}
    for arm in a.owed_arm:
        _, defines, _ = resolve_owed("", arm, "", a.owed_file)
        if defines in seen:
            print(f"ARMS: {arm} names the defines of {seen[defines]} ({defines}): built once, as {seen[defines]}")
            continue
        seen[defines] = arm
        arms.append(arm)
    rest, skip = [], False
    for x in argv:
        if skip:
            skip = False
            continue
        if x == "--owed-arm":
            skip = True
            continue
        if x.startswith("--owed-arm="):
            continue
        rest.append(x)
    if not a.dry_run:
        import release_reuse
        n_sets = len(release_reuse.scope_linux_sets(a.targets))
        pack = a.pack or "github"   # a new arm's default; an arm resumed with --pack local passes it
        need = arm_need(n_sets, pack)
        refusal = space_refusal(a.work_root, len(arms) * need,
                                f"ARMS: {len(arms)} arms x {need / GIB:.1f} GB ({n_sets} Linux sets each, pack {pack})")
        if refusal:
            raise SystemExit(refusal)
    logs = Path(a.work_root).expanduser().resolve() / f"arms-{dt.datetime.now().strftime('%Y%m%d-%H%M%S')}"
    logs.mkdir(parents=True, exist_ok=True)
    workers = len(arms) if not a.dry_run else max(1, min(len(arms), os.cpu_count() or 1))
    print(f"ARMS: {len(arms)} measurement builds ({', '.join(arms)}), {workers} at once; logs {logs}")

    def one(arm):
        log = logs / f"{arm}.log"
        with open(log, "w") as fh:
            rc = subprocess.call([sys.executable, str(ROOT / "tools" / "release.py"), "candidate", *rest,
                                  "--owed-arm", arm], cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT)
        text = log.read_text(errors="replace").splitlines()
        line = next((x for x in reversed(text) if x.startswith("ARM ")), f"ARM {arm}: no plan line")
        return arm, rc, line, log

    failed = 0
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for arm, rc, line, log in ex.map(one, arms):
            err = f", first error: {first_build_error(log)}" if rc != 0 else ""
            print(f"{line}  [rc={rc}{err}, {log}]")
            failed += rc != 0
    return 1 if failed else 0


def sh(*args, cwd=ROOT, check=True):
    p = subprocess.run(list(args), cwd=str(cwd), capture_output=True, text=True)
    if check and p.returncode != 0:
        raise SystemExit(f"candidate: {' '.join(args[:4])} failed: {p.stderr.strip()[:300]}")
    return p.stdout.strip()


def main_checkout():
    """The primary worktree of this repository (the main checkout every lane is told not to edit)."""
    common = Path(sh("git", "rev-parse", "--path-format=absolute", "--git-common-dir"))
    return common.parent.resolve()


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    return path == root or root in path.parents


def ls_remote(*args, attempts=3, backoff=(10, 30), timeout=120, sleep=None):
    """`git ls-remote ARGS` with ATTEMPTS tries and a growing pause between them (2026-11-16: one 75 s network
    timeout of the final tag check turned a finished arm into rc 1). SystemExit after the last failure."""
    import time
    sleep = sleep or time.sleep
    why = ""
    for n in range(attempts):
        try:
            p = subprocess.run(["git", "ls-remote", *args], cwd=str(ROOT), capture_output=True, text=True,
                               timeout=timeout)
            if p.returncode == 0:
                return p.stdout.strip()
            why = p.stderr.strip()[:300]
        except subprocess.TimeoutExpired:
            why = f"no answer in {timeout} s"
        if n + 1 < attempts:
            pause = backoff[min(n, len(backoff) - 1)]
            print(f"candidate: git ls-remote {' '.join(args)} failed ({why}); retry {n + 2}/{attempts} in {pause} s")
            sleep(pause)
    raise SystemExit(f"candidate: git ls-remote {' '.join(args)} failed {attempts} times: {why}")


def remote_refs():
    """(origin/main sha, sorted origin tag lines, sorted local tags): what a candidate must never change."""
    main = ls_remote("origin", "refs/heads/main").split()
    tags = sorted(ls_remote("--tags", "origin").splitlines())
    local = sorted(sh("git", "tag", "-l").splitlines())
    return dict(origin_main=main[0] if main else None,
                origin_tags_sha256=hashlib.sha256("\n".join(tags).encode()).hexdigest(), origin_tags=len(tags),
                local_tags_sha256=hashlib.sha256("\n".join(local).encode()).hexdigest(), local_tags=len(local))


def next_version(committed):
    m = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)", committed or "")
    if not m:
        raise SystemExit(f"candidate: the commit's _version.py says {committed!r}; pass --base-version X.Y.Z")
    return f"{m[1]}.{m[2]}.{int(m[3]) + 1}"


def candidate_version(base, commit, label):
    return f"{base}.dev0+{commit[:12]}" + (f".{label}" if label else "")


def default_label(defines):
    return ("d" + hashlib.sha256(defines.encode()).hexdigest()[:8]) if defines else ""


def source_tree(work, commit, dry):
    """A detached worktree of COMMIT at <work>/source/<commit12> (made once, verified clean after)."""
    tree = work / "source" / commit[:12]
    if (tree / ".git").exists():
        head = sh("git", "-C", str(tree), "rev-parse", "HEAD", check=False)
        dirty = sh("git", "-C", str(tree), "status", "--porcelain", "--untracked-files=no", check=False)
        if head != commit or dirty:
            raise SystemExit(f"candidate: {tree} is not a clean checkout of {commit}; "
                             f"git worktree remove --force {tree} and rerun")
        return tree
    if dry:
        return tree
    tree.parent.mkdir(parents=True, exist_ok=True)
    # -f only when the path is absent: a worktree directory deleted by hand stays registered ("a missing but already
    # registered worktree"), which refused the 2026-11-19 rerun of arm fx-lc-test-reuse before any work
    force = [] if tree.exists() else ["-f"]
    sh("git", "worktree", "add", *force, "--detach", str(tree), commit)
    # A worktree added from a lean (sparse) checkout inherits its patterns and lacks bench/results, which
    # holds the verification CTR models the pack stages; the source tree is the whole commit. Per worktree
    # under extensions.worktreeConfig; a no-op on a full tree. (2026-11-12: two measurement builds packed
    # 56/56 bindings and then refused on the missing models.)
    sh("git", "-C", str(tree), "sparse-checkout", "disable")
    models = tree / "bench/results/identity_break"
    if not models.is_dir():
        raise SystemExit(f"candidate: {tree} lacks {models.relative_to(tree)} after the checkout; "
                         f"git worktree remove --force {tree} and rerun")
    return tree


def timed_on(raw, targets, defines):
    """The sorted arches of --timed-on RAW (empty: every set is timed). Refused for an A set (no defines: nothing
    to take from a base), for an arch outside TARGETS, and when it names every target (that is the default)."""
    arches = sorted({x.strip() for x in (raw or "").split(",") if x.strip()})
    if not arches:
        return []
    if not defines:
        raise SystemExit("candidate: --timed-on is for a B arm (--defines or --owed-arm); an A set builds every set")
    unknown = [x for x in arches if x not in targets]
    if unknown:
        raise SystemExit(f"candidate: --timed-on {','.join(unknown)} is not in the targets {','.join(targets)}")
    return [] if set(arches) == set(targets) else arches


def untimed_record(plan, targets):
    """candidate.json's record of a B arm built for some timed sets only (release_reuse.apply_defines TIMED SETS):
    `timed_on`, and `untimed_from_base` {"vendor/arch": [binding names that read the define and ship the base's
    bytes]}. Those sets import, but an A/B timed on them would time the base. Empty for any other build."""
    timed = plan.get("timed_sets") or []
    if not timed:
        return {}
    untimed = {}
    for r in plan.get("rows") or []:
        if r.get("decision") == "REUSE" and str(r.get("reason", "")).startswith("untimed set:"):
            untimed.setdefault(f"{r['vendor']}/{r['arch']}", []).append(r["name"])
    import release_reuse
    for vendor, arch in release_reuse.ALL_LINUX_SETS:
        if arch in targets and arch not in timed:
            untimed.setdefault(f"{vendor}/{arch}", [])
    return dict(timed_on=list(timed), untimed_from_base={k: sorted(v) for k, v in sorted(untimed.items())})


def write_candidate(a, r, release, C, work, cand, vendors, version, defines, label, tooling_sha, targets, built, plan,
                    before, after, moved):
    """candidate.json beside the final wheels (the record tools/ops/measure_candidate.py stage reads), after the
    wheels' manifests prove this build's version and defines. `pack` says where the wheels were packed: the GitHub
    pack run (a B arm's pack job takes its base's wheels from that run's artifact), its runner, or this Mac.
    Returns 0."""
    final = work / a.commit[:12] / "linux" / "final"
    finals = r.split_finals()
    if not finals.get("linux") or release.wheel_commit(finals["linux"]) != a.commit:
        raise SystemExit(f"measurement build: no core wheel of {a.commit[:12]} in {final}")
    manifests = {}
    for vendor in vendors:
        wheels = [finals["linux"], finals[vendor]]
        manifests[vendor] = C.build_manifest(version, vendor, wheels, a.commit)   # RECORD, metadata, tags, payload
        for f in manifests[vendor]["files"]:
            payload = f.get("candidate") or {}
            if payload.get("local_version") != version or (payload.get("defines") or "") != defines:
                raise SystemExit(f"measurement build: {f['file']} does not record this build's version and defines")
    doc = dict(schema="mojolearn.release-candidate.v1", version=version, source_commit=a.commit, defines=defines,
               label=label, owed_arm=cand.get("owed_arm"), tooling_sha=tooling_sha, refs_moved=moved,
               wheel_dir=str(final), vendors=vendors, targets=targets, wheels_built=built, steps=list(STEPS),
               wheels={k: dict(file=w.name, sha256=release.sha256(w)) for k, w in finals.items() if w},
               manifests=manifests, refs_before=before, refs_after=after, made=release.now(),
               pack=pack_record(r.state.get("remote_pack"), a.pack_here),
               **({"base": reused_from_base(plan)} if plan and (plan.get("previous") or {}).get("base") else {}),
               **(untimed_record(plan, targets) if plan else {}))
    release.write_json(final / "candidate.json", doc)
    print(f"measurement build: {final}/candidate.json")
    return 0


def pack_record(remote, here=False):
    """candidate.json `pack`: where the wheels were packed. The GitHub pack run (run_id, url, packed_at: the base
    wheels of a later B arm are taken from its candidate-final artifact), the runner itself (here), or the Mac."""
    if here:
        return dict(where="github-runner")
    remote = remote or {}
    if remote.get("conclusion") == "success" and remote.get("run_id"):
        return dict(where="github", run_id=remote["run_id"], url=remote.get("url"), tag=remote.get("tag"),
                    packed_at=remote.get("packed_at"), tooling_sha=remote.get("tooling_sha"))
    return dict(where="local")


def parse(argv):
    ap = argparse.ArgumentParser(prog="release.py candidate", description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--commit", required=True, help="the exact source commit (40 hex), pushed to origin")
    ap.add_argument("--targets", default="", help="Linux sets to build, a comma list of sm_89, sm_80, gfx942 (default "
                    "all three; sm_89,sm_80 is the NVIDIA-only build: the nvidia wheel needs its PTX slot)")
    ap.add_argument("--vendors", default="", help="vendor wheels to validate and stage (default: the wheels of the "
                    "targets, nvidia,amd for every target)")
    ap.add_argument("--defines", default="", help="B set: '-D NAME=V -D NAME2' or NAME=V,NAME2 (MOJOLEARN_BUILD_DEFINES)")
    ap.add_argument("--timed-on", default="", help="B arms only: the arches the arm is TIMED on, a comma list of "
                    "--targets (sm_89 for the NVIDIA L40S). The define's device units build only in those sets; every "
                    "other set of --targets ships the base build's bytes (compiled without the define) so the wheel "
                    "still imports, and candidate.json names them in untimed_from_base. Default: every set is timed")
    ap.add_argument("--label", default="", help="the version's label (lower case [a-z0-9.]; a '-' becomes '.', so an "
                    "owed arm's name is a label; default: none for an A set, d<8 hex of the defines> for a B set)")
    ap.add_argument("--owed-arm", action="append", default=[], help="an owed B arm of the owed file's b_arms (e.g. "
                    "small-mlp-generic): its define, and its name as the label (small.mlp.generic) and as candidate.json "
                    "owed_arm. Repeat it for several arms: each runs as its own measurement build, all at once (ARMS)")
    ap.add_argument("--owed-file", default=OWED_FILE, help="the owed file (default %(default)s)")
    ap.add_argument("--base-version", default="", help="X.Y.Z of the .dev0 version (default: the commit's "
                    "_version.py plus one patch)")
    ap.add_argument("--ref", default="", help="the pushed tooling branch the workflow runs from (default: this "
                    "checkout's branch; it must hold this checkout's HEAD)")
    ap.add_argument("--work-root", default=str(CANDIDATE_ROOT), help="default ~/mojolearn-evidence/release-candidates")
    ap.add_argument("--dry-run", action="store_true", help="print the Release plan of the five steps; run nothing")
    ap.add_argument("--no-prebuild-check", action="store_true", help="do not run tools/prebuild_check.py (the pack "
                    "source audits and the host-parse suite) before the build is dispatched")
    ap.add_argument("--probe-wheel", default="", help="tools/prebuild_check.py --probe-wheel: an earlier packed "
                    "mojolearn_core-*.whl, smoked with this commit's Python before the build")
    ap.add_argument("--keep-intermediates", action="store_true", help="keep the leg trees, host composition, "
                    "assembled sets, audit wheel copies and source worktree after the pack (default: pruned once "
                    "candidate.json is written; see prune_set)")
    ap.add_argument("--no-verify-smoke", action="store_true", help="do not start .github/workflows/verify-smoke.yml "
                    "on the packed core wheel (an A set starts it by default; it reports and gates nothing)")
    ap.add_argument("--no-cpu-coverage", action="store_true", help="do not start the CPU coverage smoke "
                    "(.github/workflows/fast-trial.yml mode=coverage) beside the verify smoke; it reports and gates nothing")
    ap.add_argument("--pack", choices=("github", "local"), default=None, help="where linux-wait's tree checks, "
                    "linux-assemble and linux-pack run: github (default for a new build: "
                    ".github/workflows/candidate-pack.yml beside the trees; the Mac downloads linux/final) or local "
                    "(this Mac, Docker for audit.sh). A resumed build keeps the mode it started with")
    ap.add_argument("--pack-here", action="store_true", help=argparse.SUPPRESS)   # candidate-pack.yml's runner
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--census", dest="census", action="store_true", default=None, help="the verify smoke takes "
                   "the frames census even when nothing it reads changed (tools/ops/verify_smoke_dispatch.py)")
    g.add_argument("--no-census", dest="census", action="store_false", help="the verify smoke skips the frames "
                   "census (default: it runs when python/mojolearn or the census's files changed)")
    return ap.parse_args(argv)


def main(argv=None, release=None):
    a = parse(sys.argv[1:] if argv is None else argv)
    if len(a.owed_arm) > 1:
        return run_arms(a, sys.argv[1:] if argv is None else argv)
    a.owed_arm = a.owed_arm[0] if a.owed_arm else ""
    if release is None:
        import release  # noqa: F811 - tools/release.py, when this file is run directly
    import release_github_build
    import release_reuse
    import candidate_wheel_install as C
    if not re.fullmatch(r"[0-9a-f]{40}", a.commit):
        raise SystemExit("candidate: --commit must be the full 40-hex sha")
    if sh("git", "cat-file", "-t", a.commit, check=False) != "commit":
        raise SystemExit(f"candidate: {a.commit} is not a commit here; git fetch origin first")
    # (the pack runner's checkout need not list every origin branch; the Mac that dispatched it checked)
    if not a.pack_here and not sh("git", "branch", "-r", "--contains", a.commit, check=False):
        raise SystemExit(f"candidate: {a.commit[:12]} is on no origin branch; the runners check out a pushed commit")
    targets = [s for _, s in release_reuse.scope_linux_sets(a.targets)]    # every set unless --targets
    built = release_reuse.scoped_wheels()
    vendors = [v for v in (a.vendors or ",".join(built)).split(",") if v]
    if not vendors or any(v not in VENDORS for v in vendors):
        raise SystemExit("candidate: --vendors is a comma list of nvidia, amd")
    if [v for v in vendors if v not in built]:
        raise SystemExit(f"candidate: --vendors {','.join(vendors)} names a wheel --targets {','.join(targets)} "
                         f"does not build (it builds {','.join(built)})")
    defines = release_github_build.normalize_defines(a.defines)
    label, defines, owed_arm = resolve_owed(a.label, a.owed_arm, defines, a.owed_file)
    timed = timed_on(a.timed_on, targets, defines)
    for entry in [e for e in defines.split(",") if e]:
        name = entry.split("=", 1)[0]
        readers = sorted(release_reuse.define_readers(a.commit, [name], ROOT))
        if not [r for r in readers if r.endswith(".mojo") or r.startswith("bindings/")]:
            # no byte could change: the B set would be the A set under another version
            raise SystemExit(f"candidate: no .mojo or bindings/ file of {a.commit[:12]} names {name}; "
                             "that define changes nothing at this commit")
        print(f"measurement build: {name} is read by {', '.join(readers[:6])}" + (" ..." if len(readers) > 6 else ""))
    label = label or default_label(defines)
    if label and not LABEL_RE.fullmatch(label):
        raise SystemExit("candidate: --label is lower case letters and digits, dot or hyphen separated (a PEP 440 "
                         "local segment; a '-' becomes '.')")
    committed = re.search(r'^__version__ = "([^"]*)"',
                          sh("git", "show", f"{a.commit}:python/mojolearn/_version.py"), re.M)
    base = a.base_version or next_version(committed[1] if committed else None)
    if not re.fullmatch(r"\d+\.\d+\.\d+", base):
        raise SystemExit("candidate: --base-version must be X.Y.Z")
    version = candidate_version(base, a.commit, label)
    work = Path(a.work_root).expanduser().resolve() / version
    base_build = None
    if defines:
        # THE BASE BUILD (release_reuse.base_build): the A set of this commit and targets, from which every unit
        # whose closure names none of the defines is reused
        base_work = Path(a.work_root).expanduser().resolve() / candidate_version(base, a.commit, "")
        base_build = release_reuse.base_build(base_work, a.commit, targets, dry=a.dry_run)
        if base_build is None:
            print(f"measurement build: NO BASE BUILD at {base_work} (the A set of {a.commit[:12]}, targets "
                  f"{','.join(targets)}, finished): every unit of this arm builds. Build the A set first to reuse it")
            if timed:
                print(f"measurement build: --timed-on {','.join(timed)} needs the base build to take the untimed "
                      "sets from; without it every set builds with the define")
        else:
            print(f"measurement build: base build {base_build['version']}"
                  + (" (PENDING: still building; a real run waits for it to finish)" if base_build.get("pending")
                     else f" ({base_build['build_dir']})"))
    main_tree = main_checkout()
    if inside(work, main_tree):
        raise SystemExit(f"candidate: the work dir {work} is inside the main checkout {main_tree}; refused")
    src = source_tree(work, a.commit, a.dry_run)
    if inside(src, main_tree) or src.resolve() == ROOT.resolve():
        raise SystemExit(f"candidate: the source tree {src} is the main checkout or this tooling checkout; refused")

    # the pack runner pushes and tags nothing either; the Mac that dispatched it reads the refs around the whole build
    before = None if a.pack_here else remote_refs()
    state_path = work / "state.json"
    # THE PIN, written for a dry run too (the candidate's own file): Release.commit reads state["commit"], so no
    # freeze step is needed or run
    work.mkdir(parents=True, exist_ok=True)
    state = release.read_json(state_path) or {}
    if state and (state.get("commit") != a.commit or (state.get("candidate") or {}).get("defines") != defines):
        raise SystemExit(f"candidate: {state_path} is another candidate's state; pick another --label")
    # a state of every target predates --targets (no "targets" key) and is a candidate of all three sets
    had = (state.get("candidate") or {}).get("targets") or [s for _, s in release_reuse.ALL_LINUX_SETS]
    if state and had != targets:
        raise SystemExit(f"candidate: {state_path} built the targets {','.join(had)}, not {','.join(targets)}; "
                         "pick another --label")
    if state and ((state.get("candidate") or {}).get("timed_on") or []) != timed:
        raise SystemExit(f"candidate: {state_path} was timed on {','.join((state.get('candidate') or {}).get('timed_on') or ['every set'])}, "
                         f"not {','.join(timed) or 'every set'}; pick another --label")
    if not state:
        state = dict(version=version, commit=a.commit, steps={}, release_modes=["identical"],
                     candidate=dict(version=version, commit=a.commit, defines=defines, label=label, targets=targets,
                                    **({"timed_on": timed} if timed else {}),
                                    base_version=base, committed_version=committed[1] if committed else None,
                                    owed_arm=owed_arm, made=release.now(), prune=True,
                                    pack=a.pack or "github"))
    # THE TOOLING SHA, recorded at the first launch: a resume dispatches from the same pushed ref while it still
    # holds this sha with the same route files, even after the ref moved on (release_github_build.tooling_ref;
    # 2026-11-13: origin/main moved under the tooling checkout and every resume refused with "push it")
    cand = state.setdefault("candidate", {})
    if not cand.get("tooling_sha"):
        cand["tooling_sha"] = sh("git", "rev-parse", "HEAD")
        branch = a.ref or sh("git", "rev-parse", "--abbrev-ref", "HEAD")
        cand["tooling_ref"] = "" if branch == "HEAD" else branch
    if owed_arm and not cand.get("owed_arm"):
        cand["owed_arm"] = owed_arm
    # THE PACK MODE: given, else the one the build started with (a state from before the GitHub pack packed locally)
    pack = a.pack or cand.get("pack") or "local"
    if not a.pack_here:
        if a.pack and cand.get("pack", "local") != a.pack:
            print(f"measurement build: --pack {a.pack} (the build started with --pack {cand.get('pack', 'local')})")
        cand["pack"] = pack
    release.write_json(state_path, state)
    tooling_sha = cand["tooling_sha"]
    argv = [version, "--only", ",".join(STEPS), "--state-dir", str(work), "--source-checkout", str(src),
            "--build-backend", "github", "--identity-on-wheels"] + (["--dry-run"] if a.dry_run else [])
    args = release.release_parser().parse_args(argv)
    args.release_modes = release_reuse.release_modes(default=("identical",))
    args.candidate_version, args.candidate_defines = version, defines
    args.candidate_ref = a.ref or cand.get("tooling_ref") or ""
    args.candidate_base = base_build
    args.candidate_timed = ",".join(timed)
    args.candidate_tooling_sha = tooling_sha
    # release.candidate_pack: "here" on candidate-pack.yml's runner, "github" on the Mac that dispatches it
    args.candidate_pack = "here" if a.pack_here else ("github" if pack == "github" else "")
    r = release.Release(args)
    if not a.dry_run and not a.pack_here and args.candidate_pack == "github":
        repair_finished_github_pack(r)
        replan_arm_on_repaired_base(r, base_build)
    if r.commit != a.commit:
        raise SystemExit(f"measurement build: the Release resolved {r.commit}, not {a.commit}")
    print(f"measurement build {version} of {a.commit[:12]}" + (f" defines {defines}" if defines else "")
          + f"; targets {','.join(targets)}; wheels core,{','.join(built)}"
          + f"; work dir {work}; steps {', '.join(STEPS)} (every other release step is skipped)")
    if not a.dry_run and not a.no_prebuild_check and not a.pack_here:
        import prebuild_check
        pre = ["--commit", a.commit, "--ref", args.candidate_ref, "--work", str(work / "prebuild"), "--reuse-pass"]
        base_pass = Path(base_build["build_dir"]).parent / "prebuild" / "PASS.json" if base_build else None
        if base_pass and base_pass.is_file() and not (work / "prebuild" / "PASS.json").is_file():
            # the checks read the commit's source only, which a B arm shares with its base: the base's pass is
            # copied in and prebuild_check --reuse-pass still checks its commit and parts (8 arms ran them 8 times)
            (work / "prebuild").mkdir(parents=True, exist_ok=True)
            (work / "prebuild" / "PASS.json").write_bytes(base_pass.read_bytes())
        if prebuild_check.main(pre + (["--probe-wheel", a.probe_wheel] if a.probe_wheel else [])):
            raise SystemExit(f"measurement build: the pre-build checks failed (logs {work / 'prebuild'}); nothing "
                             "was dispatched. Fix and rerun, or pass --no-prebuild-check")
    rc = r.go()
    plan = r.plan() if defines else None
    if plan:
        print(arm_summary(owed_arm or label, plan))

    if a.pack_here:
        # candidate-pack.yml's runner: linux/final and its candidate.json are what the artifact carries; the Mac writes
        # the build's own candidate.json, starts the verify smoke and prints the STAGE / B-ARM lines
        if rc or a.dry_run:
            return rc
        return write_candidate(a, r, release, C, work, cand, vendors, version, defines, label, tooling_sha, targets,
                               built, plan, before=None, after=None, moved=[])
    try:
        after = remote_refs()
    except SystemExit as exc:
        # the refs are reported, never a refusal of built wheels (below); a network that stays down after the
        # retries leaves them unread, said so, and the build stands
        print(f"measurement build: the refs after the run could not be read ({exc}); recorded as unknown")
        after = {k: before[k] for k in before}
        after["unread"] = str(exc)
    moved = [k for k in ("origin_main", "origin_tags_sha256", "local_tags_sha256") if before[k] != after[k]]
    if moved:
        # This path has no push, tag or publish command, so a change here is someone else's (2026-11-12: the
        # orchestrator merged to main during a build and the finished build was refused, its wheels unrecorded).
        # It is reported and recorded in candidate.json (refs_moved), never a refusal of the built wheels.
        print("measurement build: changed by someone else during the run (this build pushes and tags nothing): "
              + ", ".join(f"{k} {before[k]} -> {after[k]}" for k in moved))
    else:
        print(f"measurement build: origin/main {str(after['origin_main'])[:12]} and {after['origin_tags']} origin / "
              f"{after['local_tags']} local tags unchanged")
    if rc and not a.dry_run:
        drop_source_after_failure(work, a.commit, r.state, pack, keep=a.keep_intermediates)
    if rc or a.dry_run:
        return rc
    write_candidate(a, r, release, C, work, cand, vendors, version, defines, label, tooling_sha, targets, built, plan,
                    before, after, moved)
    final = work / a.commit[:12] / "linux" / "final"
    if not defines and not a.no_verify_smoke:
        # THE VERIFY SMOKE (.github/workflows/verify-smoke.yml): does verify still run and do the Python boundaries
        # still work, from this build's core wheel, on a free GitHub runner. Started here because this is where the
        # wheel exists (release-linux-build.yml uploads no wheel). One per commit: a B set shares its A set's source.
        # It reports and gates nothing, so a failure to start it is printed and the build still succeeds.
        try:
            sys.path.insert(0, str(ROOT / "tools" / "ops"))
            import verify_smoke_dispatch
            verify_smoke_dispatch.dispatch(final, a.ref or cand.get("tooling_ref") or "", a.commit,
                                           census=getattr(a, "census", None))
        except (Exception, SystemExit) as exc:
            print(f"VERIFY-SMOKE not started ({exc}); by hand: python3 tools/ops/verify_smoke_dispatch.py "
                  f"--wheel-dir {final} --ref <pushed branch holding verify-smoke.yml>")
        if not getattr(a, "no_cpu_coverage", False):
            # THE CPU COVERAGE SMOKE (lane fx-coverage-smokes, 2026-11-20): every public class and method of the same
            # core wheel on a CPU runner, identical (fast-trial.yml mode=coverage). It reports and gates nothing.
            try:
                verify_smoke_dispatch.dispatch_fast_trial(final, a.ref or cand.get("tooling_ref") or "", a.commit,
                                                          shards=verify_smoke_dispatch.COVERAGE_SHARDS,
                                                          mode="coverage")
            except (Exception, SystemExit) as exc:
                print(f"CPU-COVERAGE not started ({exc}); by hand: python3 tools/ops/verify_smoke_dispatch.py "
                      f"--workflow cpu-coverage --wheel-dir {final} --ref <pushed branch holding fast-trial.yml>")
    if a.keep_intermediates:
        print("measurement build: --keep-intermediates: nothing pruned")
    elif not cand.get("prune"):
        # a work dir made before pruning existed is left exactly as it is
        print(f"measurement build: {work} predates pruning; nothing pruned")
    else:
        prune_set(work, a.commit)
    for vendor in vendors:
        if defines:
            # a B set rides with its A set's stage line (measure_candidate reads this version from candidate.json)
            print(f"B-ARM {vendor} --b-arm {cand.get('owed_arm') or '<owed arm name>'}={final}@{a.commit}")
        else:
            print(f"STAGE python3 tools/ops/measure_candidate.py stage {vendor} --version '{version}' "
                  f"--source-commit {a.commit} --tree <job worktree>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
