#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Compile on merge, not on release day (lane/prebuild-on-merge, 2026-09-25).

    python3 tools/prebuild_changed.py plan  [--commit origin/main] [--targets T,..] [--json F] [--summary F]
    python3 tools/prebuild_changed.py plan  --rent [--targets cuda-sm_89]   # fill the cache, then re-plan
    python3 tools/prebuild_changed.py check-keys --commit C --target T <leg keys dir>
    python3 tools/prebuild_changed.py capture --target T <leg keys dir>     # refresh a profile

WHAT IT ANSWERS. For one commit and each Linux release target, which binding
archives the release build leg will ask the R2 binding cache for
(tools/bincache.py) and which of them are not there yet. `--rent` then runs
ONE build per target with missing keys, through the existing leg scripts, so
the uploads land at exactly those keys; a release leg afterwards finds every
binding cached and spends minutes, not the 20 to 40 of a cold build.

HOW A KEY IS PREDICTED WITHOUT THE BOX. A bincache key is sha256 over fields
(`bincache.key_fields` plus the `declared` block build_sets.sh adds). Some
depend only on the commit (the binding's source closure, pixi.lock's Linux
toolchain, the shell and generator files the script names); the rest depend
only on the box and the leg (device arch, image, OS, repo path, and the
MOJOLEARN_*/MODULAR_* variables the release scripts export per tier). The
second half is a PROFILE, read out of a real release leg's own
keys/<key>.json by `capture` and kept in tools/prebuild_profiles.json. The
planner joins a profile with the commit's half (computed from `git archive`
of the commit, never a working tree) and hashes. MEASURED 2026-09-25: from
the 0.8.19 H100 leg's own records at 9b15a0dd0, 76 of 76 keys reproduced on
the Mac byte for byte (and 75/76/76 at 69a519c1, d61c74ea and the gfx942
CPU-pod build at 24f75178, one profile per target). PROVEN end to end at
1acc9006e: `plan --rent --targets cuda-sm_89` rented one RTX 4090 ($0.74/h,
1381 s wall, builds 995 s), built and promoted 75 of 75, the box's 75 keys
equal to the planner's, and the re-plan read 0 missing. A profile that goes stale (a new exported variable, a
new image) is caught by `check-keys` after every rented prebuild, which
exits non-zero on any key the box computed that the planner did not.

WHICH BUILDS. Enumerated from the commit's own packaging/linux/build_sets.sh
(its tier_scripts function, evaluated in bash with the release's
MOJOLEARN_PACKAGE_BYTE_LM=1) and python/mojolearn/host_surface.py
--wheel-families, so a new binding is planned the day it is merged.

TARGETS AND WHO READS THEM (the partition is <device arch>/<image slug>):
  cuda-sm_90a  the release's H100 leg (tools/gemm_remote_leg.sh, campaign 7)
  cuda-sm_89   the release's L40S leg (same); any sm_89 box (RTX 4090) keys
               identically, the image and OS being the same
  hip-gfx942   the CPU-pod route (tools/release_linux_build.sh --archs gfx942),
               partition none/<rocm image>. NOT read by the default AMD
               release leg; see AMD below.

AMD. The default gfx942 release leg (tools/do_release061_leg.sh or
tools/hotaisle_release_leg.sh) never stages the binding cache, so nothing
prebuilt reaches it whatever the key. Three alignments would be needed for a
prebuilt gfx942 entry to be served there: (1) that leg stages and promotes
like gemm_remote_leg.sh; (2) the key's `device_arch` must name the TARGET
(MOJOLEARN_GPU_ARCHS) rather than what rocminfo reports, since the CPU pod
says `none` and the MI325X `gfx942` for the same --target-accelerator
compile; (3) `image` must name the build container, not the provider
(`runpod-cpu:<rocm image>` versus a `do:`/`hotaisle:` label for the same
rocm/dev-ubuntu-22.04 digest), and the leg's exported variables must match
the CPU route's. Until then the planner reports gfx942 against the CPU-pod
partition only.

CREDENTIALS. Listing the cache needs R2_ACCOUNT_ID, R2_ACCESS_KEY_ID,
R2_SECRET_ACCESS_KEY and R2_BUCKET in the environment, or ~/.mojolearn_r2
(MOJOLEARN_R2_CREDS). Without them the plan prints every key and says the
cache state is unknown. `--rent` additionally needs the RunPod key the leg
scripts read (~/.mojolearn_runpod_key). Stdlib only.
"""
import argparse
import datetime as dt
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import bincache  # noqa: E402

PROFILE_SCHEMA = "mojolearn.prebuild-profile.v1"
PLAN_SCHEMA = "mojolearn.prebuild-plan.v1"
PROFILES = ROOT / "tools" / "prebuild_profiles.json"
TIERS = ("fast", "deterministic", "identical")
#: Parts of a key's fields that come from the commit, not the box.
COMMIT_FIELDS = ("script", "args", "source", "toolchain")
#: The cheapest box first; a RunPod "no stock" walks to the next.
GPU_WALK = {
    "sm_89": ["NVIDIA GeForce RTX 4090", "NVIDIA L40S", "NVIDIA L40", "NVIDIA RTX 6000 Ada Generation"],
    "sm_90a": ["NVIDIA H100 PCIe", "NVIDIA H100 80GB HBM3", "NVIDIA H100 NVL", "NVIDIA H200"],
}
NO_STOCK = "no instances currently available"
EVIDENCE = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", str(Path.home() / "mojolearn-evidence")))


def git(*args, root=ROOT):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, check=True).stdout


# ---------------------------------------------------------------- the commit's half
def extract_tree(commit, dest, root=ROOT):
    """The commit's tracked tree, whole (a build script may name any file)."""
    proc = subprocess.Popen(["git", "-C", str(root), "archive", "--format=tar", commit],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for member in tar:
            if (member.isfile() or member.isdir()) and ".." not in member.name.split("/"):
                tar.extract(member, dest, filter="data")
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise SystemExit("prebuild: git archive %s failed: %s" % (commit[:12], err.strip()))


def linux_toolchain(tree, plat="linux-64"):
    """bincache.toolchain as the Linux box computes it, whatever runs this."""
    lock = Path(tree) / "pixi.lock"
    if not lock.is_file():
        return dict(platform=plat, packages=[], pixi_lock_sha256="")
    text = lock.read_text(errors="replace")
    pkgs = sorted(set(re.findall(
        r"conda\.modular\.com/max/%s/((?:mojo|mojo-compiler|max|max-core)-[^/\s]+)\.conda" % re.escape(plat),
        text)))
    return dict(platform=plat, packages=pkgs, pixi_lock_sha256=bincache.sha256_file(lock))


_ASSIGN = ("SCRIPTS", "FAST_CLASSICAL_SCRIPTS", "IDENTICAL_ONLY_SCRIPTS")


def enumerate_builds(tree, tiers=TIERS):
    """[(tier, script)] exactly as the commit's build_sets.sh schedules them in
    a release (MOJOLEARN_PACKAGE_BYTE_LM=1, every tier)."""
    tree = Path(tree)
    text = (tree / "packaging" / "linux" / "build_sets.sh").read_text()
    lines = []
    for name in _ASSIGN:
        m = re.search(r'^%s="(?:\$\{MOJOLEARN_BUILD_SCRIPTS:-)?([^"}]*)\}?"' % name, text, re.M)
        if not m:
            raise SystemExit("prebuild: build_sets.sh has no %s= line; the enumerator needs updating" % name)
        lines.append("%s=%s" % (name, shlex.quote(m.group(1))))
    m = re.search(r"^tier_scripts\(\) \{\n.*?^\}\n", text, re.M | re.S)
    if not m:
        raise SystemExit("prebuild: build_sets.sh has no tier_scripts(); the enumerator needs updating")
    fams = subprocess.run([sys.executable, str(tree / "python" / "mojolearn" / "host_surface.py"),
                           "--wheel-families"], capture_output=True, text=True, check=True).stdout.split()
    body = "\n".join(lines + ["PACKAGE_BYTE_LM=1", "HOST_FAMILIES=%s" % shlex.quote(" ".join(fams)), m.group(0),
                              'for t in %s; do echo "$t $(tier_scripts "$t")"; done' % " ".join(tiers)])
    out = subprocess.run(["bash", "-c", body], capture_output=True, text=True, check=True).stdout
    builds = []
    for line in out.splitlines():
        parts = line.split()
        if parts:
            builds += [(parts[0], s) for s in parts[1:]]
    return builds


def declared_output(tier, script):
    """build_sets.sh's declared_output, the one .so a build writes."""
    if script.endswith("_host.sh"):
        return "python/mojolearn/host/_mojolearn_%s.so" % script[len("build_"):-len(".sh")]
    name = "_mojolearn" if script == "build.sh" else "_mojolearn_" + script[len("build_"):-len(".sh")]
    d = "python/mojolearn" if tier == "fast" else "python/mojolearn/" + tier
    return "%s/%s.so" % (d, name)


def kind_of(tier, script):
    return "host" if script.endswith("_host.sh") else "gpu:" + tier


class CommitFacts:
    """The commit-side half of every key, cached per script."""

    def __init__(self, tree):
        self.tree = Path(tree).resolve()
        self.toolchain = linux_toolchain(self.tree)
        self._src = {}

    def source(self, script):
        if script not in self._src:
            path = self.tree / "bindings" / script
            src, _ = bincache.source_digest(self.tree, str(path), [])
            self._src[script] = (src, bincache.local_inputs(self.tree, str(path)))
        return self._src[script]


def predict(facts, template, tier, script):
    """(key, fields) the release box will compute for this build."""
    src, inputs = facts.source(script)
    fields = json.loads(json.dumps(template))
    fields["script"] = "bindings/" + script
    fields["args"] = []
    fields["source"] = src
    fields["toolchain"] = facts.toolchain
    fields["declared"] = dict(fields.get("declared", {}), inputs=inputs, outputs=[declared_output(tier, script)])
    return bincache.key_of(fields), fields


# ---------------------------------------------------------------- profiles
def load_profiles(path=PROFILES):
    d = json.loads(Path(path).read_text())
    if d.get("schema") != PROFILE_SCHEMA:
        raise SystemExit("prebuild: %s is not a %s file" % (path, PROFILE_SCHEMA))
    return d["targets"]


def partition_of(template):
    slug = re.sub(r"[^A-Za-z0-9._-]", "-", template["image"])[:100]
    return "%s/%s" % (template["device_arch"], slug)


def capture(keys_dir):
    """A profile's templates from one leg's keys/<key>.json: every field that
    is not the commit's, per kind (host, gpu:<tier>). Refuses a leg whose
    builds of one kind disagree, since one template could not reproduce it."""
    templates = {}
    for p in sorted(Path(keys_dir).glob("*.json")):
        f = json.loads(p.read_text())
        if bincache.key_of(f) != p.stem:
            raise SystemExit("prebuild: %s does not hash to its name" % p)
        if "declared" not in f or f.get("variant"):
            continue
        script = f["script"].split("/")[-1]
        t = {k: v for k, v in f.items() if k not in COMMIT_FIELDS}
        t["declared"] = {k: v for k, v in f["declared"].items() if k not in ("inputs", "outputs")}
        kind = kind_of(f["numeric_mode"], script)
        if kind in templates and templates[kind] != t:
            raise SystemExit("prebuild: two %s builds in %s carry different box fields" % (kind, keys_dir))
        templates[kind] = t
    if not templates:
        raise SystemExit("prebuild: no declared (release) build keys in %s" % keys_dir)
    return templates


# ---------------------------------------------------------------- the cache
def r2_creds(environ=None):
    """The four R2 variables from the environment, else from ~/.mojolearn_r2
    (shell assignments), else None."""
    environ = dict(os.environ if environ is None else environ)
    names = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")
    if all(environ.get(n) for n in names):
        return {n: environ[n] for n in names}
    path = Path(environ.get("MOJOLEARN_R2_CREDS", str(Path.home() / ".mojolearn_r2")))
    if not path.is_file():
        return None
    got = {}
    for line in path.read_text().splitlines():
        m = re.match(r"^\s*(?:export\s+)?(R2_[A-Z_]+)=(.*)$", line)
        if m:
            got[m.group(1)] = m.group(2).strip().strip("'\"")
    return {n: got[n] for n in names} if all(got.get(n) for n in names) else None


def cached_keys(creds, partition, lister=None):
    """The set of production keys present in a partition."""
    lister = lister or bincache.list_objects
    prefix = "%s/%s/" % (bincache.OBJECT_PREFIX, partition)
    keys = set()
    for name, _, _ in lister(creds, prefix):
        m = re.match(r"^%s([0-9a-f]{64})\.tar\.gz$" % re.escape(prefix), name)
        if m:
            keys.add(m.group(1))
    return keys


# ---------------------------------------------------------------- the plan
def make_plan(commit, tree, profiles, targets, creds, lister=None, base_tree=None):
    facts = CommitFacts(tree)
    base = CommitFacts(base_tree) if base_tree else None
    builds = enumerate_builds(tree)
    out = dict(schema=PLAN_SCHEMA, commit=commit, builds_per_target=len(builds), targets={},
               cache="checked" if creds else "unknown (no R2 credentials)")
    for name in targets:
        prof = profiles[name]
        tmpl = prof["templates"]
        partition = partition_of(next(iter(tmpl.values())))
        have = cached_keys(creds, partition, lister) if creds else None
        rows = []
        for tier, script in builds:
            kind = kind_of(tier, script)
            if kind not in tmpl:
                rows.append(dict(tier=tier, script=script, key=None, state="no-template:" + kind))
                continue
            if not (facts.tree / "bindings" / script).is_file():
                # build_sets.sh schedules a script the commit does not have: the
                # box's build fails, so there is nothing to predict or prebuild.
                rows.append(dict(tier=tier, script=script, key=None, state="no-script"))
                continue
            key, _ = predict(facts, tmpl[kind], tier, script)
            row = dict(tier=tier, script=script, key=key,
                       state="unknown" if have is None else ("cached" if key in have else "missing"))
            if base is not None:
                row["changed"] = (not (base.tree / "bindings" / script).is_file()
                                  or predict(base, tmpl[kind], tier, script)[0] != key)
            rows.append(row)
        n = {s: sum(1 for r in rows if r["state"] == s) for s in ("cached", "missing", "unknown")}
        n["no_template"] = sum(1 for r in rows if r["state"].startswith("no-template"))
        n["no_script"] = sum(1 for r in rows if r["state"] == "no-script")
        if base is not None:
            n["changed"] = sum(1 for r in rows if r.get("changed"))
        out["targets"][name] = dict(partition=partition, served_by=prof.get("served_by", ""),
                                    route=prof["route"], counts=n, rows=rows)
    return out


def render(plan):
    lines = ["prebuild plan for %s: %d builds per target, cache %s"
             % (plan["commit"][:12], plan["builds_per_target"], plan["cache"])]
    todo = 0
    for name, t in plan["targets"].items():
        c = t["counts"]
        extra = (", %d changed since base" % c["changed"]) if "changed" in c else ""
        if plan["cache"] == "checked":
            lines.append("  %-12s %3d missing, %3d cached%s  [%s]" % (name, c["missing"], c["cached"], extra,
                                                                    t["partition"]))
            todo += c["missing"]
        else:
            lines.append("  %-12s %3d keys, cache state unknown%s  [%s]" % (name, c["unknown"], extra, t["partition"]))
        if c["no_template"]:
            lines.append("    %d builds have no profile template (capture a fresh leg)" % c["no_template"])
        if c["no_script"]:
            lines.append("    %d scheduled build scripts do not exist at this commit" % c["no_script"])
        for r in t["rows"]:
            if r["state"] == "missing":
                lines.append("    missing %-13s %-32s %s" % (r["tier"], r["script"], r["key"][:16]))
    if plan["cache"] == "checked":
        lines.append("nothing to do" if todo == 0 else "%d bindings to prebuild" % todo)
    return "\n".join(lines)


def markdown(plan):
    md = ["## Binding prebuild plan", "",
          "Commit `%s`, %d builds per target, cache %s." % (plan["commit"][:12], plan["builds_per_target"],
                                                            plan["cache"]), "",
          "| target | missing | cached | changed since base | partition | read by |",
          "|---|---:|---:|---:|---|---|"]
    for name, t in plan["targets"].items():
        c = t["counts"]
        miss = c["missing"] if plan["cache"] == "checked" else "?"
        hit = c["cached"] if plan["cache"] == "checked" else "?"
        md.append("| %s | %s | %s | %s | `%s` | %s |" % (name, miss, hit, c.get("changed", "-"),
                                                        t["partition"], t["served_by"]))
    md.append("")
    for name, t in plan["targets"].items():
        miss = [r for r in t["rows"] if r["state"] == "missing"]
        if miss:
            md.append("<details><summary>%s: %d missing</summary>\n" % (name, len(miss)))
            md += ["- `%s` %s `%s`" % (r["script"], r["tier"], r["key"][:16]) for r in miss]
            md.append("\n</details>")
    return "\n".join(md) + "\n"


# ---------------------------------------------------------------- key check
def check_keys(facts, templates, keys_dir, builds):
    """(matched, box_only, predicted_only): the box's recorded keys against
    the planner's for the same builds."""
    box = {p.stem for p in Path(keys_dir).glob("*.json")
           if "declared" in json.loads(p.read_text()) and not json.loads(p.read_text()).get("variant")}
    pred = set()
    for tier, script in builds:
        kind = kind_of(tier, script)
        if kind in templates and (facts.tree / "bindings" / script).is_file():
            pred.add(predict(facts, templates[kind], tier, script)[0])
    return sorted(box & pred), sorted(box - pred), sorted(pred - box)


# ---------------------------------------------------------------- rent
def leg_command(target, prof, commit, out, gpu):
    route = prof["route"]
    if route == "gemm-campaign7":
        arch = target.split("-", 1)[1]
        env = dict(MOJOLEARN_NVIDIA_CAMPAIGN="7", MOJOLEARN_GPU_ARCHS=arch, MOJOLEARN_GEMM_LEG_OUT=str(out),
                   MOJOLEARN_RUNPOD_KEY_FILE=os.path.expanduser(
                       os.environ.get("MOJOLEARN_RUNPOD_KEY_FILE", "~/.mojolearn_runpod_key")))
        cmd = ["sh", "tools/gemm_remote_leg.sh", "nvidia", "--payload", "mamba", "--source-ref", commit,
               "--gpu", gpu, "--allow-concurrent", "--rent",
               "--segment-lease", os.environ.get("MOJOLEARN_PREBUILD_LEASE", "90"),
               "--dollar-cap", os.environ.get("MOJOLEARN_PREBUILD_DOLLAR_CAP", "6")]
        return cmd, env, out / "remote" / "release-build" / "build" / "bincache" / "keys"
    if route == "cpu-pod":
        cmd = ["bash", "tools/release_linux_build.sh", commit, "--archs", target.split("-", 1)[1],
               "--out", str(out), "--rent"]
        return cmd, {}, out / "LEG" / "remote" / "leg_out" / "bincache" / "keys"
    raise SystemExit("prebuild: unknown route %r" % route)


#: The leg machinery the rental runs with, taken from THIS checkout. None of it
#: is native-build input (tools/check_linux_release_qualification.py
#: is_native_source), so overlaying it leaves the release source inventory the
#: campaign-7 leg compares against the archive exactly the commit's.
LEG_OVERLAY = ("tools/gemm_remote_leg.sh", "tools/bincache.py", "tools/bincache_leg.sh")


class CommitWorktree:
    """A detached worktree at the commit with LEG_OVERLAY copied in.

    The campaign-7 leg refuses to ship when the checkout it runs from carries
    a native-source inventory other than the archive's ("Release archive
    lacks or changes canonical current native source"), so it must run from
    the commit itself, as tools/release.py runs it from the release worktree.
    Removed afterwards."""

    def __init__(self, commit):
        base = Path(os.environ.get("MOJOLEARN_RELEASE_WORKTREE_ROOT", str(Path.home() / "mojolearn-wt")))
        self.path = base / ("prebuild-%s-%d" % (commit[:9], os.getpid()))
        self.commit = commit

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        git("worktree", "add", "--detach", str(self.path), self.commit)
        changed = []
        for rel in LEG_OVERLAY:
            if (self.path / rel).read_bytes() != (ROOT / rel).read_bytes():
                (self.path / rel).write_bytes((ROOT / rel).read_bytes())
                changed.append(rel)
        if changed:
            # The leg's clean-tree gate refuses a MODIFIED file among its own
            # paths, so the overlay is committed on the detached HEAD (no
            # branch moves). The box still gets `git archive` of the commit.
            git("-C", str(self.path), "add", "--", *changed)
            git("-C", str(self.path), "-c", "user.name=prebuild", "-c", "user.email=prebuild@localhost",
                "commit", "-q", "--no-verify", "-m", "prebuild leg overlay (not pushed)")
        return self.path

    def __exit__(self, *exc):
        subprocess.run(["git", "-C", str(ROOT), "worktree", "remove", "--force", str(self.path)],
                       capture_output=True)
        return False


def no_stock(out):
    for p in (out / "create_response.json", out / "remote_start.log"):
        try:
            if NO_STOCK in p.read_text(errors="replace"):
                return True
        except OSError:
            pass
    return False


def rent(plan, commit, profiles, facts, builds, say=print):
    """One leg per target with missing keys; then re-plan and check keys."""
    results = {}
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d_%H%M%S")
    for name, t in plan["targets"].items():
        if t["counts"]["missing"] == 0:
            say("%s: nothing missing, no rental" % name)
            continue
        prof = profiles[name]
        walk = GPU_WALK.get(name.split("-", 1)[1], [None]) if prof["route"] == "gemm-campaign7" else [None]
        for gpu in walk:
            out = EVIDENCE / "prebuild" / commit[:12] / ("%s-%s%s" % (name, stamp, "-" + re.sub(r"\W+", "", gpu) if gpu else ""))
            out.parent.mkdir(parents=True, exist_ok=True)
            cmd, env, keys_dir = leg_command(name, prof, commit, out, gpu)
            say("%s: %d missing; renting %s -> %s" % (name, t["counts"]["missing"], gpu or "a CPU pod", out))
            t0 = time.time()
            with open(str(out) + ".log", "w") as log:
                if prof["route"] == "gemm-campaign7":
                    with CommitWorktree(commit) as wt:
                        rc = subprocess.call(cmd, cwd=str(wt), env=dict(os.environ, **env), stdout=log,
                                             stderr=subprocess.STDOUT)
                else:   # tools/release_linux_build.sh makes its own worktree at the commit
                    rc = subprocess.call(cmd, cwd=str(ROOT), env=dict(os.environ, **env), stdout=log,
                                         stderr=subprocess.STDOUT)
            wall = time.time() - t0
            if rc != 0 and gpu and no_stock(out):
                say("  %s: no stock, walking on" % gpu)
                continue
            results[name] = dict(rc=rc, wall_seconds=round(wall), out=str(out), gpu=gpu, keys_dir=str(keys_dir))
            if keys_dir.is_dir():
                ok, box_only, pred_only = check_keys(facts, prof["templates"], keys_dir, builds)
                results[name].update(keys_matched=len(ok), box_only=box_only, predicted_only=pred_only)
                say("  key check: %d matched, %d the box computed that the planner did not, %d the reverse"
                    % (len(ok), len(box_only), len(pred_only)))
            break
    return results


# ---------------------------------------------------------------- main
def resolve_commit(ref):
    return git("rev-parse", ref + "^{commit}").strip()


def main(argv=None):
    ap = argparse.ArgumentParser(prog="prebuild_changed.py")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--commit", default="origin/main")
    p.add_argument("--base", help="also count keys that changed since this commit")
    p.add_argument("--targets", help="comma list (default: every profile)")
    p.add_argument("--json")
    p.add_argument("--summary", help="append a markdown summary here ($GITHUB_STEP_SUMMARY)")
    p.add_argument("--no-cache-check", action="store_true")
    p.add_argument("--rent", action="store_true", help="build the missing keys on one box per target")
    p.add_argument("--profiles", default=str(PROFILES))
    c = sub.add_parser("check-keys")
    c.add_argument("--commit", required=True)
    c.add_argument("--target", required=True)
    c.add_argument("--profiles", default=str(PROFILES))
    c.add_argument("keys_dir")
    k = sub.add_parser("capture")
    k.add_argument("--target", required=True)
    k.add_argument("--route", choices=("gemm-campaign7", "cpu-pod"))
    k.add_argument("--served-by")
    k.add_argument("--source", help="free text naming the leg (recorded)")
    k.add_argument("--profiles", default=str(PROFILES))
    k.add_argument("keys_dir")
    a = ap.parse_args(argv)

    if a.cmd == "capture":
        path = Path(a.profiles)
        doc = json.loads(path.read_text()) if path.is_file() else dict(schema=PROFILE_SCHEMA, targets={})
        old = doc["targets"].get(a.target, {})
        doc["targets"][a.target] = dict(
            route=a.route or old.get("route") or ("cpu-pod" if "gfx" in a.target else "gemm-campaign7"),
            served_by=a.served_by or old.get("served_by", ""),
            captured_from=a.source or str(a.keys_dir),
            templates=capture(a.keys_dir))
        path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
        t = doc["targets"][a.target]
        print("captured %s: %s, kinds %s" % (a.target, partition_of(next(iter(t["templates"].values()))),
                                             " ".join(sorted(t["templates"]))))
        return 0

    profiles = load_profiles(a.profiles)
    with tempfile.TemporaryDirectory(prefix="prebuild-") as td:
        commit = resolve_commit(a.commit)
        tree = Path(td) / "tree"
        tree.mkdir()
        extract_tree(commit, tree)
        if a.cmd == "check-keys":
            facts = CommitFacts(tree)
            ok, box_only, pred_only = check_keys(facts, profiles[a.target]["templates"], a.keys_dir,
                                                 enumerate_builds(tree))
            print("check-keys %s at %s: %d matched, %d box-only, %d predicted-only"
                  % (a.target, commit[:12], len(ok), len(box_only), len(pred_only)))
            for key in box_only:
                f = json.loads((Path(a.keys_dir) / (key + ".json")).read_text())
                print("  box-only %s %s %s" % (key[:16], f["numeric_mode"], f["script"]))
            return 1 if box_only else 0
        base_tree = None
        if a.base:
            base_tree = Path(td) / "base"
            base_tree.mkdir()
            extract_tree(resolve_commit(a.base), base_tree)
        targets = a.targets.split(",") if a.targets else sorted(profiles)
        unknown = [t for t in targets if t not in profiles]
        if unknown:
            raise SystemExit("prebuild: no profile for %s (have %s)" % (",".join(unknown), ",".join(sorted(profiles))))
        creds = None if a.no_cache_check else r2_creds()
        plan = make_plan(commit, tree, profiles, targets, creds, base_tree=base_tree)
        print(render(plan))
        if a.rent:
            if creds is None:
                raise SystemExit("prebuild: --rent needs the cache listed first (R2 credentials)")
            facts = CommitFacts(tree)
            plan["rent"] = rent(plan, commit, profiles, facts, enumerate_builds(tree))
            after = make_plan(commit, tree, profiles, targets, creds)
            plan["after"] = {n: t["counts"] for n, t in after["targets"].items()}
            print("after the rental:")
            print(render(after))
        if a.json:
            Path(a.json).write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
        if a.summary:
            with open(a.summary, "a") as fh:
                fh.write(markdown(plan))
        if a.rent:
            bad = [n for n, r in plan["rent"].items() if r.get("box_only") or r["rc"] != 0]
            return 1 if bad or any(c["missing"] for c in plan["after"].values()) else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
