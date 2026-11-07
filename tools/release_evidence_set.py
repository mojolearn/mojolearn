#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE RELEASE EVIDENCE SET: one committed copy, overwritten by every release (Andrew 2026-11-06).

The verdict-bearing outputs of the checks a release runs before (and right after) it publishes are kept as
evidence in git, as ONE set at ONE fixed path, EVIDENCE_DIR. Each release's `record` step (tools/release.py)
deletes the previous set and writes its own, then commits it; the release tag v<version> points at that
commit, so an older release's evidence is `git show v<version>:bench/results/release_evidence/manifest.json`
(or any file of the set). No dated directory per release: the dated records releases up to 0.8.65 wrote under
bench/results/release_verification/ were removed from the tree on 2026-11-06 (0.8.65's record became the
first current set); they stay in git history, and provenance_history() reads them through git.

WHY bench/results/release_evidence/: every tool already treats bench/results/ as evidence that cannot move a
bit (tools/lane_select.py INERT_PREFIXES, tools/identity_code_equal.py INERT), so the record commit never looks
like a source change to the identity-at-freeze check, the lane selector or the reuse planner; lean lane
worktrees (tools/lean_worktree.sh) leave bench/results/ out, so lanes never pay for it; nothing under bench/
is packaged into a wheel.

WHAT GOES IN is the one table ITEMS below (required vs optional is stated there and nowhere else). Small
text/JSON only. Logs, binaries, wheels, the full reuse plan, the full identity table, compile receipts and box
job trees stay in ~/mojolearn-evidence and R2: the manifest REFERENCES them by path (or R2 tag) and sha256.

FAIL CLOSED: a missing required item fails with the item's name; a copied report that names another version
or source commit is stale and fails; a file over FILE_CAP or a set over TOTAL_CAP fails; nothing is taken
from the previous set (sources under EVIDENCE_DIR are refused). SCRUBBED: tokens, signed URLs, IP addresses
and e-mail addresses are redacted from text items; in a VERBATIM item (one another tool verifies by hash, such
as binding-identities.json or the PTX proof) a hit fails instead, because editing it would break its hash.

SUMMARY.md (2026-11-07): the human-readable account at the top of the set, written by write_set from the items
it collected and the references (summary_markdown; nothing is run for it): the release, the five identity
columns (machine, lanes, the exact wheel bytes, times, lane seconds), the admission verdict, the wheels, the
multi-GPU and measurement checks, how to verify on your own machine, and where the full logs are.

Tests: tools/test_release_evidence_set.py (no network, no box).
"""
import dataclasses
import datetime as dt
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform as _platform
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent.parent
#: THE fixed directory (relative to the repository root); each release replaces its whole content
EVIDENCE_DIR = "bench/results/release_evidence"
#: the dated per-release records written up to 0.8.65; removed from the tree 2026-11-06, read through git only
LEGACY_DIR = "bench/results/release_verification"
SCHEMA = "mojolearn.release-evidence.v1"
#: the release tag the record step puts on the commit holding the set
TAG_FORMAT = "v{version}"
#: Size caps. What the items weigh today (0.8.65/0.8.66): binding-identities.json 0.63 MB (0.79 MB at its
#: largest, 0.8.29), the gzipped PTX witnesses 0.2 MB each, every gate or publish report under 50 kB; one set
#: about 1.5 MB. A single file over 2 MiB or a set over 6 MiB is something large that belongs in R2.
FILE_CAP = 2 * 1024 * 1024
TOTAL_CAP = 6 * 1024 * 1024
#: Platforms (tools/release.py PIPELINES) and the vendor names their gates and published-* steps use
GPU_VENDOR_OF = {"nvidia": "nvidia", "amd": "amd"}
PUBLISHED_VENDOR_OF = {"macos": "apple", "nvidia": "nvidia", "amd": "amd"}
#: The wheels each vendor's checks ran against (the plugin installs with the core)
CHECKED_WHEELS = {"nvidia": ("nvidia", "linux"), "amd": ("amd", "linux"), "apple": ("macos",),
                  "cpu": ("linux", "macos")}


class EvidenceError(RuntimeError):
    """The evidence set cannot be written (a missing required item, a cap, a secret in a verbatim item)."""


# ------------------------------------------------------------------------------------------- the context
@dataclasses.dataclass
class Context:
    """Everything the collector reads, from the release controller (tools/release.py Release.evidence_set_context)."""
    version: str
    commit: str                       # the frozen source commit
    tooling_commit: str
    date: str                         # the CHANGELOG publication date
    rel: Path                         # ~/mojolearn-evidence/release/<version>/<commit12>
    staged: Path                      # the record files the record step wrote for this release
    publish: str = "pypi"
    published: list = dataclasses.field(default_factory=list)   # platforms: linux nvidia amd linux-meta macos
    wheels: dict = dataclasses.field(default_factory=dict)      # platform -> wheel Path (None when not built)
    single: bool = False              # --identity-on-wheels: the one verification round on the built wheels
    gate_dirs: dict = dataclasses.field(default_factory=dict)   # nvidia/amd -> the gate report directory
    steps: dict = dataclasses.field(default_factory=dict)       # step -> recorded {done, at, result, ...}
    identity_admitted: dict = None
    adapters: bool = True             # MOJOLEARN_RELEASE_ADAPTERS != 0
    table: Path = None                # the identity table this release ships (file)
    table_ref: str = ""               # where that table is committed (commit:path)
    box_tags: list = dataclasses.field(default_factory=list)
    tools: dict = dataclasses.field(default_factory=dict)


# ------------------------------------------------------------------------------------------- THE ITEM TABLE
#: (dest in the set, producer step, when it is required, source).
#: dest may hold {platform} (each published platform), {vendor} (each gated GPU vendor whose plugin was
#: published: nvidia, amd) or {pvendor} (each published-<vendor> step: apple, nvidia, amd).
#: when:  always       every release
#:        single       --identity-on-wheels (the one verification round on the built wheels)
#:        default      without --identity-on-wheels (the gate's release-light-v1 stages)
#:        linux        the Linux core was published
#:        macos        the macOS wheel was published
#:        adapters     the macOS wheel was published with the adapter gate on
#:        optional     kept when present, listed as absent otherwise
#: source: staged:<name>   written by the record step itself (tools/release.py step_record)
#:         rel:<path>      under the release evidence directory of this freeze
#:         gate:<path>     under the vendor's gate report directory (tools/release.py gate_out)
#:         derived:<name>  computed here from the release state and reports (see DERIVED)
ITEMS = (
    # the published wheels' own records: tools/release_reuse.py previous_release reads these from the set
    ("alpha-manifest-{platform}.json", "publish-{platform}", "always", "staged:alpha-manifest-{platform}.json"),
    ("file-admission-{platform}.json", "publish-{platform}", "always", "staged:file-admission-{platform}.json"),
    ("release-provenance.json", "record", "always", "staged:release-provenance.json"),
    ("binding-identities.json", "record", "always", "staged:binding-identities.json"),
    ("self-test-{platform}.json", "macos-smoke", "optional", "staged:self-test-{platform}.json"),
    ("light-smoke-{platform}.json", "macos-smoke", "optional", "staged:light-smoke-{platform}.json"),
    ("ptx-proof/ptx-proof.json", "linux-builds", "optional", "staged:ptx-proof/ptx-proof.json"),
    ("ptx-proof/PTX_BASELINE.json.gz", "linux-builds", "optional", "staged:ptx-proof/PTX_BASELINE.json.gz"),
    ("ptx-proof/build-provenance.json.gz", "linux-builds", "optional", "staged:ptx-proof/build-provenance.json.gz"),
    ("ptx-proof/manifest.json.gz", "linux-builds", "optional", "staged:ptx-proof/manifest.json.gz"),
    # every step of this release and its recorded result
    ("steps.json", "record", "always", "derived:steps"),
    # the one verification round: identity
    ("identity/at-freeze.json", "identity-at-freeze", "always", "rel:identity-at-freeze/report.json"),
    ("identity/table-summary.json", "identity-admit", "always", "derived:table"),
    ("identity/admitted.json", "identity-admit", "single", "derived:admitted"),
    ("identity/{vendor}-identity-wheels.json", "gate-{vendor}", "single", "gate:identity/identity-wheels.json"),
    ("identity/apple-identity-wheels.json", "gate-apple", "single", "rel:identity-wheels/apple/identity-wheels.json"),
    ("identity/cpu-identity-wheels.json", "identity-cpu", "single", "derived:cpu-identity"),
    # gate verdicts per vendor
    ("gates/{vendor}/gate-smoke.json", "gate-{vendor}", "default", "gate:gate-smoke.json"),
    ("gates/{vendor}/runtime-report.json", "gate-{vendor}", "always", "gate:runtime-{vendor}/runtime-report.json"),
    ("gates/{vendor}/check-units.json", "gate-{vendor}", "single", "gate:units/check-units.json"),
    ("gates/{vendor}/shardcheck.json", "gate-{vendor}", "single", "gate:shardcheck/shardcheck.json"),
    ("gates/{vendor}/measuresmoke.json", "gate-{vendor}", "single", "gate:measuresmoke/measuresmoke.json"),
    ("gates/{vendor}/autograd-smoke.json", "gate-{vendor}", "optional", "gate:autograd-smoke.json"),
    ("gates/{vendor}/adapter-smoke.json", "gate-{vendor}", "optional", "gate:adapter-smoke/adapter-smoke.json"),
    ("gates/{vendor}/job-status.json", "gate-{vendor}", "optional", "gate:job/status.json"),
    # OWED: optional until the first release whose gate job wrote job/box-facts.txt (tools/ops/release_gate_job.sh)
    ("gates/{vendor}/box-facts.json", "gate-{vendor}", "optional", "derived:box-facts"),
    ("gates/apple/adapter-smoke.json", "gate-apple", "adapters", "rel:gate-apple/adapters/adapter-smoke.json"),
    ("gates/apple/runtime-report.json", "gate-apple", "optional", "rel:gate-apple/runtime-apple/runtime-report.json"),
    ("gates/cpu/checks.json", "identity-cpu", "single", "rel:identity-wheels/cpu/artifacts/cpu-wheel-checks/checks.json"),
    # the pre-release tool tests and the compile evidence
    ("checks/rehearsal.json", "rehearsal", "always", "derived:rehearsal"),
    ("compile/cross-compile.json", "cross-compile", "linux", "derived:cross-compile"),
    ("compile/bindings.json", "reuse-plan", "always", "derived:bindings"),
    # publication: the TestPyPI rehearsal before it, the byte checks after it
    ("publish/testpypi-rehearsal.json", "testpypi-rehearsal", "always", "rel:testpypi-rehearsal/rehearsal.json"),
    ("publish/published-{pvendor}.json", "published-{pvendor}", "always", "rel:published-{pvendor}/published-report.json"),
    ("publish/finish-line.json", "finish-line", "always", "derived:finish-line"),
)
#: Items another tool verifies by hash or reads as recorded: a secret in one fails the set, it is never edited
VERBATIM_PREFIXES = ("alpha-manifest-", "file-admission-", "release-provenance.json", "binding-identities.json",
                     "ptx-proof/")


def expand(ctx):
    """[(dest, producer, required, source, against)] of ITEMS for this release."""
    gated = [GPU_VENDOR_OF[p] for p in ctx.published if p in GPU_VENDOR_OF]
    pvendors = [PUBLISHED_VENDOR_OF[p] for p in ctx.published if p in PUBLISHED_VENDOR_OF]
    rows = []
    for dest, producer, when, source in ITEMS:
        if "{platform}" in dest:
            keys = [("platform", p) for p in ctx.published]
        elif "{vendor}" in dest:
            keys = [("vendor", v) for v in gated]
        elif "{pvendor}" in dest:
            keys = [("pvendor", v) for v in pvendors]
        else:
            keys = [(None, None)]
        for key, value in keys:
            fmt = {key: value} if key else {}
            d, p, s = dest.format(**fmt), producer.format(**fmt), source.format(**fmt)
            if when == "macos" or when == "adapters" or d.startswith("gates/apple/") or d.startswith("identity/apple"):
                if "macos" not in ctx.published:
                    continue
            if d.startswith("gates/cpu/") or d.startswith("identity/cpu"):
                if not ctx.single:
                    continue
            if d == "compile/cross-compile.json" and "linux" not in ctx.published:
                continue
            required = dict(always=True, single=ctx.single, default=not ctx.single, linux=True, macos=True,
                            adapters=ctx.adapters, optional=False)[when]
            if when in ("single",) and not ctx.single:
                continue   # not part of this release's mode at all
            rows.append((d, p, required, s, against(ctx, key, value, d)))
    return rows


def against(ctx, key, value, dest):
    """The platforms whose exact wheel bytes the item was produced against."""
    if key == "platform":
        return [value]
    if key in ("vendor", "pvendor"):
        return [p for p in CHECKED_WHEELS.get(value, ()) if p in ctx.wheels]
    if dest.startswith("gates/apple") or dest.startswith("identity/apple"):
        return ["macos"]
    if dest.startswith("gates/cpu") or dest.startswith("identity/cpu"):
        return [p for p in CHECKED_WHEELS["cpu"] if p in ctx.wheels]
    if dest.startswith(("identity/", "publish/", "steps.json", "release-provenance", "binding-identities")):
        return sorted(ctx.wheels)
    return []


# ------------------------------------------------------------------------------------------- scrubbing
#: (kind, pattern, group to redact or 0 for the whole match)
SECRET_PATTERNS = (
    ("token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"), 0),
    ("token", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"), 0),
    ("token", re.compile(r"\bpypi-[A-Za-z0-9_-]{30,}"), 0),
    ("token", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"), 0),
    ("token", re.compile(r"\brpa_[A-Za-z0-9]{20,}"), 0),
    ("token", re.compile(r"\bdop_v1_[A-Fa-f0-9]{40,}"), 0),
    ("token", re.compile(r"\bhf_[A-Za-z0-9]{30,}"), 0),
    ("token", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}"), 0),
    ("token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"), 0),
    ("token", re.compile(r"\beyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}"), 0),
    ("token", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)"), 0),
    ("token", re.compile(r"(?i)\bbearer\s+([A-Za-z0-9._~+/=-]{16,})"), 1),
    ("token", re.compile(r"(?i)\b[a-z_]*(?:token|secret|password|passwd|api[_-]?key|access[_-]?key)\\?[\"']?"
                         r"\s*[:=]\s*\\?[\"']?(?!\[REDACTED)([^\"'\\\s,}\]]{8,})"), 1),
    ("signed-url", re.compile(r"(?i)\bhttps?://[^\s\"'<>]+?(\?[^\s\"'<>]*?\b(?:x-amz-signature|x-amz-credential|"
                              r"x-amz-security-token|signature|sig|token|access_token|expires|se|sv|sp)=[^\s\"'<>]*)"), 1),
    ("email", re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b"), 0),
    ("ip", re.compile(r"(?<![\w.])(?:(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)\.){3}(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
                      r"(?!\.?\d)(?!\.[A-Za-z])"), 0),
    ("ip", re.compile(r"(?i)(?<![\w:])(?:[0-9a-f]{1,4}:){7}[0-9a-f]{1,4}(?![\w:])"), 0),
)
#: addresses that name no machine of ours: loopback and the unspecified address
HARMLESS_IP = re.compile(r"^(?:127\.\d+\.\d+\.\d+|0\.0\.0\.0)$")
#: role addresses that name no person (commit trailers that older records embed)
HARMLESS_EMAIL = frozenset({"noreply@anthropic.com", "noreply@github.com"})


def scrub_text(text):
    """(text with every secret redacted, [kinds found])."""
    found = []
    for kind, pattern, group in SECRET_PATTERNS:
        def repl(m, kind=kind, group=group):
            if kind == "ip" and HARMLESS_IP.match(m.group(0)):
                return m.group(0)
            if kind == "email" and m.group(0).lower() in HARMLESS_EMAIL:
                return m.group(0)
            found.append(kind)
            if group:
                start, end = m.span(group)
                return m.string[m.start():start] + "[REDACTED:%s]" % kind + m.string[end:m.end()]
            return "[REDACTED:%s]" % kind
        text = pattern.sub(repl, text)
    return text, found


def scan_bytes(name, data):
    """[kinds] of the secrets in DATA (a .gz is scanned decompressed)."""
    if name.endswith(".gz"):
        try:
            data = gzip.decompress(data)
        except (OSError, EOFError):
            return ["unreadable-gzip"]
    return scrub_text(data.decode("utf-8", errors="replace"))[1]


# ------------------------------------------------------------------------------------------- reading
def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError, TypeError):
        return None


def home_label(path):
    """PATH with the home directory written as ~ (no user name in committed evidence)."""
    text = str(path)
    home = os.path.expanduser("~")
    return "~" + text[len(home):] if home and home != "~" and text.startswith(home) else text


def tree_digest(path):
    """(sha256 over the sorted (relative name, sha256) of every file under PATH, file count, bytes)."""
    h, count, size = hashlib.sha256(), 0, 0
    for f in sorted(p for p in Path(path).rglob("*") if p.is_file() and not p.is_symlink()):
        h.update(("%s\0%s\n" % (f.relative_to(path).as_posix(), sha256_file(f))).encode())
        count += 1
        size += f.stat().st_size
    return h.hexdigest(), count, size


# ------------------------------------------------------------------------------------------- derived items
def derived_steps(ctx):
    rows = {}
    for step, rec in sorted((ctx.steps or {}).items()):
        if not isinstance(rec, dict) or not rec.get("done") or rec.get("commit") not in (None, ctx.commit):
            continue
        rows[step] = {k: rec.get(k) for k in ("at", "result", "tooling_commit", "platforms") if k in rec}
    if not rows:
        return None
    return dict(schema="mojolearn.release-steps.v1", version=ctx.version, source_commit=ctx.commit, steps=rows)


def derived_finish_line(ctx):
    rec = (ctx.steps or {}).get("finish-line")
    if not isinstance(rec, dict) or not rec.get("done") or rec.get("commit") not in (None, ctx.commit):
        return None
    return dict(schema="mojolearn.release-finish-line.v1", version=ctx.version, source_commit=ctx.commit,
                at=rec.get("at"), result=rec.get("result"), platforms=rec.get("platforms"))


def derived_admitted(ctx):
    doc = ctx.identity_admitted
    if not isinstance(doc, dict) or doc.get("source_commit") not in (None, ctx.commit):
        return None
    keep = ("source_commit", "commit", "branch", "recorded", "table_sha256", "at")
    out = {k: doc.get(k) for k in keep if k in doc}
    out["repacked"] = sorted((doc.get("repacked") or {}).keys()) if isinstance(doc.get("repacked"), dict) \
        else doc.get("repacked")
    return dict(schema="mojolearn.release-identity-admitted.v1", version=ctx.version, **out)


def derived_table(ctx):
    """Per identity column: the lanes and cell parts it holds, and how many parts every column agrees on (one
    admitted digest listed under every column). The table itself is referenced, not copied."""
    if not ctx.table or not Path(ctx.table).is_file():
        return None
    table = read_json(ctx.table)
    if not isinstance(table, dict) or not isinstance(table.get("cells"), dict):
        return None
    columns, parts, agree, partial = {}, 0, 0, []
    all_cols = set()
    for cell in table["cells"].values():
        for ent in (cell or {}).values():
            all_cols |= set(((ent or {}).get("cols") or {}).keys())
    for key, cell in sorted(table["cells"].items()):
        lane = key.partition("/")[0]
        for part, ent in sorted((cell or {}).items()):
            if str((ent or {}).get("ref") or "").startswith("n/a:"):
                continue
            cols = set(((ent or {}).get("cols") or {}).keys())
            parts += 1
            for c in cols:
                row = columns.setdefault(c, dict(lanes=set(), parts=0))
                row["lanes"].add(lane)
                row["parts"] += 1
            if cols == all_cols:
                agree += 1
            elif len(partial) < 50:
                partial.append(dict(cell=key, part=part, missing=sorted(all_cols - cols)))
    return dict(schema="mojolearn.release-identity-table-summary.v1", version=ctx.version, source_commit=ctx.commit,
                table=dict(ref=ctx.table_ref, sha256=sha256_file(ctx.table), bytes=Path(ctx.table).stat().st_size),
                columns={c: dict(lanes=len(r["lanes"]), parts=r["parts"]) for c, r in sorted(columns.items())},
                parts=parts, parts_equal_on_every_column=agree, parts_not_on_every_column=parts - agree,
                first_parts_not_on_every_column=partial,
                ptx_divergent=len(table.get("ptx_divergent") or []))


def derived_cpu_identity(ctx):
    base = Path(ctx.rel) / "identity-wheels" / "cpu" / "artifacts"
    shards = {}
    for f in sorted(base.glob("cpu-wheel-identity-*/identity-wheels.json")):
        doc = read_json(f)
        if doc is not None:
            shards[f.parent.name] = doc
    if not shards:
        return None
    return dict(schema="mojolearn.release-cpu-identity.v1", version=ctx.version, source_commit=ctx.commit, shards=shards)


REHEARSAL_LINE = re.compile(r"^(PASS|FAIL) (\S+) \((\d+)s(?:, exit (-?\d+))?")


def derived_rehearsal(ctx):
    """The release rehearsal's per-check verdicts (pixi run release-rehearsal; its log is referenced)."""
    log = Path(ctx.rel) / "rehearsal.log"
    try:
        text = log.read_text(errors="replace")
    except OSError:
        return None
    checks, overall = {}, None
    for line in text.splitlines():
        m = REHEARSAL_LINE.match(line)
        if m:
            checks[m.group(2)] = dict(verdict=m.group(1), seconds=int(m.group(3)),
                                      **({"exit": int(m.group(4))} if m.group(4) else {}))
        elif line.startswith("# REHEARSAL "):
            overall = line[2:].strip()
    if not checks and not overall:
        return None
    return dict(schema="mojolearn.release-rehearsal-checks.v1", version=ctx.version, source_commit=ctx.commit,
                verdict=overall, checks=checks)


def derived_cross_compile(ctx):
    doc = read_json(Path(ctx.rel) / "cross-compile.json")
    if isinstance(doc, dict):
        return doc
    rec = (ctx.steps or {}).get("cross-compile")
    if isinstance(rec, dict) and rec.get("done") and rec.get("commit") in (None, ctx.commit):
        return dict(schema="mojolearn.release-cross-compile.v1", commit=ctx.commit, verdict="PASS" if str(
            rec.get("result", "")).startswith("PASS") else None, result=rec.get("result"), base=rec.get("base"),
            run=rec.get("run"), at=rec.get("at"))
    return None


def derived_bindings(ctx):
    """Which bindings were built and which were taken from which published release, and why (the reuse plan;
    the plan itself is referenced)."""
    plan = read_json(Path(ctx.rel) / "reuse" / "plan.json")
    if not isinstance(plan, dict) or not isinstance(plan.get("rows"), list):
        return None
    rows = [{k: r.get(k) for k in ("key", "decision", "reason", "reuse_from", "identity_digest")}
            for r in plan["rows"] if isinstance(r, dict)]
    counts = {}
    for r in rows:
        counts[r["decision"]] = counts.get(r["decision"], 0) + 1
    prev = plan.get("previous") or {}
    return dict(schema="mojolearn.release-compile-summary.v1", version=ctx.version, source_commit=plan.get("commit"),
                previous=dict(version=prev.get("version"), source_commit=prev.get("source_commit"),
                              channel=prev.get("channel")) if isinstance(prev, dict) else None,
                legs=plan.get("legs"), leg_reasons=plan.get("leg_reasons"), host_toolchain=plan.get("host_toolchain"),
                counts=counts, bindings=rows)


BOX_FACT_LINE = re.compile(r"^([a-z_]+)=(.*)$")


def derived_box_facts(ctx, vendor):
    """GPU and CPU model, cores, memory and the busy-core samples the gate job recorded on its box
    (tools/ops/release_gate_job.sh: job/box-facts.txt, job/busy-cores.tsv; job/box.txt for older jobs)."""
    job = Path(ctx.gate_dirs.get(vendor) or "/nonexistent") / "job"
    facts, gpus = {}, []
    try:
        for line in (job / "box-facts.txt").read_text(errors="replace").splitlines():
            m = BOX_FACT_LINE.match(line.strip())
            if not m:
                continue
            if m.group(1) == "gpu":
                gpus.append(m.group(2).strip())
            else:
                facts[m.group(1)] = m.group(2).strip()
    except OSError:
        pass
    samples = []
    try:
        for line in (job / "busy-cores.tsv").read_text(errors="replace").splitlines():
            f = line.split("\t")
            if len(f) == 3 and all(x.isdigit() for x in f):
                samples.append([int(f[0]), int(f[1]), int(f[2])])
    except OSError:
        pass
    if not facts and not samples:
        try:   # an older job: box.txt holds `uname -a` then `nproc`; keep the core count only (no host name)
            lines = (job / "box.txt").read_text(errors="replace").splitlines()
            cores = next((x.strip() for x in lines[1:3] if x.strip().isdigit()), None)
        except OSError:
            cores = None
        if not cores:
            return None
        facts = dict(nproc=cores)
    busy = sorted(s[1] for s in samples)
    return dict(schema="mojolearn.release-box-facts.v1", vendor=vendor, version=ctx.version, source_commit=ctx.commit,
                cpu_model=facts.get("cpu_model"), cores=int(facts["nproc"]) if str(facts.get("nproc", "")).isdigit()
                else None, mem_total_kb=int(facts["mem_total_kb"]) if str(facts.get("mem_total_kb", "")).isdigit()
                else None, kernel=facts.get("kernel"), gpus=gpus,
                busy_cores=dict(samples=len(samples), median=busy[len(busy) // 2] if busy else None,
                                max=busy[-1] if busy else None, of=samples[-1][2] if samples else None),
                busy_samples=samples[:2000])


DERIVED = {"steps": derived_steps, "finish-line": derived_finish_line, "admitted": derived_admitted,
           "table": derived_table, "cpu-identity": derived_cpu_identity, "rehearsal": derived_rehearsal,
           "cross-compile": derived_cross_compile, "bindings": derived_bindings}


# ------------------------------------------------------------------------------------------- collecting
def vendor_of(dest):
    m = re.match(r"^(?:gates|identity)/(nvidia|amd)\b|^identity/(nvidia|amd)-", dest)
    return (m.group(1) or m.group(2)) if m else None


def resolve(ctx, dest, source):
    """(kind, value): ('file', Path) | ('doc', dict) | ('missing', where)."""
    kind, _, name = source.partition(":")
    if kind == "derived":
        if name == "box-facts":
            doc = derived_box_facts(ctx, vendor_of(dest))
        else:
            doc = DERIVED[name](ctx)
        return ("doc", doc) if doc is not None else ("missing", "derived " + name)
    if kind == "staged":
        path = Path(ctx.staged) / name
    elif kind == "rel":
        path = Path(ctx.rel) / name
    elif kind == "gate":
        vendor = vendor_of(dest)
        base = ctx.gate_dirs.get(vendor)
        if not base:
            return "missing", "no gate directory for " + str(vendor)
        path = Path(base) / name
    else:
        raise ValueError("unknown source " + source)
    return ("file", path) if path.is_file() else ("missing", home_label(path))


def stale(ctx, doc):
    """Why a copied report names another release, or None."""
    if not isinstance(doc, dict):
        return None
    v = doc.get("version")
    if isinstance(v, str) and re.fullmatch(r"\d+(\.\d+)+", v) and v != ctx.version:
        return "it is for version %s, not %s" % (v, ctx.version)
    for key in ("source_commit", "commit"):
        c = doc.get(key)
        if isinstance(c, str) and re.fullmatch(r"[0-9a-f]{40}", c) and c != ctx.commit:
            return "its %s is %s, not the frozen %s" % (key, c[:12], ctx.commit[:12])
    return None


def collect(ctx, root=ROOT):
    """(entries, missing_required, missing_optional, problems). An entry is
    dict(dest, producer, required, against, data (bytes), source, source_sha256, redactions)."""
    fixed = (Path(root) / EVIDENCE_DIR).resolve()
    entries, missing, absent, problems = [], [], [], []
    for dest, producer, required, source, wheels in expand(ctx):
        kind, value = resolve(ctx, dest, source)
        if kind == "missing":
            (missing if required else absent).append(dict(path=dest, producer=producer, looked_for=value))
            continue
        verbatim = dest.startswith(VERBATIM_PREFIXES)
        if kind == "file":
            path = value.resolve()
            if path == fixed or fixed in path.parents:
                problems.append("%s: its source %s is inside the previous evidence set; nothing is carried over"
                                % (dest, value))
                continue
            data = path.read_bytes()
            src, src_sha = home_label(value), sha256_bytes(data)
            if not source.startswith("staged:") and dest.endswith(".json"):
                why = stale(ctx, read_json(path))
                if why:
                    problems.append("%s (from %s) is stale: %s" % (dest, src, why))
                    continue
        else:
            data = (json.dumps(value, indent=1, sort_keys=True, default=str) + "\n").encode()
            src, src_sha = "derived", None
        redactions = []
        if dest.endswith(".gz"):
            found = scan_bytes(dest, data)
            if found:
                problems.append("%s holds %s; a verbatim item is never edited" % (dest, ", ".join(sorted(set(found)))))
                continue
        else:
            text, redactions = scrub_text(data.decode("utf-8", errors="replace"))
            if redactions and verbatim:
                problems.append("%s holds %s; a verbatim item is never edited" % (dest, ", ".join(sorted(set(redactions)))))
                continue
            if redactions:
                data = text.encode()
        if len(data) > FILE_CAP:
            problems.append("%s is %d bytes, over the %d-byte cap per file: keep it in R2 or ~/mojolearn-evidence and "
                            "reference it" % (dest, len(data), FILE_CAP))
            continue
        entries.append(dict(dest=dest, producer=producer, required=required, against=wheels, data=data,
                            source=src, source_sha256=src_sha, redactions=len(redactions)))
    return entries, missing, absent, problems


def references(ctx):
    """What stays out of git, named by path (or R2 tag) and sha256."""
    refs = []

    def add_file(what, path):
        p = Path(path)
        if p.is_file():
            refs.append(dict(what=what, path=home_label(p), sha256=sha256_file(p), bytes=p.stat().st_size))

    def add_tree(what, path):
        p = Path(path)
        if p.is_dir():
            digest, count, size = tree_digest(p)
            refs.append(dict(what=what, path=home_label(p), tree_sha256=digest, files=count, bytes=size))

    rel = Path(ctx.rel)
    for platform, wheel in sorted(ctx.wheels.items()):
        if wheel:
            add_file("wheel " + platform, wheel)
    if ctx.table:
        p = Path(ctx.table)
        if p.is_file():
            refs.append(dict(what="identity table", ref=ctx.table_ref, sha256=sha256_file(p), bytes=p.stat().st_size))
    add_file("reuse plan", rel / "reuse" / "plan.json")
    for f in sorted((rel / "legs").glob("*.provenance.json")):
        add_file("leg provenance " + f.name.split(".")[0], f)
    add_tree("compile receipts", rel / "compile-receipts")
    for log in ("rehearsal.log", "identity-at-freeze.log", "identity-admit.log", "cross-compile.log",
                "linux-pack.log", "macos-build.log", "gate-apple.log"):
        add_file("log " + log, rel / log)
    for vendor, d in sorted((ctx.gate_dirs or {}).items()):
        add_tree("gate report tree " + vendor, d)
    add_tree("TestPyPI rehearsal tree", rel / "testpypi-rehearsal")
    for tag in ctx.box_tags or []:
        refs.append(dict(what="box job results in R2", r2_tag=tag, r2_key_root="measurements/box-jobs",
                         fetch="python3 tools/box_persist.py fetch --tag " + tag))
    return refs


README = """# Release evidence

This directory holds the evidence of the checks run for one Mojolearn release: the release named in
`manifest.json`. Each release replaces the whole directory, so it only ever holds one release's evidence.

Start with `SUMMARY.md`: what was released, which wheel bytes each identity column was recorded from, whether
the columns agree, the multi-GPU and measurement checks, how to run the same check on your own machine, and
where the full logs are.

What is here:

- `SUMMARY.md`: the human-readable account of this release's evidence, written from the files below.
- `manifest.json`: the release version, the source commit the wheels were built from, every wheel's file name
  and sha256, the tools used, and for every file below its sha256, the release step that produced it and the
  wheels it was checked against. It also lists what was not copied here (logs, the wheels, the full identity
  table, build receipts, box job results) by path or storage tag plus sha256.
- `identity/`: the one identity verification round: the check at the frozen source, the per-column summary of
  the identity table that shipped, and what each column recorded.
- `gates/`: the verdicts of the installed-wheel checks on each GPU vendor and on Apple and the CPU, including
  the multi-device shard check, the measurement smoke and the facts of the machine each ran on.
- `publish/`: the TestPyPI rehearsal before publishing and the byte checks of what PyPI serves after it.
- `checks/`, `compile/`: the release's tool tests and which compiled bindings were built or reused.
- the top-level JSON files: the published wheels' manifests and admissions, the release provenance and the
  binding identities the next release reuses from.

To see an earlier release's evidence, read it from that release's tag:

    git show v0.8.67:bench/results/release_evidence/manifest.json
    git ls-tree -r --name-only v0.8.67 -- bench/results/release_evidence/

Releases up to 0.8.65 kept one dated directory each under `bench/results/release_verification/`; those are
in git history (for example `git log --format=%H -n 1 -- bench/results/release_verification/` names the
commit that removed them; read them at its parent).
"""


# ------------------------------------------------------------------------------------------- SUMMARY.md
# THE HUMAN-READABLE ACCOUNT (Andrew 2026-11-07: "how do people know it works from that?"). Written by the record
# step from the data the release already holds (the items above, the column records next to each identity summary,
# the references); nothing is run for it. Links to the JSON, never pastes it. At most SUMMARY_MAX_LINES lines.
SUMMARY_MAX_LINES = 150
#: the five identity columns, in reading order: (column, where its identity-wheels.json enters the set)
IDENTITY_COLUMNS = (("nvidia", "identity/nvidia-identity-wheels.json"), ("ptx", "identity/nvidia-identity-wheels.json"),
                    ("amd", "identity/amd-identity-wheels.json"), ("apple", "identity/apple-identity-wheels.json"),
                    ("cpu", "identity/cpu-identity-wheels.json"))
COLUMN_TITLE = {"nvidia": "NVIDIA native", "ptx": "NVIDIA PTX", "amd": "AMD", "apple": "Apple", "cpu": "CPU"}
#: the gate vendor whose box job holds each column (its box-facts.json names the machine)
COLUMN_BOX = {"nvidia": "nvidia", "ptx": "nvidia", "amd": "amd"}
PROVENANCE_DATE = re.compile(r"^date=(\S+)$", re.M)


def identity_record_dirs(ctx, column):
    """[directories] under which this column's record files (<name>.identical.json) were written on this machine:
    the gate report tree (NVIDIA, PTX, AMD), the Apple step's output, every fetched CPU shard."""
    if column in COLUMN_BOX:
        base = ctx.gate_dirs.get(COLUMN_BOX[column])
        return [Path(base) / "identity"] if base else []
    if column == "apple":
        return [Path(ctx.rel) / "identity-wheels" / "apple"]
    return sorted(Path(ctx.rel).glob("identity-wheels/cpu/artifacts/cpu-wheel-identity-*"))


def column_record_facts(dirs, label):
    """What the column's own record files say: lanes, cells, total lane seconds (sum of every cell's timing),
    first start and last end (UTC), the machine's CPU model and platform. None when no record file is found."""
    files = []
    for d in dirs:
        files += sorted(Path(d).glob("%s/**/*.identical.json" % label)) if label else []
    if not files:
        return None
    lanes, cells, seconds, starts, ends, cpus, platforms = set(), 0, 0.0, [], [], set(), set()
    for f in files:
        doc = read_json(f)
        if not isinstance(doc, dict):
            continue
        for key, cell in (doc.get("cells") or {}).items():
            cells += 1
            lanes.add(str(key).partition("/")[0])
            t = ((cell or {}).get("timing") or {}).get("total_seconds") if isinstance(cell, dict) else None
            if isinstance(t, (int, float)):
                seconds += float(t)
        for env in [doc.get("timing_env")] + list(doc.get("timing_env_attempts") or []):
            if isinstance(env, dict):
                if env.get("started_utc"):
                    starts.append(str(env["started_utc"]))
                if env.get("cpu_model"):
                    cpus.add(str(env["cpu_model"]))
        if doc.get("platform"):
            platforms.add(str(doc["platform"]))
        prov = f.with_name(f.name[:-len(".identical.json")] + ".provenance.txt")
        try:
            ends += PROVENANCE_DATE.findall(prov.read_text(errors="replace"))
        except OSError:
            ends.append(dt.datetime.fromtimestamp(f.stat().st_mtime, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))
    return dict(records=len(files), lanes=len(lanes), cells=cells, lane_seconds=seconds,
                start=min(starts) if starts else None, end=max(ends) if ends else None,
                cpu=", ".join(sorted(cpus)) or None, platform=", ".join(sorted(platforms)) or None)


def identity_column_rows(ctx, docs):
    """One row per identity column this release recorded: its label, machine, lanes, wheel bytes, times, verdict."""
    rows = []
    for column, dest in IDENTITY_COLUMNS:
        doc = docs.get(dest)
        if not isinstance(doc, dict):
            continue
        reports = list((doc.get("shards") or {}).values()) if column == "cpu" else [doc]
        cols = [r.get("columns", {}).get(column) for r in reports if isinstance(r, dict)]
        cols = [c for c in cols if isinstance(c, dict)]
        if not cols:
            continue
        label = cols[0].get("label") or column
        wheels = sorted({w for r in reports if isinstance(r, dict) for w in (r.get("wheels") or [])})
        facts = column_record_facts(identity_record_dirs(ctx, column), label)
        box = docs.get("gates/%s/box-facts.json" % COLUMN_BOX.get(column, "-")) or {}
        machine = "; ".join(x for x in (", ".join(box.get("gpus") or []), box.get("cpu_model"),
                                        "%s cores" % box["cores"] if box.get("cores") else None) if x) \
            or (facts or {}).get("cpu") or "not recorded"
        owed = sum(int(c.get("owed") or 0) for c in cols)
        reused = [c.get("reused") for c in cols]
        rc_ok = all(c.get("rc") == 0 for c in cols) and all(r.get("ok") is True for r in reports if isinstance(r, dict))
        rows.append(dict(column=column, title=COLUMN_TITLE[column], label=label, machine=machine, owed=owed,
                         reused=None if any(x is None for x in reused) else sum(int(x) for x in reused),
                         shards=len(reports), wheels=wheels, facts=facts, ok=rc_ok))
    return rows


def _seconds(s):
    s = int(round(s))
    return "%dh %02dm %02ds" % (s // 3600, s % 3600 // 60, s % 60) if s >= 3600 else "%dm %02ds" % (s // 60, s % 60)


def summary_markdown(ctx, entries, wheels, refs):
    """SUMMARY.md of this release's evidence set (text). ENTRIES are the collected items, WHEELS the manifest's
    wheel rows, REFS the manifest's references."""
    docs = {}
    for e in entries:
        if e["dest"].endswith(".json"):
            try:
                docs[e["dest"]] = json.loads(e["data"])
            except ValueError:
                pass
    have = {e["dest"] for e in entries}
    link = lambda dest: "[`%s`](%s)" % (dest, dest)
    tag = TAG_FORMAT.format(version=ctx.version)
    out = ["# Mojolearn %s: release evidence summary" % ctx.version, "",
           "Written by the release's record step from the files in this directory; nothing was run to write it. "
           "Every number below links to the file it comes from.", "",
           "| | |", "|---|---|",
           "| version | %s |" % ctx.version, "| tag | `%s` |" % tag, "| frozen source commit | `%s` |" % ctx.commit,
           "| release tooling commit | `%s` |" % ctx.tooling_commit, "| date | %s |" % ctx.date,
           "| published platforms | %s |" % (", ".join(ctx.published) or "none"), ""]
    # --- identity columns
    out += ["## Identity columns", ""]
    rows = identity_column_rows(ctx, docs)
    if rows:
        out += ["Each column installed the exact wheel files below (sha256 checked before install) and recorded the "
                "identity lanes whose code changed since the last good record; the other lanes' rows are reused.", "",
                "| column | label | machine | lanes run / reused | lane seconds | start (UTC) | end (UTC) | run |",
                "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            f = r["facts"] or {}
            out.append("| %s | `%s` | %s | %s / %s | %s | %s | %s | %s |" % (
                r["title"], r["label"], r["machine"], r["owed"], "?" if r["reused"] is None else r["reused"],
                _seconds(f["lane_seconds"]) if f else "no record files here", f.get("start") or "-",
                f.get("end") or "-", "ok" if r["ok"] else "FAILED"))
        out += ["", "Recorded from (wheel file and sha256, as each column installed it):", ""]
        for r in rows:
            out.append("- %s: %s" % (r["title"], "; ".join("`%s`" % w for w in r["wheels"]) or "not recorded"))
        out += ["", "Sources: %s." % ", ".join(sorted({link(d) for _, d in IDENTITY_COLUMNS if d in have}))]
    else:
        out.append("This release recorded no identity column on its wheels (default mode); the identity table it "
                   "ships was checked at the frozen source: %s." % (link("identity/at-freeze.json")
                                                                    if "identity/at-freeze.json" in have else "absent"))
    out.append("")
    # --- admission
    out += ["## Admission", ""]
    table = docs.get("identity/table-summary.json") or {}
    admitted = docs.get("identity/admitted.json")
    reasons = []
    if ctx.single and not isinstance(admitted, dict):
        reasons.append("no admission record")
    if any(not r["ok"] for r in rows):
        reasons.append("failed column run(s): " + ", ".join(r["title"] for r in rows if not r["ok"]))
    if table.get("parts_not_on_every_column"):
        reasons.append("%s part(s) not on every column" % table["parts_not_on_every_column"])
    if not table:
        reasons.append("no table summary")
    out.append("All columns agree: **%s**%s" % ("yes" if not reasons else "no", "" if not reasons else
                                                 " (" + "; ".join(reasons) + ")"))
    out.append("")
    if isinstance(admitted, dict):
        out.append("- admitted at commit `%s`, table sha256 `%s` (%s)" % (
            admitted.get("commit") or "?", admitted.get("table_sha256") or "?", link("identity/admitted.json")))
    if table:
        t = table.get("table") or {}
        out.append("- shipped table `%s`, sha256 `%s`: %s parts, %s equal on every column, %s PTX-divergent (%s)" % (
            t.get("ref") or "?", t.get("sha256") or "?", table.get("parts"), table.get("parts_equal_on_every_column"),
            table.get("ptx_divergent", 0), link("identity/table-summary.json")))
        cols = table.get("columns") or {}
        if cols:
            out.append("- table columns: " + "; ".join("`%s` %s lanes, %s parts" % (c, v.get("lanes"), v.get("parts"))
                                                       for c, v in sorted(cols.items())))
    out.append("")
    # --- wheels
    out += ["## Wheels", "", "| platform | file | sha256 | published |", "|---|---|---|---|"]
    for w in wheels:
        out.append("| %s | `%s` | `%s` | %s |" % (w["platform"], w["file"], w["sha256"], "yes" if w["published"] else "no"))
    out += ["", "Per-platform manifests and file admissions: %s." % ", ".join(
        link(d) for d in sorted(have) if d.startswith(("alpha-manifest-", "file-admission-"))), ""]
    # --- device checks
    out += ["## Device checks", ""]
    for vendor in sorted(GPU_VENDOR_OF.values()):
        shard = "gates/%s/shardcheck.json" % vendor
        if shard in docs:
            d = docs[shard]
            lanes = d.get("lanes") or {}
            match = sum(1 for x in lanes.values() if (x or {}).get("verdict") == "MATCH")
            out.append("- %s multi-GPU check: **%s**, %d of %d lanes MATCH across device counts %s (%s)" % (
                vendor, "PASS" if d.get("ok") is True and match == len(lanes) and lanes else "FAIL", match,
                len(lanes), d.get("counts"), link(shard)))
        measure = "gates/%s/measuresmoke.json" % vendor
        if measure in docs:
            d = docs[measure]
            c = d.get("counts") or {}
            out.append("- %s measurement smoke: **%s**, %s passed, %s unsupported, %s failed of %s recipes (%s)" % (
                vendor, "PASS" if d.get("ok") is True else "FAIL", c.get("PASS", 0), c.get("UNSUPPORTED", 0),
                c.get("FAIL", 0), d.get("plan_count"), link(measure)))
        smoke = "gates/%s/gate-smoke.json" % vendor
        if smoke in docs:
            out.append("- %s gate smoke: **%s** (%s)" % (vendor, docs[smoke].get("verdict") or "?", link(smoke)))
    other = sorted(d for d in have if d.startswith(("gates/", "self-test-", "light-smoke-", "publish/", "checks/",
                                                    "compile/"))
                   and not d.endswith(("/shardcheck.json", "/measuresmoke.json", "/gate-smoke.json")))
    if other:
        out.append("- every other check's verdict: " + ", ".join(link(d) for d in other))
    out.append("")
    # --- reproduce
    out += ["## Check it on your own machine", "",
            "    pip install \"mojolearn[verify]==%s\"" % ctx.version,
            "    python -m mojolearn verify", "",
            "The wheel ships the admitted reference table above; `verify` runs every applicable routine lane on "
            "your machine in `identical` mode and checks its output hashes against that table "
            "(`--quick` for representative checks in seconds to minutes; see the user guide, Verify on your "
            "hardware).", ""]
    # --- where the full logs are
    head = ["## Full logs and records", "",
            "Kept outside git; `manifest.json` lists each by path or R2 key with its sha256.", ""]
    lines = []
    for r in refs:
        if r.get("r2_tag"):
            lines.append("- %s: R2 `%s/<date>/%s/` (`%s`)" % (r["what"], r.get("r2_key_root"), r["r2_tag"], r["fetch"]))
        elif r.get("path"):
            lines.append("- %s: `%s`" % (r["what"], r["path"]))
    budget = SUMMARY_MAX_LINES - len(out) - len(head) - 1
    if len(lines) > budget:
        keep = max(budget - 1, 0)
        lines = lines[:keep] + ["- %d more in `manifest.json` (references)" % (len(lines) - keep)]
    out += head + lines
    return "\n".join(out[:SUMMARY_MAX_LINES]) + "\n"


def write_set(ctx, root=ROOT):
    """Delete the previous set and write this release's. Returns the manifest. Raises EvidenceError naming every
    missing required item, every cap or secret problem, before anything is deleted."""
    entries, missing, absent, problems = collect(ctx, root)
    errors = ["missing required evidence %s (step %s; looked for %s)" % (m["path"], m["producer"], m["looked_for"])
              for m in missing] + problems
    total = sum(len(e["data"]) for e in entries)
    if total > TOTAL_CAP:
        errors.append("the set is %d bytes, over the %d-byte cap" % (total, TOTAL_CAP))
    if errors:
        raise EvidenceError("release evidence for %s not written: %s" % (ctx.version, "; ".join(errors)))
    wheels = []
    for platform, wheel in sorted(ctx.wheels.items()):
        if wheel and Path(wheel).is_file():
            wheels.append(dict(platform=platform, file=Path(wheel).name, sha256=sha256_file(wheel),
                               published=platform in ctx.published))
    by_platform = {w["platform"]: w for w in wheels}
    readme = README.encode()
    refs = references(ctx)
    summary = scrub_text(summary_markdown(ctx, entries, wheels, refs))[0].encode()
    items = [dict(path="README.md", sha256=sha256_bytes(readme), bytes=len(readme), producer="record",
                  required=True, source="tools/release_evidence_set.py", against=[]),
             dict(path="SUMMARY.md", sha256=sha256_bytes(summary), bytes=len(summary), producer="record",
                  required=True, source="tools/release_evidence_set.py", against=[])]
    for e in entries:
        items.append(dict(path=e["dest"], sha256=sha256_bytes(e["data"]), bytes=len(e["data"]), producer=e["producer"],
                          required=e["required"], source=e["source"], source_sha256=e["source_sha256"],
                          redactions=e["redactions"],
                          against=[dict(platform=p, file=by_platform[p]["file"], sha256=by_platform[p]["sha256"])
                                   for p in e["against"] if p in by_platform]))
    tools = dict(python=_platform.python_version(), system=_platform.system(), machine=_platform.machine(),
                 release_evidence_py_sha256=sha256_file(__file__), **(ctx.tools or {}))
    manifest = dict(schema=SCHEMA, release=dict(version=ctx.version, source_commit=ctx.commit,
                                                tooling_commit=ctx.tooling_commit, date=ctx.date,
                                                written_at=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                                                tag=TAG_FORMAT.format(version=ctx.version)),
                    wheels=wheels, tools=tools, caps=dict(file_bytes=FILE_CAP, total_bytes=TOTAL_CAP),
                    items=items, absent_optional=absent, references=refs)
    mdata = (json.dumps(manifest, indent=1, sort_keys=True) + "\n").encode()
    mtext, found = scrub_text(mdata.decode())
    if found:
        mdata = mtext.encode()
    if len(mdata) > FILE_CAP or total + len(mdata) + len(readme) + len(summary) > TOTAL_CAP:
        raise EvidenceError("release evidence for %s not written: the manifest takes the set over its caps" % ctx.version)
    fixed = Path(root) / EVIDENCE_DIR
    tmp = fixed.parent / (".release_evidence.new-%d" % os.getpid())
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    try:
        (tmp / "README.md").write_bytes(readme)
        (tmp / "SUMMARY.md").write_bytes(summary)
        for e in entries:
            out = tmp / e["dest"]
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(e["data"])
        (tmp / "manifest.json").write_bytes(mdata)
        leaks = audit(tmp)
        if leaks:
            raise EvidenceError("release evidence for %s not written: %s" % (ctx.version, "; ".join(leaks)))
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    # THE REPLACEMENT: the previous set goes as a whole, then this one takes its place
    if fixed.exists():
        shutil.rmtree(fixed)
    tmp.rename(fixed)
    for stray in fixed.parent.glob(".release_evidence.new-*"):
        shutil.rmtree(stray, ignore_errors=True)
    return manifest


def audit(directory):
    """[problems] of a written set: a secret anywhere outside a redaction mark, a file over the cap, the total over
    the cap, a file the manifest does not list or whose sha256 differs."""
    d = Path(directory)
    problems, total = [], 0
    manifest = read_json(d / "manifest.json")
    listed = {i["path"]: i["sha256"] for i in (manifest or {}).get("items", [])} if isinstance(manifest, dict) else None
    if listed is None:
        problems.append("no manifest.json")
    for f in sorted(p for p in d.rglob("*") if p.is_file()):
        name = f.relative_to(d).as_posix()
        data = f.read_bytes()
        total += len(data)
        if len(data) > FILE_CAP:
            problems.append("%s is over the per-file cap" % name)
        found = scan_bytes(name, data)
        if found:
            problems.append("%s holds %s" % (name, ", ".join(sorted(set(found)))))
        if listed is not None and name != "manifest.json":
            if name not in listed:
                problems.append("%s is not in the manifest" % name)
            elif listed[name] != sha256_bytes(data):
                problems.append("%s does not match its manifest sha256" % name)
    if listed is not None:
        for name in sorted(set(listed) - {f.relative_to(d).as_posix() for f in d.rglob("*") if f.is_file()}):
            problems.append("%s is in the manifest but not in the set" % name)
    if total > TOTAL_CAP:
        problems.append("the set is over the total cap")
    return problems


# ------------------------------------------------------------------------------------------- readers
def current_dir(root=ROOT):
    """The current set's directory in the working tree, or None when it holds no release record yet."""
    fixed = Path(root) / EVIDENCE_DIR
    return fixed if (fixed / "release-provenance.json").is_file() or (fixed / "manifest.json").is_file() else None


def provenance_history(root, revision, limit=200):
    """[(version, source_commit)] of every release-provenance.json reachable at REVISION, read THROUGH GIT (never
    the working tree): the current set at REVISION and its earlier versions (each release overwrote it), then the
    dated records releases up to 0.8.65 committed under LEGACY_DIR (removed from the tree 2026-11-06; read from
    the last commit that held them)."""
    def git(*args):
        p = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
        return p.stdout if p.returncode == 0 else ""

    path = EVIDENCE_DIR + "/release-provenance.json"
    seen, out = set(), []

    def take(text):
        try:
            doc = json.loads(text)
        except ValueError:
            return
        if (isinstance(doc, dict) and doc.get("schema") == "mojolearn.release-provenance.v1"
                and isinstance(doc.get("version"), str) and isinstance(doc.get("source_commit"), str)):
            key = (doc["version"], doc["source_commit"])
            if key not in seen:
                seen.add(key)
                out.append(key)

    for commit in git("log", "--format=%H", "-n", str(limit), revision, "--", path).split():
        take(git("show", "%s:%s" % (commit, path)))
    # the dated records at REVISION (the ones a document still points at stay in the tree) and, for those removed,
    # at the parent of each commit that deleted files there
    points = [revision] + [c + "^" for c in git("log", "--diff-filter=D", "--format=%H", revision, "--",
                                                 LEGACY_DIR + "/").split()]
    for at in points:
        for p in git("ls-tree", "-r", "--name-only", at, LEGACY_DIR + "/").splitlines():
            if p.endswith("/release-provenance.json"):
                take(git("show", "%s:%s" % (at, p)))
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("audit", help="check a written set: manifest hashes, caps, no secrets")
    a.add_argument("dir", nargs="?", default=str(ROOT / EVIDENCE_DIR))
    sub.add_parser("items", help="print the item table (required vs optional)")
    args = ap.parse_args(argv)
    if args.cmd == "audit":
        problems = audit(args.dir)
        for p in problems:
            print("PROBLEM " + p)
        print("release-evidence audit: %s" % ("OK" if not problems else "%d problem(s)" % len(problems)))
        return 1 if problems else 0
    for dest, producer, when, source in ITEMS:
        print("%-42s %-20s %-9s %s" % (dest, producer, when, source))
    return 0


if __name__ == "__main__":
    sys.exit(main())
