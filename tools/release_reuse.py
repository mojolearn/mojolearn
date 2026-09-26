#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Which bindings a release builds and which it takes, byte for byte, from the
last PUBLISHED release (2026-09-23).

    python3 tools/release_reuse.py plan [--commit HEAD] [--json <out>] [--full]

WHY. 0.8.16 was a Python-only patch: not one .mojo file, build script,
pixi.lock line or box image had moved since 0.8.15, and the release still
rented an H100, an L40S and an MI325X to rebuild all 183 Linux bindings, then
built the 61 macOS ones again on the Mac: about 40 minutes and most of the
cost of a release, for bytes that could only come out the same (NVIDIA, host)
or, on gfx942, DIFFERENT for no reason (the Mojo compiler's AMD output is not
reproducible run to run; taking the released bytes is what freezes them).

THE IDENTITY OF A BINDING is everything that can reach its bytes:
  * its source closure, `bincache.source_digest` over the Mojo import closure
    of the file its build script compiles, the shell scripts the script
    execs, pixi.toml (dependencies, not tasks) and pixi.lock, computed from
    `git archive` of the commit and never from a working tree (a generated
    file lying around would change the walk);
  * the generator inputs when the closure reaches tokenizer/ (tokenizer/tools
    writes the unicode table that is compiled in);
  * the Mojo/MAX packages pixi.lock pins for the TARGET platform (linux-64
    for the Linux sets, osx-arm64 for the macOS wheel), never the machine
    computing the table;
  * the build flags the release scripts export for that binding (numeric
    mode, accelerator target, kernel column, CPU baseline, compiler workers);
  * the builder scripts that drive the compile and stage the result
    (packaging/linux/build_sets.sh, stage_libs.py, tools/release061_remote_build.sh,
    tools/linux_surface_qualification.sh; on macOS build_release_wheel.sh,
    stage_dylibs.py and python/setup.py), each read through THIS binding's
    view (BUILDER_RULE below: the binding lists reduced to this binding's
    own membership, comments out where unambiguous, every other byte kept),
    so another binding joining a list does not rebuild this one;
  * the pinned box image (the RunPod NVIDIA image, the ROCm container digest)
    or, on macOS, the Xcode and Metal toolchain that built it, which the
    release records; a previous release without that record is not reused.
Equal identity digests mean REUSE: the published bytes, verified by sha256
against the previous release's record and its wheel's own payload, are packed
again. Anything else, a new binding, a changed closure, flag, image, toolchain
or an unreadable pin, means BUILD. "Probably unchanged" is not a decision this
file can make.

WHAT A PLAN DECIDES. Per binding REUSE or BUILD with the reason; per Linux set
whether a build leg runs (a set with one BUILD binding runs its leg and the
leg builds the whole set, as it always did; the pack then takes the published
bytes for the set's REUSE bindings and the leg's for the rest); when only a
host binding or the runtime closure needs a build and no set does, the sm_89
leg (the cheapest box) builds it. A Python-only release therefore rents
nothing for builds; its NVIDIA and AMD release columns still run from the
packed wheel as before.

The previous release is the newest bench/results/release_verification/
<date>_pypi_<v>/ record (its alpha manifests name the published wheels and
their sha256, its light-smoke receipts the source commit); its identities
come from binding-identities.json in that record when the release wrote one,
otherwise from `git archive` of its commit.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "packaging" / "linux"))
import bincache  # noqa: E402
import stage_libs  # noqa: E402
from pack_wheel import HOST_NAMES, TIERS, tier_names  # noqa: E402

SCHEMA = "mojolearn.binding-identity.v2"
#: The identity every release up to 0.8.18 recorded: builder files by
#: whole-file sha256. A record in it is upgraded only by reproducing it
#: exactly from the release's own commit (previous_identities).
SCHEMA_V1 = "mojolearn.binding-identity.v1"
#: The tree-facts cache (per commit); v2 added the builder views.
FACTS_SCHEMA = "mojolearn.release-tree-facts.v2"
#: The builder view of the runtime closure: every list kept whole.
RUNTIME_VIEW = ""
PLAN_SCHEMA = "mojolearn.release-reuse-plan.v1"
REUSE_SCHEMA = "mojolearn.linux.reused-bindings.v1"
LINUX = "linux-64"
MACOS = "osx-arm64"
LINUX_SETS = (("cuda", "sm_90a"), ("cuda", "sm_89"), ("hip", "gfx942"))
#: The leg that builds when only a host binding or the runtime needs one.
HOST_LEG = ("cuda", "sm_89")
LINUX_BUILDERS = ("packaging/linux/build_sets.sh", "packaging/linux/stage_libs.py",
                  "tools/release061_remote_build.sh", "tools/linux_surface_qualification.sh",
                  "packaging/linux/binding_timeout.sh", "packaging/linux/ptx_contract.py",
                  "packaging/linux/cubin_contract.py")
MACOS_BUILDERS = ("packaging/macos/build_release_wheel.sh", "packaging/macos/stage_dylibs.py",
                  "python/setup.py")
PORTABLE_MATH = "packaging/portable_math"
GENERATORS = "tokenizer/tools"
NVIDIA_IMAGE_RE = re.compile(r'LEG_IMAGE_NVIDIA="\$\{MOJOLEARN_GEMM_LEG_IMAGE_NVIDIA:-([^}]+)\}"')
AMD_IMAGE_RE = re.compile(r"^IMAGE=(\S+)", re.M)
RECORD_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})_pypi_(\d+)$")
SHA_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def digest_of(identity):
    return hashlib.sha256(canonical(identity).encode()).hexdigest()


def git(*args, root=ROOT):
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True,
                          check=True).stdout


# ---------------------------------------------------------------- builder views
# A builder file is shared by every binding, so its whole-file sha256 made any
# edit to it (one more name on EXT_NAMES, a reworded comment) a new identity
# for all 240 bindings. A binding's identity now reads the builder through a
# VIEW OF THAT BINDING: the file's bytes with exactly two things taken out,
# each only where the file format makes it unambiguous, and nothing else
# (every flag, compile line, function, heredoc and embedded script stays
# byte for byte):
#
#   1. binding lists. A shell line at column 0 of the exact form
#        VAR="tok tok ..."   or   VAR="${ENV:-tok tok ...}"
#      where VAR is SCRIPTS or ends in _NAMES / _SCRIPTS and EVERY token is a
#      binding name (_mojolearn, _mojolearn_x) or a build script (build.sh,
#      build_x.sh), read at the top level of the script (not inside a quote,
#      heredoc or continuation), is replaced by the same line keeping only
#      the tokens that name THIS binding (its name or its script). So the
#      view records whether the binding is in each list, which list (the
#      tier lists are separate variables) and every list's variable and
#      environment override; another binding joining or leaving a list does
#      not reach it. The runtime closure's view keeps every list whole (a new
#      binding can need another MAX library in .libs).
#   2. comments. Python (stage_libs.py, stage_dylibs.py, setup.py): every
#      COMMENT token the tokenizer reports, except a line-1 shebang and a
#      coding cookie; comment-only lines are dropped, a trailing comment is
#      cut with the whitespace before it. Shell: only FULL-LINE comments and
#      blank lines at the top level, with the leading indentation of top
#      level lines; the scan stops taking anything out (the rest of the file
#      is hashed verbatim) at the first construct it does not model: a
#      backtick, $'...', a heredoc marker it cannot parse, a ${...} inside
#      double quotes that does not close on its own line or holds a quote or
#      another expansion, a case statement inside $( ), a heredoc whose
#      marker line ends inside a quote or a continuation. Lines inside a
#      $( ) that spans lines, heredoc bodies, multi-line strings and
#      continuation lines are never touched. The shebang stays.
#
# A file that is not UTF-8 or that Python cannot tokenize is hashed whole.
# The rule has a name (BUILDER_RULE) that is part of every identity, so a
# change to it compares as changed, never silently equal.
BUILDER_RULE = "builder-view.v1: lists by this binding's membership; full-line shell comments; python comments"
_LIST_LINE = re.compile(
    r'^(?P<var>(?:[A-Z][A-Z0-9_]*_)?(?:NAMES|SCRIPTS))='
    r'"(?:\$\{(?P<env>[A-Z][A-Z0-9_]*):-(?P<dv>[^"$`\\{}]*)\}|(?P<v>[^"$`\\{}]*))"$')
_LIST_TOKEN = re.compile(r"^(?:_mojolearn(?:_[a-z0-9_]+)?|build(?:_[a-z0-9_]+)?\.sh)$")
_HEREDOC = re.compile(r"<<(-?)[ \t]*(?:'([A-Za-z_][A-Za-z0-9_]*)'|\"([A-Za-z_][A-Za-z0-9_]*)\"|([A-Za-z_][A-Za-z0-9_]*))")
_SEPARATORS = " \t;&|()<>"


class _Unmodeled(Exception):
    """A shell construct the view does not model: stop taking anything out."""


def _close_param(line, i):
    """Index just past the `}` closing the `${` at line[i] inside double
    quotes, on this line and holding no quote, backtick, backslash or nested
    expansion; else _Unmodeled."""
    j = i + 2
    while j < len(line):
        c = line[j]
        if c in "'\"`\\${":
            raise _Unmodeled
        if c == "}":
            return j + 1
        j += 1
    raise _Unmodeled


def _scan_shell_line(line, st):
    """Advance the lexer state `st` over one line that is not a heredoc body.
    st["stack"] is the nesting, innermost last: "TOP" (the script), "CMD"
    (inside $( ), with its open-paren depth in st["depth"]), "SQ", "DQ";
    st["continued"] is a trailing backslash at a command level; heredoc
    markers seen are queued in st["heredocs"]."""
    stack, depth = st["stack"], st["depth"]
    i, n, word_start = 0, len(line), True
    st["continued"] = False
    while i < n:
        c = line[i]
        mode = stack[-1]
        if mode == "SQ":
            if c == "'":
                stack.pop()
            i += 1
            continue
        if mode == "DQ":
            if c == "\\":
                if i == n - 1:
                    return
                i += 2
            elif c == '"':
                stack.pop()
                i += 1
            elif c == "`":
                raise _Unmodeled
            elif line.startswith("$(", i):
                stack.append("CMD")
                depth.append(0)
                i += 2
                word_start = True
            elif line.startswith("${", i):
                i = _close_param(line, i)
            else:
                i += 1
            continue
        # a command level: the script itself or a $( ) substitution
        if c == "\\":
            if i == n - 1:
                st["continued"] = True
                return
            i += 2
            word_start = False
        elif c == "'":
            stack.append("SQ")
            i += 1
            word_start = False
        elif c == '"':
            stack.append("DQ")
            i += 1
            word_start = False
        elif c == "`" or line.startswith("$'", i):
            raise _Unmodeled
        elif c == "#" and word_start:
            return
        elif mode == "CMD" and word_start and re.match(r"(case|esac)\b", line[i:]):
            raise _Unmodeled
        elif line.startswith("$(", i):
            stack.append("CMD")
            depth.append(0)
            i += 2
            word_start = True
        elif mode == "CMD" and c == "(":
            depth[-1] += 1
            i += 1
            word_start = True
        elif mode == "CMD" and c == ")":
            if depth[-1] == 0:
                stack.pop()
                depth.pop()
                word_start = False
            else:
                depth[-1] -= 1
                word_start = True
            i += 1
        elif line.startswith("<<<", i):
            i += 3
            word_start = True
        elif line.startswith("<<", i):
            m = _HEREDOC.match(line, i)
            if not m:
                raise _Unmodeled
            st["heredocs"].append((m.group(2) or m.group(3) or m.group(4), m.group(1) == "-"))
            i = m.end()
            word_start = False
        else:
            word_start = c in _SEPARATORS
            i += 1


def _list_line(line, members):
    """The view of a binding-list assignment, or None when `line` is not one."""
    m = _LIST_LINE.match(line)
    if not m:
        return None
    value = m.group("dv") if m.group("env") else m.group("v")
    tokens = value.split()
    if not tokens or not all(_LIST_TOKEN.match(t) for t in tokens):
        return None
    kept = " ".join(t for t in tokens if t in members)
    if m.group("env"):
        return '%s="${%s:-%s}"' % (m.group("var"), m.group("env"), kept)
    return '%s="%s"' % (m.group("var"), kept)


def shell_view(text, members):
    """The shell builder as binding `members` ({name, script}; None keeps every
    list whole) sees it. See the rule above."""
    out = []
    st = dict(stack=["TOP"], depth=[], continued=False, heredocs=[])
    body = None  # (word, dash) of the heredoc whose body is being read
    lines = text.split("\n")
    for k, line in enumerate(lines):
        if body is not None:
            out.append(line)
            if (line.lstrip("\t") if body[1] else line) == body[0]:
                body = st["heredocs"].pop(0) if st["heredocs"] else None
            continue
        top = st["stack"] == ["TOP"] and not st["continued"] and k > 0
        stripped = line.lstrip(" \t")
        if top and stripped == "":
            continue
        if top and stripped.startswith("#"):
            continue
        view = _list_line(line, members) if top and members is not None else None
        out.append(view if view is not None else (stripped if top else line))
        try:
            _scan_shell_line(line, st)
        except _Unmodeled:
            out.extend(lines[k + 1:])
            return "\n".join(out)
        if st["heredocs"]:
            if st["stack"][-1] in ("SQ", "DQ") or st["continued"]:
                out.extend(lines[k + 1:])
                return "\n".join(out)
            body = st["heredocs"].pop(0)
    return "\n".join(out)


def python_view(text):
    """The Python builder without its comments (tokenizer-reported only), or
    None when it does not tokenize."""
    import io
    import tokenize
    cuts = {}
    try:
        for tok in tokenize.generate_tokens(io.StringIO(text).readline):
            if tok.type != tokenize.COMMENT:
                continue
            row, col = tok.start
            if row == 1 and col == 0 and tok.string.startswith("#!"):
                continue
            if row <= 2 and re.match(r"^[ \t\f]*#.*?coding[:=]", tok.line):
                continue
            cuts[row] = col
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return None
    out = []
    for row, line in enumerate(text.split("\n"), 1):
        if row in cuts:
            code = line[:cuts[row]].rstrip(" \t")
            if code.strip(" \t\f") == "":
                continue
            line = code
        out.append(line)
    return "\n".join(out)


def builder_view(rel, data, members):
    """sha256 of builder `rel`'s bytes as seen by a binding with identifiers
    `members` (None: the runtime, every list kept)."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return hashlib.sha256(data).hexdigest()
    if rel.endswith(".py"):
        view = python_view(text)
        if view is None:
            return hashlib.sha256(data).hexdigest()
    else:
        view = shell_view(text, members)
    return hashlib.sha256(("%s\0%s\0" % (BUILDER_RULE, rel) + view).encode()).hexdigest()


def binding_members(name):
    """The list tokens that name a binding: its name and its build script."""
    return frozenset((name, "build.sh" if name == "_mojolearn" else "build_" + name[len("_mojolearn_"):] + ".sh"))


# ---------------------------------------------------------------- the bindings
class Binding:
    """One shipped binary of one target: where it lands in the wheel, in a
    build set and in the package tree, and which script builds it."""

    def __init__(self, target, vendor, arch, tier, name):
        self.target, self.vendor, self.arch, self.tier, self.name = target, vendor, arch, tier, name
        self.script = "build.sh" if name == "_mojolearn" else "build_" + name[len("_mojolearn_"):] + ".sh"
        self.key = "/".join((target, vendor, arch or "-", tier, name))
        self.host = tier == "host"

    @property
    def set_rel(self):
        if self.host:
            return f"host/{self.name}.so"
        return f"{self.name}.so" if self.tier == "fast" else f"{self.tier}/{self.name}.so"

    @property
    def archive_path(self):
        if self.host:
            return f"mojolearn/host/{self.name}.so"
        if self.target == LINUX:
            return f"mojolearn/{self.vendor}/{self.arch}/{self.set_rel}"
        return f"mojolearn/{self.set_rel}"

    @property
    def package_rel(self):
        """Where the macOS build script writes it (repo-relative)."""
        return "python/" + self.archive_path

    @property
    def set_key(self):
        return f"{self.vendor}/{self.arch}" if self.target == LINUX and not self.host else self.tier

    def flags(self):
        """Exactly what tools/release061_remote_build.sh and
        packaging/linux/build_sets.sh (Linux) or build_release_wheel.sh
        (macOS) export around this binding's compile."""
        if self.target == LINUX:
            if self.host:
                return dict(numeric_mode="identical", target_column="cpu", gpu_archs=None,
                            linux_cpu="x86-64-v3", compile_jobs="2", package_byte_lm="1")
            return dict(numeric_mode=self.tier, target_column="nvidia" if self.vendor == "cuda" else "amd",
                        gpu_archs=self.arch, linux_cpu="x86-64-v3",
                        compile_jobs="1" if self.arch.startswith("gfx") else "2", package_byte_lm="1")
        if self.host:
            return dict(numeric_mode="identical", target_column="cpu", gpu_archs=None, compile_jobs="1",
                        package_byte_lm="1")
        return dict(numeric_mode=self.tier, gpu_archs=None, compile_jobs="1", package_byte_lm="1")

    def row(self):
        return dict(key=self.key, target=self.target, vendor=self.vendor, arch=self.arch, tier=self.tier,
                    name=self.name, script=self.script, archive_path=self.archive_path, set_rel=self.set_rel)


def bindings(target):
    rows = []
    if target == LINUX:
        for vendor, arch in LINUX_SETS:
            for tier in TIERS:
                for name in tier_names(tier, True):
                    rows.append(Binding(target, vendor, arch, tier, name))
    else:
        for tier in TIERS:
            for name in tier_names(tier, True):
                rows.append(Binding(target, "metal", "apple", tier, name))
    for name in HOST_NAMES:
        rows.append(Binding(target, "host", "", "host", name))
    return rows


# ---------------------------------------------------------------- tree facts
def wanted(name):
    """The members of a commit's archive that any identity can read."""
    return (name.endswith((".mojo", ".mojopkg")) or name.startswith(("bindings/", "tokenizer/", "packaging/"))
            or (name.startswith("tools/") and name.endswith(".sh"))
            or name in ("pixi.toml", "pixi.lock", "python/setup.py"))


def extract_tree(commit, dest, root=ROOT):
    """`git archive <commit>` filtered to `wanted` into dest; never a checkout,
    never the working tree."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["git", "-C", str(root), "archive", "--format=tar", commit],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    n = 0
    with tarfile.open(fileobj=proc.stdout, mode="r|") as tar:
        for member in tar:
            if member.isfile() and wanted(member.name) and ".." not in member.name.split("/"):
                tar.extract(member, dest, filter="data")
                n += 1
    err = proc.stderr.read().decode(errors="replace")
    proc.stdout.close()
    proc.stderr.close()
    if proc.wait() != 0:
        raise SystemExit(f"release_reuse: git archive {commit[:12]} failed: {err.strip()}")
    return n


def toolchain(tree, plat):
    """The Mojo/MAX packages pixi.lock pins for `plat` (bincache.toolchain's
    regex, with the platform chosen by the target, not by this machine)."""
    lock = Path(tree) / "pixi.lock"
    if not lock.is_file():
        return None
    text = lock.read_text(errors="replace")
    pkgs = sorted(set(re.findall(
        r"conda\.modular\.com/max/%s/((?:mojo|mojo-compiler|max|max-core)-[^/\s]+)\.conda" % re.escape(plat), text)))
    return dict(platform=plat, packages=pkgs, pixi_lock_sha256=sha256(lock))


def image_pins(tree):
    tree = Path(tree)
    pins = {}
    try:
        m = NVIDIA_IMAGE_RE.search((tree / "tools" / "gemm_remote_leg.sh").read_text(errors="replace"))
        pins["cuda"] = m.group(1) if m else None
    except OSError:
        pins["cuda"] = None
    try:
        m = AMD_IMAGE_RE.search((tree / "tools" / "release_ubuntu22_build.sh").read_text(errors="replace"))
        pins["hip"] = m.group(1) if m else None
    except OSError:
        pins["hip"] = None
    return pins


def dir_digest(tree, rel):
    base = Path(tree) / rel
    if not base.is_dir():
        return None
    h = hashlib.sha256()
    for p in sorted(base.rglob("*")):
        if p.is_file():
            h.update(p.relative_to(tree).as_posix().encode() + b"\0" + sha256(p).encode() + b"\n")
    return h.hexdigest()


def tree_facts(tree):
    """Everything the identities read from one commit's tree."""
    tree = Path(tree)
    closures = {}
    scripts = sorted({b.script for t in (LINUX, MACOS) for b in bindings(t)})
    for script in scripts:
        path = tree / "bindings" / script
        if not path.is_file():
            closures[script] = None
            continue
        info, rels = bincache.source_digest(tree, str(path), [])
        closures[script] = dict(info, sources=sorted(rels))
    files, views = {}, {}
    names = sorted({b.name for t in (LINUX, MACOS) for b in bindings(t)})
    for rel in LINUX_BUILDERS + MACOS_BUILDERS:
        p = tree / rel
        files[rel] = sha256(p) if p.is_file() else None
        if not p.is_file():
            views[rel] = None
            continue
        data = p.read_bytes()
        views[rel] = {name: builder_view(rel, data, binding_members(name)) for name in names}
        views[rel][RUNTIME_VIEW] = builder_view(rel, data, None)
    return dict(closures=closures, builders=files, builder_views=views, image=image_pins(tree),
                toolchain={LINUX: toolchain(tree, LINUX), MACOS: toolchain(tree, MACOS)},
                generators=dir_digest(tree, GENERATORS), portable_math=dir_digest(tree, PORTABLE_MATH))


def facts_for_commit(commit, cache_dir, root=ROOT):
    """tree_facts of a commit, cached by its full sha (the tree is a pure
    function of it)."""
    commit = git("rev-parse", commit + "^{commit}", root=root).strip()
    cache = Path(cache_dir) / f"{commit}.json"
    if cache.is_file():
        try:
            d = json.loads(cache.read_text())
            if d.get("commit") == commit and d.get("schema") == FACTS_SCHEMA:
                return d["facts"]
        except (OSError, ValueError):
            pass
    import tempfile
    with tempfile.TemporaryDirectory(prefix="mojolearn-tree-") as d:
        extract_tree(commit, d, root)
        facts = tree_facts(d)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(schema=FACTS_SCHEMA, commit=commit, facts=facts), sort_keys=True) + "\n")
    tmp.replace(cache)
    return facts


def darwin_toolchain():
    """The Mac's own compilers, which the Metal half of a binding is built
    with (bincache.darwin_toolchain); None off macOS."""
    if sys.platform != "darwin":
        return None
    return bincache.darwin_toolchain()


# ---------------------------------------------------------------- identities
def builder_digests(facts, rels, view):
    """{builder: digest} for one view (a binding name, or RUNTIME_VIEW);
    None for a builder the tree does not have or a view it has no entry for
    (unreadable() then builds)."""
    views = facts.get("builder_views") or {}
    return {rel: (views.get(rel) or {}).get(view) for rel in rels}


def identity(b, facts, host_toolchain=None, schema=SCHEMA):
    """The identity dict of binding `b` under `facts`, or None when that tree
    has no build script for it (a new binding). `schema=SCHEMA_V1` gives the
    identity 0.8.18 and earlier recorded (whole-file builder digests), used
    only to prove a recorded one reproduces before upgrading it."""
    closure = facts["closures"].get(b.script)
    if closure is None:
        return None
    builders = LINUX_BUILDERS if b.target == LINUX else MACOS_BUILDERS
    if schema == SCHEMA_V1:
        builder_ids = {rel: facts["builders"].get(rel) for rel in builders}
    else:
        builder_ids = builder_digests(facts, builders, b.name)
    ident = dict(schema=schema, target=b.target, vendor=b.vendor, arch=b.arch, tier=b.tier, name=b.name,
                 script=b.script, closure=dict(scope=closure["scope"], digest=closure["digest"]),
                 generators=facts["generators"] if any(s.startswith("tokenizer/") for s in closure["sources"]) else None,
                 toolchain=facts["toolchain"][b.target], flags=b.flags(),
                 builders=builder_ids)
    if schema != SCHEMA_V1:
        ident["builder_rule"] = BUILDER_RULE
    if b.target == LINUX:
        ident["image"] = facts["image"].get("cuda" if b.host or b.vendor == "cuda" else "hip")
        if b.host:
            # every leg builds the host bindings, and the wheel ships one copy;
            # the pack refuses copies that differ, so both images are inputs
            ident["image"] = dict(facts["image"])
    else:
        ident["host_toolchain"] = host_toolchain
    return ident


def runtime_identity(facts, schema=SCHEMA):
    """The Linux runtime closure (.libs: the MAX runtime the toolchain ships
    plus the portable math helper cc builds on the box). Its builder view
    keeps every binding list whole: a binding joining a list can need one
    more MAX library in .libs."""
    ident = dict(schema=schema, target=LINUX, what="runtime", toolchain=facts["toolchain"][LINUX],
                 image=dict(facts["image"]), portable_math=facts["portable_math"])
    if schema == SCHEMA_V1:
        ident["builders"] = {rel: facts["builders"].get(rel) for rel in LINUX_BUILDERS}
    else:
        ident["builders"] = builder_digests(facts, LINUX_BUILDERS, RUNTIME_VIEW)
        ident["builder_rule"] = BUILDER_RULE
    return ident


def unreadable(ident):
    """A pin this identity could not read. Any doubt builds."""
    if ident is None:
        return "no build script"
    if ident.get("upgrade_refused"):
        return ident["upgrade_refused"]
    if ident.get("toolchain") is None or not ident["toolchain"].get("packages"):
        return "no Mojo/MAX packages for %s in pixi.lock" % ident.get("target")
    if any(v is None for v in ident["builders"].values()):
        return "builder script missing: " + ", ".join(k for k, v in ident["builders"].items() if v is None)
    img = ident.get("image")
    if ident.get("target") == LINUX and (img is None or (isinstance(img, dict) and any(v is None for v in img.values()))):
        return "box image pin unreadable"
    if ident.get("target") == MACOS and ident.get("host_toolchain") is None:
        return "no record of the Apple toolchain"
    return None


def compare(cur, prev):
    """(decision, reason) for a binding with current and previous identities."""
    if prev is None:
        return "BUILD", "new binding (no build script in the previous release)"
    why = unreadable(cur)
    if why:
        return "BUILD", "unreadable identity: " + why
    why = unreadable(prev)
    if why:
        return "BUILD", "previous release: " + why
    if prev.get("schema") != cur.get("schema"):
        return "BUILD", "previous release: identity recorded as %s, not %s" % (prev.get("schema"), cur.get("schema"))
    if digest_of(cur) == digest_of(prev):
        return "REUSE", "identity unchanged"
    changed = [k for k in sorted(set(cur) | set(prev)) if cur.get(k) != prev.get(k)]
    return "BUILD", "changed: " + ", ".join(changed)


# ---------------------------------------------------------------- the previous release
def previous_release(root=ROOT):
    """The newest published release record: version, commit, wheels."""
    base = Path(root) / "bench" / "results" / "release_verification"
    best = None
    for d in sorted(base.iterdir()) if base.is_dir() else []:
        m = RECORD_RE.match(d.name)
        if not m or not (d / "alpha-manifest-linux.json").is_file():
            continue
        try:
            linux = json.loads((d / "alpha-manifest-linux.json").read_text())
        except (OSError, ValueError):
            continue
        version = linux.get("version")
        if not version:
            continue
        vt = tuple(int(x) for x in re.findall(r"\d+", version))
        if best is None or vt > best[0]:
            best = (vt, d, linux)
    if best is None:
        return None
    _, d, linux = best
    rec = dict(version=linux["version"], record_dir=str(d), source_commit=linux.get("light_smoke", {}).get("source_commit"))
    # "cuda" and "rocm": the split Linux plugins (python/mojolearn/gpu_plugins.py),
    # recorded beside the core's alpha-manifest-linux.json by a split release
    for platform in ("linux", "macos", "cuda", "rocm"):
        try:
            doc = json.loads((d / f"alpha-manifest-{platform}.json").read_text())
        except (OSError, ValueError):
            rec[platform] = None
            continue
        files = [(n, s) for n, s in doc.get("files", {}).items() if n.endswith(".whl")]
        rec[platform] = dict(wheel=files[0][0], sha256=files[0][1]) if len(files) == 1 else None
        if platform == "macos" and not rec["source_commit"]:
            rec["source_commit"] = doc.get("light_smoke", {}).get("source_commit")
    idents = d / "binding-identities.json"
    rec["identities"] = str(idents) if idents.is_file() else None
    return rec


def previous_identities(prev, cache_dir, root=ROOT):
    """{binding key: identity} and the runtime identity of the previous
    release: from its record when it wrote one, else from its commit's tree
    (then the Apple toolchain is unknown and no macOS binding is reused)."""
    if prev.get("identities"):
        doc = json.loads(Path(prev["identities"]).read_text())
        rows = {r["key"]: r.get("identity") for r in doc.get("rows", [])}
        runtime = doc.get("runtime", {}).get("identity")
        source = "record " + prev["identities"]
        commit = doc.get("commit") or prev.get("source_commit")
        olds = [v for v in list(rows.values()) + [runtime] if v]
        if commit and any(v.get("schema") == SCHEMA_V1 for v in olds):
            rows, runtime, n = upgrade_v1(rows, runtime, commit, cache_dir, root)
            source += " (%s identities upgraded after reproducing %d of them exactly from %s)" % (
                SCHEMA_V1, n, commit[:12])
        return rows, runtime, source
    if not prev.get("source_commit"):
        return {}, None, "no source commit recorded"
    facts = facts_for_commit(prev["source_commit"], cache_dir, root)
    rows = {b.key: identity(b, facts, None) for t in (LINUX, MACOS) for b in bindings(t)}
    return rows, runtime_identity(facts), "git archive " + prev["source_commit"][:12]


def upgrade_v1(rows, runtime, commit, cache_dir, root=ROOT):
    """Recorded v1 identities (whole-file builder digests) in today's schema.

    A v1 identity is upgraded ONLY when recomputing it in v1 from `commit`,
    the commit the record was made at (with the Apple toolchain the record
    names), gives exactly the recorded digest: that proves this code reads
    the same closure, toolchain, flags, image and builder bytes the release
    recorded, and the v2 identity is then computed from those same bytes.
    One that does not reproduce is marked refused, and compare() builds it.
    Returns (rows, runtime, number upgraded)."""
    facts = facts_for_commit(commit, cache_dir, root)
    by_key = {b.key: b for t in (LINUX, MACOS) for b in bindings(t)}
    refused = "its %s identity does not reproduce from %s" % (SCHEMA_V1, commit[:12])
    out, n = {}, 0
    for key, old in rows.items():
        b = by_key.get(key)
        if old is None or old.get("schema") != SCHEMA_V1 or b is None:
            out[key] = old
            continue
        again = identity(b, facts, old.get("host_toolchain"), schema=SCHEMA_V1)
        if again is not None and digest_of(again) == digest_of(old):
            out[key] = identity(b, facts, old.get("host_toolchain"))
            n += 1
        else:
            out[key] = dict(old, upgrade_refused=refused)
    if runtime and runtime.get("schema") == SCHEMA_V1:
        if digest_of(runtime_identity(facts, SCHEMA_V1)) == digest_of(runtime):
            runtime = runtime_identity(facts)
            n += 1
        else:
            runtime = dict(runtime, upgrade_refused=refused)
    return out, runtime, n


# ---------------------------------------------------------------- the plan
def make_plan(commit, cache_dir, root=ROOT, host_toolchain=None, prev=None, builders_override=None):
    """The REUSE/BUILD plan of COMMIT. `builders_override` ({builder: sha256})
    replaces a builder's bytes in every identity: the release's route overlay
    runs the tooling checkout's copy of a builder on the box
    (tools/release_tooling.py), so the identity is the bytes that ran."""
    commit = git("rev-parse", commit + "^{commit}", root=root).strip()
    if prev is None:
        prev = previous_release(root)
    facts = facts_for_commit(commit, cache_dir, root)
    if builders_override:
        unknown = set(builders_override) - set(facts["builders"])
        if unknown:
            raise SystemExit("release_reuse: the overlay names builders no identity reads: " + ", ".join(sorted(unknown)))
        # The v2 identities read per-binding builder views, not the whole-file
        # digest: an overridden builder's bytes are known only by their sha,
        # so every view of it (each binding's and the runtime's) becomes a
        # digest of that sha, and the overlay moves every identity that reads it.
        views = dict(facts.get("builder_views") or {})
        for rel, digest in builders_override.items():
            names = (views.get(rel) or {}).keys() or [b.name for t in (LINUX, MACOS) for b in bindings(t)] + [RUNTIME_VIEW]
            views[rel] = {name: hashlib.sha256(("%s\0%s\0override\0%s" % (BUILDER_RULE, rel, digest)).encode()).hexdigest()
                          for name in names}
        facts = dict(facts, builders=dict(facts["builders"], **builders_override), builder_views=views)
    if host_toolchain is None:
        host_toolchain = darwin_toolchain()
    if prev:
        prev_rows, prev_runtime, prev_source = previous_identities(prev, cache_dir, root)
        changed_files = set(git("diff", "--name-only", prev["source_commit"], commit, root=root).split()) \
            if prev.get("source_commit") else set()
    else:
        prev_rows, prev_runtime, prev_source, changed_files = {}, None, "no published release record", set()
    rows = []
    for target in (LINUX, MACOS):
        for b in bindings(target):
            cur = identity(b, facts, host_toolchain)
            old = prev_rows.get(b.key) if prev else None
            if prev is None:
                decision, reason = "BUILD", "no published release record"
            elif prev.get("linux" if target == LINUX else "macos") is None:
                decision, reason = "BUILD", "the previous release published no %s wheel" % target
            else:
                decision, reason = compare(cur, old)
            if decision == "BUILD" and reason.startswith("changed: closure"):
                closure = facts["closures"].get(b.script) or {}
                moved = [s for s in closure.get("sources", []) if s in changed_files][:6]
                if moved:
                    reason += " (" + ", ".join(moved) + ")"
            rows.append(dict(b.row(), decision=decision, reason=reason,
                             identity=cur, identity_digest=digest_of(cur) if cur else None,
                             previous_digest=digest_of(old) if old else None))
    rt = runtime_identity(facts)
    if prev is None or prev.get("linux") is None:
        rt_decision, rt_reason = "BUILD", "no published Linux wheel"
    elif prev_runtime is None:
        rt_decision, rt_reason = "BUILD", "previous release: no runtime identity"
    elif unreadable(prev_runtime) or prev_runtime.get("schema") != rt.get("schema"):
        rt_decision, rt_reason = "BUILD", "previous release: " + (
            unreadable(prev_runtime) or "identity recorded as %s" % prev_runtime.get("schema"))
    elif digest_of(rt) == digest_of(prev_runtime):
        rt_decision, rt_reason = "REUSE", "identity unchanged"
    else:
        rt_decision, rt_reason = "BUILD", "changed: " + ", ".join(
            k for k in sorted(set(rt) | set(prev_runtime)) if rt.get(k) != prev_runtime.get(k))
    runtime = dict(decision=rt_decision, reason=rt_reason, identity=rt, identity_digest=digest_of(rt),
                   previous_digest=digest_of(prev_runtime) if prev_runtime else None)
    legs, why = [], {}
    for vendor, arch in LINUX_SETS:
        need = [r for r in rows if r["target"] == LINUX and not r["tier"] == "host"
                and (r["vendor"], r["arch"]) == (vendor, arch) and r["decision"] == "BUILD"]
        if need:
            legs.append(f"{vendor}-{arch}")
            why[f"{vendor}-{arch}"] = f"{len(need)} binding(s) to build"
    host_build = [r for r in rows if r["target"] == LINUX and r["tier"] == "host" and r["decision"] == "BUILD"]
    if not legs and (host_build or rt_decision == "BUILD"):
        name = "%s-%s" % HOST_LEG
        legs.append(name)
        why[name] = ("%d host binding(s)" % len(host_build) if host_build else "the runtime closure") + \
            " need a build box and no set does: the cheapest leg builds them"
    return dict(schema=PLAN_SCHEMA, commit=commit, made=dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                previous=prev, previous_identities_from=prev_source, rows=rows, runtime=runtime,
                legs=legs, leg_reasons=why, host_toolchain=host_toolchain,
                builders_override=dict(sorted((builders_override or {}).items())))


def set_identity(plan, vendor, arch):
    """What a Linux build leg of (vendor, arch) produces, as one digest: the
    identity of every binding of its set, of every host binding (every leg
    builds them) and of the runtime closure it stages. Two plans with the same
    digest ask the leg for the same bytes. A binding without a readable
    identity (a new one, an unreadable pin) makes the set unreusable: None."""
    rows = {}
    for r in plan_rows(plan, LINUX):
        if r["tier"] == "host" or (r["vendor"], r["arch"]) == (vendor, arch):
            if not r.get("identity_digest") or unreadable(r.get("identity")):
                return None
            rows[r["key"]] = r["identity_digest"]
    rt = plan["runtime"]
    if not rows or not rt.get("identity_digest") or unreadable_runtime(rt.get("identity")):
        return None
    doc = dict(set=f"{vendor}/{arch}", bindings=rows, runtime=rt["identity_digest"])
    return dict(doc, digest=digest_of(doc))


def unreadable_runtime(ident):
    if not ident or not (ident.get("toolchain") or {}).get("packages"):
        return "no Mojo/MAX packages"
    if any(v is None for v in (ident.get("builders") or {}).values()):
        return "builder script missing"
    if not ident.get("image") or any(v is None for v in ident["image"].values()):
        return "box image pin unreadable"
    return None


def plan_rows(plan, target=None, decision=None):
    return [r for r in plan["rows"] if (target is None or r["target"] == target)
            and (decision is None or r["decision"] == decision)]


def set_name(r):
    if r["tier"] == "host":
        return "host" if r["target"] == LINUX else "macos host"
    return f"{r['vendor']}/{r['arch']}" if r["target"] == LINUX else "macos " + r["tier"]


def render(plan, full=False):
    """The decision table. Every BUILD row with its reason; REUSE rows one per
    line under --full, counted otherwise."""
    prev = plan.get("previous") or {}
    out = ["== binding reuse plan for %s against %s" % (
        plan["commit"][:12], f"{prev.get('version')} ({(prev.get('source_commit') or '?')[:12]})" if prev
        else "NO PUBLISHED RELEASE RECORD")]
    out.append("   previous identities from: " + plan["previous_identities_from"])
    groups = {}
    for r in plan["rows"]:
        groups.setdefault(set_name(r), []).append(r)
    out.append("   %-20s %5s %5s  %s" % ("set", "REUSE", "BUILD", "leg"))
    for name, rs in groups.items():
        n_r = sum(r["decision"] == "REUSE" for r in rs)
        n_b = len(rs) - n_r
        leg = ""
        if name in ("cuda/sm_90a", "cuda/sm_89", "hip/gfx942"):
            leg_name = name.replace("/", "-")
            leg = leg_name + ": " + plan["leg_reasons"][leg_name] if leg_name in plan["legs"] else "none"
        out.append("   %-20s %5d %5d  %s" % (name, n_r, n_b, leg))
    rt = plan["runtime"]
    out.append("   %-20s %5d %5d  %s" % ("runtime .libs", rt["decision"] == "REUSE", rt["decision"] == "BUILD", rt["reason"]))
    out.append("   legs to launch: " + (", ".join(plan["legs"]) or "none (nothing rented for builds)"))
    builds = plan_rows(plan, decision="BUILD")
    if builds:
        out.append("   BUILD:")
        for r in builds:
            out.append("     %-5s %-20s %-44s %s" % ("BUILD", set_name(r), r["set_rel"], r["reason"]))
    reuses = plan_rows(plan, decision="REUSE")
    if reuses and full:
        out.append("   REUSE (from %s, identity digest):" % prev.get("version"))
        for r in reuses:
            out.append("     %-5s %-20s %-44s %s" % ("REUSE", set_name(r), r["set_rel"], r["identity_digest"][:16]))
    elif reuses:
        out.append("   REUSE: %d bindings from %s (--full lists them)" % (len(reuses), prev.get("version")))
    return "\n".join(out)


# ---------------------------------------------------------------- published bytes
def published_wheel(prev, platform, evidence_root, download=True):
    """The previous release's wheel for `platform`, verified by sha256 against
    the record: a local copy under the evidence root when one is there,
    otherwise fetched from PyPI. Refuses anything whose digest differs."""
    info = prev.get(platform)
    if not info:
        raise SystemExit(f"release_reuse: the previous release published no {platform} wheel")
    name, want = info["wheel"], info["sha256"]
    ev = Path(evidence_root)
    store = ev / "release" / "published"
    candidates = [store / name]
    candidates += sorted((ev / "release" / prev["version"]).glob(f"*/linux/final/{name}"))
    candidates += sorted((ev / "release" / prev["version"]).glob(f"*/macos/{name}"))
    passed_over = []
    for c in candidates:
        if c.is_file():
            if sha256(c) == want:
                return c
            # a wheel of the same name from another freeze of that release (a
            # release refrozen before publication leaves one behind): not the
            # published bytes, so not this one, and no reason to stop looking
            passed_over.append(c)
    if not download:
        if passed_over:
            raise SystemExit(f"release_reuse: no local copy of the published {name} (sha256 {want[:12]}); "
                             f"passed over {len(passed_over)} of another freeze: {', '.join(str(c) for c in passed_over)}")
        return None
    project = name.split("-")[0].replace("_", "-")      # mojolearn, mojolearn-cuda, mojolearn-rocm
    with urllib.request.urlopen(f"https://pypi.org/pypi/{project}/{prev['version']}/json", timeout=30) as r:
        files = json.load(r).get("urls", [])
    urls = [f["url"] for f in files if f.get("filename") == name and f.get("digests", {}).get("sha256") == want]
    if not urls:
        raise SystemExit(f"release_reuse: PyPI does not serve {name} with sha256 {want[:12]}; refusing to reuse")
    store.mkdir(parents=True, exist_ok=True)
    tmp = store / (name + ".part")
    with urllib.request.urlopen(urls[0], timeout=120) as r, open(tmp, "wb") as fh:
        shutil.copyfileobj(r, fh)
    if sha256(tmp) != want:
        tmp.unlink()
        raise SystemExit(f"release_reuse: the download of {name} does not match the recorded sha256; refusing")
    tmp.replace(store / name)
    return store / name


def linux_payload(whl):
    with zipfile.ZipFile(whl) as z:
        names = [n for n in z.namelist() if n.endswith(".dist-info/LINUX_PAYLOAD.json")]
        if len(names) != 1:
            raise SystemExit(f"release_reuse: {whl} carries no LINUX_PAYLOAD.json")
        return json.loads(z.read(names[0]))


def extract_member(z, member, dest, want_sha=None):
    dest.parent.mkdir(parents=True, exist_ok=True)
    data = z.read(member)
    got = hashlib.sha256(data).hexdigest()
    if want_sha is not None and got != want_sha:
        raise SystemExit(f"release_reuse: {member} in the published wheel has sha256 {got[:12]}, "
                         f"its payload says {want_sha[:12]}; refusing to reuse")
    dest.write_bytes(data)
    os.chmod(dest, 0o755)
    return got


def elf_driver_libs(paths):
    """The vendor driver libraries the set's binaries NEED but do not ship
    (stage_libs.py's classes), read from the ELF headers of the published
    bytes; audit.sh excludes exactly these."""
    driver = set()
    for p in paths:
        needed, _, _ = stage_libs.elf_dynamic(p)
        for dep in needed:
            if any(dep.startswith(x) for x in stage_libs.DRIVER_PREFIXES):
                driver.add(dep)
    return sorted(driver)


def assemble_linux(plan, whl, legs, dest, say=print, leg_reuse=None):
    """The set directories pack_wheel.py packs, one per (vendor, arch): a
    leg's set copied whole when the leg ran, the set of a completed leg of an
    earlier freeze when tools/release.py admitted it (`leg_reuse`: its set
    identity and build tooling equal this freeze's), else synthesized from
    the published wheel; then the plan's REUSE bindings, host bindings and
    the runtime closure written from the published bytes, each verified
    against the wheel's payload, with reuse.json naming every reused file and
    where it came from (`source` release or leg).

    `legs` maps "vendor-arch" to the sets/<vendor>/<arch> directory of a leg
    that built at this freeze; `leg_reuse` maps "vendor-arch" to
    {"dir": that directory of the earlier leg, "origin": {source_commit, leg,
    proof_sha256, admitted_for, set_identity_digest, tooling_digest, ...}}.
    `whl` (the published wheel) may be None when nothing is taken from it.
    Returns {"vendor/arch": set dir}."""
    prev = plan["previous"] or {}
    leg_reuse = leg_reuse or {}
    need_published = bool(plan_rows(plan, LINUX, "REUSE")) or plan["runtime"]["decision"] == "REUSE" or any(
        not legs.get(f"{v}-{a}") and not leg_reuse.get(f"{v}-{a}") for v, a in LINUX_SETS)
    if need_published and whl is None:
        raise SystemExit("release_reuse: the plan takes bytes from the published wheel and none was given")
    payload, origin, members = {}, None, set()
    if whl is not None:
        payload = linux_payload(whl)
        if payload.get("source_commit") != prev.get("source_commit"):
            raise SystemExit("release_reuse: the published wheel's payload names commit %s, the record %s" % (
                (payload.get("source_commit") or "?")[:12], (prev.get("source_commit") or "?")[:12]))
        origin = dict(version=prev["version"], source_commit=prev["source_commit"], wheel=Path(whl).name,
                      wheel_sha256=sha256(whl))
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    rows = {r["archive_path"]: r for r in plan_rows(plan, LINUX)}
    host_rows = [r for r in rows.values() if r["tier"] == "host"]
    out = {}
    z = zipfile.ZipFile(whl) if whl is not None else None
    try:
        if z is not None:
            members = set(z.namelist())
        shared = payload.get("runtime_layout") == "shared"
        for vendor, arch in LINUX_SETS:
            key = f"{vendor}/{arch}"
            name = f"{vendor}-{arch}"
            leg = legs.get(name)
            old = leg_reuse.get(name)
            sdir = dest / vendor / arch
            reused, from_legs = {}, {}
            gpu = [r for r in rows.values() if r["tier"] != "host" and (r["vendor"], r["arch"]) == (vendor, arch)]
            if leg:
                shutil.copytree(leg, sdir, symlinks=False)
                say(f"  {key}: the leg's set copied from {leg}")
            elif old:
                # A COMPLETED LEG OF AN EARLIER FREEZE whose set identity and
                # build tooling equal this freeze's: every file it built is
                # taken whole, byte for byte, and named in reuse.json.
                shutil.copytree(old["dir"], sdir, symlinks=False)
                from_legs[name] = dict(old["origin"])
                for r in gpu + host_rows:
                    p = sdir / r["set_rel"]
                    if p.is_file():
                        reused[r["set_rel"]] = dict(sha256=sha256(p), source="leg", leg=name,
                                                    identity_digest=r["identity_digest"])
                for p in sorted((sdir / ".libs").glob("*")) if (sdir / ".libs").is_dir() else []:
                    if p.is_file():
                        reused[".libs/" + p.name] = dict(sha256=sha256(p), source="leg", leg=name)
                say(f"  {key}: the set of leg {name} built at {old['origin']['source_commit'][:12]} taken whole "
                    f"(set identity {old['origin']['set_identity_digest'][:12]} unchanged)")
            else:
                sdir.mkdir(parents=True)
            if not leg and not old and any(r["decision"] != "REUSE" for r in gpu):
                raise SystemExit(f"release_reuse: {key} has bindings to BUILD and no leg ran")
            for r in gpu:
                if r["decision"] != "REUSE":
                    continue
                want = payload.get("extensions", {}).get(r["archive_path"])
                if want is None or r["archive_path"] not in members:
                    raise SystemExit(f"release_reuse: {r['archive_path']} is not in the published {prev['version']} wheel")
                target = sdir / r["set_rel"]
                rebuilt = sha256(target) if target.is_file() else None
                got = extract_member(z, r["archive_path"], target, want)
                if rebuilt is not None and rebuilt != got:
                    if vendor == "cuda":
                        raise SystemExit(f"release_reuse: the {key} leg rebuilt {r['set_rel']} with different bytes "
                                         f"({rebuilt[:12]} against the published {got[:12]}) although its identity is "
                                         "unchanged; an NVIDIA rebuild must reproduce. Find the cause before reusing.")
                    say(f"  NOTE {key} {r['set_rel']}: the leg's rebuild differs ({rebuilt[:12]}); the published "
                        f"bytes {got[:12]} are packed (gfx942 codegen is not reproducible run to run)")
                reused[r["set_rel"]] = dict(sha256=got, archive_path=r["archive_path"], identity_digest=r["identity_digest"],
                                            rebuilt_sha256=rebuilt)
            # the host bindings: the published copy where the plan says so,
            # the leg's otherwise (a set without a leg takes a leg's copy)
            for r in host_rows:
                rec = payload.get("host_native", {}).get(r["name"], {})
                target = sdir / r["set_rel"]
                if r["decision"] == "REUSE":
                    want = rec.get("sha256")
                    if want is None or r["archive_path"] not in members:
                        raise SystemExit(f"release_reuse: {r['archive_path']} is not in the published {prev['version']} wheel")
                    rebuilt = sha256(target) if target.is_file() else None
                    got = extract_member(z, r["archive_path"], target, want)
                    if rebuilt is not None and rebuilt != got:
                        raise SystemExit(f"release_reuse: the {key} leg built host binding {r['name']} with bytes "
                                         f"{rebuilt[:12]}, the published copy is {got[:12]}, and its identity is "
                                         "unchanged. A host build must reproduce; find the cause.")
                    reused[r["set_rel"]] = dict(sha256=got, archive_path=r["archive_path"], identity_digest=r["identity_digest"],
                                                rebuilt_sha256=rebuilt)
                elif not target.is_file():
                    built = [(n, d / r["set_rel"]) for n, d in legs.items() if (d / r["set_rel"]).is_file()]
                    taken = [(n, o) for n, o in leg_reuse.items() if (Path(o["dir"]) / r["set_rel"]).is_file()]
                    if built:
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(built[0][1], target)
                    elif taken:
                        n, o = taken[0]
                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(Path(o["dir"]) / r["set_rel"], target)
                        from_legs[n] = dict(o["origin"])
                        reused[r["set_rel"]] = dict(sha256=sha256(target), source="leg", leg=n,
                                                    identity_digest=r["identity_digest"])
                    else:
                        raise SystemExit(f"release_reuse: host binding {r['name']} must be built and no leg built it")
            # the runtime closure
            libs = sdir / ".libs"
            if plan["runtime"]["decision"] == "REUSE":
                prefix = "mojolearn/.libs/" if shared else f"mojolearn/{vendor}/.libs/"
                if libs.exists():
                    shutil.rmtree(libs)
                for k in [k for k in reused if k.startswith(".libs/")]:
                    del reused[k]
                for member, want in sorted(payload.get("runtime_sha256", {}).items()):
                    if member.startswith(prefix):
                        got = extract_member(z, member, libs / member[len(prefix):], want)
                        reused[".libs/" + member[len(prefix):]] = dict(sha256=got, archive_path=member)
                if not libs.is_dir() or not any(libs.iterdir()):
                    raise SystemExit(f"release_reuse: no runtime closure under {prefix} in the published wheel")
            elif not libs.is_dir():
                built = [d / ".libs" for d in legs.values() if (d / ".libs").is_dir()]
                taken = [(n, Path(o["dir"]) / ".libs", o) for n, o in leg_reuse.items() if (Path(o["dir"]) / ".libs").is_dir()]
                if built:
                    shutil.copytree(built[0], libs)
                elif taken:
                    n, src, o = taken[0]
                    shutil.copytree(src, libs)
                    from_legs[n] = dict(o["origin"])
                    for p in sorted(libs.iterdir()):
                        reused[".libs/" + p.name] = dict(sha256=sha256(p), source="leg", leg=n)
                else:
                    raise SystemExit("release_reuse: the runtime closure must be built and no leg built it")
            if not leg and not old:
                exts = [sdir / r["set_rel"] for r in gpu]
                staged = [dict(name=p.name, sha256=sha256(p), bytes=p.stat().st_size) for p in sorted(libs.iterdir())]
                manifest = dict(schema="mojolearn.linux.reused-set-manifest.v1", set=str(sdir), reused_from=origin,
                                bytes_extensions=sum(p.stat().st_size for p in exts),
                                bytes_staged_libs=sum(s["bytes"] for s in staged), staged_libs=staged,
                                driver_libs_not_staged=elf_driver_libs(exts))
                (sdir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
            if origin is not None or reused:
                doc = dict(schema=REUSE_SCHEMA, set=key, from_release=origin, files=reused)
                if from_legs:
                    doc["from_legs"] = from_legs
                (sdir / "reuse.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
            n_leg = sum(1 for r in reused.values() if r.get("source") == "leg")
            say(f"  {key}: {len(reused) - n_leg} file(s) taken from {origin['wheel'] if origin else 'no published wheel'}"
                + (f", {n_leg} from the earlier leg(s) {', '.join(sorted(from_legs))}" if n_leg else "")
                + ("" if leg or old else " (no leg; set synthesized)"))
            out[key] = sdir
    finally:
        if z is not None:
            z.close()
    return out


def assemble_macos(plan, whl, dest, say=print):
    """The reused macOS bindings extracted from the published wheel into
    dest/python/mojolearn/..., verified against the wheel's RECORD, and
    dest/macos-plan.json for build_release_wheel.sh: repo-relative output
    path -> {sha256, archive_path, release}."""
    prev = plan["previous"]
    dest = Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    files = {}
    with zipfile.ZipFile(whl) as z:
        names = [n for n in z.namelist() if n.endswith(".dist-info/RECORD")]
        record = {}
        for line in z.read(names[0]).decode().splitlines():
            parts = line.split(",")
            if len(parts) == 3 and parts[1].startswith("sha256="):
                import base64
                raw = base64.urlsafe_b64decode(parts[1][7:] + "=" * (-len(parts[1][7:]) % 4))
                record[parts[0]] = raw.hex()
        for r in plan_rows(plan, MACOS, "REUSE"):
            want = record.get(r["archive_path"])
            if want is None:
                raise SystemExit(f"release_reuse: {r['archive_path']} is not in the published macOS wheel's RECORD")
            # a plan row carries archive_path, not package_rel (Binding.package_rel
            # is "python/" + archive_path, where build_release_wheel.sh writes it);
            # 0.8.19 was the first release to reuse a macOS binding and hit this
            package_rel = r.get("package_rel") or "python/" + r["archive_path"]
            got = extract_member(z, r["archive_path"], dest / package_rel, want)
            files[package_rel] = dict(sha256=got, archive_path=r["archive_path"], identity_digest=r.get("identity_digest"))
    doc = dict(schema="mojolearn.macos.reused-bindings.v1", from_release=dict(
        version=prev["version"], source_commit=prev["source_commit"], wheel=Path(whl).name, wheel_sha256=sha256(whl)),
        files=files)
    (dest / "macos-plan.json").write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    say(f"  macOS: {len(files)} binding(s) taken from {Path(whl).name}")
    return dest / "macos-plan.json"


def record_identities(plan, linux_wheel, macos_wheel, out):
    """binding-identities.json for the release record: every row's identity,
    decision and the sha256 it shipped with, so the next release can decide
    from the record alone (the Apple toolchain included)."""
    shipped = {}
    # linux_wheel is one wheel, or the split set (core and plugins) as a list
    linux = list(linux_wheel) if isinstance(linux_wheel, (list, tuple)) else [linux_wheel]
    for whl in (*linux, macos_wheel):
        if not whl or not Path(whl).is_file():
            continue
        with zipfile.ZipFile(whl) as z:
            for n in z.namelist():
                if n.endswith(".so") and n.startswith("mojolearn/"):
                    shipped[n] = hashlib.sha256(z.read(n)).hexdigest()
    rows = []
    for r in plan["rows"]:
        rows.append(dict(key=r["key"], target=r["target"], archive_path=r["archive_path"], decision=r["decision"],
                         reason=r["reason"], identity=r["identity"], identity_digest=r["identity_digest"],
                         shipped_sha256=shipped.get(r["archive_path"])))
    doc = dict(schema="mojolearn.binding-identities.v1", commit=plan["commit"], rows=rows, runtime=plan["runtime"],
               previous=(plan.get("previous") or {}).get("version"))
    Path(out).write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


# ---------------------------------------------------------------- CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("plan",))
    ap.add_argument("--commit", default="HEAD")
    ap.add_argument("--json", default="", help="write the plan here")
    ap.add_argument("--full", action="store_true", help="list every REUSE row")
    ap.add_argument("--cache", default="", help="identity cache directory (default <evidence>/release/identities)")
    a = ap.parse_args(argv)
    ev = Path(os.environ.get("MOJOLEARN_EVIDENCE_ROOT", os.path.expanduser("~/mojolearn-evidence")))
    cache = Path(a.cache) if a.cache else ev / "release" / "identities"
    plan = make_plan(a.commit, cache)
    print(render(plan, a.full))
    if a.json:
        Path(a.json).parent.mkdir(parents=True, exist_ok=True)
        Path(a.json).write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
