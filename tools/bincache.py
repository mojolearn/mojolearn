#!/usr/bin/env python3
"""tools/bincache.py -- a content-addressed cache of prebuilt Mojo bindings in
the R2 bucket that already holds the datasets (lane/r2-binding-cache,
2026-09-15). DEFAULT OFF.

WHY. Every remote GPU leg compiled about 22 bindings from source before doing
any work: a DigitalOcean MI325X leg spent 10.5 minutes building, a four-box
record about 40. A binding is a pure function of its sources, the toolchain,
the build environment and the box image, so a second box with the same inputs
can take the first box's bytes instead of recompiling them.

WHAT THE KEY IS. sha256 over a canonical JSON of every input that changes the
bytes (`key_fields`). Since 2026-11-04 the default rule is closure-v2
(KEY_RULE, the section "key rule closure-v2" below): the exact import closure
of the build's own roots (for bindings/merged_build.sh, the members of the
group it compiles), no whole-tree fallback (an unresolvable import fails the
key, naming it), and only the variables a keyed input reads. The first rule,
described next, stays computable (`rule="v1"`, MOJOLEARN_BINCACHE_SOURCE=tree)
so archives and receipts keyed under it remain verifiable:
  * the build script and the sha256 of every source file the binding can
    reach: the Mojo import closure of the file its `mojo build` line names,
    over-approximated file by file (any import that cannot be resolved falls
    back to every .mojo file in the tree, never to fewer files), plus the
    shell scripts it execs, pixi.toml and pixi.lock;
  * the Mojo/MAX toolchain packages pinned in pixi.lock for this platform;
  * the numeric mode, MOJOLEARN_GPU_ARCHS, MOJOLEARN_TARGET_COLUMN and EVERY
    other MOJOLEARN_*, MOJO_* and MODULAR_* variable in the environment
    except a short list that cannot reach a compiler (NON_BUILD_ENV);
  * the device architecture the box reports (nvidia-smi or rocminfo);
  * the container image the runner declared, and the OS the build actually
    ran in: os-release ID and VERSION_ID, glibc, machine, the first line of
    `ld --version` and `cc --version` (the Sep 13 incident: DO 24.04 linked
    a different CPU binding than RunPod 22.04 from one source), and -- on the
    machines where bindings/build_*.sh does NOT pin `--target-cpu`, which is
    every Linux that is not x86-64 -- the host CPU itself (`host_cpu`, added
    2026-09-19: on aarch64 the compiler targets whatever core ran it, so
    `machine=aarch64` alone would let a Neoverse-V2 binary be served to a
    Neoverse-N1 box);
  * the absolute repository path (the rpath into .pixi is baked in).
A build whose environment or arguments mention SABOTAGE or FAULT_INJECT is
NEVER looked up and NEVER uploaded, unless the runner opts in with
MOJOLEARN_BINCACHE_NEGATIVE=1 (tools/runpod_cpu_leg.sh, 2026-09-15). Then the
build is a NEGATIVE CONTROL: its fields carry variant=sabotage (so its key can
never equal a production key), it is looked up only among the map's `sget`
rows, which list the separate prefix bincache/sabotage-v1/, and it uploads
only there. A production build never reads an `sget` row, and an archive of
the other variant fails the fields check on placement.

CREDENTIALS NEVER REACH A BOX. This follows tools/dataset_store.sh exactly:
the Mac lists the cache and mints short-lived presigned URLs (`plan`), and the
runner pipes them INSIDE a script over ssh stdin into a 0600 file,
/root/.mojolearn_bincache/urls.tsv, which is outside every leg output
directory. The box can GET the objects that already exist and PUT only into
per-leg inbox slots; the Mac copies an inbox object to its content address
after the leg (`promote`), server side. No URL is ever printed or logged.
A key holds ONE content: promote, the hot directory and the local cache never
write a key that already holds other files (a loud BINCACHE CONFLICT line with
both sha256; promote then exits 4); equal files are a no-op.

  ON THE BOX (inside a body; a pass-through when the URL map is absent or
  MOJOLEARN_BINCACHE=0):
    python3 tools/bincache.py build bindings/build_rf.sh [args...]
    python3 tools/bincache.py key bindings/build_rf.sh     # print the fields
    python3 tools/bincache.py explain bindings/merged_build.sh gpu --group G
        # one line, before anything compiles: `hit (<where>)` or `compile: <reason>`
  ON THE MAC (through tools/bincache_leg.sh, which loads ~/.mojolearn_r2):
    python3 tools/bincache.py plan --partition P --image I --leg-id L
    python3 tools/bincache.py promote <leg out>/remote/bincache
    python3 tools/bincache.py selftest-r2
  ON THE MAC, LOCALLY (the macOS release build; no R2, no credentials):
    MOJOLEARN_BINCACHE_DIR=~/.mojolearn-bincache/macos-release \
    MOJOLEARN_BINCACHE_OUTPUTS=python/mojolearn/identical/_mojolearn_svm.so \
      python3 tools/bincache.py build bindings/build_svm.sh
    (packaging/macos/build_release_wheel.sh does this for every pair when
    MOJOLEARN_BINCACHE_DIR is set; the section "a LOCAL directory cache"
    below has the extra key fields and the rules.)

Provenance lands in $MOJOLEARN_BINCACHE_OUT (default
/root/gemm_leg_out/bincache): provenance.tsv names every build, whether it
was a cache hit, and the sha256 of every file it placed; keys/<key>.json holds
the fields. Stdlib only.
"""
import atexit
import datetime
import hashlib
import hmac
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

SCHEMA = "mojolearn-bincache-v1"
DEFAULT_MAP = "/root/.mojolearn_bincache/urls.tsv"
DEFAULT_OUT = "/root/gemm_leg_out/bincache"
OBJECT_PREFIX = "bincache/v1"
SABOTAGE_PREFIX = "bincache/sabotage-v1"
INBOX_PREFIX = "bincache/inbox"
KEY_ENV_PREFIXES = ("MOJOLEARN_", "MOJO_", "MODULAR_")
# Variables that name a commit, a thread count for RUNNING, a log or this
# cache. Nothing here reaches `mojo build`; everything else with the prefixes
# above is keyed, including the compile job counts, because a parallel
# codegen split has not been shown to be byte-neutral.
NON_BUILD_ENV = ("MOJOLEARN_COMMIT", "MOJOLEARN_CPU_THREADS", "MOJOLEARN_SKIP_BUILD_GATE",
                 "MOJOLEARN_BUILD_LOCK_HELD", "MOJOLEARN_PYTHON", "MOJOLEARN_SMOKE_SO",
                 # THE RELEASE BUILD'S SCHEDULING AND WITNESSES (2026-09-22,
                 # lane/release-cpu-build-box). MOJOLEARN_BUILD_JOBS is how many
                 # build SCRIPTS packaging/linux/build_sets.sh runs at once; no
                 # script under bindings/ reads it (each compiler keeps
                 # MOJOLEARN_COMPILE_JOBS=2, which stays keyed), and the proof
                 # at d181d9792 measured it byte-neutral: 4 jobs on the GPU
                 # boxes and 16 on the CPU box gave the same sha256 for every
                 # shipped binary. The other two are the release runner's
                 # interpreter path and the AMD leg's core-host witness.
                 "MOJOLEARN_BUILD_JOBS", "MOJOLEARN_QUALIFY_PYTHON", "MOJOLEARN_EXPECT_CORE_HOST_SHA256",
                 # THE RELEASE RUNNER'S OWN PROVISIONING SWITCHES (2026-11-05,
                 # lane/orchestrator-flag-not-a-build-input). tools/release.py
                 # reads these from its own environment to choose boxes,
                 # providers, workflow refs, state dirs, an index URL and a
                 # runtime budget; MOJOLEARN_ORCHESTRATOR=1 is REQUIRED by its
                 # gate-boxes/boxes-down steps (release.py:4359). Nothing under
                 # bindings/ or packaging/ reads any of them (checked with rg),
                 # so none reaches `mojo build`. Before this, the v1 build_env
                 # (and so release_requirements.build_flags) swept them in, and
                 # release_compile_evidence.supports_requirement refused every
                 # linux-64 compile requirement captured under
                 # MOJOLEARN_ORCHESTRATOR=1 (0.8.66 core-linux stop). KEYS: this
                 # changes a v1 key only for an environment that carried one of
                 # these; closure-v2 keys (KEY_RULE) never held them, because v2
                 # keys a MOJOLEARN_* variable only when a keyed input reads it.
                 # No rule bump.
                 "MOJOLEARN_ORCHESTRATOR", "MOJOLEARN_AMD_PROVIDER",
                 "MOJOLEARN_RELEASE_NV_BOX", "MOJOLEARN_RELEASE_AMD_BOX", "MOJOLEARN_RELEASE_NVC",
                 "MOJOLEARN_RELEASE_DO_AMD", "MOJOLEARN_RELEASE_BOX_PERSIST",
                 "MOJOLEARN_XCC_WORKFLOW_REF", "MOJOLEARN_APPLE_RUNTIME_SECONDS",
                 "MOJOLEARN_DEVPOD_STATE", "MOJOLEARN_STEWARD_DO_STATE", "MOJOLEARN_PUBLISHED_INDEX_URL")
NON_BUILD_PREFIXES = ("MOJOLEARN_BINCACHE", "MOJOLEARN_IDENTITY_", "MOJOLEARN_STAGE_",
                      "MOJOLEARN_GEMM_LEG_", "MOJOLEARN_HOTAISLE_", "MOJOLEARN_DO_",
                      # MOJOLEARN_GAP_ IS WHICH LANES THE LEG WILL RUN AFTERWARDS
                      # AND WHERE IT WILL FILE THEM (tools/gap_column_leg.sh:
                      # _LANES, _SLUG, _COMMIT_DIR, _BUDGET, _TWO_DEVICE,
                      # _DEGENERATE). Nothing under bindings/ reads any of them
                      # -- checked: `grep -rln MOJOLEARN_GAP bindings/` is
                      # empty -- so not one of them can reach `mojo build`, and
                      # a body that exports them before calling this file was
                      # partitioning the cache by its own errand.
                      #
                      # MEASURED, 2026-09-19, on the first two legs ever to
                      # read this cache. They put 110 objects into the first
                      # GPU partitions it has ever had (sm_89/ and gfx942/, 55
                      # families each, 1731 s and 1082 s of compiling) and
                      # every one is keyed to
                      #   MOJOLEARN_GAP_LANES = gbdt-ordered,...
                      #   MOJOLEARN_GAP_SLUG = gbdt-class-gaps
                      #   MOJOLEARN_GAP_COMMIT_DIR = bench/results/.../
                      #   MOJOLEARN_GAP_BUDGET = 2600
                      # (read out of their own keys/<key>.json, not inferred).
                      # `bindings/build_gbdt.sh` at one commit, one arch, one
                      # image and one OS keyed f98190e48ddcdb06 under that lane
                      # list and cf6f513b683fb1cf under the next leg's. So the
                      # only leg that could ever have hit those 110 objects is
                      # one running the SAME LANES into the SAME DIRECTORY --
                      # which is the one leg nobody ever needs to run twice.
                      "MOJOLEARN_GAP_",
                      # tools/release061_remote_build.sh's budget and route
                      # switches (MOJOLEARN_RELEASE_BUILD_SECONDS shrinks by the
                      # elapsed time inside the AMD container, so keying it
                      # would make every release a miss). Nothing under
                      # bindings/ reads a MOJOLEARN_RELEASE_ variable.
                      "MOJOLEARN_RELEASE_")
SABOTAGE_RE = re.compile(r"SABOTAGE|FAULT_INJECT", re.I)
# Import roots that are the toolchain's, not this tree's: the precompiled
# packages in .pixi/envs/default/lib/mojo/*.mojoc for mojo 1.0.0 / max 26.5.0
# (measured 2026-09-15), plus std's historical top-level names. A name not in
# this set that matches no file in the tree widens the key to every .mojo
# file, so a new toolchain package costs hits, never correctness. The
# toolchain itself is keyed through pixi.lock.
EXTERNAL_TOP = {"std", "max", "python", "builtin", "memory", "gpu", "sys", "os", "collections",
                "math", "testing", "time", "random", "algorithm", "utils", "bit", "complex",
                "pathlib", "subprocess", "tempfile", "hashlib", "benchmark", "logger", "runtime",
                "compile", "iter", "io", "format", "_cublas", "_cudnn", "_cufft", "_hal",
                "_miopen", "_rocblas", "builtin_kernels", "builtin_primitives", "comm",
                "extensibility", "internal_utils", "kv_cache", "layout", "linalg", "machine",
                "matmul_rs", "mega_ffn", "msa", "nn", "pipeline", "profiling_range",
                "quantization", "shmem", "state_space", "structured_kernels", "weights_registry"}
KEY_RE = re.compile(r"^[0-9a-f]{64}$")
SEG_RE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")


class Reject(Exception):
    """A cached archive that must not be used."""


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def key_of(fields):
    return sha256_bytes(canonical(fields).encode())


def slug(text):
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-")
    return s[:120] or "none"


# ---------------------------------------------------------------------------
# sources: the import closure of the file a build script compiles
# ---------------------------------------------------------------------------

FROM_RE = re.compile(r"^\s*from\s+(\.*[A-Za-z_][\w.]*|\.+)\s+import\s+(.*)$")
IMPORT_RE = re.compile(r"^\s*import\s+([A-Za-z_][\w.]*(?:\s+as\s+\w+)?(?:\s*,\s*[A-Za-z_][\w.]*(?:\s+as\s+\w+)?)*)\s*$")


def parse_imports(text):
    """(module, [names]) for every import statement outside a docstring."""
    out = []
    in_doc = False
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        i += 1
        n_quotes = line.count('"""')
        if in_doc:
            if n_quotes % 2 == 1:
                in_doc = False
            continue
        if n_quotes % 2 == 1:
            in_doc = True
            continue
        code = line.split("#", 1)[0]
        m = FROM_RE.match(code)
        if m:
            names = m.group(2).strip()
            if names.startswith("(") and ")" not in names:
                while i < len(lines) and ")" not in names:
                    names += " " + lines[i].split("#", 1)[0]
                    i += 1
            names = names.strip("() \t")
            out.append((m.group(1), [n.split()[0] for n in names.split(",") if n.strip()]))
            continue
        m = IMPORT_RE.match(code)
        if m:
            for part in m.group(1).split(","):
                out.append((part.split()[0], []))
    return out


#: Directory listings exact_source_file reads, keyed by directory, when a
#: caller that knows the tree does not change under it opts in
#: (cache_directory_listings; tools/lane_select.py resolves ~10^6 imports over
#: one tree). None, the default, lists the directory on every call.
_DIR_NAMES = None


def cache_directory_listings(enabled=True):
    """Opt in to (or out of, clearing it) the per-process listing cache."""
    global _DIR_NAMES
    _DIR_NAMES = {} if enabled else None


def _names_in(directory):
    if _DIR_NAMES is None:
        return {p.name for p in directory.iterdir()}
    key = str(directory)
    names = _DIR_NAMES.get(key)
    if names is None:
        names = _DIR_NAMES[key] = frozenset(p.name for p in directory.iterdir())
    return names


def exact_source_file(path, root):
    """Require on-disk spelling even on a case-insensitive filesystem.

    An imported symbol such as Tape is not the module tape.mojo. Admitting
    that alias can copy a closure under Tape.mojo, which Mojo cannot import.
    Check directory components too, without changing the caller's root.
    """
    if not path.is_file():
        return False
    parent = root
    for part in path.relative_to(root).parts:
        if part not in (".", "..") and part not in _names_in(parent):
            return False
        parent = parent / part
    return True


def resolve(module, names, importer, roots):
    """Every existing file the import could mean, or None if it names this
    tree and nothing matches (the caller then widens to the whole tree)."""
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        base = importer.parent
        for _ in range(dots - 1):
            base = base.parent
        parts = [p for p in module.lstrip(".").split(".") if p]
        search = [base]
        external = False
    else:
        parts = module.split(".")
        search = list(roots) + [importer.parent]
        external = parts[0] in EXTERNAL_TOP
    found = set()
    for root in search:
        for k in range(1, len(parts) + 1):
            init = root.joinpath(*parts[:k], "__init__.mojo")
            if exact_source_file(init, root):
                found.add(init)
        if parts:
            leaf = root.joinpath(*parts[:-1], parts[-1] + ".mojo")
            if exact_source_file(leaf, root):
                found.add(leaf)
        pkg = root.joinpath(*parts) if parts else root
        for name in names:
            if name == "*":
                if pkg.is_dir():
                    found.update(p for p in pkg.glob("*.mojo") if p.is_file())
                continue
            for cand in (pkg / (name + ".mojo"), pkg / name / "__init__.mojo"):
                if exact_source_file(cand, root):
                    found.add(cand)
    if not found and not external:
        return None
    return found


def tree_sources(repo):
    files = []
    for dirpath, dirnames, filenames in os.walk(repo):
        dirnames[:] = [d for d in dirnames if d not in (".pixi", ".git", "__pycache__")]
        for f in filenames:
            if f.endswith(".mojo"):
                files.append(Path(dirpath) / f)
    return files


def script_plan(repo, script, args):
    """(root .mojo files, import roots, shell files) for one build script, or
    None when the script's compile line cannot be read literally."""
    script = Path(script)
    shells = [script]
    text = script.read_text(errors="replace")
    m = re.search(r"build_host_family\.sh\"?\s+([a-z_]+)", text)
    family = None
    if script.name == "build_host_family.sh":
        family = args[0] if args else None
    elif m and "exec" in text:
        family = m.group(1)
        script = repo / "bindings" / "build_host_family.sh"
        shells.append(script)
        text = script.read_text(errors="replace")
    for ref in re.findall(r"(tools/[\w.-]+\.sh)", text):
        if (repo / ref).is_file():
            shells.append(repo / ref)
    joined = text.replace("\\\n", " ")
    roots, incs = [], {repo}
    for line in joined.splitlines():
        s = line.strip()
        if s.startswith("#") or "mojo build" not in s:
            continue
        srcs = [t for t in re.findall(r"(\S+\.mojo)\b", s)]
        for inc in re.findall(r"-I\s+(\S+)", s):
            if "$" in inc:
                return None
            incs.add((repo / inc).resolve())
        if family is not None and not srcs:
            srcs = ["bindings/entry/entry__mojolearn_%s_host.mojo" % family]
        if len(srcs) != 1 or "$" in srcs[0]:
            return None
        roots.append(repo / srcs[0].strip("\"'"))
    if not roots or not all(r.is_file() for r in roots):
        return None
    return roots, sorted(incs), shells


#: Why the legacy (v1) source key of a build last widened to the whole tree, per (repo, script, args):
#: `bincache.py explain` prints it. Kept outside the key fields so v1 keys stay byte-identical.
WIDENED = {}


def source_digest(repo, script, args, force_tree=False):
    repo = Path(repo).resolve()
    plan = None if force_tree else script_plan(repo, script, args)
    widened_key = (str(repo), str(Path(script).resolve()), tuple(args))
    WIDENED.pop(widened_key, None)
    if plan is None and not force_tree:
        WIDENED[widened_key] = "the compile line names no readable root; the legacy key widened to the whole tree"
    scope = "closure"
    files = set()
    if plan is not None:
        roots, incs, shells = plan
        todo = list(roots)
        while todo:
            f = todo.pop()
            if f in files:
                continue
            files.add(f)
            for module, names in parse_imports(f.read_text(errors="replace")):
                got = resolve(module, names, f, incs)
                if got is None:
                    WIDENED[widened_key] = "unresolved import %r in %s widened the closure" % (
                        module, os.path.relpath(f, repo))
                    plan = None
                    break
                todo.extend(g.resolve() for g in got if g.resolve() not in files)
            if plan is None:
                break
    if plan is None:
        scope = "tree"
        files = set(p.resolve() for p in tree_sources(repo))
        shells = [Path(script)] + [repo / "bindings" / "build_host_family.sh"]
    for extra in list(shells) + [repo / "pixi.toml", repo / "pixi.lock"]:
        if Path(extra).is_file():
            files.add(Path(extra).resolve())
    h = hashlib.sha256()
    rels = []
    for f in sorted(files):
        rel = os.path.relpath(f, repo)
        rels.append(rel)
        dig = sha256_bytes(pixi_toml_build_part(f.read_text(errors="replace")).encode()) \
            if rel == "pixi.toml" else sha256_file(f)
        h.update(rel.encode() + b"\0" + dig.encode() + b"\n")
    return dict(scope=scope, digest=h.hexdigest(), files=len(rels)), rels


def pixi_toml_build_part(text):
    """pixi.toml without its task tables and comment lines. A task is a
    command line, never a build input, and on 2026-09-14/15 a new `check-*`
    task was the only edit that invalidated all 29 closures between two
    records. Dependencies, channels, environments and the workspace stay in;
    the solved toolchain is keyed separately through pixi.lock."""
    out, in_tasks, in_string = [], False, False
    for line in text.splitlines():
        s = line.strip()
        if in_tasks and (s.count('"""') + s.count("'''")) % 2 == 1:
            in_string = not in_string
            continue
        if in_string:
            continue
        if s.startswith("[") and not s.startswith("[["):
            in_tasks = s.rstrip().endswith("tasks]")
            if in_tasks:
                continue
        if in_tasks or not s or s.startswith("#"):
            continue
        out.append(line.rstrip())
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
# key rule closure-v2 (2026-11-04): precise closures, only byte-relevant env
# ---------------------------------------------------------------------------
#
# WHY. Under the first rule (now "v1": fields without `key_rule`, still
# verifiable through source_digest/local_inputs/build_env, which are kept
# unchanged) every merged group of every release missed. Measured from the
# keys/<key>.json the releases kept (docs/build/COMPILE_CLOSURES.md):
#   * bindings/merged_build.sh compiles a generated root ("$root"), which
#     script_plan cannot read literally, so EVERY merged key hashed all ~3,950
#     .mojo files of the tree: any edit anywhere rebuilt all 13 macOS groups and
#     all 9 groups of every Linux leg;
#   * MOJOLEARN_SLOT_TOKEN, the tools/mac_slot.py lease token, is random per
#     run and was keyed: two macOS builds of one commit (0.8.65, 07:50Z and
#     09:07Z) missed 13 of 13;
#   * MOJOLEARN_MERGED_KEY hashed the whole manifest and tools/merged_bindings.py,
#     so a one-group edit or a new CLI flag in the generator changed every key;
#   * leg-level variables no bindings/ script reads (MOJOLEARN_CUDA_CODE_FORMAT,
#     MOJOLEARN_BUILD_TIERS) split the vendor-neutral host groups by leg.
# THE RULE. A key is a pure function of what the compile reads:
#   * source: the Mojo import closure from the build's actual roots (for a
#     merged build, bindings/<member>.mojo of each member of the group(s) it
#     compiles, -I . -I bindings), plus pixi.toml (build part) and pixi.lock.
#     An import that names this tree and resolves to nothing FAILS the key
#     (UnresolvedImport names the file and the import); there is no whole-tree
#     fallback. A build-time generated source (GENERATED_SOURCES) is keyed by
#     its tracked generator inputs, never by the generated bytes, so the key is
#     the same before and after the first generator run;
#   * merged: per group, the directory, stem, flags class, members and the
#     sha256 of the exact root text tools/merged_bindings.py writes;
#   * inputs: the shell and Python files the build executes (followed from the
#     script, comment-only edits of shell files ignored); the generator
#     directories are keyed by the source rule, tools/merged_bindings.py by
#     `merged`;
#   * build_env: MOJO_*/MODULAR_* variables (the toolchain reads them) and the
#     MOJOLEARN_* variables a keyed input READS (named on a non-comment line,
#     including dynamic names such as MOJOLEARN_${FAMILY}_HOST_OUTDIR), minus
#     NON_BUILD_ENV/NON_BUILD_ENV_V2/NON_BUILD_PREFIXES;
#   * device_arch is not keyed for a merged GPU build: its accelerator is the
#     explicit MOJOLEARN_GPU_ARCHS target (Linux) or the pinned Metal flags
#     (macOS), never the box's GPU; host builds keep it;
#   * everything else as in v1 (toolchain, numeric mode, archs, column, image,
#     OS, repo path, args).
# A sabotage or fault-injection variable keeps the build a negative control
# even when no script reads it (`sabotage_env`).
KEY_RULE = "closure-v2"
NON_BUILD_ENV_V2 = (
    # tools/mac_slot.py lease token: random per run (0.8.65: 13/13 misses on a rerun).
    "MOJOLEARN_SLOT_TOKEN",
    # Superseded by the `merged` field (exact members and root text per group).
    "MOJOLEARN_MERGED_KEY",
    # The interpreter that runs the build glue (tools/merged_bindings.py root/index).
    "MOJOLEARN_MERGED_PYTHON",
    # Credential path; packager routing (per-binding reuse, merged vs per-binding layout).
    "MOJOLEARN_RUNPOD_KEY_FILE", "MOJOLEARN_REUSE_DIR", "MOJOLEARN_REUSE_PLAN",
    "MOJOLEARN_MERGED_BINDINGS",
    # Leg-level packaging switches applied after the compile (build_sets.sh PTX/cubin passes).
    "MOJOLEARN_CUDA_CODE_FORMAT", "MOJOLEARN_BUILD_TIERS")
#: Keyed whether or not a script names them: which pixi environment runs the compiler.
ALWAYS_KEYED_ENV = ("MOJOLEARN_BUILD_PIXI_ENV",)
#: Build-time generated sources -> (generator directory, tracked pin). The same
#: table as tools/generated_sources.py GENERATED/GENERATOR_DIRS (a test holds
#: them equal); repeated here so this file stays stdlib-only on a box overlay.
GENERATED_SOURCES = {"tokenizer/impl/unicode_table_generated.mojo":
                     ("tokenizer/tools", "tokenizer/impl/unicode_class.mojo")}
_DIR_REF_RE = re.compile(r"\)\"?/([\w.-]+\.(?:sh|py))\b")
_VAR_RE = re.compile(r"MOJOLEARN_(?:[A-Za-z0-9_]|\$\{?[A-Za-z_]\w*\}?|\$\([^)]*\))*")
_GENERATOR_SKIP = (".pyc", ".pyo", ".swp", ".swo", ".tmp", ".orig", ".rej", "~")


class UnkeyableBuild(Exception):
    """A build whose inputs this file cannot name exactly. Never widened."""


class UnresolvedImport(UnkeyableBuild):
    def __init__(self, importer, module):
        self.importer, self.module = importer, module
        super().__init__("unresolved import %r in %s: name its file, or add a toolchain package "
                         "to EXTERNAL_TOP" % (module, importer))


def _merged_bindings():
    try:
        import merged_bindings
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import merged_bindings
    return merged_bindings


def build_platform():
    return "darwin" if sys.platform == "darwin" else "linux"


def merged_rows(repo, args, environ, plat=None):
    """(directory, [(stem, flags_class, [members])]) that
    `bindings/merged_build.sh <args>` compiles, read from REPO's manifest the way
    the script reads it (tools/merged_bindings.py plan)."""
    kind, group, layout = "gpu", None, "release"
    args = list(args)
    i = 0
    while i < len(args):
        a = args[i]
        if a in ("gpu", "host"):
            kind = a
        elif a in ("--group", "--outdir", "--layout", "--rootdir"):
            if i + 1 >= len(args):
                raise UnkeyableBuild("merged_build.sh: %s needs a value" % a)
            i += 1
            if a == "--group":
                group = args[i]
            elif a == "--layout":
                layout = args[i]
        else:
            raise UnkeyableBuild("merged_build.sh: unknown argument %r" % a)
        i += 1
    plat = plat or build_platform()
    mode = environ.get("MOJOLEARN_NUMERIC_MODE") or "identical"
    directory = "host" if kind == "host" else mode
    byte_lm = environ.get("MOJOLEARN_PACKAGE_BYTE_LM") or ("1" if plat == "darwin" else "0")
    try:
        rows = _merged_bindings().plan(directory, plat, byte_lm == "1", layout, repo=repo)
    except (KeyError, ValueError) as exc:
        raise UnkeyableBuild("merged_build.sh: no %s plan: %s" % (directory, exc))
    if group is not None:
        rows = [r for r in rows if r[0] == group]
        if not rows:
            raise UnkeyableBuild("merged_build.sh: no group %r in the %s plan" % (group, directory))
    return directory, rows


def _rel(path, repo):
    return os.path.relpath(path, repo)


def _generated_targets(module, names, importer, roots, repo):
    """GENERATED_SOURCES rels this import could name, present on disk or not."""
    if module.startswith("."):
        dots = len(module) - len(module.lstrip("."))
        base = importer.parent
        for _ in range(dots - 1):
            base = base.parent
        search, parts = [base], [p for p in module.lstrip(".").split(".") if p]
    else:
        search, parts = list(roots) + [importer.parent], module.split(".")
    tail = [os.path.join(*parts[:-1], parts[-1] + ".mojo")] if parts else []
    tail += [os.path.join(*parts, n + ".mojo") for n in names if n != "*"]
    out = set()
    top = str(repo)
    for root in search:
        for t in tail:
            r = os.path.relpath(os.path.normpath(os.path.join(str(root), t)), top)
            if r in GENERATED_SOURCES:
                out.add(r)
    return out


#: (path, mtime_ns, size, include roots) -> (dependencies, generated, unresolved
#: module or None): one parse per file per process, however many keys read it.
_NODES = {}


def _node(f, rel, incs, repo):
    st = f.stat()
    memo = (str(f), st.st_mtime_ns, st.st_size, tuple(str(i) for i in incs))
    hit = _NODES.get(memo)
    if hit is not None:
        return hit
    deps, generated, unresolved = [], set(), None
    for module, names in parse_imports(f.read_text(errors="replace")):
        gen = _generated_targets(module, names, f, incs, repo)
        generated |= gen
        got = resolve(module, names, f, incs)
        if got is None:
            if not gen and unresolved is None:
                unresolved = module
            continue
        for g in got:
            g = g.resolve()
            r = _rel(g, repo)
            if r in GENERATED_SOURCES:
                generated.add(r)
            else:
                deps.append((g, r))
    _NODES[memo] = out = (deps, frozenset(generated), unresolved)
    return out


def import_closure(repo, roots, incs):
    """({rel: sha256} of every Mojo source the roots reach, {generated rels}).
    Raises UnresolvedImport for an import of this tree that names no file."""
    repo = Path(repo).resolve()
    files, generated = {}, set()
    todo = [(Path(r).resolve(), None) for r in roots]
    while todo:
        f, rel = todo.pop()
        rel = rel or _rel(f, repo)
        if rel in files or rel in GENERATED_SOURCES:
            continue
        if rel.startswith(".."):
            raise UnkeyableBuild("build source outside the tree: %s" % f)
        if not f.is_file():
            raise UnkeyableBuild("missing build source %s" % rel)
        files[rel] = sha256_file(f)
        deps, gen, unresolved = _node(f, rel, incs, repo)
        if unresolved is not None:
            raise UnresolvedImport(rel, unresolved)
        generated |= gen
        todo.extend((g, r) for g, r in deps if r not in files)
    return files, generated


def generator_inputs(repo, generated):
    """{rel: sha256} of the tracked generator inputs of each generated source."""
    repo = Path(repo).resolve()
    out = {}
    for rel in sorted(generated):
        directory, pin = GENERATED_SOURCES[rel]
        for p in sorted((repo / directory).rglob("*")):
            if (p.is_file() and "__pycache__" not in p.parts and not p.name.startswith(".")
                    and not p.name.endswith(_GENERATOR_SKIP)):
                out[_rel(p, repo)] = sha256_file(p)
        if (repo / pin).is_file():
            out[pin] = sha256_file(repo / pin)
    return out


def executable_text(text):
    """A shell file's executable lines: comments and blank lines dropped (the
    rule of tools/development_build.py recipe_text)."""
    lines = (line.rstrip() for line in text.splitlines())
    return "\n".join(line for line in lines if line.strip() and not line.lstrip().startswith("#")) + "\n"


def executed_inputs(repo, script, merged=False):
    """({rel: sha256}, [texts]) of the shell and Python files the build runs,
    followed from SCRIPT through every shell file a non-comment line names
    (including `"$(dirname -- "$0")/build_defines.sh"`). Shell files hash their
    executable lines. Generator directories (GENERATED_SOURCES) are keyed by
    the source rule; a merged build keys tools/merged_bindings.py's output
    (`merged`) instead of the generator's own bytes. TEXTS are every executed
    file's text, for the read-set of build_env."""
    repo = Path(repo).resolve()
    gen_dirs = {repo / d for d, _ in GENERATED_SOURCES.values()}
    skip = {repo / "tools" / "merged_bindings.py"} if merged else set()
    seen, todo, files, texts = set(), [Path(script).resolve()], {}, []
    while todo:
        f = todo.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        text = f.read_text(errors="replace")
        texts.append(text)
        if f in skip or any(d in f.parents for d in gen_dirs):
            continue
        rel = _rel(f, repo)
        files[rel] = sha256_bytes(executable_text(text).encode()) if f.suffix == ".sh" else sha256_file(f)
        if f.suffix != ".sh":
            continue
        for line in text.replace("\\\n", " ").splitlines():
            if line.lstrip().startswith("#"):
                continue
            refs = [(r, (repo, f.parent)) for r in _REF_RE.findall(line)]
            refs += [(r, (f.parent,)) for r in _DIR_REF_RE.findall(line)]
            for ref, bases in refs:
                for base in bases:
                    cand = (base / ref)
                    if cand.is_file() and cand.resolve().is_relative_to(repo):
                        todo.append(cand.resolve())
                        break
    return files, texts


def read_variables(texts):
    """(names, patterns) of the MOJOLEARN_* variables the texts name outside
    comments; a name built at run time becomes a pattern."""
    names, patterns = set(), set()
    for text in texts:
        for line in text.splitlines():
            if line.lstrip().startswith("#"):
                continue  # a full-line comment; a trailing one only keys more
            for tok in _VAR_RE.findall(line):
                if "$" in tok:
                    pat = re.sub(r"\\\$(\\\{)?[A-Za-z_]\w*(\\\})?|\\\$\\\([^)]*\\\)", "[A-Za-z0-9_]*", re.escape(tok))
                    patterns.add(pat)
                else:
                    names.add(tok)
    return names, sorted(patterns)


def keyed_env(environ, readers):
    """The build_env of a closure-v2 key (see KEY_RULE above)."""
    names, patterns = readers
    compiled = [re.compile(p) for p in patterns]
    env = {}
    for k, v in environ.items():
        if not k.startswith(KEY_ENV_PREFIXES) or k in NON_BUILD_ENV or k in NON_BUILD_ENV_V2 \
                or k.startswith(NON_BUILD_PREFIXES):
            continue
        if k.startswith("MOJOLEARN_") and k not in ALWAYS_KEYED_ENV and k not in names \
                and not any(p.fullmatch(k) for p in compiled):
            continue
        env[k] = v
    return env


def source_closure(repo, script, args, environ, plat=None):
    """Everything a closure-v2 key reads from the tree, as a dict:
    source (scope/digest/files), merged (or None), inputs (digest/files),
    readers (for keyed_env), closure ({rel: sha256}, not a key field)."""
    repo = Path(repo).resolve()
    script = Path(script)
    if not script.is_absolute():
        script = (repo / script) if (repo / script).is_file() else script.resolve()
    merged = None
    if script.name == "merged_build.sh":
        directory, rows = merged_rows(repo, args, environ, plat)
        mb = _merged_bindings()
        roots = [repo / "bindings" / (m + ".mojo") for _, _, mem in rows for m in mem]
        incs = [repo, repo / "bindings"]
        merged = [dict(dir=directory, stem=stem, flags_class=cls, members=list(mem),
                       root_sha256=sha256_bytes(mb.root_text(stem, mem, repo=repo).encode()))
                  for stem, cls, mem in rows]
    else:
        plan = script_plan(repo, script, args)
        if plan is None:
            raise UnkeyableBuild("cannot read the compile line of %s" % _rel(script, repo))
        roots, incs, _ = plan
    global _DIR_NAMES
    listings = _DIR_NAMES
    if listings is None:
        _DIR_NAMES = {}  # the tree does not change while one key is computed
    try:
        closure, generated = import_closure(repo, roots, incs)
    finally:
        _DIR_NAMES = listings
    closure.update(generator_inputs(repo, generated))
    for extra in ("pixi.toml", "pixi.lock"):
        p = repo / extra
        if p.is_file():
            closure[extra] = (sha256_bytes(pixi_toml_build_part(p.read_text(errors="replace")).encode())
                              if extra == "pixi.toml" else sha256_file(p))
    h = hashlib.sha256()
    for rel in sorted(closure):
        h.update(rel.encode() + b"\0" + closure[rel].encode() + b"\n")
    inputs, texts = executed_inputs(repo, script, merged=merged is not None)
    hi = hashlib.sha256()
    for rel in sorted(inputs):
        hi.update(rel.encode() + b"\0" + inputs[rel].encode() + b"\n")
    if merged is not None:
        texts.append((repo / "tools" / "merged_bindings.py").read_text(errors="replace")
                     if (repo / "tools" / "merged_bindings.py").is_file() else "")
    return dict(source=dict(scope="closure", digest=h.hexdigest(), files=len(closure)),
                merged=merged, inputs=dict(digest=hi.hexdigest(), files=sorted(inputs)),
                readers=read_variables(texts), closure=closure)


_REVISION_TREES = {}


def revision_tree(root, revision):
    """A temporary directory holding the build inputs of one committed revision
    (every .mojo file, bindings/, tokenizer/, tools/, pixi.toml, pixi.lock),
    extracted once per process from `git archive`; never a checkout. Verifiers
    of closure-v2 keys recompute source/inputs/merged from it."""
    memo = (str(Path(root).resolve()), revision)
    if memo in _REVISION_TREES:
        return _REVISION_TREES[memo]
    dest = Path(tempfile.mkdtemp(prefix="bincache-revision-"))
    proc = subprocess.Popen(["git", "-C", memo[0], "archive", revision], stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
        for member in archive:
            name = member.name
            if member.isfile() and ".." not in name.split("/") and (
                    name.endswith(".mojo") or name.startswith(("bindings/", "tokenizer/", "tools/"))
                    or name in ("pixi.toml", "pixi.lock")):
                archive.extract(member, dest, filter="data")
    err = proc.stderr.read().decode(errors="replace")
    proc.stdout.close()
    proc.stderr.close()
    if proc.wait() != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise UnkeyableBuild("git archive %s failed: %s" % (revision, err.strip()))
    _REVISION_TREES[memo] = dest
    atexit.register(shutil.rmtree, dest, True)
    return dest


def revision_closure(root, revision, script, args, environ, plat=None):
    """source_closure at a committed revision: dict(source, inputs, merged,
    readers) for a closure-v2 verifier to compare with recorded fields."""
    tree = revision_tree(root, revision)
    sc = source_closure(tree, tree / script, args, environ, plat)
    return dict(source=sc["source"], inputs=sc["inputs"], merged=sc["merged"], readers=sc["readers"])


def merged_fields_reproduce(fields, root, revision, environ, outputs, shell="bash", plat="linux"):
    """True when the closure-v2 FIELDS of a declared merged build are exactly
    what this file computes at REVISION when the build ran under ENVIRON (the
    variables it was given, before keyed_env keeps the ones its inputs read).
    For verifiers of retained release compilations."""
    if fields.get("key_rule") != KEY_RULE:
        return False
    want = revision_closure(root, revision, fields["script"], fields.get("args") or [], environ, plat)
    return (fields.get("source") == want["source"] and fields.get("inputs") == want["inputs"]
            and fields.get("merged") == want["merged"]
            and fields.get("build_env") == keyed_env(environ, want["readers"])
            and fields.get("declared") == dict(outputs=list(outputs), shell=shell))


# ---------------------------------------------------------------------------
# the rest of the key
# ---------------------------------------------------------------------------

def toolchain(repo):
    lock = Path(repo) / "pixi.lock"
    mach = platform.machine()
    plat = {"x86_64": "linux-64", "aarch64": "linux-aarch64", "arm64": "osx-arm64"}.get(mach, mach)
    if sys.platform == "darwin":
        plat = "osx-arm64"
    pkgs = []
    if lock.is_file():
        text = lock.read_text(errors="replace")
        pkgs = sorted(set(re.findall(
            r"conda\.modular\.com/max/%s/((?:mojo|mojo-compiler|max|max-core)-[^/\s]+)\.conda" % re.escape(plat),
            text)))
        lock_sha = sha256_file(lock)
    else:
        lock_sha = ""
    return dict(platform=plat, packages=pkgs, pixi_lock_sha256=lock_sha)


def installed_toolchain(repo):
    meta = Path(repo) / ".pixi" / "envs" / "default" / "conda-meta"
    if not meta.is_dir():
        return None
    return sorted(p.name[:-5] for p in meta.glob("*.json")
                  if re.match(r"(mojo|mojo-compiler|max|max-core)-\d", p.name))


def first_line(cmd):
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30)
        return r.stdout.decode(errors="replace").splitlines()[0].strip() if r.stdout else ""
    except Exception:
        return ""


def host_cpu(machine):
    """The host CPU, keyed ONLY where `mojo build` is left to target it.

    EVERY bindings/build_*.sh runs the same `case "$(uname -m)"`: on Linux
    x86-64 it pins `--target-cpu ${MOJOLEARN_LINUX_CPU:-x86-64-v3}`, and on
    ANYTHING ELSE it sets TARGET_FLAGS to the empty string -- the comment
    beside it reads "linux arm: host cpu + its GPU". So on aarch64 the
    compiler targets WHATEVER CORE RAN IT, and two boxes that this key
    otherwise cannot tell apart (same image, same glibc, same
    machine=aarch64, same GPU) compile different instruction sets: a
    Neoverse-V2 GH200 emits things a Neoverse-N1 Altra traps on. Serving one
    box's archive to the other is the 2026-08-30 incident again -- a wheel
    built at the host default shipped AVX-512 and died with SIGILL on a Zen 3
    EPYC inside kmeans_fit -- on the architecture where the fix for it does
    not apply. So the host CPU is an input there and it is keyed there.

    It is NOT keyed on x86-64, where the pin makes it irrelevant and where
    keying it would cost every hit for nothing: a rented pod's exact EPYC or
    Xeon model changes from rental to rental, which is precisely the drift
    the pin was added to stop mattering. (os_fields() drops it on macOS for
    the same reason: `--target-cpu apple-m1` is pinned there too.)

    A machine with no /proc/cpuinfo to read returns "unknown" rather than "",
    so an unreadable host partitions the cache instead of silently joining
    every other unreadable host.
    """
    if machine in ("x86_64", "amd64"):
        return ""
    try:
        lines = Path("/proc/cpuinfo").read_text(errors="replace").splitlines()
    except OSError:
        return "unknown"
    want = ("CPU implementer", "CPU architecture", "CPU variant", "CPU part",
            "model name", "cpu model", "Features")
    seen = []
    for line in lines:
        k, sep, v = line.partition(":")
        if sep and k.strip() in want and (k.strip(), v.strip()) not in seen:
            seen.append((k.strip(), v.strip()))
    return "; ".join("%s=%s" % kv for kv in seen) or "unknown"


def os_fields():
    osr = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                osr[k] = v.strip().strip('"')
    except OSError:
        pass
    try:
        glibc = os.confstr("CS_GNU_LIBC_VERSION") or ""
    except (ValueError, OSError, AttributeError):
        glibc = ""
    machine = platform.machine()
    # macOS pins `--target-cpu apple-m1` in every bindings/build_*.sh, for the
    # same reason Linux x86 pins x86-64-v3, so its host CPU is not an input
    # either -- and nothing on a Mac stages a URL map, so bincache.py is a
    # pass-through there in any case.
    cpu = "" if sys.platform == "darwin" else host_cpu(machine)
    out = dict(id=osr.get("ID", sys.platform), version_id=osr.get("VERSION_ID", platform.release()),
               glibc=glibc, machine=machine,
               ld=first_line(["ld", "--version"]), cc=first_line(["cc", "--version"]))
    # ABSENT, not empty, where the build pins --target-cpu -- the same shape as
    # `variant` below, and for the same reason: a key that was correct before
    # this field existed stays correct, so the x86-64 entries already in
    # bincache/v1 (tools/runpod_cpu_leg.sh has been filling them since
    # 2026-09-15) are not all invalidated by adding a field that says nothing
    # about them.
    if cpu:
        out["cpu"] = cpu
    return out


def device_arch():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                           stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60)
        cc = r.stdout.decode().strip().splitlines()[0].strip() if r.returncode == 0 and r.stdout.strip() else ""
        if cc:
            return {"9.0": "sm_90a", "12.0": "sm_120a"}.get(cc, "sm_" + cc.replace(".", ""))
    except Exception:
        pass
    try:
        r = subprocess.run(["rocminfo"], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, timeout=60)
        m = re.search(r"gfx[0-9a-z]+", r.stdout.decode(errors="replace"))
        if m:
            return m.group(0)
    except Exception:
        pass
    return "none"


def build_env(environ):
    env = {}
    for k, v in environ.items():
        if not k.startswith(KEY_ENV_PREFIXES) or k in NON_BUILD_ENV or k.startswith(NON_BUILD_PREFIXES):
            continue
        env[k] = v
    return env


def key_fields(repo, script, args, environ, image, dev_arch=None, os_info=None, force_tree=False,
               rule=None, plat=None):
    """The key fields of one build. rule None (the default) is closure-v2
    (KEY_RULE); rule "v1" -- or force_tree, the explicit whole-tree opt-in
    MOJOLEARN_BINCACHE_SOURCE=tree -- is the first rule, kept so archives and
    receipts keyed under it stay verifiable."""
    if rule in (None, KEY_RULE) and not force_tree:
        return key_fields_closure(repo, script, args, environ, image, dev_arch, os_info, plat)[0]
    if rule not in ("v1", None, KEY_RULE):
        raise ValueError("unknown bincache key rule %r" % (rule,))
    return key_fields_v1(repo, script, args, environ, image, dev_arch, os_info, force_tree)


def key_fields_closure(repo, script, args, environ, image, dev_arch=None, os_info=None, plat=None):
    """(fields, {rel: sha256} closure) under closure-v2. Raises UnkeyableBuild."""
    repo = Path(repo).resolve()
    sc = source_closure(repo, script, args, environ, plat)
    fields = dict(
        schema=SCHEMA,
        key_rule=KEY_RULE,
        script=os.path.relpath(Path(script).resolve(), repo),
        args=list(args),
        source=sc["source"],
        inputs=sc["inputs"],
        toolchain=toolchain(repo),
        numeric_mode=environ.get("MOJOLEARN_NUMERIC_MODE") or "identical",
        gpu_archs=environ.get("MOJOLEARN_GPU_ARCHS", ""),
        target_column=environ.get("MOJOLEARN_TARGET_COLUMN", ""),
        build_env=keyed_env(environ, sc["readers"]),
        image=image,
        os=os_fields() if os_info is None else os_info,
        repo_path=str(repo),
    )
    if sc["merged"] is not None:
        fields["merged"] = sc["merged"]
    # A merged GPU build names its accelerator (MOJOLEARN_GPU_ARCHS on Linux,
    # the pinned Metal flags on macOS); the box it runs on is not an input.
    if not (sc["merged"] is not None and any(r["dir"] != "host" for r in sc["merged"])):
        fields["device_arch"] = device_arch() if dev_arch is None else dev_arch
    sab = {k: v for k, v in build_env(environ).items() if SABOTAGE_RE.search(k) or SABOTAGE_RE.search(v)}
    if sab:
        fields["sabotage_env"] = sab
    if is_sabotage(fields):
        fields["variant"] = "sabotage"
    return fields, sc["closure"]


def key_fields_v1(repo, script, args, environ, image, dev_arch=None, os_info=None, force_tree=False):
    repo = Path(repo).resolve()
    src, _ = source_digest(repo, script, args, force_tree=force_tree)
    fields = dict(
        schema=SCHEMA,
        script=os.path.relpath(Path(script).resolve(), repo),
        args=list(args),
        source=src,
        toolchain=toolchain(repo),
        numeric_mode=environ.get("MOJOLEARN_NUMERIC_MODE") or "identical",
        gpu_archs=environ.get("MOJOLEARN_GPU_ARCHS", ""),
        target_column=environ.get("MOJOLEARN_TARGET_COLUMN", ""),
        build_env=build_env(environ),
        device_arch=device_arch() if dev_arch is None else dev_arch,
        image=image,
        os=os_fields() if os_info is None else os_info,
        repo_path=str(repo),
    )
    if is_sabotage(fields):
        # Production keys carry no variant field, so they are unchanged; a
        # sabotage key can never collide with one.
        fields["variant"] = "sabotage"
    return fields


def is_sabotage(fields):
    for k, v in (list(fields.get("build_env", {}).items()) + list(fields.get("sabotage_env", {}).items())
                 + [("args", " ".join(fields.get("args", [])))]):
        if SABOTAGE_RE.search(k) or SABOTAGE_RE.search(str(v)):
            return True
    return False


def refusal(fields, environ):
    """Why this build must never touch the cache, or ''."""
    for k, v in (list(fields["build_env"].items()) + list(fields.get("sabotage_env", {}).items())
                 + [("args", " ".join(fields["args"]))]):
        if SABOTAGE_RE.search(k) or SABOTAGE_RE.search(v):
            return "sabotage:" + k
    return outdir_refusal(fields, environ)


def outdir_refusal(fields, environ):
    for k, v in environ.items():
        if k.endswith("_OUTDIR") and k.startswith("MOJOLEARN_") and v:
            p = Path(v)
            if p.is_absolute() and not str(p.resolve()).startswith(fields["repo_path"] + os.sep):
                return "outdir-outside-tree:" + k
    return ""


# ---------------------------------------------------------------------------
# archives
# ---------------------------------------------------------------------------

def pack(dest, key, fields, repo, outputs, extra):
    files = []
    for rel in sorted(outputs):
        p = Path(repo) / rel
        files.append(dict(path=rel, sha256=sha256_file(p), size=p.stat().st_size,
                          mode=p.stat().st_mode & 0o777))
    manifest = dict(schema=SCHEMA, key=key, fields=fields, files=files, **extra)
    raw = json.dumps(manifest, indent=1, sort_keys=True).encode()
    with tarfile.open(dest, "w:gz", compresslevel=1) as tf:
        ti = tarfile.TarInfo("manifest.json")
        ti.size = len(raw)
        tf.addfile(ti, io.BytesIO(raw))
        for f in files:
            tf.add(str(Path(repo) / f["path"]), arcname="files/" + f["path"], recursive=False)
    return manifest


def safe_rel(rel):
    if not rel or rel.startswith("/") or "\\" in rel or os.path.normpath(rel) != rel or rel.split("/")[0] == "..":
        return False
    return ".." not in rel.split("/")


def verify_archive(path, key, fields):
    """(manifest, {rel: bytes}) or raise Reject. Every member is accounted
    for, every file matches its sha256 and size, and the manifest's key and
    fields are the ones this box computed."""
    try:
        tf = tarfile.open(path, "r:gz")
    except (tarfile.TarError, OSError, EOFError) as exc:
        raise Reject("unreadable archive: %s" % type(exc).__name__)
    with tf:
        try:
            members = tf.getmembers()
        except (tarfile.TarError, OSError, EOFError, Exception) as exc:
            raise Reject("unreadable archive: %s" % type(exc).__name__)
        by_name = {}
        for m in members:
            if m.name in by_name:
                raise Reject("duplicate member " + m.name)
            if not m.isfile():
                raise Reject("non-file member " + m.name)
            by_name[m.name] = m
        if "manifest.json" not in by_name:
            raise Reject("no manifest")
        try:
            manifest = json.loads(tf.extractfile(by_name["manifest.json"]).read())
        except Exception:
            raise Reject("manifest is not JSON")
        if manifest.get("schema") != SCHEMA:
            raise Reject("schema mismatch")
        if manifest.get("key") != key or key_of(manifest.get("fields", {})) != key:
            raise Reject("key mismatch")
        if canonical(manifest.get("fields")) != canonical(fields):
            raise Reject("fields mismatch")
        listed = manifest.get("files") or []
        if not listed:
            raise Reject("manifest lists no files")
        want = {"manifest.json"} | {"files/" + f.get("path", "") for f in listed}
        if set(by_name) != want:
            raise Reject("members differ from the manifest")
        blobs = {}
        for f in listed:
            rel = f.get("path", "")
            if not safe_rel(rel):
                raise Reject("unsafe path " + rel)
            data = tf.extractfile(by_name["files/" + rel]).read()
            if len(data) != f.get("size") or sha256_bytes(data) != f.get("sha256"):
                raise Reject("sha256 mismatch " + rel)
            blobs[rel] = data
    return manifest, blobs


# ---------------------------------------------------------------------------
# ONE KEY, ONE BYTE CONTENT (2026-11-05, finding 3 of the 0.8.66 release
# state). Each GitHub release leg recompiled the vendor-neutral host groups
# and uploaded them under the SAME key; the hip leg's
# _mojolearn_native_series_host.so came out with other bytes, and the last
# promote silently replaced the first. A key now holds exactly one content:
# a write to a key that already holds an archive compares the archives' FILES
# (path -> sha256 from each manifest; the archive bytes themselves differ by
# build time and builder) and is a no-op when they are equal and a loud
# refusal, naming the key and both sha256 of every differing file, when they
# are not. The existing archive is kept.
# ---------------------------------------------------------------------------
def archive_files(path):
    """{path: sha256} of an archive's manifest, or raise Reject. Reads only the
    manifest (the content claim); verify_archive checks the bytes on placement."""
    try:
        with tarfile.open(path, "r:gz") as tf:
            try:
                m = tf.getmember("manifest.json")
            except KeyError:
                raise Reject("no manifest")
            manifest = json.loads(tf.extractfile(m).read())
    except Reject:
        raise
    except Exception as exc:
        raise Reject("unreadable archive: %s" % type(exc).__name__)
    files = manifest.get("files") if isinstance(manifest, dict) else None
    if not files or not all(isinstance(f, dict) and f.get("path") and f.get("sha256") for f in files):
        raise Reject("manifest lists no files")
    return {f["path"]: f["sha256"] for f in files}


def content_conflict(existing, new):
    """[(path, existing sha256, new sha256)] for every file the two {path:
    sha256} maps disagree on ("-" for a file one side lacks); [] when equal."""
    return [(p, existing.get(p, "-"), new.get(p, "-"))
            for p in sorted(set(existing) | set(new)) if existing.get(p) != new.get(p)]


def conflict_text(key, diffs, where=""):
    """The loud refusal line: the key and both sha256 of each differing file."""
    return ("BINCACHE CONFLICT key=%s%s holds different bytes; upload refused, existing kept: %s"
            % (key, (" at " + where) if where else "",
               "; ".join("%s existing sha256=%s upload sha256=%s" % d for d in diffs)))


def keep_or_refuse(existing_path, new_path, key, where=""):
    """Before writing NEW_PATH's archive over EXISTING_PATH (same key):
    ("write", "") when nothing usable is there (absent, or unreadable: it held
    no content a hit could place), ("same", "") when it holds the same files
    (the caller may skip the write or refresh the manifest), ("conflict",
    line) when it holds other bytes: the caller must not write."""
    if not os.path.isfile(existing_path):
        return "write", ""
    try:
        old = archive_files(existing_path)
    except Reject as exc:
        print("BINCACHE replacing unreadable archive for key %s (%s)" % (key, exc), file=sys.stderr)
        return "write", ""
    diffs = content_conflict(old, archive_files(new_path))
    if not diffs:
        return "same", ""
    line = conflict_text(key, diffs, where)
    print(line, file=sys.stderr)
    return "conflict", line


def vendor_neutral(fields):
    """True for a build keyed without any GPU input: a merged host group
    (bindings/merged_build.sh host --group G; every merged row in dir host) or
    a per-binding *_host.sh script. Every Linux release leg compiles these
    under one key (tools/release_github_build.py HOST_GROUPS_OWNER)."""
    merged = fields.get("merged")
    if merged:
        return all(isinstance(r, dict) and r.get("dir") == "host" for r in merged)
    script = os.path.basename(str(fields.get("script", "")))
    if script == "merged_build.sh":
        return list(fields.get("args") or [])[:1] == ["host"]
    return script.endswith("_host.sh")


# ---------------------------------------------------------------------------
# transport (https presigned URLs on a box; file:// in the tests)
# ---------------------------------------------------------------------------

def http_get(url, dest, tries=4):
    if url.startswith("file://"):
        src = urllib.parse.urlparse(url).path
        if not os.path.exists(src):
            return 404
        shutil.copyfile(src, dest)
        return 200
    last = 0
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as r, open(dest, "wb") as fh:
                shutil.copyfileobj(r, fh, 1 << 20)
            return 200
        except urllib.error.HTTPError as exc:
            last = exc.code
            if exc.code in (403, 404):
                return exc.code
        except (urllib.error.URLError, OSError):
            last = -1
        time.sleep(3 * (attempt + 1))
    return last


def http_put(url, src, tries=4, headers=None):
    if url.startswith("file://"):
        dst = urllib.parse.urlparse(url).path
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)
        return 200
    data = Path(src).read_bytes()
    last = 0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, data=data, method="PUT", headers=headers or {})
            with urllib.request.urlopen(req, timeout=300) as r:
                return r.status
        except urllib.error.HTTPError as exc:
            last = exc.code
            if exc.code in (400, 403):
                return exc.code
        except (urllib.error.URLError, OSError):
            last = -1
        time.sleep(3 * (attempt + 1))
    return last


def read_map(path):
    m = dict(header={}, get={}, sget={}, put=[])
    for line in Path(path).read_text().splitlines():
        parts = line.split("\t")
        if len(parts) == 2 and parts[0].startswith("#"):
            m["header"][parts[0][1:]] = parts[1]
        elif len(parts) == 3 and parts[0] in ("get", "sget") and KEY_RE.match(parts[1]):
            m[parts[0]][parts[1]] = parts[2]
        elif len(parts) == 3 and parts[0] == "put" and SEG_RE.match(parts[1]):
            m["put"].append((parts[1], parts[2]))
    return m


# ---------------------------------------------------------------------------
# the box: build through the cache
# ---------------------------------------------------------------------------

def snapshot(repo, environ):
    dirs = [Path(repo) / "python"]
    for k, v in environ.items():
        if k.startswith("MOJOLEARN_") and k.endswith("_OUTDIR") and v:
            dirs.append(Path(repo) / v if not os.path.isabs(v) else Path(v))
    snap = {}
    for d in dirs:
        if not d.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(d):
            dirnames[:] = [x for x in dirnames if x != "__pycache__"]
            for f in filenames:
                if f.endswith(".so"):
                    p = Path(dirpath) / f
                    try:
                        st = p.stat()
                    except OSError:
                        continue
                    snap[str(p.resolve())] = (st.st_ino, st.st_mtime_ns, st.st_size)
    return snap


def run_tee(cmd, env=None):
    """Run cmd with stdout+stderr forwarded to our stdout; return (rc, sha256)."""
    h = hashlib.sha256()
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env)
    out = sys.stdout.buffer
    for chunk in iter(lambda: p.stdout.read1(1 << 16) if hasattr(p.stdout, "read1") else p.stdout.read(1 << 16), b""):
        h.update(chunk)
        out.write(chunk)
        out.flush()
    p.stdout.close()
    return p.wait(), h.hexdigest()


class Record:
    def __init__(self, out_dir):
        self.dir = Path(out_dir) if out_dir else None
        if self.dir is not None:
            try:
                (self.dir / "keys").mkdir(parents=True, exist_ok=True)
            except OSError:
                self.dir = None

    def row(self, script, outcome, key, seconds, files):
        line = "\t".join([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), script, outcome,
                          key or "-", "%.1f" % seconds,
                          " ".join("%s=%s" % (f["path"], f["sha256"]) for f in files) or "-"])
        print("BINCACHE %s %s key=%s seconds=%.1f" % (outcome, script, (key or "-")[:16], seconds))
        if self.dir is not None:
            with open(self.dir / "provenance.tsv", "a") as fh:
                fh.write(line + "\n")

    def key(self, key, fields):
        if self.dir is not None:
            (self.dir / "keys" / (key + ".json")).write_text(json.dumps(fields, indent=1, sort_keys=True))

    def closure(self, key, closure):
        """closures/<key>.json: {rel: sha256} of the key's source closure. Not a
        key field (the digest is); it lets `explain` name the files that changed."""
        if self.dir is not None and closure is not None:
            (self.dir / "closures").mkdir(exist_ok=True)
            (self.dir / "closures" / (key + ".json")).write_text(json.dumps(closure, indent=0, sort_keys=True))

    def placement(self, script, outcome, key, placed, manifest):
        """One JSON line per cache HIT: the bytes placed and the archive's own
        record of where they came from (its builder, its build time and the
        commit it was compiled from), for the release provenance."""
        if self.dir is None:
            return
        with open(self.dir / "placements.jsonl", "a") as fh:
            fh.write(json.dumps(dict(script=script, outcome=outcome, key=key, files=placed,
                                     archive_source_commit=manifest.get("source_commit", ""),
                                     archive_built=manifest.get("built", ""),
                                     archive_build_seconds=manifest.get("build_seconds"),
                                     archive_builder=manifest.get("builder", {})), sort_keys=True) + "\n")

    def upload_row(self, leg, slot, dest, key, sabotage=False):
        with open(self.dir / "uploads.tsv", "a") as fh:
            fh.write("%s\t%s\t%s\t%s\tsabotage=%d\n" % (leg, slot, dest, key, 1 if sabotage else 0))


def claim_slot(map_path, puts):
    claimed = Path(map_path).parent / "claimed"
    claimed.mkdir(exist_ok=True)
    for slot, url in puts:
        try:
            fd = os.open(str(claimed / slot), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
            os.close(fd)
            return slot, url
        except FileExistsError:
            continue
    return None, None


# ---------------------------------------------------------------------------
# the Mac: a LOCAL directory cache for the macOS release build (2026-09-21)
# ---------------------------------------------------------------------------
#
# WHY. packaging/macos/build_release_wheel.sh compiles every extension from
# source on every run: about 45 builds (22 identical GPU bindings, the tree
# bindings in two more tiers, the byte LM and 32 host families), and the
# release workflow ran them one at a time, 30 to 60 minutes. Between two
# releases most bindings have the same import closure, and a rerun of a
# failed workflow has all of them. A binding is a pure function of the key
# below, so an unchanged one is taken from a directory on this Mac instead.
#
# WHAT IS DIFFERENT FROM THE R2 PATH, and why each difference is stricter:
#   * the build DECLARES its outputs (MOJOLEARN_BINCACHE_OUTPUTS, paths under
#     the repository). The R2 path diffs a snapshot of python/, which is
#     right for one build at a time and wrong for the release script, which
#     runs several builds at once in the same tree: build A would archive
#     build B's .so. A declared output that the build did not rewrite is not
#     archived, and the build still counts.
#   * the key carries `local`: every shell or Python file the build script
#     names, followed through the scripts it runs, and every generator under
#     tokenizer/tools/ it reaches (so a script that execs another, or a
#     generated source, cannot reuse a stale binary), the shell, the declared
#     outputs, and the Mac's own toolchain (macOS build, Xcode, the Metal
#     compiler, the active developer directory), because Metal AIR is
#     produced with the system's tools and the R2 key never needed them.
#   * no negative control and no upload slots: a sabotage build is refused
#     outright, as it is on the R2 path.
# Everything else is shared: the source closure, pixi.lock's toolchain,
# every MOJOLEARN_*/MOJO_*/MODULAR_* variable, the numeric mode, the OS and
# the repository path; the archive is the same format and is verified the
# same way (manifest key and fields, every file's sha256 and size) before a
# byte is placed, and a destination that already exists is never replaced.
#
# FAIL CLOSED. Any error reading, verifying or placing an archive means the
# build runs from source; an error writing one means the built binary is used
# and nothing is cached. The release script's own gates (staleness, GPU code
# present, ISA baseline, macOS floor, then verify_wheel.sh fitting every
# family on every interpreter) run on the placed bytes exactly as on built
# ones, and the build log names every hit with its key and sha256.
#
#   MOJOLEARN_BINCACHE_DIR=<dir>        turn it on (off when unset)
#   MOJOLEARN_BINCACHE_OUTPUTS="a b"    what the build writes, repo-relative
#   MOJOLEARN_BINCACHE_SHELL=bash       how to run the script (default sh)
#   MOJOLEARN_BINCACHE_KEEP=<n>         archives kept, newest first (default 400)
LOCAL_KEEP = 400

# THE ONE MAC CACHE (2026-11-07, lane/post-one-mac-cache). On 2026-11-07 the
# Mac compiled the frozen 0.8.66 commit twice: once as the development build
# (tools/development_build.py, development layout, its own key and cache) and
# again inside the release as the macOS wheel (8582 s wall, every one of 15
# merged groups "miss+built-cached" in a release-scoped cache). Andrew's rule
# is one compile per freeze. So the development build on macOS compiles the
# release layout through this cache with this environment
# (tools/development_build.py --layout release, the macOS default), writing
# every group into ONE Mac cache, and tools/release.py step_macos_build seeds
# its release-scoped cache from that same directory and copies what it built
# back into it. A group the development build compiled in the release's source
# checkout then carries the release's closure-v2 key and is placed, not
# compiled. MOJOLEARN_BINCACHE_DIR names another directory.
MAC_CACHE_DEFAULT = Path.home() / "mojolearn-evidence" / "bincache-mac"
#: The compile environment of the macOS release build that reaches a key
#: (build_env): tools/release.py step_macos_build passes both, and
#: tools/mac_slot.py run pins MOJOLEARN_COMPILE_JOBS=1. A compile with another
#: job count is keyed apart (a codegen split is not shown byte-neutral).
MACOS_RELEASE_COMPILE_ENV = {"MOJOLEARN_PACKAGE_BYTE_LM": "1", "MOJOLEARN_COMPILE_JOBS": "1"}


def mac_cache_dir(environ=None):
    """The one Mac cache: MOJOLEARN_BINCACHE_DIR, else MAC_CACHE_DEFAULT."""
    environ = os.environ if environ is None else environ
    return Path(environ.get("MOJOLEARN_BINCACHE_DIR") or MAC_CACHE_DEFAULT).expanduser()


def macos_merged_group(base, directory, stem):
    """(kind, output, environment) of one merged macOS group exactly as
    packaging/macos/build_release_wheel.sh runs it through
    `bincache.py build ./bindings/merged_build.sh <kind> --group <stem>`:
    host groups compile the CPU column with no accelerator target and no
    output directory override; GPU groups take the directory's numeric mode.
    The declared output is a key field (local.outputs), so it is the package
    path the wheel build writes."""
    env = dict(base)
    if directory == "host":
        kind, output = "host", "python/mojolearn/host/%s.so" % stem
        env.pop("MOJOLEARN_GPU_ARCHS", None)
        env.pop("MOJOLEARN_HOST_OUTDIR", None)
        env.update(MOJOLEARN_NUMERIC_MODE="identical", MOJOLEARN_TARGET_COLUMN="cpu")
    else:
        kind = "gpu"
        output = ("python/mojolearn/%s.so" % stem if directory == "fast"
                  else "python/mojolearn/%s/%s.so" % (directory, stem))
        env["MOJOLEARN_NUMERIC_MODE"] = directory
    env.update(MOJOLEARN_BINCACHE_OUTPUTS=output, MOJOLEARN_BINCACHE_SHELL="sh")
    return kind, output, env


def copy_archives(source_dir, dest_dir):
    """Copy every verified archive of SOURCE_DIR that DEST_DIR lacks into
    DEST_DIR (the release's write-back into the one Mac cache). An archive is
    verified against its own manifest before and after the copy; a key DEST_DIR
    already holds with other bytes is a BINCACHE CONFLICT and is never
    replaced. Returns {"copied": n, "present": n, "conflicts": [key],
    "rejected": [key]}. Never raises for one archive."""
    out = dict(copied=0, present=0, conflicts=[], rejected=[])
    source_dir, dest_dir = Path(source_dir), Path(dest_dir)
    if not source_dir.is_dir() or source_dir.resolve() == dest_dir.resolve():
        return out
    dest_dir.mkdir(parents=True, exist_ok=True)
    for arc in sorted(source_dir.glob("*.tar.gz")):  # driver-loop: over cache archives, file copies only
        key = arc.name[:-len(".tar.gz")]
        dst = dest_dir / arc.name
        try:
            manifest = _manifest_of(arc) or {}
            verify_archive(str(arc), key, manifest.get("fields") or {})
            verdict, _line = keep_or_refuse(dst, arc, key, "one Mac cache")
            if verdict == "same":
                out["present"] += 1
                continue
            if verdict == "conflict":
                out["conflicts"].append(key)
                continue
            tmp = dest_dir / (".copy-%d-%s.tar.gz" % (os.getpid(), key))
            shutil.copyfile(arc, tmp)
            verify_archive(str(tmp), key, manifest.get("fields") or {})
            os.replace(tmp, dst)
            out["copied"] += 1
        except (Reject, OSError, ValueError):
            out["rejected"].append(key)
    return out


def darwin_toolchain():
    """The Mac's own compilers, which the Metal half of a binding is built
    with. Empty off macOS."""
    if sys.platform != "darwin":
        return {}
    return dict(
        product=first_line(["sw_vers", "-productVersion"]),
        build=first_line(["sw_vers", "-buildVersion"]),
        xcode=" ".join(first_line(["xcodebuild", "-version"]).split()),
        developer_dir=first_line(["xcode-select", "-p"]),
        metal=first_line(["xcrun", "metal", "--version"]),
        sdk=first_line(["xcrun", "--show-sdk-version"]),
    )


_REF_RE = re.compile(r"(?<![\w./-])((?:[\w.-]+/)*[\w.-]+\.(?:sh|py))\b")


def head_commit(repo):
    """The commit REPO's checkout is at (40 hex), or "" when it is not a git
    checkout. Not a key field: a LOCAL archive carries it as `source_commit`
    so a later placement can name the revision the bytes were compiled from,
    and a verifier recomputes the key's closure at that revision
    (tools/release_requirements_macos_build.py admits a placed group so)."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    head = out.stdout.strip()
    return head if out.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head) else ""


def local_inputs(repo, script):
    """sha256 over every shell or Python file the build script names on a
    line that is not a comment, followed through the shell files it names
    (so `exec build_host_family.sh` pulls in the builder and the generator
    the builder runs), plus the whole directory of any generator under
    tokenizer/tools/, whose output is compiled in. Wider than the R2 key's
    `tools/*.sh` references on purpose: a script, helper or generator named
    by ANY path is keyed, whatever the spelling of the call. Every build
    script under bindings/ is NOT keyed wholesale: they change several times
    a day, and an edit to one says nothing about another's binary."""
    repo = Path(repo).resolve()
    seen, todo, files = set(), [Path(script).resolve()], set()
    shims = repo / "bindings" / "build_host_family.sh"
    while todo:
        f = todo.pop()
        if f in seen or not f.is_file():
            continue
        seen.add(f)
        files.add(f)
        text = f.read_text(errors="replace")
        if f.suffix == ".sh" and "build_host_family.sh" in text:
            todo.append(shims)
        if f.suffix != ".sh":
            continue
        for line in text.replace("\\\n", " ").splitlines():
            if line.lstrip().startswith("#"):
                continue
            for ref in _REF_RE.findall(line):
                for cand in (repo / ref, f.parent / ref):
                    if cand.is_file():
                        cand = cand.resolve()
                        if cand.is_relative_to(repo):
                            todo.append(cand)
                            if (repo / "tokenizer" / "tools") in cand.parents:
                                files.update(p.resolve() for p in (repo / "tokenizer" / "tools").rglob("*")
                                             if p.is_file())
    h = hashlib.sha256()
    rels = sorted(os.path.relpath(p, repo) for p in files)
    for rel in rels:
        h.update(rel.encode() + b"\0" + sha256_file(repo / rel).encode() + b"\n")
    return dict(digest=h.hexdigest(), files=rels)


#: id(fields) -> closure of the newest local_fields call, so cmd_build_local
#: keeps calling local_fields (the retained-admission tests replace it) and
#: still writes the closure sidecar when the real one ran.
_LOCAL_CLOSURES = {}


def local_fields(repo, script, args, environ, outputs, shell):
    fields, closure = build_fields(repo, script, args, environ, "local", outputs, shell, local=True)
    _LOCAL_CLOSURES.clear()
    _LOCAL_CLOSURES[id(fields)] = closure
    return fields


def build_fields(repo, script, args, environ, image, declared=None, shell="sh", local=False,
                 dev_arch=None, os_info=None, plat=None):
    """(fields, closure) of a cache build. closure-v2 unless the explicit
    whole-tree opt-in MOJOLEARN_BINCACHE_SOURCE=tree asks for the v1 rule
    (closure is then None). A LOCAL cache adds `local` (declared outputs, the
    shell, the Mac's own toolchain), a declared R2 build `declared`; under v1
    both also carry local_inputs, under closure-v2 the executed inputs are the
    top-level `inputs` field."""
    v1 = environ.get("MOJOLEARN_BINCACHE_SOURCE") == "tree"
    if v1:
        fields, closure = key_fields(repo, script, args, environ, image, dev_arch, os_info, force_tree=True), None
    else:
        fields, closure = key_fields_closure(repo, script, args, environ, image, dev_arch, os_info, plat)
    if local:
        fields["local"] = dict(outputs=sorted(declared or []), shell=shell, host_toolchain=darwin_toolchain())
        if v1:
            fields["local"]["inputs"] = local_inputs(repo, script)
    elif declared is not None:
        fields["declared"] = dict(outputs=declared, shell=shell)
        if v1:
            fields["declared"]["inputs"] = local_inputs(repo, script)
    return fields, closure


def key_error(rec, rel_script, exc, t0):
    """Record (loudly, for an unkeyable build) why no key exists; '' outcome text."""
    if isinstance(exc, UnkeyableBuild):
        print("BINCACHE ERROR %s: %s; building WITHOUT the cache" % (rel_script, exc), file=sys.stderr)
        what = "unresolved-import:%s@%s" % (exc.module, exc.importer) if isinstance(exc, UnresolvedImport) \
            else "unkeyable"
        return "error-key:" + what.replace("\t", " ").replace(" ", "_")
    return "error-key:%s" % type(exc).__name__


def seed_dirs(environ):
    """MOJOLEARN_BINCACHE_SEED_DIRS: earlier caches (os.pathsep-separated, newest
    first) a LOCAL cache may copy a verified archive from. tools/release.py
    names the earlier releases' macos-bincache directories, so a release-scoped
    cache (kept whole as that release's provenance) still hits every group
    whose key did not change since an earlier release."""
    return [Path(d) for d in environ.get("MOJOLEARN_BINCACHE_SEED_DIRS", "").split(os.pathsep) if d]


def seed_archive(cache_dir, key, fields, environ):
    """Copy KEY's archive from the first seed directory that holds a valid one
    into CACHE_DIR; the seed directory, or ''. Never raises."""
    for d in seed_dirs(environ):
        cand = d / (key + ".tar.gz")
        if not cand.is_file() or cand.resolve().parent == Path(cache_dir).resolve():
            continue
        try:
            verify_archive(str(cand), key, fields)
            tmp = Path(cache_dir) / (".seed-%d-%s.tar.gz" % (os.getpid(), key))
            shutil.copyfile(cand, tmp)
            verify_archive(str(tmp), key, fields)
            os.replace(tmp, Path(cache_dir) / (key + ".tar.gz"))
            return str(d)
        except (Reject, OSError):
            continue
    return ""


def _manifest_of(arc):
    try:
        with tarfile.open(arc, "r:gz") as tf:
            return json.loads(tf.extractfile("manifest.json").read())
    except Exception:
        return None


def _changed(old, new, prefix=""):
    """Dotted names of the fields that differ between two field dicts."""
    out = []
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        if a == b:
            continue
        if isinstance(a, dict) and isinstance(b, dict) and k not in ("source",):
            out += _changed(a, b, prefix + k + ".")
        else:
            out.append(prefix + k)
    return out


def compile_reason(fields, closure, candidates):
    """Why FIELDS has no archive: the nearest earlier build of the same script
    and arguments among CANDIDATES (archive paths), and what differs."""
    best = None
    for arc in candidates:
        m = _manifest_of(arc)
        f = (m or {}).get("fields") or {}
        if f.get("script") == fields.get("script") and f.get("args") == fields.get("args") \
                and f.get("numeric_mode") == fields.get("numeric_mode"):
            when = m.get("built", "")
            if best is None or when > best[0]:
                best = (when, m)
    if best is None:
        return "no earlier build of this group in the cache"
    old = best[1]
    if old["fields"].get("key_rule") != fields.get("key_rule"):
        return "key rule changed (%s -> %s); a one-time rebuild" % (
            old["fields"].get("key_rule", "v1"), fields.get("key_rule", "v1"))
    diffs = _changed(old["fields"], fields)
    if "source" in diffs and closure is not None and isinstance(old.get("closure"), dict):
        prev = old["closure"]
        files = sorted(r for r in set(prev) | set(closure) if prev.get(r) != closure.get(r))
        diffs.remove("source")
        shown = ", ".join(files[:6]) + (" (+%d)" % (len(files) - 6) if len(files) > 6 else "")
        return "closure changed: " + shown + ("; also " + ", ".join(diffs) if diffs else "")
    return "changed: " + ", ".join(diffs[:8]) if diffs else "archive absent"


def invalidations(repo, changed, platforms=("darwin", "linux")):
    """{changed file: ["<platform>/<set>/<stem>", ...]}: the merged groups whose closure-v2 closure holds
    each file, read from REPO's working tree (tools/compile_closures.py walk). A file in no closure maps
    to []."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import compile_closures as cc
    tree = Path(repo).resolve()
    out = {f: [] for f in changed}
    for plat in platforms:
        for d, stem, _cls, mem in cc.group_rows(tree, plat):
            closure = set()
            for m in mem:
                closure |= set(cc.closure_of(tree, [tree / "bindings" / (m + ".mojo")]))
            for f in changed:
                if f in closure:
                    out[f].append("%s/%s/%s" % (plat, d, stem))
    return out


def invalidation_lines(inv):
    return ["%s invalidates these %d groups: %s" % (f, len(g), ", ".join(g)) if g else
            "%s invalidates no merged group" % f for f, g in sorted(inv.items())]


def cmd_explain(argv, environ=None):
    """explain [--since REV] <script> [args...]: one line, `hit (<where>) key=K` or
    `compile: <reason> key=K`, for the build cmd_build would run with this
    environment (MOJOLEARN_BINCACHE_DIR/_SEED_DIRS, _HOT_DIR, _MAP, _OUTPUTS,
    _SHELL, _IMAGE). Compiles nothing, writes nothing. Exit 0, or 3 when the
    build cannot be keyed (the reason names the import).

    --since REV first prints, per file changed since REV (git diff REV, the working tree), "<file>
    invalidates these N groups: ...". With MOJOLEARN_BINCACHE_KEY_RULE=v1 it also prints the legacy key's
    scope and, when it widened to the whole tree, why ("unresolved import <x> in <file> widened the
    closure"); closure-v2 keys never widen, they fail on the import (exit 3)."""
    environ = dict(os.environ if environ is None else environ)
    since = None
    if argv[:1] == ["--since"] and len(argv) >= 2:
        since, argv = argv[1], argv[2:]
    if not argv:
        print("usage: bincache.py explain [--since REV] <script> [args...]", file=sys.stderr)
        return 2
    script, args = argv[0], argv[1:]
    repo = Path(script).resolve().parent.parent
    label = "%s %s" % (os.path.relpath(Path(script).resolve(), repo), " ".join(args))
    if since:
        diff = subprocess.run(["git", "-C", str(repo), "diff", "--name-only", since], capture_output=True, text=True)
        if diff.returncode != 0:
            print("explain: git diff %s failed: %s" % (since, diff.stderr.strip()[:200]), file=sys.stderr)
            return 2
        changed = sorted(x for x in diff.stdout.splitlines() if x.endswith(".mojo"))
        try:
            for line in invalidation_lines(invalidations(repo, changed)):
                print(line)
        except UnkeyableBuild as exc:
            print("invalidations: unkeyable: %s" % exc)
    if environ.get("MOJOLEARN_BINCACHE_KEY_RULE") == "v1":
        # The legacy key (still recomputed by the retained-compile verifiers): say what it read and,
        # when it widened to the whole tree, why. Closure-v2 never widens; it fails on the import.
        src, _rels = source_digest(repo, script, args)
        widened = WIDENED.get((str(repo.resolve()), str(Path(script).resolve()), tuple(args)))
        print("legacy key: scope=%s files=%d%s  [%s]" % (src["scope"], src["files"],
                                                          ("; " + widened) if widened else "", label))
    outputs = sorted(o for o in environ.get("MOJOLEARN_BINCACHE_OUTPUTS", "").split() if o) or None
    local_dir = environ.get("MOJOLEARN_BINCACHE_DIR") if environ.get("MOJOLEARN_BINCACHE", "") != "0" else None
    map_path = environ.get("MOJOLEARN_BINCACHE_MAP", DEFAULT_MAP)
    urls = read_map(map_path) if os.path.isfile(map_path) and not local_dir else \
        dict(header=dict(image=environ.get("MOJOLEARN_BINCACHE_IMAGE", "")), get={}, sget={}, put=[])
    try:
        if local_dir:
            fields, closure = build_fields(repo, script, args, environ, "local", outputs,
                                           environ.get("MOJOLEARN_BINCACHE_SHELL", "sh"), local=True)
        else:
            fields, closure = build_fields(repo, script, args, environ, urls["header"].get("image", ""),
                                           outputs, environ.get("MOJOLEARN_BINCACHE_SHELL", "sh"))
    except UnkeyableBuild as exc:
        print("compile: unkeyable: %s  [%s]" % (exc, label))
        return 3
    key = key_of(fields)
    dirs = ([Path(local_dir)] + seed_dirs(environ)) if local_dir else []
    if environ.get("MOJOLEARN_BINCACHE_HOT_DIR") and not local_dir:
        dirs.append(Path(environ["MOJOLEARN_BINCACHE_HOT_DIR"]))
    for d in dirs:
        if (d / (key + ".tar.gz")).is_file():
            print("hit (%s) key=%s  [%s]" % (d, key[:16], label))
            return 0
    if key in urls["get"]:
        print("hit (r2) key=%s  [%s]" % (key[:16], label))
        return 0
    cands = [p for d in dirs if d.is_dir() for p in d.glob("*.tar.gz")]
    print("compile: %s key=%s  [%s]" % (compile_reason(fields, closure, cands), key[:16], label))
    return 0


def _prune(cache_dir, keep):
    arcs = sorted(Path(cache_dir).glob("*.tar.gz"), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in arcs[keep:]:
        try:
            p.unlink()
        except OSError:
            pass


def cmd_build_local(script, args, environ, cache_dir):
    shell = environ.get("MOJOLEARN_BINCACHE_SHELL", "sh")
    if shell not in ("sh", "bash"):
        print("MOJOLEARN_BINCACHE_SHELL must be sh or bash", file=sys.stderr)
        return 2
    plain = [shell, script] + args
    t0 = time.time()
    repo = Path(script).resolve().parent.parent
    rel_script = os.path.relpath(Path(script).resolve(), repo)
    cache_dir = Path(cache_dir)
    rec = Record(environ.get("MOJOLEARN_BINCACHE_OUT") or str(cache_dir / "provenance"))
    outputs = [o for o in environ.get("MOJOLEARN_BINCACHE_OUTPUTS", "").split() if o]
    retained_device = (environ.get("MOJOLEARN_BINCACHE_RETAINED_APPLE")
                       and Path(script).name == "merged_build.sh" and args[:1] == ["gpu"])
    if not outputs or not all(safe_rel(o) for o in outputs):
        if retained_device:
            rec.row(rel_script, "retained-refused:no-declared-outputs", "", time.time() - t0, [])
            return 2
        rc = subprocess.call(plain, env=environ)
        rec.row(rel_script, "refused:no-declared-outputs", "", time.time() - t0, [])
        return rc
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        fields = local_fields(repo, script, args, environ, outputs, shell)
        closure = _LOCAL_CLOSURES.pop(id(fields), None)
    except Exception as exc:                                   # never lose a build over the cache
        why_key = key_error(rec, rel_script, exc, t0)
        if retained_device:
            rec.row(rel_script, "retained-" + why_key.replace("error-key", "refused:error-key", 1), "",
                    time.time() - t0, [])
            return 2
        rc = subprocess.call(plain, env=environ)
        rec.row(rel_script, why_key, "", time.time() - t0, [])
        return rc
    why = refusal(fields, environ)
    if why:
        if retained_device:
            rec.row(rel_script, "retained-refused:" + why, "", time.time() - t0, [])
            return 2
        rc = subprocess.call(plain, env=environ)
        rec.row(rel_script, "refused:" + why, "", time.time() - t0, [])
        return rc
    key = key_of(fields)
    rec.key(key, fields)
    rec.closure(key, closure)
    if environ.get("MOJOLEARN_BINCACHE_RETAINED_APPLE"):
        # Explicit source reconciliation may admit an unchanged Apple device
        # output while a host edit changes the broad merged tree key. Preserve
        # its original proof/key/source and never write a new cache archive.
        # An invalid explicit admission fails; it must not trigger an expensive
        # replacement compile or silently relabel retained bytes as new.
        try:
            from apple_retained_groups import try_place
            admitted = try_place(repo, script, args, environ, outputs, fields, key, rec, t0)
        except (ValueError, OSError, KeyError, TypeError) as exc:
            rec.row(rel_script, "retained-refused:" + type(exc).__name__, key, time.time() - t0, [])
            print("retained device admission refused: " + str(exc), file=sys.stderr)
            return 2
        if admitted is not None:
            return admitted
    arc = cache_dir / (key + ".tar.gz")
    miss = "miss"
    if not arc.is_file():
        seeded = seed_archive(cache_dir, key, fields, environ)
        if seeded:
            print("BINCACHE seeded %s key=%s from %s" % (rel_script, key[:16], seeded))
    if arc.is_file():
        try:
            manifest, blobs = verify_archive(str(arc), key, fields)
            if sorted(blobs) != sorted(outputs):
                raise Reject("archive files differ from the declared outputs")
            have = installed_toolchain(repo)
            if have is not None and manifest.get("installed_toolchain") not in (None, have):
                raise Reject("installed toolchain differs")
            if any((repo / rel).exists() for rel in blobs):
                miss = "bypass-destination-exists"
            else:
                placed = []
                modes = {f["path"]: f.get("mode", 0o755) for f in manifest["files"]}
                for rel, data in sorted(blobs.items()):
                    dst = repo / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    tmp = dst.parent / (".bincache-%s-%s" % (os.getpid(), dst.name))
                    tmp.write_bytes(data)
                    os.chmod(tmp, modes[rel])
                    os.replace(tmp, dst)
                    if sha256_file(dst) != sha256_bytes(data):
                        dst.unlink()
                        raise Reject("placed file differs " + rel)
                    placed.append(dict(path=rel, sha256=sha256_bytes(data)))
                    print("built %s (from the local bincache, key %s, sha256 %s)"
                          % (rel, key[:16], placed[-1]["sha256"]))
                os.utime(arc)
                # The archive's own record of its compile (revision, date,
                # seconds, builder) next to the hit row: a placed group's
                # compile evidence (tools/release_requirements_macos_build.py).
                rec.placement(rel_script, "hit", key, placed, manifest)
                rec.row(rel_script, "hit", key, time.time() - t0, placed)
                return 0
        except Reject as exc:
            miss = "rejected:" + str(exc).replace("\t", " ")
        except OSError as exc:
            miss = "rejected:io-%s" % type(exc).__name__
    before = {o: _stat(repo / o) for o in outputs}
    tb = time.time()
    rc, log_sha = run_tee(plain, environ)
    build_seconds = time.time() - tb
    if rc != 0:
        rec.row(rel_script, miss + "+build-failed", key, time.time() - t0, [])
        return rc
    after = {o: _stat(repo / o) for o in outputs}
    stale = [o for o in outputs if after[o] is None or after[o] == before[o]]
    files_meta = [dict(path=o, sha256=sha256_file(repo / o)) for o in outputs if after[o] is not None]
    if stale:
        rec.row(rel_script, miss + "+built-not-cached:declared-output-not-written:" + ",".join(stale),
                key, time.time() - t0, files_meta)
        return 0
    try:
        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=".tmp-", suffix=".tar.gz", delete=False) as tf:
            tmp = tf.name
        pack(tmp, key, fields, repo, outputs, dict(
            builder=dict(host=platform.node(), leg="local"),
            # WHICH COMMIT THESE BYTES WERE COMPILED FROM (not a key field).
            source_commit=head_commit(repo),
            build_seconds=round(build_seconds, 1), build_log_sha256=log_sha,
            installed_toolchain=installed_toolchain(repo),
            built=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **({"closure": closure} if closure is not None else {})))
        # The archive must read back as a hit would read it, or it is not kept.
        verify_archive(tmp, key, fields)
        # ONE KEY, ONE CONTENT: never replace an archive of this key that holds
        # other files (its hit was refused, e.g. installed toolchain differs or
        # the destination existed). Equal files are rewritten as before: the
        # bytes are the same, and the fresh manifest (installed toolchain)
        # lets the next lookup hit where the old one was refused.
        verdict, _line = keep_or_refuse(arc, tmp, key, "local cache")
        if verdict == "conflict":
            os.unlink(tmp)
            rec.row(rel_script, miss + "+built-not-cached:conflict", key, time.time() - t0, files_meta)
            return 0
        os.replace(tmp, arc)
        _prune(cache_dir, int(environ.get("MOJOLEARN_BINCACHE_KEEP", LOCAL_KEEP)))
    except (Reject, OSError, ValueError) as exc:
        try:
            os.unlink(tmp)
        except (OSError, UnboundLocalError):
            pass
        rec.row(rel_script, miss + "+built-not-cached:%s" % type(exc).__name__, key, time.time() - t0, files_meta)
        return 0
    rec.row(rel_script, miss + "+built-cached", key, time.time() - t0, files_meta)
    return 0


def _stat(p):
    try:
        st = p.stat()
    except OSError:
        return None
    return (st.st_ino, st.st_mtime_ns, st.st_size)


def shard_entry(mode, script, args):
    """A build's name in a shard list: "<numeric mode>:<script file name>", the
    planner's entries (tools/release_github_build.py plan_set). A merged build
    is one group: "<mode>:merged_build.sh:<stem>" from its `--group <stem>`
    (packaging/linux/build_sets.sh pool_jobs). The merged host groups build
    under MOJOLEARN_NUMERIC_MODE=identical with the same script as the
    identical tier (bindings/merged_build.sh host), so they are "host:..."."""
    name = os.path.basename(script)
    args = list(args)
    if name == "merged_build.sh":
        if args[:1] == ["host"]:
            mode = "host"
        if "--group" in args[:-1]:
            return "%s:%s:%s" % (mode, name, args[args.index("--group") + 1])
    return "%s:%s" % (mode, name)


def cmd_build(argv, environ=None):
    environ = dict(os.environ if environ is None else environ)
    if not argv:
        print("usage: bincache.py build <bindings/build_X.sh> [args...]", file=sys.stderr)
        return 2
    script, args = argv[0], argv[1:]
    if (environ.get("MOJOLEARN_BINCACHE_RETAINED_APPLE") and Path(script).name == "merged_build.sh"
            and args[:1] == ["gpu"] and (environ.get("MOJOLEARN_BINCACHE") == "0"
            or not environ.get("MOJOLEARN_BINCACHE_DIR") or environ.get("MOJOLEARN_BINCACHE_SHARD_ONLY"))):
        print("explicit retained Apple admission requires the unsharded local-cache route", file=sys.stderr)
        return 2
    # ONE SHARD OF A SET (the GitHub build route, tools/release_github_build.py,
    # 2026-09-27). MOJOLEARN_BINCACHE_SHARD_ONLY lists the builds this runner
    # compiles, each "<numeric mode>:<script file name>"; every other build of
    # the set returns at once without compiling or placing anything, so the
    # unchanged release route (build_sets.sh) runs just this shard, and the
    # hot directory collects its archives for the job that assembles the set.
    # The variable is under MOJOLEARN_BINCACHE, so it never reaches a key, and
    # a shard's own tree is never a release set: missing members fail
    # linux_surface_qualification.sh's completeness check by construction.
    only = environ.get("MOJOLEARN_BINCACHE_SHARD_ONLY", "").split()
    if only:
        me = shard_entry(environ.get("MOJOLEARN_NUMERIC_MODE") or "identical", script, args)
        if me not in only:
            print("BINCACHE skipped %s (not in this shard)" % me)
            return 0
    require_hit = environ.get("MOJOLEARN_BINCACHE_REQUIRE_HIT") == "1"
    def compile_or_refuse(command):
        if require_hit:
            print("BINCACHE retained assembly refused compilation", file=sys.stderr)
            return 3
        return subprocess.call(command, env=environ)
    if environ.get("MOJOLEARN_BINCACHE", "") != "0" and environ.get("MOJOLEARN_BINCACHE_DIR"):
        if require_hit:
            print("BINCACHE require-hit needs the verified archive-map route", file=sys.stderr)
            return 3
        return cmd_build_local(script, args, environ, environ["MOJOLEARN_BINCACHE_DIR"])
    map_path = environ.get("MOJOLEARN_BINCACHE_MAP", DEFAULT_MAP)
    shell = environ.get("MOJOLEARN_BINCACHE_SHELL", "sh")
    if shell not in ("sh", "bash"):
        print("MOJOLEARN_BINCACHE_SHELL must be sh or bash", file=sys.stderr)
        return 2
    plain = [shell, script] + args
    # THE BOX-LOCAL ROUTE (2026-11-04): with no URL map but a hot directory
    # (tools/box_merged_verify.sh keeps one on the box across jobs), the hot
    # directory alone is the cache; MOJOLEARN_BINCACHE_IMAGE names the image.
    hot_only = (not os.path.isfile(map_path) and bool(environ.get("MOJOLEARN_BINCACHE_HOT_DIR"))
                and environ.get("MOJOLEARN_BINCACHE", "") != "0")
    if environ.get("MOJOLEARN_BINCACHE", "") == "0" or (not os.path.isfile(map_path) and not hot_only):
        return compile_or_refuse(plain)
    t0 = time.time()
    repo = Path(script).resolve().parent.parent
    if hot_only:
        urls = dict(header=dict(image=environ.get("MOJOLEARN_BINCACHE_IMAGE", ""), leg="hot-only"),
                    get={}, sget={}, put=[])
    else:
        urls = read_map(map_path)
    rec = Record(environ.get("MOJOLEARN_BINCACHE_OUT", DEFAULT_OUT))
    rel_script = os.path.relpath(Path(script).resolve(), repo)
    # DECLARED OUTPUTS (MOJOLEARN_BINCACHE_OUTPUTS; the release build,
    # packaging/linux/build_sets.sh, 2026-09-22). That script runs several
    # builds at once in one tree, where the snapshot diff below would archive
    # a neighbour's .so under this key. A declared build is keyed WIDER, with
    # the local cache's `inputs` (every shell or Python file the script names,
    # followed through the scripts it runs, and the tokenizer generators) and
    # the output list, so it never shares a key with an undeclared one.
    declared = None
    if environ.get("MOJOLEARN_BINCACHE_OUTPUTS", "").strip():
        declared = sorted(o for o in environ["MOJOLEARN_BINCACHE_OUTPUTS"].split() if o)
        if not all(safe_rel(o) for o in declared):
            rc = compile_or_refuse(plain)
            rec.row(rel_script, "refused:unsafe-declared-output", "", time.time() - t0, [])
            return rc
    try:
        fields, closure = build_fields(repo, script, args, environ, urls["header"].get("image", ""),
                                       declared, shell)
    except Exception as exc:                                   # never lose a build over the cache
        why_key = key_error(rec, rel_script, exc, t0)
        rc = compile_or_refuse(plain)
        rec.row(rel_script, why_key, "", time.time() - t0, [])
        return rc
    why = refusal(fields, environ)
    negative = bool(why.startswith("sabotage:") and environ.get("MOJOLEARN_BINCACHE_NEGATIVE", "") == "1"
                    and fields.get("variant") == "sabotage" and not outdir_refusal(fields, environ))
    if why and not negative:
        rc = compile_or_refuse(plain)
        rec.row(rel_script, "refused:" + why, "", time.time() - t0, [])
        return rc
    # A negative control reads and writes ONLY the sabotage namespace; a
    # production build ONLY the production one.
    gets = urls["sget"] if negative else urls["get"]
    tag = "negative-" if negative else ""
    key = key_of(fields)
    rec.key(key, fields)
    rec.closure(key, closure)
    miss = tag + "miss"
    # THE BOX-LOCAL HOT DIRECTORY (MOJOLEARN_BINCACHE_HOT_DIR, the release
    # build, 2026-09-22): archives this box built earlier in the same run, so
    # the three release sets built one after another share the 32 host
    # bindings (vendor-neutral, keyed without any GPU variable) before the Mac
    # has promoted anything. Read and verified exactly like an R2 object.
    hot = Path(environ["MOJOLEARN_BINCACHE_HOT_DIR"]) if environ.get("MOJOLEARN_BINCACHE_HOT_DIR") and not negative else None
    if hot is not None and (hot / (key + ".tar.gz")).is_file():
        got = _try_place(str(hot / (key + ".tar.gz")), key, fields, repo, declared, rec, rel_script,
                         "hot-hit", t0)
        if got == 0:
            return 0
        miss = got
    if key in gets:
        with tempfile.TemporaryDirectory(prefix="bincache-") as td:
            arc = os.path.join(td, "a.tar.gz")
            code = http_get(gets[key], arc)
            if code != 200:
                miss = "miss-get-%s" % code
            else:
                got = _try_place(arc, key, fields, repo, declared, rec, rel_script, tag + "hit", t0, tag=tag)
                if got == 0:
                    return 0
                miss = got
    if require_hit:
        rec.row(rel_script, str(miss) + "+compile-refused", key, time.time() - t0, [])
        return compile_or_refuse(plain)
    before = snapshot(repo, environ) if declared is None else {o: _stat(repo / o) for o in declared}
    tb = time.time()
    rc, log_sha = run_tee(plain, environ)
    build_seconds = time.time() - tb
    if rc != 0:
        rec.row(rel_script, miss + "+build-failed", key, time.time() - t0, [])
        return rc
    if declared is None:
        after = snapshot(repo, environ)
        outputs = []
        for p, st in after.items():
            if before.get(p) != st:
                rp = Path(p)
                try:
                    outputs.append(str(rp.relative_to(repo.resolve())))
                except ValueError:
                    outputs = None
                    break
    else:
        # A declared output the build did not rewrite is never archived: the
        # archive would carry some other build's bytes under this key.
        after = {o: _stat(repo / o) for o in declared}
        stale = [o for o in declared if after[o] is None or after[o] == before[o]]
        if stale:
            files_meta = [dict(path=o, sha256=sha256_file(repo / o)) for o in declared if after[o] is not None]
            rec.row(rel_script, miss + "+built-not-cached:declared-output-not-written:" + ",".join(stale),
                    key, time.time() - t0, files_meta)
            return 0
        outputs = list(declared)
    files_meta = [dict(path=o, sha256=sha256_file(repo / o)) for o in sorted(outputs or [])]
    if not outputs:
        rec.row(rel_script, miss + "+built-not-cached:%s" % ("outside-tree" if outputs is None else "no-outputs"),
                key, time.time() - t0, files_meta)
        return 0
    if rec.dir is None:
        rec.row(rel_script, miss + "+built-not-uploaded:no-out-dir", key, time.time() - t0, files_meta)
        return 0
    extra = dict(builder=dict(host=platform.node(), leg=urls["header"].get("leg", "")),
                 build_seconds=round(build_seconds, 1), build_log_sha256=log_sha,
                 installed_toolchain=installed_toolchain(repo),
                 # WHICH COMMIT THESE BYTES WERE COMPILED FROM. Not a key field
                 # (a commit that changes nothing the binding reads must hit),
                 # but carried so a release's provenance can name, per binary,
                 # the commit whose build produced the bytes it ships.
                 source_commit=environ.get("MOJOLEARN_COMMIT", ""),
                 built=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if closure is not None:
        extra["closure"] = closure  # not a key field; `explain` names changed files from it
    with tempfile.TemporaryDirectory(prefix="bincache-") as td:
        arc = os.path.join(td, "a.tar.gz")
        pack(arc, key, fields, repo, outputs, extra)
        if hot is not None:
            try:
                verify_archive(arc, key, fields)
                hot.mkdir(parents=True, exist_ok=True)
                # ONE KEY, ONE CONTENT: an archive already in the hot directory
                # under this key (a shard's, or an earlier set's) is never
                # replaced by other bytes; equal files are rewritten as before.
                verdict, _line = keep_or_refuse(hot / (key + ".tar.gz"), arc, key, "hot directory")
                if verdict != "conflict":
                    tmp_hot = hot / (".tmp-%d-%s.tar.gz" % (os.getpid(), key))
                    shutil.copyfile(arc, tmp_hot)
                    os.replace(tmp_hot, hot / (key + ".tar.gz"))
                elif verdict == "conflict":
                    # The R2 upload of the same bytes is refused too: the key
                    # already names another content on this box.
                    rec.row(rel_script, miss + "+built-not-uploaded:conflict", key, time.time() - t0, files_meta)
                    return 0
            except (Reject, OSError):
                pass
        slot, put_url = claim_slot(map_path, urls["put"]) if urls["put"] else (None, None)
        if slot is None:
            rec.row(rel_script, miss + "+built-not-uploaded:no-slot", key, time.time() - t0, files_meta)
            return 0
        partition = urls["header"].get("partition", "")
        dest = "%s/%s/%s.tar.gz" % (SABOTAGE_PREFIX if negative else OBJECT_PREFIX, partition, key)
        code = http_put(put_url, arc)
    if code != 200:
        rec.row(rel_script, miss + "+built-upload-failed-%s" % code, key, time.time() - t0, files_meta)
        return 0
    rec.upload_row(urls["header"].get("leg", ""), slot, dest, key, sabotage=negative)
    rec.row(rel_script, miss + "+built-uploaded", key, time.time() - t0, files_meta)
    return 0


def _try_place(arc, key, fields, repo, declared, rec, rel_script, outcome, t0, tag=""):
    """Verify an archive and place its files; 0 on a hit, else the miss reason."""
    try:
        manifest, blobs = verify_archive(arc, key, fields)
        if declared is not None and sorted(blobs) != sorted(declared):
            raise Reject("archive files differ from the declared outputs")
        have = installed_toolchain(repo)
        if have is not None and manifest.get("installed_toolchain") not in (None, have):
            raise Reject("installed toolchain differs")
        if any((repo / rel).exists() for rel in blobs):
            return "bypass-destination-exists"
        placed = []
        modes = {f["path"]: f.get("mode", 0o755) for f in manifest["files"]}
        for rel, data in sorted(blobs.items()):
            dst = repo / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.parent / (".bincache-%s-%s" % (os.getpid(), dst.name))
            tmp.write_bytes(data)
            os.chmod(tmp, modes[rel])
            os.replace(tmp, dst)
            if sha256_file(dst) != sha256_bytes(data):
                raise Reject("placed file differs " + rel)
            placed.append(dict(path=rel, sha256=sha256_bytes(data)))
            print("built %s (from bincache key %s, sha256 %s)" % (rel, key[:16], placed[-1]["sha256"]))
        rec.row(rel_script, outcome, key, time.time() - t0, placed)
        rec.placement(rel_script, outcome, key, placed, manifest)
        return 0
    except Reject as exc:
        return tag + "rejected:" + str(exc).replace("\t", " ")


# ---------------------------------------------------------------------------
# the Mac: SigV4 presigning, listing, promotion (credentials from the env)
# ---------------------------------------------------------------------------

def _hmac(k, m):
    return hmac.new(k, m.encode(), hashlib.sha256).digest()


def presign(method, key, expires, creds, query=None, headers=None, now=None, host=None,
            region="auto", path_style=True):
    """A SigV4 query-signed URL. `headers` are signed and must be sent."""
    now = now or datetime.datetime.now(datetime.timezone.utc)
    host = host or "%s.r2.cloudflarestorage.com" % creds["R2_ACCOUNT_ID"]
    amzdate = now.strftime("%Y%m%dT%H%M%SZ")
    datestamp = now.strftime("%Y%m%d")
    scope = "%s/%s/s3/aws4_request" % (datestamp, region)
    if path_style:
        path = ("%s/%s" % (creds["R2_BUCKET"], key)) if key else creds["R2_BUCKET"]
    else:
        path = key
    uri = "/" + urllib.parse.quote(path, safe="/~")
    hdrs = {"host": host}
    for k, v in (headers or {}).items():
        hdrs[k.lower()] = v.strip()
    signed = ";".join(sorted(hdrs))
    q = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256",
         "X-Amz-Credential": "%s/%s" % (creds["R2_ACCESS_KEY_ID"], scope),
         "X-Amz-Date": amzdate, "X-Amz-Expires": str(expires), "X-Amz-SignedHeaders": signed}
    q.update(query or {})
    cqs = "&".join("%s=%s" % (urllib.parse.quote(k, safe="-_.~"), urllib.parse.quote(v, safe="-_.~"))
                   for k, v in sorted(q.items()))
    chdr = "".join("%s:%s\n" % (k, hdrs[k]) for k in sorted(hdrs))
    creq = "\n".join([method, uri, cqs, chdr, signed, "UNSIGNED-PAYLOAD"])
    sts = "\n".join(["AWS4-HMAC-SHA256", amzdate, scope, hashlib.sha256(creq.encode()).hexdigest()])
    k = _hmac(("AWS4" + creds["R2_SECRET_ACCESS_KEY"]).encode(), datestamp)
    for part in (region, "s3", "aws4_request"):
        k = _hmac(k, part)
    sig = hmac.new(k, sts.encode(), hashlib.sha256).hexdigest()
    return "https://%s%s?%s&X-Amz-Signature=%s" % (host, uri, cqs, sig)


def creds_from_env(environ=None):
    environ = os.environ if environ is None else environ
    c = {k: environ.get(k, "") for k in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")}
    missing = [k for k, v in c.items() if not v]
    if missing:
        raise SystemExit("bincache: R2 credentials missing from the environment: %s" % ",".join(missing))
    return c


def list_objects(creds, prefix):
    out, token = [], None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        q = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            q["continuation-token"] = token
        url = presign("GET", "", 600, creds, query=q)
        with urllib.request.urlopen(url, timeout=60) as r:
            root = ET.fromstring(r.read())
        for c in root.findall(ns + "Contents"):
            out.append((c.find(ns + "Key").text, c.find(ns + "LastModified").text, int(c.find(ns + "Size").text)))
        if (root.findtext(ns + "IsTruncated") or "false") != "true":
            return out
        token = root.findtext(ns + "NextContinuationToken")


def r2_request(creds, method, key, headers=None, data=None, timeout=120):
    url = presign(method, key, 600, creds, headers=headers)
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read()


def cmd_plan(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="bincache.py plan")
    ap.add_argument("--partition", required=True, help="<device arch>/<image slug>")
    ap.add_argument("--image", required=True)
    ap.add_argument("--leg-id", required=True)
    ap.add_argument("--slots", type=int, default=64)
    ap.add_argument("--max-get", type=int, default=2000)
    ap.add_argument("--expires", type=int, default=7200)
    ap.add_argument("--negative", action="store_true",
                    help="also list the sabotage namespace as sget rows (negative controls)")
    a = ap.parse_args(argv)
    lines, n_get, n_sget = plan_lines(creds_from_env(), a.partition, a.image, a.leg_id, a.slots,
                                      max_get=a.max_get, expires=a.expires, negative=a.negative)
    sys.stdout.write("\n".join(lines) + "\n")
    print("BINCACHE PLAN partition=%s entries=%d negative_entries=%d slots=%d"
          % (a.partition, n_get, n_sget, a.slots), file=sys.stderr)
    return 0


def plan_lines(creds, partition, image, leg_id, slots, max_get=2000, expires=7200, negative=False):
    """(map lines, gets, sgets): the URL map `plan` prints (and the GitHub
    build route mints, tools/release_github_build.py)."""
    parts = partition.split("/")
    if len(parts) != 2 or not all(SEG_RE.match(p) for p in parts) or not SEG_RE.match(leg_id):
        raise SystemExit("bincache plan: bad partition or leg id")
    lines = ["#partition\t" + partition, "#image\t" + image, "#leg\t" + leg_id]
    n_get = n_sget = 0
    for prefix, verb in ((OBJECT_PREFIX, "get"), (SABOTAGE_PREFIX, "sget")):
        if verb == "sget" and not negative:
            continue
        objs = list_objects(creds, "%s/%s/" % (prefix, partition))
        objs.sort(key=lambda o: o[1], reverse=True)
        for name, _, _ in objs[: max_get]:
            m = re.match(r"^%s/%s/([0-9a-f]{64})\.tar\.gz$" % (re.escape(prefix), re.escape(partition)), name)
            if m:
                lines.append("%s\t%s\t%s" % (verb, m.group(1), presign("GET", name, expires, creds)))
                if verb == "get":
                    n_get += 1
                else:
                    n_sget += 1
    for i in range(slots):
        slot = "%03d" % i
        lines.append("put\t%s\t%s" % (slot, presign("PUT", "%s/%s/%s.tar.gz" % (INBOX_PREFIX, leg_id, slot),
                                                    expires, creds)))
    return lines, n_get, n_sget


def check_upload_row(row, keys_dir):
    """(slot, dest, key) or raise ValueError. The Mac re-derives the key
    from the fields the box recorded and refuses anything sabotaged."""
    parts = row.rstrip("\n").split("\t")
    if len(parts) != 5 or parts[4] not in ("sabotage=0", "sabotage=1"):
        raise ValueError("malformed row")
    leg, slot, dest, key = parts[:4]
    negative = parts[4] == "sabotage=1"
    if not SEG_RE.match(leg) or not SEG_RE.match(slot) or not KEY_RE.match(key):
        raise ValueError("bad leg, slot or key")
    m = re.match(r"^%s/([A-Za-z0-9._-]{1,120})/([A-Za-z0-9._-]{1,120})/([0-9a-f]{64})\.tar\.gz$"
                 % re.escape(SABOTAGE_PREFIX if negative else OBJECT_PREFIX), dest)
    if not m or m.group(3) != key:
        raise ValueError("bad destination")
    fields = json.loads((Path(keys_dir) / (key + ".json")).read_text())
    if key_of(fields) != key:
        raise ValueError("recorded fields do not hash to the key")
    sab = is_sabotage(fields)
    if negative:
        # A negative control goes ONLY to the sabotage namespace, and only
        # when its own fields say sabotage.
        if not sab or fields.get("variant") != "sabotage":
            raise ValueError("row says sabotage=1 but the fields are a production build")
    elif sab or "variant" in fields:
        raise ValueError("sabotage build")
    return leg, slot, dest, key


class R2Store:
    """The promote transport: R2 through this Mac's credentials. The tests
    pass a directory-backed object with the same three methods."""

    def __init__(self, creds):
        self.creds = creds

    def fetch(self, name, dest):
        """True when NAME exists and its bytes are now at DEST, False on 404."""
        code = http_get(presign("GET", name, 600, self.creds), dest)
        if code == 404:
            return False
        if code != 200:
            raise OSError("GET %s: HTTP %s" % (name, code))
        return True

    def copy(self, src, dest):
        r2_request(self.creds, "PUT", dest, headers={
            "x-amz-copy-source": "/%s/%s" % (self.creds["R2_BUCKET"], urllib.parse.quote(src, safe="/"))})

    def delete(self, name):
        r2_request(self.creds, "DELETE", name)


def promote_row(store, leg, slot, dest, key):
    """Copy one inbox upload to its content address unless the address already
    holds an archive: then ("same", "") when its files equal the upload's (the
    inbox object is deleted, nothing is written) and ("conflict", line) when
    they differ (nothing is written, the inbox object stays for inspection).
    ("promoted", "") after a copy. Raises on a transport failure."""
    src = "%s/%s/%s.tar.gz" % (INBOX_PREFIX, leg, slot)
    with tempfile.TemporaryDirectory(prefix="bincache-promote-") as td:
        old, new = os.path.join(td, "old.tar.gz"), os.path.join(td, "new.tar.gz")
        if store.fetch(dest, old):
            if not store.fetch(src, new):
                raise OSError("inbox object %s is missing" % src)
            try:
                verdict, line = keep_or_refuse(old, new, key, dest)
            except Reject as exc:
                return "conflict", "BINCACHE CONFLICT key=%s at %s: the upload is unreadable (%s); existing kept" % (
                    key, dest, exc)
            if verdict == "same":
                store.delete(src)
                return "same", ""
            if verdict == "conflict":
                return "conflict", line
            # "write": the existing object was unreadable; the upload repairs it
    store.copy(src, dest)
    store.delete(src)
    return "promoted", ""


def cmd_promote(argv, store=None):
    if len(argv) != 1:
        raise SystemExit("usage: bincache.py promote <leg out>/remote/bincache")
    d = Path(argv[0])
    up = d / "uploads.tsv"
    if not up.is_file():
        print("BINCACHE PROMOTED 0 (no uploads.tsv)")
        return 0
    store = store or R2Store(creds_from_env())
    ok = bad = same = conflicts = 0
    for row in up.read_text().splitlines():
        if not row.strip():
            continue
        try:
            leg, slot, dest, key = check_upload_row(row, d / "keys")
        except (ValueError, OSError) as exc:
            print("  refused row: %s" % exc)
            bad += 1
            continue
        try:
            verdict, line = promote_row(store, leg, slot, dest, key)
        except Exception as exc:
            print("  promote failed %s: %s" % (dest, type(exc).__name__))
            bad += 1
            continue
        if verdict == "promoted":
            ok += 1
            print("  promoted %s" % dest)
        elif verdict == "same":
            same += 1
            print("  already holds these bytes %s" % dest)
        else:
            conflicts += 1
            print("  " + line)
    # The summary stays the last line (tools/release_github_build.py promote reads it).
    print("BINCACHE PROMOTED %d refused_or_failed=%d already_same=%d conflicts=%d" % (ok, bad, same, conflicts))
    # A key that would hold two byte contents is a loud failure of this command;
    # every current caller (the leg scripts' `|| true`, release_github_build's
    # promote) keeps going and reports the summary line.
    return 4 if conflicts else 0


def cmd_selftest_r2(argv):
    """A live round trip under bincache-selftest/: presigned PUT, list, GET,
    server-side copy, DELETE. Small bytes; no box involved."""
    creds = creds_from_env()
    tag = "%s-%d" % (time.strftime("%Y%m%dT%H%M%S"), os.getpid())
    a, b = "bincache-selftest/%s/a.bin" % tag, "bincache-selftest/%s/b.bin" % tag
    payload = os.urandom(4096)
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "src")
        Path(src).write_bytes(payload)
        assert http_put(presign("PUT", a, 300, creds), src) == 200, "presigned PUT"
        names = [o[0] for o in list_objects(creds, "bincache-selftest/%s/" % tag)]
        assert names == [a], "list"
        got = os.path.join(td, "got")
        assert http_get(presign("GET", a, 300, creds), got) == 200 and Path(got).read_bytes() == payload, "GET"
        r2_request(creds, "PUT", b, headers={"x-amz-copy-source": "/%s/%s" % (creds["R2_BUCKET"], a)})
        assert http_get(presign("GET", b, 300, creds), got) == 200 and Path(got).read_bytes() == payload, "copy"
        assert http_get(presign("GET", "bincache-selftest/%s/none" % tag, 300, creds), got) == 404, "404"
        r2_request(creds, "DELETE", a)
        r2_request(creds, "DELETE", b)
        assert list_objects(creds, "bincache-selftest/%s/" % tag) == [], "delete"
    print("BINCACHE R2 SELFTEST OK (put, list, get, copy, 404, delete)")
    return 0


def cmd_key(argv):
    if not argv:
        raise SystemExit("usage: bincache.py key <script> [args...]")
    environ = dict(os.environ)
    repo = Path(argv[0]).resolve().parent.parent
    try:
        fields = key_fields(repo, argv[0], argv[1:], environ, environ.get("MOJOLEARN_BINCACHE_IMAGE", ""),
                            force_tree=environ.get("MOJOLEARN_BINCACHE_SOURCE") == "tree")
    except UnkeyableBuild as exc:
        print("BINCACHE ERROR %s: %s" % (argv[0], exc), file=sys.stderr)
        return 3
    print(json.dumps(dict(key=key_of(fields), refusal=refusal(fields, environ), fields=fields), indent=1, sort_keys=True))
    return 0


def main(argv):
    cmds = {"build": cmd_build, "key": cmd_key, "explain": cmd_explain, "plan": cmd_plan, "promote": cmd_promote,
            "selftest-r2": cmd_selftest_r2,
            "device-arch": lambda a: print(device_arch()) or 0}
    if not argv or argv[0] not in cmds:
        print(__doc__)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
