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
bytes (`key_fields`):
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

  ON THE BOX (inside a body; a pass-through when the URL map is absent or
  MOJOLEARN_BINCACHE=0):
    python3 tools/bincache.py build bindings/build_rf.sh [args...]
    python3 tools/bincache.py key bindings/build_rf.sh     # print the fields
  ON THE MAC (through tools/bincache_leg.sh, which loads ~/.mojolearn_r2):
    python3 tools/bincache.py plan --partition P --image I --leg-id L
    python3 tools/bincache.py promote <leg out>/remote/bincache
    python3 tools/bincache.py selftest-r2

Provenance lands in $MOJOLEARN_BINCACHE_OUT (default
/root/gemm_leg_out/bincache): provenance.tsv names every build, whether it
was a cache hit, and the sha256 of every file it placed; keys/<key>.json holds
the fields. Stdlib only.
"""
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
                 "MOJOLEARN_BUILD_LOCK_HELD", "MOJOLEARN_PYTHON", "MOJOLEARN_SMOKE_SO")
NON_BUILD_PREFIXES = ("MOJOLEARN_BINCACHE", "MOJOLEARN_IDENTITY_", "MOJOLEARN_STAGE_",
                      "MOJOLEARN_GEMM_LEG_", "MOJOLEARN_HOTAISLE_", "MOJOLEARN_DO_")
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
            if init.is_file():
                found.add(init)
        if parts:
            leaf = root.joinpath(*parts[:-1], parts[-1] + ".mojo")
            if leaf.is_file():
                found.add(leaf)
        pkg = root.joinpath(*parts) if parts else root
        for name in names:
            if name == "*":
                if pkg.is_dir():
                    found.update(p for p in pkg.glob("*.mojo") if p.is_file())
                continue
            for cand in (pkg / (name + ".mojo"), pkg / name / "__init__.mojo"):
                if cand.is_file():
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
            srcs = ["bindings/_mojolearn_%s_host.mojo" % family]
        if len(srcs) != 1 or "$" in srcs[0]:
            return None
        roots.append(repo / srcs[0].strip("\"'"))
    if not roots or not all(r.is_file() for r in roots):
        return None
    return roots, sorted(incs), shells


def source_digest(repo, script, args, force_tree=False):
    repo = Path(repo).resolve()
    plan = None if force_tree else script_plan(repo, script, args)
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


def key_fields(repo, script, args, environ, image, dev_arch=None, os_info=None, force_tree=False):
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
    for k, v in list(fields.get("build_env", {}).items()) + [("args", " ".join(fields.get("args", [])))]:
        if SABOTAGE_RE.search(k) or SABOTAGE_RE.search(str(v)):
            return True
    return False


def refusal(fields, environ):
    """Why this build must never touch the cache, or ''."""
    for k, v in list(fields["build_env"].items()) + [("args", " ".join(fields["args"]))]:
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


def cmd_build(argv, environ=None):
    environ = dict(os.environ if environ is None else environ)
    if not argv:
        print("usage: bincache.py build <bindings/build_X.sh> [args...]", file=sys.stderr)
        return 2
    script, args = argv[0], argv[1:]
    map_path = environ.get("MOJOLEARN_BINCACHE_MAP", DEFAULT_MAP)
    plain = ["sh", script] + args
    if environ.get("MOJOLEARN_BINCACHE", "") == "0" or not os.path.isfile(map_path):
        return subprocess.call(plain, env=environ)
    t0 = time.time()
    repo = Path(script).resolve().parent.parent
    urls = read_map(map_path)
    rec = Record(environ.get("MOJOLEARN_BINCACHE_OUT", DEFAULT_OUT))
    rel_script = os.path.relpath(Path(script).resolve(), repo)
    try:
        fields = key_fields(repo, script, args, environ, urls["header"].get("image", ""),
                            force_tree=environ.get("MOJOLEARN_BINCACHE_SOURCE") == "tree")
    except Exception as exc:                                   # never lose a build over the cache
        rc = subprocess.call(plain, env=environ)
        rec.row(rel_script, "error-key:%s" % type(exc).__name__, "", time.time() - t0, [])
        return rc
    why = refusal(fields, environ)
    negative = bool(why.startswith("sabotage:") and environ.get("MOJOLEARN_BINCACHE_NEGATIVE", "") == "1"
                    and fields.get("variant") == "sabotage" and not outdir_refusal(fields, environ))
    if why and not negative:
        rc = subprocess.call(plain, env=environ)
        rec.row(rel_script, "refused:" + why, "", time.time() - t0, [])
        return rc
    # A negative control reads and writes ONLY the sabotage namespace; a
    # production build ONLY the production one.
    gets = urls["sget"] if negative else urls["get"]
    tag = "negative-" if negative else ""
    key = key_of(fields)
    rec.key(key, fields)
    miss = tag + "miss"
    if key in gets:
        with tempfile.TemporaryDirectory(prefix="bincache-") as td:
            arc = os.path.join(td, "a.tar.gz")
            code = http_get(gets[key], arc)
            if code != 200:
                miss = "miss-get-%s" % code
            else:
                try:
                    manifest, blobs = verify_archive(arc, key, fields)
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
                                raise Reject("placed file differs " + rel)
                            placed.append(dict(path=rel, sha256=sha256_bytes(data)))
                            print("built %s (from bincache key %s, sha256 %s)" % (rel, key[:16], placed[-1]["sha256"]))
                        rec.row(rel_script, tag + "hit", key, time.time() - t0, placed)
                        return 0
                except Reject as exc:
                    miss = tag + "rejected:" + str(exc).replace("\t", " ")
    before = snapshot(repo, environ)
    tb = time.time()
    rc, log_sha = run_tee(plain, environ)
    build_seconds = time.time() - tb
    if rc != 0:
        rec.row(rel_script, miss + "+build-failed", key, time.time() - t0, [])
        return rc
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
    files_meta = [dict(path=o, sha256=sha256_file(repo / o)) for o in sorted(outputs or [])]
    if not outputs:
        rec.row(rel_script, miss + "+built-not-cached:%s" % ("outside-tree" if outputs is None else "no-outputs"),
                key, time.time() - t0, files_meta)
        return 0
    if rec.dir is None:
        rec.row(rel_script, miss + "+built-not-uploaded:no-out-dir", key, time.time() - t0, files_meta)
        return 0
    slot, put_url = claim_slot(map_path, urls["put"])
    if slot is None:
        rec.row(rel_script, miss + "+built-not-uploaded:no-slot", key, time.time() - t0, files_meta)
        return 0
    partition = urls["header"].get("partition", "")
    dest = "%s/%s/%s.tar.gz" % (SABOTAGE_PREFIX if negative else OBJECT_PREFIX, partition, key)
    with tempfile.TemporaryDirectory(prefix="bincache-") as td:
        arc = os.path.join(td, "a.tar.gz")
        pack(arc, key, fields, repo, outputs, dict(
            builder=dict(host=platform.node(), leg=urls["header"].get("leg", "")),
            build_seconds=round(build_seconds, 1), build_log_sha256=log_sha,
            installed_toolchain=installed_toolchain(repo),
            built=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        code = http_put(put_url, arc)
    if code != 200:
        rec.row(rel_script, miss + "+built-upload-failed-%s" % code, key, time.time() - t0, files_meta)
        return 0
    rec.upload_row(urls["header"].get("leg", ""), slot, dest, key, sabotage=negative)
    rec.row(rel_script, miss + "+built-uploaded", key, time.time() - t0, files_meta)
    return 0


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


def r2_request(creds, method, key, headers=None, data=None):
    url = presign(method, key, 600, creds, headers=headers)
    req = urllib.request.Request(url, data=data, method=method, headers=headers or {})
    with urllib.request.urlopen(req, timeout=120) as r:
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
    parts = a.partition.split("/")
    if len(parts) != 2 or not all(SEG_RE.match(p) for p in parts) or not SEG_RE.match(a.leg_id):
        raise SystemExit("bincache plan: bad partition or leg id")
    creds = creds_from_env()
    lines = ["#partition\t" + a.partition, "#image\t" + a.image, "#leg\t" + a.leg_id]
    n_get = n_sget = 0
    for prefix, verb in ((OBJECT_PREFIX, "get"), (SABOTAGE_PREFIX, "sget")):
        if verb == "sget" and not a.negative:
            continue
        objs = list_objects(creds, "%s/%s/" % (prefix, a.partition))
        objs.sort(key=lambda o: o[1], reverse=True)
        for name, _, _ in objs[: a.max_get]:
            m = re.match(r"^%s/%s/([0-9a-f]{64})\.tar\.gz$" % (re.escape(prefix), re.escape(a.partition)), name)
            if m:
                lines.append("%s\t%s\t%s" % (verb, m.group(1), presign("GET", name, a.expires, creds)))
                if verb == "get":
                    n_get += 1
                else:
                    n_sget += 1
    for i in range(a.slots):
        slot = "%03d" % i
        lines.append("put\t%s\t%s" % (slot, presign("PUT", "%s/%s/%s.tar.gz" % (INBOX_PREFIX, a.leg_id, slot),
                                                    a.expires, creds)))
    sys.stdout.write("\n".join(lines) + "\n")
    print("BINCACHE PLAN partition=%s entries=%d negative_entries=%d slots=%d"
          % (a.partition, n_get, n_sget, a.slots), file=sys.stderr)
    return 0


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


def cmd_promote(argv):
    if len(argv) != 1:
        raise SystemExit("usage: bincache.py promote <leg out>/remote/bincache")
    d = Path(argv[0])
    up = d / "uploads.tsv"
    if not up.is_file():
        print("BINCACHE PROMOTED 0 (no uploads.tsv)")
        return 0
    creds = creds_from_env()
    ok = bad = 0
    for row in up.read_text().splitlines():
        if not row.strip():
            continue
        try:
            leg, slot, dest, key = check_upload_row(row, d / "keys")
        except (ValueError, OSError) as exc:
            print("  refused row: %s" % exc)
            bad += 1
            continue
        src = "%s/%s/%s.tar.gz" % (INBOX_PREFIX, leg, slot)
        try:
            r2_request(creds, "PUT", dest, headers={
                "x-amz-copy-source": "/%s/%s" % (creds["R2_BUCKET"], urllib.parse.quote(src, safe="/"))})
            r2_request(creds, "DELETE", src)
            ok += 1
            print("  promoted %s" % dest)
        except Exception as exc:
            print("  promote failed %s: %s" % (dest, type(exc).__name__))
            bad += 1
    print("BINCACHE PROMOTED %d refused_or_failed=%d" % (ok, bad))
    return 0


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
    fields = key_fields(repo, argv[0], argv[1:], environ, environ.get("MOJOLEARN_BINCACHE_IMAGE", ""),
                        force_tree=environ.get("MOJOLEARN_BINCACHE_SOURCE") == "tree")
    print(json.dumps(dict(key=key_of(fields), refusal=refusal(fields, environ), fields=fields), indent=1, sort_keys=True))
    return 0


def main(argv):
    cmds = {"build": cmd_build, "key": cmd_key, "plan": cmd_plan, "promote": cmd_promote,
            "selftest-r2": cmd_selftest_r2,
            "device-arch": lambda a: print(device_arch()) or 0}
    if not argv or argv[0] not in cmds:
        print(__doc__)
        return 2
    return cmds[argv[0]](argv[1:])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
