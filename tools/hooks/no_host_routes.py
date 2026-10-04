#!/usr/bin/env python3
"""Refuse CPU work on GPU installs. The rule is enforced in code, not in notes.

On a GPU install every fit, transform and predict runs only GPU work, in
parallel (Andrew, Oct 1-2 2026). The CPU is only for CPU-only installs,
MOJOLEARN_VENDOR=cpu verification digests and inference.

Modes:
    no_host_routes.py --tree REF [--baseline FILE]
        Scan EVERY file of REF that a GPU install runs. Fail on any finding
        that is not in the baseline, and on any baseline row that no longer
        matches (the baseline only shrinks). Prints the debt per class and
        per owner. This is what the pre-push hook and the GitHub check run.
    no_host_routes.py --prune-baseline [REF]
        Rewrite REF's baseline (default HEAD, the worktree file) without its
        stale rows, and flip in-flight rows whose code is now in the tree to
        debt. It never adds a row.
    no_host_routes.py --owed [REF]
        Print every finding of the owed rules (py-native-host, mojo-host-loop)
        in REF as TSV. These never enter the baseline: --tree reports them as
        owed and refuses only the ones a tree adds over the owed anchor and main.
    no_host_routes.py <base> <tip> [--main REF]
        Diff mode: only the lines <tip> adds over merge-base(main, tip).
    no_host_routes.py --diff FILE
        Diff mode on a unified diff (no scope: every path is GPU code).
Exit 1 with one line per finding, 0 when clean, 2 on usage errors.

Which files count as GPU code is decided by imports, not by file names:
  * Mojo: every module that a GPU binding (the entry file of a non-host
    bindings/build_*.sh, or bindings/build.sh) reaches through its imports,
    wherever it lives (checks/, host/, *host*, *multi_gpu* all count).
    A host binding's own modules that no GPU binding imports are out of scope.
  * Python: every module of python/mojolearn/ except tests/ and the fixed
    list _PY_CPU_SIDE (verification, the CPU inference product, the CPU
    backend). That list is code: adding to it is a reviewed change.

Baseline: tools/hooks/host_routes_baseline.tsv, one row per finding key
(rule, path, normalized line text, occurrence), with its class, owner and
state. `debt` rows must match the tree. `inflight@<PR head>` rows (the
replacement of _IN_FLIGHT) pre-authorize the lines one open PR adds, and only
for a tree that contains that PR head; once that code is in a tree that
carries the baseline the row must become `debt` (--prune-baseline does it).
Nothing in this tool adds a row: the baseline was written once, from the
Oct 2 2026 audit, and every later change can only delete rows.
"""
import collections
import os
import re
import subprocess
import sys

BASELINE = "tools/hooks/host_routes_baseline.tsv"

# ---------------------------------------------------------------- scope ----

# Python modules that legitimately run on the CPU: verification, the CPU
# backend loader and the CPU inference product. Exact file names, not a
# pattern: a new CPU-side module is added here in review.
_PY_CPU_SIDE = frozenset("""
__main__.py _backend.py host_surface.py _cpu_reference.py _conformance.py
_verify.py _verify_all.py _verify_causal_lm.py _verify_distributed.py
_verify_par.py _verify_parallel_cv.py _verify_reference.py _verify_resources.py
_verify_small.py _verify_worker.py _verification_catalog.py
_verification_coverage.py _verification_ctr_models.py
_verification_evidence_data.py _verification_profiles.py
_forest_host.py _gbdt_host.py _byte_lm_host.py _byte_lm_trainer_host.py
_classical_host.py _serialize.py
neural_inference.py _causal_lm_fixtures.py
""".split())
# neural_inference.py: the public CPU inference product.
# _causal_lm_fixtures.py: verification fixtures (_verify_causal_lm and tests only).
# _serialize.py: model save/load (no fit, transform or predict).

# The host thread pool's own implementation: its contents are the CPU-only
# executor. Every USE of it in GPU code is still a finding.
_INFRA = frozenset(["core/host_parallel.mojo"])

# Route switches and thresholds whose names do not say HOST. Each picks a
# CPU route (or keeps the GPU route off) on a GPU install.
_KNOWN_ROUTE_SWITCHES = (
    "MOJOLEARN_PR_SPARSE", "MOJOLEARN_GBDT_ROUTE", "MOJOLEARN_X_LINEAR_DEVICE",
    "MOJOLEARN_XN_CC_GPU", "MOJOLEARN_SGD_PER_SAMPLE", "MOJOLEARN_XD_RES_DEV_MIN",
    "MOJOLEARN_XN_SERIAL_GPU", "MOJOLEARN_MSEL3", "MOJOLEARN_GBDT_RESIDENT",
    "MOJOLEARN_HOTPATH", "MOJOLEARN_XD_QR_HOST", "MOJOLEARN_XD_LU_SOLVE_HOST",
)
_KNOWN_ROUTE_THRESHOLDS = (
    "XD_RES_DEV_MIN", "HOST_FOLD_MIN", "XD_QR_HOST_MIN", "XD_LU_SOLVE_HOST_MIN",
)

# Env names of the host binding's own plumbing, not routes.
_HOST_PLUMBING = re.compile(
    r"MOJOLEARN_(\w+_)?HOST_(DIR|ALLOW_SABOTAGE|BINARY|BLOCK_TIMING|SABOTAGE)\b"
    r"|MOJOLEARN_HOST_SABOTAGE\b"
)
_DEFINES = re.compile(r"^\s*(struct|trait|comptime|alias|def|fn|class)\s+(\w+)")

# ---------------------------------------------------------------- rules ----
# (rule, class, applies-to {"mojo","py"}, pattern, why)
_LINE_RULES = [
    ("host-module", "host-import", {"mojo", "py"},
     re.compile(r"\b(load_host_module|host_module_path)\s*\("),
     "loads a host binding from GPU code"),
    ("host-exec", "hostexec", {"mojo"},
     re.compile(r"\bHostExec\b"),
     "runs HostExec in GPU code"),
    ("host-import", "host-import", {"mojo", "py"},
     re.compile(r"\b\w+(_on_host|_host_rows)\s*\("),
     "pulls a host walk into GPU code"),
    ("py-host-import", "host-import", {"py"},
     re.compile(r"^\s*from\s+\.*[\w.]*_host\b[\w.]*\s+import\b|^\s*import\s+[\w.]*_host\b"),
     "imports a CPU-side module into GPU-path Python"),
    ("host-env", "host-env/threshold", {"mojo", "py"},
     re.compile(r"MOJOLEARN_\w*HOST\w*"),
     "a MOJOLEARN_*HOST* switch (a host route knob)"),
    ("host-threshold", "host-env/threshold", {"mojo", "py"},
     re.compile(r"\b\w*HOST\w*_(MIN|MAX)(_\w+)?\b|\bHOST_(MIN|MAX)\b"),
     "a size threshold that picks the host"),
    ("route-switch", "host-env/threshold", {"mojo", "py"},
     re.compile(r"\b(" + "|".join(_KNOWN_ROUTE_SWITCHES + _KNOWN_ROUTE_THRESHOLDS) + r")\b"),
     "a known route switch or threshold (its name does not say HOST)"),
    ("host-threads", "host-threads", {"mojo", "py"},
     re.compile(r"\bhost_parallel\w*\b"),
     "runs a step on CPU threads in GPU code"),
    ("std-parallelize", "host-threads", {"mojo"},
     re.compile(r"\b(sync_)?parallelize\b|\bfrom\s+algorithm(\.functional)?\s+import\b.*\bparallelize"),
     "stdlib parallelize: CPU threads in GPU code"),
    ("route-table", "host-env/threshold", {"mojo", "py"},
     re.compile(r"\b(_HOST_ALGOS|_HOST_ROUTE_\w+|_HOST_ONE_BORDER_\w+|_glm_host"
                r"|_small_pool_host|_fit_on_host|_sgd_on_host|HOST_RUN)\b"),
     "a named host route"),
    ("py-numpy", "py-numpy", {"py"},
     re.compile(r"\bnp\.(argsort|unique|cumsum|bincount|sort|lexsort|searchsorted)\s*\("),
     "numpy work over n in a GPU-path module"),
    ("py-sklearn", "py-numpy", {"py"},
     re.compile(r"^\s*(import\s+sklearn\b|from\s+sklearn\b)"),
     "sklearn in a GPU-path module"),
    ("py-loop", "py-numpy", {"py"},
     re.compile(r"\bfor\s+.+?\s+in\s+range\(\s*(n|n_(samples|rows|obs|nodes|points|q|test|train|edges))\s*\)"
                r"|\bfor\s+.+?\s+in\s+range\(\s*(len\(\s*(X|y|x|rows|codes|labels|idx|indices|data|y_true"
                r"|y_pred|sample_weight|weights|w|targets|points|samples)\s*\)|\w+\.shape\[0\]|self\.n_samples)\s*\)"),
     "a Python loop over n in a GPU-path module"),
    ("py-threads", "host-threads", {"py"},
     re.compile(r"\b(concurrent\.futures|ThreadPoolExecutor|ProcessPoolExecutor|multiprocessing"
                r"|threading\.Thread)\b"),
     "a Python thread or process pool in a GPU-path module"),
    ("py-vendor-cpu", "host-env/threshold", {"py"},
     re.compile(r"environ\s*\[\s*['\"]MOJOLEARN_VENDOR['\"]\s*\]\s*=(?!=)"
                r"|environ\.setdefault\(\s*['\"]MOJOLEARN_VENDOR"
                r"|putenv\(\s*['\"]MOJOLEARN_VENDOR"),
     "forces the CPU vendor from a GPU-path module"),
]

_RULE_CLASS = {r[0]: r[1] for r in _LINE_RULES}
_RULE_CLASS.update({"host-call": "host-import", "serial-launch": "serial-gpu", "host-switch": "host-env/threshold",
                    "d2h-loop": "d2h-roundtrip", "tid0-loop": "serial-gpu",
                    "d2h-host-work": "d2h-roundtrip", "one-block-n": "serial-gpu",
                    "block-per-n": "serial-gpu",
                    "py-data-loop": "py-compute", "py-np-compute": "py-compute",
                    "py-reduce": "py-compute", "py-array-method": "py-compute",
                    "py-native-host": "native-host", "mojo-host-loop": "host-loop"})
_RULE_WHY = {r[0]: r[4] for r in _LINE_RULES}
_RULE_WHY.update({
    "host-call": "calls a host-only module from GPU code",
    "host-switch": "a switch or threshold in a function that calls host code",
    "d2h-loop": "downloads, synchronizes, loops on the host, then goes back to the device",
    "tid0-loop": "one thread loops over a runtime n",
    "d2h-host-work": "copies device data to the host, then computes on it in a host loop over a data size",
    "one-block-n": "a one-block launch (grid_dim=(1, 1, 1)) over a runtime size",
    "block-per-n": "one block per class / output / problem (a small grid) walking a runtime size",
    "serial-launch": "a one-block or one-thread launch over a runtime size",
    "py-data-loop": "a Python loop or comprehension at runtime (glue only: mark a scalar or argument loop `# glue: <reason>`)",
    "py-np-compute": "numpy compute at runtime in GPU-path Python (move it into Mojo)",
    "py-reduce": "sorted()/sum()/min()/max() over a sequence at runtime in GPU-path Python",
    "py-array-method": "an array reduction or sort method at runtime in GPU-path Python",
    "py-native-host": "GPU-path Python calls a native host helper (a binding export that loops on the CPU)",
    "mojo-host-loop": "a host loop over a data size in GPU Mojo code (move it into a kernel)",
})

# ------------------------------------------------- owed (never baseline) ----
# The re-audit of 2026-10-04 (~/mojolearn-evidence/cpu-reaudit-2026-10-04.md)
# found CPU work the rules above never saw: native host helpers that GPU-path
# Python calls (binding exports whose body loops on the CPU), and host loops
# over data inside GPU Mojo code. These rules are OWED, not accepted debt:
# their findings never enter the baseline (load_baseline refuses such a row).
# A tree is charged only for what it adds over the owed anchor (the audited
# integration head) and over main; what both already carry is printed as owed
# on every run, so it stays visible until a fix lane removes it.
_OWED_RULES = ("py-native-host", "mojo-host-loop")
_OWED_ANCHOR = "1a252d6d48a3ec624e31943d1e6c34f4846ba17b"
# Host exports whose loops walk arguments, handles, shapes or a saved model,
# never rows, features, classes or tokens (reviewed, like _PY_CPU_SIDE): a
# call to one is glue. Adding a name is a reviewed change.
_NATIVE_GLUE = frozenset("""
accumulation_is_aligned byte_lm_attention_estash_gate byte_lm_attention_memory_profile
byte_lm_config_profile byte_lm_model_pool_open byte_lm_model_pool_ownership byte_lm_offload_open
byte_lm_parallel_ownership byte_lm_session_info byte_lm_step_glue_arm gbdt_model_dim
gbdt_resident_info gp_log64 gp_theta_params iforest_resident_release x_cnn_conv_shape
x_cnn_pool_shape forest_export forest_export_legacy
""".split())
# A GPU-path Python line that runs on an explicit CPU-only route (a
# CPU-only install, file I/O, a user callable) says so:
#     # cpu-route: <reason of 3+ words>
_CPU_ROUTE = re.compile(r"#\s*cpu-route:\s*(\S+\s+){2,}\S+")
# A host Mojo loop over a size that is d- or k-sized, never rows, says so on
# the `for` line, naming the bound (as small-launch does for launches):
#     # small-loop(<bound>: <what it counts>): <why it is bounded>
_SMALL_LOOP = re.compile(r"#\s*small-loop\(\s*([^:()]+?)\s*:\s*([^()]+?)\s*\)\s*:\s*(\S+\s+){2,}\S+")

# --------------------------------------------------------------- git io ----


def _git(*args, inp=None):
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          errors="replace", input=inp)


_BLOB_CACHE = {}
_LINES_CACHE = {}


def _ls(ref):
    """{path: blob sha} of every file at ref."""
    r = _git("ls-tree", "-r", "--full-tree", ref)
    if r.returncode != 0:
        raise SystemExit(f"no_host_routes: cannot list {ref}: {r.stderr.strip()}")
    out = {}
    for ln in r.stdout.splitlines():
        meta, _, path = ln.partition("\t")
        bits = meta.split()
        if len(bits) == 3 and bits[1] == "blob":
            out[path] = bits[2]
    return out


def _read_blobs(shas):
    """{sha: text}, one `git cat-file --batch` process for the uncached ones."""
    need = [s for s in dict.fromkeys(shas) if s not in _BLOB_CACHE]
    if need:
        p = subprocess.run(["git", "cat-file", "--batch"], input="".join(s + "\n" for s in need).encode(),
                           capture_output=True)
        out, i = p.stdout, 0
        for sha in need:
            nl = out.index(b"\n", i)
            head = out[i:nl].split()
            i = nl + 1
            if len(head) < 3 or head[1] == b"missing":
                _BLOB_CACHE[sha] = ""
                continue
            size = int(head[2])
            _BLOB_CACHE[sha] = out[i:i + size].decode("utf-8", "replace")
            i += size + 1
    return {s: _BLOB_CACHE[s] for s in shas}


def _lines_of(sha, text, lang):
    k = (sha, lang)
    if sha is None or k not in _LINES_CACHE:
        v = _code_lines(text, lang)
        if sha is None:
            return v
        _LINES_CACHE[k] = v
    return _LINES_CACHE[k]


# -------------------------------------------------------------- parsing ----


def _code_lines(text, lang):
    """[(line_no, code_text)] with comments and docstrings blanked."""
    out = []
    in_doc = None
    for no, raw in enumerate(text.splitlines(), 1):
        line = raw
        if in_doc:
            j = line.find(in_doc)
            if j < 0:
                continue
            line = line[j + 3:]
            in_doc = None
        # strip docstrings opening (and maybe closing) on this line
        while True:
            m = re.search(r'("""|\'\'\')', line)
            if not m:
                break
            q = m.group(1)
            k = line.find(q, m.end())
            if k < 0:
                in_doc = q
                line = line[:m.start()]
                break
            line = line[:m.start()] + line[k + 3:]
        s = line.lstrip()
        if not s or s.startswith("#") or s.startswith("//"):
            continue
        out.append((no, line.rstrip()))
    return out


_IMPORT_FROM = re.compile(r"^\s*from\s+(\.*[\w.]*)\s+import\s+(.*)$")
_IMPORT = re.compile(r"^\s*import\s+(.*)$")


def _imports(lines):
    """[(line_no, module, [(name, alias)])] for each import statement,
    joining parenthesized continuation lines."""
    res = []
    i = 0
    while i < len(lines):
        no, ln = lines[i]
        m = _IMPORT_FROM.match(ln)
        if m:
            mod, rest = m.group(1), m.group(2)
            if "(" in rest and ")" not in rest:
                j = i + 1
                while j < len(lines) and ")" not in lines[j][1]:
                    rest += " " + lines[j][1]
                    j += 1
                if j < len(lines):
                    rest += " " + lines[j][1]
                i = j
            names = []
            for part in rest.replace("(", " ").replace(")", " ").split(","):
                part = part.split("#")[0].strip()
                if not part:
                    continue
                bits = part.split()
                name = bits[0]
                alias = bits[2] if len(bits) >= 3 and bits[1] == "as" else name
                names.append((name, alias))
            res.append((no, mod, names))
        else:
            m = _IMPORT.match(ln)
            if m:
                for part in m.group(1).split(","):
                    bits = part.strip().split()
                    if bits:
                        alias = bits[2] if len(bits) >= 3 and bits[1] == "as" else bits[0]
                        res.append((no, bits[0], [("", alias)]))
        i += 1
    return res


def _resolve(mod, src, files):
    """Repo paths of a Mojo import's module (and the submodule a name may
    be), searching the package root and bindings/ (the build's -I paths)."""
    if mod.startswith("."):
        dots = len(mod) - len(mod.lstrip("."))
        base = os.path.dirname(src)
        for _ in range(dots - 1):
            base = os.path.dirname(base)
        rel = mod.lstrip(".").replace(".", "/")
        cands = [os.path.join(base, rel) if rel else base]
    else:
        rel = mod.replace(".", "/")
        cands = [rel, "bindings/" + rel]
    out = []
    for c in cands:
        c = os.path.normpath(c)
        for p in (c + ".mojo", c + "/__init__.mojo"):
            if p in files:
                out.append(p)
        if out:
            # the packages along the path load too
            parts = c.split("/")
            for k in range(1, len(parts)):
                ini = "/".join(parts[:k]) + "/__init__.mojo"
                if ini in files:
                    out.append(ini)
            return out, c
    return out, None

# ---------------------------------------------------------------- scope ----


def _gpu_roots(ref_files, texts_of):
    """Entry files of the GPU bindings, read from the build scripts."""
    scripts = [p for p in ref_files if re.fullmatch(r"bindings/build(_\w+)?\.sh", p)
               and not p.endswith("_host.sh") and p != "bindings/build_host_family.sh"]
    roots = set()
    for p, t in texts_of(scripts).items():
        for m in re.finditer(r"bindings/(_mojolearn\w*)\.mojo", t):
            q = f"bindings/{m.group(1)}.mojo"
            if not q.endswith("_host.mojo") and q in ref_files:
                roots.add(q)
    return roots


class Tree:
    """Every scoped file of a ref with its code lines, imports and scope."""

    def __init__(self, ref, overlay=None):
        self.ref = ref
        shas = _ls(ref)
        overlay = overlay or {}
        files = set(shas) | set(overlay)
        self.files = files

        def lines(paths, lang):
            got = _read_blobs([shas[p] for p in paths if p not in overlay])
            res = {}
            for p in paths:
                if p in overlay:
                    res[p] = _lines_of(None, overlay[p], lang)
                else:
                    res[p] = _lines_of(shas[p], got[shas[p]], lang)
            return res

        mojo = sorted(p for p in files if p.endswith(".mojo"))
        self.lines = lines(mojo, "mojo")
        self._sha = {p: (None if p in overlay else shas.get(p)) for p in files}
        self.imps = {}
        for p, l in self.lines.items():
            k = ("imps", self._sha[p])
            if self._sha[p] is None or k not in _LINES_CACHE:
                v = _imports(l)
                if self._sha[p] is None:
                    self.imps[p] = v
                    continue
                _LINES_CACHE[k] = v
            self.imps[p] = _LINES_CACHE[k]
        # module graph
        self.edges = collections.defaultdict(set)
        self.sym_src = {}  # (path, alias) -> (module path it came from, name)
        for p, imps in self.imps.items():
            for _, mod, names in imps:
                mods, base = _resolve(mod, p, files)
                for m in mods:
                    self.edges[p].add(m)
                for name, alias in names:
                    if base is not None and name:
                        sub, _ = _resolve(mod + name if mod.endswith(".") else mod + "." + name, p, files)
                        for s in sub:
                            self.edges[p].add(s)
                    if mods:
                        self.sym_src[(p, alias)] = (mods[0], name)
        scripts = [p for p in files if re.fullmatch(r"bindings/build(_\w+)?\.sh", p)]
        stext = {p: overlay[p] if p in overlay else _read_blobs([shas[p]])[shas[p]] for p in scripts}
        roots = _gpu_roots(files, lambda ps: {p: stext[p] for p in ps if p in stext})
        seen, stack = set(), list(roots)
        while stack:
            p = stack.pop()
            if p in seen:
                continue
            seen.add(p)
            stack.extend(self.edges.get(p, ()))
        self.gpu_mojo = seen
        self.roots = roots
        py = sorted(p for p in files if p.startswith("python/mojolearn/") and p.endswith(".py")
                    and "/tests/" not in p and os.path.basename(p) not in _PY_CPU_SIDE)
        self.lines.update(lines(py, "py"))
        self.gpu_py = set(py)
        self.host_funcs = {}
        for p in self.gpu_mojo:
            k = ("hf", p, self._sha.get(p))
            if self._sha.get(p) is None or k not in _LINES_CACHE:
                v = self._host_functions(p)
                if self._sha.get(p) is None:
                    self.host_funcs[p] = v
                    continue
                _LINES_CACHE[k] = v
            self.host_funcs[p] = _LINES_CACHE[k]

    _DEVICE_API = re.compile(r"\b(enqueue_function\w*|DeviceContext|DeviceBuffer|enqueue_copy"
                             r"|thread_idx|block_idx|global_idx|barrier)\b")
    _HOST_EXEC = re.compile(r"\bhost_parallel\w*\s*[\[(]|\bHostExec\b|\b(sync_)?parallelize\s*\["
                            r"|\b\w+(_on_host|_host_rows)\s*\(")
    _HOST_NAMED = re.compile(r"(^|/)host/|(^|/)[^/]*_host\.mojo$|(^|/)host_[^/]*\.mojo$")

    def _host_functions(self, p):
        """Top-level functions of p that run CPU work: their body runs host
        threads or HostExec (or a *_on_host / *_host_rows walk), or (in a
        module with no device code) calls such a function of the same module,
        or, in a module laid out as host code
        with no device code at all, loops."""
        lines = self.lines.get(p, [])
        named = bool(self._HOST_NAMED.search(p)) and not any(
            self._DEVICE_API.search(t) for _, t in lines)
        drivers = _driver_lines(lines)
        bodies = {}
        for a, b in _functions(lines, top_only=True):
            m = re.match(r"\s*(?:def|fn)\s+(\w+)", lines[a][1])
            if m:
                bodies[m.group(1)] = lines[a + 1:b]
        out = set()
        for name, body in bodies.items():
            if any(self._HOST_EXEC.search(t) and no not in drivers for no, t in body) or (
                    named and any(re.match(r"\s*(for|while)\s", t) for _, t in body)):
                out.add(name)
        # same-module callers count only in a module with no device code
        # (a host walk's entry points), not in a device module whose staging
        # helper happens to use host threads
        device_module = any(self._DEVICE_API.search(t) for _, t in lines)
        while not device_module:
            if not out:
                break
            call = re.compile(r"\b(" + "|".join(map(re.escape, sorted(out))) + r")\s*[\[(]")
            more = {n for n, body in bodies.items() if n not in out
                    and any(call.search(t) for _, t in body)}
            if not more:
                break
            out |= more
        return out

    def scoped(self):
        return sorted((self.gpu_mojo | self.gpu_py) - _INFRA)

    # ------------------------------------------- native host helpers ----
    _REGISTER = re.compile(r"def_function\[\s*([\w.]+)[^\]]*\]\s*\(\s*\"(\w+)\"")
    # device work in a function body: a context, a buffer, a launch, a
    # device-side helper call (`device_x(`, `x_device(`) or a kernel index
    _ON_DEVICE = re.compile(r"process_ctx|DeviceContext|DeviceBuffer|enqueue_\w+|\bctx\b"
                            r"|\bdevice_\w+\s*\(|\w+_device\w*\s*\(|thread_idx|block_idx")
    _LOOP = re.compile(r"^\s*for\s+\w+\s+in\s+range\(([^)]*)\)|^\s*while\b")
    _CALLEE = re.compile(r"\b([A-Za-z_]\w*)\s*[\[(]")

    def _top_bodies(self, p):
        k = ("tb", p)
        if k not in self.__dict__.setdefault("_tb", {}):
            ls = self.lines.get(p, [])
            out = {}
            for a, b in _functions(ls, top_only=True):
                m = re.match(r"\s*(?:def|fn)\s+(\w+)", ls[a][1])
                if m:
                    out[m.group(1)] = ls[a:b]
            self._tb[k] = out
        return self._tb[k]

    def _resolve_fn(self, p, f, depth=0):
        """(module, name) where function f, as named in module p, is defined."""
        if f in self._top_bodies(p):
            return p, f
        s = self.sym_src.get((p, f))
        if s and s[1] and s[0] != p and depth < 6:
            return self._resolve_fn(s[0], s[1], depth + 1)
        return None, None

    def _closure(self, p, f, limit=5):
        """(module, name) of f and every function it calls, transitively."""
        seen, stack = set(), [(p, f, 0)]
        while stack:
            q, g, d = stack.pop()
            if (q, g) in seen or d > limit:
                continue
            seen.add((q, g))
            for _, t in self._top_bodies(q).get(g, [])[1:]:
                for m in self._CALLEE.finditer(t):
                    r = self._resolve_fn(q, m.group(1))
                    if r[0] is not None and r not in seen:
                        stack.append((r[0], r[1], d + 1))
        return seen

    def host_exports(self):
        """{export name: (module, function)} of the GPU bindings' Python
        exports that run CPU work: no function they reach does device work,
        and one of them loops over a runtime bound. Each is a native host
        helper; a call from GPU-path Python is CPU work on the GPU route."""
        if hasattr(self, "_host_exports"):
            return self._host_exports
        out = {}
        impl = set()
        for p in sorted(self.gpu_mojo):
            if "def_function" not in "".join(t for _, t in self.lines.get(p, [])):
                continue
            txt = " ".join(t for _, t in self.lines.get(p, []))
            for m in self._REGISTER.finditer(txt):
                q, g = self._resolve_fn(p, m.group(1))
                if q is None:
                    continue
                fns = self._closure(q, g)
                bodies = [self._top_bodies(a).get(b, [])[1:] for a, b in fns]
                if any(self._ON_DEVICE.search(t) for body in bodies for _, t in body):
                    continue
                loops = False
                for body in bodies:
                    for _, t in body:
                        lm = self._LOOP.match(t)
                        if lm and (lm.group(1) is None or _runtime_bound(lm.group(1))):
                            loops = True
                            break
                    if loops:
                        break
                if loops:
                    out[m.group(2)] = (q, g)
                    impl |= fns
        self._host_exports = out
        # a function imported under a host-column alias (`f as host_f`: the
        # host fallback of a device export, its _OFF arm) is the host column,
        # with everything it calls
        for (p, alias), (src, name) in sorted(self.sym_src.items()):
            if name and src in self.gpu_mojo and _HOST_COLUMN_FN.search(alias) \
                    and name in self._top_bodies(src):
                impl |= self._closure(src, name)
        # functions that only serve a host export: their loops are charged
        # at the Python call site, once
        self._export_impl = impl
        return out

    def export_impl(self):
        self.host_exports()
        return self._export_impl

# -------------------------------------------------------------- scanning ----


def _functions(lines, top_only=False):
    """Split code lines into outermost function units: [(start_idx, end_idx)].
    top_only: only functions at column 0 (not struct methods)."""
    units, cur, ind = [], None, None
    for i, (_, t) in enumerate(lines):
        indent = len(t) - len(t.lstrip())
        m = re.match(r"\s*(def|fn)\s", t) if not top_only else re.match(r"(def|fn)\s", t)
        if cur is not None and indent <= ind and not t.strip().startswith(("@", ")", "]")):
            units.append((cur, i))
            cur = None
        if cur is None and m:
            cur, ind = i, indent
    if cur is not None:
        units.append((cur, len(lines)))
    return units


_D2H = re.compile(r"enqueue_copy\(\s*dst_ptr\s*=|\bmap_to_host\b|enqueue_copy_from_device|\bto_host\(")
_DEV_AGAIN = re.compile(r"enqueue_copy\(\s*dst_buf\s*=|enqueue_function|enqueue_copy_to_device|\bupload\w*\(")
_SYNC = re.compile(r"\.synchronize\(\)")
_HOST_FOR = re.compile(r"^\s*for\s+\w+\s+in\s+range\(")
_TID0 = re.compile(r"^\s*if\s+\(?\s*(thread_idx\.x|tid|lane|lane_id|t|local_tid|thread_id)\s*==\s*0\b")
_RUNTIME_RANGE = re.compile(r"\bfor\s+\w+\s+in\s+range\(\s*([^)]*)\)")
_ENV_READ = re.compile(r"\b(getenv|_getenv\w*|environ\.get|is_defined)\b[\[(]\s*['\"](MOJOLEARN_\w+)")
# a size test against a constant: `n < SMALL_N`, `rows >= 4096`, `LIMIT > m`
_SZ = r"(n|m|n_\w+|rows|cols|count|size|numel|nnz|cells|total|\w+_rows|\w+_cells|\w+_size|\w+_count)"
_THRESH_CMP = re.compile(r"\b" + _SZ + r"(\s*[*]\s*\w+)?\s*[<>]=?\s*([A-Z][A-Z0-9_]{2,}|\d{2,}[\d_]*|1\s*<<\s*\d+)\b"
                         r"|\b([A-Z][A-Z0-9_]{2,}|\d{2,}[\d_]*)\s*[<>]=?\s*" + _SZ + r"\b")


def _runtime_bound(expr):
    """True when a range() bound is a runtime size, not a literal/constant."""
    e = expr.split(",")[-1 if expr.count(",") >= 1 else 0].strip()
    if not e:
        return False
    if re.fullmatch(r"[\d_]+|[A-Z][A-Z0-9_]*|[\d_]+\s*[-+*]\s*[\d_]+", e):
        return False
    return True


_DRIVE = re.compile(r"\bDeviceContext\s*\(|\.enqueue_\w+|\bctx\.synchronize\(")


def _driver_lines(lines):
    """Line numbers of host_parallel* calls whose task closure drives a GPU
    (one host thread per device, each enqueueing on its own context): the
    multi-GPU dispatch Andrew allows. Decided by the closure's code."""
    out = set()
    for i, (no, t) in enumerate(lines):
        m = re.search(r"\bhost_parallel\w*\s*(?:\[[^\]]*\])?\s*\(\s*(\w+)", t)
        if not m:
            continue
        name = m.group(1)
        for j in range(i - 1, -1, -1):
            dm = re.match(r"(\s*)(?:def|fn)\s+" + re.escape(name) + r"\b", lines[j][1])
            if dm:
                ind = len(dm.group(1))
                body = []
                for _, t2 in lines[j + 1:i]:
                    if len(t2) - len(t2.lstrip()) <= ind:
                        break
                    body.append(t2)
                if any(_DRIVE.search(b) for b in body):
                    out.add(no)
                break
    return out


def _launch_statements(lines):
    """[(line_idx, joined text)] of each enqueue_function call."""
    out = []
    for i, (_, t) in enumerate(lines):
        if "enqueue_function" not in t:
            continue
        txt, depth, j = "", 0, i
        started = False
        while j < len(lines):
            seg = lines[j][1]
            if j > i:
                txt += " "
            txt += seg.strip()
            depth += seg.count("(") - seg.count(")")
            started = started or "(" in seg
            if started and depth <= 0:
                break
            j += 1
            if j - i > 40:
                break
        out.append((i, txt))
    return out


_GRID1 = re.compile(r"\bgrid_dim\s*=\s*(1\b(?!\s*[.\w])|\(\s*1\s*,\s*1\s*\)|\(\s*1\s*\))")
# THE REVIEWED EXEMPTION. A one-block or one-thread launch (or an
# `if tid == 0:` walk) that works over d- or k-sized data, never rows, says so
# on the launch statement (or the `if` line) with a trailing comment
#     # small-launch(<arg>: <what it counts>): <why it is bounded>
# naming the launch argument that bounds the work and what it counts. A
# reviewer reads the annotation in the diff; the checker requires the named
# argument to appear in the launch and a reason of at least a few words, so a
# lane annotates instead of renaming arguments to dodge the rule.
_SMALL_LAUNCH = re.compile(r"#\s*small-launch\(\s*([^:()]+?)\s*:\s*([^()]+?)\s*\)\s*:\s*(.+)$")


def _small_launch_ok(txt, first=None):
    """`first`: the statement's first physical line, which carries the note
    (the note runs to the end of that line); `txt` the joined statement."""
    first = txt if first is None else first
    m = _SMALL_LAUNCH.search(first)
    if not m:
        return False
    arg, what, why = m.group(1), m.group(2), m.group(3)
    code = txt.replace(first[m.start():].strip(), " ")
    # the named argument appears in the code, not only in the note
    is_if = code.lstrip().startswith("if ")
    if not is_if and not re.search(r"\b" + re.escape(arg) + r"\b", code):
        return False
    return len(what.split()) >= 1 and len(why.split()) >= 3


_GRID1_3D = re.compile(r"\bgrid_dim\s*=\s*(\(\s*1\s*,\s*1\s*,\s*1\s*\)|Dim\(\s*1\s*\))")
# a grid of one block per small count: classes, outputs, features, targets
_GRID_SMALL = re.compile(
    r"\bgrid_dim\s*=\s*\(?\s*(dims\.C|dims\.D|C|D|n_classes|n_class|num_classes|n_targets|n_outputs"
    r"|n_cols|n_features|d|k|cd|n_comp|n_components)\s*(,\s*1\s*,\s*1\s*\)|\)|,|$)")
_BLOCK1 = re.compile(r"\bblock_dim\s*=\s*(1\b(?!\s*[.\w])|\(\s*1\s*\))")
# a runtime size among the launch arguments (rows, samples, elements, nnz)
_SIZES = (r"(n|m|n_rows|rows|n_samples|nnz|n_nodes|n_points|n_q|numel|n_elems|n_obs|n_train"
          r"|n_test|n_items|n_edges|max_samples|r1|length|size|n_rows_total|n_cells)")
_SIZE_TOKEN = re.compile(r"\b" + _SIZES + r"\b")
# a launch argument that is a runtime size: `n` or `Int32(n)`, not `blocks(n)`
_SIZE_ARG = re.compile(r"^\s*(U?Int(8|16|32|64)?\s*\(\s*)?" + _SIZES + r"\s*\)?\s*$")


def _top_args(s):
    """Top-level comma-separated arguments of a call's argument text."""
    out, depth, cur = [], 0, ""
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
            if depth < 0:
                break
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


# any device->host copy: an explicit copy, a mapped buffer, or a download helper
_D2H_ANY = re.compile(r"enqueue_copy\(\s*dst_ptr\s*=|\bmap_to_host\b|enqueue_copy_from_device|\bto_host\w*\("
                      r"|\b\w*download\w*\s*\(|\b\w*readback\w*\s*\(")
_KERNEL_TOK = re.compile(r"\b(thread_idx|block_idx|global_idx|lane_id|warp_id)\b|\bblock_dim\.")
# a data-sized loop bound: rows, samples, queries, points, nonzeros, elements
_DATA_SIZES = re.compile(r"(?<![.\w])(" + _SIZES + r"|n_queries|n_cand|n_todo|n_pooled|n_resamples"
                         r"|n_vertices|total_labels|n_pairs|n_tokens|T)\b|\bself\.size\b")
# a loop body that computes (branches, folds, sorts), not a plain element copy
_HOST_WORK = re.compile(r"^\s*(if|elif|while)\b|[-+*/]=|\b(min|max|abs|sqrt|exp|log)\s*\("
                        r"|\bsort\w*\s*\(|\b_find\s*\(|\.insert\s*\(|[^<>=!]\s(<|>|<=|>=|==|!=)\s"
                        r"|\b(?P<acc>\w+)\s*=\s*\w*\(?\s*(?P=acc)\s*[-+*/]")
# debug-only blocks (identity traces, stage timers) are not the product path
_DEBUG_IF = re.compile(r"^\s*(comptime\s+)?(el)?if\b.*(\b(trace\w*\.enabled|_st_on|_trace\w*|verbose|debug\w*"
                       r"|timing\w*|phase_timing)\b|STAGE_TIMES|TIMERS?\b|TIMING\b)")
# a function that returns at once unless tracing is on is a trace recorder
_TRACE_ONLY = re.compile(r"^\s*if\s+not\s+\w*trace\w*\.enabled\s*:")


def _d2h_host_work(lines):
    """[(line_no, text)] of host `for` loops over a data size that compute on
    device data copied to the host earlier in the same (non-kernel) function."""
    out = []
    stack = []  # (indent, def index) of enclosing defs
    kern = {}
    owner = []
    for i, (_, t) in enumerate(lines):
        ind = len(t) - len(t.lstrip())
        while stack and ind <= stack[-1][0] and not t.strip().startswith((")", "]", "@")):
            stack.pop()
        if re.match(r"\s*(def|fn)\s", t):
            stack.append((ind, i))
            kern[i] = False
            owner.append(i)
            continue
        owner.append(stack[-1][1] if stack else None)
        if stack and _KERNEL_TOK.search(t):
            kern[stack[-1][1]] = True
    armed = set()
    ifs = []  # (indent, is_debug) of enclosing ifs
    for i, (no, t) in enumerate(lines):
        o = owner[i]
        ind = len(t) - len(t.lstrip())
        while ifs and ind <= ifs[-1][0] and not re.match(r"\s*(else|elif)\b", t):
            ifs.pop()
        if o is not None and _TRACE_ONLY.match(t):
            kern[o] = True  # not product work: skip the rest of the function
        if re.match(r"\s*(comptime\s+)?(el)?if\b|\s*else\b", t):
            if re.match(r"\s*(else|elif)\b", t) and ifs and ifs[-1][0] == ind:
                ifs.pop()
            ifs.append((ind, bool(_DEBUG_IF.match(t))))
        if o is None or kern.get(o) or re.match(r"\s*(def|fn)\s", t):
            continue
        if _D2H_ANY.search(t):
            armed.add(o)
            continue
        if o not in armed or any(d for _, d in ifs):
            continue
        m = _RUNTIME_RANGE.match(t.strip()) if t.lstrip().startswith("for ") else None
        if not m or not _runtime_bound(m.group(1)) or not _DATA_SIZES.search(m.group(1)):
            continue
        body = []
        for _, t2 in lines[i + 1:]:
            if len(t2) - len(t2.lstrip()) <= ind:
                break
            body.append(t2)
        if any(_HOST_WORK.search(b) for b in body):
            out.append((no, t))
    return out


dm_def = re.compile(r"\s*(def|fn)\s")
# a line none of the line rules can match is skipped quickly
_PREFILTER = re.compile(r"host|HOST|Host|parallel|MOJOLEARN_|_MIN|np\.|sklearn|range\(|futures|Executor"
                        r"|multiprocessing|threading")
_SCAN_CACHE = {}


def scan_file(tree, path):
    """[(rule, line_no, code_text)] for one scoped file."""
    lang = "py" if path.endswith(".py") else "mojo"
    lines = tree.lines.get(path, [])
    out = []
    host_thread_names = set()
    host_syms = set()
    import_of = {}
    if lang == "mojo":
        for no, mod, names in tree.imps.get(path, ()):
            for name, alias in names:
                leaf = name or mod.split(".")[-1]
                if leaf.startswith("host_parallel") or mod.endswith("host_parallel"):
                    host_thread_names.add(alias)
                    continue
                src = tree.sym_src.get((path, alias))
                if src and name and name in tree.host_funcs.get(src[0], ()):
                    host_syms.add(alias)
                    import_of.setdefault(no, []).append(f"{src[0]}:{name}")
    local_host = tree.host_funcs.get(path, set()) if lang == "mojo" else set()
    native = frozenset()
    impl = frozenset()
    if hasattr(tree, "host_exports"):
        if lang == "py":
            native = frozenset(n for n in tree.host_exports() if n not in _NATIVE_GLUE)
        else:
            impl = frozenset(f for q, f in tree.export_impl() if q == path)
    sha = getattr(tree, "_sha", {}).get(path)
    ck = (path, sha, frozenset(host_thread_names), frozenset(host_syms),
          tuple(sorted((k, tuple(v)) for k, v in import_of.items())), frozenset(local_host),
          native, impl)
    if sha is not None and ck in _SCAN_CACHE:
        return _SCAN_CACHE[ck]
    res = _scan_lines(lang, lines, host_thread_names, host_syms, import_of, local_host,
                      native=native, impl=impl, path=path)
    if sha is not None:
        _SCAN_CACHE[ck] = res
    return res


def _scan_lines(lang, lines, host_thread_names, host_syms, import_of, local_host,
                native=frozenset(), impl=frozenset(), path=""):
    out = []
    alias_re = (re.compile(r"\b(" + "|".join(map(re.escape, sorted(host_thread_names))) + r")\b")
                if host_thread_names else None)
    host_call_re = (re.compile(r"\b(" + "|".join(map(re.escape, sorted(host_syms))) + r")\s*[\[(]")
                    if host_syms else None)
    local_call_re = (re.compile(r"\b(" + "|".join(map(re.escape, sorted(local_host))) + r")\s*[\[(]")
                     if local_host else None)
    flagged = collections.defaultdict(set)
    drivers = _driver_lines(lines) if lang == "mojo" else set()
    uses = [no for no, t in lines if re.search(r"\bhost_parallel\w*\s*(\[[^\]]*\])?\s*\(", t)
            and not _DEFINES.match(t)] if lang == "mojo" else []
    only_drivers = bool(uses) and all(no in drivers for no in uses)

    def add(rule, no, t):
        if rule in ("host-threads",) and (no in drivers or (only_drivers and re.match(
                r"\s*(from|import)\b", t))):
            return
        if rule == "host-import" and dm_def.match(t):
            return
        if rule not in flagged[no]:
            out.append((rule, no, t))
            flagged[no].add(rule)

    for no, t in lines:
        dm = _DEFINES.match(t)
        for rule, _cls, langs, pat, _why in (_LINE_RULES if _PREFILTER.search(t) else ()):
            if lang not in langs:
                continue
            ms = [m.group(0) for m in pat.finditer(t)]
            if rule == "host-env":
                ms = [h for h in ms if not _HOST_PLUMBING.fullmatch(h)]
            if not ms:
                continue
            if dm and rule in ("host-exec", "host-threads", "std-parallelize") and \
                    re.match(r"(HostExec|host_parallel\w*|parallelize)$", dm.group(2)):
                continue
            add(rule, no, t)
        if alias_re and alias_re.search(t) and not (dm and alias_re.fullmatch(dm.group(2))):
            add("host-threads", no, t)
        if no in import_of:
            # one finding per imported host function, keyed on the symbol, so
            # dropping one name from an import line leaves the others' rows
            if not flagged[no] & {"host-threads", "host-import", "host-exec"}:
                for sym in sorted(import_of[no]):
                    out.append(("host-call", no, "import " + sym))
                flagged[no].add("host-call")
        elif host_call_re and host_call_re.search(t):
            if not flagged[no] & {"host-threads", "host-import", "host-exec"}:
                add("host-call", no, t)
    if lang != "mojo":
        for rule, no, t in _py_compute(lines):
            add(rule, no, t)
        for no, t in _py_native_host(lines, native):
            add("py-native-host", no, t)
        return out
    hostish = {"host-call", "host-import", "host-exec", "host-threads", "std-parallelize"}
    # host-switch: in a function that runs device code, an env/define read or
    # a size test against a constant whose branch reaches host code (a route)
    device_fn = set()
    for a, b in _functions(lines):
        if any(Tree._DEVICE_API.search(t) for _, t in lines[a:b]):
            device_fn.update(range(a, b))
    for k, (no, t) in enumerate(lines):
        if k not in device_fn:
            continue
        if not re.match(r"\s*(comptime\s+)?(if|elif)\b", t):
            continue
        if not (_ENV_READ.search(t) or _THRESH_CMP.search(t)) or "is_gpu()" in t:
            continue
        if flagged[no]:
            continue
        ind = len(t) - len(t.lstrip())
        reach = False
        for no2, t2 in lines[k + 1:]:
            ind2 = len(t2) - len(t2.lstrip())
            if ind2 < ind or (ind2 == ind and not re.match(r"\s*(else|elif)\b", t2)):
                break
            if flagged[no2] & hostish or (local_call_re and local_call_re.search(t2)):
                reach = True
                break
        if reach:
            add("host-switch", no, t)
    for a, b in _functions(lines):
        body = lines[a:b]
        # download -> synchronize -> host loop -> device again
        state, d2h = 0, None
        for no, t in body:
            if state == 0 and _D2H.search(t):
                state, d2h = 1, (no, t)
            elif state == 1 and _SYNC.search(t):
                state = 2
            elif state == 2 and _HOST_FOR.search(t) and _SIZE_TOKEN.search(t.split("range(", 1)[1]):
                state = 3
            elif state == 3 and _DEV_AGAIN.search(t):
                add("d2h-loop", d2h[0], d2h[1])
                state, d2h = 0, None
        # if tid == 0: for i in range(<runtime>)
        for k, (no, t) in enumerate(body):
            if not _TID0.match(t) or _small_launch_ok(t):
                continue
            ind = len(t) - len(t.lstrip())
            for no2, t2 in body[k + 1:]:
                if len(t2) - len(t2.lstrip()) <= ind:
                    break
                m = _RUNTIME_RANGE.search(t2)
                if m and _runtime_bound(m.group(1)) and _SIZE_TOKEN.search(m.group(1)):
                    add("tid0-loop", no, t)
                    break
    # download -> host loop over a data size that computes on the copy (the
    # result never needs to go back to the device: OPTICS, agglomerative and
    # HDBSCAN hid their host passes that way)
    d2h_lines = set()
    for no, t in _d2h_host_work(lines):
        add("d2h-host-work", no, t)
        d2h_lines.add(no)
    for no, t in _mojo_host_loops(lines, path, impl, d2h_lines):
        add("mojo-host-loop", no, t)
    # one-block / one-thread launches over a runtime size
    for i, txt in _launch_statements(lines):
        g1, b1 = _GRID1.search(txt), _BLOCK1.search(txt)
        if _small_launch_ok(txt, lines[i][1]):
            continue
        if not g1:
            # the 3-D spelling serial-launch's pattern never matched
            g3 = _GRID1_3D.search(txt)
            if g3:
                args = txt[:g3.start()]
                args = args[args.find("(", args.find("enqueue_function")) + 1:] if "(" in args else args
                if any(_SIZE_ARG.match(a) for a in _top_args(args)):
                    add("one-block-n", lines[i][0], lines[i][1])
                continue
            # one block per class / output cell / feature over all n rows:
            # the grid is a small count (classes, features, outputs) and a
            # launch argument is a runtime row count, so each block walks n
            gs = _GRID_SMALL.search(txt)
            if gs:
                args = txt[:gs.start()]
                args = args[args.find("(", args.find("enqueue_function")) + 1:] if "(" in args else args
                if any(_SIZE_ARG.match(a) for a in _top_args(args)):
                    add("block-per-n", lines[i][0], lines[i][1])
            continue
        args = txt[:g1.start()]
        args = args[args.find("(", args.find("enqueue_function")) + 1:] if "(" in args else args
        if any(_SIZE_ARG.match(a) for a in _top_args(args)):
            add("serial-launch", lines[i][0], lines[i][1])
    return out


# ------------------------------------------------- Python = glue only ----
# At runtime python/mojolearn only validates arguments, picks the binding,
# passes buffers and returns results (Andrew, Oct 3 2026). Inside a def, any
# loop or comprehension, numpy compute call, sorted()/sum()/one-argument
# min()/max() or array reduction method is a finding, unless the line carries
# a reviewed `# glue: <reason of 3+ words>` (a loop over arguments, kwargs,
# a handful of names, never over rows, features, classes or tokens).
_GLUE = re.compile(r"#\s*glue:\s*(\S+\s+){2,}\S+")
_PY_STR = re.compile(r"(?:[rbfuRBFU]{0,2})(\"[^\"\\\n]*(?:\\.[^\"\\\n]*)*\"|'[^'\\\n]*(?:\\.[^'\\\n]*)*')")
_PY_FOR = re.compile(r"(^\s*(async\s+)?for\s|[\[({,]\s*.*?\bfor\s+[\w\s,()*]+?\s+in\b|\S\s+for\s+[\w\s,()*]+?\s+in\b)")
# allocation, conversion, dtype and shape plumbing: not compute
_NP_GLUE = frozenset("""
asarray ascontiguousarray asfortranarray asanyarray array require frombuffer empty empty_like
zeros zeros_like ones ones_like full full_like dtype issubdtype iinfo finfo result_type can_cast
promote_types ndim shape size isscalar errstate reshape ravel atleast_1d atleast_2d broadcast_to
squeeze expand_dims moveaxis transpose ctypeslib dtypes integer floating number generic bool_
float32 float64 float16 int8 int16 int32 int64 uint8 uint16 uint32 uint64 intp uintp ndarray
str_ bytes_ object_ void nan inf pi newaxis random load save savez savez_compressed get_printoptions
set_printoptions printoptions shares_memory may_share_memory isfortran copyto
""".split())
_NP_CALL = re.compile(r"(?<![\w.])(?:np|numpy)\.([A-Za-z_][\w.]*)\s*\(")
_PY_REDUCE = re.compile(r"(?<![\w.])(sorted|sum|min|max)\s*\(")
_ARR_METHOD = re.compile(r"\.(sum|mean|std|var|argmin|argmax|argsort|cumsum|cumprod|dot|prod|nonzero"
                         r"|searchsorted|argpartition|partition|nansum|nanmean)\s*\(")


def _one_arg(s):
    """True when the call whose argument list starts at s has one positional argument."""
    depth, n, i = 0, 0, 0
    for i, c in enumerate(s):
        if c in "([{":
            depth += 1
        elif c in ")]}":
            if depth == 0:
                break
            depth -= 1
        elif c == "," and depth == 0:
            rest = s[i + 1:].lstrip()
            if not re.match(r"(key|default|start|reverse)\s*=", rest) and not rest.startswith(")"):
                n += 1
    return n == 0


def _py_compute(lines):
    out = []
    def_ind = []  # indents of enclosing defs
    for no, t in lines:
        ind = len(t) - len(t.lstrip())
        st = t.lstrip()
        while def_ind and ind <= def_ind[-1] and not st.startswith((")", "]", "}")):
            def_ind.pop()
        if re.match(r"(async\s+)?def\s", st):
            def_ind.append(ind)
            continue
        if not def_ind or _GLUE.search(t):
            continue
        code = _PY_STR.sub('""', t.split("  #", 1)[0] if "  #" in t else t)
        if _PY_FOR.search(code):
            out.append(("py-data-loop", no, t))
        for m in _NP_CALL.finditer(code):
            if m.group(1).split(".")[0] not in _NP_GLUE:
                out.append(("py-np-compute", no, t))
                break
        for m in _PY_REDUCE.finditer(code):
            if m.group(1) in ("sorted", "sum") or _one_arg(code[m.end():]):
                out.append(("py-reduce", no, t))
                break
        if _ARR_METHOD.search(code):
            out.append(("py-array-method", no, t))
    return out


_NATIVE_RE = {}


def _py_native_host(lines, names):
    """[(line_no, text)] of lines inside a def that name a native host helper:
    `.name` (a binding attribute) or the exact string "name" (`_native("name")`,
    `getattr(b, "name")`)."""
    if not names:
        return []
    if names not in _NATIVE_RE:
        alt = "|".join(sorted(map(re.escape, names), key=len, reverse=True))
        _NATIVE_RE[names] = (re.compile(r"\.(" + alt + r")\b"),
                             re.compile(r"(?<![\w\"'])[\"'](" + alt + r")[\"']"))
    attr, lit = _NATIVE_RE[names]
    out = []
    def_ind = []
    for no, t in lines:
        ind = len(t) - len(t.lstrip())
        st = t.lstrip()
        while def_ind and ind <= def_ind[-1] and not st.startswith((")", "]", "}")):
            def_ind.pop()
        if re.match(r"(async\s+)?def\s", st):
            def_ind.append(ind)
            continue
        if not def_ind or _GLUE.search(t) or _CPU_ROUTE.search(t):
            continue
        code = t.split("  #", 1)[0] if "  #" in t else t
        if attr.search(_PY_STR.sub('""', code)) or lit.search(code):
            out.append((no, t))
    return out


_DEVICE_MODULE = re.compile(r"\bDeviceContext\b|\bDeviceBuffer\b|\benqueue_\w+|\bprocess_ctx\b")
# host-column functions inside a GPU module (oracles, replays, references)
_HOST_COLUMN_FN = re.compile(r"(^|_)(host|oracle|reference|replay|cpu)(_|$)", re.I)


_DRIVES_DEVICE = re.compile(r"\benqueue_\w+|\bctx\b|\.synchronize\(|\blaunch\w*\s*[\[(]|\w+_launch\w*\s*[\[(]"
                            r"|\bdevice_\w+\s*[\[(]|\w+_(device|gpu|dev)\w*\s*[\[(]|\bDeviceBuffer\b")


def _mojo_host_loops(lines, path, impl=frozenset(), skip=frozenset()):
    """[(line_no, text)] of host `for` loops over a runtime data size whose body
    touches data (indexes it with the loop variable, appends, or computes), in
    a non-kernel function of a GPU binding (bindings/) or a module that drives
    the device. Skipped: kernels, the functions behind a native host helper
    (charged at the Python call site), host-column functions, check replays
    (checks/), trace-only and debug blocks, comptime loops, loops already
    charged as d2h-host-work, and loops marked `# small-loop(...)`."""
    if "/checks/" in "/" + path:
        return []
    if not (path.startswith("bindings/") or any(_DEVICE_MODULE.search(t) for _, t in lines)):
        return []
    # each line's innermost def; a def is skipped when it is a kernel, a
    # trace recorder, a host-column function, or (by itself or its top-level
    # function) the body of a native host helper
    stack, owner, top, skipdef = [], [], {}, {}
    for i, (_, t) in enumerate(lines):
        ind = len(t) - len(t.lstrip())
        while stack and ind <= stack[-1][0] and not t.strip().startswith((")", "]", "@")):
            stack.pop()
        dm = re.match(r"\s*(?:def|fn)\s+(\w+)", t)
        if dm:
            name = dm.group(1)
            outer = top[stack[0][1]] if stack else name
            top[i] = outer
            skipdef[i] = bool(name in impl or outer in impl or _HOST_COLUMN_FN.search(name)
                              or _HOST_COLUMN_FN.search(outer))
            stack.append((ind, i))
            owner.append(None)
            continue
        o = stack[-1][1] if stack else None
        owner.append(o)
        if o is not None and (_KERNEL_TOK.search(t) or _TRACE_ONLY.match(t)):
            skipdef[o] = True
    out = []
    ifs = []  # (indent, is_debug)
    until = None  # indent of a flagged loop whose body is skipped
    for k, (no, t) in enumerate(lines):
        ind = len(t) - len(t.lstrip())
        if until is not None:
            if ind > until:
                continue
            until = None
        while ifs and ind <= ifs[-1][0] and not re.match(r"\s*(else|elif)\b", t):
            ifs.pop()
        if re.match(r"\s*(comptime\s+)?(el)?if\b|\s*else\b", t):
            if re.match(r"\s*(else|elif)\b", t) and ifs and ifs[-1][0] == ind:
                ifs.pop()
            ifs.append((ind, bool(_DEBUG_IF.match(t))))
            continue
        o = owner[k]
        if o is None or skipdef.get(o) or not t.lstrip().startswith("for "):
            continue
        if no in skip or any(d for _, d in ifs) or lines[k - 1][1].strip().startswith("@parameter"):
            continue
        # the owner's rule counts rows, features, classes and members alike:
        # any runtime bound is data-sized; a walk over a List is too
        rm = re.match(r"\s*for\s+(\w+)\s+in\s+range\(\s*(.*)\)\s*:", t) or \
            re.match(r"\s*for\s+(\w+)\s+in\s+()([\w.\[\]]+)\s*:", t)
        if not rm or (rm.group(2) and not _runtime_bound(rm.group(2))):
            continue
        if not rm.group(2) and not re.search(r"\[\]|\w$", rm.group(3)):
            continue
        sm = _SMALL_LOOP.search(t)
        if sm and re.search(r"\b" + re.escape(sm.group(1)) + r"\b", rm.group(2)):
            continue
        var = rm.group(1)
        # the body touches data with the loop variable (an index, a load or a
        # store at it, an append), and drives no device work (a loop that
        # launches per level, step or tree is orchestration, not CPU work)
        use = re.compile(r"\[[^\]]*\b" + re.escape(var) + r"\b[^\]]*\]|\.append\(|"
                         r"(load|store|bitcast)\w*\s*[\[(].*\b" + re.escape(var) + r"\b")
        if not rm.group(2):
            use = re.compile(r"\b" + re.escape(var) + r"\b")
        inner = []
        for _, t2 in lines[k + 1:]:
            if len(t2) - len(t2.lstrip()) <= ind:
                break
            inner.append(t2)
        if any(_DRIVES_DEVICE.search(x) for x in inner):
            continue
        if any(use.search(x) for x in inner) and any(
                _HOST_WORK.search(x) or re.search(r"\]\s*=[^=]|\.append\(", x) for x in inner):
            out.append((no, t))
            until = ind
    return out


def _norm(t):
    return re.sub(r"\s+", " ", t.strip())


def tree_findings(tree, paths=None):
    """[(key, line_no)] where key = (rule, path, normalized text, occurrence)."""
    res = []
    for p in (paths if paths is not None else tree.scoped()):
        seen = collections.Counter()
        for rule, no, t in sorted(scan_file(tree, p), key=lambda x: (x[1], x[0])):
            n = _norm(t)
            occ = seen[(rule, n)]
            seen[(rule, n)] += 1
            res.append(((rule, p, n, occ), no))
    return res

# ------------------------------------------------------------- baseline ----

_HDR = "rule\tclass\towner\tstate\tpath\tocc\ttext"
# `owed2` is the same for a rule added after the 2026-10-03 hooks
# (block-per-n). `owed` is `debt` for a rule added after the 2026-10-02 hooks were
# installed: those hooks skip a state they do not know, so the row neither
# fails them as stale nor needs them to know the rule.
# `owed-py` is the same for the Python glue-only rules (Oct 3): hooks
# installed before them know `owed` but not these rules, so they skip only a
# state they do not know.
_DEBT_STATES = ("debt", "owed", "owed2", "owed-py")
# rules added after the baseline was first written; a baseline with no row of
# one predates it (see check_tree)
_LATE_RULES = ("d2h-host-work", "one-block-n", "block-per-n", "py-data-loop", "py-np-compute", "py-reduce",
               "py-array-method")


def load_baseline(text):
    """Rows of the baseline. A `# why: ...` comment line gives the reason for
    the row right after it (required for d2h-host-work rows); older readers
    skip it as a comment."""
    rows = []
    why = ""
    for ln in text.splitlines():
        if ln.startswith("# why:"):
            why = ln[len("# why:"):].strip()
            continue
        if not ln.strip() or ln.startswith("#") or ln.startswith("rule\t"):
            continue
        c = ln.split("\t")
        if len(c) != 7:
            raise SystemExit(f"no_host_routes: bad baseline row: {ln[:120]}")
        rule, cls, owner, state, path, occ, text = c
        if rule in _OWED_RULES:
            raise SystemExit(f"no_host_routes: a {rule} row is owed work, never baseline debt "
                             f"(the checker reports it as owed; fix it instead): {ln[:120]}")
        if rule in _LATE_RULES and not why:
            raise SystemExit(f"no_host_routes: a {rule} row needs a `# why:` line above it: {ln[:120]}")
        rows.append(dict(rule=rule, cls=cls, owner=owner, state=state, path=path,
                         occ=int(occ), text=text, why=why))
        why = ""
    return rows


def dump_baseline(rows):
    rows = sorted(rows, key=lambda r: (r["path"], r["rule"], r["text"], r["occ"]))
    head = ("# no_host_routes baseline: CPU work in GPU code that main still carries.\n"
            "# It only shrinks. A fix deletes its rows (no_host_routes.py --prune-baseline).\n"
            "# state debt = on main; inflight = pre-authorized lines of an open PR; owed = debt of a rule\n"
            "# newer than the installed hooks (they skip it); owed-py = the same for the Python\n"
            "# glue-only rules (py-data-loop, py-np-compute, py-reduce, py-array-method).\n"
            "# A `# why: ...` line gives the reason the next row stays (required for d2h-host-work).\n")
    return head + _HDR + "\n" + "".join(
        (f"# why: {r['why']}\n" if r.get("why") else "")
        + "\t".join([r["rule"], r["cls"], r["owner"], r["state"], r["path"], str(r["occ"]),
                     r["text"]]) + "\n" for r in rows)


def _key(r):
    return (r["rule"], r["path"], r["text"], r["occ"])


_ANC = {}


def _inflight_ok(state, ref):
    """An in-flight row (state inflight@<PR head>) only serves a tree that
    contains that PR head: the allowance belongs to the PR, not to its text."""
    if not state.startswith("inflight@"):
        return False
    sha = state.split("@", 1)[1]
    k = (sha, ref)
    if k not in _ANC:
        _ANC[k] = _git("merge-base", "--is-ancestor", sha, ref).returncode == 0
    return _ANC[k]


def summary(rows, out=sys.stderr):
    debt = [r for r in rows if r["state"] in _DEBT_STATES]
    by_cls = collections.Counter(r["cls"] for r in debt)
    by_owner = collections.Counter(r["owner"] for r in debt)
    infl = collections.Counter(r["owner"] for r in rows if r["state"].startswith("inflight"))
    print(f"no-host-routes debt: {len(debt)} rows on GPU paths "
          f"(+{sum(infl.values())} in-flight allowances)", file=out)
    print("  by class: " + ", ".join(f"{k} {v}" for k, v in by_cls.most_common()), file=out)
    print("  by owner: " + ", ".join(f"{k} {v}" for k, v in by_owner.most_common()), file=out)
    if infl:
        print("  in flight: " + ", ".join(f"{k} {v}" for k, v in infl.most_common()), file=out)


def _baseline_text(ref, explicit):
    if explicit:
        with open(explicit, encoding="utf-8") as f:
            return f.read(), True
    r = _git("show", f"{ref}:{BASELINE}")
    if r.returncode == 0:
        return r.stdout, True
    for m in ("refs/remotes/origin/main", "origin/main"):
        r = _git("show", f"{m}:{BASELINE}")
        if r.returncode == 0:
            return r.stdout, False
    return "", False


def check_tree(ref, baseline_path=None, overlay=None, quiet=False):
    """0 clean, 1 new findings or stale rows."""
    btext, own = _baseline_text(ref, baseline_path)
    rows = load_baseline(btext)
    tree = Tree(ref, overlay)
    found = tree_findings(tree)
    owed_found = [(k, no) for k, no in found if k[0] in _OWED_RULES]
    found = [(k, no) for k, no in found if k[0] not in _OWED_RULES]
    debt_keys = collections.Counter(_key(r) for r in rows if r["state"] in _DEBT_STATES)
    infl_keys = collections.Counter(_key(r) for r in rows if _inflight_ok(r["state"], ref))
    allowed_extra = collections.Counter()
    # a rule newer than the tree's baseline (no row of it, so the baseline was
    # written before the rule) is judged like a tree that predates the
    # baseline, for that rule only: what the branch adds over its merge-base
    late = {r for r in _LATE_RULES if not any(x["rule"] == r for x in rows)} if own else set()
    if not own or late:
        # the tree predates the baseline: judge only what the branch adds over
        # its merge-base with main (main's later fixes are not charged to it)
        mb = _git("merge-base", "refs/remotes/origin/main", ref)
        if mb.returncode != 0:
            mb = _git("merge-base", "origin/main", ref)
        if mb.returncode == 0:
            allowed_extra = collections.Counter(k for k, _ in tree_findings(Tree(mb.stdout.strip()))
                                                if not own or k[0] in late)
    found_keys = collections.Counter(k for k, _ in found)
    new, matched_infl = [], []
    for k, no in found:
        if debt_keys[k] > 0:
            debt_keys[k] -= 1
        elif infl_keys[k] > 0:
            infl_keys[k] -= 1
            matched_infl.append((k, no))
        elif allowed_extra[k] > 0:
            allowed_extra[k] -= 1
        else:
            new.append((k, no))
    stale = [k for k, c in debt_keys.items() for _ in range(c)] if own else []
    allowed_owed = owed_allowance(ref)
    owed, new_owed = [], []
    for k, no in owed_found:
        if allowed_owed[k] > 0:
            allowed_owed[k] -= 1
            owed.append((k, no))
        else:
            new_owed.append((k, no))
    new += new_owed
    bad = bool(new or stale or (own and matched_infl))
    if not quiet:
        summary(rows)
        owed_summary(owed)
    if new:
        print(f"no-host-routes: REFUSED. {len(new)} finding(s) of CPU work in GPU code "
              "are not in the baseline:", file=sys.stderr)
        for (rule, p, n, _), no in new:
            print(f"  {p}:{no}: [{rule}] {_RULE_WHY.get(rule, '')}: {n[:120]}", file=sys.stderr)
        print("  Put the step on the GPU, in parallel. The baseline never grows.", file=sys.stderr)
    if stale:
        print(f"no-host-routes: REFUSED. {len(stale)} baseline row(s) no longer match "
              "(the CPU work is gone: delete the rows with "
              "`python3 tools/hooks/no_host_routes.py --prune-baseline`):", file=sys.stderr)
        for rule, p, n, occ in stale[:40]:
            print(f"  {p}: [{rule}] #{occ}: {n[:110]}", file=sys.stderr)
        if len(stale) > 40:
            print(f"  ... {len(stale) - 40} more", file=sys.stderr)
    if own and matched_infl:
        print(f"no-host-routes: REFUSED. {len(matched_infl)} in-flight row(s) are now in the "
              "tree: mark them debt (`--prune-baseline`):", file=sys.stderr)
        for (rule, p, n, _), no in matched_infl[:20]:
            print(f"  {p}:{no}: [{rule}] {n[:110]}", file=sys.stderr)
    return 1 if bad else 0


_OWED_CACHE = {}


def _owed_findings(sha):
    if sha not in _OWED_CACHE:
        _OWED_CACHE[sha] = collections.Counter(
            k for k, _ in tree_findings(Tree(sha)) if k[0] in _OWED_RULES)
    return _OWED_CACHE[sha]


def owed_allowance(ref):
    """Owed findings a tree may carry: those already in its merge-base with the
    owed anchor and in its merge-base with main (the larger count of each).
    Everything else of an owed rule is new CPU work and is refused."""
    allowed = collections.Counter()
    bases = set()
    for other in (_OWED_ANCHOR, "refs/remotes/origin/main", "origin/main"):
        mb = _git("merge-base", other, ref)
        if mb.returncode == 0 and mb.stdout.strip():
            bases.add(mb.stdout.strip())
    for b in sorted(bases):
        allowed |= _owed_findings(b)
    return allowed


def owed_summary(owed, out=sys.stderr):
    """The owed findings, by rule and by file: CPU work that a fix lane owes
    (~/mojolearn-evidence/cpu-reaudit-2026-10-04/checker-new-findings.tsv),
    never accepted as debt."""
    if not owed:
        return
    by_rule = collections.Counter(k[0] for k, _ in owed)
    by_path = collections.Counter(k[1] for k, _ in owed)
    print(f"no-host-routes OWED (not baseline debt): {len(owed)} findings of CPU work on the GPU route "
          "(`--owed` lists them)", file=out)
    print("  by rule: " + ", ".join(f"{k} {v}" for k, v in by_rule.most_common()), file=out)
    print("  top files: " + ", ".join(f"{k} {v}" for k, v in by_path.most_common(8)), file=out)


def owed_rows(ref="HEAD"):
    """TSV of every owed finding in ref: rule, class, path, line, text."""
    tree = Tree(ref) if ref != "HEAD" else Tree("HEAD", _worktree_overlay())
    lines = ["rule\tclass\tpath\tline\ttext"]
    for (rule, p, n, _), no in tree_findings(tree):
        if rule in _OWED_RULES:
            lines.append("\t".join([rule, _RULE_CLASS[rule], p, str(no), n]))
    return "\n".join(lines) + "\n"


def prune_baseline(ref="HEAD", path=BASELINE):
    with open(path, encoding="utf-8") as f:
        rows = load_baseline(f.read())
    tree = Tree(ref) if ref != "HEAD" else Tree("HEAD", _worktree_overlay())
    found = collections.Counter(k for k, _ in tree_findings(tree))
    keep = []
    removed = flipped = 0
    for r in rows:
        k = _key(r)
        if found[k] > 0:
            found[k] -= 1
            if r["state"].startswith("inflight"):
                r["state"] = "debt"
                flipped += 1
            keep.append(r)
        elif r["state"].startswith("inflight"):
            keep.append(r)
        else:
            removed += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write(dump_baseline(keep))
    print(f"no-host-routes: pruned {removed} stale row(s), flipped {flipped} in-flight row(s) "
          f"to debt; {len(keep)} rows remain", file=sys.stderr)
    summary(keep)
    return 0


def _worktree_overlay():
    """Uncommitted edits of tracked files, so --prune-baseline sees the worktree."""
    r = _git("diff", "--name-only", "HEAD")
    over = {}
    top = _git("rev-parse", "--show-toplevel").stdout.strip()
    for p in r.stdout.splitlines():
        fp = os.path.join(top, p)
        if os.path.exists(fp):
            with open(fp, encoding="utf-8", errors="replace") as f:
                over[p] = f.read()
    return over

# ------------------------------------------------------------ diff mode ----


def _walk(diff_text):
    """(path, line number, text) of every added line of a -U0 diff."""
    path = None
    line_no = 0
    for raw in diff_text.splitlines():
        if raw.startswith("+++ "):
            p = raw[4:]
            path = None if p == "/dev/null" else p[2:] if p.startswith("b/") else p
            continue
        if raw.startswith("@@"):
            m = re.search(r"\+(\d+)", raw)
            line_no = int(m.group(1)) if m else 0
            continue
        if not raw.startswith("+") or raw.startswith("+++") or path is None:
            continue
        line_no += 1
        yield path, line_no - 1, raw[1:]


def findings(diff_text, exempt=frozenset()):
    """Diff of added lines, no scope (every .mojo/.py path counts): line rules only."""
    out = []
    for path, here, text in _walk(diff_text):
        lang = "py" if path.endswith(".py") else "mojo" if path.endswith(".mojo") else None
        if lang is None:
            continue
        cl = _code_lines(text, lang)
        if not cl:
            continue
        t = cl[0][1]
        if (path, _norm(t)) in exempt:
            continue
        dm = _DEFINES.match(t)
        for rule, _cls, langs, pat, why in _LINE_RULES:
            if lang not in langs:
                continue
            hits = [m.group(0) for m in pat.finditer(t)]
            if rule == "host-env":
                hits = [h for h in hits if not _HOST_PLUMBING.fullmatch(h)]
            if not hits:
                continue
            if dm and rule in ("host-exec", "host-threads", "std-parallelize") and \
                    re.match(r"(HostExec|host_parallel\w*|parallelize)$", dm.group(2)):
                continue
            out.append(f"{path}:{here}: [{rule}] {why}: {t.strip()[:120]}")
    return out


def diff_mode(base, tip, main_ref):
    """Findings on the lines tip adds, in tip's GPU scope (full rule set)."""
    mb = _git("merge-base", main_ref, tip)
    if mb.returncode == 0:
        base = mb.stdout.strip()
    r = _git("diff", "-U0", "--no-color", "--no-ext-diff", "--diff-filter=AMR", base, tip)
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return 2
    added = collections.defaultdict(set)
    for p, no, _ in _walk(r.stdout):
        added[p].add(no)
    if not added:
        return 0
    btext, _ = _baseline_text(tip, None)
    infl = {(r_["path"], r_["text"]) for r_ in load_baseline(btext) if _inflight_ok(r_["state"], tip)}
    tree = Tree(tip)
    scoped = set(tree.scoped())
    out = []
    for p in sorted(added):
        if p not in scoped:
            continue
        for rule, no, t in scan_file(tree, p):
            if no in added[p] and (p, _norm(t)) not in infl:
                out.append(f"{p}:{no}: [{rule}] {_RULE_WHY.get(rule, '')}: {_norm(t)[:120]}")
    if not out:
        return 0
    print("no-host-routes: REFUSED. These added lines add CPU work to GPU code:", file=sys.stderr)
    for f in out:
        print("  " + f, file=sys.stderr)
    return 1

# ----------------------------------------------------------------- main ----


def main(argv):
    args = argv[1:]
    if args[:1] == ["--tree"]:
        ref = args[1] if len(args) > 1 and not args[1].startswith("--") else "HEAD"
        bl = None
        if "--baseline" in args:
            bl = args[args.index("--baseline") + 1]
        return check_tree(ref, bl)
    if args[:1] == ["--prune-baseline"]:
        return prune_baseline(args[1] if len(args) > 1 else "HEAD")
    if args[:1] == ["--owed"]:
        sys.stdout.write(owed_rows(args[1] if len(args) > 1 else "HEAD"))
        return 0
    if args[:1] == ["--summary"]:
        btext, _ = _baseline_text(args[1] if len(args) > 1 else "HEAD", None)
        summary(load_baseline(btext), out=sys.stdout)
        return 0
    if args[:1] == ["--diff"] and len(args) == 2:
        with open(args[1], encoding="utf-8", errors="replace") as f:
            found = findings(f.read())
        if found:
            print("no-host-routes: REFUSED. These added lines add CPU work to GPU code:",
                  file=sys.stderr)
            for x in found:
                print("  " + x, file=sys.stderr)
            return 1
        return 0
    main_ref = "origin/main"
    if "--main" in args:
        i = args.index("--main")
        main_ref = args[i + 1]
        args = args[:i] + args[i + 2:]
        if _git("rev-parse", "--verify", "-q", main_ref).returncode != 0:
            main_ref = "origin/main"
    if len(args) == 2:
        return diff_mode(args[0], args[1], main_ref)
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
