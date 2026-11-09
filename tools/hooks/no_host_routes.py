#!/usr/bin/env python3
"""Refuse CPU work on GPU installs. The rule is enforced in code, not in notes.

On a GPU install every fit, transform and predict runs only GPU work, in
parallel (Andrew, Oct 1-2 2026). The CPU is only for CPU-only installs,
MOJOLEARN_VENDOR=cpu verification digests and inference.

Modes:
    no_host_routes.py --tree REF [--baseline FILE] [--branch]
        Scan EVERY file of REF that a GPU install runs. Fail on any finding
        that is not in the baseline, and on any baseline row that no longer
        matches (the baseline only shrinks). Prints the debt per class and
        per owner. This is what the pre-push hook and the GitHub check run.
        --branch (the pre-push hook passes it for every ref except main):
        findings already in merge-base(origin/main, REF) are main's, not the
        branch's, and are reported but not refused. A push to main is always
        judged on the whole tree (Andrew, 2026-10-07: 15 findings that reached
        main refused every unrelated lane push for hours).
        The pre-push hook does not run --tree at all for archive refs
        (refs/heads/archive/*, refs/tags/archive-*, refs/tags/archive/*,
        2026-10-09) or salvage/* refs: they are never merged. See the ref
        table in tools/hooks/pre-push.
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

Parsing and scanning are split over worker processes: MOJOLEARN_HOOK_WORKERS=N
sets their number (any platform); unset, it is 4 on macOS and the core count
elsewhere.

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
Oct 2 2026 audit, and every later change can only delete rows. On 2026-11-09,
when the loop tags started to be judged (the LOOP-TAG VOCABULARY), the findings
of the rules that stopped trusting free-text notes were written once more, as
`owed-tags` rows, from the integration tree _REGEN_ANCHOR. A row main's
baseline lacks holds only for a finding main's tree (or, for a tree that
contains it, _REGEN_ANCHOR's tree) carries, so a branch can never grandfather
a finding of its own.
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
_verify_distributed_program.py _verify_partitioned_program.py _verify_native_runtime.py
_verify_par.py _verify_parallel_cv.py _verify_reference.py _verify_resources.py
_verify_small.py _verify_worker.py _verification_catalog.py
_verify_graph.py _verify_native_runtime.py _verify_generation.py
_verification_coverage.py _verification_ctr_models.py
_verification_evidence_data.py _verification_profiles.py
_verification_sources.py _verify_public.py _verify_light.py
_verify_light_command.py _verify_light_faults.py _verify_api.py
_verify_api_runner.py _verify_api_availability.py _verify_api_metadata.py
_verify_api_estimators.py _verify_api_utilities.py _verify_api_optimizers.py
_verify_api_neural.py _verify_api_training.py _verify_api_programs.py
_verify_api_models.py _verify_api_misc.py _verify_api_faults.py
_runtime_fallbacks.py
_forest_host.py _gbdt_host.py _byte_lm_host.py _byte_lm_trainer_host.py
_classical_host.py _serialize.py
neural_inference.py _causal_lm_fixtures.py
_bpe_trainer.py _tokenizer_synthetic.py
""".split())
# The bounded verifier modules above prepare test fixtures, compare outputs and
# inject test faults. They are not product runtime implementations. Exact names
# keep future modules subject to review; the scope test rejects product imports.
# neural_inference.py: the public CPU inference product.
# _causal_lm_fixtures.py: verification fixtures (_verify_causal_lm and tests only).
# _serialize.py: model save/load (no fit, transform or predict).
# _bpe_trainer.py: the BPE trainer's verification oracle (refused as a run-time route).
# _tokenizer_synthetic.py: tokenizer fixtures for the gates and the identity lane.
# _runtime_fallbacks.py: verifier-only exception descriptions and code identities;
# no product caller imports it (enforced by test_runtime_fallback_scope.py).
# The actual Array.__len__ product callback remains in scope and is reported.

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
                    "py-fsum": "py-compute", "py-map-arith": "py-compute",
                    "py-while-loop": "py-compute", "py-iter-call": "py-compute", "py-scalar-arith": "py-compute",
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
    "py-data-loop": "a Python loop or comprehension at runtime (only a model-structure loop tagged "
                    "`# driver-loop: over <what>` or a constant-bound `# small-loop(<CONST>)` holds; "
                    "`# glue:` and `# cpu-route:` notes exempt nothing)",
    "py-while-loop": "a Python while loop at runtime (move it into Mojo; only a true driver-loop tag holds)",
    "py-iter-call": "Python iteration inside a call at runtime (map(f, ...), list(<iterable>), dict.fromkeys, "
                    "itertools.chain, zip, enumerate)",
    "py-scalar-arith": "binary arithmetic on numbers at runtime in GPU-path Python (every computed number comes from Mojo)",
    "py-np-compute": "numpy compute at runtime in GPU-path Python (move it into Mojo)",
    "py-reduce": "sorted()/sum()/min()/max() over a sequence at runtime in GPU-path Python",
    "py-array-method": "an array reduction or sort method at runtime in GPU-path Python",
    "py-fsum": "an fsum (math.fsum or a portable twin) over values at runtime in GPU-path Python",
    "py-map-arith": "arithmetic on data through map(operator.*), itertools.accumulate/starmap or reduce at runtime",
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
accumulation_is_aligned byte_lm_attention_memory_profile
byte_lm_config_profile byte_lm_model_pool_open byte_lm_model_pool_ownership byte_lm_offload_open
byte_lm_parallel_ownership byte_lm_session_info byte_lm_step_glue_arm gbdt_model_dim
gbdt_resident_info gp_theta_params iforest_resident_release x_cnn_conv_shape
x_cnn_pool_shape forest_export forest_export_legacy
""".split())
# 2026-11-09: the `# cpu-route:` note and the free-text `# small-loop(<bound>:
# <what>): <why>` note no longer exempt a line (AGENTS.md: cpu-route tags are
# debt, never exemptions). A small-loop note holds only for a constant bound
# (_small_bound_ok); its bound is the text before `:` or `)`.
_SMALL_LOOP_ANY = re.compile(r"#\s*small-loop\(\s*([^:()]*?)\s*(?::[^()]*)?\)")
# THE DRIVER-LOOP RULE (Andrew, 2026-10-17; docs/plans/GPU_PATH_HOST_LOOPS.md).
# A host loop on the GPU path that walks model or launch structure (features,
# columns, words, classes, layers, leaves, depth, trees, iterations, configs,
# argument spans, buffer handles) carries, on its `for` line or the line above:
#     # driver-loop: over <what it walks>
# The tag is checked, not trusted: its range must name no data dimension
# (_DRIVER_DENY, or the length of a buffer a d2h copy filled) and its body must
# index no downloaded buffer (_downloaded_names), and (2026-11-09) the loop
# must walk structure by the LOOP-TAG VOCABULARY below. A tag that fails is a
# `mojo-host-loop` finding ("false driver-loop tag"). Accepted tags are a count
# in the --tree summary (class `driver-loop`, with the "could not judge" ones
# counted apart), never debt and never baseline rows. Readback reductions and
# data loops are defects to fix, never tagged. `while` loops are judged the
# same way (their condition is the source).
_DRIVER_TAG = re.compile(r"#\s*driver-loop:\s*over\s+\S")
# data dimensions a driver loop may never run over (extend as cases appear)
_DRIVER_DENY = re.compile(
    r"(?<!\w)(n|rows|n_rows|n_samples|samples|elements|n_elements|tokens|n_tokens|nnz|n_nnz|count|numel"
    r"|n_elems|n_obs|n_points|n_queries|n_q|\w+_rows|\w+_samples|\w+_elements|\w+_tokens|\w+_nnz)(?!\w)"
    r"|\blen\(\s*(x|y|X|Y)\s*\)")

# ------------------------------------------------- loop-tag vocabulary ----
# THE LOOP-TAG VOCABULARY (2026-11-09). A `# driver-loop: over <what>` tag
# claims that a loop walks model structure, never data
# (docs/plans/GPU_PATH_HOST_LOOPS.md). The checker judges the claim from the
# loop: first its source (the `for x in <expr>` iterable, the `range(<expr>)`
# bound, the `while` condition), then, when the source says nothing, its loop
# variables, then the tag text. Every identifier is split into lower-case
# words (on `_`, digits and case changes):
#   _V_DATA_PATTERNS   spellings that are data by themselves
#   _V_DATA_NAMES      whole identifiers, or first words, that name the data (X, y)
#   _V_DATA_WORDS      a word that names data: rows, samples, labels, values, ...
#   _V_VALUE_WORDS     a COUNT of these is structure (`range(n_classes)`), the
#                   sequence itself is data (`for c in classes_`)
#   _V_STRUCTURE_WORDS a word that names model structure: layers, blocks, heads, ...
#   _V_COUNT_WORDS     a first word that makes the identifier a count of the rest
# Before the words are read, a dict view (`.keys()`, `.values()`, `.items()`)
# is judged by its receiver and an attribute chain by its last name.
# In the loop's source or variables a data word anywhere makes it data; else a
# structure word makes it structure. In the tag text a structure word wins and
# a data word refutes the tag only alone. Otherwise the verdict is "unknown": the tag is kept, as
# before, and counted as "could not judge". This table is the one place to
# extend; tools/hooks/no_compute_in_shells.py carries the same table (each hook is
# installed as one self-contained file), keep the two in step.
_V_DATA_PATTERNS = re.compile(r"\.shape\s*\[\s*0\s*\]|\.tolist\b|\blen\(\s*[XxYy]\w*\s*\)|\bn_samples\b|\bnnz\b"
                           r"|\bdownload\w*|\breadback\w*")
_V_DATA_NAMES = frozenset("x y".split())
_V_DATA_WORDS = frozenset("""
row rows sample samples label labels value values val vals element elements elem elems token tokens
tok toks record records nnz obs observation observations point points query queries pixel pixels
datum data text texts document documents doc docs sentence sentences char chars byte bytes batch
batches chunk chunks numel target targets instance instances example examples
""".split())
_V_VALUE_WORDS = frozenset("class classes category categories unique uniques".split())
_V_STRUCTURE_WORDS = frozenset("""
layer layers block blocks head heads estimator estimators member members fold folds split splits
candidate candidates device devices gpu gpus rank ranks shard shards field fields param params
parameter parameters stage stages tree trees feature features column columns col cols dim dims
dimension dimensions axis axes ndim shape output outputs component components iteration iterations
iter iters epoch epochs step steps level levels depth leaf leaves arg args argument arguments kwarg
kwargs key keys name names option options config configs worker workers kernel kernels launch
launches pass passes round rounds model models module modules child children attr attrs attribute
attributes slot slots spec specs plan plans op ops phase phases kind kinds dtype dtypes metric
metrics handle handles span spans bin bins expert experts context contexts stream streams
""".split())
_V_COUNT_WORDS = frozenset("n num nb count total".split())


def _words(ident):
    """Lower-case words of one identifier: `n_samplesIn2` -> [n, samples, in]."""
    out = []
    for part in re.split(r"[_\d]+", ident):
        out.extend(w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", part))
    return out


def _judge_text(text, tag_text=False):
    """("data" | "structure" | "unknown", the deciding word) for one text.
    tag_text: the words of a tag's <what>, where a value word (classes) is a
    count, so structure."""
    if not text or not text.strip():
        return "unknown", ""
    m = _V_DATA_PATTERNS.search(text)
    if m:
        return "data", m.group(0)
    # a dict view walks its receiver (`kwargs.items()` walks kwargs); an
    # attribute chain is judged by its last name (`value.ndim` is a rank)
    text = re.sub(r"\.(?:keys|values|items)\(\s*\)", "", text)
    text = re.sub(r"(?<![\w.])(?:[A-Za-z_]\w*(?:\[\])?\.)+(?=[A-Za-z_])", "", text)
    struct = ""
    data = ""
    for ident in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", text):
        words = _words(ident)
        if not words:
            continue
        if ident == "n" or ident.lower() in _V_DATA_NAMES or (words[0] in _V_DATA_NAMES and ident[:1] in "XxYy"
                                                             and len(words) > 1 and not tag_text):
            if not tag_text:
                return "data", ident
            data = data or ident
            continue
        if words[0] in _V_COUNT_WORDS and len(words) > 1:
            rest = words[1:]
            if any(w in _V_DATA_WORDS for w in rest):
                if not tag_text:
                    return "data", ident
                data = data or ident
                continue
            if not struct and any(w in _V_STRUCTURE_WORDS or w in _V_VALUE_WORDS for w in rest):
                struct = ident
            continue
        if any(w in _V_DATA_WORDS for w in words) or (not tag_text and any(w in _V_VALUE_WORDS for w in words)):
            if not tag_text:
                return "data", ident
            data = data or ident
            continue
        if not struct and any(w in _V_STRUCTURE_WORDS or (tag_text and w in _V_VALUE_WORDS) for w in words):
            struct = ident
    # in a tag's own text a structure word wins ("the batch's classes"): the
    # text refutes the tag only when it names data and no structure
    if struct:
        return "structure", struct
    return ("data", data) if data else ("unknown", "")


def _judge_loop(source, targets="", what=""):
    """The verdict on a tagged loop: its source, else its loop variables, else
    the tag text decides (see the vocabulary above)."""
    for text, tag in ((source, False), (targets, False), (what, True)):
        v = _judge_text(text, tag)
        if v[0] != "unknown":
            return v
    return "unknown", ""


# A `# small-loop(<bound>...)` note holds only when its bound is a literal or a
# named constant (UPPER_CASE) that the loop line uses (2026-11-09): a loop over
# a variable size is a driver loop, judged by the vocabulary above, or a defect.
_CONST_BOUND = re.compile(r"[\d_]+|[A-Z][A-Z0-9_]*")


def _small_bound_ok(bound, code):
    bound = (bound or "").strip()
    return bool(_CONST_BOUND.fullmatch(bound)) and re.search(
        r"(?<!\w)" + re.escape(bound) + r"(?!\w)", code) is not None


# the loop tags a comment-only line may carry for the loop line right below it
_TAG_ABOVE = re.compile(r"#\s*(driver-loop:\s*over\s+\S|small-loop\()")
_DRIVER_WHAT = re.compile(r"#\s*driver-loop:\s*over\s+(.*)$")


def _loop_source(code):
    """(source, loop variables) of a loop: the `for x in <expr>` iterable (or a
    comprehension's first one), else the `while` condition, else the code."""
    m = re.match(r"\s*(?:async\s+)?for\s+(.+?)\s+in\s+(.+?):?\s*$", code)
    if m:
        return m.group(2), m.group(1)
    m = re.match(r"\s*while\b\s*(.*?):?\s*$", code)
    if m:
        return m.group(1), ""
    m = re.search(r"\bfor\s+([\w\s,()*]+?)\s+in\s+(.+?)(?=\s+(?:for|if)\b|[\]})]|$)", code)
    if m:
        return m.group(2), m.group(1)
    return code, ""


def _tag_verdict(t):
    """("data" | "structure" | "unknown", word) of the driver-loop tag on line t
    (its loop source first, then its loop variables, then the tag text)."""
    w = _DRIVER_WHAT.search(t)
    code = t.split("#", 1)[0]
    source, targets = _loop_source(code)
    return _judge_loop(source, targets, w.group(1) if w else "")


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
    tag = None  # a `# driver-loop:` / `# small-loop(` line waiting for its loop line
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
        # a loop tag on a comment-only line serves the code line right below
        # it (a Mojo `for`/`while`, or any Python line: a comprehension's line
        # is its loop line); any other line in between drops it
        if s.startswith("#") and _TAG_ABOVE.match(s):
            tag = s.strip()
            continue
        if not s or s.startswith("#") or s.startswith("//"):
            tag = None
            continue
        if tag is not None and (lang == "py" or re.match(r"(for|while)\b", s)) and not _TAG_ABOVE.search(line):
            line = line.rstrip() + "  " + tag
        tag = None
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
        # bindings/entry/entry_<name>.mojo (the per-binding entry, tools/merged_bindings.py)
        # only exports the PyInit of the module body bindings/<name>.mojo: the body is the root
        for m in re.finditer(r"bindings/(?:entry/entry_)?(_mojolearn\w*)\.mojo", t):
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
            # parse the uncached files across all cores (forked workers share
            # the blob cache, so only the sha travels)
            todo = sorted({shas[p] for p in paths if p not in overlay and (shas[p], lang) not in _LINES_CACHE})
            for sha, v, imps in _pool_map(None, _parse_blob, [(sha, lang) for sha in todo]):
                _LINES_CACHE[(sha, lang)] = v
                if imps is not None:
                    _LINES_CACHE[("imps", sha)] = imps
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
        self.mod_alias = {}  # (path, alias) -> module path (`from pkg import mod as alias`)
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
                        if sub:
                            self.mod_alias[(p, alias)] = sub[0]
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
        todo = sorted(p for p in self.gpu_mojo
                      if self._sha.get(p) is not None and ("hf", p, self._sha.get(p)) not in _LINES_CACHE)
        for p, v in zip(todo, _pool_map(self, "_host_functions", todo)):
            _LINES_CACHE[("hf", p, self._sha[p])] = v
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
    # (or a device executor whose methods launch on its own context: the
    # resident decomp kit `DKit`, x_decomp/kit_device.mojo; `DeviceExec`)
    _ON_DEVICE = re.compile(r"process_ctx|DeviceContext|DeviceBuffer|enqueue_\w+|\bctx\b|\bDKit\s*\(|\bDeviceExec\s*\("
                            r"|\bdevice_\w+\s*\(|\w+_device\w*\s*\(|thread_idx|block_idx"
                            r"|\b\w+\[\s*DeviceExec\s*(?:,|\])")
    _LOOP = re.compile(r"^\s*for\s+\w+\s+in\s+range\(([^)]*)\)|^\s*while\b")
    _CALLEE = re.compile(r"\b([A-Za-z_]\w*)\s*[\[(]")
    # a session object whose struct holds its own device contexts or buffers
    # (`var contexts: List[DeviceContext]`): an export that downcasts a
    # PythonObject to it drives that device through its methods, which the
    # call closure (top-level functions only) cannot follow (lane cpu4-python)
    _STRUCT = re.compile(r"^struct\s+(\w+)")
    _DEV_FIELD = re.compile(r"^\s+var\s+\w+\s*:\s*[^=#]*\b(DeviceContext|DeviceBuffer)\b")
    _DOWNCAST = re.compile(r"\bdowncast_value_ptr\[\s*(\w+)\s*\]")

    def device_structs(self):
        """Names of GPU-module structs with a DeviceContext/DeviceBuffer field."""
        if hasattr(self, "_dev_structs"):
            return self._dev_structs
        out = set()
        for p in self.gpu_mojo:
            cur = None
            for _, t in self.lines.get(p, []):
                m = self._STRUCT.match(t)
                if m:
                    cur = m.group(1)
                    continue
                if cur is None:
                    continue
                if t[:1].strip() and not t.startswith((")", "]", "#", "@")):
                    cur = None
                    continue
                if self._DEV_FIELD.match(t):
                    out.add(cur)
        self._dev_structs = out
        return out
    _QCALLEE = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*[\[(]")

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
                # a module-qualified call (`from pkg import mod as m`, `m.f(`)
                for m in self._QCALLEE.finditer(t):
                    mp = self.mod_alias.get((q, m.group(1)))
                    if mp is not None:
                        r = self._resolve_fn(mp, m.group(2))
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
        files = [p for p in sorted(self.gpu_mojo)
                 if "def_function" in "".join(t for _, t in self.lines.get(p, []))]
        self.device_structs()
        # a function imported under a host-column alias (`f as host_f`: the
        # host fallback of a device export, its _OFF arm) is the host column,
        # with everything it calls
        aliased = sorted({(src, name) for (p, alias), (src, name) in self.sym_src.items()
                          if name and src in self.gpu_mojo and _HOST_COLUMN_FN.search(alias)})
        # each registering module and each aliased function is one unit
        units = [("x", p) for p in files] + [("a", sn) for sn in aliased]
        for items, fns in _pool_map(self, "_export_unit", units):
            for name, qg in items:
                out[name] = qg
            impl |= fns
        self._host_exports = out
        # functions that only serve a host export: their loops are charged
        # at the Python call site, once
        self._export_impl = impl
        return out

    def _export_unit(self, unit):
        """([(export name, (module, function))], functions reached) of one
        host_exports unit: a registering module, or a host-column alias."""
        kind, it = unit
        if kind == "x":
            _p, items, fns = self._exports_of(it)
            return items, fns
        src, name = it
        return [], (self._closure(src, name) if name in self._top_bodies(src) else set())

    def _exports_of(self, p):
        """(p, [(export name, (module, function))], functions they reach) of
        the host exports registered in module p (see host_exports)."""
        items, impl = [], set()
        txt = " ".join(t for _, t in self.lines.get(p, []))
        for m in self._REGISTER.finditer(txt):
            q, g = self._resolve_fn(p, m.group(1))
            if q is None:
                continue
            fns = self._closure(q, g)
            bodies = [self._top_bodies(a).get(b, [])[1:] for a, b in fns]
            if any(self._ON_DEVICE.search(t) for body in bodies for _, t in body):
                continue
            dev_structs = self.device_structs()
            if any(m2.group(1) in dev_structs for body in bodies for _, t in body
                   for m2 in self._DOWNCAST.finditer(t)):
                continue
            # 2026-11-09: a `small-loop` note on a runtime bound no longer
            # clears the export (a note holds only for a constant bound, which
            # is not runtime)
            if any(lm and (lm.group(1) is None or _runtime_bound(lm.group(1)))
                   for body in bodies for lm in (self._LOOP.match(t) for _, t in body)):
                items.append((m.group(2), (q, g)))
                impl |= fns
        return p, items, impl

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


# A captured-program worker owns an Exec: DeviceExec in the GPU binding,
# HostExec only in the CPU binding (which admits one physical worker). Its
# explicit drain is the same native GPU-driver boundary as ctx.synchronize.
_DRIVE = re.compile(r"\bDeviceContext\s*\(|\.enqueue_\w+|\bctx\.synchronize\(|\.session\.ex\.sync\(")


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
# a grid of one block per small count: classes, outputs, features, targets.
# A tuple grid matches only when every further dimension is 1: a grid
# `(n_cols, ceildiv(rows, B))` splits the rows over blocks and is not one
# block walking them (lane debt-block-per-n, 2026-10-21).
_GRID_SMALL_NAMES = (r"(dims\.C|dims\.D|C|D|n_classes|n_class|num_classes|n_targets|n_outputs"
                     r"|n_cols|n_features|d|k|cd|n_comp|n_components)")
_GRID_SMALL = re.compile(
    r"\bgrid_dim\s*=\s*(\(\s*" + _GRID_SMALL_NAMES + r"\s*(,\s*1\s*)*\)"
    r"|" + _GRID_SMALL_NAMES + r"\s*(,|\)|$))")
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

# ------------------------------------------------------------ parallel ----
# Independent files are split across every core (Andrew, 2026-11-05: a serial
# loop over independent units is a defect). The workers are forked, so they
# share the parent's Tree and blob cache; each returns plain results. Where
# fork is unavailable the same work runs in the parent.
_POOL_TREE = None


def hook_workers():
    """Worker processes for the parallel scan: MOJOLEARN_HOOK_WORKERS=N when set
    (any platform, at least 1), else 4 on macOS (the hooks run on a shared,
    often loaded laptop) and os.cpu_count() elsewhere (rented hosts and CI)."""
    env = os.environ.get("MOJOLEARN_HOOK_WORKERS", "").strip()
    if env:
        try:
            return max(1, int(env))
        except ValueError:
            pass
    return 4 if sys.platform == "darwin" else (os.cpu_count() or 1)


def _pool_call(arg):
    method, chunk = arg
    f = getattr(_POOL_TREE, method) if not callable(method) else (lambda it: method(_POOL_TREE, it))
    _FALSE_TAG_WHY.clear()  # only what this chunk adds travels back
    res = [f(it) for it in chunk]
    return res, dict(_FALSE_TAG_WHY)


def _pool_map(tree, method, items):
    """[method(tree, item)] for each item, in item order, across all cores.
    method: a Tree method name or a function f(tree, item)."""
    global _POOL_TREE
    items = list(items)
    workers = min(hook_workers(), len(items) // 4)
    if workers > 1:
        import multiprocessing
        _POOL_TREE = tree
        try:
            ctx = multiprocessing.get_context("fork")
            # a few chunks per worker balance uneven files; each chunk sends
            # its results back in one message
            n = workers * 4
            chunks = [items[i::n] for i in range(n)]
            # freeze the parent's objects so the workers' collector never
            # touches (and copies) the shared pages, and stop the workers as
            # soon as their results are in
            import gc
            gc.freeze()
            pool = ctx.Pool(workers, initializer=gc.disable)
            try:
                got = pool.map(_pool_call, [(method, c) for c in chunks], chunksize=1)
            finally:
                pool.terminate()
                gc.unfreeze()
            for _r, why in got:
                _FALSE_TAG_WHY.update(why)
            res = [None] * len(items)
            for i, (r, _w) in enumerate(got):
                res[i::n] = r
            return res
        except (OSError, ValueError, AttributeError):
            pass
        finally:
            _POOL_TREE = None
    f = (lambda it: getattr(tree, method)(it)) if not callable(method) else (lambda it: method(tree, it))
    return [f(it) for it in items]


def _parse_blob(_tree, item):
    """(sha, code lines, imports or None) of one blob: one worker's unit."""
    sha, lang = item
    v = _code_lines(_BLOB_CACHE.get(sha, ""), lang)
    return sha, v, (_imports(v) if lang == "mojo" else None)


def _scan_one(tree, path):
    return path, scan_file(tree, path, with_driver=True)


def _scan_all(tree, paths):
    """Scan every path of tree once, across all cores; results live on tree."""
    done = tree.__dict__.setdefault("_scanned", {})
    need = [p for p in paths if p not in done]
    if not need:
        return
    if hasattr(tree, "host_exports"):
        tree.host_exports()  # once, before the fork: every worker shares it
    keys = {}
    todo = []
    for p in need:
        ck = scan_file(tree, p, key_only=True)
        if ck is not None and ck in _SCAN_CACHE:
            done[p] = _SCAN_CACHE[ck]
            continue
        keys[p] = ck
        todo.append(p)
    for p, r in _pool_map(tree, _scan_one, todo):
        done[p] = r
        if keys[p] is not None:
            _SCAN_CACHE[keys[p]] = r


def scan_file(tree, path, with_driver=False, key_only=False):
    """[(rule, line_no, code_text)] for one scoped file; with_driver: also the
    accepted driver-loop tags, as (findings, [(line_no, text)]). key_only:
    only the scan cache key (None for an uncommitted file)."""
    hit = None if key_only else getattr(tree, "__dict__", {}).get("_scanned", {}).get(path)
    if hit is not None:
        return hit if with_driver else hit[0]
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
    if key_only:
        return ck if sha is not None else None
    if sha is not None and ck in _SCAN_CACHE:
        res, acc = _SCAN_CACHE[ck]
        return (res, acc) if with_driver else res
    acc = []
    res = _scan_lines(lang, lines, host_thread_names, host_syms, import_of, local_host,
                      native=native, impl=impl, path=path, accepted=acc)
    if sha is not None:
        _SCAN_CACHE[ck] = (res, acc)
    return (res, acc) if with_driver else res


def _scan_lines(lang, lines, host_thread_names, host_syms, import_of, local_host,
                native=frozenset(), impl=frozenset(), path="", accepted=None):
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
        # 2026-11-09: a `# cpu-route:` note on a host-module load exempts
        # nothing any more (AGENTS.md: cpu-route tags are debt to remove)
        # lane py-runtime round 3: sklearn's own interface (the tags it asks
        # an estimator for) imports sklearn only when sklearn calls in; a
        # reviewed `# glue:` note on that import line says so
        if rule == "py-sklearn" and _GLUE.search(t):
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
        for rule, no, t in _py_compute(lines, accepted):
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
    for no, t in _mojo_host_loops(lines, path, impl, d2h_lines, accepted):
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
# A d- or k-sized walk over fitted metadata (per-feature category sizes,
# per-class offsets: values the binding returned or the user passed, never
# rows or tokens) carries `# small-loop(<bound>: <what>): <why>`, where the
# bound names a variable of the line; an explicit CPU-only input step (text,
# files, user callables) carries `# cpu-route: <reason>`.
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
# lane py-runtime round 3: the two spellings that walked data in C while the
# rules above only saw comprehensions: an exact fsum (math.fsum, _pm.fsum,
# a module's `_fsum` alias) and operator arithmetic mapped over sequences
_PY_FSUM = re.compile(r"(?<![\w.])(?:\w+\.)?_?fsum\s*\(")
_PY_MAP_ARITH = re.compile(
    r"\bmap\(\s*operator\.(add|sub|mul|truediv|floordiv|mod|pow|neg|abs|ne|eq|lt|le|gt|ge|or_|and_|xor)\b"
    r"|\bitertools\.(accumulate|starmap)\s*\(|\b(functools\.)?reduce\(\s*operator\.")
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


# 2026-11-09: `# glue:` and `# cpu-route:` notes exempt nothing here any more
# (AGENTS.md: cpu-route tags are debt, never exemptions). A Python loop holds
# only with a loop tag on its line or on the comment-only line above, and the
# tag is checked: `# driver-loop: over <what>` must walk model structure (the
# LOOP-TAG VOCABULARY), `# small-loop(<bound>...)` needs a constant bound the
# line uses. Also seen now: `while` loops (py-while-loop), iteration inside one
# call (py-iter-call: map(f, ...) with a non-operator callable, list(<iterable>),
# dict.fromkeys, itertools.chain, zip, enumerate) and binary arithmetic on
# numbers (py-scalar-arith).
_PY_WHILE = re.compile(r"^\s*while\b")
_PY_ITER_CALL = re.compile(
    r"(?<![\w.])map\(\s*(?!operator\.)[^\s)]"
    r"|(?<![\w.])list\(\s*[^\s)]"
    r"|\.fromkeys\s*\("
    r"|\bitertools\.chain\b|(?<![\w.])chain(?:\.from_iterable)?\s*\("
    r"|(?<![\w.])(zip|enumerate)\s*\(")
# a binary arithmetic operator between two operands (strings are blanked
# first, so `"a" + b` is not arithmetic), or an augmented assignment
_PY_ARITH = re.compile(r"(?<=[\w)\]])\s*(\*\*|//|[-+*/%])(?![=*/])\s*(?=[\w(.]|\[(?!\s*\]))"
                       r"|(?<=[\w)\]])\s*(\*\*|//|[-+*/%])=")
_PY_NOT_ARITH = re.compile(r"^\s*(@|class\s|import\s|from\s+\S+\s+import\b|raise\b|assert\b)")


def _py_arith(code):
    """True when the code line does binary arithmetic on numbers."""
    if _PY_NOT_ARITH.match(code):
        return False
    c = re.sub(r"\b\d+(?:\.\d*)?[eE][-+]?\d+\b", "1", code)  # 1e-6 is one number
    c = re.sub(r"->\s*[\w\[\], .]+", "", c)  # a return annotation
    for m in _PY_ARITH.finditer(c):
        before = c[:m.start()].rstrip()
        # `*args`/`**kw` and keyword unpacking start an argument, not arithmetic
        if m.group(1) in ("*", "**") and before.endswith((",", "(")):
            continue
        # a sign or an unpacking after a keyword (`return -1`, `x if c else -y`)
        if re.search(r"(?<![\w.])(return|yield|in|not|and|or|if|else|elif|is|await|lambda|print)$", before):
            continue
        return True
    return False


def _py_tag(t, code, accepted=None, no=0):
    """None when line t carries a loop tag that holds (the tag is counted in
    accepted), else the reason the loop is a finding."""
    if _DRIVER_TAG.search(t):
        verdict, word = _tag_verdict(t)
        if verdict == "data":
            return f"false driver-loop tag: the loop walks data (`{word}`), not model structure"
        if accepted is not None:
            accepted.append((no, t))
        return None
    sm = _SMALL_LOOP_ANY.search(t)
    if sm:
        if _small_bound_ok(sm.group(1), code):
            if accepted is not None:
                accepted.append((no, t))
            return None
        return "small-loop note without a constant bound the line uses"
    return ""


def _py_compute(lines, accepted=None):
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
        if not def_ind:
            continue
        code = _PY_STR.sub('""', t.split("  #", 1)[0] if "  #" in t else t)
        is_for = bool(_PY_FOR.search(code))
        if is_for or _PY_WHILE.match(code):
            why = _py_tag(t, code, accepted, no)
            if why is not None:
                out.append(("py-data-loop" if is_for else "py-while-loop", no, t))
                if why:
                    _FALSE_TAG_WHY[("", _norm(t))] = why
        elif _PY_ITER_CALL.search(code):
            why = _py_tag(t, code, accepted, no)
            if why is not None:
                out.append(("py-iter-call", no, t))
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
        if _PY_FSUM.search(code):
            out.append(("py-fsum", no, t))
        if _PY_MAP_ARITH.search(code):
            out.append(("py-map-arith", no, t))
        if _py_arith(code):
            out.append(("py-scalar-arith", no, t))
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
        if not def_ind:
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


# a kernel by name, or by a thread-index helper (`_tid()`, `tid()`) in its body
_KERNEL_NAME = re.compile(r"(^|_)kernel(_\w*)?$|_kern$")
_KERNEL_IDX = re.compile(r"\b_?(tid|gtid|thread_id|global_id)\s*\(\s*\)")


# A per-thread device helper (an @always_inline function a kernel calls for
# one row, chain or leaf) is device code, though it has no thread index of its
# own. Its def line carries `# device-helper: <reason of 3+ words>`, and the
# note holds only when every caller in the file is a kernel, another such
# helper or a host-column twin, with at least one kernel or helper among them
# when the file calls it at all; one other caller in the file and its loops
# are charged again. Callers in other modules are the note's reviewed claim.
_DEVICE_HELPER = re.compile(r"#\s*device-helper:\s*(\S+\s+){2,}\S+")


def _device_helpers(lines, owner, top, skipdef):
    """Mark noted device helpers as skipped defs (see _DEVICE_HELPER)."""
    names = {}
    for i, (_, t) in enumerate(lines):
        dm = re.match(r"\s*(?:def|fn)\s+(\w+)", t)
        if dm:
            names[i] = dm.group(1)
    first = {}
    for i in sorted(names):
        first.setdefault(top[i], i)  # the top-level def of each name
    noted = [i for i in names if _DEVICE_HELPER.search(lines[i][1]) and not skipdef.get(i)]
    helpers = set()
    changed = True
    while changed:
        changed = False
        for i in noted:
            if i in helpers:
                continue
            call = re.compile(r"\b" + re.escape(names[i]) + r"\s*[\[(]")
            by_device, ok, seen = False, True, False
            for k, (_, t) in enumerate(lines):
                if k == i or not call.search(t.split("#", 1)[0]):
                    continue
                seen = True
                o = owner[k]
                if o is None:
                    ok = False  # a module-level use, or another def's line naming it
                    break
                outer = first.get(top[o], o)
                if outer == i or o == i:
                    continue  # its own body
                if o in helpers or outer in helpers:
                    by_device = True
                elif skipdef.get(o) or skipdef.get(outer):
                    # a kernel, or a host-column twin / native host helper body
                    if not (_HOST_COLUMN_FN.search(names[o]) or _HOST_COLUMN_FN.search(names[outer])):
                        by_device = True
                else:
                    ok = False
                    break
            if ok and (by_device or not seen):
                helpers.add(i)
                skipdef[i] = True
                changed = True


_HOST_BUF_DECL = re.compile(r"\b(\w+)\s*(?::[^=]*)?=\s*[\w.]*enqueue_create_host_buffer\b|\b(\w+)\s*:\s*HostBuffer\[")


def _copy_statement(lines, i):
    """The enqueue_copy statement starting on lines[i], joined (at most 12 lines)."""
    txt, depth = "", 0
    for j in range(i, min(i + 12, len(lines))):
        seg = lines[j][1].split("#", 1)[0]
        txt += " " + seg.strip()
        depth += seg.count("(") - seg.count(")")
        if depth <= 0 and "(" in txt:
            break
    return txt


def _buf_name(expr):
    """(name, is_field) of the host buffer an argument expression names:
    `h.unsafe_ptr() + k` -> ("h", False), `s.h_est.unsafe_ptr()` -> ("h_est", True)."""
    e = expr.strip().split("+", 1)[0].strip()
    e = re.sub(r"\.unsafe_ptr\(\s*\)", "", e).strip(" ()")
    m = re.fullmatch(r"([\w.\[\]]+)", e)
    if not m:
        return None, False
    parts = re.findall(r"\w+", re.sub(r"\[[^\]]*\]", "", e))
    if not parts:
        return None, False
    return parts[-1], len(parts) > 1


def _downloaded_names(lines, topidx):
    """({top-level def index: local names}, file-wide field names) of host
    buffers a device-to-host copy fills (enqueue_copy with a host destination,
    a download/readback/to_host helper's result, a map_to_host alias), plus
    local pointer aliases of them."""
    hostbufs = set()
    for _, t in lines:
        for m in _HOST_BUF_DECL.finditer(t.split("#", 1)[0]):
            hostbufs.add(m.group(1) or m.group(2))
    local, fields = {}, set()

    def add(fn, expr):
        name, field = _buf_name(expr)
        if name:
            (fields.add(name) if field else local.setdefault(fn, set()).add(name))

    for i, (_, t) in enumerate(lines):
        code = t.split("#", 1)[0]
        fn = topidx[i]
        if fn is None:
            continue
        if "enqueue_copy(" in code:
            st = _copy_statement(lines, i)
            args = st[st.find("enqueue_copy(") + len("enqueue_copy("):]
            kw = dict(re.findall(r"\b(dst_ptr|dst_buf|src_buf|src_ptr)\s*=\s*([^,]+?)\s*(?=,|\)\s*$|\)\s*\Z)", args))
            if "dst_ptr" in kw:
                add(fn, kw["dst_ptr"])
            elif "dst_buf" in kw and "src_buf" in kw:
                name, _f = _buf_name(kw["dst_buf"])
                if name and (name in hostbufs or name.startswith("h_")):
                    add(fn, kw["dst_buf"])
            elif not kw:
                first = _top_args(args)[0] if args else ""
                name, _f = _buf_name(first)
                if name and (name in hostbufs or name.startswith("h_")):
                    add(fn, first)
        m = re.search(r"enqueue_copy_from_device\(\s*([^,]+),", code)
        if m:
            add(fn, m.group(1))
        m = re.match(r"\s*var\s+(\w+)\s*(?::[^=]*)?=\s*.*\b(\w*download\w*|\w*readback\w*|to_host\w*)\s*\(", code)
        if m:
            local.setdefault(fn, set()).add(m.group(1))
        m = re.search(r"map_to_host\(\s*\)\s+as\s+(\w+)", code)
        if m:
            local.setdefault(fn, set()).add(m.group(1))
    # pointer aliases: `var hp = h.unsafe_ptr()`, `var hp = F32Ptr(... h.unsafe_ptr() ...)`
    for i, (_, t) in enumerate(lines):
        fn = topidx[i]
        if fn is None or fn not in local and not fields:
            continue
        m = re.match(r"\s*var\s+(\w+)\s*(?::[^=]*)?=\s*(.*)$", t.split("#", 1)[0])
        if not m or re.search(r"unsafe_load|\[", m.group(2)):
            continue
        rhs = m.group(2)
        if not re.search(r"unsafe_ptr\(|bitcast|^\s*\w+\s*$", rhs):
            continue
        names = local.get(fn, set())
        if any(re.search(r"(?<![\w.])" + re.escape(n) + r"\b", rhs) for n in names) or \
                any(re.search(r"\." + re.escape(n) + r"\b", rhs) for n in fields):
            local.setdefault(fn, set()).add(m.group(1))
    return local, fields


def _driver_tag_reason(rng, inner, local, fields, line=""):
    """None when a tagged loop is a driver loop, else why the tag is false."""
    dm = _DRIVER_DENY.search(rng)
    if dm:
        return f"its range names the data size `{dm.group(0)}`"
    if line:
        verdict, word = _tag_verdict(line)
        if verdict == "data":
            return f"it walks data (`{word}`), not model structure"
    for n in sorted(local):
        if re.search(r"\blen\(\s*" + re.escape(n) + r"\s*\)", rng):
            return f"its range is the length of the downloaded buffer `{n}`"
    for x in inner:
        code = x.split("#", 1)[0]
        for n in sorted(local):
            if re.search(r"(?<![\w.])" + re.escape(n) + r"\b", code):
                return f"its body reads the downloaded buffer `{n}`"
        for n in sorted(fields):
            if re.search(r"\." + re.escape(n) + r"\b", code):
                return f"its body reads the downloaded buffer `.{n}`"
    return None


# why each false driver-loop tag was refused, by (path, normalized line)
_FALSE_TAG_WHY = {}


def _mojo_host_loops(lines, path, impl=frozenset(), skip=frozenset(), accepted=None):
    """[(line_no, text)] of host `for` loops over a runtime data size whose body
    touches data (indexes it with the loop variable, appends, or computes), in
    a non-kernel function of a GPU binding (bindings/) or a module that drives
    the device. Skipped: kernels, the functions behind a native host helper
    (charged at the Python call site), host-column functions, check replays
    (checks/), trace-only and debug blocks, comptime loops, loops already
    charged as d2h-host-work, loops marked `# small-loop(...)`, and the
    bodies of verified `# device-helper:` functions (_device_helpers).
    A `# driver-loop: over <what>` tag is checked (_driver_tag_reason): a
    true tag goes to `accepted` (the --tree count), a false one is a finding."""
    if "/checks/" in "/" + path:
        return []
    if not (path.startswith("bindings/") or any(_DEVICE_MODULE.search(t) for _, t in lines)):
        return []
    # each line's innermost def; a def is skipped when it is a kernel, a
    # trace recorder, a host-column function, or (by itself or its top-level
    # function) the body of a native host helper
    stack, owner, top, skipdef = [], [], {}, {}
    topidx = []  # each line's top-level def index (the scope of a download)
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
                              or _HOST_COLUMN_FN.search(outer) or _KERNEL_NAME.search(name))
            topidx.append(stack[0][1] if stack else i)
            stack.append((ind, i))
            owner.append(None)
            continue
        o = stack[-1][1] if stack else None
        owner.append(o)
        topidx.append(stack[0][1] if stack else None)
        if o is not None and (_KERNEL_TOK.search(t) or _KERNEL_IDX.search(t) or _TRACE_ONLY.match(t)):
            skipdef[o] = True
    _device_helpers(lines, owner, top, skipdef)
    dl = None  # downloaded buffers, computed on the first driver-loop tag
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
        is_while = t.lstrip().startswith("while ")
        if o is None or skipdef.get(o) or not (t.lstrip().startswith("for ") or is_while):
            continue
        if no in skip or any(d for _, d in ifs) or lines[k - 1][1].strip().startswith("@parameter"):
            continue
        if _DRIVER_TAG.search(t):
            code = t.split("#", 1)[0]
            dr = re.match(r"\s*for\s+.+?\s+in\s+(.*):\s*$", code) or re.match(r"\s*while\s+(.*?):?\s*$", code)
            rng = dr.group(1) if dr else code
            inner = []
            for _, t2 in lines[k + 1:]:
                if len(t2) - len(t2.lstrip()) <= ind:
                    break
                inner.append(t2)
            if dl is None:
                dl = _downloaded_names(lines, topidx)
            why = _driver_tag_reason(rng, inner, dl[0].get(topidx[k], set()), dl[1], t)
            if why is None:
                if accepted is not None:
                    accepted.append((no, t))
            else:
                _FALSE_TAG_WHY[(path, _norm(t))] = why
                out.append((no, t))
            continue
        code = t.split("#", 1)[0]
        if is_while:
            # an untagged `while` over a data size whose body stores or appends
            # (2026-11-09; `while` loops were invisible before)
            cond = re.match(r"\s*while\s+(.*?):?\s*$", code)
            cond = cond.group(1) if cond else ""
            if not (_DRIVER_DENY.search(cond) or _judge_text(cond)[0] == "data"):
                continue
            inner = []
            for _, t2 in lines[k + 1:]:
                if len(t2) - len(t2.lstrip()) <= ind:
                    break
                inner.append(t2)
            if any(_DRIVES_DEVICE.search(x) for x in inner):
                continue
            if any(re.search(r"\]\s*=[^=]|\.append\(|\bstore\w*\s*[\[(]", x) for x in inner):
                out.append((no, t))
                until = ind
            continue
        # the owner's rule counts rows, features, classes and members alike:
        # any runtime bound is data-sized; a walk over a List is too
        rm = re.match(r"\s*for\s+(\w+)\s+in\s+range\(\s*(.*)\)\s*:", code) or \
            re.match(r"\s*for\s+(\w+)\s+in\s+()([\w.\[\]]+)\s*:", code)
        if not rm or (rm.group(2) and not _runtime_bound(rm.group(2))):
            continue
        if not rm.group(2) and not re.search(r"\[\]|\w$", rm.group(3)):
            continue
        # 2026-11-09: the `small-loop` note no longer clears a runtime bound
        # (it holds only for a constant bound, which never reaches this line)
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
    ps = list(paths if paths is not None else tree.scoped())
    _scan_all(tree, ps)
    for p in ps:
        seen = collections.Counter()
        for rule, no, t in sorted(scan_file(tree, p), key=lambda x: (x[1], x[0])):
            n = _norm(t)
            occ = seen[(rule, n)]
            seen[(rule, n)] += 1
            res.append(((rule, p, n, occ), no))
    return res

def tree_driver_loops(tree, paths=None, verdicts=None):
    """Counter {path: accepted loop tags} (Mojo and Python). verdicts, a
    Counter, receives them by verdict: structure, unknown (could not judge),
    small-loop (a constant bound)."""
    c = collections.Counter()
    ps = list(paths if paths is not None else tree.scoped())
    _scan_all(tree, ps)
    for p in ps:
        _, acc = scan_file(tree, p, with_driver=True)
        if acc:
            c[p] += len(acc)
            if verdicts is not None:
                for _no, t in acc:
                    verdicts["small-loop" if not _DRIVER_TAG.search(t) else _tag_verdict(t)[0]] += 1
    return c


def _why(rule, path, text):
    """The refusal reason of one finding (a false driver-loop tag says why)."""
    w = _FALSE_TAG_WHY.get((path, text))
    if rule == "mojo-host-loop" and w:
        return f"false driver-loop tag: {w}"
    w = _FALSE_TAG_WHY.get(("", text))
    if rule in ("py-data-loop", "py-while-loop") and w:
        return w
    return _RULE_WHY.get(rule, "")

# ------------------------------------------------------------- baseline ----

_HDR = "rule\tclass\towner\tstate\tpath\tocc\ttext"
# `owed2` is the same for a rule added after the 2026-10-03 hooks
# (block-per-n). `owed` is `debt` for a rule added after the 2026-10-02 hooks were
# installed: those hooks skip a state they do not know, so the row neither
# fails them as stale nor needs them to know the rule.
# `owed-py` is the same for the Python glue-only rules (Oct 3): hooks
# installed before them know `owed` but not these rules, so they skip only a
# state they do not know.
# `owed-tags` is the same for the rules of 2026-11-09 (judged loop tags, no
# free-text exemptions, while loops, iteration inside calls, scalar
# arithmetic): main's findings of them were written once, from main's tree, and
# hooks installed before them skip the state.
_DEBT_STATES = ("debt", "owed", "owed2", "owed-py", "owed-tags")
# rules added after the baseline was first written; a baseline with no row of
# one predates it (see check_tree)
_LATE_RULES = ("d2h-host-work", "one-block-n", "block-per-n", "py-data-loop", "py-np-compute", "py-reduce",
               "py-array-method", "py-fsum", "py-map-arith")


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
            "# glue-only rules (py-data-loop, py-np-compute, py-reduce, py-array-method, py-fsum,\n"
            "# py-map-arith); owed-tags = the same for the 2026-11-09 loop-tag rules (py-while-loop,\n"
            "# py-iter-call, py-scalar-arith, and lines a glue/cpu-route/small-loop note no longer exempts).\n"
            "# A row main's baseline lacks holds only for a finding main's own tree carries.\n"
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


def check_tree(ref, baseline_path=None, overlay=None, quiet=False, branch=False):
    """0 clean, 1 new findings or stale rows. branch: main's own findings are not charged."""
    btext, own = _baseline_text(ref, baseline_path)
    rows = load_baseline(btext)
    tree = Tree(ref, overlay)
    found = tree_findings(tree)
    if not overlay:
        tip = _git("rev-parse", ref + "^{commit}")
        if tip.returncode == 0:
            _FINDINGS_CACHE.setdefault(tip.stdout.strip(), collections.Counter(k for k, _ in found))
    owed_found = [(k, no) for k, no in found if k[0] in _OWED_RULES]
    found = [(k, no) for k, no in found if k[0] not in _OWED_RULES]
    debt_keys = collections.Counter(_key(r) for r in rows if r["state"] in _DEBT_STATES)
    grown_refused = collections.Counter()
    if own:
        debt_keys, grown_refused = _restrict_to_main(debt_keys, ref, quiet)
    infl_keys = collections.Counter(_key(r) for r in rows if _inflight_ok(r["state"], ref))
    allowed_extra = collections.Counter()
    # a rule newer than the tree's baseline (no row of it, so the baseline was
    # written before the rule) is judged like a tree that predates the
    # baseline, for that rule only: what the branch adds over its merge-base
    late = {r for r in _LATE_RULES if not any(x["rule"] == r for x in rows)} if own else set()
    if branch:
        # a branch push is judged on what it adds: findings already in its
        # merge-base with main belong to main (and are refused on main's push)
        mb = _git("merge-base", "refs/remotes/origin/main", ref)
        if mb.returncode != 0:
            mb = _git("merge-base", "origin/main", ref)
        if mb.returncode == 0:
            base_found = [(k, 0) for k in _findings_at(mb.stdout.strip()).elements()
                          if k[0] not in _OWED_RULES]
            base_debt = collections.Counter(_key(r) for r in rows if r["state"] in _DEBT_STATES)
            for k, _ in base_found:
                if base_debt[k] > 0:
                    base_debt[k] -= 1
                else:
                    allowed_extra[k] += 1
            if allowed_extra and not quiet:
                print(f"no-host-routes: {sum(allowed_extra.values())} finding(s) already on main "
                      "are main's to fix, not charged to this branch", file=sys.stderr)
    elif not own or late:
        # the tree predates the baseline: judge only what the branch adds over
        # its merge-base with main (main's later fixes are not charged to it)
        mb = _git("merge-base", "refs/remotes/origin/main", ref)
        if mb.returncode != 0:
            mb = _git("merge-base", "origin/main", ref)
        # main never grandfathers itself: when the merge-base IS the tree
        # (main, or a tree at main), a late rule's findings are its own, so
        # they are reported (lane H2, 2026-10-09: `--tree origin/main` passed
        # with 14 Python glue findings because main's baseline has no py-* row)
        tip = _git("rev-parse", ref + "^{commit}")
        at_main = (own and mb.returncode == 0 and tip.returncode == 0
                   and tip.stdout.strip() == mb.stdout.strip())
        if mb.returncode == 0 and not at_main:
            allowed_extra = collections.Counter(k for k in _findings_at(mb.stdout.strip()).elements()
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
    bad = bool(new or stale or (own and matched_infl) or grown_refused)
    if not quiet:
        summary(rows)
        owed_summary(owed)
        verdicts = collections.Counter()
        drv = tree_driver_loops(tree, verdicts=verdicts)
        if drv:
            print(f"no-host-routes driver-loop (allowed, not debt): {sum(drv.values())} tagged host loops over "
                  f"model structure ({verdicts['structure']} judged structure, {verdicts['unknown']} could not "
                  f"be judged and are kept, {verdicts['small-loop']} constant-bound small-loop notes); top files: "
                  + ", ".join(f"{k} {v}" for k, v in drv.most_common(8)), file=sys.stderr)
    if new:
        print(f"no-host-routes: REFUSED. {len(new)} finding(s) of CPU work in GPU code "
              "are not in the baseline:", file=sys.stderr)
        for (rule, p, n, _), no in new:
            print(f"  {p}:{no}: [{rule}] {_why(rule, p, n)}: {n[:120]}", file=sys.stderr)
        print("  Put the step on the GPU, in parallel. The baseline never grows.", file=sys.stderr)
    if stale:
        print(f"no-host-routes: REFUSED. {len(stale)} baseline row(s) no longer match "
              "(the CPU work is gone: delete the rows with "
              "`python3 tools/hooks/no_host_routes.py --prune-baseline`):", file=sys.stderr)
        for rule, p, n, occ in stale[:40]:
            print(f"  {p}: [{rule}] #{occ}: {n[:110]}", file=sys.stderr)
        if len(stale) > 40:
            print(f"  ... {len(stale) - 40} more", file=sys.stderr)
    if grown_refused:
        print(f"no-host-routes: REFUSED. {sum(grown_refused.values())} baseline row(s) are not in main's "
              "baseline and main's tree does not carry them (the baseline only shrinks):", file=sys.stderr)
        for (rule, p, n, occ), _c in sorted(grown_refused.items())[:40]:
            print(f"  {p}: [{rule}] #{occ}: {n[:110]}", file=sys.stderr)
    if own and matched_infl:
        print(f"no-host-routes: REFUSED. {len(matched_infl)} in-flight row(s) are now in the "
              "tree: mark them debt (`--prune-baseline`):", file=sys.stderr)
        for (rule, p, n, _), no in matched_infl[:20]:
            print(f"  {p}:{no}: [{rule}] {n[:110]}", file=sys.stderr)
    return 1 if bad else 0


def _main_sha():
    for m in ("refs/remotes/origin/main", "origin/main"):
        r = _git("rev-parse", "--verify", "--quiet", m + "^{commit}")
        if r.returncode == 0 and r.stdout.strip():
            return r.stdout.strip()
    return None


# The tree the 2026-11-09 baseline was regenerated from (the integration head
# the loop-tag rules were written on). Its findings were debt before the rules
# could see them; a tree that contains it may carry them until main does.
_REGEN_ANCHOR = "e68654a7579a3abeef7a97c63ca1a001a7de8a8f"


def _restrict_to_main(debt_keys, ref, quiet=False):
    """(debt keys that hold, keys refused): the baseline only shrinks. A row
    main's baseline (and the ref's baseline where it left main) lacks holds only for a finding main's own tree carries
    under this checker, or the regeneration anchor's tree when ref contains
    the anchor (debt that a stricter rule made visible); a finding a branch
    adds never does."""
    main = _main_sha()
    if main is None:
        return debt_keys, collections.Counter()
    r = _git("show", f"{main}:{BASELINE}")
    if r.returncode != 0:
        return debt_keys, collections.Counter()
    main_keys = collections.Counter(_key(x) for x in load_baseline(r.stdout) if x["state"] in _DEBT_STATES)
    # rows the ref already had where it left main (an old tree judged later,
    # after main fixed them) are its own history, not growth
    mb = _git("merge-base", main, ref)
    if mb.returncode == 0 and mb.stdout.strip() and mb.stdout.strip() != main:
        r2 = _git("show", f"{mb.stdout.strip()}:{BASELINE}")
        if r2.returncode == 0:
            main_keys |= collections.Counter(_key(x) for x in load_baseline(r2.stdout)
                                             if x["state"] in _DEBT_STATES)
    grown = debt_keys - main_keys
    if not grown:
        return debt_keys, collections.Counter()
    tip = _git("rev-parse", ref + "^{commit}").stdout.strip()
    on_main = grown if tip == main else _findings_at(main)
    refused = grown - on_main
    if refused and _git("merge-base", "--is-ancestor", _REGEN_ANCHOR, ref).returncode == 0:
        refused = refused - _findings_at(_REGEN_ANCHOR)
    if not quiet and grown - refused:
        print(f"no-host-routes: {sum((grown - refused).values())} baseline row(s) main's baseline lacks hold "
              "because main's tree carries the finding", file=sys.stderr)
    return debt_keys - refused, refused


_FINDINGS_CACHE = {}


def _findings_at(sha):
    """Counter of the finding keys (owed rules included) of a commit."""
    if sha not in _FINDINGS_CACHE:
        _FINDINGS_CACHE[sha] = collections.Counter(k for k, _ in tree_findings(Tree(sha)))
    return _FINDINGS_CACHE[sha]


_OWED_CACHE = {}


def _owed_findings(sha):
    if sha not in _OWED_CACHE:
        _OWED_CACHE[sha] = collections.Counter(
            {k: c for k, c in _findings_at(sha).items() if k[0] in _OWED_RULES})
    return _OWED_CACHE[sha]


def owed_allowance(ref):
    """Owed findings a tree may carry: those already in its merge-base with the
    owed anchor, with the 2026-11-09 regeneration tree (_REGEN_ANCHOR: loops
    the judged tags made visible) and with main (the larger count of each).
    Everything else of an owed rule is new CPU work and is refused."""
    allowed = collections.Counter()
    bases = set()
    for other in (_OWED_ANCHOR, _REGEN_ANCHOR, "refs/remotes/origin/main", "origin/main"):
        mb = _git("merge-base", other, ref)
        if mb.returncode == 0 and mb.stdout.strip():
            bases.add(mb.stdout.strip())
    for b in sorted(bases):
        allowed |= _owed_findings(b)
    return allowed


def owed_summary(owed, out=None):
    """The owed findings, by rule and by file: CPU work that a fix lane owes
    (~/mojolearn-evidence/cpu-reaudit-2026-10-04/checker-new-findings.tsv),
    never accepted as debt."""
    out = out or sys.stderr
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
                out.append(f"{p}:{no}: [{rule}] {_why(rule, p, _norm(t))}: {_norm(t)[:120]}")
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
        return check_tree(ref, bl, branch="--branch" in args)
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
