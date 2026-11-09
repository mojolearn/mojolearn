#!/usr/bin/env python3
"""Refuse compute in shell files: a Python shell .mojo file is a passthrough.

The rule (Andrew, 2026-10-21): a SHELL file is the Python door

    bindings/**            the Python extension entry points
    **/*pyapi*.mojo        a kernel's sibling Python shell (<name>_pyapi.mojo)
    **/*_py.mojo           the same, older spelling

and may only unpack PythonObject arguments into typed values, call one typed
kernel entry and convert the typed result back. It must not loop over data or
launch work. This checker reports, in every shell file:

    loop     a line starting with `for` or `while` that carries no accepted tag.
             The tag sits on the loop line itself or on a comment-only line
             directly above it, and it is checked against the loop:
               `# driver-loop: over <what>` holds only when the loop walks model
               structure (the LOOP-TAG VOCABULARY below judges the iterable,
               then the loop variables, then the tag text); a tag on a loop
               over data is a `false-tag` hit;
               `# small-loop(<bound>...)` holds only when <bound> is a literal
               or a named constant (UPPER_CASE) that the loop line uses; any
               other bound is a `small-bound` hit.
             `# cpu-route:` and any other comment exempt nothing: a cpu-route
             loop is debt to remove (AGENTS.md, "No Python in the runtime").
    inline   Python iteration inside one expression (the audit's c-inline
             class): `b.sorted`, `b.map`, `b.zip` (also through `bi.` and
             `builtins.`), `.tolist`, `PyObject_GetIter`, `PyIter_Next`,
             `itertools.repeat`, `itertools.chain`, `.fromkeys` (never taggable)
    launch   `enqueue_function`, `DeviceContext(`, `parallelize(`/`parallelize[`,
             `vectorize(`/`vectorize[`, anywhere in code (never taggable)

Comment lines and triple-quoted docstrings are skipped. Hits that main carries
are listed in tools/hooks/compute_in_shells_baseline.txt as `path:line: code`
(written 2026-10-21, regenerated once on 2026-11-09 when the tags started to be
checked); a hit is grandfathered by a row with the same path and the same code
text (the line number is informational, so edits above a row do not break it).
The thin-* lanes move the work into kernel files and the list only shrinks.

Modes:
    no_compute_in_shells.py --tree [REF] [--branch] [--baseline FILE]
        Scan every shell .mojo file committed at REF (default HEAD). Fail (exit 1)
        on any hit not covered by the baseline. The baseline is REF's file,
        restricted to the rows main's baseline also has or, for a row main's
        baseline lacks, a hit main's own tree carries under this checker, or the
        tree the 2026-11-09 regeneration was made from when REF contains it
        (debt that a stricter rule made visible), so a branch cannot grow it
        with a hit of its own. Prints hit counts per kind and directory, and how many
        driver-loop tags were judged structure or could not be judged. Rows no
        hit matches are stale.
        --branch (the pre-push hook passes it for every ref except main): judge
        only the shell files REF adds or changes over merge-base(origin/main, REF);
        a hit that was already in that file at the merge-base is main's debt.
    no_compute_in_shells.py --prune-baseline [REF]
        Rewrite the worktree baseline without its stale rows and with current line
        numbers. It never adds a row.
Exit 0 clean, 1 on findings, 2 on usage errors. Pure git and regex; no Mojo, no build.
The files are scanned by worker processes: MOJOLEARN_HOOK_WORKERS=N sets their
number (any platform); unset, it is 4 on macOS and the core count elsewhere.
"""
import os
import re
import subprocess
import sys
from collections import Counter

BASELINE = "tools/hooks/compute_in_shells_baseline.txt"
LOOP = re.compile(r"^\s*(for|while)\b")
LAUNCH = re.compile(r"enqueue_function|\bDeviceContext\s*\(|\b(?:sync_)?parallelize\s*[\[(]|\bvectorize\s*[\[(]")
# Python iteration inside one expression (audit A's c-inline class, 2026-11-08)
INLINE = re.compile(r"\b(?:b|bi|builtins)\.(?:sorted|map|zip)\b|\.tolist\b|\bPyObject_GetIter\b|\bPyIter_Next\b"
                    r"|\bitertools\.(?:repeat|chain)\b|\.fromkeys\b")
DRIVER_TAG = re.compile(r"#\s*driver-loop:\s*(?:over\s+)?(?P<what>.*)$")
SMALL_TAG = re.compile(r"#\s*small-loop\(\s*(?P<bound>[^:()]*?)\s*(?::[^()]*)?\)")
CONST_BOUND = re.compile(r"[\d_]+|[A-Z][A-Z0-9_]*")
ROW = re.compile(r"^(?P<path>[^:\s]+\.mojo):(?P<line>\d+):\s?(?P<code>.*)$")

# ------------------------------------------------- loop-tag vocabulary ----
# THE LOOP-TAG VOCABULARY (2026-11-09). A `# driver-loop: over <what>` tag
# claims that a loop walks model structure, never data
# (docs/plans/GPU_PATH_HOST_LOOPS.md). The checker judges the claim from the
# loop: first its source (the `for x in <expr>` iterable, the `range(<expr>)`
# bound, the `while` condition), then, when the source says nothing, its loop
# variables, then the tag text. Every identifier is split into lower-case
# words (on `_`, digits and case changes):
#   DATA_PATTERNS   spellings that are data by themselves
#   DATA_NAMES      whole identifiers, or first words, that name the data (X, y)
#   DATA_WORDS      a word that names data: rows, samples, labels, values, ...
#   VALUE_WORDS     a COUNT of these is structure (`range(n_classes)`), the
#                   sequence itself is data (`for c in classes_`)
#   STRUCTURE_WORDS a word that names model structure: layers, blocks, heads, ...
#   COUNT_WORDS     a first word that makes the identifier a count of the rest
# Before the words are read, a dict view (`.keys()`, `.values()`, `.items()`)
# is judged by its receiver and an attribute chain by its last name.
# In the loop's source or variables a data word anywhere makes it data; else a
# structure word makes it structure. In the tag text a structure word wins and
# a data word refutes the tag only alone. Otherwise the verdict is "unknown": the tag is kept, as
# before, and counted as "could not judge". This table is the one place to
# extend; tools/hooks/no_host_routes.py carries the same table (each hook is
# installed as one self-contained file), keep the two in step.
DATA_PATTERNS = re.compile(r"\.shape\s*\[\s*0\s*\]|\.tolist\b|\blen\(\s*[XxYy]\w*\s*\)|\bn_samples\b|\bnnz\b"
                           r"|\bdownload\w*|\breadback\w*")
DATA_NAMES = frozenset("x y".split())
DATA_WORDS = frozenset("""
row rows sample samples label labels value values val vals element elements elem elems token tokens
tok toks record records nnz obs observation observations point points query queries pixel pixels
datum data text texts document documents doc docs sentence sentences char chars byte bytes batch
batches chunk chunks numel target targets instance instances example examples
""".split())
VALUE_WORDS = frozenset("class classes category categories unique uniques".split())
STRUCTURE_WORDS = frozenset("""
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
COUNT_WORDS = frozenset("n num nb count total".split())


def _words(ident):
    """Lower-case words of one identifier: `n_samplesIn2` -> [n, samples, in]."""
    out = []
    for part in re.split(r"[_\d]+", ident):
        out.extend(w.lower() for w in re.findall(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", part))
    return out


def judge_text(text, tag_text=False):
    """("data" | "structure" | "unknown", the deciding word) for one text.
    tag_text: the words of a tag's <what>, where a value word (classes) is a
    count, so structure."""
    if not text or not text.strip():
        return "unknown", ""
    m = DATA_PATTERNS.search(text)
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
        if ident == "n" or ident.lower() in DATA_NAMES or (words[0] in DATA_NAMES and ident[:1] in "XxYy"
                                                             and len(words) > 1 and not tag_text):
            if not tag_text:
                return "data", ident
            data = data or ident
            continue
        if words[0] in COUNT_WORDS and len(words) > 1:
            rest = words[1:]
            if any(w in DATA_WORDS for w in rest):
                if not tag_text:
                    return "data", ident
                data = data or ident
                continue
            if not struct and any(w in STRUCTURE_WORDS or w in VALUE_WORDS for w in rest):
                struct = ident
            continue
        if any(w in DATA_WORDS for w in words) or (not tag_text and any(w in VALUE_WORDS for w in words)):
            if not tag_text:
                return "data", ident
            data = data or ident
            continue
        if not struct and any(w in STRUCTURE_WORDS or (tag_text and w in VALUE_WORDS) for w in words):
            struct = ident
    # in a tag's own text a structure word wins ("the batch's classes"): the
    # text refutes the tag only when it names data and no structure
    if struct:
        return "structure", struct
    return ("data", data) if data else ("unknown", "")


def judge_loop(source, targets="", what=""):
    """The verdict on a tagged loop: its source, else its loop variables, else
    the tag text decides (see the vocabulary above)."""
    for text, tag in ((source, False), (targets, False), (what, True)):
        v = judge_text(text, tag)
        if v[0] != "unknown":
            return v
    return "unknown", ""


def loop_parts(code):
    """(source, loop variables) of a `for`/`while` line's code."""
    m = re.match(r"\s*for\s+(.+?)\s+in\s+(.+?):?\s*$", code)
    if m:
        return m.group(2), m.group(1)
    m = re.match(r"\s*while\b\s*(.*?):?\s*$", code)
    if m:
        return m.group(1), ""
    return code, ""


def small_bound_ok(bound, code):
    """A `small-loop(<bound>)` note holds when the bound is a literal or a named
    constant and the loop line uses it."""
    bound = (bound or "").strip()
    return bool(CONST_BOUND.fullmatch(bound)) and re.search(r"(?<!\w)" + re.escape(bound) + r"(?!\w)", code) is not None


def is_shell(path):
    """True for the Python door files this checker judges (see the module docstring)."""
    name = path.rsplit("/", 1)[-1]
    return path.startswith("bindings/") or "pyapi" in name or name.endswith("_py.mojo")


def _code_part(line):
    """The line without its trailing `#` comment (a `#` inside a string literal is kept)."""
    quote = None
    i = 0
    while i < len(line):
        c = line[i]
        if quote:
            if c == "\\":
                i += 2
                continue
            if c == quote:
                quote = None
        elif c in "\"'":
            quote = c
        elif c == "#":
            return line[:i]
        i += 1
    return line


def scan(text, stats=None):
    """[(line number, kind, code)] of the loop, inline and launch hits in one
    shell file (kinds: loop, false-tag, small-bound, inline, launch). stats, a
    Counter, receives the accepted tags: driver-structure, driver-unknown
    (could not judge) and small-loop."""
    out = []
    lines = text.splitlines()
    in_doc = None
    for n, raw in enumerate(lines, 1):
        s = raw.strip()
        if in_doc:
            if in_doc in s:
                in_doc = None
            continue
        q = '"""' if '"""' in s else ("'''" if "'''" in s else None)
        if q:
            # a line with a triple quote is docstring or string text, never a loop or a launch
            if s.count(q) % 2 == 1:
                in_doc = q
            continue
        if not s or s.startswith("#"):
            continue
        code = _code_part(raw)
        if LOOP.match(code):
            kind = _loop_kind(raw, code, lines[n - 2] if n >= 2 else "", stats)
            if kind:
                out.append((n, kind, s))
        if LAUNCH.search(code):
            out.append((n, "launch", s))
        elif INLINE.search(code):
            out.append((n, "inline", s))
    return out


def _loop_kind(raw, code, prev, stats=None):
    """None for a loop whose tag holds, else the hit kind (see scan)."""
    comment = raw[len(code):]
    above = prev.strip() if prev.strip().startswith("#") else ""
    for tag_text in (comment, above):
        d = DRIVER_TAG.search(tag_text)
        if d:
            source, targets = loop_parts(code)
            verdict, _word = judge_loop(source, targets, d.group("what"))
            if verdict == "data":
                return "false-tag"
            if stats is not None:
                stats["driver-" + verdict] += 1
            return None
        m = SMALL_TAG.search(tag_text)
        if m:
            if not small_bound_ok(m.group("bound"), code):
                return "small-bound"
            if stats is not None:
                stats["small-loop"] += 1
            return None
    return "loop"


def _git(*args, inp=None):
    return subprocess.run(["git", *args], capture_output=True, input=inp)


def _resolve(ref):
    r = _git("rev-parse", "--verify", "--quiet", ref + "^{commit}")
    return r.stdout.decode().strip() if r.returncode == 0 else None


def _shell_blobs(ref):
    """{path: blob sha} of every shell .mojo file at ref."""
    r = _git("ls-tree", "-r", "--full-tree", ref)
    if r.returncode != 0:
        raise SystemExit(f"no_compute_in_shells: cannot list {ref}: {r.stderr.decode().strip()}")
    out = {}
    for ln in r.stdout.decode("utf-8", "replace").splitlines():
        meta, _, path = ln.partition("\t")
        bits = meta.split()
        if len(bits) == 3 and bits[1] == "blob" and path.endswith(".mojo") and is_shell(path):
            out[path] = bits[2]
    return out


def _read(shas):
    """{sha: text}, one `git cat-file --batch` process."""
    need = list(dict.fromkeys(shas))
    if not need:
        return {}
    out = _git("cat-file", "--batch", inp="".join(s + "\n" for s in need).encode()).stdout
    texts, i = {}, 0
    for sha in need:
        nl = out.index(b"\n", i)
        head = out[i:nl].split()
        i = nl + 1
        if len(head) < 3 or head[1] == b"missing":
            texts[sha] = ""
            continue
        size = int(head[2])
        texts[sha] = out[i:i + size].decode("utf-8", "replace")
        i += size + 1
    return texts


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


def _scan_many(items):
    """[(path, hits, stats)] for [(path, text)]: one worker's share."""
    out = []
    for p, t in items:
        st = Counter()
        out.append((p, scan(t, st), st))
    return out


def hits_at(ref, paths=None, stats=None):
    """[(path, line, kind, code)] at ref, sorted by path and line (only `paths`,
    if given). The files are split across every core (one worker per core);
    stats, a Counter, receives the accepted tag counts."""
    blobs = _shell_blobs(ref)
    if paths is not None:
        blobs = {p: s for p, s in blobs.items() if p in paths}
    texts = _read(blobs.values())
    items = [(p, texts[blobs[p]]) for p in sorted(blobs)]
    workers = min(hook_workers(), max(1, len(items) // 64))
    if workers > 1:
        import multiprocessing
        chunks = [items[i::workers] for i in range(workers)]
        try:
            with multiprocessing.get_context("fork").Pool(workers) as pool:
                parts = [r for part in pool.map(_scan_many, chunks) for r in part]
        except (OSError, ValueError):
            parts = _scan_many(items)
    else:
        parts = _scan_many(items)
    res = []
    for p, hs, st in sorted(parts, key=lambda r: r[0]):
        if stats is not None:
            stats.update(st)
        res.extend((p, n, k, c) for n, k, c in hs)
    return res


def parse_baseline(text):
    """[(path, line, code)] rows of a baseline file; comments and blank lines skipped."""
    rows = []
    for ln in text.splitlines():
        m = ROW.match(ln.strip())
        if m and not ln.lstrip().startswith("#"):
            rows.append((m["path"], int(m["line"]), m["code"].strip()))
    return rows


def format_row(path, line, code):
    return f"{path}:{line}: {code}"


def _keys(rows):
    return Counter((r[0], r[-1]) for r in rows)


def _baseline_at(ref):
    r = _git("show", f"{ref}:{BASELINE}")
    return parse_baseline(r.stdout.decode("utf-8", "replace")) if r.returncode == 0 else None


# The tree the 2026-11-09 baseline was regenerated from (the integration head
# the judged tags were written on). Its hits were debt before the rules could
# see them; a tree that contains it may carry them until main does.
REGEN_ANCHOR = "e68654a7579a3abeef7a97c63ca1a001a7de8a8f"


def _main_ref():
    for m in ("refs/remotes/origin/main", "origin/main", "refs/heads/main"):
        if _resolve(m):
            return m
    return None


def _per_dir(hits):
    c = Counter(h[0].rsplit("/", 1)[0] if "/" in h[0] else "." for h in hits)
    return ", ".join(f"{d} {n}" for d, n in sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))) or "none"


_WHAT = {
    "loop": "untagged loop in a shell file; move it into the kernel file (a tag holds only for a model-structure "
            "driver loop `# driver-loop: over <what>` or a constant-bound `# small-loop(<CONST>)`; `# cpu-route:` "
            "and other comments exempt nothing)",
    "false-tag": "false driver-loop tag: the loop walks data, not model structure; move it into the kernel file",
    "small-bound": "small-loop note without a constant bound (it needs a literal or an UPPER_CASE constant the "
                   "loop uses); move the loop into the kernel file",
    "inline": "Python iteration inside an expression in a shell file (b.sorted/b.map/b.zip/.tolist/"
              "PyObject_GetIter/PyIter_Next/itertools/.fromkeys); move it into the kernel file",
    "launch": "kernel launch in a shell file; move it into the kernel file",
}


def check_tree(ref, branch=False, baseline_file=None):
    sha = _resolve(ref)
    if not sha:
        print(f"no_compute_in_shells: unknown ref {ref}", file=sys.stderr)
        return 2
    if baseline_file:
        with open(baseline_file, encoding="utf-8") as f:
            base = _keys(parse_baseline(f.read()))
    else:
        base = _keys(_baseline_at(sha) or [])
    main = _main_ref()
    main_rows = _baseline_at(main) if main else None
    grown = Counter()
    if main_rows is not None:
        main_base = _keys(main_rows)
        grown = base - main_base
        if grown:
            # a row main's baseline lacks holds only for a hit main's own tree
            # carries (main's debt that a stricter rule made visible, as on the
            # 2026-11-09 regeneration); a hit a branch adds is never grandfathered
            on_main = _keys(hits_at(main)) if _resolve(main) != sha else Counter(grown)
            if _resolve(REGEN_ANCHOR) and _git("merge-base", "--is-ancestor", REGEN_ANCHOR, sha).returncode == 0:
                on_main = on_main | _keys(hits_at(REGEN_ANCHOR))
            main_base = main_base | (grown & on_main)
            grown = base - main_base
        base = base & main_base  # the baseline only shrinks: rows main lacks are not grandfathered
    stats = Counter()
    hits = hits_at(sha, stats=stats)
    judged = hits
    allow = Counter(base)
    if branch and main:
        mb = _git("merge-base", main, sha).stdout.decode().strip()
        if mb and mb != sha:
            changed = set(_git("diff", "--name-only", "--no-renames", mb, sha, "--", "*.mojo").stdout.decode().split())
            judged = [h for h in hits if h[0] in changed]
            old = _keys(hits_at(mb, changed))
            allow = Counter({k: max(base[k], old[k]) for k in set(base) | set(old)})
    seen = Counter()
    new = []
    for h in judged:
        key = (h[0], h[3])
        seen[key] += 1
        if seen[key] > allow[key]:
            new.append(h)
    stale = Counter() if branch else base - _keys(hits)
    kinds = Counter(h[2] for h in hits)
    print(f"no_compute_in_shells: {len(hits)} hits in {len({h[0] for h in hits})} of {len(_shell_blobs(sha))} shell .mojo files scanned at {sha[:9]} "
          f"({kinds['loop']} untagged loops, {kinds['false-tag']} false driver-loop tags, "
          f"{kinds['small-bound']} small-loop notes without a constant bound, {kinds['inline']} inline iterations, "
          f"{kinds['launch']} launches; baseline {sum(base.values())}); "
          f"per directory: {_per_dir(hits)}", file=sys.stderr)
    print(f"no_compute_in_shells: accepted tags: {stats['driver-structure']} driver-loop tags judged structure, "
          f"{stats['driver-unknown']} driver-loop tags could not be judged (kept), "
          f"{stats['small-loop']} small-loop notes with a constant bound", file=sys.stderr)
    for (p, c), n in sorted(grown.items()):
        print(f"no_compute_in_shells: baseline row {p}: {c} (x{n}) is not in main's baseline; "
              f"the baseline only shrinks", file=sys.stderr)
    for (p, c), n in sorted(stale.items()):
        print(f"no_compute_in_shells: stale baseline row {p}: {c} (x{n}) matches no hit: "
              f"remove it with --prune-baseline", file=sys.stderr)
    for p, n, k, c in new:
        print(f"{p}:{n}: {_WHAT[k]}: {c}", file=sys.stderr)
    return 1 if new else 0


def prune_baseline(ref):
    sha = _resolve(ref)
    if not sha:
        print(f"no_compute_in_shells: unknown ref {ref}", file=sys.stderr)
        return 2
    top = _git("rev-parse", "--show-toplevel").stdout.decode().strip()
    path = os.path.join(top, BASELINE)
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    current = {}
    for p, n, _k, c in hits_at(sha):
        current.setdefault((p, c), []).append(n)
    used = Counter()
    keep, dropped = [], 0
    for ln in lines:
        m = ROW.match(ln.strip())
        if not m or ln.lstrip().startswith("#"):
            keep.append(ln)
            continue
        key = (m["path"], m["code"].strip())
        nums = current.get(key, [])
        if used[key] < len(nums):
            keep.append(format_row(key[0], nums[used[key]], key[1]))
            used[key] += 1
        else:
            dropped += 1
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(keep) + "\n")
    print(f"no_compute_in_shells: removed {dropped} stale rows from {BASELINE}", file=sys.stderr)
    return 0


def main(argv):
    args = list(argv)
    branch = "--branch" in args
    args = [a for a in args if a != "--branch"]
    baseline_file = None
    if "--baseline" in args:
        i = args.index("--baseline")
        if i + 1 >= len(args):
            print(__doc__, file=sys.stderr)
            return 2
        baseline_file = args[i + 1]
        del args[i:i + 2]
    if args and args[0] == "--tree" and len(args) <= 2:
        return check_tree(args[1] if len(args) == 2 else "HEAD", branch, baseline_file)
    if args and args[0] == "--prune-baseline" and len(args) <= 2:
        return prune_baseline(args[1] if len(args) == 2 else "HEAD")
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
