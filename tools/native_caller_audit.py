#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""NATIVE CALLER AUDIT: a registered native entry or dispatch name that nothing can call.

    python3 tools/native_caller_audit.py [--commit <sha>] [--root DIR] [--list]

A native entry is a module export (`def_function[...]("name")`, `def_py_function`, `def_py_c_function` in a .mojo
file) or a dispatch arm (`operation == "name"`, `owner == "name"`). It is CALLED when its name appears as a whole
word on a code line anywhere else in the tracked .py/.mojo/.sh/.yml/.toml sources (not a comment, not the
registration or the comparison itself, not a manifest that only lists names), or when it is one of the names a
CONSTRUCTED FAMILY below can build at run time. Lanes fb7-dead-entries and fb12-dead-names (2026-11-17) deleted
~100 entries nothing could call; this audit fails when one regrows.

A constructed family is a call site that builds the name from a fixed prefix and a variable part
(`"family_mix_gmm_" + method`, `getattr(module, "xn_" + name)`). Each entry of FAMILIES names that site
(path + the literal that must still be there, so a deleted site retires its family) and the sources of the
variable part: regexes whose first group is the suffix, over named files. A name in a family's prefix whose suffix
no source yields is dead (reported), and a suffix a source yields that no registration has is a latent runtime
error (reported as UNBOUND). Names reached from outside the sources (the verify harness, a framework) and entries
kept on purpose (paused Apple `fast` arms, experiment probes) are listed in KEEP with the reason.

Every tracked source file (git ls-files, or the blobs of --commit) is scanned, split across os.cpu_count() worker
processes. tools/prebuild_check.py runs it in its `audits` part (seconds).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
# A door that dispatches on integer operation codes (bindings/py_keys.mojo) is read in its by-name form, so its
# arms (`operation == "name"`) and its callers are found as before the door was converted.
from boundary_keys_plain import plain  # noqa: E402
SUFFIXES = (".py", ".mojo", ".sh", ".yml", ".yaml", ".toml")
#: never a caller: prose, removed code, stored results
SKIP_DIRS = ("docs/", "experiments/", "bench/results/", "docs-site/")
#: files that only LIST native names (manifests checked against the binding, generated op tables, coverage
#: catalogues); a name they hold is not called by them
MANIFESTS = (
    re.compile(r"^python/mojolearn/_surface_[a-z_]+\.py$"),
    re.compile(r"^python/mojolearn/host_surface\.py$"),
    re.compile(r"^tools/verify_coverage_audit\.py$"),
    re.compile(r"^bindings/native_operation_codes\.mojo$"),   # the name -> integer code table; callers use _OP.<name>
    re.compile(r"^bindings/[a-z0-9_]+_keys_unpack\.mojo$"),   # a family's held keys and operation codes (py_keys.mojo)
    re.compile(r"^bindings/operation_names_[a-z0-9_]+\.mojo$"),  # a family owner's operation names and leaf codes
    re.compile(r"^tools/native_caller_audit\.py$"),
    re.compile(r"^tools/test_native_caller_audit\.py$"),
)
EXPORT = re.compile(r"""\bdef_(?:py_c_|py_)?function\s*(?:\[[^\]]*\]\s*\(|\(\s*[A-Za-z_][A-Za-z0-9_.]*\s*,)\s*"([A-Za-z_][A-Za-z0-9_]*)"\s*\)""")
DISPATCH = re.compile(r"""\b(?:operation|owner)\s*[!=]=\s*"([A-Za-z_][A-Za-z0-9_]*)\"""")
WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
STRING = re.compile(r'"(?:[^"\\]|\\.)*"' + r"|'(?:[^'\\]|\\.)*'")

#: REGISTRATIONS in these trees are compile probes and native test suites, not product entries
REGISTRATION_SKIP = ("tests/", "checks/")


def _src(path_re, regex):
    """A family part read from the sources: group 1 of REGEX over every file whose path matches PATH_RE (a group
    holding quoted words is a tuple of parts)."""
    return ("src", path_re, regex)


#: Every constructed family: SITE holds LITERAL (the family is void, and reported, once it does not); the names it
#: can build are every prefix + suffix. A part is a list of literals or a _src(); prefixes "MODULE" are the module
#: basenames of every bindings/_mojolearn_<module>.mojo. A STRICT family calls without a default (a built name no
#: registration has is an UNBOUND runtime error); the others probe with a default (getattr(..., None), has_entry).
_PUBLIC = r"^python/mojolearn/(?!tests/).*\.py$"
FAMILIES = (
    dict(site="python/mojolearn/_expansion_neighbors.py", literal='getattr(self._bind(), "xn_" + name)', strict=True,
         prefixes=["xn_"],
         suffixes=_src(r"^python/mojolearn/(_expansion_neighbors|tests/_legacy_neighbors_[a-z]+)\.py$",
                       r"""\._op\(\s*["']([a-z0-9_]+)\b"""),
         note="_op(name, ...) calls the export xn_<name>; tests/_legacy_neighbors_*.py are the frozen test oracles"),
    dict(site="providers/python/x_neighbors_descriptor_pyapi_door.mojo", literal='var export_name = "xn_" + name',
         strict=True, prefixes=["xn_"], suffixes=_src(r"\.mojo$", r"""\bxn_execute\(\s*[a-z_]+\s*,\s*"([a-z0-9_]+)\""""),
         note="xn_execute(binding, name, ...) calls the export xn_<name> of a module binding"),
    dict(site="bindings/neighbors_local_native_door.mojo", literal="xn_execute(binding, name, addrs, ints, floats)",
         strict=True, prefixes=["xn_"],
         suffixes=_src(r"^(bindings/neighbors_[a-z_]+_door|neighbors/neighbors_[a-z_]+)\.mojo$",
                       r"""(?:\b_op\(\s*[a-z_]+\s*,\s*"([a-z0-9_]+)"|^\s*return ("[a-z0-9_]+" if [a-z_]+ else "[a-z0-9_]+")$)"""),
         note="the neighbors doors' _op(binding, name, ...) (and imp_cells_operation's two names) -> xn_execute"),
    dict(site="x_neighbors/gen.py", literal="def _add_ops(mut m: PythonModuleBuilder)", strict=False,
         prefixes=["xn_", "x_neighbors_"],
         suffixes=_src(r"^(core|neighbors|bindings)/[a-z0-9_/]+\.mojo$",
                       r"""\b(?:native|driver)\.run\(\s*"([a-z0-9_]+)\""""),
         note="x_neighbors/gen.py registers an export for every op of its OPS table; an op the XnDriver descriptor "
              "path runs (core/*_plan.mojo, core/agglo_native_api.mojo) keeps its generated export"),
    dict(site="bindings/mixture_native_door.mojo", literal='"family_mix_gmm_" + method', strict=True,
         prefixes=["family_mix_gmm_"],
         suffixes=_src(_PUBLIC, r"""(?:["']|_OP\.)family_mix_gmm_public_(fit|fit_predict|bic|aic)\b"""),
         note="GaussianMixture public fit/fit_predict/bic/aic (the queries take _gmm_public_query)"),
    dict(site="python/mojolearn/_native_parallel_public.py", literal="{name:'family_mix_gmm_'+name for name in",
         strict=True, prefixes=["family_mix_gmm_"],
         suffixes=_src(r"^python/mojolearn/_native_parallel_public\.py$",
                       r"""\{name:'family_mix_gmm_'\+name for name in (\([^)]*\))"""),
         note="the parallel GaussianMixture output entries"),
    dict(site="providers/python/hdbscan_public_lifecycle_pyapi_door.mojo", literal='"family_cfa_hdbscan_" + method',
         strict=True, prefixes=["family_cfa_hdbscan_"],
         suffixes=_src(_PUBLIC, r"""(?:["']|_OP\.)family_hdbscan_public_(fit|fit_predict)\b"""),
         note="HDBSCAN public fit/fit_predict"),
    dict(site="providers/python/density_public_lifecycle_pyapi_pack.mojo", literal='"family_cfb_dbscan_" + method',
         strict=True, prefixes=["family_cfb_dbscan_"],
         suffixes=_src(_PUBLIC, r"""(?:["']|_OP\.)family_density_dbscan_(fit|fit_predict|predict|score_samples|score)\b"""),
         note="DBSCAN public fit/fit_predict/predict (getstate/setstate/initialize return before the site)"),
    dict(site="providers/python/density_public_lifecycle_pyapi_pack.mojo", literal='"family_xn_kde_" + method',
         strict=True, prefixes=["family_xn_kde_"],
         suffixes=_src(_PUBLIC, r"""(?:["']|_OP\.)family_density_kde_(fit|fit_predict|predict|score_samples|score)\b"""),
         note="KernelDensity public fit/score/score_samples (getstate/setstate/release/sample return before)"),
    dict(site="providers/python/tree_boost_public_lifecycle_pyapi_pack.mojo", literal='"trees_boost_" + method',
         strict=True, prefixes=["trees_boost_"],
         suffixes=_src(_PUBLIC, r"""(?:["']|_OP\.)trees_public_boost_(fit|fit_transform|predict|predict_proba|"""
                                r"""decision_function|transform)\b"""),
         note="boosting public methods (score and apply return before the site)"),
    dict(site="bindings/trees_public_meta_lifecycle_pack.mojo", literal='"trees_meta_" + method', strict=True,
         prefixes=["trees_meta_"],
         suffixes=_src(_PUBLIC, r"""(?:["']|_OP\.)trees_public_meta_(fit|fit_transform|predict|predict_proba|transform|score)\b"""),
         note="meta-estimator public methods"),
    dict(site="bindings/iforest_public_lifecycle_pack.mojo", literal='"trees_iforest_" + method', strict=True,
         prefixes=["trees_iforest_"],
         suffixes=_src(_PUBLIC, r"""(?:["']|_OP\.)trees_public_iforest_(fit|fit_predict|score_samples|decision_function|predict)\b"""),
         note="IsolationForest public methods"),
    dict(site="python/mojolearn/tokenizer.py", literal='getattr(module, "bpe_" + name, None)', strict=True,
         prefixes=["bpe_"], suffixes=_src(r"^python/mojolearn/tokenizer\.py$", r"""^_ABI = \(([^)]*)\)"""),
         note="_Entries binds bpe_<name> for every name of tokenizer._ABI"),
    dict(site="bindings/mamba_public_native_door.mojo", literal='kind_name + "_session_create"', strict=False,
         prefixes=["mamba1", "mamba2", "mamba3"],
         suffixes=_src(r"^bindings/mamba_public_native_door\.mojo$",
                       r"""(?:kind_name|\bname|_kind_name\([^)]*\))\s*\+\s*"(_[a-z0-9_]+)\""""),
         note="the Mamba session entries of _kind_name(kind)"),
    dict(site="python/mojolearn/_backend.py", literal='getattr(module, prefix + "_" + fn, None)', strict=False,
         prefixes="MODULE",
         suffixes=["_vendor", "_numeric_mode", "_column", "_sabotage", "_detected_column", "_gbdt_ctr_sabotage",
                   "_groves_sabotage"],
         note="the read-back every binding exports under its own prefix (load_host_module, load_door_host, "
              "backend_public_native_door, the runtime report)"),
    dict(site="python/mojolearn/_native_arena_rows.py", literal='getattr(module, prefix + "_dev_put")', strict=False,
         prefixes="MODULE", suffixes=["_dev_put", "_dev_free", "_dev_live", "_dev_take_rows"],
         note="the device arena entries a binding exports under its own prefix (arena rows, arena_prog, x_prep owner)"),
)


#: name -> reason: entries with no caller in the sources that stay on purpose
KEEP = {
    "x_neighbors_kfeat_schi2_draw": "paused Apple fast arm (lane apple-fast-w3-kfeat, XN_FAST_SCHI2_LAZYW, opt-in): "
                                    "kept until Andrew reopens Apple fast",
    "x_cluster_mbk_devscan": "paused Apple fast read-back (MiniBatchKMeans' FAST + Apple NaN/inf device scan)",
    "x_cluster_mbk_policy": "paused Apple fast read-back (MINIBATCH_FAST_DEV policies, x_cluster/minibatch_fast.mojo)",
    "x_decomp_lu_resid": "paused Apple fast arm (lane apple-fast-q-linalg, LU_QFIX)",
    "x_decomp_dev_orth_ws": "paused Apple fast arm (lane apple-fast-s-linalg, DECOMP_FAST_ORTH_WS)",
    "gbdt_fit_rowmajor": "paused Apple fast arm (lane apple-fast-sym-feat, GBDT_QUANT_DEVICE)",
}


def _code(line, suffix):
    """LINE without its comment (a # outside quotes); "" for a comment-only line."""
    if suffix not in (".py", ".mojo", ".sh", ".yml", ".yaml", ".toml"):
        return line
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


def _caller_words(code, suffix):
    """The words of CODE that can call a native name. A string counts only when it is exactly one name (a py_get key,
    an operation literal, a getattr name): a message or docstring that mentions a name calls nothing. In Mojo a bare
    word is a Mojo function of the same name, never the Python export; in Python a bare word is an attribute."""
    exact = [m.group(0)[1:-1] for m in STRING.finditer(code) if IDENT.fullmatch(m.group(0)[1:-1])]
    if suffix == ".mojo":
        return exact
    return WORD.findall(STRING.sub(" ", code)) + exact


def _is_manifest(path):
    return any(p.search(path) for p in MANIFESTS)


def scan_text(path, text):
    """(registrations [(name, kind, path, line)], Counter of caller words) of one file."""
    regs, words = [], collections.Counter()
    manifest = _is_manifest(path)
    suffix = os.path.splitext(path)[1]
    for k, line in enumerate(text.splitlines(), 1):
        code = _code(line, suffix)
        if not code.strip():
            continue
        if suffix == ".mojo" and not path.startswith(REGISTRATION_SKIP):
            for m in EXPORT.finditer(code):
                regs.append((m.group(1), "export", path, k))
            for m in DISPATCH.finditer(code):
                regs.append((m.group(1), "dispatch", path, k))
        if suffix == ".mojo":
            code = DISPATCH.sub(" ", EXPORT.sub(" ", code))
        if manifest:
            continue
        words.update(_caller_words(code, suffix))
    return regs, words


def callers_of(files, names):
    """{name: ['path:line: code']} of every line that counts as a caller of each of NAMES (the --why listing)."""
    names, out = set(names), collections.defaultdict(list)
    for path, text in files:
        if _is_manifest(path):
            continue
        suffix = os.path.splitext(path)[1]
        for k, line in enumerate(text.splitlines(), 1):
            code = _code(line, suffix)
            if suffix == ".mojo":
                code = DISPATCH.sub(" ", EXPORT.sub(" ", code))
            hits = set(_caller_words(code, suffix))
            for name in hits & names:
                out[name].append(f"{path}:{k}: {line.strip()[:160]}")
    return out


def _scan_many(items):
    regs, words = [], collections.Counter()
    for path, text in items:
        r, w = scan_text(path, text)
        regs += r
        words.update(w)
    return regs, words


def _wanted(path):
    return path.endswith(SUFFIXES) and not path.startswith(SKIP_DIRS)


def read_files(root=ROOT, commit=None):
    """[(path, text)] of every tracked source file in ROOT's work tree, or of COMMIT's blobs."""
    if commit is None:
        names = subprocess.run(["git", "ls-files", "-z"], cwd=str(root), capture_output=True,
                               check=True).stdout.decode().split("\0")
        return [(n, plain(n, (Path(root) / n).read_text(errors="replace"))) for n in names
                if n and _wanted(n) and (Path(root) / n).is_file()]
    listing = subprocess.run(["git", "ls-tree", "-r", "-z", commit], cwd=str(root), capture_output=True,
                             check=True).stdout.decode().split("\0")
    rows = [r.split("\t", 1) for r in listing if " blob " in r]
    shas = [(meta.split()[2], path) for meta, path in rows if _wanted(path)]
    proc = subprocess.run(["git", "cat-file", "--batch"], cwd=str(root), check=True, capture_output=True,
                          input="".join(f"{sha}\n" for sha, _ in shas).encode())
    data, pos, out = proc.stdout, 0, []
    for _, path in shas:
        header_end = data.index(b"\n", pos)
        size = int(data[pos:header_end].split()[2])
        out.append((path, plain(path, data[header_end + 1:header_end + 1 + size].decode(errors="replace"))))
        pos = header_end + 1 + size + 1
    return out


def scan(files, workers=None):
    """(registrations, caller word Counter) over FILES, split across WORKERS processes (default os.cpu_count())."""
    workers = max(1, min(workers or os.cpu_count() or 1, len(files) or 1))
    chunks = [files[i::workers] for i in range(workers) if files[i::workers]]
    regs, words = [], collections.Counter()
    if workers == 1:
        return _scan_many(files)
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        for r, w in pool.map(_scan_many, chunks):
            regs += r
            words.update(w)
    return regs, words


def _parts(part, files):
    """{part: 'path:line'} of a family part."""
    if isinstance(part, (list, tuple)) and (not part or part[0] != "src"):
        return {p: "literal" for p in part}
    _, path_re, regex = part
    pr, sr, out = re.compile(path_re), re.compile(regex, re.M), {}
    for path, text in files:
        if not pr.search(path):
            continue
        for m in sr.finditer(text):
            line = text.count("\n", 0, m.start()) + 1
            group = next(g for g in m.groups() if g is not None)
            for p in re.findall(r"""["']([a-z0-9_]+)["']""", group) or [group]:
                out.setdefault(p, f"{path}:{line}")
    return out


def family_names(files, families=FAMILIES):
    """({name: 'site (where its part comes from)'}, {strict-family name: where}, [void family messages])."""
    texts = dict(files)
    modules = sorted(Path(p).stem[len("_mojolearn_"):] for p, _ in files
                     if re.fullmatch(r"bindings/_mojolearn_[a-z0-9_]+\.mojo", p))
    built, strict, void = {}, {}, []
    for fam in families:
        if fam["literal"] not in texts.get(fam["site"], ""):
            void.append(f"{fam['site']} no longer holds {fam['literal']!r}")
            continue
        prefixes = {m: "module" for m in modules} if fam["prefixes"] == "MODULE" else _parts(fam["prefixes"], files)
        suffixes = _parts(fam["suffixes"], files)
        for p, pw in prefixes.items():
            for s, sw in suffixes.items():
                where = sw if sw not in ("literal", "module") else pw
                built.setdefault(p + s, f"{fam['site']} ({where})")
                if fam["strict"]:
                    strict.setdefault(p + s, where)
    return built, strict, void


def verdict(regs, words, built, strict, keep=KEEP):
    """(dead [(name, kind, where)], unbound [(name, where)], stale keep names)."""
    seen = collections.defaultdict(list)
    for name, kind, path, line in regs:
        seen[name].append((kind, f"{path}:{line}"))
    dead = []
    for name in sorted(seen):
        if words.get(name, 0) or name in built or name in keep:
            continue
        kinds = sorted({k for k, _ in seen[name]})
        dead.append((name, "+".join(kinds), seen[name][0][1]))
    unbound = sorted((n, where) for n, where in strict.items() if n not in seen)
    stale = sorted(n for n in keep if n not in seen or words.get(n, 0) or n in built)
    return dead, unbound, stale


def audit(root=ROOT, commit=None, workers=None, out=print, families=FAMILIES, keep=KEEP):
    files = read_files(root, commit)
    regs, words = scan(files, workers)
    built, strict, void = family_names(files, families)
    dead, unbound, stale = verdict(regs, words, built, strict, keep)
    for message in void:
        out(f"NATIVECALL FAIL void family: {message}")
    for name, kind, where in dead:
        out(f"NATIVECALL FAIL no caller: {name} ({kind}) {where}")
    for name, where in unbound:
        out(f"NATIVECALL FAIL unbound: {name} is built at {where} and registered nowhere")
    for name in stale:
        out(f"NATIVECALL FAIL stale KEEP entry: {name}")
    bad = len(void) + len(dead) + len(unbound) + len(stale)
    out(f"NATIVECALL {'FAIL' if bad else 'PASS'} {len({r[0] for r in regs})} native names in {len(files)} files, "
        f"{len(dead)} without a caller, {len(unbound)} unbound, {len(void)} void families, {len(stale)} stale keeps")
    return 1 if bad else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--commit", default=None, help="scan this commit's blobs instead of the work tree")
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--workers", type=int, default=None, help="default os.cpu_count()")
    ap.add_argument("--why", nargs="*", default=None, help="print the lines that count as callers of these names")
    a = ap.parse_args(argv)
    if a.why:
        files = read_files(Path(a.root), a.commit)
        found = callers_of(files, a.why)
        for name in a.why:
            print(f"== {name}")
            for line in found.get(name, ()):
                print("  " + line)
        return 0
    return audit(Path(a.root), a.commit, a.workers)


if __name__ == "__main__":
    sys.exit(main())
