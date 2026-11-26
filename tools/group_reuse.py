#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn
"""GROUP REUSE IN THE GITHUB LEGS (lane build-reuse-1, 2026-11-26): a measurement build's leg gives no shard job to a
merged group whose binding-cache key is already in R2, and the leg prints one line of what its cache did.

WHY. Round 15 (0.8.67.dev0+acdf0ad30312, 2026-11-26) dispatched 15 shard jobs per leg on sm_89 and sm_80, one per
merged group, whatever the cache held. A shard job costs 7 to 12 minutes of a free runner before its compile starts
or after it ends (checkout, image pull, pixi install, the route's setup, the artifact upload: shard 1 ran 812 s for a
394 s compile), and the two legs' 30 jobs queue on about 20 free runners. A group whose key is in R2 only fetched its
archive there. The assemble job already places every group from the cache by its REAL key (tools/bincache.py build:
hot directory, then R2, each archive checked by verify_archive against the key and its fields), so a group without a
shard job is either placed there or, when the cache does not hold its real key, compiled there: slower, never wrong.

HOW A KEY IS PREDICTED ON THE MAC (the method of tools/prebuild_changed.py predict, measured there at 76 of 76 keys,
here for the merged groups). A closure-v2 key (tools/bincache.py KEY_RULE) is sha256 over fields of two halves:
  * the commit's: the group's import closure (`source`), the build scripts it executes (`inputs`), the merged plan
    (`merged`: members and root text), the Linux toolchain of pixi.lock (`toolchain`), and which variables those
    inputs read (the read-set of `build_env`);
  * the leg's: image, OS of the build container, repo path, device arch, numeric mode, GPU archs, target column,
    declared outputs and the values of the variables the route exports.
The leg's half is read from the newest earlier leg of the SAME arch (its GHA/bincache/keys/<key>.json, each checked
to hash to its file name). The commit's half is computed by bincache.build_fields over the commit's source tree with
the commit's own tools/merged_bindings.py. So every byte-deciding input is in the predicted key exactly as the box
keys it: a closure file, an executed script line, the toolchain, the flags and defines the scripts read from the
environment, the target. A predicted key the store does not hold is a shard job, as before.

WHAT IS NEVER USED. A record of a sabotage or fault-injected build (bincache.is_sabotage, `variant`), and a record
of a B arm (MOJOLEARN_BUILD_DEFINES in its build_env): a define set is never the leg half of an A set. Fault builds
never upload in any case (bincache refusal). A release dispatches exactly as before (tools/release.py passes
--predict-skips only for a measurement build's A set): its legs still give every group a shard job, and each shard
takes a hit from R2 under the release's own key, so the release key and the provenance of each shard are unchanged.

THE LINE. After the run, tools/release_github_build.py run prints, from the shards' and the assemble job's
bincache provenance.tsv:
    group cache: hits=H misses=M compiles=C uploads=U secs=S
H groups placed from R2 (in a shard, or in the assemble job for a group with no shard job), M groups whose key was
not in the cache, C groups compiled (in a shard or, after a wrong prediction, in the assemble job), U archives
uploaded to the inbox, S the compiler seconds those compiles took. Hits inside the run (a shard's own archive placed
by the assemble job, `hot-hit`) are not counted. Before the dispatch it prints
    group cache plan: units=N predicted_hits=P shard_jobs=N-P records=<leg dir>
"""
import contextlib
import importlib.util
import json
import re
from pathlib import Path

import bincache

#: the candidate arm's define channel (tools/release_github_build.py normalize_defines): a record that carries it is a
#: B arm's and is never the leg half of an A set
ARM_DEFINES = "MOJOLEARN_BUILD_DEFINES"
#: earlier legs read for records, newest first
MAX_LEGS = 6


def entry_of(fields):
    """The shard entry ("<tier>:merged_build.sh:<group>", release_github_build.plan_set) a recorded merged build
    compiles, or None for any other build."""
    merged = fields.get("merged") or []
    if Path(fields.get("script") or "").name != "merged_build.sh" or len(merged) != 1:
        return None
    return "%s:merged_build.sh:%s" % (merged[0]["dir"], merged[0]["stem"])


def usable(fields, arch):
    """True when FIELDS can be the leg half of a production A-set key of ARCH."""
    if fields.get("key_rule") != bincache.KEY_RULE or fields.get("variant") or fields.get("sabotage_env"):
        return False
    if bincache.is_sabotage(fields) or (fields.get("build_env") or {}).get(ARM_DEFINES):
        return False
    if "declared" not in fields or entry_of(fields) is None:
        return False
    merged = fields["merged"][0]
    # a device group names its target; a host group rides the leg of its arch (each leg's keys dir is its own)
    return merged["dir"] == "host" or fields.get("gpu_archs") == arch


def leg_records(roots, arch, max_legs=MAX_LEGS):
    """[(leg dir, {entry: fields})] of the newest earlier legs of ARCH under the measurement-build work roots, newest
    first: every <root>/<version>/<c12>/legs/<vendor>-<arch>/GHA/bincache/keys/<key>.json whose fields hash to its
    name and are usable. Unreadable files are passed over."""
    legs = []
    for root in roots:
        for keys in Path(root).expanduser().glob("*/*/legs/*-%s/GHA/bincache/keys" % arch):
            if keys.is_dir():
                legs.append(keys)
    legs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    out = []
    for keys in legs[:max_legs]:
        recs = {}
        for p in sorted(keys.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
            try:
                f = json.loads(p.read_text())
            except (OSError, ValueError):
                continue
            if not isinstance(f, dict) or bincache.key_of(f) != p.stem or not usable(f, arch):
                continue
            recs.setdefault(entry_of(f), f)
        if recs:
            out.append((keys.parent.parent.parent, recs))
    return out


def leg_environ(recs):
    """The route's exported variables as one leg's records show them: the union of their build_env (closure-v2 keeps
    only what each group's inputs read; tools/prebuild_changed.py capture), None when two records disagree on a
    value (no one environment reproduces that leg)."""
    env = {}
    for f in recs.values():
        for k, v in (f.get("build_env") or {}).items():
            if env.setdefault(k, v) != v:
                return None
    return env


@contextlib.contextmanager
def commit_merged_bindings(tree):
    """bincache's merged plan read through the COMMIT's tools/merged_bindings.py (the box runs the commit's copy;
    the tooling checkout's may differ), restored after."""
    path = Path(tree) / "tools" / "merged_bindings.py"
    saved = bincache._merged_bindings
    if path.is_file():
        spec = importlib.util.spec_from_file_location("group_reuse_commit_merged_bindings", str(path))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        bincache._merged_bindings = lambda: module
    try:
        yield
    finally:
        bincache._merged_bindings = saved


def linux_toolchain(tree, plat):
    """bincache.toolchain as the Linux build container computes it, whatever machine runs this."""
    lock = Path(tree) / "pixi.lock"
    if not lock.is_file():
        return dict(platform=plat, packages=[], pixi_lock_sha256="")
    text = lock.read_text(errors="replace")
    pkgs = sorted(set(re.findall(
        r"conda\.modular\.com/max/%s/((?:mojo|mojo-compiler|max|max-core)-[^/\s]+)\.conda" % re.escape(plat), text)))
    return dict(platform=plat, packages=pkgs, pixi_lock_sha256=bincache.sha256_file(lock))


def predict(tree, template, environ=None):
    """(key, fields) the build container will compute for TEMPLATE's group at the commit checked out in TREE: the
    commit's half from TREE (bincache.build_fields, plat linux, the commit's merged_bindings.py), the leg's half from
    TEMPLATE. ENVIRON is the leg's exported variables (leg_environ; default the template's own build_env); keyed_env
    keeps only what this commit's inputs read, as the box does. None when the commit has no such group or the key
    cannot be computed."""
    tree = Path(tree).resolve()
    env = dict(template.get("build_env") or {}) if environ is None else dict(environ)
    env["MOJOLEARN_NUMERIC_MODE"] = template.get("numeric_mode") or "identical"
    for name, field in (("MOJOLEARN_GPU_ARCHS", "gpu_archs"), ("MOJOLEARN_TARGET_COLUMN", "target_column")):
        if template.get(field):
            env[name] = template[field]
    declared = template.get("declared") or {}
    try:
        with commit_merged_bindings(tree):
            fields, _ = bincache.build_fields(tree, str(tree / template["script"]), list(template.get("args") or []),
                                              env, template["image"], declared=list(declared.get("outputs") or []),
                                              shell=declared.get("shell", "bash"),
                                              dev_arch=template.get("device_arch", "none"), os_info=template["os"],
                                              plat="linux")
    except (bincache.UnkeyableBuild, OSError, KeyError, ValueError):
        return None
    if entry_of(fields) != entry_of(template):
        return None
    fields["toolchain"] = linux_toolchain(tree, (template.get("toolchain") or {}).get("platform") or "linux-64")
    for k in ("numeric_mode", "gpu_archs", "target_column", "repo_path"):
        fields[k] = template[k]
    if "device_arch" not in template:
        fields.pop("device_arch", None)
    return bincache.key_of(fields), fields


def skip_plan(tree, arch, entries, stored, roots):
    """dict(skip=[entries], keys={entry: predicted key}, records=<leg dir> or None, units=N): the shard ENTRIES of
    this leg whose predicted key is in STORED (the R2 partition's keys). An entry with no record, or whose
    prediction fails, keeps its shard job. Only the newest leg's records are the leg half (one environment); an
    entry it lacks takes the newest older leg's record with the same environment."""
    legs = leg_records(roots, arch)
    out = dict(skip=[], keys={}, records=None, units=len(entries))
    if not legs:
        return out
    newest_dir, newest = legs[0]
    env = leg_environ(newest)
    if env is None:
        return out
    out["records"] = str(newest_dir)
    templates = {}
    for entry in entries:
        template = newest.get(entry)
        if template is None:
            for _, recs in legs[1:]:
                f = recs.get(entry)
                if f is not None and all(env.get(k, v) == v for k, v in (f.get("build_env") or {}).items()):
                    template = f
                    break
        if template is not None:
            templates[entry] = template
    for entry, got in zip(templates, predict_all(tree, list(templates.values()), env)):
        if got is None:
            continue
        key, _ = got
        out["keys"][entry] = key
        if key in stored:
            out["skip"].append(entry)
    return out


def predict_all(tree, templates, env, workers=None):
    """predict() of every template, one process per group (each walks and hashes its closure, about 800 files for
    the big groups; bincache's closure walk keeps module state, so processes, not threads). Workers: half the cores
    (the two legs of a build predict at once on one Mac), at most one per group; the memory per worker is a Python
    interpreter and one closure's file list."""
    import concurrent.futures
    import os
    if not templates:
        return []
    workers = workers or max(1, min(len(templates), (os.cpu_count() or 2) // 2))
    if workers == 1:
        return [predict(tree, t, env) for t in templates]
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(predict, [tree] * len(templates), templates, [env] * len(templates)))


def stored_keys(listing, partition):
    """The production keys of a partition listing ([(name, modified, size)], bincache.list_objects)."""
    prefix = "%s/%s/" % (bincache.OBJECT_PREFIX, partition)
    keys = set()
    for name, _, _ in listing:
        m = re.match(r"^%s([0-9a-f]{64})\.tar\.gz$" % re.escape(prefix), name)
        if m:
            keys.add(m.group(1))
    return keys


def provenance_rows(path):
    """[(outcome, key, seconds)] of one bincache provenance.tsv (date, script, outcome, key, seconds, files)."""
    rows = []
    try:
        text = Path(path).read_text(errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        cells = line.split("\t")
        if len(cells) >= 5:
            try:
                secs = float(cells[4])
            except ValueError:
                secs = 0.0
            rows.append((cells[2], cells[3], secs))
    return rows


def cache_counts(shard_files, assemble_files):
    """dict(hits, misses, compiles, uploads, secs) over the run's groups. A group is counted once, by key: its shard
    row when it had a shard job, else its assemble row. `hot-hit` (the run's own archive) is not a hit."""
    shard = [r for f in shard_files for r in provenance_rows(f)]
    assemble = [r for f in assemble_files for r in provenance_rows(f)]
    hits = misses = compiles = uploads = 0
    secs = 0.0
    shard_keys = {k for _, k, _ in shard}
    # outcomes (tools/bincache.py rec.row): hit (R2 or a seed), hot-hit (this run's own archive), miss and
    # error-key:<why> followed by +built-uploaded, +built-not-uploaded:<why>, +built-cached, +build-failed, ...
    built = lambda o: "+built" in o
    uploaded = lambda o: o.endswith("+built-uploaded")
    for outcome, key, s in shard + [r for r in assemble if r[1] not in shard_keys]:
        if built(outcome):
            compiles += 1
            secs += s
        uploads += uploaded(outcome)
        if outcome.startswith(("miss", "error-key")):
            misses += 1
        elif outcome == "hit":
            hits += 1
    # a group with a shard job compiled again by the assemble job (its archive lost between the jobs) is a compile
    for outcome, key, s in assemble:
        if key in shard_keys and built(outcome):
            compiles += 1
            secs += s
            uploads += uploaded(outcome)
    return dict(hits=hits, misses=misses, compiles=compiles, uploads=uploads, secs=int(round(secs)))


def cache_line(counts):
    return "group cache: hits=%(hits)d misses=%(misses)d compiles=%(compiles)d uploads=%(uploads)d secs=%(secs)d" \
        % counts


def plan_line(plan):
    return "group cache plan: units=%d predicted_hits=%d shard_jobs=%d records=%s" % (
        plan["units"], len(plan["skip"]), plan["units"] - len(plan["skip"]), plan["records"] or "none")
