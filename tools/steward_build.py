#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""BUILD ONCE, RUN MANY: the stewards' shared binding store
(lane/steward-throughput, 2026-09-27).

A steward request used to build every binding its lanes run inside its own
worktree, so on do-amd a build held a slot (and the FIFO) while the other
identity slots sat idle, and two requests at one commit built the same
bindings twice. The store separates the build from the run:

    <store>/objects/<binding>-<digest>/{<binding>.so, stamp.json}
        a built binding, keyed by the digest of its source closure
        (tools/binding_stamps.py: the digest `stale()` compares), so any
        request whose tree has that closure can use it, whatever its commit
    <store>/objects/libMojolearnMath-<hash>/...     the portable math library
    <store>/commits/<commit>/lanes/<lane>        this lane's bindings are all in
                                                 the store for this commit (READY)
    <store>/commits/<commit>/nobuild/<lane>      the prebuild could not build them;
                                                 the run builds (and reports) itself
    <store>/commits/<commit>/build-*.log         the one build's log

    steward_build.py build   --root WT --store S --commit C --lanes a,b
        derives each lane's bindings with WT's own tools/algos_lane_check.py,
        seeds WT from the store, builds what is still stale (the lane check's
        own build()), publishes, and marks each lane READY or NOBUILD
    steward_build.py seed    --root WT --store S
        copies into WT every stored binding whose digest equals WT's closure
        digest (and that WT does not already hold fresh): the lane check then
        finds it fresh and does not rebuild it
    steward_build.py publish --root WT --store S
        copies WT's fresh bindings (stamp digest == WT's closure) into the store

A binding is only ever copied when its stamp's digest equals the digest of
the tree it lands in, so a stored .so can never answer for different
sources. Copies land by write-then-rename. The check still compares NUMBERS
(CPU == GPU); nothing here compares .so digests (gfx942 codegen from a cold
Mojo cache varies run to run: the store only saves rebuilding what the box
already built from the same sources).

It imports the lane check and binding_stamps FROM THE TREE (WT/tools), so a
request at an older commit is built exactly as its own check would build it.
Run it in the tree's pixi default environment (the builds need mojo).
"""
import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time
from pathlib import Path

# the tree's tools, never this file's directory (a steward runs this from its
# own clone against a worktree at another commit)
if sys.path and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    sys.path.pop(0)


def load_check(root):
    """The tree's tools/algos_lane_check.py as a module (it puts root/tools on sys.path)."""
    path = Path(root) / "tools" / "algos_lane_check.py"
    spec = importlib.util.spec_from_file_location("steward_tree_lane_check", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _stamps_mod(root):
    import binding_stamps          # root/tools is first on sys.path once load_check ran
    if Path(binding_stamps.__file__).resolve().parents[1] != Path(root).resolve():
        raise SystemExit(f"binding_stamps came from {binding_stamps.__file__}, not {root}/tools")
    return binding_stamps


def _copy(src, dst):
    """write-then-rename: a reader never sees half a file, and a .so another
    process has mapped is replaced, never rewritten in place"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.{os.getpid()}.tmp")
    shutil.copy2(src, tmp)
    os.replace(tmp, dst)


def _math_paths(root):
    pkg = Path(root) / "python" / "mojolearn"
    lib = pkg / (".dylibs/libMojolearnMath.dylib" if sys.platform == "darwin" else ".libs/libMojolearnMath.so")
    return lib, lib.with_name(lib.name + ".lanecheck-stamp")


def _math_hash(root):
    src = Path(root) / "packaging" / "portable_math"
    h = hashlib.sha256()
    for name in ("portable_math.c", "powers_of_ten.h", "stage.py"):
        p = src / name
        if not p.is_file():
            return None
        h.update(p.read_bytes())
    return h.hexdigest()


class Tree:
    """One worktree's view: which binding is which script, and its digests."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.lc = load_check(self.root)
        self.bs = _stamps_mod(self.root)
        self._dig = {}

    def digest(self, binding):
        if binding not in self._dig:
            script = self.lc.script_for(binding)
            if not (self.root / "bindings" / script).is_file():
                self._dig[binding] = None
            else:
                self._dig[binding] = self.bs.digest(script, self.root)["digest"]
        return self._dig[binding]

    def so(self, binding):
        return self.lc.output_for(binding)

    def stamp(self, binding):
        return self.bs.stamp_path(self.so(binding), self.bs.STAMPS, self.bs.PKG)

    def fresh(self, binding):
        return self.lc.stale(binding) is None

    def built(self):
        """every binding .so in the tree that carries a stamp"""
        out = []
        for so in self.bs.bindings(self.bs.PKG):
            b = so.name[:-len(".so")] if so.name.endswith(".so") else None
            if b and so.resolve() == self.so(b).resolve() and self.stamp(b).is_file():
                out.append(b)
        return out


def _obj(store, binding, digest):
    return Path(store) / "objects" / f"{binding}-{digest}"


def import_one(tree, store, binding):
    """Copy `binding` in from the store when an object with this tree's
    closure digest is there. True when it landed (and is now fresh)."""
    d = tree.digest(binding)
    if d is None:
        return False
    obj = _obj(store, binding, d)
    so, st = obj / f"{binding}.so", obj / "stamp.json"
    if not (so.is_file() and st.is_file()):
        return False
    if json.loads(st.read_text()).get("digest") != d:
        return False
    _copy(so, tree.so(binding))
    _copy(st, tree.stamp(binding))
    os.utime(obj)                      # pruning keeps what is used
    return tree.fresh(binding)


def publish_one(tree, store, binding):
    """Copy the tree's fresh `binding` into the store (once per digest)."""
    if not tree.fresh(binding):
        return False
    d = json.loads(tree.stamp(binding).read_text())["digest"]
    obj = _obj(store, binding, d)
    if (obj / f"{binding}.so").is_file() and (obj / "stamp.json").is_file():
        os.utime(obj)
        return False
    tmpdir = obj.with_name(f".{obj.name}.{os.getpid()}.tmp")
    shutil.rmtree(tmpdir, ignore_errors=True)
    tmpdir.mkdir(parents=True)
    shutil.copy2(tree.so(binding), tmpdir / f"{binding}.so")
    shutil.copy2(tree.stamp(binding), tmpdir / "stamp.json")
    try:
        os.rename(tmpdir, obj)        # atomic; a concurrent publisher of the same digest wins, same sources
    except OSError:
        shutil.rmtree(tmpdir, ignore_errors=True)
        return False
    return True


def math_seed(tree, store):
    h = _math_hash(tree.root)
    lib, stamp = _math_paths(tree.root)
    if h is None or (lib.is_file() and stamp.is_file() and stamp.read_text().strip() == h):
        return False
    obj = Path(store) / "objects" / f"libMojolearnMath-{h}"
    if not (obj / lib.name).is_file():
        return False
    _copy(obj / lib.name, lib)
    stamp.write_text(h + "\n")
    return True


def math_publish(tree, store):
    h = _math_hash(tree.root)
    lib, stamp = _math_paths(tree.root)
    if h is None or not (lib.is_file() and stamp.is_file() and stamp.read_text().strip() == h):
        return False
    obj = Path(store) / "objects" / f"libMojolearnMath-{h}"
    if (obj / lib.name).is_file():
        return False
    _copy(lib, obj / lib.name)
    return True


def _stored_bindings(store):
    """binding names that have at least one object in the store"""
    names = set()
    for p in (Path(store) / "objects").glob("_mojolearn*-*"):
        if p.is_dir() and not p.name.startswith("."):
            names.add(p.name.rsplit("-", 1)[0])
    return sorted(names)


def seed(tree, store, only=None):
    """Every stored binding (or those in `only`) this tree lacks fresh: copied in
    when its digest matches. Returns the bindings seeded."""
    got = []
    for b in (sorted(only) if only else _stored_bindings(store)):
        if tree.fresh(b):
            continue
        if import_one(tree, store, b):
            got.append(b)
    if math_seed(tree, store):
        got.append("libMojolearnMath")
    return got


def publish(tree, store, only=None):
    out = [b for b in (sorted(only) if only else tree.built()) if publish_one(tree, store, b)]
    if math_publish(tree, store):
        out.append("libMojolearnMath")
    return out


def prune(store, keep=4, keep_commits=40):
    """At most `keep` objects per binding (newest use first) and the newest
    `keep_commits` commit marker dirs."""
    objs = {}
    for p in (Path(store) / "objects").glob("*-*"):
        if p.is_dir() and not p.name.startswith("."):
            objs.setdefault(p.name.rsplit("-", 1)[0], []).append(p)
    for ps in objs.values():
        for p in sorted(ps, key=lambda x: x.stat().st_mtime, reverse=True)[keep:]:
            shutil.rmtree(p, ignore_errors=True)
    cs = sorted((p for p in (Path(store) / "commits").glob("*") if p.is_dir()),
                key=lambda x: x.stat().st_mtime, reverse=True)
    for p in cs[keep_commits:]:
        shutil.rmtree(p, ignore_errors=True)


def build(tree, store, commit, lanes):
    """The one build for (commit, lanes): READY or NOBUILD marker per lane."""
    cdir = Path(store) / "commits" / commit
    for sub in ("lanes", "nobuild"):
        (cdir / sub).mkdir(parents=True, exist_ok=True)
    log = cdir / f"build-{time.strftime('%Y%m%dT%H%M%S')}-{os.getpid()}.log"
    log.touch()
    print(f"[steward-build] {commit[:12]} lanes {','.join(lanes)} log {log}", flush=True)
    needed = {}
    for lane in lanes:
        try:
            needed[lane] = tree.lc.needed_bindings([lane])[lane]
        except Exception as exc:          # the lane check's Fail (no CPU arm, ...): the run reports it
            if isinstance(exc, KeyError):
                exc = f"lane {lane} is not in this tree's lane map"
            (cdir / "nobuild" / lane).write_text(f"{exc}\n")
            print(f"[steward-build] {lane}: NOBUILD ({str(exc)[:200]})", flush=True)
    every = sorted(set().union(*needed.values())) if needed else []
    seeded = seed(tree, store, every)
    if seeded:
        print(f"[steward-build] seeded from the store: {', '.join(seeded)}", flush=True)
    t0 = time.time()
    for lane, bs in needed.items():
        try:
            tree.lc.ensure_built(set(bs), log)
        except Exception as exc:
            (cdir / "nobuild" / lane).write_text(f"{exc}\n")
            print(f"[steward-build] {lane}: NOBUILD ({str(exc)[:200]})", flush=True)
            continue
        published = publish(tree, store, bs)
        missing = [b for b in bs if not tree.fresh(b)]
        if missing:
            (cdir / "nobuild" / lane).write_text(f"not fresh after the build: {', '.join(missing)}\n")
            continue
        (cdir / "lanes" / lane).write_text(json.dumps({"bindings": bs, "published": published,
                                                       "log": str(log)}) + "\n")
        print(f"[steward-build] {lane}: READY ({len(bs)} bindings; published {', '.join(published) or 'none new'})",
              flush=True)
    print(f"[steward-build] done in {time.time() - t0:.0f}s", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("cmd", choices=("build", "seed", "publish", "prune"))
    ap.add_argument("--root", default=".")
    ap.add_argument("--store", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--lanes", default="")
    a = ap.parse_args(argv)
    if a.cmd == "prune":
        prune(a.store)
        return 0
    tree = Tree(a.root)
    if a.cmd == "build":
        lanes = [x for x in a.lanes.split(",") if x]
        if not (a.commit and lanes):
            ap.error("build needs --commit and --lanes")
        return build(tree, a.store, a.commit, lanes)
    if a.cmd == "seed":
        got = seed(tree, a.store)
        print(f"SEEDED {len(got)}: {', '.join(got)}", flush=True)
        return 0
    got = publish(tree, a.store)
    print(f"PUBLISHED {len(got)}: {', '.join(got)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
