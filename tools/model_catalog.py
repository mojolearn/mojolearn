#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""THE VERIFIED-MODEL CATALOG (2026-10-20, lane ag-model-catalog): python/mojolearn/models/catalog.json
built from tools/real_model_import.py box runs.

    python3 tools/model_catalog.py build --runs RUN_DIR [RUN_DIR ...] [--catalog PATH] [--check]
    python3 tools/model_catalog.py show [--catalog PATH]

Each RUN_DIR is one `real_model_import.py run --out RUN_DIR` (one JSON per model, schema
mojolearn-real-model/1), typically one per vendor (Apple, NVIDIA, AMD) for one mojolearn version. `build`
merges them into the catalog: per model the exact source (torchvision weights enum member and the torch /
torchvision versions; Hugging Face repo and commit sha; ONNX file URL and sha256, and the converted file's
sha256 when the opset was raised), the input shapes, the import call, the check input (recipe, seed,
digest), the accuracy (torchvision's published top-1; our top-1 on the stated ImageNet slice, the value
only), and per mojolearn version and numeric mode the output digest of each vendor and the latency,
import time and device memory of each vendor's run. Other versions already in the catalog are kept.

THE BUILD REFUSES (exit 1, nothing written) when, for one model, version, numeric mode, batch and seed:
  * the `identical` output digests differ between vendors, between two runs of one vendor, or from the
    digest the catalog already holds for that cell;
  * a run's second output differs from its first (repeat_equal false) in `identical` mode;
  * the check inputs differ (input_digest), or the runs name different sources (weights, revision, sha256);
  * the runs used different batch or seed values.
A failed run (status failed) is reported and left out; it never makes an entry.

Status of an entry, from its newest version's `identical` cell:
  verified          ok on NVIDIA (cuda) and AMD (hip), digests equal (Apple, when present, equal too);
  refused: <ops>    the import refuses the model (the ops it does not support, from the run);
  unverified: ...   ok but not yet run on both NVIDIA and AMD (left out of the docs page).

Only revisions, hashes and numbers are written: never weights, model files or evaluation images (the
catalog ships in the wheel; no third-party data in the distribution).
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "python" / "mojolearn" / "models" / "catalog.json"
SCHEMA = "mojolearn-model-catalog/1"
RUN_SCHEMA = "mojolearn-real-model/1"
GPU_VENDORS = ("cuda", "hip")  # the two that must agree for `verified`
VENDOR_NAMES = {"metal": "Apple", "cuda": "NVIDIA", "hip": "AMD", "cpu": "CPU"}
# the per-run numbers copied into a vendor's cell (whole clock, docs in real_model_import.py)
_PERF = ("gpu_arch", "latency_ms", "first_run_ms", "import_ms", "device_memory_mb", "device_memory_method",
         "repeat_equal")
# source fields that name exactly what was imported; they must agree across the runs of one model
_SOURCE_PIN = ("weights", "weights_url", "repo", "revision", "url", "sha256", "converted_sha256")
_SOURCE_KEEP = _SOURCE_PIN + ("opset", "opset_conversion", "published_top1", "published_eval")
_ENV = ("torch_version", "torchvision_version", "transformers_version", "onnx_version")


class Refused(Exception):
    pass


def _version_key(v):
    out = []
    for part in str(v).replace("+", ".").replace("-", ".").split("."):
        out.append((0, int(part), "") if part.isdigit() else (1, 0, part))
    return out


def load_runs(dirs):
    """{model: [run doc]} from every mojolearn-real-model/1 JSON under the run directories."""
    by_model = {}
    failed = []
    for d in dirs:
        d = Path(d)
        files = sorted(d.glob("*.json")) if d.is_dir() else [d]
        if not files:
            raise Refused(f"{d}: no run JSON files")
        for f in files:
            try:
                doc = json.loads(f.read_text())
            except (OSError, ValueError) as exc:
                raise Refused(f"{f}: unreadable ({exc})") from None
            if doc.get("schema") != RUN_SCHEMA:
                continue  # summary.json and anything else
            doc["_file"] = str(f)
            if doc.get("status") == "failed":
                failed.append(f"{doc.get('model')} ({doc.get('vendor')}, {f}): {str(doc.get('error'))[:160]}")
                continue
            for k in ("model", "vendor", "numeric_mode", "mojolearn_version"):
                if not doc.get(k):
                    raise Refused(f"{f}: no {k}")
            by_model.setdefault(doc["model"], []).append(doc)
    return by_model, failed


def _one(values, what, model):
    vals = {json.dumps(v, sort_keys=True) for v in values}
    if len(vals) > 1:
        raise Refused(f"{model}: the runs disagree on {what}: {sorted(vals)}")
    return values[0] if values else None


def _source(model, runs):
    src = {}
    for k in _SOURCE_KEEP:
        vals = [r["source"][k] for r in runs if k in (r.get("source") or {})]
        if not vals:
            continue
        src[k] = _one(vals, "source." + k, model) if k in _SOURCE_PIN else vals[0]
    return src


def _import_call(kind, src, shapes):
    if kind == "torchvision":
        cls, _, member = (src.get("weights") or "?.?").partition(".")
        ctor = cls.replace("_Weights", "").lower()
        return (f"mojolearn.import_model(torch.export.export(torchvision.models.{ctor}(weights="
                f"torchvision.models.{cls}.{member}).eval(), (x,)))")
    if kind == "hf-causal-lm":
        return (f"mojolearn.import_model(torch.export.export(<logits of transformers.AutoModelForCausalLM."
                f"from_pretrained({src.get('repo')!r}, revision={src.get('revision')!r})>, (input_ids,)))")
    name = (src.get("url") or "model.onnx").rsplit("/", 1)[-1]
    if src.get("converted_sha256"):
        name = name.rsplit(".", 1)[0] + "-opset13.onnx"
    shape_arg = ", shapes=%r" % ({k: tuple(v) for k, v in shapes.items()},) if shapes else ""
    return f"mojolearn.import_model({name!r}{shape_arg})"


def _input_recipe(kind_of_input, shape, seed):
    if kind_of_input == "ids":
        return (f"numpy.random.default_rng({seed}).integers(0, vocab_size, {tuple(shape)!r})"
                ".astype(numpy.int64)")
    return f"numpy.random.default_rng({seed}).standard_normal({tuple(shape)!r}, dtype=numpy.float32)"


def merge_model(entry, model, runs):
    """Merge one model's runs into its catalog entry (a new dict). Raises Refused."""
    entry = copy.deepcopy(entry) if entry else {"name": model}
    kind = _one([r.get("kind") for r in runs], "kind", model)
    batch = _one([r.get("batch") for r in runs], "batch", model)
    seed = _one([r.get("seed") for r in runs], "seed", model)
    input_digest = _one([r.get("input_digest") for r in runs], "the check input (input_digest)", model)
    input_shapes = _one([r.get("input_shapes") for r in runs], "input_shapes", model)
    import_shapes = _one([r.get("import_shapes") for r in runs], "import_shapes", model)
    old_check = entry.get("check")
    if old_check and (old_check.get("batch"), old_check.get("seed"), old_check.get("input_digest")) != \
            (batch, seed, input_digest):
        raise Refused(f"{model}: the runs check batch {batch}, seed {seed}, input {input_digest}; the catalog "
                      f"holds batch {old_check.get('batch')}, seed {old_check.get('seed')}, input "
                      f"{old_check.get('input_digest')}")
    src = _source(model, runs)
    old_src = entry.get("source") or {}
    for k in _SOURCE_PIN:
        if k in src and k in old_src and src[k] != old_src[k]:
            raise Refused(f"{model}: source.{k} is {src[k]!r} in the runs, {old_src[k]!r} in the catalog "
                          "(a new source is a new entry name)")
    old_src.update(src)
    entry["kind"] = kind
    entry["source"] = old_src
    ok = [r for r in runs if r["status"] == "ok"]
    refused = [r for r in runs if str(r["status"]).startswith("refused")]
    if ok:
        ins = _one([r.get("inputs") for r in ok], "the program inputs", model)
        outs = _one([r.get("outputs") for r in ok], "the program outputs", model)
        entry["inputs"] = [{"name": n, "shape": s, "dtype": d} for n, s, d in ins]
        entry["outputs"] = [{"name": n, "shape": s, "dtype": d} for n, s, d in outs]
    elif input_shapes and "inputs" not in entry:
        entry["inputs"] = [{"name": None, "shape": s, "dtype": None} for s in input_shapes]
    in_kind = "ids" if kind == "hf-causal-lm" else "image"
    shape0 = (entry.get("inputs") or [{"shape": (input_shapes or [[]])[0]}])[0]["shape"]
    entry["check"] = {"batch": batch, "seed": seed, "input": _input_recipe(in_kind, shape0, seed),
                      "input_kind": in_kind, "input_digest": input_digest, "import_shapes": import_shapes,
                      "digest": "sha256 of the output bytes in order, first 16 hex digits"}
    entry["import"] = _import_call(kind, old_src, entry["check"]["import_shapes"])
    versions = entry.setdefault("versions", {})
    for r in runs:
        cell = versions.setdefault(r["mojolearn_version"], {}).setdefault(r["numeric_mode"], {})
        vendor = r["vendor"]
        if str(r["status"]).startswith("refused"):
            cell["status"] = "refused"
            cell["unsupported_ops"] = sorted(set(cell.get("unsupported_ops", [])) | set(r.get("unsupported_ops") or []))
            continue
        ours = r.get("ours") or {}
        d = ours.get("digest")
        if r["numeric_mode"] == "identical":
            if not ours.get("repeat_equal", False):
                raise Refused(f"{model} {vendor} {r['mojolearn_version']}: the second run's output differs "
                              f"from the first ({r['_file']})")
            digests = cell.setdefault("digests", {})
            have = set(digests.values()) | ({d} if d else set())
            if len(have) > 1:
                raise Refused(f"{model} {r['mojolearn_version']} identical: the output digests disagree: "
                              + ", ".join(f"{v}={x}" for v, x in sorted(digests.items()))
                              + f", {vendor}={d} ({r['_file']})")
            digests[vendor] = d
        cell["status"] = "ok"
        perf = {k: (r.get("gpu_arch") if k == "gpu_arch" else ours.get(k)) for k in _PERF}
        perf["import_ms"] = r.get("import_ms")
        env = {k: (r.get("source") or {}).get(k) for k in _ENV if (r.get("source") or {}).get(k)}
        if env:
            perf["environment"] = env
        cell.setdefault("vendors", {})[vendor] = perf
        acc = r.get("accuracy")
        if acc and "value" in acc:
            meas = cell.setdefault("accuracy", {})
            prev = meas.get("value")
            if prev is not None and prev != acc["value"] and r["numeric_mode"] == "identical":
                raise Refused(f"{model} {r['mojolearn_version']} identical: top-1 {acc['value']} on {vendor}, "
                              f"{prev} on {meas.get('vendors')} with equal digests: the eval inputs differ")
            meas.update({k: acc[k] for k in ("metric", "value", "n", "slice", "slice_files_sha256", "preprocess")})
            meas["vendors"] = sorted(set(meas.get("vendors", [])) | {vendor})
    if refused and ok:
        raise Refused(f"{model}: refused on {[r['vendor'] for r in refused]}, ok on {[r['vendor'] for r in ok]}")
    entry["status"] = entry_status(entry)
    acc = {}
    if old_src.get("published_top1") is not None:
        acc["published"] = {"metric": "top1", "value": old_src["published_top1"],
                            "eval": old_src.get("published_eval")}
    newest = newest_cell(entry)
    if newest and newest.get("accuracy"):
        acc["measured"] = newest["accuracy"]
    entry["accuracy"] = acc or None
    return entry


def newest_cell(entry, mode="identical"):
    vs = sorted((v for v in entry.get("versions", {}) if mode in entry["versions"][v]), key=_version_key)
    return entry["versions"][vs[-1]][mode] if vs else None


def entry_status(entry):
    cell = newest_cell(entry)
    if cell is None:
        return "unverified: no identical-mode run"
    if cell.get("status") == "refused":
        return "refused: " + ", ".join(cell.get("unsupported_ops") or ["unsupported model"])
    digests = cell.get("digests") or {}
    missing = [VENDOR_NAMES[v] for v in GPU_VENDORS if v not in digests]
    if missing:
        return "unverified: no %s run" % " or ".join(missing)
    return "verified"


def build(catalog_path, run_dirs):
    path = Path(catalog_path)
    cat = json.loads(path.read_text()) if path.is_file() else {"schema": SCHEMA, "models": []}
    if cat.get("schema") != SCHEMA:
        raise Refused(f"{path}: schema {cat.get('schema')!r}, expected {SCHEMA}")
    by_model, failed = load_runs(run_dirs)
    entries = {e["name"]: e for e in cat.get("models", [])}
    for model in sorted(by_model):
        entries[model] = merge_model(entries.get(model), model, by_model[model])
    cat["models"] = [entries[k] for k in sorted(entries)]
    cat["schema"] = SCHEMA
    return cat, sorted(by_model), failed


def _write(cat, path):
    path = Path(path)
    tmp = str(path) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(cat, fh, indent=1, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def show(cat):
    for e in cat.get("models", []):
        cell = newest_cell(e) or {}
        dg = cell.get("digests") or {}
        print("CATALOG %s %s %s" % (e["name"], e["status"].replace(" ", ""),
                                    ",".join(f"{v}={d}" for v, d in sorted(dg.items())) or "-"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="merge real_model_import.py run directories into the catalog")
    b.add_argument("--runs", nargs="+", required=True, help="run directories (or single run JSON files)")
    b.add_argument("--catalog", default=str(CATALOG))
    b.add_argument("--check", action="store_true", help="build and print, write nothing")
    s = sub.add_parser("show", help="one CATALOG line per entry")
    s.add_argument("--catalog", default=str(CATALOG))
    a = ap.parse_args(argv)
    if a.cmd == "show":
        show(json.loads(Path(a.catalog).read_text()))
        return 0
    try:
        cat, models, failed = build(a.catalog, a.runs)
    except Refused as exc:
        print("CATALOG_REFUSED %s" % exc)
        print("model_catalog: nothing written")
        return 1
    for f in failed:
        print("CATALOG_SKIPPED_FAILED_RUN %s" % f)
    show({"models": [e for e in cat["models"] if e["name"] in models]})
    if not a.check:
        _write(cat, a.catalog)
        print("model_catalog: %d models updated in %s" % (len(models), a.catalog))
    return 0


if __name__ == "__main__":
    sys.exit(main())
