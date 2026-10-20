#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""REAL PRETRAINED MODELS THROUGH THE IMPORTER (2026-10-20, lane ag-real-models), a box tool.

Each model in MODELS is a real, unmodified pretrained network: torchvision's resnet18, squeezenet1_1 and
mobilenet_v3_small with their published weights (torch.export of the eval module, a fixed example
input), two ONNX model-zoo files (downloaded once, the sha256 recorded) and HuggingFaceTB/SmolLM2-135M
(torch.export of the causal LM returning logits). For each one the tool

  1. exports it (torch.export.export with a fixed example input) or downloads it (ONNX);
  2. imports it: mojolearn.import_model(source, shapes=...);
  3. runs our Program on a seeded input: one untimed run (the digest), then one timed run, whole clock
     (the host -> device copy of the input, the program, the device -> host copy of the output);
  4. runs the source framework on the same input, same clock (torch: the input moved to the device inside
     the clock, the output read back inside it; onnxruntime: one session.run), TF32 off;
  5. compares the two: max_abs, max_rel and top-1 agreement along the last axis. REFERENCE ONLY, never a
     gate: torch and onnxruntime sum in another order and use their own exp/erf/tanh;
  6. records our output digest (sha256 of the output words, 16 hex): the identity claim is that this
     digest is equal on NVIDIA, AMD and Apple for the same model, batch and seed.

A model the import cannot run is recorded as `refused`, with every unsupported node by name and op
(mojolearn.importers.UnsupportedModel.problems), and one `REAL_MODEL_UNSUPPORTED` line per node so the
op table can grow. Nothing is approximated: a depthwise Conv (group > 1) is a refusal, not a rewrite.
SmolLM2's token embedding (a Gather by a model input) is `refused: embedding` until the program can hold
an embedding (lane ag-program-embedding); the row needs no change when it lands.

Every model runs in its own child process (one `one` call), so its device memory reading is its own and a
crash stops one row only. Output: one JSON per model under --out (schema mojolearn-real-model/1) and one
line per model:

    REAL_MODEL <name> <status> <digest|-> <top1_agree|-> <max_abs|-> <latency_ms|->

    python3 tools/real_model_import.py run --out DIR [--models resnet18,squeezenet1_1] [--batch 1]
                                         [--cache DIR] [--seed 0] [--timeout 1800]
                                         [--imagenet-val DIR] [--installed]
    python3 tools/real_model_import.py list

By default the runner tests this source checkout. --installed tests the current
interpreter's non-editable installation in isolation, including in each model
child. Each model JSON records the imported package and loaded native bindings.
Use a virtual environment with the base wheel and the required GPU plugin
installed from PyPI or local wheel files. This tool never installs packages.

Accuracy (optional, --imagenet-val DIR): the ImageNet classifiers (the three torchvision models and
resnet50-v1-12) are also scored on a stated public slice, the first 256 images of the ILSVRC2012
validation set by file name (ILSVRC2012_val_00000001 .. 00000256), read from DIR in the ImageFolder
layout (DIR/<wnid>/<file>.JPEG; the class index is the wnid's position in sorted order, the order the
weights use). Only the metric value is recorded (accuracy.value, with the slice's file-list sha256): the
images stay on the box and never enter the repository or the wheel. torchvision's own published top-1
(the weights' meta) is recorded beside it as source.published_top1. tools/model_catalog.py builds
python/mojolearn/models/catalog.json from these JSON files.

Box needs: torch, torchvision, transformers (SmolLM2), onnxruntime (the ONNX reference), onnx (to
convert a model-zoo file below opset 13: onnx.version_converter.convert_version(model, 13), saved beside
the download as <stem>-opset13.onnx with its sha256, and that file is imported and run by onnxruntime;
without onnx such a row fails with the reason instead of importing the old opset). Network for the first download of each weight file. Exit
status 0 when every row has a status (ok or refused); 1 when a row failed (its log tail in the JSON).
"""
import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PACKAGE_EVIDENCE = ROOT / "python/mojolearn/tests/_package_evidence.py"
SCHEMA = "mojolearn-real-model/1"
_MB = 1024.0 * 1024.0
ONNX_ZOO = "https://github.com/onnx/models/raw/main/validated/vision/classification"

#: name -> how to get it. `input` is the example input for a batch b: (shape(b), "image" | "ids").
#: `sha256`: a pinned digest of the downloaded file (None: recorded on the first download, then checked
#: against the cache's .sha256 file on every later run).
MODELS = {
    "resnet18": {"kind": "torchvision", "ctor": "resnet18", "weights": "ResNet18_Weights", "eval": "imagenet",
                 "input": (lambda b: (b, 3, 224, 224), "image")},
    "squeezenet1_1": {"kind": "torchvision", "ctor": "squeezenet1_1", "weights": "SqueezeNet1_1_Weights", "eval": "imagenet",
                      "input": (lambda b: (b, 3, 224, 224), "image")},
    "mobilenet_v3_small": {"kind": "torchvision", "ctor": "mobilenet_v3_small",
                           "weights": "MobileNet_V3_Small_Weights", "eval": "imagenet",
                           "input": (lambda b: (b, 3, 224, 224), "image")},
    "resnet50-v1-12": {"kind": "onnx", "url": ONNX_ZOO + "/resnet/model/resnet50-v1-12.onnx", "sha256": None,
                       "eval": "imagenet", "input": (lambda b: (b, 3, 224, 224), "image")},
    "mnist-12": {"kind": "onnx", "url": ONNX_ZOO + "/mnist/model/mnist-12.onnx", "sha256": None,
                 "input": (lambda b: (1, 1, 28, 28), "image")},  # the file fixes its batch at 1
    "smollm2-135m": {"kind": "hf-causal-lm", "repo": "HuggingFaceTB/SmolLM2-135M",
                     "input": (lambda b: (b, 32), "ids")},
}
DEFAULT_MODELS = tuple(MODELS)


def _sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _digest(*arrays):
    """Our output words: sha256 of the contiguous bytes, 16 hex (the IMPORT_DIGEST convention)."""
    import numpy as np
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]


# ------------------------------------------------------------------ sources ----

def _torch_flags(torch):
    # an fp32 reference: no TF32 in convolutions or matmuls
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except AttributeError:
        pass
    try:
        torch.set_float32_matmul_precision("highest")
    except AttributeError:
        pass


def _example(spec, batch, seed, vocab=None):
    """The seeded input of a model: standard normals for an image, ids in [0, vocab) for a language model."""
    import numpy as np
    shape_of, kind = spec["input"]
    shape = tuple(shape_of(batch))
    rng = np.random.default_rng(seed)
    if kind == "ids":
        return rng.integers(0, vocab, shape).astype(np.int64)
    return rng.standard_normal(shape, dtype=np.float32)


def source_torchvision(spec, batch, seed, info):
    import torch
    import torchvision
    _torch_flags(torch)
    weights = getattr(torchvision.models, spec["weights"]).DEFAULT
    info["weights"] = str(weights)
    info["weights_url"] = getattr(weights, "url", None)
    try:  # torchvision's published number for these weights (its own full ImageNet-1K val run)
        info["published_top1"] = float(weights.meta["_metrics"]["ImageNet-1K"]["acc@1"]) / 100.0
        info["published_eval"] = "ImageNet-1K validation (50,000 images), as published with the torchvision weights"
    except (AttributeError, KeyError, TypeError, ValueError):
        pass
    model = getattr(torchvision.models, spec["ctor"])(weights=weights).eval()
    x = _example(spec, batch, seed)
    t0 = time.perf_counter()
    with torch.no_grad():
        ep = torch.export.export(model, (torch.from_numpy(x),))
    info["export_ms"] = (time.perf_counter() - t0) * 1e3
    info["torch_version"] = torch.__version__
    info["torchvision_version"] = torchvision.__version__
    return ep, None, [x], ("torch", model)


class _LogitsOnly:
    """A causal LM as input_ids -> logits (no cache, no attention mask: every position attends causally)."""

    @staticmethod
    def wrap(torch, lm):
        class Logits(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids):
                return self.m(input_ids=input_ids, use_cache=False, return_dict=True).logits

        return Logits(lm).eval()


def source_hf_causal_lm(spec, batch, seed, info):
    import torch
    import transformers
    _torch_flags(torch)
    lm = transformers.AutoModelForCausalLM.from_pretrained(spec["repo"], torch_dtype=torch.float32)
    lm.config.use_cache = False
    model = _LogitsOnly.wrap(torch, lm.eval())
    info["repo"] = spec["repo"]
    info["revision"] = getattr(lm.config, "_commit_hash", None)
    info["transformers_version"] = transformers.__version__
    info["torch_version"] = torch.__version__
    vocab = int(lm.config.vocab_size)
    x = _example(spec, batch, seed, vocab=vocab)
    t0 = time.perf_counter()
    with torch.no_grad():
        try:
            ep = torch.export.export(model, (torch.from_numpy(x),))
        except Exception as exc:  # noqa: BLE001  transformers models often need the non-strict tracer
            info["export_strict_error"] = repr(exc)[:400]
            ep = torch.export.export(model, (torch.from_numpy(x),), strict=False)
    info["export_ms"] = (time.perf_counter() - t0) * 1e3
    return ep, None, [x], ("torch", model)


def _download(url, dest):
    tmp = str(dest) + ".part"
    with urllib.request.urlopen(url, timeout=600) as r, open(tmp, "wb") as fh:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            fh.write(chunk)
    os.replace(tmp, dest)


def _onnx_opset(path):
    """The default-domain opset of an ONNX file: read by the onnx package when it is installed (the graph
    only, no external data), else by mojolearn's own test-oracle wire reader."""
    try:
        import onnx
    except ImportError:
        onnx = None
    if onnx is not None:
        m = onnx.load(str(path), load_external_data=False)
        ops = [(o.domain, o.version) for o in m.opset_import]
    else:
        from mojolearn.tests import _onnx_proto_oracle as proto  # the test oracle's wire reader (no binding needed)
        ops = [(o["domain"], o["version"]) for o in proto.model(Path(path).read_bytes())["opset_import"]]
    for domain, version in ops:
        if domain in ("", "ai.onnx"):
            return int(version)
    return None


#: the importer reads opset 13 and later; a model-zoo file below it is converted once with onnx's own converter
MIN_OPSET = 13


def _convert_opset(path, opset, info):
    """`path` (default-domain opset `opset` < 13) converted with onnx.version_converter.convert_version(model,
    13), saved beside the original as <stem>-opset13.onnx with its sha256 in <that>.sha256 (written on the
    conversion, checked on every later run), recorded in `info`. Returns the converted file's path. Raises
    when the onnx package is missing or the converted file is still below opset 13: never imports the
    unconverted file (the importer would refuse it by opset)."""
    try:
        import onnx
        from onnx import version_converter
    except ImportError as e:
        raise RuntimeError(f"{path} is opset {opset}: the onnx package is needed to convert it to opset "
                           f"{MIN_OPSET} (pip install onnx): {e}") from e
    conv = path.with_name(path.stem + f"-opset{MIN_OPSET}.onnx")
    pin = conv.with_name(conv.name + ".sha256")
    if conv.is_file() and pin.is_file():
        sha = _sha256_file(conv)
        if sha != pin.read_text().strip():
            raise RuntimeError(f"{conv} has sha256 {sha}, recorded {pin.read_text().strip()}: delete both to "
                               f"convert again")
    else:
        model = version_converter.convert_version(onnx.load(str(path)), MIN_OPSET)
        tmp = conv.with_name(conv.name + ".part")
        onnx.save(model, str(tmp))
        os.replace(tmp, conv)
        sha = _sha256_file(conv)
        pin.write_text(sha + "\n")
    got = _onnx_opset(conv)
    if got is None or got < MIN_OPSET:
        raise RuntimeError(f"{conv}: onnx.version_converter left opset {got} (opset {MIN_OPSET} is needed)")
    info.update(opset_conversion=f"onnx.version_converter.convert_version(model, {MIN_OPSET})",
                converted_file=str(conv), converted_sha256=sha, converted_opset=got,
                onnx_version=onnx.__version__)
    return conv


def _onnx_shapes(path, x):
    """`shapes` for import_model: the example's shape for the (single) image input when the file leaves an
    axis open, else None. A file the reader refuses gives None; import_model then names its problems."""
    from mojolearn.importers import UnsupportedModel
    from mojolearn.importers import onnx as reader
    try:
        g = reader.read(path)
    except (UnsupportedModel, ValueError):
        return None
    if len(g.inputs) != 1:
        return None
    name, _, dims = g.inputs[0]
    return None if all(isinstance(d, int) for d in dims) else {name: tuple(x.shape)}


def source_onnx(spec, batch, seed, info, cache):
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    name = spec["url"].rsplit("/", 1)[1]
    path = cache / name
    pin = cache / (name + ".sha256")
    if not path.is_file():
        _download(spec["url"], path)
        info["downloaded"] = True
    sha = _sha256_file(path)
    want = spec.get("sha256") or (pin.read_text().strip() if pin.is_file() else None)
    if want and want != sha:
        raise RuntimeError(f"{path} has sha256 {sha}, recorded {want}: delete it to download again")
    if not pin.is_file():
        pin.write_text(sha + "\n")
    info.update(url=spec["url"], file=str(path), sha256=sha)
    src = path
    try:
        opset = _onnx_opset(path)
    except Exception as e:  # noqa: BLE001  recorded; the importer's own error then names the problem
        opset = None
        info["opset_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    info["opset"] = opset
    if opset is not None and opset < MIN_OPSET:
        src = _convert_opset(path, opset, info)
    x = _example(spec, batch, seed)
    return str(src), _onnx_shapes(str(src), x), [x], ("onnxruntime", str(src))


# ----------------------------------------------------------- device memory ----

def device_memory_mb(vendor):
    """(MB, method) of this process's device memory now, or (None, why). Read right after our arm, before
    the reference touches the device; a reading at the arm's end, not a peak."""
    pid = os.getpid()
    if vendor == "cuda":
        try:
            out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                                  "--format=csv,noheader,nounits"], capture_output=True, text=True,
                                 timeout=30, check=False).stdout
        except (OSError, subprocess.SubprocessError) as exc:
            return None, "nvidia-smi unavailable (%s)" % exc.__class__.__name__
        for line in out.splitlines():
            f = [x.strip() for x in line.split(",")]
            if len(f) >= 2 and f[0] == str(pid):
                try:
                    return float(f[1]), "nvidia-smi used_memory for this pid after our arm (context and pools)"
                except ValueError:
                    break
        return None, "nvidia-smi lists no compute app with this pid (a container's pid namespace hides it)"
    if vendor == "hip":
        kfd = "/sys/class/kfd/kfd/proc/%d" % pid
        try:
            names = [n for n in os.listdir(kfd) if n.startswith("vram_")]
        except OSError:
            names = []
        if names:
            used = 0
            for n in names:
                with open(os.path.join(kfd, n)) as fh:
                    used += int(fh.read().strip() or 0)
            return used / _MB, "KFD %s/vram_* bytes for this pid after our arm" % kfd
        return None, "no KFD counters for this pid"
    if vendor == "metal":
        return None, "Apple unified memory: no per-process GPU counter"
    return None, "no device (%s)" % vendor


# ---------------------------------------------------------------- the arms ----

def _compare(ours, ref):
    import numpy as np
    ours = np.asarray(ours, dtype=np.float32)
    ref = np.asarray(ref, dtype=np.float32)
    if ours.shape != ref.shape:
        return {"error": "shape differs", "shape": list(ours.shape), "reference_shape": list(ref.shape)}
    d = np.abs(ours.astype(np.float64) - ref.astype(np.float64))
    rel = d / np.maximum(np.abs(ref.astype(np.float64)), np.finfo(np.float32).tiny)
    row = {"shape": list(ours.shape), "max_abs": float(d.max(initial=0.0)),
           "max_rel": float(rel.max(initial=0.0)), "finite": bool(np.isfinite(ours).all())}
    if ours.ndim >= 2 and ours.shape[-1] > 1:
        a = ours.reshape(-1, ours.shape[-1]).argmax(axis=1)
        b = ref.reshape(-1, ref.shape[-1]).argmax(axis=1)
        row["top1_agree"] = float((a == b).mean())
        row["top1_rows"] = int(a.size)
    return row


def ours_arm(prog, arrays, info):
    """One untimed run (the digest), one timed run; the whole clock includes both copies."""
    import numpy as np
    t0 = time.perf_counter()
    y0 = prog.run(*arrays)
    info["first_run_ms"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    y = prog.run(*arrays)
    info["latency_ms"] = (time.perf_counter() - t0) * 1e3
    y0 = y0 if isinstance(y0, tuple) else (y0,)
    y = y if isinstance(y, tuple) else (y,)
    info["digest"] = _digest(*y0)
    info["repeat_equal"] = _digest(*y) == info["digest"]
    info["output_shapes"] = [list(np.shape(a)) for a in y]
    return y


def torch_arm(model, arrays, info):
    import torch
    if torch.cuda.is_available():
        dev = "cuda"
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        dev = "mps"
    else:
        dev = "cpu"
    info["device"] = dev
    model = model.to(dev)

    def once():
        with torch.no_grad():
            xs = [torch.from_numpy(a).to(dev) for a in arrays]
            out = model(*xs)
            out = out if isinstance(out, (tuple, list)) else (out,)
            return [o.detach().float().cpu().numpy() for o in out]

    if dev == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    once()
    info["first_run_ms"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    refs = once()
    info["latency_ms"] = (time.perf_counter() - t0) * 1e3
    if dev == "cuda":
        info["peak_device_mb"] = torch.cuda.max_memory_allocated() / _MB
        info["peak_device_method"] = "torch.cuda.max_memory_allocated (the caching allocator's peak)"
    elif dev == "mps":
        info["peak_device_mb"] = None
        info["peak_device_method"] = "Apple unified memory: no per-process GPU counter"
    return refs


def onnxruntime_arm(path, input_names, arrays, info):
    import onnxruntime as ort
    avail = ort.get_available_providers()
    gpu = [p for p in ("CUDAExecutionProvider", "ROCMExecutionProvider", "CoreMLExecutionProvider") if p in avail]
    sess = ort.InferenceSession(path, providers=gpu[:1] + ["CPUExecutionProvider"])
    info["providers"] = sess.get_providers()
    info["onnxruntime_version"] = ort.__version__
    feeds = dict(zip(input_names, arrays))
    names = [o.name for o in sess.get_outputs()]
    t0 = time.perf_counter()
    sess.run(names, feeds)
    info["first_run_ms"] = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    refs = sess.run(names, feeds)
    info["latency_ms"] = (time.perf_counter() - t0) * 1e3
    return refs


# ---------------------------------------------------------------- accuracy ----

IMAGENET_SLICE = 256
IMAGENET_SLICE_NAME = ("ImageNet-1K (ILSVRC2012) validation, the first 256 images by file name "
                       "(ILSVRC2012_val_00000001 .. 00000256)")


def imagenet_slice(root, n=IMAGENET_SLICE):
    """[(path, class index)] of the slice: every image of an ImageFolder-layout validation directory,
    sorted by file name, the first n; the class index is the wnid directory's position in sorted order."""
    root = Path(root)
    wnids = sorted(d.name for d in root.iterdir() if d.is_dir())
    if len(wnids) != 1000:
        raise RuntimeError(f"{root}: {len(wnids)} class directories, an ImageNet-1K validation set has 1000")
    cls = {w: i for i, w in enumerate(wnids)}
    files = sorted((p for w in wnids for p in (root / w).iterdir() if p.is_file()), key=lambda p: p.name)
    if len(files) < n:
        raise RuntimeError(f"{root}: {len(files)} images, the slice needs {n}")
    return [(p, cls[p.parent.name]) for p in files[:n]]


def _imagenet_transform(spec):
    """The weights' own preprocessing for a torchvision model; the standard ImageNet one (resize 256,
    center crop 224, the ImageNet mean and std) for an ONNX model-zoo classifier."""
    import torchvision
    if spec["kind"] == "torchvision":
        return getattr(torchvision.models, spec["weights"]).DEFAULT.transforms(), "the torchvision weights' transforms()"
    from torchvision import transforms as T
    return (T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                       T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
            "resize 256, center crop 224, ImageNet mean/std normalization")


def imagenet_accuracy(spec, prog, root):
    """Top-1 of our program on the slice, at the program's batch (a partial last batch is padded with
    copies of its first image and the padding is not scored). Returns the accuracy record."""
    import numpy as np
    from PIL import Image
    rows = imagenet_slice(root)
    tf, how = _imagenet_transform(spec)
    batch = int(prog.inputs[0][1][0])
    correct = 0
    t0 = time.perf_counter()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        xs = [tf(Image.open(p).convert("RGB")).numpy().astype(np.float32) for p, _ in chunk]
        xs += [xs[0]] * (batch - len(xs))
        y = prog.run(np.stack(xs))
        y = y[0] if isinstance(y, tuple) else y
        pred = np.asarray(y).reshape(batch, -1).argmax(axis=1)[:len(chunk)]
        correct += int(sum(int(p) == c for p, (_, c) in zip(pred, chunk)))
    names = "\n".join(p.name for p, _ in rows)
    return {"metric": "top1", "value": correct / len(rows), "n": len(rows), "slice": IMAGENET_SLICE_NAME,
            "slice_files_sha256": hashlib.sha256(names.encode()).hexdigest(), "preprocess": how,
            "eval_ms": (time.perf_counter() - t0) * 1e3}


# ------------------------------------------------------------ one model ----

def _refusal_kind(problems):
    for name, op, why in problems:
        if op in ("Gather", "aten.embedding") and ("input" in why or "index" in why):
            return "refused: embedding"
    return "refused"


def run_one(name, batch, seed, cache, out_dir, imagenet_val=None, *, installed=False):
    spec = MODELS[name]
    doc = {"schema": SCHEMA, "model": name, "kind": spec["kind"], "batch": batch, "seed": seed,
           "gate": False, "status": "failed",
           "clock": {"ours": "whole: input host->device copy + Program.run + output device->host copy",
                     "reference": "whole: torch.from_numpy(x).to(device) + forward + .cpu(); onnxruntime "
                                  "session.run from host arrays",
                     "runs_per_arm": "one untimed (digest / warm), then one timed"}}
    out = Path(out_dir) / ("%s.json" % name)
    try:
        import mojolearn
        package_evidence = runpy.run_path(str(_PACKAGE_EVIDENCE))["package_evidence"]
        doc["package"] = package_evidence(mojolearn, require_installed=installed)
        doc["vendor"] = mojolearn.vendor()
        doc["numeric_mode"] = mojolearn.numeric_mode()
        doc["mojolearn_version"] = getattr(mojolearn, "__version__", None)
        try:
            doc["gpu_arch"] = mojolearn.gpu_arch()
        except Exception:  # noqa: BLE001  a CPU install has none
            doc["gpu_arch"] = None
        src_info = {}
        doc["source"] = src_info
        if spec["kind"] == "torchvision":
            source, shapes, arrays, ref = source_torchvision(spec, batch, seed, src_info)
        elif spec["kind"] == "hf-causal-lm":
            source, shapes, arrays, ref = source_hf_causal_lm(spec, batch, seed, src_info)
        else:
            source, shapes, arrays, ref = source_onnx(spec, batch, seed, src_info, cache)
        doc["input_digest"] = _digest(*arrays)
        doc["import_shapes"] = {k: list(v) for k, v in shapes.items()} if shapes else None
        doc["input_shapes"] = [list(a.shape) for a in arrays]
        from mojolearn.importers import UnsupportedModel
        t0 = time.perf_counter()
        try:
            prog = mojolearn.import_model(source, shapes=shapes)
        except UnsupportedModel as e:
            doc["package"] = package_evidence(mojolearn, require_installed=installed)
            doc["status"] = _refusal_kind(e.problems)
            doc["unsupported"] = [{"node": n, "op": op, "reason": why} for n, op, why in e.problems]
            doc["unsupported_ops"] = sorted({p[1] for p in e.problems if p[1] is not None})
            return doc, out
        doc["import_ms"] = (time.perf_counter() - t0) * 1e3
        doc["inputs"] = [[n, list(s), d] for n, s, d in prog.inputs]
        doc["outputs"] = [[n, list(s), d] for n, s, d in prog.outputs]
        want = [tuple(s) for _, s, _ in prog.inputs]
        if want != [tuple(a.shape) for a in arrays]:
            raise RuntimeError(f"the program takes {want}, the example input is {[a.shape for a in arrays]}")
        ours = {}
        doc["ours"] = ours
        y = ours_arm(prog, arrays, ours)
        ours["device_memory_mb"], ours["device_memory_method"] = device_memory_mb(doc["vendor"])
        reference = {"library": ref[0]}
        doc["reference"] = reference
        try:
            if ref[0] == "torch":
                refs = torch_arm(ref[1], arrays, reference)
            else:
                refs = onnxruntime_arm(ref[1], [n for n, _, _ in prog.inputs], arrays, reference)
        except ImportError as exc:
            reference["error"] = "not installed: %s" % exc
            refs = None
        if refs is not None:
            doc["compare"] = [_compare(o, r) for o, r in zip(y, refs)]
        if imagenet_val and spec.get("eval") == "imagenet":
            try:
                doc["accuracy"] = imagenet_accuracy(spec, prog, imagenet_val)
            except Exception as exc:  # noqa: BLE001  the digest row stands without its accuracy
                doc["accuracy"] = {"error": repr(exc)[:400]}
        doc["package"] = package_evidence(mojolearn, require_installed=installed)
        doc["status"] = "ok"
    except Exception as exc:  # noqa: BLE001  a failed row records why; the other models still run
        import traceback
        doc["status"] = "failed"
        doc["error"] = repr(exc)[:2000]
        doc["traceback"] = traceback.format_exc().splitlines()[-12:]
    return doc, out


def _fmt(v, f):
    return "-" if v is None else (f % v)


def line(doc):
    cmp0 = (doc.get("compare") or [{}])[0]
    status = doc["status"].replace(" ", "")
    return "REAL_MODEL %s %s %s %s %s %s" % (
        doc["model"], status, (doc.get("ours") or {}).get("digest", "-"),
        _fmt(cmp0.get("top1_agree"), "%.4f"), _fmt(cmp0.get("max_abs"), "%.3e"),
        _fmt((doc.get("ours") or {}).get("latency_ms"), "%.3f"))


def unsupported_lines(doc):
    return ["REAL_MODEL_UNSUPPORTED %s %s %s: %s" % (doc["model"], u["op"] or "-", u["node"], u["reason"])
            for u in doc.get("unsupported", ())]


def _write(doc, out):
    tmp = str(out) + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(doc, fh, indent=1, sort_keys=True)
    os.replace(tmp, out)


def cmd_one(a):
    Path(a.out).mkdir(parents=True, exist_ok=True)
    doc, out = run_one(a.name, a.batch, a.seed, a.cache, a.out, a.imagenet_val, installed=a.installed)
    _write(doc, out)
    print(line(doc), flush=True)
    for text in unsupported_lines(doc):
        print(text, flush=True)
    return 0 if doc["status"] != "failed" else 1


def _tail(path, n=12, width=300):
    try:
        lines = [l for l in Path(path).read_text(errors="replace").splitlines() if l.strip()]
    except OSError:
        return []
    return [l[:width] for l in lines[-n:]]


def cmd_run(a):
    out_dir = Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    names = [n for n in a.models.split(",") if n] if a.models else list(DEFAULT_MODELS)
    unknown = [n for n in names if n not in MODELS]
    if unknown:
        print("unknown model(s): %s (tools/real_model_import.py list)" % ", ".join(unknown))
        return 2
    bad = 0
    summary = []
    for name in names:  # one child process per model: its own device memory, its own crash
        log = out_dir / ("%s.log" % name)
        cmd = [sys.executable, *(['-I'] if a.installed else []),
               str(Path(__file__).resolve()), "one", name, "--out", str(out_dir),
               "--batch", str(a.batch), "--seed", str(a.seed), "--cache", str(a.cache)]
        if a.installed:
            cmd += ["--installed"]
        if a.imagenet_val:
            cmd += ["--imagenet-val", str(a.imagenet_val)]
        with open(log, "wb") as fh:
            try:
                rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, timeout=a.timeout).returncode
            except subprocess.TimeoutExpired:
                rc = None
        path = out_dir / ("%s.json" % name)
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            doc = {"schema": SCHEMA, "model": name, "status": "failed", "batch": a.batch, "seed": a.seed,
                   "error": "timed out after %ss" % a.timeout if rc is None else "exit %s, no JSON" % rc,
                   "tail": _tail(log)}
            _write(doc, path)
        # A previous JSON cannot turn a crashed or timed-out new child green.
        if rc != 0:
            doc["status"] = "failed"
            doc.setdefault("error", "timed out after %ss" % a.timeout if rc is None else "child exit %s" % rc)
        if doc["status"] == "failed":
            bad += 1
            doc.setdefault("tail", _tail(log))
            _write(doc, path)
        print(line(doc), flush=True)
        for text in unsupported_lines(doc):
            print(text, flush=True)
        summary.append({"model": name, "status": doc["status"], "json": str(path), "log": str(log)})
    _write({"schema": SCHEMA + "-summary", "rows": summary, "batch": a.batch, "seed": a.seed,
            "package_selection": "installed" if a.installed else "checkout"},
           out_dir / "summary.json")
    return 1 if bad else 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    default_cache = os.path.expanduser("~/.cache/mojolearn-real-models")
    for p in (sub.add_parser("run", help="every model (or --models), one child process each"),
              sub.add_parser("one", help="one model in this process (what run starts)")):
        if p.prog.endswith("one"):
            p.add_argument("name", choices=sorted(MODELS))
        else:
            p.add_argument("--models", default=None, help="comma-separated names (default: all)")
            p.add_argument("--timeout", type=float, default=1800.0, help="seconds per model")
        p.add_argument("--out", required=True)
        p.add_argument("--batch", type=int, default=1)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--cache", default=default_cache, help="where the ONNX files are downloaded once")
        p.add_argument("--installed", action="store_true",
                       help="test this interpreter's installed wheels in isolation; reject checkout or editable imports")
        p.add_argument("--imagenet-val", default=None,
                       help="an ImageNet-1K validation directory (ImageFolder layout) on this box: score the "
                            "ImageNet classifiers on its first 256 images by file name")
    sub.add_parser("list", help="the models, one line each")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for n, s in MODELS.items():
            print("%-20s %-13s %s" % (n, s["kind"], s.get("url") or s.get("repo") or "torchvision." + s["ctor"]))
        return 0
    if a.installed:
        if not sys.flags.isolated:
            # -I ignores PYTHONPATH, the working directory and user-site
            # packages. Re-enter before importing mojolearn or loading models.
            os.execv(sys.executable, [sys.executable, "-I", str(Path(__file__).resolve()), *argv])
    else:
        repo_python = str(ROOT / "python")
        sys.path.insert(0, repo_python)
        os.environ["PYTHONPATH"] = repo_python + (
            os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    return cmd_one(a) if a.cmd == "one" else cmd_run(a)


if __name__ == "__main__":
    sys.exit(main())
