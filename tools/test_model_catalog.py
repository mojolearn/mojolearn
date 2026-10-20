# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""tools/model_catalog.py: verified only when NVIDIA and AMD agree; any digest disagreement refuses."""
import json

import pytest

import model_catalog as mc


def _run(vendor, digest="aaaa", status="ok", version="0.9.0", mode="identical", repeat=True, **extra):
    doc = {"schema": mc.RUN_SCHEMA, "model": "resnet18", "kind": "torchvision", "batch": 1, "seed": 0,
           "status": status, "vendor": vendor, "numeric_mode": mode, "mojolearn_version": version,
           "gpu_arch": vendor + "-arch", "input_digest": "in0", "input_shapes": [[1, 3, 224, 224]],
           "import_shapes": None, "import_ms": 5.0,
           "source": {"weights": "ResNet18_Weights.IMAGENET1K_V1", "weights_url": "https://w/r.pth",
                      "torch_version": "2.5.0", "published_top1": 0.6976, "published_eval": "ImageNet-1K val"}}
    if status == "ok":
        doc.update(inputs=[["x", [1, 3, 224, 224], "float32"]], outputs=[["y", [1, 1000], "float32"]],
                   ours={"digest": digest, "repeat_equal": repeat, "latency_ms": 3.0, "device_memory_mb": 100.0})
    else:
        doc["unsupported_ops"] = ["Gather"]
    doc.update(extra)
    return doc


def _dirs(tmp_path, *docs):
    out = []
    for i, d in enumerate(docs):
        p = tmp_path / ("run%d" % i)
        p.mkdir(parents=True)
        (p / ("%s.json" % d["model"])).write_text(json.dumps(d))
        (p / "summary.json").write_text(json.dumps({"schema": mc.RUN_SCHEMA + "-summary"}))
        out.append(str(p))
    return out


def test_agreeing_vendors_verify(tmp_path):
    cat, models, failed = mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda"), _run("hip"), _run("metal")))
    (e,) = cat["models"]
    assert models == ["resnet18"] and not failed
    assert e["status"] == "verified"
    assert e["versions"]["0.9.0"]["identical"]["digests"] == {"cuda": "aaaa", "hip": "aaaa", "metal": "aaaa"}
    assert e["accuracy"]["published"]["value"] == 0.6976
    assert "ResNet18_Weights.IMAGENET1K_V1" in e["import"]


def test_disagreeing_digest_refuses(tmp_path):
    with pytest.raises(mc.Refused, match="disagree"):
        mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda"), _run("hip", digest="bbbb")))


def test_apple_disagreeing_refuses_too(tmp_path):
    with pytest.raises(mc.Refused, match="disagree"):
        mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda"), _run("hip"), _run("metal", digest="cccc")))


def test_not_repeat_equal_refuses(tmp_path):
    with pytest.raises(mc.Refused, match="second run"):
        mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda", repeat=False)))


def test_one_gpu_vendor_is_unverified(tmp_path):
    cat, _, _ = mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda")))
    assert cat["models"][0]["status"] == "unverified: no AMD run"


def test_refused_model(tmp_path):
    cat, _, _ = mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda", status="refused: embedding"),
                                                   _run("hip", status="refused: embedding")))
    assert cat["models"][0]["status"] == "refused: Gather"


def test_failed_run_is_skipped(tmp_path):
    cat, models, failed = mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda"), _run("hip", status="failed")))
    assert len(failed) == 1 and cat["models"][0]["status"] == "unverified: no AMD run"


def test_merge_keeps_versions_and_refuses_a_changed_digest(tmp_path):
    path = tmp_path / "c.json"
    cat, _, _ = mc.build(path, _dirs(tmp_path / "a", _run("cuda"), _run("hip")))
    mc._write(cat, path)
    cat2, _, _ = mc.build(path, _dirs(tmp_path / "b", _run("cuda", digest="dddd", version="0.9.1"),
                                      _run("hip", digest="dddd", version="0.9.1")))
    assert set(cat2["models"][0]["versions"]) == {"0.9.0", "0.9.1"}
    mc._write(cat2, path)
    with pytest.raises(mc.Refused, match="disagree"):
        mc.build(path, _dirs(tmp_path / "c", _run("metal", digest="eeee", version="0.9.1")))


def test_different_source_refuses(tmp_path):
    other = _run("hip")
    other["source"]["weights"] = "ResNet18_Weights.IMAGENET1K_V2"
    with pytest.raises(mc.Refused, match="source.weights"):
        mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda"), other))


def test_shipped_catalog_has_the_schema():
    doc = json.loads(mc.CATALOG.read_text())
    assert doc["schema"] == mc.SCHEMA
    for e in doc["models"]:
        assert e["status"] == mc.entry_status(e)
