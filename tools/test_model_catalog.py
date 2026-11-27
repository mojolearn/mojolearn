# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn.
"""tools/model_catalog.py: verified only when NVIDIA and AMD agree; any digest disagreement refuses."""
import json

import pytest

import model_catalog as mc


OUTPUT = "a" * 16


def _run(vendor, digest=OUTPUT, status="ok", version="0.9.0", mode="identical", repeat=True, **extra):
    doc = {"schema": mc.RUN_SCHEMA, "model": "resnet18", "kind": "torchvision", "batch": 1, "seed": 0,
           "status": status, "vendor": vendor, "numeric_mode": mode, "mojolearn_version": version,
           "gpu_arch": vendor + "-arch", "input_digest": "0" * 16, "input_shapes": [[1, 3, 224, 224]],
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
    assert e["versions"]["0.9.0"]["identical"]["digests"] == dict.fromkeys(("cuda", "hip", "metal"), OUTPUT)
    assert e["accuracy"]["published"]["value"] == 0.6976
    assert "ResNet18_Weights.IMAGENET1K_V1" in e["import"]


def test_disagreeing_digest_refuses(tmp_path):
    with pytest.raises(mc.Refused, match="disagree"):
        mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda"), _run("hip", digest="b" * 16)))


def test_apple_disagreeing_refuses_too(tmp_path):
    with pytest.raises(mc.Refused, match="disagree"):
        mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda"), _run("hip"), _run("metal", digest="c" * 16)))


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
    cat2, _, _ = mc.build(path, _dirs(tmp_path / "b", _run("cuda", digest="d" * 16, version="0.9.1"),
                                      _run("hip", digest="d" * 16, version="0.9.1")))
    assert set(cat2["models"][0]["versions"]) == {"0.9.0", "0.9.1"}
    mc._write(cat2, path)
    with pytest.raises(mc.Refused, match="disagree"):
        mc.build(path, _dirs(tmp_path / "c", _run("metal", digest="e" * 16, version="0.9.1")))


@pytest.mark.parametrize(("options", "error"), [
    ({"digest": None}, "output digest"),
    ({"digest": "too short"}, "output digest"),
    ({"input_digest": None}, "input digest"),
    ({"input_digest": "z" * 16}, "input digest"),
    ({"repeat": "false"}, "second run"),
    ({"status": "pending", "ours": {"digest": OUTPUT, "repeat_equal": True}}, "unknown run status"),
])
def test_incomplete_runs_cannot_verify_a_model(tmp_path, options, error):
    dirs = _dirs(tmp_path, _run("cuda", **options), _run("hip", **options))
    with pytest.raises(mc.Refused, match=error):
        mc.build(tmp_path / "c.json", dirs)


@pytest.mark.parametrize("statuses", [("ok", "refused"), ("refused", "ok")])
def test_conflicting_import_in_a_later_merge_preserves_the_catalog(tmp_path, statuses):
    path = tmp_path / "c.json"
    first = _dirs(tmp_path / "a", _run("cuda", status=statuses[0]))
    second = _dirs(tmp_path / "b", _run("hip", status=statuses[1]))
    assert mc.main(["build", "--catalog", str(path), "--runs", *first]) == 0
    saved = path.read_bytes()
    assert mc.main(["build", "--catalog", str(path), "--runs", *second]) == 1
    assert path.read_bytes() == saved


def test_import_can_succeed_in_a_later_release(tmp_path):
    dirs = _dirs(tmp_path, _run("cuda", status="refused", version="0.8.9"), _run("cuda"), _run("hip"))
    cat, _, _ = mc.build(tmp_path / "c.json", dirs)
    entry, = cat["models"]
    assert entry["status"] == "verified"
    assert entry["versions"]["0.8.9"]["identical"]["status"] == "refused"


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
        for version, modes in e["versions"].items():
            mc.entry_status(dict(e, versions={version: modes}))


def test_inconsistent_retained_evidence_cannot_supply_a_verified_status(tmp_path):
    cat, _, _ = mc.build(tmp_path / "c.json", _dirs(tmp_path, _run("cuda"), _run("hip")))
    entry, = cat["models"]
    cell = mc.newest_cell(entry)
    cell["digests"]["hip"] = "b" * 16
    with pytest.raises(mc.Refused, match="disagree"):
        mc.entry_status(entry)
    cell["digests"]["hip"] = OUTPUT
    del cell["vendors"]["hip"]["repeat_equal"]
    with pytest.raises(mc.Refused, match="confirmed repeat"):
        mc.entry_status(entry)
