# SPDX-License-Identifier: Apache-2.0
"""Mapped checkpoint reads keep all package work inside the native entry."""
import array
import json

import pytest

from mojolearn import Array
from mojolearn._verify_native_runtime import audit_native_call
from mojolearn.models import Checkpoint, SafetensorsFile
from mojolearn.models.safetensors import write_safetensors


def test_native_checkpoint_reads_all_storage_types_and_closes_maps(tmp_path, monkeypatch):
    bf = array.array("H", [0, 0x8000, 0x3F81, 0x7FC1])
    half = array.array("H", [0, 0x8000, 0x0001, 0x3C00])
    tensors = {
        "f32": ("F32", (2, 2), array.array("f", [0.0, -0.0, 1.0, -2.0]).tobytes()),
        "bf16": ("BF16", (2, 2), bf.tobytes()),
        "f16": ("F16", (4,), half.tobytes()),
        "i32": ("I32", (), array.array("i", [-7]).tobytes()),
        "i64": ("I64", (2,), array.array("q", [-(2**40), 2**40]).tobytes()),
        "empty": ("F32", (2, 0), b""),
    }
    path = tmp_path / "model.safetensors"
    write_safetensors(path, tensors)
    file = SafetensorsFile.__new__(SafetensorsFile)
    audit_native_call(SafetensorsFile.__init__, file, path).assert_native()
    checkpoint = audit_native_call(Checkpoint.open, tmp_path).assert_native()
    assert sorted(audit_native_call(checkpoint.names).assert_native()) == sorted(tensors)

    def forbidden(*args, **kwargs):
        raise AssertionError("Python checkpoint callback executed")

    with monkeypatch.context() as patch:
        for name in ("__init__", "copy", "from_buffer", "reshape"):
            patch.setattr(Array, name, forbidden)
        patch.setattr(json, "loads", forbidden)
        values = {name: audit_native_call(checkpoint.read, name).assert_native() for name in tensors}
        packed = audit_native_call(file.read, "bf16", bf16="bits").assert_native()
        info = audit_native_call(file.info, "f32").assert_native()
        assert info.size == 4 and info.nbytes == 16
        audit_native_call(checkpoint.close).assert_native()
        audit_native_call(file.close).assert_native()
    assert values["f32"].tobytes() == tensors["f32"][2]
    assert values["bf16"].tobytes() == array.array("I", [v << 16 for v in bf]).tobytes()
    assert values["f16"].tobytes() == array.array("I", [0, 0x80000000, 0x33800000, 0x3F800000]).tobytes()
    assert values["i32"].shape == () and values["i32"].tobytes() == tensors["i32"][2]
    assert values["i64"].tobytes() == tensors["i64"][2]
    assert values["empty"].shape == (2, 0) and values["empty"].nbytes == 0
    assert packed.bits.tobytes() == bf.tobytes()
    assert file._mm is None and file._fh is None


def test_sharded_index_must_name_the_shard_that_holds_each_tensor(tmp_path):
    raw = array.array("f", [1.0]).tobytes()
    write_safetensors(tmp_path / "a.safetensors", {"a": ("F32", (1,), raw)})
    write_safetensors(tmp_path / "b.safetensors", {"b": ("F32", (1,), raw)})
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"a": "b.safetensors", "b": "a.safetensors"}}))
    checkpoint = Checkpoint.open(tmp_path)
    with pytest.raises(ValueError, match="does not hold"):
        checkpoint.names()
