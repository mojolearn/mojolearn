# SPDX-License-Identifier: Apache-2.0
"""The loader recognises an architecture set whose binaries live only at the tier level (identical-only releases, 0.8.56)."""
from mojolearn import _backend as b


def test_identical_only_architecture_set_is_recognised(tmp_path):
    arch = tmp_path / "cuda_native" / "sm_89"
    (arch / "identical").mkdir(parents=True)
    (arch / "identical" / "_mojolearn_native.so").write_bytes(b"\x7fELF")
    (arch / "_adapters").mkdir()
    (arch / "_adapters" / "libmojolearn_adapter.so").write_bytes(b"\x7fELF")
    assert b._has_binaries(str(arch))
    assert b._arch_dirs(str(tmp_path / "cuda_native")) == ["sm_89"]


def test_a_bare_architecture_directory_is_not_a_set(tmp_path):
    (tmp_path / "cuda_native" / "sm_89" / "identical").mkdir(parents=True)
    assert not b._has_binaries(str(tmp_path / "cuda_native" / "sm_89"))
    assert b._arch_dirs(str(tmp_path / "cuda_native")) == []
