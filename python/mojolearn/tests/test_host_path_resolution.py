# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE resolution path for every host binding (lane/host-path-resolution,
2026-09-16).

Every door onto a CPU host binding must name the same directory. Until this
lane there were two answers. `_backend.host_dir()` honored
`MOJOLEARN_HOST_DIR` and `load_host_module` went through it, while
`_byte_lm_host.binary_path()` and `_forest_host.binary_path()` (the byte LM
and forest INFERENCE doors) and `_backend.host_binding_path()` /
`forest_host_binding_path()` built their own path from the package directory
and never read the variable. Under the override the training door opened and
the inference door did not: measured on this commit's parent, the `byte-lm`
lane's train cell was STABLE while its `infer` and `batch` parts read
REFUSED, every refusal naming a file under the package directory rather than
the override. With a host set present in the package the same split is
silent instead: the inference door loads the PACKAGE binding while the
process was told to use another set, which is how a forest sabotage column
once read IDENTICAL (the guard in tools/identity_break.py:6210).

These are path assertions plus one end-to-end arm: under the override, in a
subprocess, the byte LM inference door must load the override's file and
answer loss bits rather than refuse. The subprocess is what makes the arm
honest -- the module caches its binding for the life of a process.

    cd python && python -m pytest mojolearn/tests/test_host_path_resolution.py
"""
import json
import os
import subprocess
import sys

import pytest

from mojolearn import _backend, _byte_lm_host, _forest_host

#: label -> (resolver, the file it names). Every door onto a host binding
#: that answers a PATH. A new one belongs here, or it can drift apart again.
RESOLVERS = {
    "_backend.host_module_path(byte_lm)":
        (lambda: _backend.host_module_path("_mojolearn_byte_lm_host"), "_mojolearn_byte_lm_host.so"),
    "_backend.host_module_path(forest)":
        (lambda: _backend.host_module_path("_mojolearn_forest_host"), "_mojolearn_forest_host.so"),
    "_backend.host_binding_path":
        (_backend.host_binding_path, "_mojolearn_byte_lm_host.so"),
    "_backend.forest_host_binding_path":
        (_backend.forest_host_binding_path, "_mojolearn_forest_host.so"),
    "_byte_lm_host.binary_path":
        (_byte_lm_host.binary_path, "_mojolearn_byte_lm_host.so"),
    "_forest_host.binary_path":
        (_forest_host.binary_path, "_mojolearn_forest_host.so"),
}

#: The per-binding variables that name ONE file. They must keep winning over
#: the directory: the CPU identity gate points them at its sabotage build.
NAMED_BINARY_ENV = ("MOJOLEARN_BYTE_LM_HOST_BINARY", "MOJOLEARN_FOREST_HOST_BINARY")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """Neither the directory nor a named binary leaks in from the caller's
    environment; each test sets what it means to measure."""
    for key in ("MOJOLEARN_HOST_DIR",) + NAMED_BINARY_ENV:
        monkeypatch.delenv(key, raising=False)


def _resolved():
    return {label: fn() for label, (fn, _) in RESOLVERS.items()}


def test_under_the_override_every_resolver_follows_it(monkeypatch, tmp_path):
    """THE DEFECT. One directory, named once, is where every door looks."""
    monkeypatch.setenv("MOJOLEARN_HOST_DIR", str(tmp_path))
    wrong = {label: path for label, path in _resolved().items()
             if os.path.dirname(path) != str(tmp_path)}
    assert not wrong, (
        "MOJOLEARN_HOST_DIR=%s and these resolvers looked somewhere else:\n%s"
        % (tmp_path, "".join(f"    {k}() -> {v}\n" for k, v in sorted(wrong.items()))))
    for label, (fn, basename) in RESOLVERS.items():
        assert os.path.basename(fn()) == basename, label


def test_a_relative_override_is_made_absolute(monkeypatch, tmp_path):
    """`host_dir()` answers an absolute path for a relative override (the
    gate and the legs pass `python/mojolearn/host-sabotage`); every door
    answers the same one, so a chdir cannot move the binding."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "host-sab").mkdir()
    monkeypatch.setenv("MOJOLEARN_HOST_DIR", "host-sab")
    expected = str(tmp_path / "host-sab")
    for label, path in _resolved().items():
        assert os.path.isabs(path), label
        assert os.path.dirname(path) == expected, label


def test_without_the_override_every_resolver_stays_in_the_package(monkeypatch):
    """THE UNCHANGED PATH. With no override every door names
    `<package>/host/`, exactly as before this lane, which is what the wheel
    and every recorded hash rest on."""
    expected = os.path.join(_backend._pkg_dir(), "host")
    for label, path in _resolved().items():
        assert os.path.dirname(path) == expected, f"{label} -> {path}"


def test_a_named_binary_still_wins_over_the_directory(monkeypatch, tmp_path):
    """The CPU identity gate points MOJOLEARN_FOREST_HOST_BINARY at one file
    while MOJOLEARN_HOST_DIR names a directory. The file wins, and the
    directory still answers for everything else."""
    monkeypatch.setenv("MOJOLEARN_HOST_DIR", str(tmp_path / "dir"))
    named = {"MOJOLEARN_BYTE_LM_HOST_BINARY": tmp_path / "named_byte_lm.so",
             "MOJOLEARN_FOREST_HOST_BINARY": tmp_path / "named_forest.so"}
    for key, path in named.items():
        monkeypatch.setenv(key, str(path))
    assert _byte_lm_host.binary_path() == str(named["MOJOLEARN_BYTE_LM_HOST_BINARY"])
    assert _forest_host.binary_path() == str(named["MOJOLEARN_FOREST_HOST_BINARY"])
    # The directory is still the answer for a door that takes no named file.
    assert os.path.dirname(_backend.host_module_path("_mojolearn_core_host")) == str(tmp_path / "dir")


#: Run in a subprocess: `_byte_lm_host` caches its binding for the life of a
#: process, so the door can only be opened once per environment.
_DOOR = r"""
import hashlib, json, os, sys
import numpy as np
import mojolearn
from mojolearn import _backend, _byte_lm_host, _forest_host

out = {"host_dir": _backend.host_dir(), "binary_path": _byte_lm_host.binary_path()}
shape = mojolearn.ByteLanguageModelConfig()
flat = np.array([((i % 17) - 8) / 64.0 for i in range(shape.n_total)], dtype=np.float32)
ids = np.array([[(r * 7 + c) % 256 for c in range(shape.length + 1)]
                for r in range(shape.batch)], dtype=np.int32)
m = mojolearn.LanguageModelInference(flat, shape=shape, threaded=False)
out["module_file"] = m._binding.__file__
out["loss_bits"] = "0x%08x" % int(m.loss_bits(ids))
logits = np.asarray(m.logits(ids[:, :-1]))
out["logits_sha256"] = hashlib.sha256(
    str(logits.dtype).encode() + str(logits.shape).encode() + logits.tobytes()).hexdigest()
if os.path.exists(_forest_host.binary_path()):
    out["forest_module_file"] = _forest_host._load().__file__
print(json.dumps(out))
"""


def _mirror(tmp_path):
    """A complete host set beside the package's own, by symlink, so the
    override names REAL bindings that are not the package's files."""
    built = _backend.host_families_built()
    if "_mojolearn_byte_lm_host" not in built:
        pytest.skip("no _mojolearn_byte_lm_host.so built; "
                    "build it with bindings/build_byte_lm_host.sh")
    mirror = tmp_path / "host-mirror"
    mirror.mkdir()
    for basename in built:
        os.symlink(_backend.host_module_path(basename), mirror / (basename + ".so"))
    return mirror


def test_the_inference_door_opens_under_the_override(tmp_path):
    """THE CELL THAT REFUSED. Under the override the byte LM inference door
    must load the OVERRIDE's binding and answer loss bits and a logits hash.
    Unfixed it loads the package's own file, which is the silent half of
    this defect, and refuses outright when the package has no host set."""
    mirror = _mirror(tmp_path)
    env = dict(os.environ)
    for key in NAMED_BINARY_ENV:
        env.pop(key, None)
    env.update(MOJOLEARN_HOST_DIR=str(mirror), MOJOLEARN_NUMERIC_MODE="identical",
               PYTHONPATH=os.path.dirname(os.path.dirname(os.path.abspath(_backend.__file__))),
               OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
               NUMEXPR_NUM_THREADS="1", MOJOLEARN_CPU_THREADS="1")
    done = subprocess.run([sys.executable, "-c", _DOOR], env=env, timeout=600,
                          capture_output=True, text=True)
    assert done.returncode == 0, f"the inference door refused:\n{done.stdout}\n{done.stderr}"
    out = json.loads(done.stdout.strip().splitlines()[-1])
    assert out["host_dir"] == str(mirror)
    assert os.path.dirname(out["binary_path"]) == str(mirror), out["binary_path"]
    # The binding that ANSWERED, not the one that was asked for.
    assert os.path.dirname(out["module_file"]) == str(mirror), (
        "the byte LM inference door loaded %s while MOJOLEARN_HOST_DIR named %s"
        % (out["module_file"], mirror))
    if "forest_module_file" in out:
        assert os.path.dirname(out["forest_module_file"]) == str(mirror), out["forest_module_file"]
    # Hashes, not a refusal.
    assert len(out["logits_sha256"]) == 64
    int(out["loss_bits"], 16)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
