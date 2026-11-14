# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the Transformer block lanes
(lane/cpu-training-transformer, 2026-09-15): transformer and
transformer-window. Checked from SOURCE so it runs on a box with nothing
built.

What the source checks hold: the manifest covers both lanes in the
transformer family, which routes `_mojolearn_transformer` to its own host
binding; that binding registers every entry the GPU binding registers (the
three forward spellings, the backward and the two read-backs) and every
name the manifest exports; the host glue and the two oracles import no GPU
module; the glue calls the two oracles and hands the window to both; the
sabotage define reaches the family through the gemm oracle; the CPU identity
gate workflow triggers on every host module; and the no-CPU-path sentence no
longer names the Transformer block.

The runtime checks (stateless forward == carried prefill, split prefill,
decode == prefill row, backward repeat, NaN refusal, full causal and window
8) moved to checks/native_tests/transformer_host_block.mojo
(2026-11-14); they call the host binding's typed RUN directly.

    cd python && python3 -m mojolearn.tests.test_cpu_training_transformer
"""

import re
import sys
from pathlib import Path

from mojolearn import host_surface

ROOT = Path(__file__).resolve().parents[3]

LANES = ("transformer", "transformer-window")
GLUE = "transformer/host/transformer_block_host.mojo"
ORACLES = ("transformer/checks/transformer_oracle.mojo", "transformer/checks/transformer_backward_oracle.mojo")
ENTRIES = ("transformer_forward", "transformer_forward_fresh", "transformer_decode_step",
           "transformer_backward", "transformer_vendor", "transformer_numeric_mode")
GPU_IMPORTS = re.compile(r"^\s*from\s+(max\.gpu|std\.gpu)", re.M)


def _read(rel):
    return (ROOT / rel).read_text(encoding="utf-8")


def _registers(src, name):
    return re.search(r'\(\s*"' + re.escape(name) + r'"\s*\)', src) is not None


def test_manifest_covers_the_transformer_lanes():
    fam = host_surface.family("transformer")
    assert fam["routes"] == "_mojolearn_transformer" and fam["binding"] == "_mojolearn_transformer_host"
    covered = host_surface.covered_lanes()
    for lane in LANES:
        assert lane in fam["training_lanes"] and lane in covered, lane
        assert host_surface.TRAINING_LANE_NAMES[lane] in host_surface.training_sentence()
    for rel in (GLUE,) + ORACLES:
        assert rel in fam["host_modules"] and (ROOT / rel).is_file(), rel
    assert "TransformerBlock" in fam["classes"]
    assert host_surface.routed_modules()["_mojolearn_transformer"] == "_mojolearn_transformer_host"


def test_binding_registers_the_gpu_entries():
    src = _read(host_surface.binding_source("transformer"))
    gpu = _read("bindings/_mojolearn_transformer.mojo")
    for name in ENTRIES:
        assert _registers(gpu, name), f"{name} is not a transformer GPU binding name"
        assert _registers(src, name), f"the transformer host binding does not register {name}"
    for name in host_surface.family("transformer")["exports"]:
        assert _registers(src, name), f"the manifest exports {name} and the binding does not register it"


def test_host_modules_import_no_gpu_module():
    for rel in (GLUE, "bindings/_mojolearn_transformer_host.mojo"):
        text = _read(rel)
        assert not GPU_IMPORTS.search(text), f"{rel} imports a GPU module"
        assert "DeviceContext" not in re.sub(r'"""(.|\n)*?"""', "", text), f"{rel} names DeviceContext"
    for rel in ORACLES:
        assert not GPU_IMPORTS.search(_read(rel)), f"{rel} imports a GPU module"


def test_glue_calls_the_oracles_with_the_window():
    text = _read(GLUE)
    assert "transformer_block_oracle(w, x, b, l, cache, rope, ScorePlant.none())" in text
    assert "transformer_block_backward_oracle(w, fwd, d_out, b, l, 0, rope, window)" in text
    assert "TransformerKVCache(b, dims, smax, window)" in text
    assert "TransformerKVCache(b, dims, l, window)" in text
    # The device backward recomputes the forward at positions [0, L) with a
    # rotary table of L positions; the host builds its dims at L.
    assert "dm, nh, nkv, hd, it, l," in _read("bindings/_mojolearn_transformer_host.mojo")


def test_sabotage_define_reaches_the_family():
    fam = host_surface.family("transformer")
    assert fam["sabotage_define"] == "MOJOLEARN_HOST_SABOTAGE"
    assert "gemm/host/gemm_oracle.mojo" in fam["host_modules"]
    # 390e1cbbf: the define is read once in gemm/contract.mojo (the profile
    # contract the device GEMMs and the host oracles both import)
    assert 'GEMM_ORACLE_HOST_SABOTAGE = is_defined["MOJOLEARN_HOST_SABOTAGE"]()' in _read("gemm/contract.mojo")
    assert "from gemm.contract import GEMM_ORACLE_HOST_SABOTAGE" in _read(GLUE)
    assert "TRANSFORMER_HOST_SABOTAGE = GEMM_ORACLE_HOST_SABOTAGE" in _read(GLUE)


def test_workflow_triggers_on_the_host_modules():
    text = _read(".github/workflows/cpu-identity-gate.yml")
    # Since Sep 15 the gate has no push trigger (it runs by hand, weekly or
    # from the release workflow), so there is no path filter to check; a
    # push trigger, if one returns, must still list the host modules.
    if "\n  push:" not in text:
        return
    for rel in host_surface.family("transformer")["host_modules"]:
        assert f'- "{rel}"' in text, f"cpu-identity-gate.yml does not trigger on {rel}"


def test_no_cpu_path_sentence_drops_the_transformer():
    sentence = host_surface.no_cpu_path_sentence()
    assert "Transformer" not in sentence, sentence


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
