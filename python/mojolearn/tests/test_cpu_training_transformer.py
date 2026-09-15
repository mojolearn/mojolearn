# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""CPU training for the Transformer block lanes
(lane/cpu-training-transformer, 2026-09-15): transformer and
transformer-window. Checked from SOURCE so it runs on a box with nothing
built, plus runtime checks that run only where the transformer host binding
is built and the package took the CPU-only path.

What the source checks hold: the manifest covers both lanes in the
transformer family, which routes `_mojolearn_transformer` to its own host
binding; that binding registers every entry the GPU binding registers (the
three forward spellings, the backward and the two read-backs) and every
name the manifest exports; the host glue and the two oracles import no GPU
module; the glue calls the two oracles and hands the window to both; the
sabotage define reaches the family through the gemm oracle; the CPU identity
gate workflow triggers on every host module; and the no-CPU-path sentence no
longer names the Transformer block.

The runtime checks (skipped, and SAID to be skipped, when the binding is
absent or a GPU set loaded), at the lanes' shape (d_model 32, two heads, one
kv head, intermediate 64), full causal and with window 8: the stateless
forward equals the carried-state prefill; a prefill split in two equals the
whole prefill; a prefill of 16 tokens then one decode step equals the 17th
row of the 17-token prefill (contract section 7.2's decode == prefill, which
the host inherits from the one oracle spelling); the ring state holds the
`min(cached_tokens, window)` newest keys; the backward returns the input and
nine weight gradients in their shapes and the same bytes twice; a NaN input
is refused by name. The bit claim against the GPU columns is the CPU identity
gate's, not this file's.

    cd python && python3 -m mojolearn.tests.test_cpu_training_transformer
"""
import re
import sys
from pathlib import Path

import mojolearn
from mojolearn import _backend, host_surface

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
    assert 'is_defined["MOJOLEARN_HOST_SABOTAGE"]()' in _read("gemm/host/gemm_oracle.mojo")
    assert "TRANSFORMER_HOST_SABOTAGE = GEMM_ORACLE_HOST_SABOTAGE" in _read(GLUE)


def test_workflow_triggers_on_the_host_modules():
    text = _read(".github/workflows/cpu-identity-gate.yml")
    for rel in host_surface.family("transformer")["host_modules"]:
        assert f'- "{rel}"' in text, f"cpu-identity-gate.yml does not trigger on {rel}"


def test_no_cpu_path_sentence_drops_the_transformer():
    sentence = host_surface.no_cpu_path_sentence()
    assert "Transformer" not in sentence and "Mamba and Samba blocks" in sentence, sentence


def _built():
    if _backend._CPU_ONLY is None:
        print("SKIP: a GPU set loaded; the host route is not taken here")
        return False
    if "_mojolearn_transformer_host" not in _backend.host_families_built():
        print("SKIP: _mojolearn_transformer_host is not built")
        return False
    return True


def _block(np, window):
    rng = np.random.default_rng(11)
    dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
    shapes = {
        "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
        "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm), "v_proj.weight": (nkv * hd, dm),
        "o_proj.weight": (dm, nh * hd), "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
        "down_proj.weight": (dm, it)}
    w = {k: (np.ones(s, np.float32) if k.endswith("layernorm.weight")
             else (rng.random(s, dtype=np.float32) * np.float32(0.25) - np.float32(0.125)))
         for k, s in shapes.items()}
    return mojolearn.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv, window=window)


def test_block_forward_and_backward_on_the_host_when_built():
    if not _built():
        return
    import numpy as np
    rng = np.random.default_rng(5)
    x = rng.standard_normal((2, 17, 32)).astype(np.float32)
    g = rng.standard_normal((2, 16, 32)).astype(np.float32)
    for window in (0, 8):
        blk = _block(np, window)
        whole16 = np.asarray(blk.forward(x[:, :16]))
        st = blk.allocate_state(2, 32)
        prefill = np.asarray(blk.forward(x[:, :16], st))
        assert whole16.tobytes() == prefill.tobytes(), f"window {window}: stateless forward != carried prefill"
        assert st.cached_tokens == 16
        held = np.asarray(st.keys())
        assert held.shape == (2, 1, min(16, window) if window else 16, 16), held.shape
        step = np.asarray(blk.step(x[:, 16:17], st))
        assert st.cached_tokens == 17
        whole17 = np.asarray(blk.forward(x))
        assert step.tobytes() == whole17[:, 16:17].tobytes(), f"window {window}: decode step != prefill row 16"
        assert whole17[:, :16].tobytes() == whole16.tobytes(), f"window {window}: a later token moved an earlier row"
        st2 = blk.allocate_state(2, 32)
        a = np.asarray(blk.forward(np.ascontiguousarray(x[:, :8]), st2))
        b = np.asarray(blk.forward(np.ascontiguousarray(x[:, 8:16]), st2))
        assert np.concatenate([a, b], axis=1).tobytes() == whole16.tobytes(), f"window {window}: split prefill moved"
        grads = blk.backward(x[:, :16], g)
        again = blk.backward(x[:, :16], g)
        assert sorted(grads) == sorted(("x",) + mojolearn.TransformerBlock._W_NAMES)
        assert np.asarray(grads["x"]).shape == (2, 16, 32)
        for k in blk._W_NAMES:
            assert np.asarray(grads[k]).shape == tuple(blk._w[blk._W_NAMES.index(k)].shape), k
        for k in grads:
            assert np.asarray(grads[k]).tobytes() == np.asarray(again[k]).tobytes(), k
            assert np.isfinite(np.asarray(grads[k])).all(), k
        bad = x[:, :16].copy()
        bad[0, 3, 5] = np.float32(np.nan)
        try:
            blk.forward(bad)
        except Exception as e:  # the oracle's by-name refusal
            assert "non-finite" in str(e) or "NaN" in str(e) or "nan" in str(e), str(e)
        else:
            raise AssertionError("a NaN input was not refused")


if __name__ == "__main__":
    names = [n for n in sorted(globals()) if n.startswith("test_")]
    for name in names:
        globals()[name]()
        print("ok", name)
    print(f"{len(names)} passed")
    sys.exit(0)
