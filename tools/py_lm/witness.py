# SPDX-License-Identifier: Apache-2.0
"""lane/py-lm witness: digests and timings for the three items the lane moves.

    python tools/py_lm/witness.py --device gpu|cpu --out FILE [--timing]

One run is one column (one tree, one device). Two columns compare with
`--compare A B`: every digest must be equal. Run it on the BASE tree and on
the lane tree, same box, same job, both devices, and compare base == lane per
device (the before == after rule of docs/lanes/progress/py-lm.md).

  causal   CausalLM over the eight synthetic checkpoint families
           (`_causal_lm_fixtures`), float32, bfloat16 and int8 weights:
           `generate(prompt, 40)` token streams, plus the prefill/step
           logits the per-layer route produces (untouched by the lane, a
           control). On a GPU the transformer rows take the resident session.
  samba    SambaStack (Mamba-3 + attention) three AdamW steps with ignored
           targets and clipping, then an accumulation_steps=2 stack: losses,
           the flat parameter buffer, forward logits (GPU only; SambaStack is
           a GPU surface).
  bytelm   SmallByteLanguageModelTrainer, stateless and resident, three
           steps: losses, parameters, flat gradients, logits.
  timing   (with --timing) generate at a GPT-2-small-like shape, a Samba
           step at vocab 32000, and a stateless byte LM step at ~13M
           parameters; seconds, not digests.
"""
import argparse
import hashlib
import json
import os
import struct
import sys
import tempfile
import time


def _h(*arrays):
    d = hashlib.sha256()
    for a in arrays:
        d.update(bytes(memoryview(a).cast("B")) if not isinstance(a, (bytes, bytearray)) else a)
    return d.hexdigest()


def _ids(n, seed, lo, hi):
    """Deterministic ids in [lo, hi) from a 64-bit LCG."""
    x = (seed * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
    out = []
    for _ in range(n):
        x = (x * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append(lo + (x >> 33) % (hi - lo))
    return out


def causal(ml, device, out):
    from mojolearn import Array
    from mojolearn.models import CausalLM
    from mojolearn import _causal_lm_fixtures as fx
    arches = (("llama", False), ("llama", True), ("mistral", False), ("qwen2", False),
              ("qwen3", False), ("phi3", False), ("mamba", True), ("mamba2", True))
    with tempfile.TemporaryDirectory() as root:
        for arch, tied in arches:
            cfg, tensors = fx.family_fixture(arch, tied)
            path = os.path.join(root, f"{arch}-{tied}")
            fx._write_checkpoint(path, cfg, tensors)
            for fmt in ("float32", "bfloat16", "int8"):
                key = f"causal/{arch}-{'tied' if tied else 'untied'}-{fmt}"
                try:
                    lm = CausalLM.load(path, device=device, weight_format=fmt)
                    prompt = Array.from_list([[1, 7, 3], [2, 9, 2]], "<i4")
                    gen = lm.generate(prompt, 40)
                    ids = Array.from_list([[1, 7, 3, 11, 5], [2, 9, 2, 4, 8]], "<i4")
                    state = lm.allocate_state(2, 8)
                    pre = lm.forward(ids[:, :3], state)
                    s1 = lm.step(ids[:, 3:4], state)
                    s2 = lm.step(ids[:, 4:5], state)
                    out[key] = dict(generate=_h(gen), prefill=_h(pre), step1=_h(s1), step2=_h(s2),
                                    stream=gen.tolist()[0][3:])
                except Exception as exc:  # noqa: BLE001
                    out[key] = dict(error=f"{type(exc).__name__}: {exc}"[:300])


def samba(ml, out):
    from mojolearn import Array
    T = ml.training
    cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64)
    m = ml.SambaStack(cfg, generator=T.Generator(1), lr=1e-3, max_norm=1.0)
    losses = []
    for k in range(3):
        raw = _ids(2 * 17, 100 + k, 0, 256)
        x = Array.from_list([raw[0:16], raw[17:33]], "<i4")
        y = [raw[1:17], raw[18:34]]
        y[1][5] = -100
        if k == 2:
            y[0][0] = -100
        y = Array.from_list(y, "<i4")
        losses.append(struct.pack("<d", float(m.train_step(x, y)["loss"])))
    probe = Array.from_list([_ids(16, 7, 0, 256), _ids(16, 8, 0, 256)], "<i4")
    out["samba/steps"] = dict(loss=_h(b"".join(losses)), params=_h(m.flat), logits=_h(m.forward(probe)))
    m2 = ml.SambaStack(cfg, generator=T.Generator(2), lr=1e-3, accumulation_steps=2)
    raw = _ids(4 * 17, 55, 0, 256)
    x = Array.from_list([raw[r * 17:r * 17 + 16] for r in range(4)], "<i4")
    y = Array.from_list([raw[r * 17 + 1:r * 17 + 17] for r in range(4)], "<i4")
    try:
        loss = m2.train_step(x, y)["loss"]
        out["samba/accum2"] = dict(loss=_h(struct.pack("<d", float(loss))), params=_h(m2.flat))
    except Exception as exc:  # noqa: BLE001
        out["samba/accum2"] = dict(error=f"{type(exc).__name__}: {exc}"[:300])


def _byte_params(ml, shape):
    import numpy as np
    named = {}
    for i, (name, shp) in enumerate(zip(shape.parameter_names, shape.parameter_shapes)):
        n = 1
        for s in shp:
            n *= s
        if name.endswith("norm1_w") or name.endswith("norm2_w"):
            vals = np.ones(n, dtype=np.float32)
        else:
            vals = np.array([(v / float(1 << 30)) * 0.25 - 0.125 for v in _ids(n, 9 + i, 0, 1 << 30)],
                            dtype=np.float32)
        named[name] = vals.reshape(tuple(shp))
    return named


def bytelm(ml, out):
    shape = ml.ByteLanguageModelConfig(n_layers=1, d_model=16, n_heads=2, n_kv=1, head_dim=8, intermediate=32)
    raw = _ids(3 * shape.batch * (shape.length + 1), 3, 0, 256)
    w = shape.length + 1
    for resident in (False, True):
        key = "bytelm/" + ("resident" if resident else "stateless")
        try:
            m = ml.SmallByteLanguageModelTrainer(_byte_params(ml, shape),
                                                 data_schedule={"dataset": "py-lm", "order": "sequential"},
                                                 shape=shape, resident=resident)
            losses, grads = [], None
            for k in range(3):
                base = k * shape.batch * w
                ids = ml.Array.from_list([raw[base + r * w:base + (r + 1) * w] for r in range(shape.batch)], "<i4")
                r = m.train_step(ids)
                losses.append(struct.pack("<d", float(r["loss"])))
                grads = r.get("flat_gradients")
            probe = ml.Array.from_list([raw[r * w:r * w + shape.length] for r in range(shape.batch)], "<i4")
            nb = list(m.next_bytes(probe)) if hasattr(m, "next_bytes") else None
            out[key] = dict(loss=_h(b"".join(losses)), params=_h(m.parameters_), logits=_h(m.logits(probe)),
                            grads=_h(grads) if grads is not None else None, next_bytes=nb)
        except Exception as exc:  # noqa: BLE001
            out[key] = dict(error=f"{type(exc).__name__}: {exc}"[:300])


def timing(ml, device, out):
    import numpy as np
    from mojolearn import Array
    from mojolearn.models import CausalLM
    from mojolearn import _causal_lm_fixtures as fx
    rng = np.random.default_rng(0)

    def t(shape, scale=0.02):
        a = (rng.standard_normal(shape, dtype=np.float32) * scale).astype(np.float32)
        return ("F32", tuple(shape), a.tobytes())
    dm, nl, v, it = 768, 12, 50257, 3072
    cfg = fx._llama_config(hidden_size=dm, num_attention_heads=12, num_key_value_heads=12,
                           intermediate_size=it, num_hidden_layers=nl, vocab_size=v,
                           max_position_embeddings=1024, tie_word_embeddings=True)
    tensors = {"model.embed_tokens.weight": t((v, dm)), "model.norm.weight": fx._ones((dm,))}
    for i in range(nl):
        p = f"model.layers.{i}."
        tensors[p + "input_layernorm.weight"] = fx._ones((dm,))
        tensors[p + "post_attention_layernorm.weight"] = fx._ones((dm,))
        for name, shp in (("self_attn.q_proj.weight", (dm, dm)), ("self_attn.k_proj.weight", (dm, dm)),
                          ("self_attn.v_proj.weight", (dm, dm)), ("self_attn.o_proj.weight", (dm, dm)),
                          ("mlp.gate_proj.weight", (it, dm)), ("mlp.up_proj.weight", (it, dm)),
                          ("mlp.down_proj.weight", (dm, it))):
            tensors[p + name] = t(shp)
    n_new = 64
    with tempfile.TemporaryDirectory() as root:
        fx._write_checkpoint(root, cfg, tensors, shards=1)
        lm = CausalLM.load(root, device=device)
        prompt = Array.from_list([_ids(16, 1, 0, v)], "<i4")
        lm.generate(prompt, 2)  # warm the kernels
        t0 = time.perf_counter()
        gen = lm.generate(prompt, n_new)
        dt = time.perf_counter() - t0
        out["timing/generate_gpt2s_b1_p16_n64"] = dict(seconds=dt, per_token_ms=1e3 * dt / n_new, stream=_h(gen))
    if device == "gpu":
        T = ml.training
        cfg = ml.SambaConfig(vocab=32000, d_model=256, layers=("mamba3", "attention"), n_heads=2, intermediate=1024)
        m = ml.SambaStack(cfg, generator=T.Generator(1), lr=1e-3)
        raw = _ids(4 * 129, 11, 0, 32000)
        x = Array.from_list([raw[r * 129:r * 129 + 128] for r in range(4)], "<i4")
        y = Array.from_list([raw[r * 129 + 1:r * 129 + 129] for r in range(4)], "<i4")
        m.train_step(x, y)
        t0 = time.perf_counter()
        losses = [m.train_step(x, y)["loss"] for _ in range(3)]
        dt = (time.perf_counter() - t0) / 3
        out["timing/samba_step_v32000_d256_b4_l128"] = dict(seconds=dt, loss=_h(struct.pack("<3d", *losses)),
                                                            params=_h(m.flat))
    shape = ml.ByteLanguageModelConfig(n_layers=4, d_model=512, n_heads=8, n_kv=8, head_dim=64, intermediate=1536)
    named = {}
    for name, shp in zip(shape.parameter_names, shape.parameter_shapes):
        if name.endswith("norm1_w") or name.endswith("norm2_w"):
            named[name] = np.ones(shp, dtype=np.float32)
        else:
            named[name] = (rng.standard_normal(shp, dtype=np.float32) * 0.02).astype(np.float32)
    m = ml.SmallByteLanguageModelTrainer(named, data_schedule={"dataset": "py-lm", "order": "sequential"},
                                         shape=shape)
    w = shape.length + 1
    raw = _ids(shape.batch * w, 21, 0, 256)
    ids = ml.Array.from_list([raw[r * w:(r + 1) * w] for r in range(shape.batch)], "<i4")
    m.train_step(ids)
    t0 = time.perf_counter()
    loss = m.train_step(ids)["loss"]
    dt = time.perf_counter() - t0
    out["timing/bytelm_stateless_step_%dparams" % shape.n_total] = dict(seconds=dt, loss=repr(float(loss)))


def compare(a, b, common=False):
    A, B = json.load(open(a)), json.load(open(b))
    bad, same = [], 0
    keys = (set(A) & set(B)) if common else (set(A) | set(B))
    for k in sorted(keys):
        if k.startswith("timing/") or k.startswith("_"):
            continue
        x, y = A.get(k), B.get(k)
        if x is None or y is None or "error" in (x or {}) or "error" in (y or {}):
            bad.append((k, "MISSING/ERROR", x, y))
            continue
        for f in sorted(set(x) | set(y)):
            if x.get(f) != y.get(f):
                bad.append((k, f, x.get(f), y.get(f)))
        same += 1
    for k in sorted(A):
        if k.startswith("timing/") and k in B:
            ta, tb = A[k].get("seconds"), B[k].get("seconds")
            print(f"TIMING {k}: {ta:.4f} s -> {tb:.4f} s ({ta / tb:.1f}x)"
                  + ("" if A[k].get("stream", A[k].get("params")) == B[k].get("stream", B[k].get("params"))
                     else "  DIGEST DIFFERS"))
    for row in bad:
        print("DIFF", *row)
    print(f"RESULT {'AGREE' if not bad and same else 'DISAGREE'} cells={same} diffs={len(bad)}")
    return 0 if not bad and same else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=("gpu", "cpu"))
    ap.add_argument("--out")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--only", default="causal,samba,bytelm")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--common", action="store_true", help="compare only cells both columns hold")
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare, common=a.common)
    import mojolearn as ml
    out = {"_device": a.device, "_vendor": str(ml._backend.vendor())}
    only = a.only.split(",")
    if "causal" in only:
        causal(ml, a.device, out)
    if "samba" in only and a.device == "gpu":
        samba(ml, out)
    if "bytelm" in only:
        bytelm(ml, out)
    if a.timing:
        timing(ml, a.device, out)
    json.dump(out, open(a.out, "w"), indent=1, sort_keys=True)
    errs = [k for k, v in out.items() if isinstance(v, dict) and "error" in v]
    print(f"witness {a.device}: {len(out) - 2} cells, {len(errs)} errors {errs[:6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
