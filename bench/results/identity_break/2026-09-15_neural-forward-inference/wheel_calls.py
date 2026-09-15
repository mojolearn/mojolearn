"""Every new public call, hashed: run once from the source tree and once
from the installed wheel; the two JSONs must be equal."""
import hashlib, json, os, sys, tempfile
import numpy as np
import mojolearn as ml
out = {"mojolearn": os.path.dirname(ml.__file__), "vendor": ml.vendor()}
h = lambda a: hashlib.sha256(np.ascontiguousarray(np.asarray(a)).tobytes()).hexdigest()[:16]
rng = np.random.default_rng(2026)
def mw(kind, dm=32):
    di = 2 * dm; nh = di // 64
    if kind == "m1":
        r = -(-dm // 16)
        s = {"norm.weight": (dm,), "in_proj.weight": (2 * di, dm), "conv1d.weight": (di, 1, 4), "conv1d.bias": (di,),
             "x_proj.weight": (r + 32, di), "dt_proj.weight": (di, r), "dt_proj.bias": (di,), "A_log": (di, 16),
             "D": (di,), "out_proj.weight": (dm, di)}
    elif kind == "m2":
        s = {"block_norm.weight": (dm,), "in_proj.weight": (2 * di + 256 + nh, dm), "conv1d.weight": (di + 256, 1, 4),
             "conv1d.bias": (di + 256,), "dt_bias": (nh,), "A_log": (nh,), "D": (nh,), "norm.weight": (di,),
             "out_proj.weight": (dm, di)}
    else:
        s = {"block_norm.weight": (dm,), "in_proj.weight": (2 * di + 256 + 3 * nh + 32, dm), "dt_bias": (nh,),
             "B_norm.weight": (128,), "C_norm.weight": (128,), "B_bias": (nh, 128), "C_bias": (nh, 128), "D": (nh,),
             "out_proj.weight": (dm, di)}
    return {n: (rng.standard_normal(v) * 0.1).astype(np.float32) for n, v in s.items()}
x = rng.standard_normal((3, 70, 32)).astype(np.float32)
lens = [70, 9, 65]
for name, blk in (("Mamba1BlockInference", ml.Mamba1BlockInference(mw("m1"))),
                  ("Mamba2BlockInference", ml.Mamba2BlockInference(mw("m2"))),
                  ("Mamba2BlockInference.dt_limit", ml.Mamba2BlockInference(mw("m2"), dt_limit=(0.01, 0.1))),
                  ("Mamba3BlockInference", ml.Mamba3BlockInference(mw("m3")))):
    out[name + ".forward"] = h(blk.forward(x))
    out[name + ".forward(lengths)"] = h(blk.forward(x, lengths=lens))
for tied in (True, False):
    cfg = ml.SambaConfig(vocab=256, d_model=32, layers=("mamba3", "attention"), n_heads=2, intermediate=64,
                         tie_embeddings=tied)
    w = {n: (rng.standard_normal(s) * 0.1).astype(np.float32) for n, s in cfg.registry()}
    inf = ml.SambaInference(cfg, w)
    ids = rng.integers(0, 256, (3, 20)).astype(np.int32)
    out[f"SambaInference(tied={tied}).forward"] = h(inf.forward(ids))
    out[f"SambaInference(tied={tied}).forward(lengths)"] = h(inf.forward(ids, lengths=[20, 1, 7]))
shape = ml.ByteLanguageModelConfig()
p = (rng.standard_normal(shape.n_total) * 0.05).astype(np.float32)
ids = rng.integers(0, 256, (4, shape.length)).astype(np.int32)
for threaded in (False, True):
    lm = ml.LanguageModelInference(p, shape=shape, threaded=threaded)
    out[f"LanguageModelInference(threaded={threaded}).logits"] = h(lm.logits(ids))
    out[f"LanguageModelInference(threaded={threaded}).logits(lengths)"] = h(lm.logits(ids, lengths=[32, 1, 5, 17]))
json.dump(out, open(sys.argv[1], "w"), indent=1, sort_keys=True)
print(len(out) - 2, "calls hashed from", out["mojolearn"])
