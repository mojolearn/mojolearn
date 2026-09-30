#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""FAST MAY DIFFER IN BITS, NEVER IN QUALITY: the neural family's paired check.

The neural bindings (training, mamba, transformer, embedding) build FAST since
2026-09-27 (lane neural). FAST is the same kernels with the pins in
checks/numerics.mojo on the free schedule. This tool is the quality rule: at least 5 seeds on at least
2 datasets, FAST against IDENTICAL from the SAME initial weights and the SAME
batches, and both against torch eager float32 (TF32 off) from those same
weights and batches.

Subcommands (each writes one JSON; `compare` reads a directory of them):

  samba        SambaStack (Mamba-3 + attention + tied embedding, AdamW,
               clip_grad_norm_, warmup-cosine) trained on a byte corpus.
               Runs in the process tier MOJOLEARN_NUMERIC_MODE (fast or
               identical) and refuses a --mode that differs from it.
               Per-step loss, held-out loss before/after. --init-out writes
               the initial weights (npz, registry names) for the torch twin.
  samba-torch  The same stack in torch (tools/samba_torch_reference.py's
               modules), loaded from --init, same batches, same schedule.
  mlp-data     sklearn's bundled wine and digits (classes 0-2) standardized
               and projected to 8 principal components (SmallMLPTrainer's
               8 -> 16 -> 3 shape). Needs scikit-learn; writes one npz.
  mlp          SmallMLPTrainer on one dataset, one seed: a seeded train/test
               split, seeded init (torch Linear's default bound), minibatch
               AdamW; test loss and accuracy.
  mlp-torch    The same in torch from the same init, split and batches.
  blocks       Block-level accuracy against the float64 references: every
               Mamba-1/2/3 corpus case (mamba/corpus, ref64 from the
               reference algorithms) and TransformerBlock forward against a
               float64 NumPy restatement of HF's LlamaDecoderLayer at 5
               seeds x 2 shapes. Error = max|y - ref64| / max|ref64|.
  compare      The verdict table over a directory of JSON files.

THE RULE `compare` APPLIES. For each (task, dataset): the paired difference
d_s = FAST_s - IDENTICAL_s of the final held-out loss (lower is better) over
the seeds. FAST PASSES when mean(d) <= 2 * stderr(d) + 1e-4 * mean(IDENTICAL)
(no significant loss of quality) and no single seed is worse by more than 1%.
The same rule is applied to torch vs IDENTICAL and printed, so the reader sees
where the reference sits. For accuracy the sign flips. For `blocks`, FAST's
error must not exceed max(2 x IDENTICAL's error, the case's torch-float32
error, 1e-6).
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CORPORA = ("enwik8", "pile_github", "tinyshakespeare")
SCHEMA = "mojolearn.neural-fast-quality.v1"


def _sha(b):
    return hashlib.sha256(b).hexdigest()


def corpus_bytes(name):
    path = ROOT / "training" / "corpus" / name / "input.txt"
    if not path.is_file():
        sys.exit(f"neural_fast_quality: {path} is missing; stage it from R2 "
                 f"(tools/dataset_store.sh stage <box> corpus/{name}/input.txt)")
    return path.read_bytes()


def schedule(n, seed, steps, batch, seq, heldout_rows):
    """Training row starts inside the first 90% of the corpus, drawn from a
    seeded generator (so each seed also sees its own data order), and
    held-out row starts evenly spaced in the last 5%, never trained on."""
    import numpy as np
    lo, hi = 0, int(n * 0.90) - seq - 1
    rng = np.random.default_rng(0x6E657572 + seed)
    starts = rng.integers(lo, hi, size=(steps, batch))
    vlo, vhi = int(n * 0.95), n - seq - 1
    held = [vlo + (vhi - vlo) * j // max(1, heldout_rows - 1) for j in range(heldout_rows)]
    return starts, held


def rows_of(raw, starts, seq):
    import numpy as np
    ids = np.stack([np.frombuffer(raw[s:s + seq + 1], dtype=np.uint8).astype(np.int32) for s in starts])
    return np.ascontiguousarray(ids[:, :-1]), np.ascontiguousarray(ids[:, 1:])


def lr_at(t, lr, warmup, steps, min_lr):
    """WarmupCosineLR's rule (tools/samba_torch_reference.py's lr_at)."""
    if t <= warmup:
        return lr * t / warmup
    if t >= steps:
        return min_lr
    p = (t - warmup) / (steps - warmup)
    return min_lr + (lr - min_lr) * (1 + math.cos(math.pi * p)) / 2


def samba_shape(args):
    return dict(vocab=256, d_model=args.d_model, layers=args.layers.split(","),
                n_heads=args.n_heads, intermediate=args.intermediate)


def base(args, kind):
    return dict(schema=SCHEMA, kind=kind, host=os.uname().nodename,
                harness_sha256=_sha(Path(__file__).read_bytes()),
                args={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()})


# ----------------------------------------------------------------- samba
def cmd_samba(args):
    import numpy as np
    env = os.environ.get("MOJOLEARN_NUMERIC_MODE", "").lower()
    if env != args.mode:
        sys.exit(f"neural_fast_quality samba: MOJOLEARN_NUMERIC_MODE={env!r} but --mode {args.mode}")
    sys.path.insert(0, str(ROOT / "python"))
    import mojolearn
    from mojolearn.training import Generator, WarmupCosineLR, SambaConfig, SambaStack
    raw = corpus_bytes(args.corpus)
    starts, held = schedule(len(raw), args.seed, args.steps, args.batch, args.seq, args.heldout_rows)
    cfg = SambaConfig(**samba_shape(args))
    if args.init:
        # FAST from IDENTICAL's initial weights: the pair differs only in
        # the training arithmetic (the initializer is a FAST kernel too).
        init_npz = np.load(args.init)
        source = dict(weights={n: init_npz[n] for n in init_npz.files})
    else:
        source = dict(generator=Generator(args.seed))
    stack = SambaStack(cfg, **source, lr=args.lr,
                       lr_schedule=WarmupCosineLR(args.lr, args.warmup, args.steps, args.min_lr),
                       max_norm=args.max_norm)
    init = np.array(stack.flat, dtype=np.float32, copy=True)
    if args.init_out:
        args.init_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(args.init_out, **{n: np.array(stack.arrays[n], dtype=np.float32) for n in stack.names})
    hx, hy = rows_of(raw, held, args.seq)

    def heldout():
        return float(np.mean([stack.loss(hx[i:i + 8], hy[i:i + 8]) for i in range(0, len(hx), 8)]))

    before = heldout()
    losses, t0 = [], time.perf_counter()
    for s in range(args.steps):
        x, y = rows_of(raw, starts[s], args.seq)
        losses.append(float(stack.train_step(x, y)["loss"]))
        if s < 2 or (s + 1) % 50 == 0:
            print(json.dumps(dict(step=s + 1, loss=losses[-1])), flush=True)
    wall = time.perf_counter() - t0
    after = heldout()
    return dict(base(args, "samba"), mode=args.mode, dataset=args.corpus, seed=args.seed,
                numeric_mode_used=mojolearn.training.numeric_mode_used(),
                init_sha256=_sha(init.tobytes()), n_parameters=int(init.size),
                losses=losses, heldout_before=before, heldout_after=after,
                final_sha256=_sha(np.array(stack.flat, dtype=np.float32).tobytes()),
                train_seconds=wall)


def cmd_samba_torch(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    sys.path.insert(0, str(ROOT / "tools"))
    import samba_torch_reference as ref
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dev = torch.device(args.device)
    shp = samba_shape(args)
    model = ref.Stack(256, shp["d_model"], shp["layers"], shp["n_heads"], shp["intermediate"], True).to(dev)
    init = np.load(args.init)
    # SambaStack registry name -> the torch module's parameter.
    mapping = {"embed.weight": model.embed.weight, "norm_f.weight": model.norm_f}
    attn = {"input_layernorm.weight": "norm1", "post_attention_layernorm.weight": "norm2",
            "q_proj.weight": "q.weight", "k_proj.weight": "k.weight", "v_proj.weight": "v.weight",
            "o_proj.weight": "o.weight", "gate_proj.weight": "gate.weight",
            "up_proj.weight": "up.weight", "down_proj.weight": "down.weight"}
    m3 = {"block_norm.weight": "block_norm", "in_proj.weight": "in_proj.weight", "dt_bias": "dt_bias",
          "B_norm.weight": "B_norm", "C_norm.weight": "C_norm", "B_bias": "B_bias",
          "C_bias": "C_bias", "D": "D", "out_proj.weight": "out_proj.weight"}
    params = dict(model.named_parameters())
    for i, kind in enumerate(shp["layers"]):
        for ours, theirs in (m3 if kind == "mamba3" else attn).items():
            mapping[f"layers.{i}.{ours}"] = params[f"blocks.{i}.{theirs}"]
    if set(mapping) != set(init.files):
        sys.exit(f"samba-torch: registry mismatch {sorted(set(init.files) ^ set(mapping))}")
    with torch.no_grad():
        for n, p in mapping.items():
            src = torch.from_numpy(init[n])
            if tuple(src.shape) != tuple(p.shape):
                sys.exit(f"samba-torch: {n} shape {tuple(src.shape)} vs torch {tuple(p.shape)}")
            p.copy_(src.to(dev))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    raw = corpus_bytes(args.corpus)
    starts, held = schedule(len(raw), args.seed, args.steps, args.batch, args.seq, args.heldout_rows)
    hx, hy = rows_of(raw, held, args.seq)

    def t(a):
        return torch.from_numpy(a.astype(np.int64)).to(dev)

    def heldout():
        with torch.no_grad():
            return float(np.mean([float(F.cross_entropy(model(t(hx[i:i + 8])).reshape(-1, 256),
                                                        t(hy[i:i + 8]).reshape(-1)))
                                  for i in range(0, len(hx), 8)]))

    before = heldout()
    losses = []
    for s in range(args.steps):
        x, y = rows_of(raw, starts[s], args.seq)
        for g in opt.param_groups:
            g["lr"] = lr_at(s + 1, args.lr, args.warmup, args.steps, args.min_lr)
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(t(x)).reshape(-1, 256), t(y).reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        opt.step()
        losses.append(float(loss))
    after = heldout()
    return dict(base(args, "samba"), mode="torch", dataset=args.corpus, seed=args.seed,
                torch_version=torch.__version__, init_sha256=_sha(b"".join(
                    np.ascontiguousarray(init[n]).tobytes() for n in init.files)),
                losses=losses, heldout_before=before, heldout_after=after)


# ------------------------------------------------------------------- mlp
def cmd_mlp_data(args):
    import numpy as np
    from sklearn.datasets import load_digits, load_wine
    out = {}
    for name, loader in (("wine", load_wine), ("digits3", load_digits)):
        d = loader()
        X, y = d.data.astype(np.float64), d.target.astype(np.int64)
        keep = y < 3
        X, y = X[keep], y[keep]
        X = (X - X.mean(0)) / np.where(X.std(0) > 0, X.std(0), 1.0)
        _, _, vt = np.linalg.svd(X, full_matrices=False)
        Z = X @ vt[:8].T
        Z = Z / Z.std(0)
        out[name + "_X"] = Z.astype(np.float32)
        out[name + "_y"] = y.astype(np.int32)
    np.savez(args.out_npz, **out)
    print("wrote", args.out_npz, {k: v.shape for k, v in out.items()})
    return None


def mlp_setup(args):
    """Seeded split (70/30), seeded init (torch Linear default: U(-1/sqrt(fan_in), +)),
    seeded minibatch order. Shared by both sides."""
    import numpy as np
    d = np.load(args.data)
    X, y = d[args.dataset + "_X"], d[args.dataset + "_y"]
    rng = np.random.default_rng(1000 + args.seed)
    perm = rng.permutation(len(X))
    cut = int(0.7 * len(X))
    tr, te = perm[:cut], perm[cut:]

    def u(shape, fan_in):
        b = 1.0 / math.sqrt(fan_in)
        return rng.uniform(-b, b, size=shape).astype(np.float32)
    w = dict(weight1=u((16, 8), 8), bias1=u((16,), 8), weight2=u((3, 16), 16), bias2=u((3,), 16))
    order = [rng.permutation(tr) for _ in range(args.epochs)]
    return X, y, tr, te, w, order


def _softmax_loss_acc(logits, y):
    import numpy as np
    z = logits.astype(np.float64)
    z = z - z.max(1, keepdims=True)
    lse = np.log(np.exp(z).sum(1))
    return float(np.mean(lse - z[np.arange(len(y)), y])), float(np.mean(z.argmax(1) == y))


def cmd_mlp(args):
    import numpy as np
    env = os.environ.get("MOJOLEARN_NUMERIC_MODE", "").lower()
    if env != args.mode:
        sys.exit(f"neural_fast_quality mlp: MOJOLEARN_NUMERIC_MODE={env!r} but --mode {args.mode}")
    sys.path.insert(0, str(ROOT / "python"))
    from mojolearn.neural_network import SmallMLPTrainer
    X, y, tr, te, w, order = mlp_setup(args)
    m = SmallMLPTrainer(w["weight1"], w["bias1"], w["weight2"], w["bias2"],
                        data_schedule={"fixture": "neural_fast_quality", "seed": args.seed},
                        lr=args.lr, weight_decay=0.01)
    losses = []
    for ep in order:
        for i in range(0, len(ep), args.batch):
            idx = ep[i:i + args.batch]
            r = m.train_step(np.ascontiguousarray(X[idx]), np.ascontiguousarray(y[idx]))
            losses.append(float(r["loss"]))
    loss, acc = _softmax_loss_acc(np.asarray(m.predict_logits(np.ascontiguousarray(X[te]))), y[te])
    return dict(base(args, "mlp"), mode=args.mode, dataset=args.dataset, seed=args.seed,
                losses=losses, heldout_after=loss, accuracy=acc)


def cmd_mlp_torch(args):
    import numpy as np
    import torch
    import torch.nn.functional as F
    torch.backends.cuda.matmul.allow_tf32 = False
    dev = torch.device(args.device)
    X, y, tr, te, w, order = mlp_setup(args)
    P = {k: torch.nn.Parameter(torch.from_numpy(v).to(dev)) for k, v in w.items()}
    opt = torch.optim.AdamW(list(P.values()), lr=args.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)

    def fwd(xb):
        h = F.relu(xb @ P["weight1"].T + P["bias1"])
        return h @ P["weight2"].T + P["bias2"]
    Xt = torch.from_numpy(X).to(dev)
    yt = torch.from_numpy(y.astype(np.int64)).to(dev)
    losses = []
    for ep in order:
        for i in range(0, len(ep), args.batch):
            idx = torch.from_numpy(ep[i:i + args.batch]).to(dev)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(fwd(Xt[idx]), yt[idx])
            loss.backward()
            opt.step()
            losses.append(float(loss))
    with torch.no_grad():
        logits = fwd(Xt[torch.from_numpy(te).to(dev)]).cpu().numpy()
    loss, acc = _softmax_loss_acc(logits, y[te])
    return dict(base(args, "mlp"), mode="torch", dataset=args.dataset, seed=args.seed,
                losses=losses, heldout_after=loss, accuracy=acc)


# ---------------------------------------------------------------- blocks
def _llama64(x, w, nh, eps=1e-6, theta=10000.0):
    """HF LlamaDecoderLayer (modeling_llama.py: RMSNorm, rotate_half RoPE,
    causal eager softmax attention, SwiGLU), in float64 NumPy."""
    import numpy as np
    f = {k: np.asarray(v, dtype=np.float64) for k, v in w.items()}
    b, l, dm = x.shape
    hd = dm // nh

    def rms(t, g):
        return g * (t / np.sqrt(np.mean(t * t, -1, keepdims=True) + eps))
    h = rms(x, f["input_layernorm.weight"])
    q = (h @ f["q_proj.weight"].T).reshape(b, l, nh, hd).transpose(0, 2, 1, 3)
    k = (h @ f["k_proj.weight"].T).reshape(b, l, nh, hd).transpose(0, 2, 1, 3)
    v = (h @ f["v_proj.weight"].T).reshape(b, l, nh, hd).transpose(0, 2, 1, 3)
    inv = 1.0 / theta ** (np.arange(0, hd, 2) / hd)
    fr = np.outer(np.arange(l), inv)
    emb = np.concatenate([fr, fr], -1)
    cos, sin = np.cos(emb), np.sin(emb)

    def rope(t):
        t1, t2 = t[..., :hd // 2], t[..., hd // 2:]
        return t * cos + np.concatenate([-t2, t1], -1) * sin
    q, k = rope(q), rope(k)
    s = q @ k.transpose(0, 1, 3, 2) / math.sqrt(hd)
    s = np.where(np.triu(np.ones((l, l), bool), 1), -np.inf, s)
    s = np.exp(s - s.max(-1, keepdims=True))
    s = s / s.sum(-1, keepdims=True)
    o = (s @ v).transpose(0, 2, 1, 3).reshape(b, l, dm)
    x = x + o @ f["o_proj.weight"].T
    h = rms(x, f["post_attention_layernorm.weight"])
    g = h @ f["gate_proj.weight"].T
    return x + ((g / (1 + np.exp(-g))) * (h @ f["up_proj.weight"].T)) @ f["down_proj.weight"].T


def cmd_blocks(args):
    import numpy as np
    env = os.environ.get("MOJOLEARN_NUMERIC_MODE", "").lower()
    if env != args.mode:
        sys.exit(f"neural_fast_quality blocks: MOJOLEARN_NUMERIC_MODE={env!r} but --mode {args.mode}")
    sys.path.insert(0, str(ROOT / "python"))
    sys.path.insert(0, str(ROOT / "python" / "mojolearn" / "tests"))
    from mojolearn import Mamba1Block, Mamba2Block, Mamba3Block, TransformerBlock
    import test_mamba_surface as tm
    cases = []
    root = ROOT / "mamba" / "corpus"
    fams = [("mamba1", root, Mamba1Block, tm.m1_weights, "block.out"),
            ("mamba2", root / "mamba2", Mamba2Block, tm.m2_weights, "residual.out"),
            ("mamba3", root / "mamba3", Mamba3Block, tm.m3_weights_corpus, "residual.out")]
    for fam, d, cls, weights_of, stage in fams:
        for case in sorted(p for p in d.iterdir() if (p / "manifest.json").is_file()):
            m = json.loads((case / "manifest.json").read_text())
            st = m.get("stages", {}).get(stage)
            rec = dict(family=fam, case=case.name)
            if st is None or not (case / "x.f32").is_file():
                cases.append(dict(rec, skipped=f"no {stage} stage or no x.f32"))
                continue
            if "init_states" in case.name:
                # its reference starts from a planted state this zero-state
                # forward does not load
                cases.append(dict(rec, skipped="needs the case's initial state"))
                continue
            B, L, dm = m["B"], m["L"], m["d_model"]
            try:
                kw = {}
                if "dt_limit" in m:
                    kw["dt_limit"] = tuple(float(v["value"]) for v in m["dt_limit"])
                blk = cls(weights_of(str(case), dm), **kw)
                x = tm.f32(str(case / "x.f32"), (B, L, dm))
                y = np.asarray(blk.forward(x, blk.allocate_state(B)), dtype=np.float64).reshape(-1)
            except Exception as exc:  # recorded, never a pass
                cases.append(dict(rec, error=repr(exc)[:300]))
                continue
            r64 = np.fromfile(case / st["ref64"], dtype="<f8").reshape(-1)
            r32 = np.fromfile(case / st["ref32"], dtype="<f4").astype(np.float64).reshape(-1)
            scale = float(np.max(np.abs(r64))) or 1.0
            cases.append(dict(rec, err=float(np.max(np.abs(y - r64))) / scale,
                              torch32_err=float(np.max(np.abs(r32 - r64))) / scale))
    for (dm, nh, l) in ((64, 4, 16), (128, 2, 64)):
        for seed in range(args.seeds):
            rng = np.random.default_rng(7000 + seed)
            it = 2 * dm

            def lin(o, i):
                return rng.uniform(-1 / math.sqrt(i), 1 / math.sqrt(i), (o, i)).astype(np.float32)
            w = {"input_layernorm.weight": (1 + 0.1 * rng.standard_normal(dm)).astype(np.float32),
                 "post_attention_layernorm.weight": (1 + 0.1 * rng.standard_normal(dm)).astype(np.float32),
                 "q_proj.weight": lin(dm, dm), "k_proj.weight": lin(dm, dm), "v_proj.weight": lin(dm, dm),
                 "o_proj.weight": lin(dm, dm), "gate_proj.weight": lin(it, dm), "up_proj.weight": lin(it, dm),
                 "down_proj.weight": lin(dm, it)}
            x = rng.standard_normal((2, l, dm)).astype(np.float32)
            blk = TransformerBlock(w, n_heads=nh)
            y = np.asarray(blk.forward(x, blk.allocate_state(2, l)), dtype=np.float64)
            r64 = _llama64(x.astype(np.float64), w, nh)
            scale = float(np.max(np.abs(r64)))
            cases.append(dict(family="transformer", case=f"d{dm}_h{nh}_l{l}_seed{seed}",
                              err=float(np.max(np.abs(y - r64))) / scale))
    return dict(base(args, "blocks"), mode=args.mode, cases=cases)


# --------------------------------------------------------------- compare
def paired(fast, ident, lower_is_better=True):
    """(verdict, mean d, stderr d, worst relative) for FAST - IDENTICAL."""
    d = [(f - i) if lower_is_better else (i - f) for f, i in zip(fast, ident)]
    n = len(d)
    mean = sum(d) / n
    sd = math.sqrt(sum((v - mean) ** 2 for v in d) / (n - 1)) if n > 1 else 0.0
    se = sd / math.sqrt(n)
    ref = sum(abs(v) for v in ident) / n
    worst = max((v / abs(i) if i else 0.0) for v, i in zip(d, ident))
    ok = mean <= 2 * se + 1e-4 * ref and worst <= 0.01
    return ok, mean, se, worst


def cmd_compare(args):
    runs = [json.loads(p.read_text()) for p in sorted(Path(args.dir).glob("*.json"))]
    groups = {}
    for r in runs:
        if r.get("kind") in ("samba", "mlp"):
            groups.setdefault((r["kind"], r["dataset"]), {}).setdefault(r["mode"], {})[r["seed"]] = r
    lines, bad = [], 0
    lines.append("| task | dataset | seeds | IDENTICAL held-out | FAST held-out | torch held-out | "
                  "FAST-IDENT mean (se) | worst seed | verdict |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for (kind, ds), by in sorted(groups.items()):
        seeds = sorted(set(by.get("fast", {})) & set(by.get("identical", {})))
        if len(seeds) < 5:
            lines.append(f"| {kind} | {ds} | {len(seeds)} | | | | | | TOO FEW SEEDS |")
            bad += 1
            continue
        fi = [by["fast"][s]["heldout_after"] for s in seeds]
        ii = [by["identical"][s]["heldout_after"] for s in seeds]
        tt = [by["torch"][s]["heldout_after"] for s in seeds if s in by.get("torch", {})]
        ok, mean, se, worst = paired(fi, ii)
        extra = ""
        if kind == "mlp":
            fa = [by["fast"][s]["accuracy"] for s in seeds]
            ia = [by["identical"][s]["accuracy"] for s in seeds]
            ok_a, ma, sa, wa = paired(fa, ia, lower_is_better=False)
            ok = ok and ok_a
            extra = f"; acc {sum(ia)/len(ia):.4f} vs {sum(fa)/len(fa):.4f}"
        bad += 0 if ok else 1
        tm = f"{sum(tt)/len(tt):.5f}" if tt else "-"
        lines.append(f"| {kind} | {ds} | {len(seeds)} | {sum(ii)/len(ii):.5f} | {sum(fi)/len(fi):.5f} | {tm} | "
                     f"{mean:+.2e} ({se:.1e}) | {100*worst:+.3f}% | {'PASS' if ok else 'FAIL'}{extra} |")
    blocks = {r["mode"]: r for r in runs if r.get("kind") == "blocks"}
    if blocks:
        lines.append("")
        lines.append("| block case | IDENTICAL err | FAST err | torch fp32 err | verdict |")
        lines.append("|---|---|---|---|---|")
        fb = {(c["family"], c["case"]): c for c in blocks.get("fast", {}).get("cases", [])}
        for c in blocks.get("identical", {}).get("cases", []):
            k = (c["family"], c["case"])
            f = fb.get(k, {})
            if "err" not in c or "err" not in f:
                why = c.get("skipped") or c.get("error") or f.get("skipped") or f.get("error") or "missing"
                ok = "skipped" in c and "skipped" in f
                bad += 0 if ok else 1
                lines.append(f"| {k[0]}/{k[1]} | | | | {'SKIPPED' if ok else 'FAIL'}: {why} |")
                continue
            bound = max(2 * c["err"], c.get("torch32_err", 0.0), 1e-6)
            ok = f["err"] <= bound
            bad += 0 if ok else 1
            t32 = f"{c['torch32_err']:.2e}" if "torch32_err" in c else "-"
            lines.append(f"| {k[0]}/{k[1]} | {c['err']:.2e} | {f['err']:.2e} | {t32} | {'PASS' if ok else 'FAIL'} |")
    text = "\n".join(lines) + f"\n\nVERDICT: {'PASS' if bad == 0 else 'FAIL (%d)' % bad}\n"
    sys.stdout.write(text)
    return 0 if bad == 0 else 1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("samba", "samba-torch"):
        p = sub.add_parser(name)
        p.add_argument("--corpus", choices=CORPORA, required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--steps", type=int, default=300)
        p.add_argument("--batch", type=int, default=8)
        p.add_argument("--seq", type=int, default=128)
        p.add_argument("--d-model", type=int, default=64)
        p.add_argument("--layers", default="mamba3,attention")
        p.add_argument("--n-heads", type=int, default=2)
        p.add_argument("--intermediate", type=int, default=128)
        p.add_argument("--lr", type=float, default=3e-3)
        p.add_argument("--warmup", type=int, default=20)
        p.add_argument("--min-lr", type=float, default=3e-4)
        p.add_argument("--max-norm", type=float, default=1.0)
        p.add_argument("--heldout-rows", type=int, default=64)
        p.add_argument("--out", type=Path, required=True)
        if name == "samba":
            p.add_argument("--mode", choices=("fast", "identical"), required=True)
            p.add_argument("--init-out", type=Path)
            p.add_argument("--init", type=Path, help="start from these weights (an --init-out npz)")
        else:
            p.add_argument("--init", type=Path, required=True)
            p.add_argument("--device", default="cuda")
    p = sub.add_parser("mlp-data")
    p.add_argument("--out-npz", type=Path, required=True)
    for name in ("mlp", "mlp-torch"):
        p = sub.add_parser(name)
        p.add_argument("--data", type=Path, required=True)
        p.add_argument("--dataset", choices=("wine", "digits3"), required=True)
        p.add_argument("--seed", type=int, required=True)
        p.add_argument("--epochs", type=int, default=60)
        p.add_argument("--batch", type=int, default=16)
        p.add_argument("--lr", type=float, default=1e-2)
        p.add_argument("--out", type=Path, required=True)
        if name == "mlp":
            p.add_argument("--mode", choices=("fast", "identical"), required=True)
        else:
            p.add_argument("--device", default="cuda")
    p = sub.add_parser("blocks")
    p.add_argument("--mode", choices=("fast", "identical"), required=True)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--out", type=Path, required=True)
    p = sub.add_parser("compare")
    p.add_argument("dir")
    args = ap.parse_args()
    if args.cmd == "compare":
        return cmd_compare(args)
    if args.cmd == "mlp-data":
        cmd_mlp_data(args)
        return 0
    if args.out.exists():
        ap.error(f"{args.out} exists; evidence is never overwritten")
    fn = {"samba": cmd_samba, "samba-torch": cmd_samba_torch, "mlp": cmd_mlp,
          "mlp-torch": cmd_mlp_torch, "blocks": cmd_blocks}[args.cmd]
    record = fn(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(record, indent=1, allow_nan=False) + "\n"
    with args.out.open("x") as fh:
        fh.write(text)
    print(json.dumps(dict(out=str(args.out), heldout_after=record.get("heldout_after"))), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
