#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Held-out perplexity of SmolLM2-360M under each candidate arithmetic.

Lane lane/lowbit-quality, 2026-09-29. Runs on the pod, never on the laptop.

    python3 bench/lowbit_quality/infer_eval.py --model /root/models/SmolLM2-360M \\
        --corpus <enwik8 input.txt> --out <dir> --arms a,floor,b,c,d,e,f,c+attn,d+attn,e+attn,f+attn

THE EVALUATION SET. Bytes of `corpus/enwik8/input.txt` (R2 dataset store,
sha256 pinned in bench/results/dataset_store/manifest.tsv) from the first
byte after the first newline at or after `--tail-from` (default 99,000,000)
to the last newline of the file, decoded as strict UTF-8, tokenized by the
model's own `tokenizer.json` with no special token added. The first
`--windows` times `--length` ids are cut into windows that do not overlap.
In each window every position but the first is scored, each from the ids
before it in the same window. The record states the byte range, the sha256
of those bytes, the id count and the sha256 of the ids used.

THE METRIC. Perplexity `exp(mean nll)` over the scored positions, the nll
computed in float64 from the float32 logits. Per arm: the relative change
against arm `a` on the same positions, and the fraction of scored positions
where the arm's top-1 id equals the baseline's. The interval is 1.96
standard errors of the per-window mean differences.

THE ARMS are in `ARMS` below. A name that carries a profile name follows
the contract's rule with nothing substituted. A follow-up arm has its own
name and its own row.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402

from arith import DROPPED_NOTE, Spec  # noqa: E402
from llama_sim import LlamaSim, load_config, load_weights  # noqa: E402

FP = ("fp32", "fp32")
SWEEP_WIDTHS = (8, 10, 12, 15)


def pieces(bits):
    """int8 pieces a code of `bits` bits is carried in: 8 bits is one piece,
    anything up to 15 bits is two."""
    return 1 if bits <= 8 else 2


def _arms():
    arms = {}

    def add(key, spec, letter):
        spec.letter = letter
        arms[key] = spec

    add("a", Spec("fp32.v1", note="baseline: float32 operands, torch float32 accumulation"), "a")
    add("floor", Spec("fp32-acc64", acc64=True,
                      note="float32 operands, every product accumulated in float64 and rounded once: "
                           "how far the ORDER of a float32 accumulation can move the metric"), "floor")
    add("b", Spec("bf16f32.v1", w="bf16", note="bf16 weights, fp32 activations, fp32 accumulation"), "b")
    for key, name, w, a, note in (
            ("c", "bf16-both", "bf16", "bf16", "bf16 both operands, fp32 accumulation"),
            ("d", "int8i32.v1", "int8", "int8", "contract L-3 to L-7 on both operands"),
            ("e", "int15-both", "int15", "int15", "15-bit codes on both operands, same scale rule"),
            ("f", "int15w-int8a", "int15", "int8", "15-bit weight codes, int8 activation codes")):
        add(key, Spec(name, w=w, a=a, note=note), key)
        add(key + "+attn", Spec(name + "+attn", w=w, a=a, attn=True,
                                note=note + "; attention QK and PV products replaced too"), key)
    # ---- THE TWO FINALISTS (orchestrator, 2026-09-29 03:35Z), each ONE COMPLETE CONFIGURATION:
    # every projection and every attention product replaced at once. Errors interact, so a
    # pass of the projections alone and of the attention alone does not establish these.
    add("F1", Spec("F1-int15-all", w="int15", a="int15", attn=True,
                   note="FINALIST F1: 15-bit codes on every projection and on every attention product (QK, PV)"), "F1")
    add("F2", Spec("F2-int15proj-int8attn", w="int15", a="int15", attn=True,
                   overrides={"attn_qk": ("int8", "int8"), "attn_pv": ("int8", "int8")},
                   note="FINALIST F2: 15-bit codes on every projection, int8 codes (int8i32.v1's rule) on "
                        "every attention product (QK, PV)"), "F2")
    # ---- follow-up arms: their own names, their own rows
    add("int8w-fp32a", Spec("int8w-fp32a", w="int8",
                            note="int8 weight codes materialized (what weight_format='int8' ships), fp32 activations"), "-")
    add("fp32w-int8a", Spec("fp32w-int8a", a="int8", note="fp32 weights, int8 activation codes materialized"), "-")
    add("int15w-fp32a", Spec("int15w-fp32a", w="int15", note="15-bit weight codes materialized, fp32 activations"), "-")
    add("fp32w-int15a", Spec("fp32w-int15a", a="int15", note="fp32 weights, 15-bit activation codes materialized"), "-")
    add("d-fp32head", Spec("int8i32.v1-fp32head", w="int8", a="int8", overrides={"lm_head": FP},
                           note="int8 both operands, the LM head product kept fp32"), "-")
    add("d-fp32mlp", Spec("int8i32.v1-fp32mlp", w="int8", a="int8",
                          overrides={"gate_proj": FP, "up_proj": FP, "down_proj": FP},
                          note="int8 both operands, the three SwiGLU products of every block kept fp32"), "-")
    add("d-fp32down", Spec("int8i32.v1-fp32down", w="int8", a="int8", overrides={"down_proj": FP},
                           note="int8 both operands, down_proj of every block kept fp32"), "-")
    add("d-fp32attnproj", Spec("int8i32.v1-fp32attnproj", w="int8", a="int8",
                               overrides={"q_proj": FP, "k_proj": FP, "v_proj": FP, "o_proj": FP},
                               note="int8 both operands, the four attention projections of every block kept fp32"), "-")
    add("d-int15down", Spec("int8i32.v1-int15down", w="int8", a="int8",
                            overrides={"down_proj": ("int15", "int15")},
                            note="int8 both operands, down_proj of every block on 15-bit codes"), "-")
    add("d-int15head-down", Spec("int8i32.v1-int15head-down", w="int8", a="int8",
                                 overrides={"down_proj": ("int15", "int15"), "lm_head": ("int15", "int15")},
                                 note="int8 both operands, down_proj and the LM head on 15-bit codes"), "-")
    add("f-int15down-a", Spec("int15w-int8a-int15down", w="int15", a="int8",
                              overrides={"down_proj": ("int15", "int15")},
                              note="15-bit weights, int8 activations, but down_proj's activation on 15-bit codes"), "-")
    add("e-qk", Spec("int15-both+qk", w="int15", a="int15", overrides={"attn_qk": ("int15", "int15")},
                     note="15-bit both operands, QK replaced, PV kept fp32"), "-")
    add("e-pv", Spec("int15-both+pv", w="int15", a="int15", overrides={"attn_pv": ("int15", "int15")},
                     note="15-bit both operands, PV replaced, QK kept fp32"), "-")
    add("int8w-int15a", Spec("int8w-int15a", w="int8", a="int15",
                             note="int8 weight codes, 15-bit activation codes (more integer pieces on the activation only)"), "-")
    add("int8w-int12a", Spec("int8w-int12a", w="int8", a="int12",
                             note="int8 weight codes, 12-bit activation codes, same scale rule"), "-")
    add("int8w-int10a", Spec("int8w-int10a", w="int8", a="int10",
                             note="int8 weight codes, 10-bit activation codes, same scale rule"), "-")
    add("int8w-int8s1a", Spec("int8w-int8s1a", w="int8", a="int8s1",
                              note="int8 both; the ACTIVATION scale saturates: row absmax to [128, 256), clamp at 127"), "-")
    add("int8w-int8s2a", Spec("int8w-int8s2a", w="int8", a="int8s2",
                              note="int8 both; the ACTIVATION scale saturates: row absmax to [256, 512), clamp at 127"), "-")
    add("int8m-both", Spec("int8m-both", w="int8m", a="int8m",
                           note="int8 both, a finer scale that is NOT a power of two: row absmax maps to exactly 127"), "-")
    # ---- the WEIGHT side, with the activation held at 15 bits (which alone costs nothing measurable)
    add("int10w-int15a", Spec("int10w-int15a", w="int10", a="int15",
                              note="10-bit weight codes, 15-bit activation codes, same scale rule"), "-")
    add("int12w-int15a", Spec("int12w-int15a", w="int12", a="int15",
                              note="12-bit weight codes, 15-bit activation codes, same scale rule"), "-")
    add("int8mw-int15a", Spec("int8mw-int15a", w="int8m", a="int15",
                              note="int8 weight codes on the finer scale that is NOT a power of two "
                                   "(row absmax maps to exactly 127), 15-bit activation codes"), "-")
    add("int8s1w-int15a", Spec("int8s1w-int15a", w="int8s1", a="int15",
                               note="int8 weight codes on the saturating scale (row absmax to [128, 256), "
                                    "clamp at 127), 15-bit activation codes"), "-")
    add("int8w-int15a-fp32head", Spec("int8w-int15a-fp32head", w="int8", a="int15", overrides={"lm_head": FP},
                                      note="int8 weights, 15-bit activations, the LM head product kept fp32"), "-")
    add("int8w-int15a-int15head", Spec("int8w-int15a-int15head", w="int8", a="int15",
                                       overrides={"lm_head": ("int15", "int15")},
                                       note="int8 weights, 15-bit activations, the LM head (the tied table) on 15-bit codes"), "-")
    add("int8w-int15a-int15mlp", Spec("int8w-int15a-int15mlp", w="int8", a="int15",
                                      overrides={k: ("int15", "int15") for k in ("gate_proj", "up_proj", "down_proj")},
                                      note="int8 weights on the four attention projections and the head, "
                                           "15-bit weights on the three SwiGLU products, 15-bit activations"), "-")
    add("int8w-int15a-int15attnproj", Spec("int8w-int15a-int15attnproj", w="int8", a="int15",
                                           overrides={k: ("int15", "int15") for k in ("q_proj", "k_proj", "v_proj", "o_proj")},
                                           note="int8 weights on the SwiGLU products and the head, 15-bit weights "
                                                "on the four attention projections, 15-bit activations"), "-")
    # ---- THE WIDTH SWEEP (orchestrator, 2026-09-29): weight width by activation width, the
    # contract's scale rule at every width, projections only. A width of 8 bits is one int8
    # piece; any width up to 15 bits is two. Cells that are also a lettered arm repeat it.
    for wb in SWEEP_WIDTHS:
        for ab in SWEEP_WIDTHS:
            add("sweep-w%d-a%d" % (wb, ab),
                Spec("int%dw-int%da" % (wb, ab), w="int%d" % wb, a="int%d" % ab,
                     note="width sweep: %d-bit weight codes, %d-bit activation codes, same scale rule" % (wb, ab)), "-")
    add("d-qk", Spec("int8i32.v1+qk", w="int8", a="int8", overrides={"attn_qk": ("int8", "int8")},
                     note="int8 both operands, QK replaced, PV kept fp32"), "-")
    add("d-pv", Spec("int8i32.v1+pv", w="int8", a="int8", overrides={"attn_pv": ("int8", "int8")},
                     note="int8 both operands, PV replaced, QK kept fp32"), "-")
    add("attn-only-int8", Spec("fp32+attn-int8", overrides={"attn_qk": ("int8", "int8"), "attn_pv": ("int8", "int8")},
                               note="every projection fp32, QK and PV on int8 codes"), "-")
    add("attn-only-int15", Spec("fp32+attn-int15",
                                overrides={"attn_qk": ("int15", "int15"), "attn_pv": ("int15", "int15")},
                                note="every projection fp32, QK and PV on 15-bit codes"), "-")
    return arms


ARMS = _arms()


def sha256(b):
    return hashlib.sha256(b).hexdigest()


def source_sha256():
    """The sha256 of the source files this record was computed by, read at
    run time, so a record names its code even when the tree on the box was
    synced after the job that wrote it had started."""
    here = os.path.dirname(os.path.abspath(__file__))
    return {name: hashlib.sha256(open(os.path.join(here, name), "rb").read()).hexdigest()
            for name in ('arith.py', 'llama_sim.py', 'infer_eval.py')}


def evaluation_ids(corpus, tokenizer_path, tail_from, windows, length, corpus_key="corpus/enwik8/input.txt"):
    from tokenizers import Tokenizer
    with open(corpus, "rb") as fh:
        raw = fh.read()
    nl = raw.find(b"\n", tail_from)
    if nl < 0:
        raise ValueError(f"no newline at or after byte {tail_from}")
    start = nl + 1
    end = raw.rfind(b"\n") + 1
    piece = raw[start:end]
    # One character per invalid byte, so a character offset maps back to a
    # byte offset exactly; the tokenizer is handed U+FFFD in its place.
    exact = piece.decode("utf-8", errors="surrogateescape")
    invalid = sum(1 for ch in exact if 0xDC80 <= ord(ch) <= 0xDCFF)
    text = exact if invalid == 0 else "".join(
        "\ufffd" if 0xDC80 <= ord(ch) <= 0xDCFF else ch for ch in exact)
    tok = Tokenizer.from_file(tokenizer_path)
    enc = tok.encode(text, add_special_tokens=False)
    ids = enc.ids
    need = windows * length
    if len(ids) < need:
        raise ValueError(f"the tail tokenizes to {len(ids)} ids; {need} were asked for")
    used = ids[:need]
    offsets = enc.offsets
    # the character span the used ids cover, then its byte length
    last_char = offsets[need - 1][1]
    used_bytes = len(exact[:last_char].encode("utf-8", errors="surrogateescape"))
    invalid_used = sum(1 for ch in exact[:last_char] if 0xDC80 <= ord(ch) <= 0xDCFF)
    ids_bytes = b"".join(int(t).to_bytes(4, "little", signed=True) for t in used)
    record = dict(
        corpus_key=corpus_key, corpus_bytes=len(raw), corpus_sha256=sha256(raw),
        decoding="UTF-8; an invalid byte is one U+FFFD to the tokenizer",
        invalid_bytes_in_tail=invalid, invalid_bytes_in_used=invalid_used,
        tail_from=tail_from, byte_start=start, byte_end=end, tail_bytes=len(piece),
        tail_sha256=sha256(piece), tail_ids=len(ids),
        used_byte_start=start, used_byte_end=start + used_bytes,
        used_bytes_sha256=sha256(raw[start:start + used_bytes]),
        windows=windows, length=length, ids_used=need, ids_sha256=sha256(ids_bytes),
        scored_positions=windows * (length - 1),
        tokenizer_sha256=sha256(open(tokenizer_path, "rb").read()),
        rule="non-overlapping windows; every position but the first of a window is scored "
             "from the ids before it in the same window; no special token added",
    )
    return torch.tensor(used, dtype=torch.int64).reshape(windows, length), record


def score(logits, ids):
    """Per scored position: float64 nll and the top-1 id. `logits [B, L, V]`."""
    bsz, length, vocab = logits.shape
    rows = logits[:, :-1].reshape(-1, vocab)
    target = ids[:, 1:].reshape(-1)
    nll, top = [], []
    for lo in range(0, rows.shape[0], 1024):
        part = rows[lo:lo + 1024].to(torch.float64)
        logp = torch.log_softmax(part, dim=-1)
        nll.append(-logp.gather(1, target[lo:lo + 1024, None]).squeeze(1))
        top.append(part.argmax(-1))
    return torch.cat(nll), torch.cat(top), bool(torch.isfinite(rows).all().item())


def run_arm(weights, cfg, spec, ids, device, batch, diag_batches=1, rope_theta=None, limit=None):
    sim = LlamaSim(weights, cfg, spec, device, rope_theta=rope_theta)
    nll, top, finite = [], [], True
    n = ids.shape[0] if limit is None else min(limit, ids.shape[0])
    diag = {}
    t0 = time.time()
    for bi, lo in enumerate(range(0, n, batch)):
        chunk = ids[lo:min(lo + batch, n)].to(device)
        sim.diag = diag if bi < diag_batches else None
        logits = sim.forward(chunk)
        a, b, ok = score(logits, chunk)
        nll.append(a.cpu())
        top.append(b.cpu())
        finite = finite and ok
        del logits
    if device != "cpu":
        torch.cuda.synchronize()
    seconds = time.time() - t0
    sim.release()
    del sim
    if device != "cpu":
        torch.cuda.empty_cache()
    rel = {k: math.sqrt(v[0] / v[1]) if v[1] > 0 else 0.0 for k, v in diag.items()}
    return torch.cat(nll), torch.cat(top), finite, rel, seconds


INTERVAL = ("the per-position difference of nll (arm minus baseline) is averaged within each window; the "
            "interval is the mean over the windows plus and minus 1.96 standard errors of those window means "
            "(sample standard deviation over the windows divided by the root of their number), a normal "
            "approximation at the 95 percent level, mapped through exp(x) - 1. The windows are the unit that "
            "is resampled. It bounds the SAMPLING ERROR ON THIS TEXT and says nothing about other text or "
            "about tasks.")


def compare(nll, top, base_nll, base_top, length):
    per = length - 1
    d = (nll - base_nll).reshape(-1, per).mean(1)
    mean = d.mean().item()
    se = (d.std(unbiased=True) / math.sqrt(d.numel())).item() if d.numel() > 1 else float("nan")
    return dict(
        mean_nll=nll.mean().item(), perplexity=math.exp(nll.mean().item()),
        delta_nll=mean, delta_nll_se=se,
        rel_ppl_change=math.expm1(mean),
        rel_ppl_change_lo=math.expm1(mean - 1.96 * se), rel_ppl_change_hi=math.expm1(mean + 1.96 * se),
        top1_agreement=(top == base_top).double().mean().item(),
        top1_agreement_is="the share of scored positions, each with the TRUE context supplied, where the "
                          "arm's top token equals the baseline's; not a rate of changed tokens in generated "
                          "text, which diverges from the first changed token on",
        interval=INTERVAL,
    )


def summarize_diag(rel):
    """Relative Frobenius error of each product against the float64 product
    of THE SAME INPUTS, grouped by product family; first batch only."""
    groups = {}
    for name, value in rel.items():
        family = name.split(".")[-1]
        groups.setdefault(family, []).append(value)
    return {k: dict(mean=sum(v) / len(v), max=max(v), n=len(v)) for k, v in sorted(groups.items())}


def validate_hf(model_dir, weights, cfg, ids, device, out):
    """This forward against transformers' LlamaForCausalLM, float32, on the
    first batch, with an arm that must fail (RoPE theta 10000 in place of
    the config's 100000)."""
    import transformers
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.float32).to(device).eval()
    chunk = ids[:4].to(device)
    with torch.no_grad():
        ref = model(chunk).logits.to(torch.float32)
    ref_nll, ref_top, _ = score(ref, chunk)
    del model

    def against(rope_theta):
        sim = LlamaSim(weights, cfg, ARMS["a"], device, rope_theta=rope_theta)
        mine = sim.forward(chunk)
        nll, top, _ = score(mine, chunk)
        return dict(max_abs_logit_diff=(mine - ref).abs().max().item(),
                    delta_mean_nll=(nll.mean() - ref_nll.mean()).item(),
                    top1_agreement=(top == ref_top).double().mean().item())

    gate = dict(max_abs_delta_mean_nll=1e-5, min_top1_agreement=0.999)

    def passes(r):
        return abs(r["delta_mean_nll"]) <= gate["max_abs_delta_mean_nll"] and \
            r["top1_agreement"] >= gate["min_top1_agreement"]

    real, sabotage = against(None), against(10000.0)
    rec = dict(transformers=transformers.__version__, windows=int(chunk.shape[0]), gate=gate,
               forward=real, forward_passes=passes(real),
               sabotage_rope_theta_10000=sabotage, sabotage_fails=not passes(sabotage))
    rec["verdict"] = "PASS" if rec["forward_passes"] and rec["sabotage_fails"] else "FAIL"
    with open(os.path.join(out, "validate_hf.json"), "w") as fh:
        json.dump(rec, fh, indent=1)
    print("validate_hf", json.dumps(rec), flush=True)
    torch.cuda.empty_cache()
    return rec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpus-key", default="corpus/enwik8/input.txt",
                    help="the key of --corpus in the R2 dataset store, for the record")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="a,floor,b,c,d,e,f,c+attn,d+attn,e+attn,f+attn")
    ap.add_argument("--tail-from", type=int, default=99_000_000)
    ap.add_argument("--windows", type=int, default=200)
    ap.add_argument("--length", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit-windows", type=int, default=None, help="smoke only; the record says so")
    ap.add_argument("--validate-hf", action="store_true")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--commit", default="unknown")
    args = ap.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    names = [a for a in args.arms.split(",") if a]
    for a in names:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; one of {sorted(ARMS)}")
    if names[0] != "a":
        names.insert(0, "a")
    # Andrew, 2026-09-29: int8 and the int8-attention mix are not offered by the flag. Their
    # measurement is still finished: nothing is skipped here, the record carries the note.
    skipped = []

    cfg = load_config(args.model)
    ids, evalset = evaluation_ids(args.corpus, os.path.join(args.model, "tokenizer.json"),
                                  args.tail_from, args.windows, args.length, corpus_key=args.corpus_key)
    print("evaluation set", json.dumps(evalset), flush=True)
    weights, dtypes = load_weights(args.model, args.device)
    record = dict(
        schema="mojolearn.lowbit_quality.inference.v2", commit=args.commit,
        stamp_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        model=dict(key="models/SmolLM2-360M", path=args.model,
                   weights_sha256=sha256(open(os.path.join(args.model, "model.safetensors"), "rb").read()),
                   stored_dtypes=dtypes, config=cfg),
        source_sha256=source_sha256(),
        skipped_arms={a: DROPPED_NOTE for a in skipped},
        evaluation_set=evalset, limit_windows=args.limit_windows, batch=args.batch,
        library=dict(torch=torch.__version__, cuda=torch.version.cuda,
                     device_name=torch.cuda.get_device_name(0) if args.device == "cuda" else None,
                     matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                     float32_matmul_precision=torch.get_float32_matmul_precision()),
        threshold=dict(rel_ppl_change_max=0.01, rel_ppl_change_hi_max=0.01,
                       rule="the change and the upper end of its interval both under 1 percent, on two texts"),
        arms={},
    )
    if args.validate_hf:
        record["validate_hf"] = validate_hf(args.model, weights, cfg, ids, args.device, args.out)

    base_nll = base_top = None
    for key in names:
        spec = ARMS[key]
        nll, top, finite, rel, seconds = run_arm(weights, cfg, spec, ids, args.device, args.batch,
                                                 limit=args.limit_windows)
        if key == "a":
            base_nll, base_top = nll, top
        cell = compare(nll, top, base_nll, base_top, args.length)
        cell.update(spec=spec.describe(), letter=spec.letter, finite_logits=finite,
                    product_relative_error=summarize_diag(rel), seconds=seconds,
                    scored_positions=int(nll.numel()))
        if key != "a":
            cell["bit_equal_nll_to_baseline"] = bool(torch.equal(nll, base_nll))
        record["arms"][key] = cell
        torch.save(dict(nll=nll, top=top), os.path.join(args.out, f"arm_{key.replace('+', '_')}.pt"))
        print("arm", key, json.dumps({k: cell[k] for k in (
            "perplexity", "rel_ppl_change", "rel_ppl_change_lo", "rel_ppl_change_hi",
            "top1_agreement", "finite_logits", "seconds")}), flush=True)
        print("   product error", json.dumps(cell["product_relative_error"]), flush=True)
        with open(os.path.join(args.out, "inference.json"), "w") as fh:
            json.dump(record, fh, indent=1)

    floor = record["arms"].get("floor")
    for key, cell in record["arms"].items():
        if key in ("a", "floor"):
            continue
        change = cell["rel_ppl_change"]
        # A CANDIDATE NEEDS MARGIN (orchestrator, 2026-09-29): the change AND the
        # upper end of its interval are both under 1 percent. This is the verdict
        # on THIS text; a profile passes only when it passes on both texts.
        cell["verdict"] = "PASS" if (cell["finite_logits"] and change < 0.01
                                     and cell["rel_ppl_change_hi"] < 0.01) else "MISS"
        if floor is not None:
            cell["numerical_floor"] = abs(floor["rel_ppl_change"])
            cell["inside_numerical_floor"] = abs(change) <= abs(floor["rel_ppl_change"])
    with open(os.path.join(args.out, "inference.json"), "w") as fh:
        json.dump(record, fh, indent=1)
    print("DONE", os.path.join(args.out, "inference.json"), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
