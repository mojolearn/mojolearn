#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""ONE REAL OPEN MODEL through mojolearn.models, one column, one record.

    python3 bench/model/harness.py --model <local dir> \\
        --formats float32,bfloat16,int8 --prompts bench/model/prompts.txt \\
        --max-new 64 --column nvidia-h100-sm_90a --out ours.nvidia-h100-sm_90a.json

For every weight format: load the model, greedy-generate every prompt, and
record per prompt the SHA-256 of the generated ids and of the first-step
logits bytes, the token counts, and the wall time per token for the prefill
and for the decode separately (median of three runs, one untimed warm-up
first). The record also carries the commit, the box, the device, the
numeric mode READ BACK from the binaries, the binding sha256s, and the
model's config hash. bench/model/_common.py names every field.

WHAT THIS FILE ASSUMES ABOUT mojolearn.models (lane B2, built in parallel;
each assumption is a line here, so a change there is a one-line change
here):
  CausalLM.load(path, weight_format=<fmt>, max_positions=None, device=<dev>)
  lm.forward(ids)                       ids (1, L) int32 Array; logits (1, L, V) float32 Array, C order
  lm.generate(ids, max_new_tokens=N, greedy=True)
                                        the prompt followed by the continuation, or the
                                        continuation alone (both read; the record says which)
  Tokenizer.from_pretrained(path); tok.encode(text) -> ids; tok.decode(ids) -> text
  optional read-backs, recorded when present: lm.numeric_mode, lm.weight_format,
  lm.device, lm.vocab_size; mojolearn.numeric_mode(), mojolearn._backend.vendor()
The step API (allocate_state, step) is NOT timed here: its per-step contract
(whether the first call takes the whole prompt) is not fixed, and the decode
time is the generate time minus the prefill time of the same run, which is
what a user's generate call costs. Nothing here moves a bit.

`--model` may be a Hugging Face repo id; then `huggingface_hub` fetches it
into --model-dir (default $MOJOLEARN_MODEL_DIR/<repo with / as __>, else
~/.cache/mojolearn-models/<repo>) BEFORE any clock, and the leg scripts fetch
it before this file runs anyway. The fetch is never inside a timing, and
nothing lands under bench/results.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _common as C  # noqa: E402

FORMATS = ("float32", "bfloat16", "int8")


def resolve_model(model, model_dir, no_download):
    if os.path.isdir(model):
        return os.path.abspath(model), model
    target = model_dir or os.path.join(os.environ.get("MOJOLEARN_MODEL_DIR", os.path.expanduser("~/.cache/mojolearn-models")),
                                       model.replace("/", "__"))
    if os.path.isfile(os.path.join(target, "config.json")):
        return target, model
    if no_download:
        raise SystemExit(f"REFUSING: {model} is not on disk at {target} and --no-download is set")
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        raise SystemExit("huggingface_hub is not importable here; fetch the model first "
                         "(tools/model_leg/*.sh do) and pass the directory") from None
    snapshot_download(repo_id=model, local_dir=target,
                      allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*"],
                      token=os.environ.get("HF_TOKEN") or None)
    return target, model


def read_back(lm):
    """Every mode and vendor statement the process can make about the
    binaries it loaded: the package's cross-checked numeric_mode(), the
    vendor, and each loaded `_mojolearn_*` module's own `<prefix>_numeric_mode()`
    (1 is identical, 0 fast, 2 deterministic) and the file's sha256."""
    import mojolearn
    lib = dict(mojolearn_version=getattr(mojolearn, "__version__", None), numeric_mode=None,
               vendor=None, requested_mode=os.environ.get("MOJOLEARN_NUMERIC_MODE", "identical"),
               bindings=[], model_read_back={})
    try:
        lib["numeric_mode"] = mojolearn.numeric_mode()
    except Exception as e:  # noqa: BLE001
        lib["numeric_mode_error"] = str(e)[:300]
    try:
        lib["vendor"] = mojolearn._backend.vendor()
    except Exception as e:  # noqa: BLE001
        lib["vendor_error"] = str(e)[:300]
    for attr in ("numeric_mode", "weight_format", "device", "vocab_size", "max_positions"):
        if hasattr(lm, attr):
            try:
                v = getattr(lm, attr)
                lib["model_read_back"][attr] = v() if callable(v) else v
            except Exception as e:  # noqa: BLE001
                lib["model_read_back"][attr] = f"error: {str(e)[:120]}"
    codes = {0: "fast", 1: "identical", 2: "deterministic"}
    for name, mod in sorted(sys.modules.items()):
        f = getattr(mod, "__file__", None)
        if not f or "_mojolearn" not in os.path.basename(f) or not f.endswith((".so", ".dylib", ".pyd")):
            continue
        entry = dict(module=name, file=f, sha256=C.sha256_file(f), numeric_mode=None)
        for attr in dir(mod):
            if attr.endswith("_numeric_mode") and callable(getattr(mod, attr, None)):
                try:
                    entry["numeric_mode"] = codes.get(int(getattr(mod, attr)()), "unknown")
                except Exception as e:  # noqa: BLE001
                    entry["numeric_mode"] = f"error: {str(e)[:80]}"
                break
        lib["bindings"].append(entry)
    return lib


def make_ids(ids_list):
    import mojolearn
    return mojolearn.Array.from_list([[int(t) for t in ids_list]], dtype="<i4")


def first_step_logits_bytes(logits, n_prompt):
    """The bytes of logits[0, L-1, :] for a (1, L, V) float32 C-order
    Array: the row that chooses the first generated token."""
    shape = tuple(int(s) for s in logits.shape)
    if len(shape) != 3 or shape[0] != 1:
        raise RuntimeError(f"forward returned logits of shape {shape}; expected (1, L, V)")
    L, V = shape[1], shape[2]
    if L != n_prompt:
        raise RuntimeError(f"forward returned L={L} for a prompt of {n_prompt} tokens")
    raw = C.bytes_of(logits)
    if len(raw) != L * V * 4:
        raise RuntimeError(f"logits are {len(raw)} bytes, not float32 (1, {L}, {V})")
    return raw[(L - 1) * V * 4: L * V * 4], V


def argmax_f32_row(raw):
    import struct
    n = len(raw) // 4
    vals = struct.unpack("<%df" % n, raw)
    best, bi = vals[0], 0
    for i in range(1, n):
        if vals[i] > best:
            best, bi = vals[i], i
    return bi


def run_prompt(lm, tok, text, max_new, runs, clock):
    ids = C.flatten_row(tok.encode(text))
    if not ids:
        return C.refused_cell("the tokenizer returned no ids for this prompt", 0)
    n = len(ids)
    arr = make_ids(ids)
    ids_runs, pre, dec, first_sha, first_argmax, gen_text, returns = [], [], [], None, None, None, None
    for _ in range(runs):
        p_ms, logits = clock.ms(lm.forward, arr)
        row, _v = first_step_logits_bytes(logits, n)
        sha = C.sha256_bytes(row)
        if first_sha is None:
            first_sha, first_argmax = sha, argmax_f32_row(row)
        elif sha != first_sha:
            first_sha = "MOVED:" + first_sha[:16] + "/" + sha[:16]
        g_ms, out = clock.ms(lm.generate, arr, max_new_tokens=max_new, greedy=True)
        new, returns = C.split_generated(ids, C.flatten_row(out))
        ids_runs.append(new)
        pre.append(p_ms)
        dec.append(max(g_ms - p_ms, 0.0))
        if gen_text is None:
            try:
                gen_text = tok.decode(new)
            except Exception as e:  # noqa: BLE001
                gen_text = f"<decode error: {str(e)[:80]}>"
    n_gen = len(ids_runs[0])
    extra = dict(generate_returns=returns, generate_ms_runs=[p + d for p, d in zip(pre, dec)],
                 first_token_is_logits_argmax=(bool(ids_runs[0]) and ids_runs[0][0] == first_argmax))
    return C.cell_from_runs(n, n_gen, ids_runs, first_sha, gen_text, pre, dec, extra)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local model directory, or a Hugging Face repo id")
    ap.add_argument("--model-dir", default=None, help="where a repo id is fetched to (before any clock)")
    ap.add_argument("--no-download", action="store_true")
    ap.add_argument("--formats", default="float32,bfloat16,int8")
    ap.add_argument("--prompts", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.txt"))
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--runs", type=int, default=3, help="timed runs per prompt (median reported)")
    ap.add_argument("--column", required=True, help="the column label, e.g. nvidia-h100-sm_90a")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-positions", type=int, default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    for f in formats:
        if f not in FORMATS:
            raise SystemExit(f"unknown format {f!r}; one of {FORMATS}")
    path, model_id = resolve_model(args.model, args.model_dir, args.no_download)
    C.require_model_dir(path)
    prompts = C.read_prompts(args.prompts)

    from mojolearn.models import CausalLM, Tokenizer  # the lane B2 surface
    tok = Tokenizer.from_pretrained(path)
    protocol = dict(max_new=args.max_new, greedy=True, runs=args.runs, statistic="median",
                    warm_prompt=prompts[0][0], prompts_file=os.path.abspath(args.prompts),
                    prompts_sha256=C.prompts_sha256(args.prompts), device_requested=args.device,
                    formats=formats, timing="prefill = forward over the prompt; decode = generate minus "
                    "that run's prefill; per token = whole phase over its token count")
    rec = C.new_record("ours", args.column, C.model_record(path, model_id), protocol, library={})
    rec["model"]["tokenizer"] = type(tok).__name__
    clock = C.Clock()
    for fmt in formats:
        arm = dict(prompts={}, weight_format=fmt)
        try:
            lm = CausalLM.load(path, weight_format=fmt, max_positions=args.max_positions, device=args.device)
        except Exception as e:  # noqa: BLE001
            arm["error"] = f"load refused: {str(e)[:600]}"
            for pid, _ in prompts:
                arm["prompts"][pid] = C.refused_cell(arm["error"])
            rec["arms"][fmt] = arm
            C.write_record(args.out, rec)
            print(f"# ARM {fmt} REFUSED {arm['error']}", flush=True)
            continue
        if not rec["library"]:
            rec["library"] = read_back(lm)
            C.write_record(args.out, rec)
        # the warm-up: the first prompt once, never recorded
        try:
            run_prompt(lm, tok, prompts[0][1], args.max_new, 1, clock)
        except Exception as e:  # noqa: BLE001
            arm["warmup_error"] = str(e)[:300]
        for pid, text in prompts:
            try:
                cell = run_prompt(lm, tok, text, args.max_new, args.runs, clock)
            except Exception as e:  # noqa: BLE001
                cell = C.refused_cell(e)
            arm["prompts"][pid] = cell
            v = cell["verdict"]
            shown = cell.get("ids_sha256", "")[:16] if v != "REFUSED" else cell.get("error", "")[:60]
            print(f"# CELL {fmt}/{pid} {v} {shown} prefill={cell.get('prefill_ms_per_token')} "
                  f"decode={cell.get('decode_ms_per_token')}", flush=True)
            C.write_record(args.out, rec)
        rec["arms"][fmt] = arm
        C.write_record(args.out, rec)
        if hasattr(lm, "close"):
            try:
                lm.close()
            except Exception:  # noqa: BLE001
                pass
    rec["complete"] = True
    C.write_record(args.out, rec)
    cells = [c for a in rec["arms"].values() for c in a["prompts"].values()]
    n_ref = sum(1 for c in cells if c["verdict"] == "REFUSED")
    n_mov = sum(1 for c in cells if c["verdict"] == "MOVED")
    print(f"cells={len(cells)} stable={len(cells) - n_ref - n_mov} moved={n_mov} refused={n_ref} "
          f"numeric_mode={rec['library'].get('numeric_mode')} vendor={rec['library'].get('vendor')}")
    print(f"wrote {args.out}")
    return 1 if n_mov else 0


if __name__ == "__main__":
    sys.exit(main())
