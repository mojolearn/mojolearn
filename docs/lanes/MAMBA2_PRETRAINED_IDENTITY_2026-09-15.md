# Mamba-2 pretrained identity, 2026-09-15

A real public pretrained language model, `state-spaces/mamba2-130m`, run through
mojolearn's GPU `Mamba2Block` in IDENTICAL mode on an Apple M4 (Metal) and on a
rented NVIDIA H100 (CUDA, DigitalOcean). The experiment asks one question. Are
the logits and the greedy tokens bit-identical across the two vendors?

Owner approved. Nothing below has run yet.

Files

- `tools/mamba2_pretrained_identity.py` is the harness, the comparison and the optional torch sanity check.
- `tools/mamba2_pretrained_leg.sh` is the H100 leg (a Mac launcher plus the on-droplet body).
- This brief.

## 1. What it claims, if both runs agree

For the pinned checkpoint, the pinned tokenizer, the default prompt
`The capital of France is` (ids `510 5347 273 6181 310`) and 16 greedy tokens,
every one of these is byte-identical between the M4 and the H100.

- The output of each of the 24 `Mamba2Block.forward` calls, at every step.
- The final hidden state, the `norm_f` output and the logits at every prompt position.
- The last-position logits at every generation step and the chosen token.

Batch 1, sequence lengths 5 through 20, profile
`mojolearn.identical.mamba2.fp32.v1` (`mamba/IDENTICAL_MAMBA2_CONTRACT.md`).

## 2. What it does not claim

- Not the decode path. Every step re-runs the whole sequence from a zero state, so each step is the same prefill contract. The carried-state decode gates (contract section 8(d)) are a different claim.
- Not other prompts, other lengths, batch sizes above 1, or the multi-chunk path. Every length here is under 256, so every forward is ONE padded chunk.
- Not AMD, and not any other NVIDIA or Apple part.
- Not agreement with PyTorch or mamba_ssm. The torch phase is a sanity check of the wiring, and a nonzero logit difference there is expected.
- Not model quality, not speed, not float16 behavior. The published file stores float16 and is widened to float32 before anything runs (section 4).
- Not a FAST-mode contrast. The Mamba binding is identical-only (`bindings/build_mamba.sh:124-139`, `python/mojolearn/_backend.py` `_IDENTICAL_ONLY`). `--mode fast` therefore records the library's refusal with exit 3. That refusal is a record and is never a pass.
- The tokenizer is not certified. It only chooses the prompt ids, which both runs record and `--compare` requires to be equal. It gives the known GPT-NeoX ids (`Hello world` is `12092 1533`), and the torch phase checks parity with `tokenizers` when that package imports.

## 3. The three checks the owner asked for

**(1) eps. MATCHES.** The checkpoint's `config.json` names no `norm_epsilon`, so
mamba_ssm's default holds.

- mamba_ssm uses 1e-5 for both norms: `mixer_seq_simple.py:36` and `:134` for `norm_epsilon`, `ops/triton/layer_norm.py:957` for `RMSNorm`, and `modules/mamba2.py:144` for the gated norm (`eps=1e-5`). HF agrees at `configuration_mamba2.py:68`.
- mojolearn uses one kernel for both norms. `mamba/checks/mamba_fixture.mojo:46` sets `RMS_EPS = 1e-5` (bits `0x3727C5AC`). It is used at `mamba/impl/modeling/modeling_mamba.mojo:787`, and `mamba_rms_norm` is called for the block norm at `mamba/impl/modules/mamba2.mojo:901` and for the gated norm in S21 (`:1175` onward).
- The spelling also matches mamba_ssm's `rms_norm_ref` (`layer_norm.py:120-121`), which is `1/sqrt(mean + eps)` then `(x * rstd) * weight`.
- The harness refuses to run if any of those source lines changes (`SOURCE_FACTS`).

**(2) residual. MATCHES, with no host residual add.**

- In the fused path, mamba_ssm's `Block.forward` (`block.py:57-67`) computes `x = hidden + residual` inside `layer_norm_fn` (`layer_norm.py:118-119`). It then runs the mixer on `norm(x)` and returns `(mixer_out, x)`.
- `MixerModel.forward` (`mixer_seq_simple.py:196-217`) sends the last pair through the same fused add into `norm_f`.
- `Mamba2Block` returns `x + out_proj` (S22, `mamba2.mojo:1244`, HF `modeling_mamba2.py:630`).
- So block i's output is exactly the residual mamba_ssm carries into block i+1, and block 24's output is exactly `norm_f`'s input.
- Addition commutes bit for bit, and `residual_in_fp32` changes nothing when everything is float32.

**(3) DEVIATIONS 2712 and 2713. They affected this experiment's forward, and HEAD is fixed.**

- `SUPPORT_MATRIX.md:134-158` records the defect. `m2_ydiag_kernel` read X_d rows `T..Q-1` past the allocation.
- That read happens on every forward whose length is not a multiple of 256, which is every forward here.
- It was hidden when the memory held finite garbage (0 times finite is 0). It produced NaN on a warm Metal process and a fault on the MI325X.
- The fix bounds the read at `mamba/impl/modules/ssd_minimal.mojo:486`. The fix commit 82510c470 and its follow-up b6e117ea8 are both ancestors of this branch's HEAD (checked with `git merge-base --is-ancestor`).
- The harness requires that guard line in the source it imports. It also refuses a `_mojolearn_mamba.so` older than `ssd_minimal.mojo`, `mamba2.mojo`, `modeling_mamba.mojo`, `checks/numerics.mojo` or the binding source.

## 4. What the harness does, and why nothing on the host moves a bit

- **Weights.** `pytorch_model.bin` is read without torch: a zip, plus an unpickler that resolves only `OrderedDict`, `_rebuild_tensor_v2` and storage types, so no code in the archive runs.
  - The file stores **195 of its 219 tensors as HalfStorage (float16)** and 24 as FloatStorage, measured on the local copy.
  - Widening binary16 to binary32 is exact, and every tensor is round-trip checked.
  - Only Float and Half storages are admitted.
  - The 219 names and shapes, finiteness, and the byte-equality of the tied `lm_head` are checked.
  - `config.json` must equal the profile exactly: `ssm_cfg` is only `{"layer": "Mamba2"}`, so every Mamba2 default in `mamba2.py:41-59` applies.
- **Pins.** The checkpoint, config and tokenizer are pinned by sha256 and HF revision. `MAMBA2_130M_PYTORCH_MODEL_SHA256` is left empty for the owner to fill, and every run refuses until it is. The local HF cache names the file's blob `f786aff9...6fc501`, which is the git-LFS sha256; confirm it before pinning.
- **GPU.** 24 `Mamba2Block(weights, numeric_mode="identical")`, each built once. `forward(x)` gets no state. The binding's own read-back of vendor and mode is recorded, along with the `.so` path and sha256.
- **Host.** `norm_f` and the tied head use elementwise float32 numpy ufuncs only.
  - Sums are `np.cumsum(..., axis=-1)[..., -1]`. Each partial depends on the previous one, so the order is the index order.
  - No `np.sum`, `np.mean`, `@`, `np.dot`, `einsum` or BLAS.
  - Each arithmetic step is its own ufunc call, so there is nothing to contract into an FMA.
  - Every host stage is scanned for subnormal values. If any appears, the run refuses to claim identity, because an FTZ/DAZ flag set by some runtime could change exactly those bits.
  - Greedy is `np.argmax` over the first 50277 logits, and the first index wins a tie.
- **Generation.** Full re-prefill of the growing sequence each step. No EOS stop, so both runs make the same number of steps.
- **Provenance.**
  - Platform, Python and numpy versions.
  - The commit (`git rev-parse`, or `MOJOLEARN_GATE_COMMIT` on the box).
  - The same `*.mojo` source sha256 recipe `tools/do_extra_leg.sh` uses.
  - The sha256 of the harness, `_mamba_impl.py` and `_buffer.py`.
  - A device hint from `nvidia-smi` or `sysctl`, labeled as not coming from mojolearn.

## 5. Build `_mojolearn_mamba` for Metal on the Mac

Run this plugged in, with the lid open, and with no other GPU job or `mojo build` running.

```sh
cd /Users/andrewhendel/CascadeProjects/mojolearn-llm

# 1. Clear the content-addressed Mojo cache ONLY if no other build is running.
#    Its key ignores MACOSX_DEPLOYMENT_TARGET, so one poisoned build serves
#    empty 134-byte metallibs to later builds (bindings/build.sh:64-85).
MH=$(pixi run sh -c 'printf %s "$MODULAR_HOME"')
[ -n "$MH" ] && [ -d "$MH/cache/.mojo_cache" ] && rm -rf "$MH/cache/.mojo_cache"

# 2. The build. MACOSX_DEPLOYMENT_TARGET must be UNSET (set at all, it
#    suppresses ahead-of-time Metal compilation). The script already pins
#    --target-cpu apple-m1, passes no --target-accelerator on macOS and
#    compiles -D MOJOLEARN_NUMERIC_IDENTICAL=1 into python/mojolearn/identical/.
env -u MACOSX_DEPLOYMENT_TARGET -u MOJOLEARN_MAMBA_DEFINES -u MOJOLEARN_MAMBA_OUTDIR \
    -u MOJOLEARN_GPU_ARCHS -u MOJOLEARN_BUILD_EXTRA_DEFINES \
    MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2 \
    nice -n 19 sh bindings/build_mamba.sh

# 3. Count the Metal kernels yourself. build_mamba.sh exports
#    MOJOLEARN_SKIP_BUILD_GATE=1 on every run (line 141), so its own AIR floor
#    (mamba >= 16) and smoke never execute. Zero here means a poisoned cache or
#    a deployment target in the environment.
strings -a python/mojolearn/identical/_mojolearn_mamba.so \
  | grep -oE '[0-9A-Za-z_]+_[0-9a-f]{16}air' \
  | sed -e 's/^_gpu_shared_mem//' -e 's/^0//' | sort -u | grep -c '^mamba'
```

The Mamba forward path calls no base-binding helper, so only this binding
should be needed. If `import mojolearn` complains that the base binding is
missing, build it the same way with
`env -u MACOSX_DEPLOYMENT_TARGET MOJOLEARN_NUMERIC_MODE=identical MOJOLEARN_COMPILE_JOBS=2 MOJOLEARN_SKIP_BUILD_GATE=1 nice -n 19 sh bindings/build.sh`.
The leg does exactly this fallback on the box.

## 6. Run the harness on the Mac

First fill `MAMBA2_130M_PYTORCH_MODEL_SHA256`. The model and tokenizer default to the local HF snapshots.

```sh
cd /Users/andrewhendel/CascadeProjects/mojolearn-llm
STAMP=$(date -u +%Y-%m-%d_%H%M%S)
OUT=/Users/andrewhendel/CascadeProjects/mojolearn-evidence/mamba2-pretrained/$STAMP-apple-m4
nice -n 19 pixi run python tools/mamba2_pretrained_identity.py \
    --mode identical --save-logits --out $OUT/identical.json
nice -n 19 pixi run python tools/mamba2_pretrained_identity.py \
    --mode identical --out $OUT/identical_repeat.json
pixi run python tools/mamba2_pretrained_identity.py --compare $OUT/identical.json $OUT/identical_repeat.json
```

Use `pixi run python` rather than a system Python, so numpy comes from the
same `pixi.lock` as on the box. Commit first. Otherwise the recorded commit is
not the source that ran, although `--compare` still checks the Mojo source,
harness and Python-layer hashes directly.

## 7. Run the H100 leg

```sh
cd /Users/andrewhendel/CascadeProjects/mojolearn-llm
# after committing tools/mamba2_pretrained_identity.py and this leg:
sh tools/mamba2_pretrained_leg.sh dry-run      # rents nothing
nohup sh tools/mamba2_pretrained_leg.sh rent > /tmp/mamba2_pretrained_leg.log 2>&1 &
```

The launcher hands the rental to `tools/do_extra_leg.sh nv --minutes 60 --skip-gates`. That runner provides the following, and the leg script's header cites each item by line.

- `gpu-h100x1-80gb` in `nyc2`, with the repository's ssh key fingerprint.
- The token file `~/.mojolearn_do_token`, passed by path and never printed.
- The one-GPU-droplet lock.
- A local dead-man armed before the create.
- An EXIT trap that deletes the droplet and waits for a GET 404.
- An on-droplet dead-man. Before any work starts it is verified three ways: its process is alive, the droplet id is baked in, and a token GET returns 200.

The source ships as a sha256-checked `git archive` of the commit, about 10 MB. The box downloads the model and tokenizer from huggingface.co and checks each against the harness pins. It then builds the binding for the H100's architecture and runs the phases listed in the leg header:
1. identical
2. a repeat run
3. a repeat compare
4. the fast refusal record
5. the torch sanity check in its own venv

Evidence comes home to
`/Users/andrewhendel/CascadeProjects/mojolearn-evidence/mamba2-pretrained/<stamp>-nvidia-h100/`.
The body's files are under `remote/mamba2-pretrained/`.

`launcher_verdict.txt` re-reads the teardown and dead-man records. It exits
nonzero if the destroy was not confirmed by a 404.

## 8. Compare

```sh
pixi run python tools/mamba2_pretrained_identity.py --compare \
    <mac>/identical.json <h100>/remote/mamba2-pretrained/identical.json
```

`--compare` first refuses (exit 2) unless both runs are identical-mode, status ok and subnormal-free, and share all of these:
- checkpoint sha256
- prompt ids
- token count
- harness sha256
- Mojo source sha256
- `_mamba_impl.py` and `_buffer.py` hashes

It then reports each item below as IDENTICAL or DIVERGENT.
- the embedding input
- each of the 24 layers
- the final hidden state
- `norm_f`
- the prompt logits
- every step's layers, logits and token

It names the first divergent layer, and it says whether the pair is cross-vendor or a same-vendor repeat. Exit 0 only if everything is identical.

A divergence is attributable by construction.
- A differing layer hash with an identical embedding names the first block whose GPU output moved.
- Identical layer hashes with a differing `norm_f` or logits hash puts the difference on the host, which the subnormal scan and the ufunc-only spelling are there to exclude.
- If step k's token differs, every later step compares different sequences and is not compared.
