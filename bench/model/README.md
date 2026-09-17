# The model leg: one real open model, four columns, the incumbent's fast default

Lane `lane/model-leg`, 2026-09-17, written against the `mojolearn.models`
surface lane B2 is building (`CausalLM`, `Tokenizer`). Nothing in this
directory moves a bit and nothing in it has been executed: every command
below is RUN OWED, and every cell of the results table reads RUN OWED until a
record replaces it.

## The question

Does one Llama-architecture checkpoint, loaded by mojolearn in each of its
three weight formats (`python/mojolearn/lowbit.py`: `float32`, `bfloat16`,
`int8`), generate THE SAME BITS on an Apple M4 (Metal), an NVIDIA H100, an
AMD MI325X and a CPU, and what does our identical mode cost against the
incumbent's fast default on the same box?

The claim is bit equality of the generated ids and of the first-step logits
across columns, per prompt and per format, in the words of
`tools/identity_break.py --diff`: IDENTICAL xN or DIVERGENT. The timing is
one sentence per phase, the only comparison this repository reports
(`bench/OPPONENT_REFERENCE.md`, `ENGINEERING_RULES.md` section 9): "identical
mode takes X times the incumbent's time", ours over theirs, where above 1.0
ours takes longer. No other speed statement is made in this directory.

## The model

Default `HuggingFaceTB/SmolLM2-360M` (Llama architecture, ungated, about
720 MB of bf16 safetensors). Selectable: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
(ungated) and `meta-llama/Llama-3.2-1B` (gated: `HF_TOKEN` must reach the
harness, which none of the guarded runners passes to a rented box, so that
model is a local-M4 or CPU-pod run only). The record carries the model's
`config_sha256` (canonical config.json) and `weights_sha256` (the sorted list
of safetensors file hashes), so two columns that loaded different bytes are
refused by the diff rather than read as DIVERGENT.

### The model source

The leg scripts fetch the files with `huggingface_hub.snapshot_download` into
a directory OUTSIDE the checkout (`/root/model-leg-models/<repo with / as __>`
on a box, `$MOJOLEARN_EVIDENCE_ROOT/model-leg/models/<slug>` on the M4),
before any clock starts. A download is never inside a timing; the harness
refuses a path without `config.json`.

OPTIONAL, documented, not a default: `MODEL_SOURCE_R2=<bucket/prefix>` pulls
the files from a Cloudflare R2 bucket in the S3-compatible form

    AWS_ACCESS_KEY_ID=$R2_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY=$R2_SECRET_ACCESS_KEY \
      aws s3 cp --recursive s3://<bucket/prefix> <model dir> --endpoint-url $R2_ENDPOINT

when `R2_ENDPOINT`, `R2_ACCESS_KEY_ID` and `R2_SECRET_ACCESS_KEY` are set.
The owner has not said what the bucket holds. Two facts bound it: the RunPod
runner passes no environment to the body and the DigitalOcean runner passes
only `MOJOLEARN_*`/`MODULAR_*` names, so on those legs the variables never
reach the box and the Hugging Face fetch runs; and this repository's rule is
that credentials never reach a rented box (`tools/dataset_store.sh` mints
presigned URLs instead). So the option is for the M4 and for a box the
operator holds a shell on. `R2_ENDPOINT` is the full endpoint
(`https://<account id>.r2.cloudflarestorage.com`, the form
`tools/dataset_store.sh` derives from `R2_ACCOUNT_ID`).

## The prompts

`bench/model/prompts.txt`, one prompt per line as `<id><TAB><text>`: `p01`
to `p16` are English, 8 to 64 tokens under a Llama-style tokenizer; `a01` to
`a04` are adversarial (a single period; one token repeated forty times; a
paragraph of about 300 words with a question; non-ASCII text with CJK,
accented Latin, emoji and punctuation marks). The file's SHA-256 is in every
record (`protocol.prompts_sha256`) and the diff refuses two records that were
not handed the same file. Never edit a prompt; add one under a new id.

## The protocol

For each column and each weight format:

1. load; read back `mojolearn.numeric_mode()` and `mojolearn._backend.vendor()`
   and every loaded `_mojolearn_*` binding's own `<prefix>_numeric_mode()` and
   file sha256 (a column whose read-back is not `identical` is named by the
   diff and is not an identical column);
2. warm up: the first prompt generated once, untimed;
3. every prompt, three runs each: `forward` over the prompt (the prefill,
   timed alone; the last row of its logits, as float32 bytes, is the
   first-step logits hash), then `generate(max_new_tokens=64, greedy=True)`
   (timed; the decode time is this run's generate time minus its prefill
   time). Per token means the whole phase over its token count. The record
   keeps every run and reports the median and the range. Three runs that do
   not hash the same ids read MOVED, the column disagreeing with itself.
4. the incumbent, same box, same prompts, same greedy decode, through
   `transformers` + PyTorch: the fast default as shipped (bf16 weights and
   activations on a GPU with every switch left where torch ships it and read
   back; float32 on a CPU), then a second arm under
   `torch.use_deterministic_algorithms(True)`, TF32 off on both switches,
   `float32_matmul_precision("highest")` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
   each arm in its own process. `torch` and `transformers` versions, CUDA or
   HIP, the device name and the SDPA switches are in the record.

Timing is wall time by `perf_counter`, device-synchronized around the torch
calls; mojolearn's calls return after the bytes are on the host.

## The records

One JSON per column per model, schema `mojolearn.model_leg.v1`
(`bench/model/_common.py` names every field): `column`, `commit`, `box`
(hostname, cpu_model, gpu), `model` (id, config_sha256, weights_sha256),
`protocol`, `library` (ours: numeric_mode and vendor read back, binding
sha256s; torch: the pins and switches), and `arms.<format>.prompts.<id>` with
`ids_sha256`, `ids_sha256_runs`, `first_logits_sha256`, `n_prompt_tokens`,
`n_generated`, `prefill_ms_per_token` and `decode_ms_per_token` (medians),
their ranges, every run, the decoded text (for reading, never for the
verdict) and the verdict STABLE, MOVED or REFUSED. `complete` is false on a
record a lease cut.

The records live OUTSIDE the checkout, under
`$MOJOLEARN_EVIDENCE_ROOT/model-leg/<stamp>-<box>/` (default
`~/mojolearn-evidence`), as every leg's raw output does. What comes into the
repository is a dated directory under `bench/results/model/` holding the JSON
records (about 100 KB each), the diff output and a README in the shape of
`bench/results/identity_break/2026-09-14_136-lanes/README.md`: a column table
(column, box, how, cells) and the diff summary.

## The verdicts

    python3 bench/model/diff.py --diff a.json b.json c.json d.json --require-columns 4

Per format and per prompt: IDENTICAL xN when every column's generated-id
hash and first-step-logits hash agree, DIVERGENT otherwise (and which of the
two differs is named), MOVED when a column disagreed with itself, ONE-COLUMN
with one hash, REFUSED with none. `summary:` counts them, one
`summary (<format>):` line per format, and `--require-columns N` prints
REQUIRE FAIL for a cell resting on fewer than N real hashes, which is not a
pass. Exit 1 on DIVERGENT, MOVED or REQUIRE FAIL. A torch record is refused
by `--diff`: the incumbent is never compared for bits.

    python3 bench/model/diff.py --ratio ours.<label>.json torch.<label>.json

prints, per format and per incumbent arm, for the prefill and for the decode,

    identical mode (float32) takes X.XX times the incumbent's time (torch-bf16-shipped) for prefill on <box>: median over N prompts, range A to B

over the prompts both records hashed STABLE, plus the incumbent's own
determinism arm against its fast arm in the same words ("times its fast
default's time"), so the record shows what that setting costs it.
`bench/model/tests/test_diff.py` holds the wording to this (RUN OWED:
`pixi run -e test test-model-diff` once the pixi task below is added).

## The incumbent, named as bench/OPPONENT_REFERENCE.md names opponents

| arm | what it is | where |
|---|---|---|
| `torch-bf16-shipped` | transformers `AutoModelForCausalLM`, `torch_dtype=bfloat16`, `generate(do_sample=False, num_beams=1)`, every torch switch at its shipped value and read back (`library.switches`), the SDPA backend transformers chose | NVIDIA H100 (torch 2.4.1+cu124, the RunPod image's, the other torch rows' pin), AMD MI325X (torch 2.6.0+rocm6.4.1, the pin `tools/torch_lm_step_opponent_leg.sh` installed on that droplet) |
| `torch-bf16-mps-shipped` | the same on torch's `mps` device | Apple M4 |
| `torch-fp32-cpu-shipped` | the same in float32 on the CPU (torch's CPU wheel) | the CPU pod |
| `torch-deterministic` | the same dtype under `use_deterministic_algorithms(True)`, TF32 off, `highest`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`; an op with no deterministic kernel reads REFUSED with its message | every box |

`transformers` is installed unpinned on the FIRST leg and the version it
resolved is the pin from then on (`MOJOLEARN_MODEL_LEG_TRANSFORMERS_PIN`),
recorded in `library.transformers`; the opponent table rule applies: a row
is measured once per (GPU model, driver, torch, transformers, model) and
never re-run to refresh it.

## The commands, in order (every one RUN OWED; none was executed by this lane)

Apple M4 first, on the release Mac, from a clean worktree (it compiles the
four GPU bindings and four host bindings, fetches the model, runs the Metal
column, the CPU column and the incumbent on `mps`):

    RUN OWED: sh tools/model_leg/run_local_m4.sh
    RUN OWED: python3 bench/model/diff.py --diff ~/mojolearn-evidence/model-leg/<stamp>-apple-m4/model-leg/ours.apple-m4-metal.json ~/mojolearn-evidence/model-leg/<stamp>-apple-m4/model-leg/ours.apple-m4-metal-cpu.json --require-columns 2
    RUN OWED: python3 bench/model/diff.py --ratio ~/mojolearn-evidence/model-leg/<stamp>-apple-m4/model-leg/ours.apple-m4-metal.json ~/mojolearn-evidence/model-leg/<stamp>-apple-m4/model-leg/torch.apple-m4-metal.json

NVIDIA H100 on RunPod (dry run first; `--rent` bills one hour; the runner's
gemm gate needs an existing Apple card):

    RUN OWED: sh tools/model_leg/run_leg.sh
    RUN OWED: MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_MODEL_LEG_LOCAL_CARD=bench/results/e1g/<stamp>/local/apple.card sh tools/model_leg/run_leg.sh --rent

AMD MI325X on DigitalOcean (dry run first; `--rent` bills one hour):

    RUN OWED: sh tools/model_leg/run_leg_amd.sh
    RUN OWED: MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token sh tools/model_leg/run_leg_amd.sh --rent

The CPU column on a RunPod CPU pod (dry run first; the rate scales with `--vcpu`):

    RUN OWED: sh tools/model_leg/run_leg_cpu.sh
    RUN OWED: sh tools/model_leg/run_leg_cpu.sh --rent --vcpu 16

Then, on the Mac, with the four records copied into one dated directory:

    RUN OWED: python3 bench/model/diff.py --diff ours.apple-m4-metal.json ours.nvidia-h100-80gb-hbm3-sm_90a.json ours.amd-gfx942.json ours.cpu-<model>.json --require-columns 4
    RUN OWED: python3 bench/model/diff.py --ratio ours.nvidia-h100-80gb-hbm3-sm_90a.json torch.nvidia-h100-80gb-hbm3-sm_90a.json
    RUN OWED: python3 bench/model/diff.py --ratio ours.amd-gfx942.json torch.amd-gfx942.json
    RUN OWED: python3 bench/model/diff.py --ratio ours.cpu-<model>.json torch.cpu-<model>.json
    RUN OWED: pixi run -e test test-model-diff

Each leg's `remote/model-leg/status.tsv` names every phase with its exit code
and seconds; `gate.txt` the label, commit and the `cells=` lines; `ratio.txt`
the ratio output. A phase that failed is a finding, not a reason to re-run
silently.

## What the harness assumes about `mojolearn.models`

    CausalLM.load(path, weight_format="float32" | "bfloat16" | "int8", max_positions=None, device="auto" | "cpu")
    lm.forward(ids)      ids a (1, L) int32 mojolearn.Array; returns (1, L, vocab) float32 Array, C order
    lm.generate(ids, max_new_tokens=64, greedy=True)   the prompt followed by the continuation, or the continuation alone
    Tokenizer.from_pretrained(path); tok.encode(text) -> ids; tok.decode(ids) -> text

Read when present: `lm.numeric_mode`, `lm.weight_format`, `lm.device`,
`lm.vocab_size`, `lm.close()`. The step API (`allocate_state`, `step`) is
not timed: its per-step contract is not fixed and the decode time is what a
user's `generate` costs. The CPU column loads the package from a copy that
carries no GPU set with `MOJOLEARN_HOST_DIR` naming the host bindings
(`bench/results/identity_break/2026-09-17_lowbit-m4/README.md`'s recipe), and
passes `device="cpu"`.

## Results

Every cell RUN OWED. A cell is filled only from a record in
`bench/results/model/<date>/`, never from a log line.

### Bits: `--diff` over the four columns, `--require-columns 4`

| format | prompts | apple-m4-metal | apple-m4-cpu | nvidia-h100-sm_90a | amd-mi325x-gfx942 | cpu (RunPod) | verdict |
|---|---|---|---|---|---|---|---|
| float32 | p01 to p16 | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED |
| float32 | a01 to a04 | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED |
| bfloat16 | p01 to p16 | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED |
| bfloat16 | a01 to a04 | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED |
| int8 | p01 to p16 | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED |
| int8 | a01 to a04 | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED | RUN OWED |

### Time: identical mode against the incumbent's fast default, same box, ours over theirs

| box | format | phase | identical mode takes X times the incumbent's time (median, range) | incumbent arm | incumbent's own determinism arm takes X times its fast default's time |
|---|---|---|---|---|---|
| Apple M4 | float32 | prefill | RUN OWED | torch-bf16-mps-shipped | RUN OWED |
| Apple M4 | float32 | decode | RUN OWED | torch-bf16-mps-shipped | RUN OWED |
| Apple M4 | bfloat16 | prefill | RUN OWED | torch-bf16-mps-shipped | RUN OWED |
| Apple M4 | bfloat16 | decode | RUN OWED | torch-bf16-mps-shipped | RUN OWED |
| Apple M4 | int8 | prefill | RUN OWED | torch-bf16-mps-shipped | RUN OWED |
| Apple M4 | int8 | decode | RUN OWED | torch-bf16-mps-shipped | RUN OWED |
| NVIDIA H100 | float32 | prefill | RUN OWED | torch-bf16-shipped | RUN OWED |
| NVIDIA H100 | float32 | decode | RUN OWED | torch-bf16-shipped | RUN OWED |
| NVIDIA H100 | bfloat16 | prefill | RUN OWED | torch-bf16-shipped | RUN OWED |
| NVIDIA H100 | bfloat16 | decode | RUN OWED | torch-bf16-shipped | RUN OWED |
| NVIDIA H100 | int8 | prefill | RUN OWED | torch-bf16-shipped | RUN OWED |
| NVIDIA H100 | int8 | decode | RUN OWED | torch-bf16-shipped | RUN OWED |
| AMD MI325X | float32 | prefill | RUN OWED | torch-bf16-shipped | RUN OWED |
| AMD MI325X | float32 | decode | RUN OWED | torch-bf16-shipped | RUN OWED |
| AMD MI325X | bfloat16 | prefill | RUN OWED | torch-bf16-shipped | RUN OWED |
| AMD MI325X | bfloat16 | decode | RUN OWED | torch-bf16-shipped | RUN OWED |
| AMD MI325X | int8 | prefill | RUN OWED | torch-bf16-shipped | RUN OWED |
| AMD MI325X | int8 | decode | RUN OWED | torch-bf16-shipped | RUN OWED |
| CPU (RunPod) | float32 | prefill | RUN OWED | torch-fp32-cpu-shipped | RUN OWED |
| CPU (RunPod) | float32 | decode | RUN OWED | torch-fp32-cpu-shipped | RUN OWED |
| CPU (RunPod) | bfloat16 | prefill | RUN OWED | torch-fp32-cpu-shipped | RUN OWED |
| CPU (RunPod) | bfloat16 | decode | RUN OWED | torch-fp32-cpu-shipped | RUN OWED |
| CPU (RunPod) | int8 | prefill | RUN OWED | torch-fp32-cpu-shipped | RUN OWED |
| CPU (RunPod) | int8 | decode | RUN OWED | torch-fp32-cpu-shipped | RUN OWED |

Pins per box (torch, transformers, driver, CUDA or HIP): RUN OWED, from
`library` of each torch record.

## What this record is not

Not a quality claim about the model, not a claim that a bf16 or int8 column
equals the float32 column (they are different weights by construction,
`python/mojolearn/lowbit.py`), and not a throughput claim of any kind beyond
the one sentence per phase above. The 60-minute lease may cut a column short
(`complete` false, the cells it has are kept and the missing ones are absent,
not clean); `MOJOLEARN_MODEL_LEG_FORMATS` and `MOJOLEARN_MODEL_LEG_RUNS`
narrow a second leg to what is owed.
