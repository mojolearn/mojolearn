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
(`bench/OPPONENT_REFERENCE.md`, `CONTRIBUTING.md` (Performance claims)): "identical
mode takes X times the incumbent's time", ours over theirs, where above 1.0
ours takes longer. No other speed statement is made in this directory.

## The model

Default `HuggingFaceTB/SmolLM2-360M` (Llama architecture, ungated, about
720 MB of bf16 safetensors). Selectable: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
and `meta-llama/Llama-3.2-1B` (gated on Hugging Face; the gate is met ONCE, on
the Mac that populates the store with `HF_TOKEN`, and never on a box). The
record carries the model's `config_sha256` (canonical config.json) and
`weights_sha256` (the sorted list of safetensors file hashes), so two columns
that loaded different bytes are refused by the diff rather than read as
DIVERGENT.

### The model source: the R2 dataset store, never a download on a box

DEVIATION 2704 (`tools/stage_from_r2.sh`; Andrew, 2026-09-13: "cloudflare has
datasets already saved and when using runpod we should ALWAYS use them; make
sure all of our shit ships corpora from R2 instead of downloading"). The model
is held in the store like every corpus, pinned by size and sha256 in
`bench/results/dataset_store/manifest.tsv`, one key per checkpoint file under
`models/<name>/`, and staged onto a rented box by the runner right after the
source is unpacked (`tools/gemm_remote_leg.sh` and `tools/do_extra_leg.sh`
call `sh tools/stage_from_r2.sh "<ssh target>"` there and read
`MOJOLEARN_STAGE_KEYS`; the model-leg wrappers export
`MOJOLEARN_STAGE_KEYS=models/<name>` and `MOJOLEARN_STAGE_STRICT=1`). The
store presigns on the Mac; the box receives only a short-lived URL inside a
piped script and refuses any file whose size or sha256 differs from the pin.
No credential ever reaches a box, and no `R2_*` variable exists in this
directory. A `HOME/`-rooted key lands at `/root/<path under $HOME>` on a box,
so the group `models/<name>` (local `$HOME/models/<name>/`) is
`/root/models/<name>/` there, and that is the harness's `--model` and the
twin's model path on every column. The body REFUSES an unstaged model; the
runners' `--allow-hf-download` is the one opt-in (a Hugging Face fetch on
the box, off by default, printing a warning that names DEVIATION 2704 and
marking the record `model_source=huggingface-on-box`).

The keys, one per checkpoint file (the store's key scheme, `<group>/<file>`):

    models/SmolLM2-360M/config.json
    models/SmolLM2-360M/generation_config.json
    models/SmolLM2-360M/model.safetensors
    models/SmolLM2-360M/tokenizer.json
    models/SmolLM2-360M/tokenizer_config.json
    models/SmolLM2-360M/special_tokens_map.json
    models/TinyLlama-1.1B-Chat-v1.0/config.json
    models/TinyLlama-1.1B-Chat-v1.0/generation_config.json
    models/TinyLlama-1.1B-Chat-v1.0/model-0000N-of-0000M.safetensors   (one key per shard)
    models/TinyLlama-1.1B-Chat-v1.0/model.safetensors.index.json
    models/TinyLlama-1.1B-Chat-v1.0/tokenizer.json
    models/TinyLlama-1.1B-Chat-v1.0/tokenizer.model
    models/TinyLlama-1.1B-Chat-v1.0/tokenizer_config.json
    models/TinyLlama-1.1B-Chat-v1.0/special_tokens_map.json
    models/Llama-3.2-1B/config.json
    models/Llama-3.2-1B/generation_config.json
    models/Llama-3.2-1B/model-0000N-of-0000M.safetensors               (one key per shard)
    models/Llama-3.2-1B/model.safetensors.index.json
    models/Llama-3.2-1B/tokenizer.json
    models/Llama-3.2-1B/tokenizer_config.json
    models/Llama-3.2-1B/special_tokens_map.json

The shard names are whatever the repository ships at fetch time (a single
`model.safetensors` where there is no index); the `ls` in step 2 below is
the list, and every file listed becomes a key.

The store declares a multi-file model as a GROUP (a group's rows are carried
forward by `manifest` on a Mac that lacks the files; a catalog key missing
locally would be dropped). Patch text for `tools/dataset_store.sh`, `groups()`,
one row per model, tab separated, the third field the file names from step 2:

    models/SmolLM2-360M	HOME/models/SmolLM2-360M	config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json special_tokens_map.json
    models/TinyLlama-1.1B-Chat-v1.0	HOME/models/TinyLlama-1.1B-Chat-v1.0	config.json generation_config.json model-00001-of-0000M.safetensors ... model.safetensors.index.json tokenizer.json tokenizer.model tokenizer_config.json special_tokens_map.json
    models/Llama-3.2-1B	HOME/models/Llama-3.2-1B	config.json generation_config.json model-00001-of-0000M.safetensors ... model.safetensors.index.json tokenizer.json tokenizer_config.json special_tokens_map.json

and the same group names in the header's key list beside
`corpus/fineweb-edu-10BT`.

POPULATING THE STORE, ONCE, on the Mac (every line RUN OWED; `manifest`
hashes only catalog keys, so a group's rows are written the way the FineWeb
shards' were, from the bytes' own size and sha256):

    RUN OWED (1): python3 -m pip install --user huggingface_hub   # or a throwaway venv
    RUN OWED (2): python3 -c 'from huggingface_hub import snapshot_download; import os; snapshot_download("HuggingFaceTB/SmolLM2-360M", local_dir=os.path.expanduser("~/models/SmolLM2-360M"), allow_patterns=["*.json","*.safetensors","*.txt","*.model","tokenizer*"])' && ls -l ~/models/SmolLM2-360M
    RUN OWED (3): add the groups() row above to tools/dataset_store.sh with exactly the files (2) listed
    RUN OWED (4): (cd ~/models && for f in SmolLM2-360M/*; do printf 'models/%s\t%s\t%s\n' "$f" "$(wc -c < "$f" | tr -d ' ')" "$(shasum -a 256 "$f" | cut -d' ' -f1)"; done) >> bench/results/dataset_store/manifest.tsv && LC_ALL=C sort -o bench/results/dataset_store/manifest.tsv bench/results/dataset_store/manifest.tsv
    RUN OWED (5): sh tools/dataset_store.sh push models/SmolLM2-360M        # expands the group, one object per key
    RUN OWED (6): sh tools/dataset_store.sh verify models/SmolLM2-360M      # every shard against its pin
    RUN OWED (7): sh tools/dataset_store.sh manifest                        # rewrites the pins, carrying the group rows forward; must print the same rows
    RUN OWED (8): commit tools/dataset_store.sh and bench/results/dataset_store/manifest.tsv together

The same eight lines for `TinyLlama/TinyLlama-1.1B-Chat-v1.0` and, with
`HF_TOKEN` set for step (2) only, `meta-llama/Llama-3.2-1B`. The rows the
orchestrator adds to `manifest.tsv` in step (4) have this shape (size and
sha256 from the bytes on the Mac):

    models/SmolLM2-360M/config.json	<size>	<sha256>
    models/SmolLM2-360M/generation_config.json	<size>	<sha256>
    models/SmolLM2-360M/model.safetensors	<size>	<sha256>
    models/SmolLM2-360M/special_tokens_map.json	<size>	<sha256>
    models/SmolLM2-360M/tokenizer.json	<size>	<sha256>
    models/SmolLM2-360M/tokenizer_config.json	<size>	<sha256>

The FineWeb route is the alternative when the Mac's uplink is the cost (a
1.1B checkpoint is about 2.2 GB): a rented box fetches the repository, hashes
each file, and PUTs it to a write URL minted here with
`sh tools/dataset_store.sh presign-put models/<name>` (one URL per shard),
then the rows are added from the box's `sha256sum` output; the bytes never
pass through the Mac. Either way the model is pinned in
`bench/results/dataset_store/manifest.tsv` like every corpus before any leg
rents a box, and the wrappers refuse to rent without a `models/<name>/` row.

On the M4, `tools/model_leg/run_local_m4.sh` runs `tools/dataset_store.sh
verify` on every key of the group at `$HOME/models/<name>/` and `pull`s a
missing one; on the CPU pod, whose runner has no staging hook,
`run_leg_cpu.sh` writes the store's own `box-cmd` fetch per key, presigned on
the Mac, into the generated command file (0600, outside the checkout), which
is the `stage` mechanism spelled out.

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

Before any column: the store holds the model (the eight RUN OWED lines of
"The model source" above), and `bench/results/dataset_store/manifest.tsv`
carries its `models/<name>/` rows; every wrapper refuses to rent without them.

Apple M4 first, on the release Mac, from a clean worktree (it compiles the
four GPU bindings and four host bindings, verifies the model against the
store's pins at `$HOME/models/<name>/`, runs the Metal column, the CPU column
and the incumbent on `mps`):

    RUN OWED: sh tools/model_leg/run_local_m4.sh
    RUN OWED: python3 bench/model/diff.py --diff ~/mojolearn-evidence/model-leg/<stamp>-apple-m4/model-leg/ours.apple-m4-metal.json ~/mojolearn-evidence/model-leg/<stamp>-apple-m4/model-leg/ours.apple-m4-metal-cpu.json --require-columns 2
    RUN OWED: python3 bench/model/diff.py --ratio ~/mojolearn-evidence/model-leg/<stamp>-apple-m4/model-leg/ours.apple-m4-metal.json ~/mojolearn-evidence/model-leg/<stamp>-apple-m4/model-leg/torch.apple-m4-metal.json

NVIDIA H100 on RunPod (dry run first; `--rent` bills one hour; the runner
stages `models/<name>` from the store right after the source is unpacked,
strict; its gemm gate needs an existing Apple card):

    RUN OWED: sh tools/model_leg/run_leg.sh
    RUN OWED: MOJOLEARN_RUNPOD_KEY_FILE=$HOME/.mojolearn_runpod_key MOJOLEARN_MODEL_LEG_LOCAL_CARD=bench/results/e1g/<stamp>/local/apple.card sh tools/model_leg/run_leg.sh --rent

AMD MI325X on DigitalOcean (dry run first; `--rent` bills one hour; the same
staging by the runner):

    RUN OWED: sh tools/model_leg/run_leg_amd.sh
    RUN OWED: MOJOLEARN_DO_TOKEN_FILE=$HOME/.mojolearn_do_token sh tools/model_leg/run_leg_amd.sh --rent

The CPU column on a RunPod CPU pod (dry run first; the rate scales with
`--vcpu`; the wrapper presigns the group's keys on the Mac and the pod fetches
and verifies them before the body):

    RUN OWED: sh tools/model_leg/run_leg_cpu.sh
    RUN OWED: sh tools/model_leg/run_leg_cpu.sh --rent --vcpu 16

Then, on the Mac, with the four records copied into one dated directory:

    RUN OWED: python3 bench/model/diff.py --diff ours.apple-m4-metal.json ours.nvidia-h100-80gb-hbm3-sm_90a.json ours.amd-gfx942.json ours.cpu-<model>.json --require-columns 4
    RUN OWED: python3 bench/model/diff.py --ratio ours.nvidia-h100-80gb-hbm3-sm_90a.json torch.nvidia-h100-80gb-hbm3-sm_90a.json
    RUN OWED: python3 bench/model/diff.py --ratio ours.amd-gfx942.json torch.amd-gfx942.json
    RUN OWED: python3 bench/model/diff.py --ratio ours.cpu-<model>.json torch.cpu-<model>.json
    RUN OWED: pixi run -e test test-model-diff

Each leg's `remote/model-leg/status.tsv` names every phase with its exit code
and seconds; `gate.txt` the label, commit, the `cells=` lines and
`model_source=` when the opt-in download ran; the runner's `stage.log` the
staging summary line; `ratio.txt` the ratio output. A phase that failed is a
finding, not a reason to re-run silently; a body that read
`refused=model-not-staged` is a staging failure, and the leg is re-run only
after the store is fixed, never with `--allow-hf-download` as a shortcut.

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
