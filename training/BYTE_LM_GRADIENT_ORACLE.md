# Independent two-block gradient and optimizer gate

`tools/byte_lm_gradient_oracle.py` is authored source, not an executed qualification. The author ran no tests, builds, models, measurements or provisioning. Root alone executes it on an authorized remote NVIDIA/AMD device, under the corresponding serial guard. CPU and Apple model execution are refused by the tool.

The mathematical model is exactly `mojolearn.byte-lm.b2-l32-d32-h4-kv2-ff64-v256-blocks2.fp32.v1`: embedding → two independent Llama decoder blocks → untied head → mean causal next-byte cross entropy over 64 targets. It includes RoPE, grouped-query attention, causal softmax, RMSNorm, SiLU gating and both residual paths. No final RMSNorm, bias, dropout or carried cache is added. Every tensor is computed in PyTorch FP64 on the remote GPU. No native model/oracle/arithmetic module is imported.

## Reusable functions

```python
loss64, gradients64 = reference(initial_params, ids)
result = evaluate_capture(arrays, config, reference_sink=new_npz_path)
```

`reference` accepts finite flat FP32 parameters with 34,944 cells and int32 IDs of shape `(66,)` or `(2,33)`. It returns one Python scalar loss and a dictionary of 20 shaped FP64 gradient arrays in the registry exported by `registry()`.

`evaluate_capture` expects exactly these one-dimensional arrays:

| Array keys | dtype | Cells |
|---|---|---:|
| `initial_p`, `initial_m`, `initial_v`, `grad`, `post_p`, `post_m`, `post_v` | float32 | 34944 each |
| `loss` | float32 | 1 |
| `ids` | int32 | 66 |
| `initial_flags`, `post_flags` | int32, exactly 0 or 1 | 20 each |

All state must be finite and second moments nonnegative. Config contains `profile`, `numeric_mode` (`identical`), `vendor` (`cuda` or `hip`), `completed_steps` (before the step), `post_completed_steps` (exactly one larger), and `optimizer`. The optimizer dictionary must explicitly contain `kind=2`, `lr`, `beta1`, `beta2`, `eps`, `weight_decay`, `momentum=0`, `dampening=0`, `nesterov=false`, `max_norm=0`. Scalars are normalized to their actual native FP32 values before the FP64 optimizer calculation, and those normalized values are retained. Source/fixture/run provenance remains the root harness's responsibility.

The independent AdamW reference consumes the **actual captured pre-update gradients and pre-state**, isolating optimizer mathematics from model-gradient errors. It computes decoupled weight decay, updated first/second moments, bias correction at `completed_steps+1`, and epsilon outside the square root. It compares every parameter and moment cell separately by named tensor. It also requires flags to remain unchanged and at least one parameter bit to move.

## Preset admission thresholds and controls

Every cell must satisfy `abs(actual-reference) <= atol + rtol*abs(reference)`. These provisional criteria were selected before execution; no observed failure may be silently used to enlarge them.

| Quantity | atol | rtol |
|---|---:|---:|
| All 20 model-gradient tensors | 2e-6 | 2e-4 |
| Mean loss | 2e-6 | 2e-6 |
| Updated parameters | 2e-6 | 2e-5 |
| Updated first moments | 2e-7 | 2e-5 |
| Updated second moments | 2e-9 | 2e-5 |

The sign control negates captured native gradients and must fail the same gradient gate. Two separate nonlinear controls replace SiLU's backward in block 0 or block 1 by detaching its sigmoid factor. Each preserves forward values, must preserve the reference loss under the declared loss threshold, and must fail its altered block's own `w_gate` gradient comparison. Thus a forward-only or insensitive fixture cannot qualify the gate. These are independent-reference/comparator sensitivity controls, not native sabotage builds. They do not establish exhaustive correctness for all possible inputs, optimizer settings or model sizes.

`evaluate_capture` returns a JSON-safe structured verdict including all named comparisons, first failing flat indices, error statistics, controls, exact input-array hashes and oracle-source hash. An optional new NPZ path retains the raw FP64 gradients, both nonlinear-control gradient sets, reference loss and all updated-state arrays. Passing this gate means **tolerance correctness for one supplied step**, not external bitwise equality, training progress or language quality. Bitwise cross-vendor comparisons remain separate.

## Optional file adapter

`load_capture(directory)` reads `capture.json` and exact bounded raw files. The manifest schema is:

```json
{
  "schema": "mojolearn.byte-lm.gradient-capture.v1",
  "registry": "replace with registry() result",
  "config": "replace with the config dictionary described above",
  "arrays": {
    "initial_p": {"file": "initial_p.f32", "count": 34944, "sha256": "actual SHA-256"}
  }
}
```

The example is schematic: all array keys in the table are mandatory, `registry` is the full list, and `config` is an object. Each filename is exactly `<key>.f32` or `<key>.i32`; bytes are little-endian. SHA-256 binds the actual bytes. The manifest is bounded to one MiB, and every array read is bounded to its exact expected size plus one refusal byte.

Root-only remote NVIDIA command, after the native capture exists:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python3 tools/nvidia_serial_guard.py --seconds 300 --rss-gib 12 -- \
  python3 tools/byte_lm_gradient_oracle.py /artifacts/byte-lm-capture \
  --expected-vendor cuda --output /artifacts/byte-lm-oracle-new.json
```

On AMD use `tools/amd_serial_guard.py`, `--expected-vendor hip`, and fresh artifact paths. Outputs refuse existing paths. Root must retain command, logs, source/binary/runtime provenance and complete successful guard exit evidence; a JSON written before teardown is insufficient admission on its own. No Apple command is authorized.
