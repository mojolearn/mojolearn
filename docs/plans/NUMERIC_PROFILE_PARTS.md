# Numeric profile: the fixed15_v1 default and per-part configuration (lane fixed15-train, 2026-11-25)

Andrew, 2026-11-25: "YES ACROSS EVERYTHING DEFAULT. I don't give a fuck about apple", and "configurable for EACH
part", with the verifier covering it.

## 1. The default flip

- `fixed15_v1` becomes the INFERENCE default (`_numeric_profile.DEFAULT`) for the families its row lists in
  `default_for` (the transformer models: `CausalLM`, `ParallelCausalLM`), on every vendor and the CPU column.
- `fp32_v1` is the explicit opt-out: `numeric_profile="fp32_v1"`, `mojolearn.set_numeric_profile("fp32_v1")`,
  `MOJOLEARN_NUMERIC_PROFILE=fp32_v1`.
- Training stays `fp32_v1` (`TRAINING_DEFAULT`). Families `fixed15_v1` does not compute (Mamba, Samba, every
  non-neural family, a bare `TransformerBlock` named without a profile) stay `fp32_v1`.
- Decided on existing measurements (no new A/B): perplexity change +0.0006 % (enwik8) and +0.0033 % (pile_github)
  on SmolLM2-360M; time over fp32_v1 0.34 to 0.68 on NVIDIA, 0.85 to 0.89 on AMD, 3.40 to 3.65 on Apple (accepted;
  `NumericProfileSpeedWarning` still says so on Apple).
- Bits change for every no-keyword run of those models, on every column together. Verify references: RECORD OWED.

## 2. Per-part configuration

```python
CausalLM.load(path)                                              # the default preset, fixed15_v1
CausalLM.load(path, numeric_profile="fp32_v1")                   # a preset
CausalLM.load(path, numeric_profile={"preset": "fixed15_v1", "lm_head": "fp32_v1"})
CausalLM.load(path, numeric_profile={"preset": "fp32_v1", "q_proj": "fixed15_v1"})
model.numeric_profile        # the canonical form: a preset name, or a dict with only the overrides that differ
```

- A spec is a preset name, or an exact `dict` with `"preset"` (a profile name) and part overrides
  `part -> profile name`. The part's format is that profile's arithmetic for that product.
- Parts (`_numeric_profile.PARTS`, same order as `core/numeric_profile_parts.mojo`):
  `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, lm_head, attn_scores, attn_context`.
- Refused by name at the line that asked: an unknown part; a part the family does not have (`lm_head` on a bare
  `TransformerBlock`; `gate_proj` on an ungated MLP); a format the part cannot use (`attn_context` is `fp32_v1`
  only); a non-registered profile name as a preset or a part value.
- Canonical form: overrides equal to the preset's own format are dropped; with none left the canonical form is the
  preset string. The canonical form is what `model.numeric_profile` returns and what a checkpoint state stores
  (`STATE_KEY`); a state with no field is `fp32_v1` (unchanged). Reload without a keyword adopts the saved spec;
  reload naming a different spec is refused by name.
- Resolved ONCE at construction/load into the native PARTS MASK (`core/numeric_profile_parts.mojo`,
  `bindings/numeric_profile_mask_unpack.mojo::profile_parts_mask`): bit set = int15i64.v1, clear = fp32.v1.
  `fixed15_v1` = 511 (bits 0..8), `fp32_v1` = 0. Native code reads the mask, never a name.
- The training default and trainers are unchanged; a trainer named a dict is refused (no training profile).

## 3. Verify (both sides through `device=`, light fixtures)

1. The default preset (no keyword) on each transformer model.
2. One data-driven lane, one case per part: the part overridden away from the preset (fixed15 -> fp32 from the
   fixed15_v1 preset; fp32 -> fixed15 from the fp32_v1 preset where allowed; `attn_context` is a refusal case).
3. Save and reload with a per-part spec: the same digest.
4. Refusals at the boundary in pytest.
5. Sabotage arms per side: an override applied to the wrong part; an override ignored; the saved spec dropped on
   reload. Each must be caught.

## 4. Next (not in this change)

Training under fixed15 (forward and input gradient in fixed15, weight gradient fp32, a new versioned name), built on
`autograd/fixed15_host.mojo` (the host arm, written, not wired).
