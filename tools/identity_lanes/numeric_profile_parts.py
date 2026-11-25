# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
# THE NUMERIC PROFILE PARTS LANES (lane fixed15-parts-verify, Andrew 2026-11-25; the interface is
# docs/plans/NUMERIC_PROFILE_PARTS.md). `fixed15_v1` is the inference default of the transformer models on every
# vendor and `fp32_v1` the opt-out; each product of a transformer layer (a PART) takes its own format. These lanes
# verify that through `device=` on both sides, on the hf-causal-lm lane's synthetic Llama checkpoint (untied, so it
# has an `lm_head`) and the `transformer` lane's block shape.
#
#   numeric-profile-parts             ONE data-driven lane: the default (no keyword), the two presets, then one CASE
#                                     per part (_NPP_CASES, built from _NPP_PARTS: a part added to the table is a case
#                                     with no code): the part overridden away from the preset (fixed15 -> fp32 from
#                                     the fixed15_v1 preset; fp32 -> fixed15 from the fp32_v1 preset), `attn_context`
#                                     as a refusal case, an override equal to the preset's own format dropped from the
#                                     canonical form, a CausalLM checkpoint saved with its per-part spec and reloaded
#                                     naming nothing, and a TransformerBlock with a per-part spec pickled and reloaded.
#                                     A part is `<case>` (the logits' digest); `flags` holds every in-cell check (the
#                                     canonical form read back, each override moving the logits away from both
#                                     presets, the reload equal to the original, the refusals by name) as a hashed
#                                     flag, so a fault reads MOVED, never a raise.
#   numeric-profile-parallel-default  ParallelCausalLM (no device=, GPU only: one-sided) under the default and under
#                                     a per-part spec, each held to the plain CausalLM of the same spec.
#
# SABOTAGE (tools/identity_lanes/coverage_sabotage.tsv, block "fixed15-parts"): which case catches which arm.
#   MOJOLEARN_NP_PARTS_SABOTAGE_WRONG_PART_HOST / _DEVICE (the q_proj bit applied to k_proj and vice versa, one
#     side): parts f15-q_proj, f15-k_proj, f32-q_proj, f32-k_proj MOVE on that side (each case computes the other
#     projection's override) and stay on the other side.
#   MOJOLEARN_NP_PARTS_SABOTAGE_IGNORE_HOST / _DEVICE (overrides ignored, any mask with a bit set becomes 511, one
#     side): every f15-<part> case equals the fixed15_v1 preset and every f32-<part> case equals it too, so their
#     "differs from the preset" / "differs from the other preset" flags flip: `flags` and every f32-<part> part MOVE
#     on that side.
#   MOJOLEARN_NUMERIC_PROFILE_SABOTAGE_DROP_SAVED (the profile door's saved-state reader keeps only the preset of a
#     saved dict; door code serves both sides): case save-reload adopts "fixed15_v1" instead of the saved spec, so
#     its part and the reload flags (adopted == saved, reloaded profile, logits and digest_ equal) MOVE on both sides.
#
# Every reference is OWED (python/mojolearn/host_surface.py PUBLIC_PENDING_LANES: "no reference"): the release's one
# verification records them. A fragment of tools/identity_break.py (loaded by `_load_lane_fragments`); np, os,
# tempfile, _h, _fit, _ids, _seq, _block_weights, _hf_refused, _one_sided, lane and HF_LM_* are that namespace's.

#: The parts, in the order of core/numeric_profile_parts.mojo and `_numeric_profile.PARTS`; the lane checks the two
#: agree (a flag), so a part added to the package without a case here moves the cell.
_NPP_PARTS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head",
              "attn_scores", "attn_context")
#: Parts whose only format is fp32_v1 (plan section 2): naming fixed15_v1 for one is refused by name.
_NPP_FP32_ONLY = ("attn_context",)

#: THE CASES, as data. A case is {"name", "spec", "canonical", "base", "refused"}: `spec` is what the lane hands
#: `numeric_profile=`; `canonical` what `model.numeric_profile` must return; `base` the preset whose logits the case
#: must equal ("same") or differ from ("differs": and from the other preset too); `refused` the word the refusal must
#: name (the case then loads nothing).
_NPP_CASES = tuple(
    [dict(name="f15-" + p, spec={"preset": "fixed15_v1", p: "fp32_v1"},
          canonical={"preset": "fixed15_v1", p: "fp32_v1"}, base="fixed15_v1", relation="differs", refused=None)
     for p in _NPP_PARTS if p not in _NPP_FP32_ONLY]
    + [dict(name="f32-" + p, spec={"preset": "fp32_v1", p: "fixed15_v1"},
            canonical={"preset": "fp32_v1", p: "fixed15_v1"}, base="fp32_v1", relation="differs", refused=None)
       for p in _NPP_PARTS if p not in _NPP_FP32_ONLY]
    + [dict(name=pre + "-" + p + "-refused", spec={"preset": pre, p: "fixed15_v1"}, canonical=None, base=None,
            relation=None, refused=p)
       for p in _NPP_FP32_ONLY for pre in ("fp32_v1", "fixed15_v1")]
    # an override equal to the preset's own format is dropped: the canonical form is the preset string
    + [dict(name="f15-" + p + "-same", spec={"preset": "fixed15_v1", p: "fp32_v1"}, canonical="fixed15_v1",
            base="fixed15_v1", relation="same", refused=None)
       for p in _NPP_FP32_ONLY]
    + [dict(name="f32-q_proj-same", spec={"preset": "fp32_v1", "q_proj": "fp32_v1"}, canonical="fp32_v1",
            base="fp32_v1", relation="same", refused=None)])

#: The save/reload case's spec: two overrides, so a reader that keeps one of them still moves the cell.
_NPP_SAVED_SPEC = {"preset": "fixed15_v1", "lm_head": "fp32_v1", "attn_scores": "fp32_v1"}
#: The pickled block's spec (a TransformerBlock has no lm_head).
_NPP_BLOCK_SPEC = {"preset": "fp32_v1", "attn_scores": "fixed15_v1", "down_proj": "fixed15_v1"}


def _npp_checkpoint(d):
    """The hf-causal-lm lane's untied Llama fixture written under `d`; its root."""
    from mojolearn import _causal_lm_fixtures as fx
    cfg, tensors = fx.family_fixture("llama", False)
    return fx._write_checkpoint(os.path.join(d, "llama"), cfg, tensors)


@lane("numeric-profile-parts")
def _(ml, X, yc, yr, Xh=None):
    """The fixed15_v1 default and the per-part numeric profile of `models.CausalLM` and `TransformerBlock`, on this
    side's device. See the fragment's header for the cases and the sabotage arms each one catches."""
    import json
    import pickle
    from mojolearn import _numeric_profile as npm
    A = ml.Array.from_buffer
    CLM = ml.models.causal_lm.CausalLM
    ids = _ids(X, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)
    parts, flags = {}, []
    with tempfile.TemporaryDirectory(prefix="ib-numeric-profile-parts-") as d:
        root = _npp_checkpoint(d)

        def run(spec):
            m = CLM.load(root) if spec is None else CLM.load(root, numeric_profile=spec)
            return m, np.asarray(m.forward(A(ids)))

        default_m, default_y = run(None)
        presets = {"fixed15_v1": run("fixed15_v1")[1], "fp32_v1": run("fp32_v1")[1]}
        parts["default"] = _h(default_y)
        parts["fixed15_v1"] = _h(presets["fixed15_v1"])
        parts["fp32_v1"] = _h(presets["fp32_v1"])
        flags += [tuple(npm.PARTS) == _NPP_PARTS,
                  default_m.numeric_profile == "fixed15_v1",
                  default_y.tobytes() == presets["fixed15_v1"].tobytes(),
                  presets["fixed15_v1"].tobytes() != presets["fp32_v1"].tobytes()]
        for case in _NPP_CASES:  # driver-loop: over the cases (data)
            spec = case["spec"]
            if case["refused"] is not None:
                flags.append(_hf_refused(lambda: CLM.load(root, numeric_profile=spec), case["refused"]))
                continue
            m, y = run(spec)
            base = presets[case["base"]]
            other = presets["fp32_v1" if case["base"] == "fixed15_v1" else "fixed15_v1"]
            flags.append(m.numeric_profile == case["canonical"])
            if case["relation"] == "same":
                flags.append(y.tobytes() == base.tobytes())
            else:
                parts[case["name"]] = _h(y)
                flags += [y.tobytes() != base.tobytes(), y.tobytes() != other.tobytes()]

        # SAVE AND RELOAD with a per-part spec: the weights and the profile field a checkpoint writer stores
        # (`state_field`), written to files and read back; the reader names nothing and adopts the saved spec.
        m, y = run(_NPP_SAVED_SPEC)
        weights = {str(k): np.ascontiguousarray(np.asarray(v)) for k, v in m.state_dict().items()}
        np.savez(os.path.join(d, "weights.npz"), **weights)
        with open(os.path.join(d, "profile.json"), "w", encoding="utf-8") as fh:
            json.dump(npm.state_field(m.numeric_profile), fh, sort_keys=True)
        with open(os.path.join(d, "profile.json"), encoding="utf-8") as fh:
            saved = json.load(fh)
        with np.load(os.path.join(d, "weights.npz"), allow_pickle=False) as z:
            loaded = {k: A(np.ascontiguousarray(z[k])) for k in z.files}
        adopted = npm.adopt_saved(saved)
        again = CLM.load(root, numeric_profile=adopted)
        again.load_state_dict(loaded)
        y2 = np.asarray(again.forward(A(ids)))
        parts["save-reload"] = _h(y2)
        flags += [saved.get(npm.STATE_KEY) == _NPP_SAVED_SPEC,
                  adopted == _NPP_SAVED_SPEC,
                  again.numeric_profile == m.numeric_profile == _NPP_SAVED_SPEC,
                  y2.tobytes() == y.tobytes(),
                  again.digest_ == m.digest_,
                  _hf_refused(lambda: npm.adopt_saved(saved, "fp32_v1"), "fp32_v1")]

    # A TransformerBlock with a per-part spec, pickled and reloaded; lm_head is not a block part.
    dm, nh, nkv, hd, it = 32, 2, 1, 16, 64
    w = _block_weights("numeric-profile-parts", {
        "input_layernorm.weight": (dm,), "post_attention_layernorm.weight": (dm,),
        "q_proj.weight": (nh * hd, dm), "k_proj.weight": (nkv * hd, dm), "v_proj.weight": (nkv * hd, dm),
        "o_proj.weight": (dm, nh * hd), "gate_proj.weight": (it, dm), "up_proj.weight": (it, dm),
        "down_proj.weight": (dm, it)},
        near_one=("input_layernorm.weight", "post_attention_layernorm.weight"))
    x = _seq(X, 2, 16, dm)
    blk = ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv, numeric_profile=_NPP_BLOCK_SPEC)
    plain = ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv)
    yb = np.asarray(blk.forward(x))
    twin = pickle.loads(pickle.dumps(blk))
    parts["block"] = _h(yb)
    flags += [blk.numeric_profile == _NPP_BLOCK_SPEC,
              plain.numeric_profile == "fp32_v1",
              twin.numeric_profile == _NPP_BLOCK_SPEC,
              np.asarray(twin.forward(x)).tobytes() == yb.tobytes(),
              yb.tobytes() != np.asarray(plain.forward(x)).tobytes(),
              _hf_refused(lambda: ml.TransformerBlock(w, n_heads=nh, n_kv_heads=nkv,
                                                      numeric_profile={"preset": "fixed15_v1",
                                                                       "lm_head": "fp32_v1"}), "lm_head")]
    parts["flags"] = _h(np.asarray(flags, dtype=np.int64))
    return _fit(parts)


@lane("numeric-profile-parallel-default")
def _(ml, X, yc, yr, Xh=None):
    """`models.ParallelCausalLM` (every layer owned by device 0) under the default, naming no profile, and under a
    per-part spec: its profile read back and its logits held to the plain CausalLM loaded with the same spec."""
    from mojolearn.models.parallel_causal_lm import ParallelCausalLM
    A = ml.Array.from_buffer
    CLM = ml.models.causal_lm.CausalLM
    ids = _ids(X, HF_LM_BATCH, HF_LM_LEN, HF_LM_VOCAB)
    parts, flags = {}, []
    with tempfile.TemporaryDirectory(prefix="ib-numeric-profile-parallel-") as d:
        root = _npp_checkpoint(d)
        for name, spec in (("default", None), ("parts", _NPP_SAVED_SPEC)):  # driver-loop: over two specs
            plain = CLM.load(root, numeric_profile="fixed15_v1" if spec is None else spec)
            owners = (0,) * int(plain.plan.n_layers)
            par = (ParallelCausalLM.load(root, layer_devices=owners) if spec is None
                   else ParallelCausalLM.load(root, layer_devices=owners, numeric_profile=spec))
            with par:
                if str(par.vendor_used()) != str(plain.vendor_used()):
                    raise AssertionError(f"ParallelCausalLM ran on {par.vendor_used()!r} and this side's CausalLM "
                                         f"on {plain.vendor_used()!r}: ParallelCausalLM has no device=")
                y = np.asarray(par.forward(A(ids)))
                profile = par.numeric_profile
            parts[name] = _h(y)
            flags += [profile == ("fixed15_v1" if spec is None else spec),
                      y.tobytes() == np.asarray(plain.forward(A(ids))).tobytes()]
    parts["flags"] = _h(np.asarray(flags, dtype=np.int64))
    return _fit(parts)


_one_sided("numeric-profile-parallel-default", missing="cpu", reason="models.ParallelCausalLM keeps devices= (a GPU "
           "list) and takes no device=, by the owner's direction of 2026-11-24")


_batch_decl("n/a:profile-cases (per-part numeric profile cases of whole loaded models and one block: each case "
            "loads its own model; outputs digested whole)",
            "numeric-profile-parts", "numeric-profile-parallel-default")
