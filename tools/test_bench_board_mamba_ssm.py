# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The library opponents of the neural lanes (FAIRNESS FIX, 2026-10-13 audit): the mamba-ssm arms
(tools/bench_board_neural.py MAMBA_SSM_VENDOR_ARMS, from lane/mamba-opp facde18cc, opened to AMD's
ROCm build), the refused pure-PyTorch references, the transformers weight maps, the compile
warm-up deadline and the compile-timeout rule. Standard library and pytest only (no torch, no
mamba_ssm, no transformers): nothing here runs a kernel."""
import importlib.util
import os
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)


def _load(name):
    spec = importlib.util.spec_from_file_location("t_mssm_" + name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bb = _load("bench_board")
N = bb.NEURAL
BP = _load("bench_board_params")
PROBE = _load("bench_board_probe")

LANES = ("mamba1-forward", "mamba2-forward", "mamba3-forward")
ARMS = ("mamba-ssm-fp32", "mamba-ssm-tf32")


def test_planned_where_mamba_ssm_is_built():
    assert N.MAMBA_SSM_ARMS == ARMS and N.MAMBA_SSM_LANES == LANES
    want = {"nvidia": list(ARMS), "amd": ["mamba-ssm-fp32"], "apple": []}
    for lane in N.LANES:
        for vendor in N.VENDORS:
            has = [a for a in N.opponents(vendor, lane) if a.startswith("mamba-ssm-")]
            assert has == (want[vendor] if lane in LANES else []), (vendor, lane)
    for arm in ARMS:
        assert arm in N.ARMS
        assert bb.arm_library(arm) == "mamba-ssm" and bb.arm_device(arm, "amd") == "gpu"


def test_references_are_refused_and_samba_is_not():
    for lane in LANES + ("mamba1-infer", "mamba2-infer", "mamba3-infer", "samba-infer"):
        assert lane in N.REFERENCE_TWIN_LANES and "reference" in N.REFERENCE_TWIN_LANES[lane]
        assert N.lane_settings(lane)["opponent_refused"] == N.REFERENCE_TWIN_LANES[lane]
    for lane in ("samba-train-step", "samba-forward"):
        assert lane not in N.REFERENCE_TWIN_LANES
        assert "mamba_ssm" in N.LANE_TEXT[lane][1]
    assert any("mamba-ssm-fp32 on NVIDIA and AMD" in x for x in N.NOT_COVERED)
    assert any(x.startswith("mamba-ssm-* on mamba*-forward") for x in N.NOT_PLANNED["apple"])
    assert any(x.startswith("mamba-ssm-tf32") for x in N.NOT_PLANNED["amd"])


def test_settings_precision():
    assert N.mamba_ssm_setting("mamba-ssm-fp32") == {"tf32": False, "triton_f32_default": "ieee"}
    assert N.mamba_ssm_setting("mamba-ssm-tf32") == {"tf32": True, "triton_f32_default": "tf32"}
    with pytest.raises(ValueError):
        N.mamba_ssm_setting("torch-eager-fp32")


def test_worker_parser_accepts_the_arms():
    for arm in ARMS:
        a = N.build_parser().parse_args(["worker", "--arm", arm, "--lane", "mamba2-forward",
                                         "--shape", "small", "--data", "x.npz"])
        assert a.arm == arm


def test_refuses_a_non_mamba_lane_before_touching_torch():
    for lane in ("transformer-forward", "mamba1-infer", "lm-forward"):
        with pytest.raises(RuntimeError, match="REFUSED: mamba-ssm-fp32 races the Mamba forward"):
            N.MambaSsmArm(lane, "small", {}, "mamba-ssm-fp32")


def test_every_weight_maps_to_the_block_state_dict():
    want = {
        "mamba1": {"norm.weight": "norm.weight", "in_proj.weight": "mixer.in_proj.weight",
                   "conv1d.weight": "mixer.conv1d.weight", "conv1d.bias": "mixer.conv1d.bias",
                   "x_proj.weight": "mixer.x_proj.weight", "dt_proj.weight": "mixer.dt_proj.weight",
                   "dt_proj.bias": "mixer.dt_proj.bias", "A_log": "mixer.A_log", "D": "mixer.D",
                   "out_proj.weight": "mixer.out_proj.weight"},
        "mamba2": {"block_norm.weight": "norm.weight", "in_proj.weight": "mixer.in_proj.weight",
                   "conv1d.weight": "mixer.conv1d.weight", "conv1d.bias": "mixer.conv1d.bias",
                   "dt_bias": "mixer.dt_bias", "A_log": "mixer.A_log", "D": "mixer.D",
                   "norm.weight": "mixer.norm.weight", "out_proj.weight": "mixer.out_proj.weight"},
        "mamba3": {"block_norm.weight": "norm.weight", "in_proj.weight": "mixer.in_proj.weight",
                   "dt_bias": "mixer.dt_bias", "B_norm.weight": "mixer.B_norm.weight",
                   "C_norm.weight": "mixer.C_norm.weight", "B_bias": "mixer.B_bias",
                   "C_bias": "mixer.C_bias", "D": "mixer.D", "out_proj.weight": "mixer.out_proj.weight"},
    }
    for model, names in N.MAMBA_NAMES.items():
        got = {n: N.mamba_ssm_state_name(model, n) for n in names}
        assert got == want[model], model
        assert len(set(got.values())) == len(got)


def _corpus():
    src = open(os.path.join(REPO, "mamba", "corpus", "gen_corpus.py")).read()
    ns = {}
    for name in ("D_STATE", "D_CONV", "EXPAND", "EPS", "M2_D_STATE", "M2_D_CONV", "M2_EXPAND",
                 "M2_HEADDIM", "M2_NGROUPS", "M2_CHUNK", "M2_EPS", "M3_D_STATE", "M3_EXPAND",
                 "M3_HEADDIM", "M3_NGROUPS", "M3_CHUNK", "M3_EPS", "M3_A_FLOOR"):
        line = next(ln for ln in src.splitlines() if ln.startswith(name + " ="))
        ns[name] = eval(line.split("=", 1)[1].split("#")[0])  # noqa: S307 (a numeric literal)
    return types.SimpleNamespace(**ns)


def test_mixer_config_is_explicit():
    c = _corpus()
    m1, e1 = N.mamba_ssm_config("mamba1", c, 384)
    assert m1 == dict(d_state=16, d_conv=4, expand=2, dt_rank=24, conv_bias=True, bias=False,
                      use_fast_path=True) and e1 == 1e-5
    m2, _ = N.mamba_ssm_config("mamba2", c, 384)
    assert m2["chunk_size"] == 256 and m2["use_mem_eff_path"] is True
    m3, e3 = N.mamba_ssm_config("mamba3", c, 384)
    assert m3 == dict(d_state=128, expand=2, headdim=64, ngroups=1, rope_fraction=0.5,
                      A_floor=1e-4, is_outproj_norm=False, is_mimo=False, chunk_size=64)
    assert e3 == 1e-5


def _fake_ours_module():
    return types.SimpleNamespace(
        _M1_D_STATE=16, _M1_D_CONV=4, _M1_EXPAND=2, _M2_D_STATE=128, _M2_D_CONV=4, _M2_EXPAND=2,
        _M2_HEADDIM=64, _M2_NGROUPS=1, _M2_CHUNK_SIZE=256, _M3_D_STATE=128, _M3_EXPAND=2,
        _M3_HEADDIM=64, _M3_NGROUPS=1, _M3_CHUNK_SIZE=64, _M3_NUM_ROPE_ANGLES=32)


def _fake_block(model, **over):
    norm = types.SimpleNamespace(eps=1e-5)
    m = dict(d_state=128, expand=2, d_conv=4, dt_rank=24, headdim=64, ngroups=1, num_bc_heads=1,
             chunk_size={"mamba2": 256, "mamba3": 64}.get(model, 0), dt_limit=(0.0, float("inf")),
             num_rope_angles=32, norm=norm, B_norm=norm, C_norm=norm)
    if model == "mamba1":
        m["d_state"] = 16
    m.update(over)
    return types.SimpleNamespace(norm=norm, mixer=types.SimpleNamespace(**m))


@pytest.mark.parametrize("model", ["mamba1", "mamba2", "mamba3"])
def test_board_params_match_ours_and_refuse_a_difference(model):
    lane = model + "-forward"
    ours = N._ours_record(lane)
    ours.update(N.ours_ssm_record(model, _fake_ours_module(),
                                  types.SimpleNamespace(dt_rank=24, dt_limit=(0.0, float("inf")))))
    back, eps = N.mamba_ssm_readback(model, _fake_block(model))
    assert set(eps.values()) == {1e-5}
    rec = {"__library__": "mamba-ssm", "seed": 7}
    rec.update(back)
    rep = BP.check("neural/" + lane, {"ours": ours, "mamba-ssm-fp32": rec}, family="neural")
    assert rep["verdict"] == "MATCHED", rep["problems"]
    key = "d_state" if model == "mamba1" else "chunk_size"
    bad, _ = N.mamba_ssm_readback(model, _fake_block(model, **{key: 999}))
    rep = BP.check("neural/" + lane, {"ours": ours, "mamba-ssm-fp32": dict(rec, **bad)},
                   family="neural")
    assert rep["verdict"] == "REFUSED" and "ssm_" + key in " ".join(rep["problems"])


def test_install_pins_and_steps(tmp_path):
    assert bb.MAMBA_SSM_COMMIT in open(os.path.join(REPO, "mamba", "corpus", "gen_corpus.py")).read()
    assert bb.mamba_ssm_install_steps("py", "apple") == []
    for vendor in ("nvidia", "amd"):
        (deps, e0), (build, e1) = bb.mamba_ssm_install_steps("py", vendor)
        assert deps[:4] == ["py", "-m", "pip", "install"] and e0 == {}
        assert "--no-build-isolation" in build and "--no-deps" in build
        assert "causal-conv1d==1.7.0" in build
        assert "mamba-ssm @ git+https://github.com/state-spaces/mamba@" + bb.MAMBA_SSM_COMMIT in build
        assert e1["MAMBA_FORCE_BUILD"] == "TRUE" and e1["CAUSAL_CONV1D_FORCE_BUILD"] == "TRUE"
        assert (e1.get("PYTORCH_ROCM_ARCH") == "gfx942") == (vendor == "amd")
    assert "transformers==5.17.0" in bb.NEURAL_PINS and N.TRANSFORMERS_PIN in bb.NEURAL_PINS
    local = bb.mamba_ssm_install_steps("py", "amd", str(tmp_path))
    assert all("--no-index" in a for a, _ in local)
    names = bb.pinned_package_names()
    for n in ("mamba-ssm", "causal-conv1d", "transformers", "torchvision"):
        assert n in names


def test_memory_probe_reads_torch_allocator_for_mamba_ssm():
    assert "mamba-ssm" in PROBE.TORCH_LIBRARIES
    assert PROBE.IMPORT_NAME["mamba-ssm"] == "mamba_ssm"


def test_transformers_weight_maps():
    shapes = [n for n, _ in _lm_registry()]
    hf = [N.hf_lm_state_name(n) for n in shapes]
    assert hf[0] == "model.embed_tokens.weight" and hf[-1] == "lm_head.weight"
    assert "model.layers.1.self_attn.k_proj.weight" in hf and "model.layers.0.mlp.down_proj.weight" in hf
    assert len(set(hf)) == len(hf) and not any(h == "model.norm.weight" for h in hf)
    assert set(N.HF_LAYER_NAME) == set(N.TRANSFORMER_NAMES)
    assert len(set(N.HF_LAYER_NAME.values())) == len(N.HF_LAYER_NAME)


def _lm_registry():
    twin = _load("torch_lm_step_opponent")
    return twin.registry([1, 8, 16, 2, 2, 8, 32, 2, 32])


def test_compile_warmup_deadline_and_timeout_rule():
    a = N.build_parser().parse_args(["race", "--lane", "lm-train-step", "--out", "o", "--work", "w"])
    assert N.warmup_deadline("torch-compile-bf16", a) == 8 * a.round_seconds
    assert N.warmup_deadline("torch-eager-bf16", a) == a.warmup_seconds
    a = N.build_parser().parse_args(["race", "--lane", "lm-train-step", "--out", "o", "--work", "w",
                                     "--compile-warmup-seconds", "99999"])
    assert N.warmup_deadline("torch-compile-fp32", a) == 99999
    r = {"ours": {"status": "ok"}, "torch-eager-fp32": {"status": "ok"},
         "torch-compile-fp32": {"status": "timeout"}, "mamba-ssm-fp32": {"status": "ok"}}
    marked = N.mark_not_opponents("samba-train-step", r)
    assert set(marked) == {"torch-eager-fp32", "mamba-ssm-fp32"} and "ours" not in marked
    assert N.mark_not_opponents("lm-train-step", {"ours": {"status": "ok"},
                                                  "torch-eager-fp32": {"status": "ok"}}) == {}
    race = bb.plan_races("nvidia", ["identical"], ["neural"], ["samba-train-step"])[0]
    ctx = {"python": "py", "neural_driver": "drv", "vendor": "nvidia", "rounds": 1, "out": "/o",
           "round_seconds": 0}
    cmd, _, ceiling = bb.neural_cmd(ctx, race)
    assert cmd[cmd.index("--compile-warmup-seconds") + 1] == str(bb.COMPILE_WARMUP_FACTOR * 1800)
    race = bb.plan_races("nvidia", ["identical"], ["neural"], ["mamba1-forward"])[0]
    assert "--compile-warmup-seconds" not in bb.neural_cmd(ctx, race)[0]
