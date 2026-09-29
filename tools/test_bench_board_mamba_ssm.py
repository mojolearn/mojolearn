# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""The mamba-ssm opponent arms (tools/bench_board_neural.py MAMBA_SSM_ARMS):
planning, the install pins, the weight mapping, the constructor config, the
BOARD-PARAMS record, and that the Mamba lanes' settings (the opponent store's
key) do not move. Standard library and pytest only (no torch, no mamba_ssm)."""
import importlib.util
import json
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


def test_planned_on_nvidia_mamba_forward_only():
    assert N.MAMBA_SSM_ARMS == ARMS and N.MAMBA_SSM_LANES == LANES
    for lane in N.LANES:
        for vendor in N.VENDORS:
            opp = N.opponents(vendor, lane)
            has = [a for a in opp if a.startswith("mamba-ssm-")]
            assert has == (list(ARMS) if vendor == "nvidia" and lane in LANES else []), (vendor, lane)
            if has:   # the torch reference arms stay, the new arms follow them
                assert opp[:-2] and all(a.startswith("torch-") for a in opp[:-2])
    for arm in ARMS:
        assert arm in N.ARMS
        assert bb.arm_library(arm) == "mamba-ssm" and bb.arm_device(arm, "nvidia") == "gpu"
    # mamba1: the compile arms stay off (the reference loop); the mamba-ssm arms are eager
    assert N.opponents("nvidia", "mamba1-forward") == ("torch-eager-fp32", "torch-eager-tf32",
                                                        "torch-eager-bf16") + ARMS


def test_not_planned_is_named_on_amd_and_apple():
    for v in ("amd", "apple"):
        lines = [x for x in N.NOT_PLANNED[v] if x.startswith("mamba-ssm-* on mamba*-forward")]
        assert len(lines) == 1, v
    assert "ROCm" in " ".join(N.NOT_PLANNED["amd"]) and "gfx942" in " ".join(N.NOT_PLANNED["amd"])
    assert not [x for x in N.NOT_PLANNED["nvidia"] if x.startswith("mamba-ssm-* on")]
    assert any("mamba-ssm-" in x and "NVIDIA" in x for x in N.NOT_COVERED)


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
    """our names -> mamba_ssm Block names: the block RMSNorm is block.norm, the
    rest the mixer's own names (mamba2's gated `norm.weight` stays the mixer's)."""
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
    """mamba/corpus/gen_corpus.py's constants, read from its source (the real
    module needs torch)."""
    src = open(os.path.join(REPO, "mamba", "corpus", "gen_corpus.py")).read()
    ns = {}
    for name in ("D_STATE", "D_CONV", "EXPAND", "EPS", "M2_D_STATE", "M2_D_CONV", "M2_EXPAND",
                 "M2_HEADDIM", "M2_NGROUPS", "M2_CHUNK", "M2_EPS", "M3_D_STATE", "M3_EXPAND",
                 "M3_HEADDIM", "M3_NGROUPS", "M3_CHUNK", "M3_EPS", "M3_A_FLOOR"):
        line = next(ln for ln in src.splitlines() if ln.startswith(name + " ="))
        ns[name] = eval(line.split("=", 1)[1].split("#")[0])  # noqa: S307 (a numeric literal)
    return types.SimpleNamespace(**ns)


def test_mixer_config_is_explicit_and_the_references():
    c = _corpus()
    m1, e1 = N.mamba_ssm_config("mamba1", c, 384)
    assert m1 == dict(d_state=16, d_conv=4, expand=2, dt_rank=24, conv_bias=True, bias=False,
                      use_fast_path=True) and e1 == 1e-5
    assert N.mamba_ssm_config("mamba1", c, 16)[0]["dt_rank"] == 1
    m2, e2 = N.mamba_ssm_config("mamba2", c, 384)
    assert m2 == dict(d_state=128, d_conv=4, expand=2, headdim=64, ngroups=1, D_has_hdim=False,
                      rmsnorm=True, norm_before_gate=False, dt_limit=(0.0, float("inf")),
                      bias=False, conv_bias=True, chunk_size=256, use_mem_eff_path=True)
    assert e2 == 1e-5
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
    ours_block = types.SimpleNamespace(dt_rank=24, dt_limit=(0.0, float("inf")))
    ours = N._ours_record(lane)
    ours.update(N.ours_ssm_record(model, _fake_ours_module(), ours_block))
    back, eps = N.mamba_ssm_readback(model, _fake_block(model))
    assert set(eps.values()) == {1e-5}
    rec = {"__library__": "mamba-ssm", "seed": 7}
    rec.update(back)
    rep = BP.check("neural/" + lane, {"ours": ours, "mamba-ssm-fp32": rec,
                                      "torch-eager-fp32": {"__library__": "torch", "seed": 7}},
                   family="neural")
    assert rep["verdict"] == "MATCHED", rep["problems"]
    compared = {c["param"] for c in rep["compared"] if c["arm"] == "mamba-ssm-fp32"}
    assert compared == set(back) and len(compared) >= 4
    assert not [c for c in rep["compared"] if c["arm"] == "torch-eager-fp32"]
    # a mixer constructed with another chunk size / d_state refuses the race by name
    key = "d_state" if model == "mamba1" else "chunk_size"
    bad, _ = N.mamba_ssm_readback(model, _fake_block(model, **{key: 999}))
    rep = BP.check("neural/" + lane, {"ours": ours, "mamba-ssm-fp32": dict(rec, **bad)},
                   family="neural")
    assert rep["verdict"] == "REFUSED" and "ssm_" + key in " ".join(rep["problems"])
    # and a wrong seed
    rep = BP.check("neural/" + lane, {"ours": ours, "mamba-ssm-tf32": dict(rec, seed=0)},
                   family="neural")
    assert rep["verdict"] == "REFUSED"


def test_ssm_aliases_do_not_touch_other_libraries():
    names = [k for k in BP.ALIASES["*"] if k.startswith("ssm_")]
    assert sorted(names) == sorted(["ssm_d_state", "ssm_d_conv", "ssm_expand", "ssm_dt_rank",
                                    "ssm_headdim", "ssm_ngroups", "ssm_chunk_size",
                                    "ssm_dt_limit", "ssm_rope_angles"])
    assert all(BP.ALIASES["*"][k] == k for k in names)


def test_mamba_lane_settings_do_not_move_for_the_store():
    """The opponent store keys the torch arms by the race's settings; adding the
    mamba-ssm arms must not change them (their stored cells are reused)."""
    ctx = {"python": "py", "neural_driver": "drv", "vendor": "nvidia", "rounds": 5, "out": "/o",
           "round_seconds": 0}
    for lane in LANES:
        race = bb.plan_races("nvidia", ["identical"], ["neural"], [lane], cpu_arm=False)[0]
        assert race["opponents"][-2:] == list(ARMS)
        s = bb.race_settings(ctx, race)
        assert "mamba-ssm" not in json.dumps(s, default=str), lane
        cmd, _, _ = bb.neural_cmd(ctx, race)
        assert cmd[cmd.index("--arms") + 1].endswith(",mamba-ssm-fp32,mamba-ssm-tf32")


def test_install_pins_and_steps(tmp_path):
    assert bb.MAMBA_SSM_COMMIT and len(bb.MAMBA_SSM_COMMIT) == 40
    # the same commit mamba/corpus/gen_corpus.py cites for its references
    assert bb.MAMBA_SSM_COMMIT in open(os.path.join(REPO, "mamba", "corpus", "gen_corpus.py")).read()
    assert bb.mamba_ssm_install_steps("py", "amd") == [] and bb.mamba_ssm_install_steps("py", "apple") == []
    steps = bb.mamba_ssm_install_steps("py", "nvidia")
    assert len(steps) == 2
    (deps, e0), (build, e1) = steps
    assert deps[:4] == ["py", "-m", "pip", "install"] and e0 == {}
    assert "transformers==5.17.0" in deps
    assert "--no-build-isolation" in build and "--no-deps" in build
    assert "causal-conv1d==1.7.0" in build
    assert "mamba-ssm @ git+https://github.com/state-spaces/mamba@" + bb.MAMBA_SSM_COMMIT in build
    assert e1["MAMBA_KEEP_CUDA_BUILD"] == "TRUE" and e1["MAMBA_FORCE_BUILD"] == "TRUE"
    assert e1["CAUSAL_CONV1D_FORCE_BUILD"] == "TRUE" and int(e1["MAX_JOBS"]) >= 1
    # every requirement is pinned exactly (a git build pins its commit)
    for r in deps[6:] + build[8:]:
        assert "==" in r or "@" in r, r
    # --opponent-wheels: the prebuilt wheels by name, no index
    local = bb.mamba_ssm_install_steps("py", "nvidia", str(tmp_path))
    assert all("--no-index" in a for a, _ in local)
    assert "mamba-ssm" in local[1][0] and not any("git+" in x for x in local[1][0])
    names = bb.pinned_package_names()
    for n in ("mamba-ssm", "causal-conv1d", "transformers"):
        assert n in names


def test_memory_probe_reads_torch_allocator_for_mamba_ssm():
    assert "mamba-ssm" in PROBE.TORCH_LIBRARIES
    assert PROBE.IMPORT_NAME["mamba-ssm"] == "mamba_ssm"
