"""The neural headline column choice, stored-hash quality, and the race's host-twin naming (lane/board-quality-bf16).

Metadata only: no bindings, no box, no arrays beyond tiny log files.
"""
import importlib.util
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bb = _load("bench_board")
hq = _load("bench_board_host_quality")


def _cell(arm, ms, library=None, mode=None, status="ok", hash_=None, quality=None):
    return {"arm": arm, "library": library or ("mojolearn" if arm.startswith("ours") else "torch"),
            "mode": mode or ("identical" if arm.startswith("ours") else None), "status": status,
            "median_ms": ms, "min_ms": ms, "max_ms": ms, "rounds": 1, "device": "gpu",
            "hash": hash_, "quality": quality if quality is not None else {}}


def _neural_race(cells, family="algos", lane="conv2d"):
    return {"family": family, "lane": lane, "dataset": "synthetic", "cells": cells}


def test_headline_is_fastest_bf16_arm_with_fp32_twin_second():
    rr = _neural_race([
        _cell("ours", 9.66),
        _cell("torch-eager-fp32", 2.40), _cell("torch-compile-fp32", 2.30),
        _cell("torch-eager-tf32", 1.10),                      # TF32 is neither column
        _cell("torch-eager-bf16", 1.50), _cell("torch-compile-bf16", 1.33),
        _cell("torch-cpu-eager-bf16", 0.50),                  # a CPU torch arm never heads a GPU board
        _cell("torch-compile-bf16-x", 0.10, status="error"),  # not completed: never chosen
    ])
    hl = bb.neural_headline(rr)
    assert hl["bf16"]["arm"] == "torch-compile-bf16"
    assert abs(hl["bf16"]["ratio"] - 9.66 / 1.33) < 1e-12
    assert hl["fp32"]["arm"] == "torch-compile-fp32"
    assert abs(hl["fp32"]["ratio"] - 9.66 / 2.30) < 1e-12
    assert "identity tax" in hl["note"] and "tensor cores" in hl["note"]


def test_headline_eager_bf16_wins_when_it_is_the_lower_median():
    rr = _neural_race([_cell("ours", 5.0), _cell("torch-eager-bf16", 1.0), _cell("torch-compile-bf16", 2.0)],
                      family="neural", lane="mlp-train-step")
    assert bb.neural_headline(rr)["bf16"]["arm"] == "torch-eager-bf16"


def test_fp32_only_neural_lane_has_no_bf16_headline():
    rr = _neural_race([_cell("ours", 13.0), _cell("torch-eager-fp32", 20.8)], lane="nadam")
    hl = bb.neural_headline(rr)
    assert hl["bf16"] is None and hl["fp32"]["arm"] == "torch-eager-fp32"


def test_classical_races_get_no_headline_and_keep_their_columns():
    cells = [_cell("ours", 10.0), _cell("cuml-gpu", 5.0, library="cuml")]
    for rr in ({"family": "classical", "lane": "kmeans", "dataset": "taxi", "cells": cells},
               {"family": "algos", "lane": "ridge", "dataset": "taxi", "cells": cells}):
        assert bb.neural_headline(rr) is None
    assert bb.render_neural_headline({"classical/kmeans/taxi": {"family": "classical", "lane": "kmeans",
                                                                 "dataset": "taxi", "cells": cells}}) == []
    out = bb.add_ratios([dict(c) for c in cells])
    assert out[1]["ratio_ours_identical_over"] == 2.0    # the per-arm ratio column is unchanged


def test_headline_table_rows():
    races = {"algos/conv2d/synthetic": _neural_race([_cell("ours", 9.66), _cell("torch-compile-bf16", 1.33),
                                                     _cell("torch-eager-fp32", 2.30)])}
    lines = bb.render_neural_headline(races)
    assert any(l.startswith("## Neural headline") for l in lines)
    row = [l for l in lines if l.startswith("| conv2d |")][0]
    assert "torch-compile-bf16" in row and "torch-eager-fp32" in row and "identity tax" in row
    assert not any(w in " ".join(lines).lower().split() for w in ("faster", "slower"))


def test_stored_hash_equal_gives_identical_to_quality():
    h = "2753e61a6dde36e3"
    ours = _cell("ours", 2.65, hash_=h, quality={"error": "Own host reference failed (x); see host.log"})
    cells = bb.stored_identity_quality([ours, _cell("torch-eager-fp32", 1.88, hash_=h + "0" * 48)])
    assert cells[0]["quality"] == {"identical_to": "torch-eager-fp32"}
    assert cells[0]["quality_superseded"]["error"].startswith("Own host reference failed")


def test_stored_hash_rule_leaves_numeric_quality_and_unequal_hashes():
    q = {"rel_fro_vs_torch_eager_fp32": 1e-7}
    a = _cell("ours", 1.0, hash_="a" * 16, quality=dict(q))
    b = _cell("ours", 1.0, hash_="b" * 16)
    bb.stored_identity_quality([a, _cell("torch-eager-fp32", 1.0, hash_="a" * 16)])
    bb.stored_identity_quality([b, _cell("torch-eager-fp32", 1.0, hash_="c" * 16)])
    assert a["quality"] == q and b["quality"] == {}


def test_missing_host_bindings_reads_both_refusals(tmp_path):
    log = tmp_path / "host.log"
    log.write_text("Traceback ...\nImportError: mojolearn: /root/t/python/mojolearn/host/_mojolearn_x_cnn_host.so "
                   "is not built. Build it with bindings/build_x_cnn_host.sh\n"
                   "ImportError: mojolearn: no CPU implementation of _mojolearn_linalg.matmul yet; see "
                   "SUPPORT_MATRIX.md (no host binding covers _mojolearn_linalg; host bindings built here: none)\n")
    assert hq.missing_host_bindings(log) == ["_mojolearn_x_cnn_host", "_mojolearn_linalg_host"]
    assert hq.failure_line(log).startswith("ImportError: mojolearn: no CPU implementation")
    assert hq.missing_host_bindings(tmp_path / "absent.log") == []


def test_race_builds_add_the_host_twin_only_for_host_reference_lanes():
    algos = bb.ALGOS
    assert algos.race_builds("nadam") == ["build_x_sequence", "build_x_sequence_host"]
    assert algos.race_builds("conv2d") == ["build_x_cnn", "build_x_cnn_host"]
    assert algos.race_builds("embedding") == ["build_embedding", "build_embedding_host"]
    # torch-eager-fp32 in the race: quality compares against it, no host twin
    assert algos.race_builds("conv2d", ("ours", "torch-eager-fp32")) == ["build_x_cnn"]
