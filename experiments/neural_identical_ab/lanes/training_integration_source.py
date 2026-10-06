"""Author concrete training A/B selectors; no runtime or verification work."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write():
    rows = json.loads((HERE / "training.json").read_text())["experiments"]
    experiments = []
    for source in rows:
        number = int(source["id"][2:])
        runtime = {}
        lanes = ["lm-train-step"]
        if number == 53:
            runtime = {"operation": "lm_chunked_head_v2", "chunked_lm_head_v2": True}
        elif number == 57:
            lanes = ["samba-train-step"]
            runtime = {"operation": "samba_training_config", "samba_max_norm": 1.0}
        elif number in (58, 63):
            lanes = ["samba-train-step"]
            runtime = {"operation": "samba_training_config", "samba_accumulation_steps": 2}
        elif number == 59:
            lanes = ["transformer-forward"]
            runtime = {"operation": "residual_dropout_forward_vjp", "dropout_probability": 0.1,
                       "seed": 7, "stream": 0, "offset": 0}
        elif number == 64:
            lanes = ["mlp-forward"]
            runtime = {"operation": "mlp_inference_sessions", "session_count": 4}
        elif number == 60:
            lanes = ["lm-train-step", "lm-forward"]
        entry = dict(id=source["id"], implementation_status=source["implementation_status"],
                     implementation_paths=source["implementation_paths"], caller_paths=source["implementation_paths"],
                     legacy_experiments=(["I10"] if number in (55, 61) else []),
                     baseline=dict(compile_defines=source["baseline_defines"], environment={}, runtime=dict(runtime)),
                     candidate=dict(compile_defines=source["candidate_defines"], environment={}, runtime=dict(runtime)),
                     limitations=list(source["remaining"]),
                     workloads=[dict(harness="tools/bench_board_neural.py", lane=lane, shape="full",
                                     dataset_recipe="make_inputs / shape_record; resolve actual input hash and intrinsic caps before future execution",
                                     coverage="selected caller only; all transitive affected models and combinations remain unmeasured")
                                for lane in lanes])
        if number == 53:
            entry["variants"] = {
                "panel512": {"candidate": {"compile_defines": ["MOJOLEARN_NN53_HEAD_CHUNK512=1"]}},
                "panel2048": {"candidate": {"compile_defines": ["MOJOLEARN_NN53_HEAD_CHUNK2048=1"]}},
            }
        if number == 59:
            entry["baseline"]["compile_defines"] = ["MOJOLEARN_NN59_DROPOUT_RESIDUAL=1", "MOJOLEARN_NN59_CONTROL=1"]
            entry["limitations"].append("The selected operation is a complete explicit layer and its VJP; it is not a transformer model train step or model-quality evidence.")
        if number == 60:
            entry["variants"] = {row["name"]: {"candidate": {"compile_defines": row["defines"]}}
                                 for row in source["arms"]}
        if number == 63:
            entry["limitations"].append("This recipe reaches canonical accumulation on one device. Existing training/accumulate_multi_gpu.mojo owns multi-device transport; topology coverage remains a separate future workload.")
        if number == 64:
            entry["limitations"].append("The saved MLP input fixture is reused in full through the GPU mlp-forward lane. The explicit operation uses the selected training binding. The CPU-only MLPInference lane remains separate; host arithmetic qualification needs its own matching binding recipe.")
        experiments.append(entry)
    (HERE / "training_integration_inventory.json").write_text(json.dumps(dict(
        schema=1, lane="training", qualification="uncompiled_unverified_unmeasured",
        experiments=experiments), indent=2) + "\n")


if __name__ == "__main__":
    write()
