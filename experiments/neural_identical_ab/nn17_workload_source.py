"""Author NN17's explicit full GQA block recipe; no ML execution or checks."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def write():
    # Workload dimensions, not a dispatch rule. Four query heads per K/V
    # head reach the declared shared-page arm; full sequence length follows
    # the saved neural block recipe. Both arms consume this entire model.
    dimensions = dict(batch=1, length=2048, d_model=512, n_heads=8,
                      n_kv=2, head_dim=64, intermediate=1536)
    recipe = dict(schema=1, id="NN17", harness="tools/bench_board_neural.py",
                  lane="transformer-forward", shape="full", seed=7,
                  runtime=dict(operation="transformer_configured_forward", transformer_config=dimensions),
                  input_source="make_inputs: complete declared Gaussian block input and initialized weights, same saved file for both arms",
                  scope="supplemental full-block GQA workload; does not replace complete LM corpus training/quality or every affected estimator",
                  input_hash="pending_future_fixture_authoring", execution="not_run_by_request")
    destination = HERE / "workloads/NN17-full-gqa.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(recipe, indent=2) + "\n")
    path = HERE / "lanes/attention_integration_inventory.json"
    content = json.loads(path.read_text())
    for row in content["experiments"]:
        if row["id"] != "NN17":
            continue
        for arm in ("baseline", "candidate"):
            row[arm]["runtime"] = dict(recipe["runtime"])
        row["workloads"] = [dict(recipe, recipe_path=str(destination.relative_to(HERE.parents[1])))]
        row["limitations"] = [s for s in row["limitations"] if not s.startswith("Ordinary transformer-forward fixture")]
        row["limitations"].append("The ordinary full transformer fixture has GQA ratio one. The explicit saved NN17-full-gqa recipe uses ratio four in both arms and reaches the public configured model; it is a supplemental full block workload, not full LM corpus coverage. Its input hash, branch reach and all execution evidence remain pending.")
    path.write_text(json.dumps(content, indent=2) + "\n")


if __name__ == "__main__":
    write()
