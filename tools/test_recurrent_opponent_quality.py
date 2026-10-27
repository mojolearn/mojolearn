"""Exercise real Torch recurrent opponents through fit, infer and quality export."""
import copy
import os

import numpy as np
import pytest

import bench_board_algos as A


@pytest.mark.parametrize("cell", ["gru", "lstm", "rnn"])
@pytest.mark.parametrize("task", ["clf", "reg"])
def test_recurrent_opponent_retains_scored_outputs(monkeypatch, cell, task):
    torch = pytest.importorskip("torch")
    device = os.environ.get("MOJOLEARN_QUALITY_TEST_DEVICE", "cpu")
    lane = cell + "-" + task
    spec = copy.deepcopy(A.LANES[lane])
    spec["params"].update(hidden_size=4, max_epochs=1, batch_size=4)
    monkeypatch.setitem(A.LANES, lane, spec)
    monkeypatch.setattr(A, "_torch_device", lambda _: device)
    nn_cell = {"gru": torch.nn.GRU, "lstm": torch.nn.LSTM, "rnn": torch.nn.RNN}[cell]
    recurrent = nn_cell(1, 4, batch_first=True)
    head = torch.nn.Linear(4, 2 if task == "clf" else 1)
    initial = {k: v.detach().numpy().copy() for k, v in recurrent.state_dict().items()}
    initial.update({"head." + k: v.detach().numpy().copy() for k, v in head.state_dict().items()})
    monkeypatch.setattr(A, "_read_ours_init", lambda *args: (
        initial, [np.arange(8, dtype=np.int64)], {"fixture": True}))
    rng = np.random.default_rng(8)
    data = {"X": rng.normal(size=(8, 3, 1)).astype(np.float32),
            "y": np.array([0, 1] * 4, dtype=np.float32),
            "Xq": rng.normal(size=(4, 3, 1)).astype(np.float32),
            "yq": np.array([0, 1, 1, 0], dtype=np.float32)}
    runner = A._build_seqmodel(lane, "torch-eager-fp32", data)
    runner.fit()
    runner.infer()
    outputs = runner.outputs()
    quality = A.quality(lane, data, {"torch-eager-fp32": outputs})["torch-eager-fp32"]
    if task == "clf":
        assert outputs["logits"].shape == (4, 2)
        expected = torch.softmax(torch.from_numpy(outputs["logits"]), dim=1).numpy()[:, 1]
        np.testing.assert_allclose(outputs["proba1"], expected, rtol=1e-14)
        np.testing.assert_array_equal(outputs["pred"], outputs["logits"].argmax(1))
        assert np.all((outputs["proba1"] > 0) & (outputs["proba1"] < 1))
        labels = data["yq"].astype(np.float64)
        expected_loss = -(labels * np.log(expected) + (1 - labels) * np.log1p(-expected)).mean()
        assert quality["logloss"] == pytest.approx(expected_loss)
        assert quality["accuracy"] == np.mean(outputs["pred"] == labels)
    else:
        assert set(outputs) == {"pred"}
        assert np.isfinite(quality["rmse"])
        assert "logloss" not in quality
