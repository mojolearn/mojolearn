"""CPU stub regressions for neural preparation, scored/save receipts and exits.

No mojolearn binding, GPU, opponent or measurement workload is executed.
"""
import json
import os
from pathlib import Path
import signal
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

import bench_board_neural as neural
import classical_two_datasets as ctd


FAKE_WORKER = r'''
import sys, types
import numpy as np
import bench_board_neural as n
class Runner:
    info = {"device": "test-cpu", "library": "test-stub"}
    def __init__(self, data): self.data = data; self.y = None
    def call(self): self.y = self.data["a"] @ self.data["b"]
    def sync(self): pass
    def outputs(self): return {"y": self.y}
    def digest(self): return n.outputs_digest(self.outputs())
class Mem:
    def __init__(self, *a, **k): pass
    def start(self): pass
    def stop(self): return {"test_stub": True}
original_load = n._load
def load(name, *a, **k):
    if name == "bench_board_probe":
        return types.SimpleNamespace(MemProbe=Mem, library_identity=lambda info: {})
    if name == "bench_board_compile_guard":
        return types.SimpleNamespace(read=lambda: {}, compiles=lambda a,b: 0)
    return original_load(name, *a, **k)
n._load = load
n.build_runner = lambda lane, arm, shape, data, ab_config=None: Runner(data)
args = n.build_parser().parse_args(sys.argv[1:])
if __import__("os").environ.get("TEST_RECEIPT_FAILURE"):
    original_receipt = n._neural_ab_receipt
    def receipt(runner, config, executed=False):
        if executed: raise RuntimeError("test receipt failed")
        return original_receipt(runner, config, executed)
    n._neural_ab_receipt = receipt
sys.exit(n.worker(args))
'''


class NeuralWorkerProtocolTests(unittest.TestCase):
    def setUp(self):
        retained = os.environ.get("MOJOLEARN_TEST_LOG_DIR")
        if retained:
            Path(retained).mkdir(parents=True, exist_ok=True)
            self.directory = Path(tempfile.mkdtemp(prefix=self._testMethodName + "-", dir=retained))
        else:
            temporary = tempfile.TemporaryDirectory()
            self.addCleanup(temporary.cleanup)
            self.directory = Path(temporary.name)
        self.workers = []

    def tearDown(self):
        for worker in self.workers:
            if worker.proc.poll() is None:
                worker.kill("test_cleanup", None)
            worker.close()
            try:
                worker.proc.stdin.close()
            except BrokenPipeError:
                pass
            worker.proc.stdout.close()

    def start(self, source, *args, env=None):
        child_env = dict(os.environ, PYTHONPATH=str(Path(neural.__file__).parent))
        # Stub tests never write into a controller's actual measured receipts.
        child_env.pop("MOJOLEARN_MEASUREMENT_RECEIPTS", None)
        child_env.update(env or {})
        worker = ctd.Worker("ours", [sys.executable, "-u", "-c", source, *args], child_env,
                            str(self.directory / ("worker-%d.log" % len(self.workers))), str(self.directory))
        self.workers.append(worker)
        return worker

    def stub(self, config=None, **kwargs):
        data = self.directory / "input.npz"
        np.savez(data, a=np.array([[1., 2.]], np.float32), b=np.array([[3.], [4.]], np.float32))
        args = ["worker", "--lane", "gemm", "--arm", "ours", "--shape", "small", "--data", str(data)]
        if config:
            path = self.directory / "config.json"
            path.write_text(json.dumps(config))
            args += ["--neural-ab-config", str(path)]
        return self.start(FAKE_WORKER, *args, **kwargs)

    def roundtrip(self, config=None):
        worker = self.stub(config)
        ready = neural._worker_message(worker, 10, "ready")
        self.assertEqual(ready["event"], "ready")
        rounds = []
        for index in (0, 1):
            message = neural._worker_message(worker, 10, "round", "round %d" % index)
            self.assertEqual(message["event"], "round", message)
            self.assertEqual(message["round"], index)
            self.assertGreaterEqual(message["operation"]["ms"], message["ms"])
            rounds.append(message)
        cold = rounds[0]["operation"]
        self.assertGreaterEqual(cold["preparation_ms"], 0)
        self.assertAlmostEqual(cold["cold_total_ms"], cold["preparation_ms"] + cold["ms"])
        self.assertEqual(rounds[1]["operation"]["preparation_ms"], 0)
        self.assertIsNone(rounds[1]["operation"]["cold_total_ms"])
        self.assertEqual(rounds[1]["state_receipt"]["output_status"], "ok")
        output = self.directory / "saved.npz"
        saved = neural._worker_message(worker, 10, "save", "save " + str(output))
        self.assertEqual(saved["event"], "saved", saved)
        with np.load(output) as arrays:
            np.testing.assert_array_equal(arrays["y"], [[11.]])
        self.assertEqual(neural._worker_message(worker, 10, "quit", "quit")["event"], "bye")
        self.assertEqual(worker.proc.wait(timeout=10), 0)
        (self.directory / "protocol.json").write_text(json.dumps([ready, *rounds, saved], indent=2))
        return ready, rounds

    def test_normal_warmup_scored_and_save(self):
        ready, rounds = self.roundtrip()
        self.assertIsNone(ready["neural_ab"])
        self.assertTrue(all(r["neural_ab"] is None for r in rounds))
        self.assertEqual(rounds[0]["operation"]["scope"], "training-step-or-forward-consume")

    def test_configured_ab_receipts_survive_ready_scope(self):
        config = {"id": "test", "arm": "candidate", "mode": "identical", "compile_defines": []}
        ready, rounds = self.roundtrip(config)
        self.assertFalse(ready["neural_ab"]["public_operation_completed"])
        for row in rounds:
            self.assertTrue(row["neural_ab"]["public_operation_completed"])
            self.assertEqual(row["neural_ab"]["configured"], config)
            self.assertEqual(row["operation"]["scope"], "model_default")

    def test_receipt_error_is_reported_as_round_error(self):
        worker = self.stub(env={"TEST_RECEIPT_FAILURE": "1"})
        self.assertEqual(neural._worker_message(worker, 10, "ready")["event"], "ready")
        message = neural._worker_message(worker, 10, "round", "round 0")
        self.assertEqual(message["event"], "error")
        self.assertIn("test receipt failed", message["error"])
        self.assertEqual(worker.proc.wait(timeout=10), 1)

    def test_exception_exit_is_not_timeout(self):
        worker = self.start("raise NameError('ab_config regression')")
        message = neural._worker_message(worker, 10, "round 0")
        self.assertEqual(message["reason"], "worker_exited")
        self.assertEqual(message["returncode"], 1)
        self.assertIsNone(message["signal"])
        self.assertEqual(message["read_failure"]["reason"], "eof")
        self.assertIn("ab_config regression", Path(message["worker_log"]).read_text())

    def test_signal_exit_is_not_timeout(self):
        worker = self.start("import os, signal; os.kill(os.getpid(), signal.SIGTERM)")
        message = neural._worker_message(worker, 10, "round 0")
        self.assertEqual(message["reason"], "worker_exited")
        self.assertEqual(message["returncode"], -signal.SIGTERM)
        self.assertEqual(message["signal_name"], "SIGTERM")

    def test_alive_worker_deadline_is_timeout_and_cleanup_is_separate(self):
        worker = self.start('import time; print(\'{"event":"ready"}\', flush=True); time.sleep(60)')
        self.assertEqual(neural._worker_message(worker, 10, "ready")["event"], "ready")
        message = neural._worker_message(worker, 0.05, "round 0")
        self.assertEqual(message["reason"], "timeout")
        self.assertIsNone(message["returncode"])
        self.assertEqual(message["read_failure"]["reason"], "deadline")
        row = {}
        neural._stop_failed_worker(worker, "timeout", message, row)
        self.assertIsNone(row["worker_process"]["returncode_before_cleanup"])
        self.assertEqual(row["worker_process"]["returncode_after_cleanup"], -signal.SIGKILL)

    def test_alive_worker_closing_stdout_is_eof_not_timeout(self):
        worker = self.start("import os, time; os.close(1); time.sleep(60)")
        message = neural._worker_message(worker, 10, "ready")
        self.assertEqual(message["reason"], "worker_eof")
        self.assertIsNone(message["returncode"])

    def test_broken_pipe_preserves_child_exit(self):
        worker = self.start("import sys; sys.exit(7)")
        self.assertEqual(worker.proc.wait(timeout=10), 7)
        message = neural._worker_message(worker, 10, "round 0", "round 0")
        self.assertEqual(message["reason"], "worker_exited")
        self.assertEqual(message["returncode"], 7)
        self.assertIsNotNone(message["send_error"])

    def conductor(self, source, expected_status):
        out, work = self.directory / "out", self.directory / "work"
        args = neural.build_parser().parse_args([
            "race", "--lane", "gemm", "--shape", "small", "--arms", "ours", "--rounds", "1",
            "--out", str(out), "--work", str(work), "--keep-outputs"])
        def inputs(lane, shape, rounds, path):
            np.savez(path, a=np.array([[1., 2.]], np.float32), b=np.array([[3.], [4.]], np.float32))
            return {"test_stub": True, "actual_shapes": {"a": [1, 2], "b": [2, 1]}}
        def start(arm, command, env, log_path, cwd, **kwargs):
            # Replace only the GPU implementation; the real conductor and
            # persistent worker protocol run from ready through saved outputs.
            return self.start(source, *command[2:])
        original_load = neural._load
        def load(name, *a, **k):
            if name == "classical_two_datasets":
                return types.SimpleNamespace(start_worker=start)
            if name == "bench_board_probe":
                return types.SimpleNamespace(refuse_our_cpu_arms=lambda *a: None)
            if name == "bench_board_params":
                return types.SimpleNamespace(enforce=lambda *a, **k: {"test_stub": True}, ParamsRefused=ValueError)
            return original_load(name, *a, **k)
        with patch.object(neural, "_load", side_effect=load), \
             patch.object(neural, "_worker_env", return_value={}), \
             patch.object(neural, "make_inputs", side_effect=inputs), \
             patch.object(neural, "scorer_preflight", return_value={"test_stub": True}):
            code = neural.race(args)
        result = json.loads((out / "gemm-gaussian.json").read_text())
        self.assertEqual(code, 0 if expected_status == "ok" else 1)
        row = result["arms"]["ours"]
        self.assertEqual(row["status"], expected_status)
        self.assertEqual(result["rounds_requested"], 1)
        return result, row

    def test_conductor_completes_one_scored_round_and_retains_outputs(self):
        result, row = self.conductor(FAKE_WORKER, "ok")
        self.assertEqual(len(row["ms"]), 1)
        self.assertEqual(len(row["operations"]), 2)
        self.assertEqual(len(row["digests"]), 2)
        self.assertEqual(len(row["state_receipts"]), 1)
        self.assertIsNotNone(row["median_ms"])
        self.assertTrue(Path(row["outputs_npz"]).is_file())
        self.assertTrue(Path(result["inputs_npz"]).is_file())
        self.assertEqual(result["quality"]["ours"]["max_rel_err_vs_fp64"], 0)

    def test_conductor_crash_retains_exit_and_zero_scored_coverage(self):
        source = ('import sys; print(\'{"event":"ready","info":{}}\', flush=True); '
                  'sys.stdin.readline(); raise NameError("ab_config regression")')
        result, row = self.conductor(source, "error")
        self.assertEqual(row["error"]["reason"], "worker_exited")
        self.assertEqual(row["error"]["returncode"], 1)
        self.assertEqual(row["worker_process"]["returncode_before_cleanup"], 1)
        self.assertEqual(row["worker_process"]["returncode_after_cleanup"], 1)
        self.assertEqual(row["failed_round"], 0)
        self.assertEqual(row["ms"], [])
        self.assertIsNone(row["median_ms"])
        self.assertNotIn("warmup_deadline_s", row)
        self.assertTrue(Path(result["inputs_npz"]).is_file())


if __name__ == "__main__":
    unittest.main()
