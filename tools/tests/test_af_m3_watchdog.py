import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("watchdog", Path(__file__).resolve().parents[1] / "af_m3_watchdog.py")
watchdog = importlib.util.module_from_spec(spec)
spec.loader.exec_module(watchdog)


class RestartGuardTest(unittest.TestCase):
    home = Path("/Users/ec2-user")
    queue = ["CMD lane/apple-fast current true", "CMD lane/apple-fast next true"]

    def test_nested_runner_shell_is_not_a_second_runner(self):
        ps = watchdog.process_table("10 1 0:01 bash /Users/ec2-user/mq.sh\n11 10 0:00 bash /Users/ec2-user/mq.sh")
        self.assertEqual(watchdog.classify(ps, self.home, 1, self.queue), ([10], [], False))

    def test_orphan_worker_prevents_restart(self):
        for command in ("python tools/bench_board_algos.py race", "mojo build -j 2 bindings/x.mojo", "bash tools/aft_ab.sh x"):
            ps = watchdog.process_table("10 1 0:01 " + command)
            self.assertFalse(watchdog.classify(ps, self.home, 1, self.queue)[2])

    def test_empty_queue_and_explicit_stop_prevent_restart(self):
        self.assertFalse(watchdog.classify({}, self.home, 2, self.queue)[2])
        self.assertFalse(watchdog.classify({}, self.home, 1, ["STOP", self.queue[1]])[2])

    def test_missing_runner_with_pending_work_can_restart(self):
        self.assertTrue(watchdog.classify({}, self.home, 1, self.queue)[2])


if __name__ == "__main__":
    unittest.main()
