"""A failed generated-source preparation names its cause and its generator log's tail in the error itself, because
the evidence directory it points at stays on the build box (lane fb9-arm-builds, 2026-11-17: the nv-fma-asm arm's
assemble job failed with only "preparation failed for core/archive_unicode_names_generated.mojo; see <runner path>").

CPU only, no network, no compiler.

    python3 -m unittest tools/tests/test_generated_sources_failure.py
"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
import generated_sources as gs  # noqa: E402


@patch.object(gs, "GENERATOR_RETRY_SECONDS", 0)
class FailureNamesItsCause(unittest.TestCase):
    def test_a_failed_generator_run_is_retried(self):
        """A transient failure (a refused download) is retried; the run that succeeds is then verified as usual."""
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for name in gs.GENERATED[gs.ARCHIVE_TABLE]:
                (repo / name).parent.mkdir(parents=True, exist_ok=True)
                (repo / name).write_text("# input\n")
            runs = []

            def runner(command, repo, env, log):
                runs.append(command)
                Path(log).write_text("run %d\n" % len(runs))
                return 6 if len(runs) < 3 else 0

            with self.assertRaises(gs.GeneratedSourceError) as caught:    # the stub writes no output: not verified
                gs.prepare_sources(repo, repo / "evidence", [gs.ARCHIVE_TABLE], runner=runner)
            self.assertEqual(len(runs), 3)
            self.assertIn("generator did not create", str(caught.exception))
            self.assertTrue((repo / "evidence" / (gs.LOG_NAMES[gs.ARCHIVE_TABLE] + ".attempt1")).is_file())

    def test_the_error_carries_the_generator_exit_and_log_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp)
            for name in gs.GENERATED[gs.ARCHIVE_TABLE]:
                (repo / name).parent.mkdir(parents=True, exist_ok=True)
                (repo / name).write_text("# input\n")

            def runner(command, repo, env, log):
                Path(log).write_text("downloading\ncurl: (6) Could not resolve host: www.unicode.org\n")
                return 6

            with self.assertRaises(gs.GeneratedSourceError) as caught:
                gs.prepare_sources(repo, repo / "evidence", [gs.ARCHIVE_TABLE], runner=runner)
            text = str(caught.exception)
            self.assertIn(gs.ARCHIVE_TABLE, text)
            self.assertIn("generator exited 6", text)
            self.assertIn("Could not resolve host", text)


if __name__ == "__main__":
    unittest.main()
