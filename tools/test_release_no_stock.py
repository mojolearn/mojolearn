"""release.no_stock sees RunPod's "no instances" answer where the legs keep it:
the leg's log says only "create returned HTTP 500"; RunPod's words are in
create_response.json under the leg's directory (0.8.19, 2026-09-25: both
NVIDIA legs failed on stock and never walked to the next GPU type)."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

sys.argv = [sys.argv[0]]
SPEC = importlib.util.spec_from_file_location("release", Path(__file__).resolve().parent / "release.py")
rel = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rel)

BODY = '{"error":"create pod: There are no instances currently available","status":500}'


class NoStock(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = self.tmp / "cuda-sm_89.log"
        self.log.write_text("  create returned HTTP 500 and no id this script could parse.\n")
        self.leg = self.tmp / "cuda-sm_89"
        self.leg.mkdir()

    def test_the_log_alone_does_not_say_it(self):
        (self.leg / "create_response.json").write_text(BODY)
        self.assertFalse(rel.no_stock(self.log))

    def test_the_leg_directory_says_it(self):
        (self.leg / "create_response.json").write_text(BODY)
        self.assertTrue(rel.no_stock(self.log, self.leg))

    def test_nested_and_missing_dirs(self):
        (self.leg / "a").mkdir()
        (self.leg / "a" / "create_response.json").write_text(BODY)
        self.assertTrue(rel.no_stock(self.log, None, self.tmp / "nope", self.leg))

    def test_another_failure_is_not_no_stock(self):
        (self.leg / "create_response.json").write_text('{"error":"unauthorized","status":401}')
        self.assertFalse(rel.no_stock(self.log, self.leg))

    def test_the_log_still_counts(self):
        self.log.write_text("There are no instances currently available\n")
        self.assertTrue(rel.no_stock(self.log))


if __name__ == "__main__":
    unittest.main()
