"""Synthetic evidence controls; main lane runs these without a GPU."""
from pathlib import Path
import tempfile
import unittest

from knn_smallk_price_compare import ARMS, QUERIES, ROUNDS, compare_directories, parse_log


def fixture(q, arm, milliseconds=2):
    lines = [f"SMALLK_PRICE experimental {ARMS.index(arm)} mode IDENTICAL fixture dyadic-v1 "
             f"index 100000 queries {q} features 32 k 10 index_salt 0 query_salt 593 "
             "requested_tile 256 warmups 2 timed_calls 1 "
             "scope native-public-upload-search-download-synchronize"]
    lines += [f"PRICE_CELL dyadic-v1 100000 {q} 32 10 {cell} 1065353216 {cell % 10}"
              for cell in range(q * 10)]
    lines += [f"PRICE_MS dyadic-v1 100000 {q} 32 10 {milliseconds} used_tile {min(q, 256)}",
              f"KNN SMALL-K PUBLIC PRICE PASS queries {q} selected_pairs {q * 10}"]
    return "\n".join(lines) + "\n"


def campaign(root):
    root.mkdir()
    (root / "completion.txt").write_text("COMPLETE\n")
    statuses = ["build-legacy\t0", "build-experimental\t0"]
    for q in QUERIES:
        for r in ROUNDS:
            for arm in ARMS:
                name = f"q{q}-r{r}-{arm}"
                (root / f"{name}.log").write_text(fixture(q, arm, 4 if arm == "legacy" else 2))
                statuses.append(name + "\t0")
    (root / "status.tsv").write_text("\n".join(statuses) + "\n")


class PriceEvidenceTests(unittest.TestCase):
    def test_complete_campaign_and_paired_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vendor"
            campaign(root)
            result = compare_directories(root, root)
            self.assertEqual(result["cross_vendor_bits"], "PASS")
            self.assertEqual(result["primary"]["log_count"], 54)
            shape = result["primary"]["shapes"]["1000"]
            self.assertEqual(shape["paired_legacy_over_experimental"]["median"], 2)
            self.assertEqual(shape["milliseconds"]["legacy"]["iqr"], 0)
            self.assertEqual(len(shape["samples"]["experimental"]), 9)

    def test_malformed_individual_evidence(self):
        original = fixture(32, "legacy")
        cell = "PRICE_CELL dyadic-v1 100000 32 32 10 0 1065353216 0\n"
        timing = "PRICE_MS dyadic-v1 100000 32 32 10 2 used_tile 32\n"
        variants = {
            "wrong activation": original.replace("experimental 0", "experimental 1"),
            "wrong mode": original.replace("IDENTICAL", "FAST"),
            "missing cell": original.replace(cell, ""),
            "duplicate cell": original.replace(cell, cell + cell),
            "nonfinite": original.replace(cell, cell.replace("1065353216", "2139095040")),
            "negative": original.replace(cell, cell.replace("1065353216", "3212836864")),
            "bounds": original.replace(cell, cell.replace("1065353216 0", "1065353216 100000")),
            "wrong fixture": original.replace("100000", "100001"),
            "no timing": original.replace(timing, ""),
            "duplicate timing": original.replace(timing, timing + timing),
            "nan timing": original.replace(timing, timing.replace(" 2 used", " nan used")),
            "zero timing": original.replace(timing, timing.replace(" 2 used", " 0 used")),
            "invalid tile": original.replace("used_tile 32", "used_tile 257"),
            "no pass": original.rsplit("KNN SMALL-K", 1)[0],
            "postpass failure": original + "GPU ERROR\n",
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "case.log"
            for name, content in variants.items():
                with self.subTest(name=name):
                    path.write_text(content)
                    with self.assertRaises(ValueError):
                        parse_log(path, 32, "legacy")

    def test_incomplete_or_changed_campaign_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vendor"
            campaign(root)
            path = root / "q128-r8-experimental.log"
            original = path.read_text()
            path.unlink()
            with self.assertRaisesRegex(ValueError, "54"):
                compare_directories(root)
            path.write_text(original.replace(" 0 1065353216 0\n", " 0 1065353217 0\n"))
            with self.assertRaisesRegex(ValueError, "bytes differ"):
                compare_directories(root)
            path.write_text(original)
            status = root / "status.tsv"
            text = status.read_text()
            status.write_text(text.replace("q32-r0-legacy\t0", "q32-r0-legacy\t134"))
            with self.assertRaisesRegex(ValueError, "unsuccessful"):
                compare_directories(root)
            status.write_text(text)
            (root / "completion.txt").write_text("PARTIAL\n")
            with self.assertRaisesRegex(ValueError, "completion"):
                compare_directories(root)

    def test_consistent_vendor_difference_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            first, second = Path(tmp) / "first", Path(tmp) / "second"
            campaign(first)
            campaign(second)
            for path in second.glob("q32-*.log"):
                path.write_text(path.read_text().replace(" 0 1065353216 0\n", " 0 1065353217 0\n"))
            with self.assertRaisesRegex(ValueError, "cross-vendor"):
                compare_directories(first, second)


if __name__ == "__main__":
    unittest.main()
