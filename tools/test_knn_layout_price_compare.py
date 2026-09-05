"""Main-run synthetic negative controls; no GPU or benchmark execution."""
from pathlib import Path
import tempfile
import unittest

from knn_layout_price_compare import ARMS, compare_directories, parse_check, parse_price
from test_knn_smallk_price_compare import fixture


def check_fixture(arm):
    selector, transpose = ARMS[arm]
    rows = [f"LAYOUT_FLAGS mode IDENTICAL selector {selector} transpose {transpose}",
            f"SMALLK_DISPATCH experimental {selector} mode IDENTICAL specialized_k 8,10,16 fallback_k 4,15"]
    for profile, n in (("dyadic", 257), ("duplicates", 1025)):
        for q in (1, 257, 1000):
            for k in (4, 8, 10, 15, 16):
                key = f"{profile} {n} {q} 8 {k}"
                rows.append(f"DISPATCH_CALL {key} 0 tile {min(q,256)}")
                rows.extend(f"DISPATCH_CELL {key} {i} 1065353216 {i % k}" for i in range(q*k))
                rows.extend((f"DISPATCH_CALL {key} 1 tile {min(q,128)}", f"DISPATCH_FIXTURE_PASS {key}"))
    rows.append("KNN SMALL-K PUBLIC DISPATCH PASS fixtures 30 selected_pairs 133348")
    for metric in (0, 1, 3, 2):
        rows.extend(f"LAYOUT_CELL {metric} 65 257 17 10 {i} 1065353216 {i % 10}" for i in range(2570))
    rows.append("KNN LAYOUT PUBLIC DISPATCH PASS metric_fixtures 4 additional_selected_pairs 10280")
    return "\n".join(rows) + "\n"


def price_fixture(q, arm):
    selector, transpose = ARMS[arm]
    return (f"LAYOUT_FLAGS mode IDENTICAL selector {selector} transpose {transpose}\n" +
            fixture(q, "experimental" if selector else "legacy", 4 if arm == "baseline" else 2))


def campaign(root):
    root.mkdir()
    statuses = []
    for arm in ARMS:
        for kind in ("check", "price"):
            name = f"build-{kind}-{arm}"
            (root / (name + ".log")).write_text("synthetic successful build\n")
            statuses.append(name + "\t0")
        name = f"check-{arm}"
        (root / (name + ".log")).write_text(check_fixture(arm))
        statuses.append(name + "\t0")
        for q in (32, 128, 1000):
            for r in range(9):
                name = f"q{q}-r{r}-{arm}"
                (root / (name + ".log")).write_text(price_fixture(q, arm))
                statuses.append(name + "\t0")
    (root / "status.tsv").write_text("\n".join(statuses) + "\n")
    (root / "completion.txt").write_text("COMPLETE\n")


class LayoutEvidenceTests(unittest.TestCase):
    def test_activation_order_and_completeness(self):
        original = check_fixture("both")
        cell = "LAYOUT_CELL 0 65 257 17 10 0 1065353216 0\n"
        variants = (
            original.replace("transpose 1", "transpose 0"),
            original.replace(cell, ""), original.replace(cell, cell + cell),
            original.replace(cell, cell.replace(" 10 0 ", " 10 1 ")),
            original.rsplit("KNN LAYOUT", 1)[0],
            original.replace(cell, cell.replace("1065353216", "2139095040")),
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "check.log"
            path.write_text(original)
            parse_check(path, "both")
            for index, text in enumerate(variants):
                with self.subTest(variant=index):
                    path.write_text(text)
                    with self.assertRaises(ValueError): parse_check(path, "both")

    def test_price_activation_and_invalid_times(self):
        original = price_fixture(32, "transpose")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "price.log"
            path.write_text(original)
            parse_price(path, 32, "transpose")
            for text in (original.replace("transpose 1", "transpose 0"),
                         original.replace(" 2 used_tile", " nan used_tile"),
                         original.replace(" 2 used_tile", " 0 used_tile")):
                path.write_text(text)
                with self.assertRaises(ValueError): parse_price(path, 32, "transpose")

    def test_campaign_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "vendor"
            campaign(root)
            result = compare_directories(root, root)
            self.assertEqual(result["cross_vendor_bits"], "PASS")
            self.assertEqual(result["primary"]["shapes"]["1000"]["paired_baseline_over_arm"]["both"]["median"], 2)
            status = root / "status.tsv"
            original = status.read_text()
            status.write_text(original + "check-both\t0\n")
            with self.assertRaises(ValueError): compare_directories(root)
            status.write_text(original)
            price = root / "q32-r0-both.log"
            price.write_text(price.read_text().replace(" 0 1065353216 0\n", " 0 1065353217 0\n"))
            with self.assertRaisesRegex(ValueError, "bytes differ"): compare_directories(root)
            (root / "build-price-both.log").unlink()
            with self.assertRaisesRegex(ValueError, "logs"): compare_directories(root)


if __name__ == "__main__":
    unittest.main()
