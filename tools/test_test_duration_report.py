import json

import test_duration_report as report


def _write(directory, name, tests):
    (directory / name).write_text(json.dumps({"tests": tests}), encoding="utf-8")


def test_report_sums_files_and_estimates_the_loadfile_wall(tmp_path, capsys):
    _write(tmp_path, "durations-1.json", {
        "a/test_one.py::test_x": {"seconds": 30.0, "outcome": "passed"},
        "a/test_one.py::test_y": {"seconds": 10.0, "outcome": "failed"},
        "a/test_two.py::test_z": {"seconds": 20.0, "outcome": "passed"},
    })
    _write(tmp_path, "durations-2.json", {"a/test_three.py::test_s": {"seconds": 5.0, "outcome": "skipped"}})
    tests = report.load(tmp_path)
    assert len(tests) == 4
    assert report.file_totals(tests) == {"a/test_one.py": 40.0, "a/test_two.py": 20.0, "a/test_three.py": 5.0}
    assert report.loadfile_wall(report.file_totals(tests), 2) == 40.0
    assert report.loadfile_wall(report.file_totals(tests), 1) == 65.0
    out = tmp_path / "all.tsv"
    assert report.main([str(tmp_path), "--workers", "2", "--top", "2", "--tsv", str(out)]) == 0
    text = capsys.readouterr().out
    assert "tests 4  test-seconds 65" in text and "failed        1 tests       10 s" in text
    assert out.read_text().splitlines()[0] == "30.000\tpassed\ta/test_one.py::test_x"


def test_report_refuses_an_empty_directory(tmp_path):
    assert report.main([str(tmp_path)]) == 2
