"""The two-GPU checks are not verify commands and not in the wheel (Andrew, 2026-11-26: everything verify ships is one
GPU only). `verify-distributed` and `verify-cross-validation` are not registered on the CLI; their code is the
multi-GPU tools tools/multi_gpu/verify_distributed.py and verify_parallel_cv.py.
"""
import importlib.util

import pytest
from mojolearn import __main__ as cli


@pytest.mark.parametrize("command", ["verify-distributed", "verify-cross-validation"])
def test_two_gpu_commands_are_not_verify_commands(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args([command, "--devices", "0,1"])
    assert exc.value.code == 2


@pytest.mark.parametrize("name", ["mojolearn._verify_distributed", "mojolearn._verify_parallel_cv",
                                  "mojolearn._parallel_cv_witness"])
def test_the_two_gpu_modules_are_not_in_the_package(name):
    assert importlib.util.find_spec(name) is None


@pytest.mark.parametrize("command", ["verify-distributed-program", "verify-partitioned-program"])
def test_single_device_program_commands_take_no_device_list(command):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args([command, "--devices", "0,1", "--output", "r.json"])
    assert exc.value.code == 2
    args = cli.build_parser().parse_args([command, "--output", "r.json"])
    assert not hasattr(args, "devices")
