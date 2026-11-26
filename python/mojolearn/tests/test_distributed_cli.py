"""The two-GPU checks are not verify commands (Andrew, 2026-11-26: everything verify ships is one GPU only).

`verify-distributed` and `verify-cross-validation` are not registered on the CLI; their modules keep their own
`main(argv)` for the multi-GPU tools (`python -m mojolearn._verify_distributed`, `._verify_parallel_cv`).
"""
import pytest
from mojolearn import __main__ as cli, _verify_distributed as proof, _verify_parallel_cv as cv


@pytest.mark.parametrize("command", ["verify-distributed", "verify-cross-validation"])
def test_two_gpu_commands_are_not_verify_commands(command, capsys):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args([command, "--devices", "0,1"])
    assert exc.value.code == 2


def test_the_two_gpu_modules_keep_their_own_entry_points():
    assert callable(proof.main) and callable(cv.main)


@pytest.mark.parametrize("command", ["verify-distributed-program", "verify-partitioned-program"])
def test_single_device_program_commands_take_no_device_list(command):
    with pytest.raises(SystemExit) as exc:
        cli.build_parser().parse_args([command, "--devices", "0,1", "--output", "r.json"])
    assert exc.value.code == 2
    args = cli.build_parser().parse_args([command, "--output", "r.json"])
    assert not hasattr(args, "devices")
