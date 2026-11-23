# SPDX-License-Identifier: Apache-2.0
"""DEVIATION 3120: the `into=` forms of `ParallelByteLanguageModelTrainer`'s
exports (`export_raw`, `export_gradients`, `fold_export`).

The earlier host-mock tests injected a Python stand-in binding through the
trainer's `_entries` dictionary. That route no longer exists: every byte-LM
trainer keeps one native owner record and drives the binding's private
schedule descriptor table, and the native owner refuses Python-supplied
callables or storage. These source checks pin the native contract instead;
the device proof is the H100 leg in
bench/results/byte_lm_export_fast_2026-09-25/."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from boundary_keys_plain import plain  # noqa: E402


def _read(rel):
    # Held-key doors (bindings/parallel_keys_unpack.mojo) read in their by-name form.
    return plain(rel, (ROOT / rel).read_text(encoding="utf-8"))


def test_owner_has_no_instance_callables():
    owner = _read("bindings/parallel_byte_native_door.mojo")
    assert "_entries" not in owner
    assert "py_call_typed" not in owner
    assert '"byte_lm_schedule_api_v1"' in owner
    # The session operations are typed RUN (no Python object, no Python
    # import); the owner door only unpacks, calls them and packs.
    run = (ROOT / "bindings/parallel_byte_native_run.mojo").read_text(encoding="utf-8")
    assert "neural_descriptor_record_execute(" in run
    assert "PythonObject" not in run and "std.python" not in run
    assert "parallel_session_step_begin(" in owner
    assert "neural_descriptor_execute(" not in owner
    assert "def parallel_byte_admit_class(" in owner
    assert 'py_get(py_fields(owner), "_native")' in owner


def test_into_targets_are_refused_natively_before_the_session():
    owner = _read("bindings/parallel_byte_native_door.mojo")
    for message in ("export target has wrong dtype",
                    "export target must have the exact contiguous one-dimensional shape",
                    "export target is read-only",
                    "into must be a dict with parameters, m, v and flags"):
        assert message in owner, message
    export = owner[owner.index("def _parallel_export("):owner.index("def parallel_byte_state(")]
    assert export.index("_parallel_target(") < export.index("parallel_byte_open(owner)")


def test_schedule_descriptor_exports():
    binding = _read("bindings/_mojolearn_byte_lm.mojo")
    for name in ("byte_lm_schedule_api_v1", "byte_lm_schedule_create_v1", "byte_lm_schedule_handle_v1"):
        assert f'("{name}")' in binding, name
    # lane rs-embedding-bytelm: the schedule table is RUN code beside the door.
    run = _read("training/byte_lm_door_run.mojo")
    assert 'comptime BYTE_LM_SCHEDULE_ABI_NAME = "mojolearn.training.byte_lm_schedule.v1"' in run


def test_schedule_is_one_constructor_option():
    # The pooled and offloaded schedules are options of the one trainer class
    # (lane neural-consolidate, 2026-11-22): their wrapper modules are gone and
    # the schedule is resolved once, at construction, into the native owner.
    for rel in ("python/mojolearn/model_pool_training.py", "python/mojolearn/offload_training.py"):
        assert not (ROOT / rel).exists(), rel
    source = _read("python/mojolearn/parallel_training.py")
    assert 'runtime("admit_class", (ParallelByteLanguageModelTrainer,))' in source
    assert 'schedule="replica"' in source
    owner = _read("bindings/parallel_byte_native_door.mojo")
    init = owner[owner.index("def parallel_byte_init("):owner.index("def _parallel_mirror_session(")]
    assert "_parallel_schedule_index(String(schedule_name))" in init
    record = owner[owner.index("def _parallel_record("):owner.index("def _parallel_int(")]
    assert "schedule" not in record
