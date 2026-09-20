# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""`python -m mojolearn verify --par`: the two-device column, on the user's box.

THE CLAIM THIS COMMAND STATES. `identity_break._par_devices` says it in one
sentence: a `par-*` lane's two-device column "must hash equal, cell for cell,
to the one-device column of the same commit; that equality is the drivers'
whole claim". Nothing else in the shipped verifier states it. `verify --all`
compares this box against our recorded table and every `par-*` cell in that
table is a ONE-device run (`identity_break.RECORD_EXCLUDED_PREFIXES`), so a
release record cannot carry the claim and never has.

NO REFERENCE TABLE IS INVOLVED, and that is the point. Both sides are produced
here, in this process, minutes apart, off one build: the local ONE-device
column IS the reference. `_verify_reference.admit(..., par_axis=True)` exists
for exactly this reason, and this command needs neither it nor the table.

HOW THE COLUMN IS FLIPPED. Two `run_cell` calls with `MOJOLEARN_PAR_DEVICES`
set to `"0"` and then to `"0,1"`. That is enough and it was checked rather
than assumed: `identity_break._par_devices()` reads the variable at FIT time,
inside the lane body, and `_parallel_pool.DevicePool._start` builds each
worker's environment from `dict(os.environ, ...)` at spawn time, so a pool
created after the variable changes carries the new value. No re-import, no
second process, no reference regeneration.

WHY A SILENT ONE-DEVICE FALLBACK IS THE FAILURE THIS GUARDS AGAINST. If the
two-device column quietly runs on one device, every hash agrees, the command
prints AGREE over 59 lanes and has checked nothing. This repo has shipped
that failure once already. So the two-device column is watched: every
`DevicePool` started while it runs is inventoried through `_gpu_witness`, and
the run is REFUSED unless at least one pool started, every pool that started
asked for the requested device group, and each worker resolved to its own
process and (on CUDA and HIP) its own physical GPU by UUID and PCI id. A
column that reports AGREE without a witness is not reported at all.

APPLE IS EXCLUDED STRUCTURALLY, NOT BY POLICY. `DevicePool._start` admits the
metal vendor only at `group == (0,)` and raises "device selection is
unavailable for this vendor/device group" for anything else. There is no Mac,
present or future, on which a two-device `par-*` column can be produced by
this code, so the refusal says so by name rather than sending someone to look
for hardware that would never have worked.

WHAT A CPU-ONLY INSTALL GETS. `_parallel_pool.CPU_OPERATIONS` declares a real
CPU route for the drivers whose split and merge live in Python, and on that
route a "device" is one worker PROCESS with no device selection at all. This
command runs there and says so in those words: it checks the drivers' own
partition and merge across two processes and it is NOT a device claim. Lanes
whose operation has no CPU route refuse BY NAME and are listed, never hidden.
"""
import contextlib
import json
import os
import sys
import time

from ._verify_all import CannotRun, EXIT_CANNOT_RUN, EXIT_MISMATCH, EXIT_USAGE, EXIT_VERIFIED
from ._verify_all import _emit, _finish, load_harness, run_cell, family_map
from . import _verify_reference as vref

#: the variable `identity_break._par_devices()` reads at fit time
PAR_DEVICES_ENV = "MOJOLEARN_PAR_DEVICES"

#: the prefix that names a multi-device driver lane. The same constant
#: `identity_break.RECORD_EXCLUDED_PREFIXES` uses to keep them OUT of a
#: release record, which is why they need this command.
PAR_PREFIX = "par-"

#: the two-device group. Two is what the claim is about and what every
#: two-device leg has ever recorded; `--par-devices` overrides it.
DEFAULT_PAR_DEVICES = (0, 1)

#: a `--par quick` column, one lane per driver family, is seconds per lane;
#: the default is every `par-*` lane on the base fixture.
QUICK_FIXTURE = "base"


# --------------------------------------------------------------------------
# which lanes
# --------------------------------------------------------------------------

def par_lanes(harness, scope="default"):
    """(chosen, every, per_family) for a `--par` scope.

    Derived from `harness.LANES` at call time, never hand listed, so a lane
    added tomorrow is in the default column tomorrow. `quick` is one lane per
    family (families come from the same `_verify_all.family_map` the rest of
    the verifier uses), `default` and `all` are every `par-*` lane; `all`
    differs only in the fixtures the caller then runs.
    """
    every = sorted(n for n in harness.LANES if n.startswith(PAR_PREFIX))
    fam = family_map(every)
    per_family = {}
    for lane in every:
        per_family.setdefault(fam[lane], lane)
    if scope == "quick":
        return sorted(per_family.values()), every, dict(per_family)
    if scope in ("default", "all"):
        return every, every, dict(per_family)
    raise CannotRun(f"--par scope {scope!r} is not one of quick, default, all")


# --------------------------------------------------------------------------
# the guard against a silent one-device fallback
# --------------------------------------------------------------------------

class PoolWitness:
    """Every `DevicePool` started inside the `watching()` block.

    `_verify_distributed.start_pool` proved this shape on the distributed
    capture: wrap `DevicePool._start`, and for each pool that actually spawns
    workers, ask each worker for its own `device_inventory` before any real
    request reaches it. What is lifted here is that, plus the rule that made
    it worth anything, `_gpu_witness.require_distinct_workers`: one visible
    physical GPU per worker, no repeated UUID, PCI id or pid.

    THE NEW PART IS THE EMPTY CASE. A driver that never starts a pool, or
    starts one at a single device, produces a perfectly stable column that
    agrees with the one-device column in one second and says nothing. That is
    the failure this class exists to name, so `refusal()` reports it first.
    """

    def __init__(self, vendor, devices):
        self.vendor = vendor
        self.devices = tuple(devices)
        self.pools = []
        self.errors = []

    # -- the placement rule -------------------------------------------------

    def _admit(self, pool, inventory):
        """Refuse by name unless this pool's workers sit where they were asked
        to sit. A COOPERATIVE pool is one worker with the whole group visible
        to it, so `require_distinct_workers`, which demands one device per
        worker at local ordinal zero, is the wrong rule there and is not used.
        """
        from ._gpu_witness import require_distinct_workers
        asked = tuple(pool.devices)
        if asked != self.devices:
            raise RuntimeError(f"a pool ran on devices {list(asked)} while the column asked for "
                               f"{list(self.devices)}")
        if self.vendor not in ("cuda", "hip"):
            # The CPU route has no device identity to check. Distinct worker
            # PROCESSES is the whole of what it can say, and it says only that.
            pids = [record["pid"] for record in inventory]
            if len(set(pids)) != len(pids) or any(type(p) is not int or p < 1 for p in pids):
                raise RuntimeError("two logical workers share one process")
            return
        if getattr(pool, "cooperative", False):
            if len(inventory) != 1:
                raise RuntimeError("a cooperative pool started more than one worker")
            seen = inventory[0].get("devices") or []
            if len(seen) != len(asked):
                raise RuntimeError(f"the cooperative worker sees {len(seen)} device(s), not "
                                   f"{len(asked)}; this column ran on fewer devices than it names")
            if len({d.get("uuid") for d in seen}) != len(seen):
                raise RuntimeError("the cooperative worker sees one physical GPU twice")
            return
        require_distinct_workers(inventory, self.vendor, len(asked))

    # -- the wrapper --------------------------------------------------------

    @contextlib.contextmanager
    def watching(self):
        from ._parallel_pool import DevicePool
        original_start, original_call = DevicePool._start, DevicePool._call

        def start(pool):
            started = bool(pool._workers)
            original_start(pool)
            if started:
                return
            try:
                if self.vendor in ("cuda", "hip"):
                    # ASK THE WORKER, not the parent. `device_inventory` runs
                    # after that worker's visibility mask is set, which is the
                    # only place the answer means anything.
                    inventory = [original_call(worker, ("device_inventory", None, ()))
                                 for worker in pool._workers]
                    for worker, record in zip(pool._workers, inventory):
                        if record.get("pid") != worker.pid:
                            raise RuntimeError("worker inventory pid differs from the launched "
                                               "process")
                else:
                    # The CPU route has no driver to ask: `visible_gpu_inventory`
                    # raises NotImplementedError there BY NAME, and sending it
                    # anyway kills the column with an error about the witness
                    # rather than about the work (measured, 2026-09-20).
                    inventory = [dict(kind="process-only", pid=worker.pid, devices=[])
                                 for worker in pool._workers]
                self._admit(pool, inventory)
            except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                self.errors.append(f"{type(exc).__name__}: {exc}")
                raise
            self.pools.append(dict(devices=list(pool.devices),
                                   cooperative=bool(getattr(pool, "cooperative", False)),
                                   workers=[dict(pid=r.get("pid"),
                                                 devices=[d.get("uuid") for d in (r.get("devices") or [])],
                                                 visibility=r.get("visibility", {}))
                                            for r in inventory]))

        DevicePool._start = start
        try:
            yield self
        finally:
            DevicePool._start = original_start

    # -- the verdict --------------------------------------------------------

    def refusal(self):
        """None when the two-device column was genuinely produced on the
        devices it names, else the sentence that says it was not."""
        if self.errors:
            return "a worker was not where the column said it was: " + self.errors[0]
        if not self.pools:
            return ("the two-device column started NO device pool at all, so it ran on one "
                    "device and agreeing with the one-device column proves nothing. Either "
                    "the lane does not shard on this install or the driver fell back "
                    "silently; both are failures of this check, not passes")
        bad = [p for p in self.pools if tuple(p["devices"]) != self.devices]
        if bad:
            return f"a pool ran on devices {bad[0]['devices']}, not {list(self.devices)}"
        return None

    def summary(self):
        return dict(pools=len(self.pools),
                    workers=sum(len(p["workers"]) for p in self.pools),
                    devices=list(self.devices),
                    placement=("physical" if self.vendor in ("cuda", "hip") else "process-only"),
                    detail=self.pools)


# --------------------------------------------------------------------------
# the two columns
# --------------------------------------------------------------------------

@contextlib.contextmanager
def par_devices(devices):
    """`MOJOLEARN_PAR_DEVICES` set to this group for the block, restored after.

    The value is what `identity_break._par_devices()` parses, so this is the
    one place the column is chosen and it is chosen the same way a two-device
    leg chooses it on a rented box.
    """
    previous = os.environ.get(PAR_DEVICES_ENV)
    os.environ[PAR_DEVICES_ENV] = ",".join(str(int(d)) for d in devices)
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(PAR_DEVICES_ENV, None)
        else:
            os.environ[PAR_DEVICES_ENV] = previous


def _is_na(value):
    return isinstance(value, str) and value.startswith("n/a")


#: The sentence `_parallel_pool._cpu_refusal` raises when the CPU route has
#: no implementation of an operation, or has one only at a single device
#: (`CPU_SINGLE_DEVICE_COOPERATIVE`, which is `mlp_update` and `samba_update`).
#: It is a DECLARED limit of that route, written down in that module's
#: docstring, and it fires only on a CPU-only install.
CPU_ROUTE_REFUSAL = "no CPU implementation of"


def compare_parts(one, two, cpu_route=False):
    """[(part, verdict, one, two)] for one cell, in `_verify_reference.PARTS`
    order.

    IDENTICAL       both sides produced the same hash
    DIVERGENT       both produced a hash and the hashes differ. Sharding
                    changed the bits, which is the strongest thing this
                    command can find and it gates.
    MOVED           a side disagreed with itself across repeats
    ONE-COLUMN      one side produced a hash and the other REFUSED. This is
                    the `par-queries-nn` batch shape: a real shipped defect
                    that only the two-device column can see, and it gates.
    CPU-ROUTE-LIMIT the two-device side refused with the CPU route's OWN
                    by-name sentence. Measured 2026-09-20 on a CPU-only
                    install: par-mlp, par-samba and par-samba-clip refuse
                    above one device because `mlp_update` and `samba_update`
                    split their gradient columns inside the GPU binding and no
                    host binding restates that. Reading those three as the
                    same finding as `par-queries-nn` would bury the one real
                    defect under three declared limits, so they are named
                    separately and do not gate. On a GPU install this verdict
                    cannot occur: `_cpu_refusal` never fires there.
    REFUSED         both sides refused, the same way. Not a difference, so
                    not a gate; the reason is printed.
    N/A             the lane declares the part absent on both sides
    """
    rows = []
    for part in vref.PARTS:
        (v1, e1), (v2, e2) = one.get(part, (None, "missing")), two.get(part, (None, "missing"))
        if _is_na(v1) and _is_na(v2) and v1 == v2:
            verdict = "N/A"
        elif v1 is None and v2 is None:
            verdict = "REFUSED"
        elif v1 is None or v2 is None:
            missing_error = str(e2 if v2 is None else e1)
            verdict = ("CPU-ROUTE-LIMIT" if cpu_route and CPU_ROUTE_REFUSAL in missing_error
                       else "ONE-COLUMN")
        elif "MOVED" in str(v1) or "MOVED" in str(v2):
            verdict = "MOVED"
        elif v1 == v2:
            verdict = "IDENTICAL"
        else:
            verdict = "DIVERGENT"
        rows.append(dict(part=part, verdict=verdict, one=v1, two=v2,
                         one_error=e1, two_error=e2))
    return rows


#: the verdicts that make `--par` exit non-zero. A DIVERGENT means sharding
#: changed the bits. A ONE-COLUMN means the work refused on exactly one of the
#: two columns, which is how `par-queries-nn`'s batch part has been failing on
#: two AMD devices since 2026-09-19 while every release record read clean.
GATING = ("DIVERGENT", "MOVED", "ONE-COLUMN")

#: the states this command can end in, worst first. It is `_verify_all.verdict`'s
#: ladder, applied to the two columns instead of to the reference table.
PAR_STATES = ("MISMATCH", "INCOMPLETE", "NOTHING COMPARED", "VERIFIED")


def par_state(counts, compared):
    """`(state, passed)` from the per-verdict counts of every compared part.

    A REFUSED PART DID NOT RUN, AND A RUN THAT RAN NOTHING IS NOT A PASS.
    Until 2026-09-20 this was `passed = None if not cells else (gated == 0)`,
    and `cells` is appended once per PART whatever the part read, so `None`
    could only mean that the LANE LIST was empty. Every other way of running
    nothing -- every part REFUSED on both columns, every part CPU-ROUTE-LIMIT,
    every part N/A -- counted zero gating verdicts and printed
    `RESULT: THE TWO COLUMNS AGREE` at exit 0. MEASURED the same day: a
    CPU-only install outside `reference_training()` produced 65 REFUSED parts
    and exited 0 as VERIFIED, and the docstring of `par_check` records that
    exact state as a thing that happens.

    The ladder, which is `_verify_all.verdict`'s and not a new invention:

      MISMATCH          a GATING verdict. A wrong answer outranks an absent
                        one, so this is read first.
      INCOMPLETE        nothing gated, but a part REFUSED on both columns.
                        Absence is not agreement. It is EXIT_CANNOT_RUN, not
                        EXIT_MISMATCH: the part did not disagree, it did not
                        run, and `compare_parts` still says REFUSED is not a
                        difference.
      NOTHING COMPARED  no part produced two hashes to compare. A column of
                        N/A and CPU-ROUTE-LIMIT establishes nothing, and
                        `_verify_all.verdict` already refuses to call that a
                        pass.
      VERIFIED          at least one part was compared and every compared
                        part agreed.

    CPU-ROUTE-LIMIT is deliberately NOT in the INCOMPLETE arm: it is a limit
    the CPU route declares by name in its own docstring, it is reported rather
    than hidden, and reading it as a refusal would bury `par-queries-nn` under
    three declared limits. It cannot make a run VERIFIED either, because only
    an IDENTICAL part can.
    """
    if not compared:
        return "NOTHING COMPARED", None
    if sum(counts.get(v, 0) for v in GATING):
        return "MISMATCH", False
    if counts.get("REFUSED", 0):
        return "INCOMPLETE", None
    if not counts.get("IDENTICAL", 0):
        return "NOTHING COMPARED", None
    return "VERIFIED", True


def par_check(harness, ml, lanes, fixtures, devices=DEFAULT_PAR_DEVICES,
              repeats=1, log=None, perturb=False):
    """The one-device column, then the two-device column, then the comparison.

    `perturb=True` moves every value of the input's first column up by one ULP
    for the TWO-device arm only. It is the `--par-self-test` arm and it exists
    because a comparison never seen to fail is not a comparison: with it on,
    every cell must read DIVERGENT or this machinery is not doing its job.
    """
    import numpy as np
    from . import _backend
    from ._cpu_reference import reference_training
    log = log or (lambda s: None)
    vendor = _backend.vendor()
    cells, counts = [], {}
    cell_witnesses = []
    for lane in lanes:
        for fx in fixtures:
            t0 = time.time()
            # Placement from another lane or fixture cannot certify this cell.
            witness = PoolWitness(vendor, devices)
            X, yc, yr = harness.fixture(fx)
            held = harness.heldout(fx)
            # The same scope `self_test` opens. On a GPU install it changes
            # nothing; on a CPU-only install it is what lets the verifier fit
            # at all, and without it every cell reads REFUSED on BOTH columns
            # (measured, 2026-09-20: 65 refused parts). That run now ends
            # INCOMPLETE at a non-zero exit; until the same day `par_state`
            # closed it, it printed `THE TWO COLUMNS AGREE on 0 compared cell
            # parts` and exited 0.
            with reference_training():
                with par_devices((devices[0],)):
                    one = run_cell(harness, ml, lane, fx, (X, yc, yr), held, repeats)
                data = (X, yc, yr)
                if perturb:
                    Xp = np.array(X, copy=True)
                    Xp[:, 0] = np.nextafter(Xp[:, 0], np.asarray(np.inf, dtype=Xp.dtype))
                    if not (Xp != X).any():
                        raise CannotRun("--par-self-test: the one-ULP perturbation did not change "
                                        "the input at all, so the perturbed arm would prove nothing")
                    data = (Xp, yc, yr)
                with par_devices(devices), witness.watching():
                    two = run_cell(harness, ml, lane, fx, data, held, repeats)
            cell_witnesses.append(dict(lane=lane, fixture=fx,
                                       witness=witness.summary(),
                                       witness_refusal=witness.refusal()))
            secs = round(time.time() - t0, 3)
            rows = compare_parts(one, two, cpu_route=(vendor == "cpu"))
            for row in rows:
                counts[row["verdict"]] = counts.get(row["verdict"], 0) + 1
                cells.append(dict(lane=lane, fixture=fx, seconds=secs, **row))
            shown = " ".join(f"{r['part']}={r['verdict']}" for r in rows
                             if r["verdict"] != "N/A")
            log(f"  {lane:<26} {fx:<8} {shown:<58} {secs:6.2f}s")
    state, passed = par_state(counts, cells)
    failures = [c for c in cell_witnesses if c["witness_refusal"]]
    witness_refusal = (f"{failures[0]['lane']}/{failures[0]['fixture']}: "
                       f"{failures[0]['witness_refusal']}" if failures else None)
    if witness_refusal and state != "MISMATCH":
        state, passed = "INCOMPLETE", None
    # Retain the aggregate report shape for existing readers, but admission
    # above requires a witness for every individual lane/fixture.
    summaries = [c["witness"] for c in cell_witnesses]
    aggregate = dict(pools=sum(s["pools"] for s in summaries),
                     workers=sum(s["workers"] for s in summaries),
                     devices=list(devices),
                     placement="physical" if vendor in ("cuda", "hip") else "process-only",
                     detail=[p for s in summaries for p in s["detail"]])
    return dict(ran=True, vendor=vendor, devices=list(devices), repeats=repeats,
                lanes=sorted({c["lane"] for c in cells}), fixtures=list(fixtures),
                compared=len(cells), counts=counts, cells=cells,
                perturbed=bool(perturb), witness=aggregate,
                cell_witnesses=cell_witnesses, witness_refusal=witness_refusal,
                state=state, passed=passed)


# --------------------------------------------------------------------------
# what the box can run at all
# --------------------------------------------------------------------------

def refuse_to_run(vendor, devices):
    """The sentence that says this box cannot produce a two-device column, or
    None. Structural reasons only; no policy lives here."""
    if vendor == "metal":
        return ("Apple is excluded STRUCTURALLY, not by policy: "
                "`_parallel_pool.DevicePool._start` admits the metal vendor only at "
                "`group == (0,)` and raises 'device selection is unavailable for this "
                "vendor/device group' for any other group. No Mac can produce a "
                "two-device `par-*` column with this code, so there is no machine to go "
                "looking for. Run this on a CUDA or HIP box with two GPUs.")
    if vendor not in ("cuda", "hip", "cpu"):
        return f"unknown vendor read-back {vendor!r}"
    if vendor == "cpu":
        return None
    from ._gpu_witness import require_device_count
    try:
        require_device_count(vendor, len(devices))
    except RuntimeError as exc:
        return str(exc)
    return None


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------

def format_par_check(r):
    lines = ["# python -m mojolearn verify --par", ""]
    if not r["ran"]:
        lines.append("NOT RUN: " + r["reason"])
        return "\n".join(lines)
    lines.append(f"Ran each `par-*` lane TWICE in this process on this box ({r['vendor']}): once with")
    lines.append(f"MOJOLEARN_PAR_DEVICES={r['devices'][0]} and once with "
                 f"MOJOLEARN_PAR_DEVICES={','.join(str(d) for d in r['devices'])}, and compared the")
    lines.append("two columns cell for cell. No reference table is involved: the one-device column")
    lines.append("produced here, minutes ago, off this build, IS the reference. That equality is")
    lines.append("the drivers' whole claim and no release record states it.")
    lines.append("")
    if r["vendor"] == "cpu":
        lines.append("THIS IS THE CPU ROUTE AND IT IS NOT A DEVICE CLAIM. Each `device` here is one")
        lines.append("worker PROCESS with no device selection. What it checks is the drivers' own")
        lines.append("partition and merge across two processes, which is real and is where a")
        lines.append("sharding bug lives; it says nothing about two physical GPUs.")
        lines.append("")
    if r["perturbed"]:
        lines.append("SELF-TEST ARM: the two-device column was fed an input whose first column was")
        lines.append("moved up by one ULP. Every compared part MUST read DIVERGENT. A part that")
        lines.append("reads IDENTICAL here means this comparison cannot fail and is worthless.")
        lines.append("")
    for c in r["cells"]:
        if c["verdict"] == "N/A":
            continue
        lines.append(f"  {c['lane']:<26} {c['fixture']:<8} {c['part']:<9} {c['verdict']:<11} "
                     f"one={c['one']}  two={c['two']}")
        for side in ("one", "two"):
            if c[f"{side}_error"]:
                lines.append(f"      {side}-device error: {str(c[f'{side}_error'])[:400]}")
    if r["counts"].get("CPU-ROUTE-LIMIT"):
        lines.append(f"{r['counts']['CPU-ROUTE-LIMIT']} part(s) read CPU-ROUTE-LIMIT: the "
                     "two-device side refused with the CPU route's own by-name sentence "
                     "('no CPU implementation of ...'), which is a declared limit of that route "
                     "and not a defect. They are named rather than hidden, and they do not gate. "
                     "On a GPU install this verdict cannot occur.")
        lines.append("")
    w = r["witness"]
    lines.append(f"Device witness: {w['pools']} pool(s), {w['workers']} worker(s), "
                 f"placement={w['placement']}.")
    if r["witness_refusal"]:
        lines.append("WITNESS REFUSED: " + r["witness_refusal"])
        lines.append("")
        lines.append("RESULT: CANNOT RUN. The two-device column was not shown to have run on two")
        lines.append("devices, so its agreement with the one-device column is not evidence. This is")
        lines.append("reported rather than counted, because a clean AGREE from a column that never")
        lines.append("sharded is the exact failure this command exists to prevent.")
        return "\n".join(lines)
    lines.append("")
    state = r.get("state") or ("NOTHING COMPARED" if r["passed"] is None
                               else "VERIFIED" if r["passed"] else "MISMATCH")
    if state == "INCOMPLETE":
        lines.append(f"RESULT: INCOMPLETE. {r['counts'].get('REFUSED', 0)} part(s) REFUSED on BOTH")
        lines.append("columns, so they were never compared. Absence is not agreement: the parts that")
        lines.append("did run do not make up for the ones that did not, and there is no threshold")
        lines.append("below which a part that never ran counts as checked. The refusals are above,")
        lines.append("with their errors.")
    elif state == "NOTHING COMPARED":
        lines.append("RESULT: NOTHING COMPARED. No lane produced both columns, so there is nothing")
        lines.append("to agree or disagree about. This is not a pass and not a failure.")
    elif r["passed"]:
        lines.append(f"RESULT: THE TWO COLUMNS AGREE on {r['counts'].get('IDENTICAL', 0)} compared "
                     f"cell parts across {len(r['lanes'])} lane(s)")
        lines.append(f"and {len(r['fixtures'])} fixture(s), " +
                     ", ".join(f"{k}={v}" for k, v in sorted(r["counts"].items())) + ".")
        lines.append("Sharding this work across the devices did not move a bit.")
    else:
        gated = {k: r["counts"][k] for k in GATING if k in r["counts"]}
        lines.append("RESULT: THE TWO COLUMNS DISAGREE. " +
                     ", ".join(f"{k}={v}" for k, v in sorted(gated.items())) + ".")
        lines.append("A DIVERGENT part means sharding the work changed the bits. A ONE-COLUMN part")
        lines.append("means the work refused on exactly one of the two columns, which is a defect")
        lines.append("only a two-device run can see. Both are above, with the hashes and the error.")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# the command
# --------------------------------------------------------------------------

def _parse_devices(text):
    if not text:
        return DEFAULT_PAR_DEVICES
    try:
        devices = tuple(int(x) for x in str(text).split(",") if x.strip() != "")
    except ValueError:
        raise CannotRun(f"--par-devices {text!r} is not a list of integer device indices") from None
    if len(devices) < 2 or len(set(devices)) != len(devices) or any(d < 0 for d in devices):
        raise CannotRun(f"--par-devices {text!r} must name at least two distinct "
                        "nonnegative device indices; the claim is about more than one device")
    return devices


def cmd_par_check(args, ml):
    """`verify --par [quick|default|all]`: the two-device column against the
    one-device column, both produced here."""
    from . import _backend
    json_out = getattr(args, "json", False)
    log = (lambda s: _emit(s, sys.stderr)) if json_out else _emit
    started = time.time()
    scope = getattr(args, "par", None) or "default"
    perturb = bool(getattr(args, "par_self_test", False))
    try:
        devices = _parse_devices(getattr(args, "par_devices", "") or "")
        harness = load_harness(par_axis=True)
    except (FileNotFoundError, CannotRun) as exc:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN", str(exc))

    vendor = _backend.vendor()
    refusal = refuse_to_run(vendor, devices)
    if refusal:
        result = dict(ran=False, vendor=vendor, devices=list(devices), reason=refusal)
        if json_out:
            _emit(json.dumps(dict(format="mojolearn.verify-par.v1", **result), indent=1, sort_keys=True))
        else:
            _emit(format_par_check(result))
        return EXIT_CANNOT_RUN

    try:
        chosen, every, per_family = par_lanes(harness, scope)
    except CannotRun as exc:
        return _finish(args, EXIT_USAGE, "USAGE", str(exc))
    asked = [x for x in (getattr(args, "lanes", "") or "").split(",") if x]
    if asked:
        unknown = [l for l in asked if l not in every]
        if unknown:
            _emit(f"USAGE: --lanes names lanes that are not `par-*` driver lanes: {unknown}; "
                  f"this command covers the {len(every)} of them", sys.stderr)
            return EXIT_USAGE
        chosen = [l for l in every if l in asked]
    fixtures = [x for x in (getattr(args, "fixtures", "") or "").split(",") if x] or [QUICK_FIXTURE]
    bad = [f for f in fixtures if f not in harness.FIXTURES]
    if bad:
        _emit(f"USAGE: --fixtures names fixtures the harness does not define: {bad}", sys.stderr)
        return EXIT_USAGE
    if perturb:
        chosen, fixtures = chosen[:1], fixtures[:1]

    log(f"# par ({scope}): {len(chosen)} of {len(every)} par-* lanes x {len(fixtures)} fixture(s), "
        f"devices {devices[0]} then {','.join(str(d) for d in devices)}")
    try:
        result = par_check(harness, ml, chosen, fixtures, devices=devices,
                           repeats=max(1, int(getattr(args, "repeats", 1) or 1)),
                           log=log, perturb=perturb)
    except CannotRun as exc:
        return _finish(args, EXIT_CANNOT_RUN, "CANNOT RUN", str(exc))
    result.update(scope=scope, par_lanes=every, representative_of=per_family,
                  elapsed_s=round(time.time() - started, 2))
    if perturb:
        # THE ARM THAT MUST FAIL. Every compared part has to read DIVERGENT;
        # an IDENTICAL one says the comparison cannot discriminate.
        compared = [c for c in result["cells"]
                    if c["verdict"] not in ("N/A", "REFUSED", "CPU-ROUTE-LIMIT")]
        result["self_test_passed"] = bool(compared) and all(
            c["verdict"] == "DIVERGENT" for c in compared)
        result["self_test_compared"] = len(compared)
    if json_out:
        _emit(json.dumps(dict(format="mojolearn.verify-par.v1", **result), indent=1, sort_keys=True))
    else:
        _emit(format_par_check(result))
    if result["witness_refusal"] or not result["cells"]:
        return EXIT_CANNOT_RUN
    if perturb:
        # THE SELF-TEST ARM IS JUDGED BY ITS OWN POSITIVE REQUIREMENT and is
        # read before the state ladder, because every part of a perturbed run
        # is SUPPOSED to read DIVERGENT: the ladder would call that a MISMATCH,
        # which it is, and say nothing about whether the comparison can fail.
        # `self_test_passed` is already false when nothing was compared.
        if result["self_test_passed"]:
            _emit("SELF-TEST PASSED: the perturbed two-device column read DIVERGENT on "
                  f"{result['self_test_compared']} compared part(s). This comparison can fail.")
            return EXIT_VERIFIED
        _emit("SELF-TEST FAILED: a perturbed two-device column did not read DIVERGENT, so this "
              "comparison cannot catch a wrong answer.")
        return EXIT_MISMATCH
    # THE LADDER, not `passed` alone. `passed` is None for both INCOMPLETE and
    # NOTHING COMPARED and neither is a pass; True only when a part was
    # actually compared and every compared part agreed.
    if result["state"] == "MISMATCH":
        return EXIT_MISMATCH
    if result["state"] != "VERIFIED":
        return EXIT_CANNOT_RUN
    return EXIT_VERIFIED
