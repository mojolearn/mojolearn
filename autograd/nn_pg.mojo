# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Andrew Hendel. Part of mojolearn, https://doi.org/10.5281/zenodo.22068632
"""Optimizer parameter groups of `mojolearn.ops.Session` (lane nn-blocks-pg, 2026-11-25).

One optimizer, several groups of parameters, each group with its own learning rate, weight decay,
betas, momentum, eps (torch.optim's `param_groups`): a new head at a higher rate, a pretrained backbone
lower, biases without weight decay.

THE TABLE. `OpsSession.param_group_optimizer` resolves the groups ONCE, at construction, into a
`ParamGroupTable`: one member optimizer per group (an ordinary single-group session optimizer, its
configuration validated by the same `_optimizer_entry` as `Session.optimizer`), the parameters of every
group in group order (the state-dict index order) and a handle -> group map. Nothing is resolved per
element; a step resolves each listed parameter's group once (one dictionary read, the same kind of read
`_state_of` already makes per parameter).

THE STEP. Per parameter, the member's own update, unchanged: `step_lr` (ONE OP_OPT launch) at the
member's scheduled rate and step count, or, when the optimizer clips, `step_lr_clipped` (ONE
OP_AGH_OPT_SCALED launch). No new kernel and no per-element group lookup: two groups with different
settings give, word for word, what two separately configured optimizers give on the same parameters.
A single-group optimizer runs exactly the launches it ran before (its step path tests one integer,
`_OpsOptimizer.pg`, once per call).

THE GLOBAL CLIP. clip_grad_norm is ONE norm over every gradient of every group the step lists, in the
order the step lists them (autograd/optim.mojo's cross-tensor fold, `clip_norm_accumulate`: the block
partials of each tensor, then the running total in the first tensor's cell, two launches a tensor, no
read-back), then ONE coefficient that every group's update reads from that cell. This is the same fold
`Session.step` already runs for one group, so the fold order is fixed and the same on every vendor and
the host column. Two clips (one per group) would not equal one: that is what the sabotage arm
PG_SABOTAGE_CLIP_PER_GROUP does, and what the `ops-param-groups` verify lane catches.

SCHEDULES. The optimizer's schedule (constant, step, exponential) scales every group's rate, as a
torch scheduler scales every `param_group["lr"]`: `scheduler_step` advances every member's epoch.

STATE. Each (group member, parameter) pair keeps the session's ordinary per-parameter state
(`_OpsOptState`); `optimizer_state` / `load_optimizer_state` on the grouped optimizer route to the
parameter's member. The state dict (`optimizer_state_dict`) is torch's layout:
{"state": {index: per-parameter state}, "param_groups": [{"params": [indices], "lr": ..., ...}]}.

Launches per step: per listed parameter one update launch, plus two norm launches when clip_grad_norm is
set (the single-group step's counts). No host read-back, no sync.
"""
from std.sys import is_defined

from sequence.ops import (
    OPT_SGD, OPT_ADAM, OPT_ADAMW, OPT_LION, OPT_ADAMAX, OPT_NADAM, OPT_RMSPROP, OPT_ADAGRAD,
)


# Sabotage arms (tools/identity_lanes/coverage_sabotage.tsv coverage-autograd-pg_clip_per_group_gpu / _cpu and
# coverage-autograd-pg_group_lr_gpu / _cpu, lane ops-param-groups). Off in every build; an arm's patch switches
# one on. CLIP_PER_GROUP: the norm restarts at each group's first listed tensor and closes at its last, so each
# group is clipped by its own norm (two clips, not one). GROUP_LR: every group steps at group 0's rate.
comptime PG_SABOTAGE_CLIP_PER_GROUP = is_defined["MOJOLEARN_PG_SABOTAGE_CLIP_PER_GROUP"]()
comptime PG_SABOTAGE_CLIP_PER_GROUP_HOST = is_defined["MOJOLEARN_PG_SABOTAGE_CLIP_PER_GROUP_HOST"]()
comptime PG_SABOTAGE_GROUP_LR = is_defined["MOJOLEARN_PG_SABOTAGE_GROUP_LR"]()
comptime PG_SABOTAGE_GROUP_LR_HOST = is_defined["MOJOLEARN_PG_SABOTAGE_GROUP_LR_HOST"]()
# The training module's twins (coverage-training-pg_clip_per_group_gpu / _cpu, lane training-param-groups): the
# parameter-group step of SGD / Adam / AdamW clips each group by its own norm (bindings/optimizer_native_router_door.mojo
# pg_training_operation). GPU: a GPU image's device-side optimizer; CPU: a CPU-column image or a device="cpu" one.
comptime PG_SABOTAGE_TRAINING_CLIP = is_defined["MOJOLEARN_PG_SABOTAGE_TRAINING_CLIP"]()
comptime PG_SABOTAGE_TRAINING_CLIP_HOST = is_defined["MOJOLEARN_PG_SABOTAGE_TRAINING_CLIP_HOST"]()
comptime PG_COLUMN_CPU = is_defined["MOJOLEARN_COLUMN_CPU"]()


struct ParamGroupTable(Copyable, Movable):
    """The groups of one optimizer, resolved at construction.

    members: the session optimizer index (`OpsSession.opts`) of each group's member, in group order.
    params: every group's parameter handles, group 0 first (the state-dict index order).
    starts: group g's parameters are params[starts[g]:starts[g + 1]] (len = groups + 1).
    group_of: parameter handle (with its generation) -> group index."""

    var members: List[Int]
    var params: List[Int]
    var starts: List[Int]
    var group_of: Dict[Int, Int]

    def __init__(out self):
        self.members = List[Int]()
        self.params = List[Int]()
        self.starts = List[Int]()
        self.starts.append(0)
        self.group_of = Dict[Int, Int]()

    def groups(self) -> Int:
        return len(self.members)


def pg_kind_name(kind: Int) -> String:
    """The `kind` argument's name of an OPT_* code."""
    if kind == OPT_SGD:
        return "sgd"
    if kind == OPT_ADAM:
        return "adam"
    if kind == OPT_ADAMW:
        return "adamw"
    if kind == OPT_RMSPROP:
        return "rmsprop"
    if kind == OPT_ADAGRAD:
        return "adagrad"
    if kind == OPT_LION:
        return "lion"
    if kind == OPT_ADAMAX:
        return "adamax"
    if kind == OPT_NADAM:
        return "nadam"
    return "optimizer " + String(kind)


def _adam_like(kind: Int) -> Bool:
    return kind == OPT_ADAM or kind == OPT_ADAMW or kind == OPT_ADAMAX or kind == OPT_NADAM


def pg_key_taken(kind: Int, key: String) -> Bool:
    """Whether a parameter group of an optimizer of this kind takes `key` (torch.optim's per-group options
    of that optimizer; `betas` or `beta1` / `beta2` for the Adam family and Lion)."""
    if key == "params" or key == "lr" or key == "weight_decay":
        return True
    if kind == OPT_SGD:
        return key == "momentum" or key == "nesterov" or key == "dampening"
    if _adam_like(kind):
        return key == "betas" or key == "beta1" or key == "beta2" or key == "eps"
    if kind == OPT_LION:
        return key == "betas" or key == "beta1" or key == "beta2"
    if kind == OPT_RMSPROP:
        return key == "alpha" or key == "eps" or key == "momentum" or key == "centered"
    if kind == OPT_ADAGRAD:
        return key == "eps" or key == "lr_decay" or key == "initial_accumulator_value"
    return False


def pg_admit_key(kind: Int, key: String, group: Int) raises:
    """Refuse a per-group key in clear words: optimizer-wide settings, keys of another optimizer kind and
    unknown keys, naming the key and the group index."""
    var where = "param_groups[" + String(group) + "]"
    if pg_key_taken(kind, key):
        return
    if key == "schedule" or key == "step_size" or key == "gamma":
        raise Error("ValueError: " + where + ": '" + key + "' is optimizer-wide: one schedule scales every "
                    + "group's learning rate; pass it to param_group_optimizer, not to a group")
    if key == "clip_grad_norm" or key == "clip_grad_value":
        raise Error("ValueError: " + where + ": '" + key + "' is optimizer-wide: the clip is computed over "
                    + "the gradients of every group; pass it to param_group_optimizer, not to a group")
    var known = (key == "momentum" or key == "nesterov" or key == "dampening" or key == "betas" or key == "beta1"
                 or key == "beta2" or key == "eps" or key == "alpha" or key == "centered" or key == "lr_decay"
                 or key == "initial_accumulator_value")
    if known:
        raise Error("ValueError: " + where + ": the " + pg_kind_name(kind) + " optimizer does not take '"
                    + key + "'")
    raise Error("ValueError: " + where + ": unknown parameter group key '" + key + "'")


def pg_check_membership(groups: List[List[Int]]) raises -> ParamGroupTable:
    """The table's parameter lists and handle map, refusing an empty group, a parameter listed twice in one
    group and a parameter in two groups (naming both group indices). `members` is left empty: the caller
    appends each group's member optimizer."""
    var table = ParamGroupTable()
    if len(groups) < 1:
        raise Error("ValueError: param_groups must hold at least one group")
    for g in range(len(groups)):  # driver-loop: over parameter groups (construction, once)
        if len(groups[g]) < 1:
            raise Error("ValueError: param_groups[" + String(g) + "] has no parameters")
        for i in range(len(groups[g])):  # driver-loop: over one group's parameter handles (construction, once)
            var h = groups[g][i]
            if h in table.group_of:
                var other = table.group_of[h]
                if other == g:
                    raise Error("ValueError: param_groups[" + String(g) + "] lists parameter " + String(i)
                                + " twice")
                raise Error("ValueError: parameter " + String(i) + " of param_groups[" + String(g)
                            + "] is also in param_groups[" + String(other) + "]; a parameter belongs to one group")
            table.group_of[h] = g
            table.params.append(h)
        table.starts.append(len(table.params))
    return table^
