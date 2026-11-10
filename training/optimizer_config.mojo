# SPDX-License-Identifier: Apache-2.0
"""Native public optimizer configuration and parameter-name decisions."""


@fieldwise_init
struct OptimizerConstructorStart(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var accumulation: Int
    var beta1: Float64
    var beta2: Float64
    var nesterov: Bool
    var maximize: Bool

    def name(self) -> String:
        return String("SGD") if self.kind == 0 else (String("Adam") if self.kind == 1 else String("AdamW"))


@fieldwise_init
struct OptimizerConstructorConfig(Copyable, ImplicitlyCopyable, Movable):
    var kind: Int
    var accumulation: Int
    var lr: Float64
    var beta1: Float64
    var beta2: Float64
    var eps: Float64
    var weight_decay: Float64
    var momentum: Float64
    var dampening: Float64
    var nesterov: Bool
    var maximize: Bool

    def name(self) -> String:
        return String("SGD") if self.kind == 0 else (String("Adam") if self.kind == 1 else String("AdamW"))

    def validate(self) raises:
        optimizer_validate_accumulation(self.kind, self.accumulation)
        if self.kind == 0 and self.nesterov:
            optimizer_validate_nesterov_momentum(self.momentum)
            optimizer_validate_nesterov_dampening(self.dampening)


def optimizer_validate_accumulation(kind: Int, accumulation: Int) raises:
    if accumulation < 1 or (accumulation & (accumulation - 1)) != 0:
        var name = String("SGD") if kind == 0 else (String("Adam") if kind == 1 else String("AdamW"))
        raise Error("ValueError: mojolearn." + name + ": accumulation_steps must be a power of two >= 1 (optimizer contract clause 9.2 condition 4)")


def optimizer_validate_nesterov_momentum(momentum: Float64) raises:
    if momentum == 0.0:
        raise Error("ValueError: mojolearn.SGD: nesterov=True needs momentum > 0 and dampening == 0")


def optimizer_validate_nesterov_dampening(dampening: Float64) raises:
    if dampening != 0.0:
        raise Error("ValueError: mojolearn.SGD: nesterov=True needs momentum > 0 and dampening == 0")


def optimizer_parameter_names(kind: Int) -> List[String]:
    # The public lexical insertion order is part of get_params packing.
    if kind == 0:
        return ["accumulation_steps", "dampening", "lr", "lr_schedule", "maximize", "momentum", "nesterov", "numeric_mode", "params", "resident", "weight_decay"]
    return ["accumulation_steps", "betas", "eps", "lr", "lr_schedule", "maximize", "numeric_mode", "params", "resident", "weight_decay"]


def optimizer_parameter_update(kind: Int, key: String) raises:
    var separator = key.find("__")
    var head = String(key[byte=0:separator]) if separator >= 0 else String(key)
    var names = optimizer_parameter_names(kind)
    var found = False
    for i in range(len(names)):  # driver-loop: fixed public constructor parameter names
        if names[i] == head:
            found = True
            break
    if not found:
        raise Error("ValueError: invalid optimizer parameter '" + head + "'")
    if separator >= 0:
        raise Error("TypeError: optimizer nested Python set_params hooks cannot execute; pass a complete native schedule or buffer configuration")


def optimizer_profile_available(available: Bool) raises:
    if not available:
        raise Error("NotImplementedError: the selected numeric profile is unavailable for optimizer training")


def optimizer_keyword_refusal(key: String, where: String) raises:
    """The fixed public keyword policy, independent of Python module dictionaries."""
    if key == "dropout":
        raise Error("NotImplementedError: mojolearn." + where + ": " + "dropout is not an OPTIMIZER option. It is an op on this surface: `Generator(seed).dropout(x, p)` draws its mask from the position-keyed Philox stream in core/philox_neural.mojo (a pure function of seed, stream id and element index, never of the launch geometry) and `Generator.dropout_backward(dy, key)` replays it. Stochastic depth and any other sampling stay uncovered.")
    if key == "distributed" or key == "world_size" or key == "rank":
        raise Error("NotImplementedError: mojolearn." + where + ": " + "DISTRIBUTED TRAINING IS NOT COVERED. Nothing here is aware of a second process or a second device: there is no all-reduce, no gradient bucket and no rank. A cross-device gradient sum is another reduction and would need its own contract clause, its own fixture and its own sabotage before it could carry the identity claim the single-device path carries.")
    if key == "amsgrad":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "the profile EXCLUDES amsgrad and the exclusion is load bearing: `max(v_max, v)` would create the profile's first -0.0 vs +0.0 selection hazard, where the order of a compare decides which of two equal-comparing values survives (optimizer contract 8c, and clip_coefficient's docstring in training/checks/optimizer_oracle.mojo)" + " (python/mojolearn/_training_impl.py)")
    if key == "foreach":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "an EXECUTION knob. Adam here is ONE launch over the whole flat buffer and SGD is one launch per tensor, because SGD carries a per-tensor flag; the choice is the profile's and not the caller's (training/checks/optimizer.mojo::identical_optimizer_step)" + " (python/mojolearn/_training_impl.py)")
    if key == "fused":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "an EXECUTION knob, and in torch it is a different arithmetic as well" + " (python/mojolearn/_training_impl.py)")
    if key == "capturable":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "CUDA-graph capture; nothing on this surface captures, and the step synchronizes before it returns (training/checks/optimizer.mojo::identical_optimizer_step)" + " (python/mojolearn/_training_impl.py)")
    if key == "differentiable":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "there is no autograd in this library, which is also why `step` takes the gradients as an argument" + " (python/mojolearn/_training_impl.py)")
    if key == "params_groups":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "one `OptimizerConfig` per optimizer. Two parameter groups would be two calls with two registries and two clips, and two clips do not equal one: the clip coefficient is a function of every gradient in ITS registry (optimizer contract 3.5)" + " (python/mojolearn/_training_impl.py)")
    if key == "param_groups":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "one `OptimizerConfig` per optimizer; see `params_groups`" + " (python/mojolearn/_training_impl.py)")
    if key == "weight":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "a per-class weight vector is REFUSED BY NAME (loss contract section 11): a weighted mean's denominator is a SUM OF FLOATS and would need a fold, a clause, a fixture and a sabotage of its own, where `count` today is an INTEGER and therefore exact, order-free and vendor-free (contract 5.5, `ce_count` in training/checks/loss_oracle.mojo)" + " (python/mojolearn/_training_impl.py)")
    if key == "size_average":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "torch's own deprecated alias for `reduction`. Refused rather than translated, because translating it silently reproduces the ambiguity torch deprecated it for" + " (python/mojolearn/_training_impl.py)")
    if key == "reduce":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "torch's own deprecated alias for `reduction`; see `size_average`" + " (python/mojolearn/_training_impl.py)")
    if key == "dropout":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + null + " (python/mojolearn/_training_impl.py)")
    if key == "device":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "the device is chosen by the binary that loaded, and its accelerator API is readable with `vendor_used()` (python/mojolearn/_backend.py)" + " (python/mojolearn/_training_impl.py)")
    if key == "dtype":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + "float32 only. MIXED PRECISION IS NOT COVERED and a non-float32 array is refused by name rather than cast" + " (python/mojolearn/_training_impl.py)")
    if key == "distributed":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + null + " (python/mojolearn/_training_impl.py)")
    if key == "world_size":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + null + " (python/mojolearn/_training_impl.py)")
    if key == "rank":
        raise Error("TypeError: mojolearn." + where + ": " + key + "= is REFUSED BY NAME -- " + null + " (python/mojolearn/_training_impl.py)")
    raise Error("TypeError: mojolearn." + where + ": unexpected keyword " + key + "=. This surface takes only "
        + "the parameters in its WHAT IS HONORED table; an unknown keyword is refused rather than "
        + "ignored, because a keyword that is accepted and does nothing is a caller who believes "
        + "something about their run that is not true (python/mojolearn/_training_impl.py)")


def optimizer_beta_shape(rank: Int, count: Int) raises:
    if rank != 1 or count != 2:
        raise Error("ValueError: betas must be a pair (beta1, beta2)")
