# SPDX-License-Identifier: Apache-2.0
"""Native public optimizer configuration and parameter-name decisions."""


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
        if self.accumulation < 1 or (self.accumulation & (self.accumulation - 1)) != 0:
            raise Error("ValueError: mojolearn." + self.name() + ": accumulation_steps must be a power of two >= 1 (optimizer contract clause 9.2 condition 4)")
        if self.kind == 0 and self.nesterov and (self.momentum == 0.0 or self.dampening != 0.0):
            raise Error("ValueError: mojolearn.SGD: nesterov=True needs momentum > 0 and dampening == 0")


def optimizer_parameter_names(kind: Int) -> List[String]:
    # The public lexical insertion order is part of get_params packing.
    if kind == 0:
        return ["accumulation_steps", "dampening", "lr", "lr_schedule", "maximize", "momentum", "nesterov", "numeric_mode", "params", "resident", "weight_decay"]
    return ["accumulation_steps", "betas", "eps", "lr", "lr_schedule", "maximize", "numeric_mode", "params", "resident", "weight_decay"]
