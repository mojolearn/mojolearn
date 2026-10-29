# Native-only compile root: no provider or Python dependency; never executed.
from mamba.impl.mamba_decode_plan import MambaDecodePlan


def main():
    var plan = MambaDecodePlan(2)
    var addresses = plan.step_addresses(True, 0, 0, [])
    var parameters = plan.step_parameters()
    var floats = plan.step_floats(0.0, 1.0)
    _ = addresses
    _ = parameters
    _ = floats
