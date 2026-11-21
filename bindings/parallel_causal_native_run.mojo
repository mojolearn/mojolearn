# SPDX-License-Identifier: Apache-2.0
"""Typed rules and request plans of the multi-device causal language model
(RUN): the route a vendor gives a layer map, the layer map's admission, the
closed / lost refusals, the agreement of the layer owners on weight format,
vendor and numeric mode, the constructor keyword and weight-format rules,
the index bounds, the worker's frame, route and tensor-name rules, and the
words of the `run` and `run_layers` requests with the rules that cut a token
batch into one request per device.
`bindings/parallel_causal_native_door.mojo` unpacks the Python values into
these arguments, sends the requests to the process pool and packs the
replies; no Python object reaches this file. Each function here is word or
text comparison done once per driver call or once per device run; the layers
themselves run in the workers, one worker per device."""
from core.parallel_worker_request import WorkerRequest, WORKER_NO_HANDLE, PC_RUN, PC_RUN_LAYERS
from core.parallel_worker_frame import WIRE_REQUEST

comptime CAUSAL_ROUTE_GPU = 0
comptime CAUSAL_ROUTE_CPU = 1
# The word UNPACK stores for a device entry that is not an exact integer.
comptime CAUSAL_DEVICE_INEXACT = -1
comptime CAUSAL_LAYERS_HEADER_WORDS = 4


# ------------------------------------------------------------ route and layer map
def causal_route_reads_devices(vendor: String) -> Bool:
    """Whether the vendor's route rule looks at the device indices (so the
    door reads them only then, as it did)."""
    return vendor == "metal"


def causal_route_plan(vendor: String, devices: List[Int]) raises -> Int:
    """The worker route of a layer map on this installation's vendor.
    `devices` is read for Metal only; an entry that is not an exact integer
    is CAUSAL_DEVICE_INEXACT."""
    if vendor == "metal":
        if len(devices) == 0:
            raise Error("Metal ParallelCausalLM requires every layer owner to be device 0")
        for i in range(len(devices)):  # driver-loop: over explicit layer ownership metadata
            if devices[i] != 0:
                raise Error("Metal ParallelCausalLM requires every layer owner to be device 0")
        return CAUSAL_ROUTE_GPU
    if vendor == "cuda" or vendor == "hip":
        return CAUSAL_ROUTE_GPU
    if vendor == "cpu":
        return CAUSAL_ROUTE_CPU
    raise Error("NotImplementedError: ParallelCausalLM requires CUDA or HIP device isolation, single-device Metal, or a CPU-only install")


def causal_route_name(route: Int) -> String:
    return String("cpu") if route == CAUSAL_ROUTE_CPU else String("gpu")


def causal_layer_map_admit(devices: List[Int], expected: Int) raises:
    """One nonnegative exact device index per layer (`expected` < 0: any
    positive count)."""
    if len(devices) == 0 or (expected >= 0 and len(devices) != expected):
        raise Error("layer_devices requires one nonnegative device index per layer")
    for i in range(len(devices)):  # driver-loop: over layer-to-worker indices
        if devices[i] < 0:
            raise Error("layer_devices requires one nonnegative device index per layer")


# ------------------------------------------------------------ owner rules
def causal_closed_admit(closed: Bool) raises:
    """A closed model is a RuntimeError on every public call."""
    if closed:
        raise Error("RuntimeError: ParallelCausalLM is closed")


def causal_lost_admit(failed: Bool) raises:
    """So is a lost worker state; an invalid argument stays a ValueError."""
    if failed:
        raise Error("RuntimeError: ParallelCausalLM worker state was lost; reload the model")


struct CausalRegistryState(Movable):
    """The typed state of one model's worker registry: the device of each
    layer, the distinct devices in first-appearance order (a device's
    position is its worker's index in the pool), and the two lifecycle
    words. Built once at construction; the operations read it instead of
    re-admitting and copying the model's fields."""
    var layer_devices: List[Int]
    var workers: List[Int]
    var closed: Bool
    var failed: Bool

    def __init__(out self, layer_devices: List[Int]):
        self.layer_devices = layer_devices.copy()
        self.workers = List[Int]()
        self.closed = False
        self.failed = False
        for i in range(len(layer_devices)):  # driver-loop: stable first-appearance physical worker ordering
            var seen = False
            for j in range(len(self.workers)):  # small-loop(the distinct devices met so far)
                if self.workers[j] == layer_devices[i]:
                    seen = True
                    break
            if not seen:
                self.workers.append(layer_devices[i])


def causal_registry_usable(state: CausalRegistryState) raises:
    """A closed model or a lost worker state is a RuntimeError on every
    public call. One check so the calls cannot disagree."""
    causal_closed_admit(state.closed)
    causal_lost_admit(state.failed)


def causal_registry_worker(state: CausalRegistryState, device: Int) raises -> Int:
    """The pool index of the worker that owns `device`."""
    for j in range(len(state.workers)):  # small-loop(the model's distinct devices)
        if state.workers[j] == device:
            return j
    raise Error("KeyError: " + String(device))


def causal_layer_format_admit(reported: String, requested: String) raises:
    if reported != requested:
        raise Error("ParallelCausalLM: layer reports a different weight_format than requested")


def causal_layer_owner_admit(mode: Int, vendor: String, first_mode: Int, first_vendor: String) raises:
    """Every later layer's worker answers the first layer's vendor and mode."""
    if mode != first_mode or vendor != first_vendor:
        raise Error("ParallelCausalLM: layer owners disagree on vendor or numeric mode")


def causal_option_admit(name: String) raises:
    if name != "weight_format" and name != "max_positions" and name != "numeric_profile":
        raise Error("TypeError: ParallelCausalLM: unexpected constructor keyword " + name)


def causal_weight_format_admit(format: String) raises:
    if format != "float32" and format != "bfloat16" and format != "int8":
        raise Error("ParallelCausalLM: unsupported weight_format")


def causal_block_index_admit(index: Int, layers: Int) raises:
    if index < 0 or index >= layers:
        raise Error("ParallelCausalLM: remote block index outside its layer map")


def causal_allocation_index_admit(index: Int, blocks: Int) raises:
    if index < 0 or index >= blocks:
        raise Error("ParallelCausalLM: layer allocation index outside model")


# ------------------------------------------------------------ worker rules
# Removed 2026-11-21 (lane pools-frame): `causal_frame_admit`. The worker's
# receive is `process_worker_next` (bindings/native_process_pool_door.mojo),
# which refuses a frame that is neither a request nor the close.


def causal_worker_route(vendor: String) -> Int:
    """The route a worker process answers for: its own connected vendor."""
    return CAUSAL_ROUTE_CPU if vendor == "cpu" else CAUSAL_ROUTE_GPU


def causal_worker_route_admit(exact: Bool, reported: String, vendor: String) raises:
    """`configure` names the owner's route; it is this worker's or refused."""
    if not exact or reported != causal_route_name(causal_worker_route(vendor)):
        raise Error("ParallelCausalLM: worker route differs from its connected backend")


def causal_top_tensor_admit(name: String) raises:
    if name != "embed" and name != "norm" and name != "head":
        raise Error("ParallelCausalLM: unknown top-level tensor")


# ------------------------------------------------------------ run requests
def causal_step_word(method: String) -> Int:
    """`step` runs each block's decode entry; anything else its forward."""
    return 1 if method == "step" else 0


def causal_run_request(index: Int, step: Int, token: Int) -> WorkerRequest:
    """One layer's "run": words [step, state token or WORKER_NO_HANDLE]."""
    var run = WorkerRequest(PC_RUN, index)
    run.words.append(step)
    run.words.append(token)
    return run^


def causal_layers_request(step: Int, has_ids: Bool, with_head: Bool, items: List[Int]) -> WorkerRequest:
    """One device run's "run_layers": words [step, has ids, final, n,
    (layer index, state token or WORKER_NO_HANDLE) * n]. `items` holds the
    pairs flat, in layer order."""
    var run = WorkerRequest(PC_RUN_LAYERS)
    run.words.append(step)
    run.words.append(1 if has_ids else 0)
    run.words.append(1 if with_head else 0)
    run.words.append(len(items) // 2)
    for k in range(len(items)):  # small-loop(two words per layer of this device run)
        run.words.append(items[k])
    return run^


def causal_layers_count(words: List[Int]) raises -> Int:
    """The worker's admission of a "run_layers" request: its layer count."""
    if len(words) < CAUSAL_LAYERS_HEADER_WORDS:
        raise Error("ParallelCausalLM: malformed run_layers request")
    var count = words[3]
    if count < 0 or len(words) != CAUSAL_LAYERS_HEADER_WORDS + 2 * count:
        raise Error("ParallelCausalLM: malformed run_layers request")
    return count


def causal_run_joins(taken: Int, layer_device: Int, run_device: Int) -> Bool:
    """Whether the next layer goes in the device run being built: the first
    layer opens it; a later one joins while it is on the run's device."""
    return taken == 0 or layer_device == run_device


def causal_run_embeds(device: Int, embed_device: Int) -> Bool:
    """The embedding rides in the first run when that run is on its device."""
    return device == embed_device


def causal_run_final(next_layer: Int, layers: Int, device: Int, head_device: Int) -> Bool:
    """The head rides in the last run when that run is on the head's device."""
    return next_layer == layers and device == head_device
