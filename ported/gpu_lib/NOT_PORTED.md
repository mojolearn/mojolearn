# `cuda_lib` coverage, file by file

Their `catboost/cuda/cuda_lib/` is **57 headers**, not 47. Counted
2026-08-19 by

    find . \( -name '*.h' -o -name '*.cuh' \) -not -path './ut/*' | wc -l

which is 30 at the top level and 27 in the eight subdirectories
(`cuda_buffer_helpers/` 5, `future/` 4, `kernel/` 2 `.cuh`, `memory_pool/` 2,
`mpi/` 1, `serialization/` 1, `tasks_impl/` 10, `tasks_queue/` 2). Their
`ut/` directory holds tests and no headers. The previous count of 47 in this
file was wrong and the list under it named only 30 of the 57; both are fixed
below. `PORTING_RULES.md` repeats the 47 and needs the same correction.

The directory is named `gpu_lib` here, not `cuda_lib`, because nothing in it
is CUDA. That is the only rename; every CLASS keeps its CatBoost name
(`TCudaManager`, `TCudaStream`, `TGpuOneDeviceWorker`, `TCudaProfiler`,
`TStripeMapping`) so a reviewer can grep their tree with the symbol they see
in ours.

## Ported (11 of their headers, 10 of our files)

| ours | theirs | status |
|---|---|---|
| `fwd.mojo` | `fwd.h` | transliterated |
| `gpu_base.mojo` | `cuda_base.h` | partial, streams only |
| `slice.mojo` | `slice.h` | transliterated, all of it |
| `mapping.mojo` | `mapping.h` | partial, the three mappings and the three builders |
| `device_id.mojo` | `device_id.h` | transliterated |
| `task.mojo` | `task.h` + `tasks_impl/request_stream_task.h` | partial+replaced, tagged union for `ICommand` |
| `tasks_queue/single_host_task_queue.mojo` | same | replaced, no thread |
| `gpu_single_worker.mojo` | `gpu_single_worker.h`/`.cpp` | partial, the dispatch switch |
| `gpu_manager.mojo` | `cuda_manager.h`/`.cpp` | partial, single device |
| `gpu_profiler.mojo` | `cuda_profiler.h` | transliterated, guard is a `with` block |

`single_device.h` is not one of our files, but it is not untouched either:
`TCudaSingleDevice::WaitComplete` (`single_device.h:349`),
`StreamSynchronize` (`single_device.h:343`), `FreeStream`
(`single_device.h:335`) and `Stop` (`single_device.h:240`) are what
`gpu_manager.mojo` and `gpu_single_worker.mojo` implement between them, since
with one device the manager, the device and the worker collapse into two
objects instead of three.

## Not ported, and the reason

**Single host, so no MPI** (5). Their whole serialization dimension exists to
ship a command to another host. `mpi/mpi_manager.h`,
`serialization/task_factory.h`, `future/mpi_promise_future.h`,
`tasks_queue/mpi_task_queue.h`, `remote_objects.h`, and the `Save`/`Load` half
of every command in `task.h`.

**Single device, so no topology or peering** (9). `peer_devices.h`,
`inter_device_stream_section.h`, `devices_provider.h`, `devices_list.h`,
`hwloc_wrapper.h`, `device_subtasks_helper.h`, `tasks_impl/enable_peers.h`,
`memory_copy_performance.h`, `cpu_reducers.h`.

**Owned by `DeviceContext` instead** (7). Mojo's context allocates and frees,
so `memory_pool/cuda_malloc_wrapper.h`,
`memory_pool/stack_like_memory_pool.h`, `memory_provider_trait.h`,
`tasks_impl/memory_allocation.h`, `tasks_impl/memory_state_func.h`,
`cuda_kernel_buffer.h`, and `worker_state.h`'s `TMemoryState`, which only
reports what those providers hold. The `MemoryAllocation` and
`MemoryDeallocation` cases of the switch RAISE rather than silently accepting.

**Real gaps, wanted, not yet written** (21). These are not excused by the
port being single device. They are simply not done. The `mapping.h` row is a
part of an already-ported file and is not counted again.

| theirs | what it is | why it matters here |
|---|---|---|
| `cuda_buffer.h` (693 lines) | `TCudaBuffer`, the typed device buffer | the driver still passes raw `DeviceBuffer` around |
| `cuda_buffer_helpers/buffer_reader.h`, `buffer_writer.h` | the read and write paths of that buffer | same |
| `cuda_buffer_helpers/all_reduce.h`, `reduce_scatter.h`, `buffer_resharding.h` | collectives over a buffer | degenerate at one device, but they are how `TCudaBuffer` moves data |
| `read_and_write_helpers.h` | `Read`/`Write` over a vector of buffers | needs `cuda_buffer.h` |
| `mapping.h`'s `Transform`, `Apply`, `At`, `NonEmptyDevices` (not counted; the file is ported) | callback and metadata halves of a mapping | their only caller is `TCudaBuffer` |
| `tasks_impl/kernel_task.h` (341) | `IGpuKernelTask`, `PrepareExec` temp memory | our launches carry no temp-memory contract |
| `tasks_impl/memory_copy_tasks.h` (504), `memory_copy_staged_operation.h` | async copies as stream tasks | our copies are still bare `enqueue_copy` |
| `tasks_impl/cpu_func.h`, `host_tasks.h` | the host-task bodies the `HostTask` case runs | our `HostTask` case does the blocking guard and the caller runs the body |
| `future/future.h`, `local_promise_future.h`, `promise_factory.h` | how a command returns a value to the host | `RequestStream` writes its answer into the command instead |
| `kernel.h` | `NKernelHost::IMemoryManager` and the kernel-side context | pairs with `kernel_task.h` |
| `kernel/kernel.cuh`, `kernel/reduce.cuh` | their device-side helpers | our kernels are in `ported/*/kernel/` |
| `cache.h`, `helpers.h`, `column_aligment_helper.h` | small utilities: a guid-keyed cache, `ParseRangeString`, the 256-byte column alignment | pulled in as callers need them. The 256-byte alignment is the one with teeth, and `ported/gpu_data/` decides its own today |

The four groups plus the ported eleven plus `single_device.h` plus the three
below account for every one of the 57:
`11 + 1 + 5 + 9 + 7 + 21 + 3 = 57`.

## The ones that cannot be ported today (3)

`cuda_events_provider.h` and `stream_section_tasks_launcher.h` and
`tasks_impl/stream_section_task.h` all exist to order work ACROSS streams. On
Apple there is one stream: `DeviceContext.stream()` raises `Metal stream not
implemented` (`device_context.mojo:2172`), measured 2026-08-19. Ordering
across streams is not a thing that can be under-delivered here, it is a thing
with no referent. All three light up on CUDA and HIP with no caller change.

`cuda_base.h`'s `TCudaDeviceProperties` and `NCudaHelpers`
(`cuda_base.h:262-320`) are the same shape of problem, and like the `mapping.h`
row above they are part of an already-ported file and are not counted again:
their only reader is the `Reset` handler
(`gpu_single_worker.h:315`), which sizes the memory providers, and the memory
providers are `DeviceContext`'s here. It becomes portable the day a Mojo
device-properties query is established, and is dead weight until then.
